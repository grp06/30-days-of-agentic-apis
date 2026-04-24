from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel

from .arena import ArenaLaneRecord, ArenaMetadata, ArenaRecorder, ArenaStatus
from .atomic_files import write_text_atomic
from .config import Settings
from .coordinator import (
    FixtureRunSource,
    RunCoordinator,
    SnapshotRunSource,
    utc_filename,
)
from .recorder import RunMetadata, RunStatus, RunSummary, prepare_run_directory
from .run_lifecycle import (
    PendingRunState,
    build_launch_failed_state,
    build_pending_run_state,
)
from .run_store import RunStore
from .model_client import OllamaModelClient


class ModelCatalog(Protocol):
    async def list_models(self) -> list[str]: ...

    def resolve_model_name(
        self, requested: str, available: list[str]
    ) -> str | None: ...


class LaunchResponse(BaseModel):
    run_id: str
    parent_run_id: str | None = None
    status: str


class ArenaLaunchLane(BaseModel):
    lane_id: str
    run_id: str
    model_name: str


class ArenaLaunchResponse(BaseModel):
    arena_id: str
    status: str
    lanes: list[ArenaLaunchLane]


class LaunchHint(BaseModel):
    run_id: str
    state: str
    current_model_name: str
    latest_sequence: int = 0
    preview_url: str | None = None
    checkpoint_id: str | None = None
    is_fork: bool = False


@dataclass
class ActiveLaunch:
    task: asyncio.Task[None]
    hint: LaunchHint


@dataclass
class PreparedArenaLane:
    run_id: str
    source: FixtureRunSource
    pending: PendingRunState


@dataclass
class PreparedArenaLaunch:
    arena_id: str
    lanes: list[ArenaLaunchLane] = field(default_factory=list)
    lane_records: list[ArenaLaneRecord] = field(default_factory=list)
    pending_lanes: list[PreparedArenaLane] = field(default_factory=list)
    cleanup_run_ids: list[str] = field(default_factory=list)
    cleanup_arena_id: str | None = None

    def track_created_run(self, run_id: str) -> None:
        self.cleanup_run_ids.append(run_id)

    def add_lane(
        self,
        *,
        lane_id: str,
        run_id: str,
        model_name: str,
        source: FixtureRunSource,
        pending: PendingRunState,
    ) -> None:
        self.lanes.append(
            ArenaLaunchLane(
                lane_id=lane_id,
                run_id=run_id,
                model_name=model_name,
            )
        )
        self.lane_records.append(
            ArenaLaneRecord(
                lane_id=lane_id,
                run_id=run_id,
                model_name=model_name,
            )
        )
        self.pending_lanes.append(
            PreparedArenaLane(run_id=run_id, source=source, pending=pending)
        )

    def metadata(self, fixture_name: str, task: str) -> ArenaMetadata:
        return ArenaMetadata(
            arena_id=self.arena_id,
            fixture_name=fixture_name,
            task=task,
            lanes=self.lane_records,
        )

    def initial_status(self) -> ArenaStatus:
        return ArenaStatus(
            arena_id=self.arena_id,
            state="running",
            total_lanes=len(self.lane_records),
            completed_lanes=0,
            lane_states={lane.lane_id: "running" for lane in self.lane_records},
        )

    def cleanup_runs(self, run_root: Path) -> None:
        for run_id in self.cleanup_run_ids:
            shutil.rmtree(run_root / run_id, ignore_errors=True)

    def track_arena_record_creation(self, arena_root: Path) -> None:
        arena_dir = self._owned_arena_dir(arena_root)
        if arena_dir is not None and not arena_dir.exists():
            self.cleanup_arena_id = self.arena_id

    def cleanup_arena_record(self, arena_root: Path) -> None:
        if self.cleanup_arena_id is None:
            return
        arena_dir = self._owned_arena_dir(arena_root)
        if arena_dir is None:
            return
        shutil.rmtree(arena_dir, ignore_errors=True)

    def _owned_arena_dir(self, arena_root: Path) -> Path | None:
        resolved_root = arena_root.resolve()
        arena_dir = (resolved_root / self.arena_id).resolve()
        try:
            arena_dir.relative_to(resolved_root)
        except ValueError:
            return None
        return arena_dir

    def response(self) -> ArenaLaunchResponse:
        return ArenaLaunchResponse(
            arena_id=self.arena_id, status="running", lanes=self.lanes
        )


class RunLauncher:
    def __init__(
        self,
        settings: Settings,
        coordinator_factory: Callable[[], RunCoordinator] | None = None,
        model_catalog_factory: Callable[[], ModelCatalog] | None = None,
    ) -> None:
        self.settings = settings
        self.run_store = RunStore(settings.run_root)
        self._tasks: dict[str, ActiveLaunch] = {}
        self._coordinator_factory = coordinator_factory or (
            lambda: RunCoordinator(settings)
        )
        self._model_catalog_factory = model_catalog_factory or (
            lambda: OllamaModelClient(settings)
        )

    async def start_fixture_run(
        self,
        fixture_name: str,
        task_override: str | None = None,
    ) -> LaunchResponse:
        run_id = utc_filename()
        source = FixtureRunSource(
            fixture_name=fixture_name, task_override=task_override
        )
        pending = self._register_pending_run(run_id=run_id, source=source)
        self._spawn(run_id, source, pending)
        return LaunchResponse(run_id=run_id, status="running")

    async def start_arena_run(
        self,
        fixture_name: str,
        task: str,
        lane_count: int = 4,
    ) -> ArenaLaunchResponse:
        arena_id = utc_filename()
        lane_models = self._arena_lane_models(lane_count)
        await self._validate_arena_models(lane_models)

        prepared = self._prepare_arena_launch(
            arena_id=arena_id,
            fixture_name=fixture_name,
            task=task,
            lane_models=lane_models,
        )
        prepared.track_arena_record_creation(self.settings.arena_root)
        try:
            arena_recorder = ArenaRecorder(
                self.settings.arena_root,
                arena_id,
                prepared.metadata(fixture_name=fixture_name, task=task),
            )
            arena_recorder.initialize_status(prepared.initial_status())
        except Exception:
            prepared.cleanup_runs(self.settings.run_root)
            prepared.cleanup_arena_record(self.settings.arena_root)
            raise

        for lane in prepared.pending_lanes:
            self._spawn(lane.run_id, lane.source, lane.pending)

        return prepared.response()

    async def start_fork_run(
        self,
        parent_run_id: str,
        checkpoint_sequence: int,
        instruction_override: str,
    ) -> LaunchResponse:
        parent_metadata = self.run_store.load_metadata(parent_run_id)
        checkpoint_path = self.run_store.resolve_checkpoint(
            parent_run_id, checkpoint_sequence
        )
        checkpoint_payload = self.run_store.load_checkpoint_payload(checkpoint_path)
        source = SnapshotRunSource(
            fixture_name=parent_metadata.fixture_name,
            parent_run_id=parent_run_id,
            source_snapshot_id=str(checkpoint_payload["snapshot_id"]),
            source_checkpoint_sequence=checkpoint_sequence,
            instruction_override=instruction_override,
            parent_task=parent_metadata.task,
            model_name=parent_metadata.model_name,
        )
        run_id = utc_filename()
        pending = self._register_pending_run(run_id=run_id, source=source)
        self._spawn(run_id, source, pending)
        return LaunchResponse(
            run_id=run_id, parent_run_id=parent_run_id, status="running"
        )

    def get_launch_hint(self, run_id: str) -> LaunchHint | None:
        active = self._tasks.get(run_id)
        if active is None:
            return None
        if active.task.done():
            self._tasks.pop(run_id, None)
            return None
        return active.hint

    def _register_pending_run(
        self,
        *,
        run_id: str,
        source: FixtureRunSource | SnapshotRunSource,
    ) -> PendingRunState:
        pending = build_pending_run_state(
            run_id=run_id,
            task=self._metadata_task(source),
            model_name=self._model_name_for_source(source),
            fixture_name=source.fixture_name,
            is_fork=isinstance(source, SnapshotRunSource),
            parent_run_id=source.parent_run_id
            if isinstance(source, SnapshotRunSource)
            else None,
            source_snapshot_id=(
                source.source_snapshot_id
                if isinstance(source, SnapshotRunSource)
                else None
            ),
            source_checkpoint_sequence=(
                source.source_checkpoint_sequence
                if isinstance(source, SnapshotRunSource)
                else None
            ),
            instruction_override=(
                source.instruction_override
                if isinstance(source, SnapshotRunSource)
                else None
            ),
        )
        prepare_run_directory(
            self.settings.run_root, run_id, pending.metadata, pending.status
        )
        return pending

    def _prepare_arena_launch(
        self,
        *,
        arena_id: str,
        fixture_name: str,
        task: str,
        lane_models: list[str],
    ) -> PreparedArenaLaunch:
        prepared = PreparedArenaLaunch(arena_id=arena_id)
        try:
            for index, model_name in enumerate(lane_models):
                run_id = utc_filename()
                prepared.track_created_run(run_id)
                lane_id = f"lane-{index + 1}"
                source = FixtureRunSource(
                    fixture_name=fixture_name,
                    task_override=task,
                    model_name=model_name,
                )
                pending = self._register_pending_run(run_id=run_id, source=source)
                prepared.add_lane(
                    lane_id=lane_id,
                    run_id=run_id,
                    model_name=model_name,
                    source=source,
                    pending=pending,
                )
        except Exception:
            prepared.cleanup_runs(self.settings.run_root)
            raise
        return prepared

    def _arena_lane_models(self, lane_count: int) -> list[str]:
        if lane_count < 1:
            raise ValueError("lane_count must be at least 1")
        models = self.settings.ollama_arena_models
        if len(models) < lane_count:
            raise ValueError(
                "OLLAMA_ARENA_MODELS must provide at least "
                f"{lane_count} model(s); got {len(models)}"
            )
        return models[:lane_count]

    async def _validate_arena_models(self, lane_models: list[str]) -> None:
        catalog = self._model_catalog_factory()
        available = await catalog.list_models()
        missing: list[str] = []
        suggestions: dict[str, list[str]] = {}
        resolved_models: list[str] = []
        for model in lane_models:
            resolved_model = catalog.resolve_model_name(model, available)
            if resolved_model is not None:
                resolved_models.append(resolved_model)
                continue
            missing.append(model)
            suggestions[model] = get_close_matches(
                model.removesuffix(":cloud"),
                available,
                n=3,
                cutoff=0.45,
            )
        if not missing:
            duplicate_models = _duplicate_values(resolved_models)
            if duplicate_models:
                raise ValueError(
                    "OLLAMA_ARENA_MODELS must resolve to distinct models for each "
                    "lane; duplicate resolved model(s): "
                    + ", ".join(repr(model) for model in duplicate_models)
                )
            return
        details = []
        for model in missing:
            close = suggestions[model]
            suffix = f" Did you mean: {', '.join(close)}?" if close else ""
            details.append(f"{model!r}.{suffix}")
        raise ValueError(
            "Configured arena model(s) were not found in Ollama /tags: "
            + " ".join(details)
        )

    def _model_name_for_source(
        self, source: FixtureRunSource | SnapshotRunSource
    ) -> str:
        return getattr(source, "model_name", None) or self.settings.ollama_model

    def _spawn(
        self,
        run_id: str,
        source: FixtureRunSource | SnapshotRunSource,
        pending: PendingRunState,
    ) -> None:
        task = asyncio.create_task(self._run_and_cleanup(run_id, source))
        self._tasks[run_id] = ActiveLaunch(
            task=task,
            hint=self._hint_from_pending_state(pending.status),
        )

    def _hint_from_pending_state(self, status: RunStatus) -> LaunchHint:
        return LaunchHint(
            run_id=status.run_id,
            state=status.state,
            current_model_name=status.current_model_name,
            latest_sequence=status.latest_sequence,
            preview_url=status.preview_url,
            checkpoint_id=status.checkpoint_id,
            is_fork=status.is_fork,
        )

    async def _run_and_cleanup(
        self,
        run_id: str,
        source: FixtureRunSource | SnapshotRunSource,
    ) -> None:
        coordinator = self._coordinator_factory()
        try:
            await coordinator.run_once(source, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            self._mark_launch_failed(run_id, source, str(exc))
        finally:
            self._tasks.pop(run_id, None)

    def _metadata_task(self, source: FixtureRunSource | SnapshotRunSource) -> str:
        if isinstance(source, SnapshotRunSource):
            return (
                f"{source.parent_task}\n\n"
                "Fork instruction override:\n"
                f"- {source.instruction_override}\n"
            )
        if source.task_override:
            return source.task_override
        fixture_path = self.settings.fixture_root / source.fixture_name / "TASK.md"
        return fixture_path.read_text(encoding="utf-8")

    def _mark_launch_failed(
        self,
        run_id: str,
        source: FixtureRunSource | SnapshotRunSource,
        error: str,
    ) -> None:
        run_dir = self.settings.run_root / run_id
        preview_evidence = self.run_store.load_preview_evidence(run_id)
        status_path = run_dir / "status.json"
        status = _load_optional_json_model(status_path, RunStatus)
        metadata_path = run_dir / "metadata.json"
        metadata = _load_optional_json_model(metadata_path, RunMetadata)
        summary_path = run_dir / "summary.json"
        summary = _load_optional_json_model(summary_path, RunSummary)
        _cleanup_owned_run_outputs(run_dir)
        failed_state = build_launch_failed_state(
            run_id=run_id,
            fixture_name=source.fixture_name,
            default_model_name=self._model_name_for_source(source),
            failure_reason=error,
            metadata=metadata,
            status=status,
            summary=summary,
            evidence=preview_evidence,
        )
        write_text_atomic(
            status_path,
            failed_state.status.model_dump_json(indent=2) + "\n",
        )
        write_text_atomic(
            summary_path,
            failed_state.summary.model_dump_json(indent=2) + "\n",
        )


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _load_optional_json_model(path: Path, model):  # type: ignore[no-untyped-def]
    if not path.exists():
        return None
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cleanup_owned_run_outputs(run_dir: Path) -> None:
    resolved_run_dir = run_dir.resolve()
    _unlink_owned_file_or_symlink(resolved_run_dir, run_dir / "events.jsonl")
    _unlink_owned_directory_files(resolved_run_dir, run_dir / "artifacts")
    _unlink_owned_directory_files(
        resolved_run_dir, run_dir / "checkpoints", suffix=".json"
    )
    _unlink_owned_directory_files(resolved_run_dir, run_dir / "diffs", suffix=".patch")


def _unlink_owned_directory_files(
    resolved_run_dir: Path,
    directory: Path,
    *,
    suffix: str | None = None,
) -> None:
    if directory.is_symlink():
        _unlink_owned_file_or_symlink(resolved_run_dir, directory)
        return
    if not directory.exists():
        return
    try:
        resolved_directory = directory.resolve()
        resolved_directory.relative_to(resolved_run_dir)
    except (OSError, ValueError):
        return
    for child in directory.rglob("*"):
        if child.is_dir() and not child.is_symlink():
            continue
        if suffix is not None and child.suffix != suffix:
            continue
        _unlink_owned_file_or_symlink(
            resolved_run_dir,
            child,
            required_parent=resolved_directory,
        )


def _unlink_owned_file_or_symlink(
    resolved_run_dir: Path,
    path: Path,
    *,
    required_parent: Path | None = None,
) -> None:
    if not path.exists() and not path.is_symlink():
        return
    try:
        path.parent.resolve().relative_to(resolved_run_dir)
    except (OSError, ValueError):
        return
    if path.is_symlink():
        path.unlink(missing_ok=True)
        return
    try:
        resolved_path = path.resolve()
        resolved_path.relative_to(resolved_run_dir)
        if required_parent is not None:
            resolved_path.relative_to(required_parent)
    except (OSError, ValueError):
        return
    if path.is_file():
        path.unlink(missing_ok=True)
