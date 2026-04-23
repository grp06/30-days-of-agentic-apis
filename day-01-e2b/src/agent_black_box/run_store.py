from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .fixture_policy import fixture_preview_port
from .preview_execution import PreviewEvidence, build_preview_evidence
from .recorder import RunMetadata, RunStatus, RunSummary
from .run_projection import LoadedRunFacts, ProjectedRun, RunListItem, project_run


class RunStore:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root

    def list_runs(self) -> list[RunListItem]:
        items: list[RunListItem] = []
        children_by_parent = self._children_by_parent()
        for run_dir in sorted(self._iter_run_dirs(), reverse=True):
            projected = self.load_projected_run(
                run_dir.name,
                child_run_ids=children_by_parent.get(run_dir.name, []),
                checkpoint_count=0,
            )
            if projected is None:
                continue
            items.append(projected.list_item)
        return items

    def get_run_dir(self, run_id: str) -> Path:
        run_root = self.run_root.resolve()
        run_dir = (run_root / run_id).resolve()
        try:
            run_dir.relative_to(run_root)
        except ValueError as exc:
            raise FileNotFoundError(f"Run directory not found: {self.run_root / run_id}") from exc
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    def load_metadata(self, run_id: str) -> RunMetadata:
        return self._load_run_json(run_id, "metadata.json", RunMetadata)

    def load_summary(self, run_id: str) -> RunSummary | None:
        projected = self.load_projected_run(run_id)
        return projected.summary if projected is not None else None

    def load_status(self, run_id: str) -> RunStatus | None:
        projected = self.load_projected_run(run_id)
        return projected.status if projected is not None else None

    def load_projected_run(
        self,
        run_id: str,
        *,
        child_run_ids: list[str] | None = None,
        checkpoint_count: int | None = None,
    ) -> ProjectedRun | None:
        status = self._load_optional_run_json(run_id, "status.json", RunStatus)
        summary = self._load_optional_run_json(run_id, "summary.json", RunSummary)
        metadata = self._load_optional_run_json(run_id, "metadata.json", RunMetadata)
        if metadata is None or (status is None and summary is None):
            return None
        event_lines = self.load_event_lines(run_id)
        checkpoints = self.list_checkpoints(run_id)
        return project_run(
            LoadedRunFacts(
                metadata=metadata,
                status=status,
                summary=summary,
                preview_evidence=build_preview_evidence(
                    status=status,
                    summary=summary,
                    event_lines=event_lines,
                    default_preview_port=self._default_preview_port(metadata),
                ),
                artifact_names=self.list_artifacts(run_id),
                child_run_ids=child_run_ids if child_run_ids is not None else self.list_children(run_id),
                checkpoint_count=checkpoint_count if checkpoint_count is not None else len(checkpoints),
                latest_checkpoint_sequence=_latest_checkpoint_sequence(checkpoints),
                event_lines=event_lines,
            )
        )

    def load_preview_evidence(self, run_id: str) -> PreviewEvidence:
        summary = self._load_optional_run_json(run_id, "summary.json", RunSummary)
        status = self._load_optional_run_json(run_id, "status.json", RunStatus)
        metadata = self._load_optional_run_json(run_id, "metadata.json", RunMetadata)
        return build_preview_evidence(
            status=status,
            summary=summary,
            event_lines=self.load_event_lines(run_id),
            default_preview_port=self._default_preview_port(metadata),
        )

    def infer_preview_url(self, run_id: str) -> str | None:
        return self.load_preview_evidence(run_id).preview_url

    def list_children(self, run_id: str) -> list[str]:
        return self._children_by_parent().get(run_id, [])

    def resolve_diff(self, run_id: str, diff_id: str) -> Path:
        diff_dir = self._diff_dir(run_id)
        diff_path = (diff_dir / diff_id).resolve()
        try:
            diff_path.relative_to(diff_dir.resolve())
        except ValueError as exc:
            raise FileNotFoundError(f"Diff not found: {diff_dir / diff_id}") from exc
        if diff_path.suffix != ".patch" or not diff_path.exists():
            raise FileNotFoundError(f"Diff not found: {diff_path}")
        return diff_path

    def resolve_artifact(self, run_id: str, artifact_name: str) -> Path:
        artifact_dir = self._artifact_dir(run_id)
        artifact_path = (artifact_dir / artifact_name).resolve()
        try:
            artifact_path.relative_to(artifact_dir.resolve())
        except ValueError as exc:
            raise FileNotFoundError(f"Artifact not found: {artifact_dir / artifact_name}") from exc
        if not artifact_path.exists() or not artifact_path.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        return artifact_path

    def list_artifacts(self, run_id: str) -> list[str]:
        try:
            artifact_dir = self._artifact_dir(run_id)
        except FileNotFoundError:
            return []
        if not artifact_dir.exists():
            return []
        artifact_names: list[str] = []
        resolved_artifact_dir = artifact_dir.resolve()
        for path in artifact_dir.rglob("*"):
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
                relative = resolved.relative_to(resolved_artifact_dir)
            except ValueError:
                continue
            artifact_names.append(relative.as_posix())
        return sorted(artifact_names)

    def list_checkpoints(self, run_id: str) -> list[Path]:
        try:
            checkpoint_dir = self._checkpoint_dir(run_id)
        except FileNotFoundError:
            return []
        if not checkpoint_dir.exists():
            return []
        checkpoints: list[Path] = []
        for path in checkpoint_dir.glob("*.json"):
            try:
                resolved = path.resolve()
                resolved.relative_to(checkpoint_dir.resolve())
            except ValueError:
                continue
            if _checkpoint_sequence(resolved) is None:
                continue
            try:
                self.load_checkpoint_payload(resolved)
            except FileNotFoundError:
                continue
            checkpoints.append(resolved)
        return sorted(checkpoints)

    def resolve_checkpoint(self, run_id: str, checkpoint_sequence: int) -> Path:
        try:
            checkpoint_dir = self._checkpoint_dir(run_id)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Checkpoint {checkpoint_sequence} not found for run {run_id}"
            ) from exc
        checkpoint_path = (checkpoint_dir / f"{checkpoint_sequence:04d}.json").resolve()
        try:
            checkpoint_path.relative_to(checkpoint_dir.resolve())
        except ValueError as exc:
            raise FileNotFoundError(
                f"Checkpoint {checkpoint_sequence} not found for run {run_id}"
            ) from exc
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint {checkpoint_sequence} not found for run {run_id}"
            )
        return checkpoint_path

    def load_checkpoint_payload(self, checkpoint_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FileNotFoundError(
                f"Checkpoint payload not readable: {checkpoint_path.name}"
            ) from exc
        if not isinstance(payload, dict):
            raise FileNotFoundError(
                f"Checkpoint payload is not an object: {checkpoint_path.name}"
            )
        snapshot_id = payload.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise FileNotFoundError(
                f"Checkpoint payload missing snapshot_id: {checkpoint_path.name}"
            )
        return payload

    def load_event_lines(self, run_id: str) -> list[str]:
        try:
            events_path = self._run_file(run_id, "events.jsonl")
        except FileNotFoundError:
            return []
        if not events_path.exists():
            return []
        try:
            return events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

    def infer_preview_port(self, run_id: str) -> int | None:
        return self.load_preview_evidence(run_id).preview_port

    def infer_preview_failure_reason(self, run_id: str) -> str | None:
        return self.load_preview_evidence(run_id).preview_failure_reason

    def _default_preview_port(self, metadata: RunMetadata | None) -> int:
        if metadata is None:
            return 4173
        return fixture_preview_port(metadata.fixture_name, default=4173)

    def _iter_run_dirs(self) -> list[Path]:
        if not self.run_root.exists():
            return []
        run_dirs: list[Path] = []
        for path in self.run_root.iterdir():
            if not path.is_dir():
                continue
            try:
                run_dirs.append(self.get_run_dir(path.name))
            except FileNotFoundError:
                continue
        return run_dirs

    def _children_by_parent(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for run_dir in self._iter_run_dirs():
            metadata = self._load_optional_run_json(run_dir.name, "metadata.json", RunMetadata)
            if metadata is None or metadata.parent_run_id is None:
                continue
            mapping.setdefault(metadata.parent_run_id, []).append(metadata.run_id)
        for child_ids in mapping.values():
            child_ids.sort(reverse=True)
        return mapping

    def _load_json(self, path: Path, model):  # type: ignore[no-untyped-def]
        return model.model_validate_json(path.read_text(encoding="utf-8"))

    def _load_run_json(self, run_id: str, relative_name: str, model):  # type: ignore[no-untyped-def]
        return self._load_json(self._run_file(run_id, relative_name), model)

    def _load_optional_run_json(self, run_id: str, relative_name: str, model):  # type: ignore[no-untyped-def]
        try:
            path = self._run_file(run_id, relative_name)
        except FileNotFoundError:
            return None
        if not path.exists():
            return None
        try:
            return self._load_json(path, model)
        except (OSError, ValueError):
            return None

    def _checkpoint_dir(self, run_id: str) -> Path:
        run_dir = self.get_run_dir(run_id)
        checkpoint_dir = (run_dir / "checkpoints").resolve()
        try:
            checkpoint_dir.relative_to(run_dir)
        except ValueError as exc:
            raise FileNotFoundError(f"Checkpoint directory not found for run {run_id}") from exc
        return checkpoint_dir

    def _diff_dir(self, run_id: str) -> Path:
        run_dir = self.get_run_dir(run_id)
        diff_dir = (run_dir / "diffs").resolve()
        try:
            diff_dir.relative_to(run_dir)
        except ValueError as exc:
            raise FileNotFoundError(f"Diff directory not found for run {run_id}") from exc
        return diff_dir

    def _artifact_dir(self, run_id: str) -> Path:
        run_dir = self.get_run_dir(run_id)
        artifact_dir = (run_dir / "artifacts").resolve()
        try:
            artifact_dir.relative_to(run_dir)
        except ValueError as exc:
            raise FileNotFoundError(f"Artifact directory not found for run {run_id}") from exc
        return artifact_dir

    def _run_file(self, run_id: str, relative_name: str) -> Path:
        run_dir = self.get_run_dir(run_id)
        file_path = (run_dir / relative_name).resolve()
        try:
            file_path.relative_to(run_dir)
        except ValueError as exc:
            raise FileNotFoundError(f"{relative_name} not found for run {run_id}") from exc
        return file_path


def _latest_checkpoint_sequence(checkpoints: list[Path]) -> int | None:
    sequences = [
        sequence
        for checkpoint in checkpoints
        if (sequence := _checkpoint_sequence(checkpoint)) is not None
    ]
    if not sequences:
        return None
    return max(sequences)


def _checkpoint_sequence(checkpoint_path: Path) -> int | None:
    try:
        return int(checkpoint_path.stem)
    except ValueError:
        return None
