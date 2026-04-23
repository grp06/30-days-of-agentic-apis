from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .atomic_files import write_bytes_atomic, write_text_atomic
from .events import Event, dump_event, utc_now


_UNSET = object()


class RunMetadata(BaseModel):
    run_id: str
    task: str
    model_name: str
    fixture_name: str
    started_at: str = Field(default_factory=lambda: utc_now().isoformat())
    preview_expected: bool = False
    sandbox_id: str | None = None
    parent_run_id: str | None = None
    source_snapshot_id: str | None = None
    source_checkpoint_sequence: int | None = None
    instruction_override: str | None = None


class RunSummary(BaseModel):
    run_id: str
    status: str
    model_name: str
    fixture_name: str
    sandbox_id: str | None = None
    preview_url: str | None = None
    preview_state: str = "unavailable"
    preview_last_error: str | None = None
    preview_expected: bool = False
    preview_failure_reason: str | None = None
    sandbox_retained: bool = False
    checkpoint_id: str | None = None
    failure_reason: str | None = None
    command_count: int = 0
    diff_count: int = 0
    tool_call_count: int = 0
    completed_at: str = Field(default_factory=lambda: utc_now().isoformat())


class RunStatus(BaseModel):
    run_id: str
    state: str
    current_model_name: str
    latest_sequence: int = 0
    preview_url: str | None = None
    preview_state: str = "unavailable"
    preview_last_error: str | None = None
    preview_expected: bool = False
    preview_failure_reason: str | None = None
    sandbox_retained: bool = False
    preview_attempted: bool = False
    preview_refresh_allowed: bool = False
    checkpoint_id: str | None = None
    is_fork: bool = False
    updated_at: str = Field(default_factory=lambda: utc_now().isoformat())


class Recorder:
    def __init__(
        self,
        run_root: Path,
        run_id: str,
        metadata: RunMetadata,
        *,
        allow_existing: bool = False,
    ) -> None:
        self.run_root = run_root
        self.run_id = run_id
        self.run_dir = self._resolve_run_dir(run_root, run_id)
        self.artifacts_dir = self.run_dir / "artifacts"
        self.diffs_dir = self.run_dir / "diffs"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.status_path = self.run_dir / "status.json"
        self.metadata_path = self.run_dir / "metadata.json"
        self._counters: dict[str, int] = {"command": 0, "diff": 0, "tool_call": 0}
        self._status: RunStatus | None = None

        if self.run_dir.exists() and not allow_existing:
            raise FileExistsError(f"Run directory already exists: {self.run_dir}")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.diffs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        if metadata is not None:
            if allow_existing and self.metadata_path.exists():
                existing_metadata = RunMetadata.model_validate_json(
                    self._read_text(self.metadata_path)
                )
                metadata = metadata.model_copy(update={"started_at": existing_metadata.started_at})
            self._write_text(self.metadata_path, metadata.model_dump_json(indent=2) + "\n")

    def append(self, event: Event) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self._append_text_handle(self.events_path) as handle:
            handle.write(dump_event(event))
            handle.write("\n")
        if event.type.startswith("command_") and event.type != "command_stream":
            if event.type == "command_started":
                self._counters["command"] += 1
        elif event.type == "file_diff":
            self._counters["diff"] += 1
        elif event.type == "tool_call":
            self._counters["tool_call"] += 1
        if self._status is not None:
            self.update_status(latest_sequence=event.sequence)

    def write_diff(self, sequence: int, slug: str, patch: str) -> Path:
        path = self._diff_path(f"{sequence:04d}-{slug}.patch")
        self._write_text(path, patch)
        return path

    def write_checkpoint(self, sequence: int, checkpoint_id: str, extra: dict[str, Any]) -> Path:
        path = self.checkpoints_dir / f"{sequence:04d}.json"
        payload = {"snapshot_id": checkpoint_id, **extra}
        self._write_text(path, json.dumps(payload, indent=2) + "\n")
        return path

    def write_artifact_text(self, relative_name: str, content: str) -> Path:
        path = self._artifact_path(relative_name)
        self._write_text(path, content)
        return path

    def write_artifact_bytes(self, relative_name: str, content: bytes) -> Path:
        path = self._artifact_path(relative_name)
        self._write_bytes(path, content)
        return path

    def initialize_status(self, status: RunStatus) -> None:
        self._status = status
        self._write_status(status)

    @classmethod
    def open_existing(cls, run_root: Path, run_id: str) -> Recorder:
        run_dir = cls._resolve_run_dir(run_root, run_id)
        metadata = RunMetadata.model_validate_json(
            cls._read_text_static(run_dir, run_dir / "metadata.json")
        )
        recorder = cls(run_root, run_id, metadata, allow_existing=True)
        if recorder.status_path.exists():
            recorder._status = RunStatus.model_validate_json(
                recorder._read_text(recorder.status_path)
            )
        return recorder

    @property
    def current_status(self) -> RunStatus | None:
        return self._status

    def update_status(self, **fields: Any) -> None:
        if self._status is None:
            raise RuntimeError("status must be initialized before it can be updated")
        merged = self._status.model_copy(
            update={**fields, "updated_at": utc_now().isoformat()}
        )
        self._status = merged
        self._write_status(merged)

    def finalize(self, summary: RunSummary) -> None:
        merged = summary.model_copy(
            update={
                "command_count": summary.command_count or self._counters["command"],
                "diff_count": summary.diff_count or self._counters["diff"],
                "tool_call_count": summary.tool_call_count or self._counters["tool_call"],
                "completed_at": utc_now().isoformat(),
            }
        )
        preview_artifact_path = self.artifacts_dir / "preview-url.txt"
        if merged.preview_url is None:
            if preview_artifact_path.exists():
                self._unlink(preview_artifact_path)
        else:
            self.write_artifact_text("preview-url.txt", merged.preview_url + "\n")
        self._write_text(self.summary_path, merged.model_dump_json(indent=2) + "\n")
        if self._status is not None:
            self.update_status(
                state=merged.status,
                current_model_name=merged.model_name,
                preview_url=merged.preview_url,
                preview_state=merged.preview_state,
                preview_last_error=merged.preview_last_error,
                preview_expected=merged.preview_expected,
                preview_failure_reason=merged.preview_failure_reason,
                sandbox_retained=merged.sandbox_retained,
                checkpoint_id=merged.checkpoint_id,
            )

    def persist_terminal_state(self, *, status: RunStatus, summary: RunSummary) -> None:
        completed_at = utc_now().isoformat()
        merged_summary = summary.model_copy(
            update={
                "command_count": summary.command_count or self._counters["command"],
                "diff_count": summary.diff_count or self._counters["diff"],
                "tool_call_count": summary.tool_call_count or self._counters["tool_call"],
                "completed_at": completed_at,
            }
        )
        merged_status = status.model_copy(update={"updated_at": completed_at})
        preview_artifact_path = self.artifacts_dir / "preview-url.txt"
        if merged_summary.preview_url is None:
            if preview_artifact_path.exists():
                self._unlink(preview_artifact_path)
        else:
            self.write_artifact_text("preview-url.txt", merged_summary.preview_url + "\n")
        self._write_text(self.summary_path, merged_summary.model_dump_json(indent=2) + "\n")
        self._status = merged_status
        self._write_status(merged_status)

    def persist_preview_state(
        self,
        *,
        preview_state: str,
        preview_url: str | None | object = _UNSET,
        preview_last_error: str | None = None,
        preview_failure_reason: str | None | object = _UNSET,
        sandbox_retained: bool | None = None,
    ) -> None:
        preview_artifact_path = self.artifacts_dir / "preview-url.txt"
        if preview_url is not _UNSET:
            if preview_url is None:
                if preview_artifact_path.exists():
                    self._unlink(preview_artifact_path)
            else:
                self.write_artifact_text("preview-url.txt", preview_url + "\n")
        if self.summary_path.exists():
            summary = RunSummary.model_validate_json(self._read_text(self.summary_path))
            summary = summary.model_copy(
                update={
                    "preview_url": summary.preview_url if preview_url is _UNSET else preview_url,
                    "preview_state": preview_state,
                    "preview_last_error": preview_last_error,
                    "preview_failure_reason": (
                        summary.preview_failure_reason
                        if preview_failure_reason is _UNSET
                        else preview_failure_reason
                    ),
                    "sandbox_retained": (
                        sandbox_retained if sandbox_retained is not None else summary.sandbox_retained
                    ),
                }
            )
            self._write_text(self.summary_path, summary.model_dump_json(indent=2) + "\n")
        if self._status is None and self.status_path.exists():
            self._status = RunStatus.model_validate_json(self._read_text(self.status_path))
        if self._status is not None:
            self.update_status(
                preview_url=self._status.preview_url if preview_url is _UNSET else preview_url,
                preview_state=preview_state,
                preview_last_error=preview_last_error,
                preview_failure_reason=(
                    self._status.preview_failure_reason
                    if preview_failure_reason is _UNSET
                    else preview_failure_reason
                ),
                sandbox_retained=(
                    sandbox_retained if sandbox_retained is not None else self._status.sandbox_retained
                ),
            )

    def _write_status(self, status: RunStatus) -> None:
        self._write_text(self.status_path, status.model_dump_json(indent=2) + "\n")

    def _diff_path(self, relative_name: str) -> Path:
        return self._owned_child_path(
            self.diffs_dir,
            relative_name,
            "diffs",
        )

    def _artifact_path(self, relative_name: str) -> Path:
        return self._owned_child_path(
            self.artifacts_dir,
            relative_name,
            "artifacts",
        )

    def _owned_child_path(
        self,
        root: Path,
        relative_name: str,
        label: str,
    ) -> Path:
        child_root = root.resolve()
        path = (child_root / relative_name).resolve()
        try:
            path.relative_to(child_root)
        except ValueError as exc:
            raise FileNotFoundError(
                f"Path not owned by run {label}: {relative_name}"
            ) from exc
        return path

    @staticmethod
    def _resolve_run_dir(run_root: Path, run_id: str) -> Path:
        resolved_root = run_root.resolve()
        run_dir = (resolved_root / run_id).resolve()
        try:
            run_dir.relative_to(resolved_root)
        except ValueError as exc:
            raise FileNotFoundError(f"Run directory not found: {run_root / run_id}") from exc
        return run_dir

    @staticmethod
    def _read_text_static(run_dir: Path, path: Path) -> str:
        resolved = Recorder._owned_path(run_dir, path)
        return resolved.read_text(encoding="utf-8")

    def _read_text(self, path: Path) -> str:
        return self._read_text_static(self.run_dir, path)

    def _write_text(self, path: Path, content: str) -> None:
        resolved = self._owned_path(self.run_dir, path)
        write_text_atomic(resolved, content)

    def _write_bytes(self, path: Path, content: bytes) -> None:
        resolved = self._owned_path(self.run_dir, path)
        write_bytes_atomic(resolved, content)

    def _append_text_handle(self, path: Path):  # type: ignore[no-untyped-def]
        resolved = self._owned_path(self.run_dir, path)
        return resolved.open("a", encoding="utf-8")

    def _unlink(self, path: Path) -> None:
        resolved = self._owned_path(self.run_dir, path)
        resolved.unlink()

    @staticmethod
    def _owned_path(run_dir: Path, path: Path) -> Path:
        resolved_parent = path.parent.resolve()
        try:
            resolved_parent.relative_to(run_dir)
        except ValueError as exc:
            raise FileNotFoundError(f"Path not owned by run directory: {path}") from exc
        if path.exists():
            resolved = path.resolve()
            try:
                resolved.relative_to(run_dir)
            except ValueError as exc:
                raise FileNotFoundError(f"Path not owned by run directory: {path}") from exc
            return resolved
        return resolved_parent / path.name


def prepare_run_directory(
    run_root: Path,
    run_id: str,
    metadata: RunMetadata,
    status: RunStatus,
) -> Path:
    recorder = Recorder(run_root, run_id, metadata)
    recorder.initialize_status(status)
    return recorder.run_dir
