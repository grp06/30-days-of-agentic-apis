from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .atomic_files import write_text_atomic
from .events import utc_now


class ArenaLaneRecord(BaseModel):
    lane_id: str
    run_id: str
    model_name: str | None = None


class ArenaMetadata(BaseModel):
    arena_id: str
    fixture_name: str
    task: str
    started_at: str = Field(default_factory=lambda: utc_now().isoformat())
    lanes: list[ArenaLaneRecord]


class ArenaStatus(BaseModel):
    arena_id: str
    state: str
    total_lanes: int
    completed_lanes: int = 0
    lane_states: dict[str, str] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=lambda: utc_now().isoformat())


class ArenaRecorder:
    def __init__(
        self,
        arena_root: Path,
        arena_id: str,
        metadata: ArenaMetadata | None = None,
        *,
        allow_existing: bool = False,
    ) -> None:
        self.arena_root = arena_root
        self.arena_id = arena_id
        self.arena_dir = self._resolve_arena_dir(arena_root, arena_id)
        self.metadata_path = self.arena_dir / "metadata.json"
        self.status_path = self.arena_dir / "status.json"
        self._status: ArenaStatus | None = None

        if self.arena_dir.exists() and not allow_existing:
            raise FileExistsError(f"Arena directory already exists: {self.arena_dir}")
        self.arena_dir.mkdir(parents=True, exist_ok=True)
        if metadata is not None:
            self._write_text(
                self.metadata_path, metadata.model_dump_json(indent=2) + "\n"
            )

    @classmethod
    def open_existing(cls, arena_root: Path, arena_id: str) -> ArenaRecorder:
        return cls(arena_root, arena_id, allow_existing=True)

    def initialize_status(self, status: ArenaStatus) -> None:
        self._status = status
        self._write_status(status)

    def update_status(self, **fields: object) -> None:
        if self._status is None:
            if not self.status_path.exists():
                raise RuntimeError(
                    "arena status must be initialized before it can be updated"
                )
            self._status = ArenaStatus.model_validate_json(
                self._read_text(self.status_path)
            )
        merged = self._status.model_copy(
            update={**fields, "updated_at": utc_now().isoformat()}
        )
        self._status = merged
        self._write_status(merged)

    def _write_status(self, status: ArenaStatus) -> None:
        self._write_text(self.status_path, status.model_dump_json(indent=2) + "\n")

    @staticmethod
    def _resolve_arena_dir(arena_root: Path, arena_id: str) -> Path:
        resolved_root = arena_root.resolve()
        arena_dir = (resolved_root / arena_id).resolve()
        try:
            arena_dir.relative_to(resolved_root)
        except ValueError as exc:
            raise FileNotFoundError(
                f"Arena directory not found: {arena_root / arena_id}"
            ) from exc
        return arena_dir

    def _read_text(self, path: Path) -> str:
        resolved = self._owned_path(path)
        return resolved.read_text(encoding="utf-8")

    def _write_text(self, path: Path, content: str) -> None:
        resolved = self._owned_path(path)
        write_text_atomic(resolved, content)

    def _owned_path(self, path: Path) -> Path:
        resolved_parent = path.parent.resolve()
        try:
            resolved_parent.relative_to(self.arena_dir)
        except ValueError as exc:
            raise FileNotFoundError(
                f"Path not owned by arena directory: {path}"
            ) from exc
        if path.exists():
            resolved = path.resolve()
            try:
                resolved.relative_to(self.arena_dir)
            except ValueError as exc:
                raise FileNotFoundError(
                    f"Path not owned by arena directory: {path}"
                ) from exc
            return resolved
        return resolved_parent / path.name
