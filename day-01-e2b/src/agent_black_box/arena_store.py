from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from .arena import ArenaMetadata, ArenaStatus
from .run_projection import ArenaLaneSummary, project_arena_lane
from .run_store import RunStore


class ArenaProjection(BaseModel):
    metadata: ArenaMetadata
    status: ArenaStatus
    demo_summary: str | None = None
    recommended_lane_id: str | None = None
    lanes: list[ArenaLaneSummary]


class ArenaListItem(BaseModel):
    arena_id: str
    fixture_name: str
    state: str
    total_lanes: int
    completed_lanes: int


class ArenaStore:
    def __init__(self, arena_root: Path, run_store: RunStore) -> None:
        self.arena_root = arena_root
        self.run_store = run_store

    def list_arena_ids(self) -> list[str]:
        if not self.arena_root.exists():
            return []
        arena_ids: list[str] = []
        for path in self.arena_root.iterdir():
            if not path.is_dir():
                continue
            try:
                self.get_arena_dir(path.name)
            except FileNotFoundError:
                continue
            arena_ids.append(path.name)
        return sorted(arena_ids, reverse=True)

    def get_arena_dir(self, arena_id: str) -> Path:
        arena_root = self.arena_root.resolve()
        arena_dir = (arena_root / arena_id).resolve()
        try:
            arena_dir.relative_to(arena_root)
        except ValueError as exc:
            raise FileNotFoundError(f"Arena directory not found: {self.arena_root / arena_id}") from exc
        if not arena_dir.exists():
            raise FileNotFoundError(f"Arena directory not found: {arena_dir}")
        return arena_dir

    def load_metadata(self, arena_id: str) -> ArenaMetadata:
        return self._load_arena_json(arena_id, "metadata.json", ArenaMetadata)

    def load_status(self, arena_id: str) -> ArenaStatus | None:
        return self._load_optional_arena_json(arena_id, "status.json", ArenaStatus)

    def load_lane_summaries(self, arena_id: str) -> list[ArenaLaneSummary]:
        metadata = self.load_metadata(arena_id)
        return [self._load_lane_summary(lane.lane_id, lane.run_id) for lane in metadata.lanes]

    def _load_lane_summary(self, lane_id: str, run_id: str) -> ArenaLaneSummary:
        try:
            projected = self.run_store.load_projected_run(run_id)
        except FileNotFoundError:
            projected = None
        if projected is None:
            return ArenaLaneSummary(lane_id=lane_id, run_id=run_id, state="missing")
        return project_arena_lane(projected, lane_id=lane_id)

    def _load_arena_json(self, arena_id: str, relative_name: str, model):  # type: ignore[no-untyped-def]
        return model.model_validate_json(
            self._arena_file(arena_id, relative_name).read_text(encoding="utf-8")
        )

    def _load_optional_arena_json(self, arena_id: str, relative_name: str, model):  # type: ignore[no-untyped-def]
        try:
            path = self._arena_file(arena_id, relative_name)
        except FileNotFoundError:
            return None
        if not path.exists():
            return None
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _arena_file(self, arena_id: str, relative_name: str) -> Path:
        arena_dir = self.get_arena_dir(arena_id)
        file_path = (arena_dir / relative_name).resolve()
        try:
            file_path.relative_to(arena_dir)
        except ValueError as exc:
            raise FileNotFoundError(f"{relative_name} not found for arena {arena_id}") from exc
        return file_path
