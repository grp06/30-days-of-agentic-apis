from __future__ import annotations

from .arena import ArenaRecorder, ArenaStatus
from .arena_store import ArenaListItem, ArenaProjection, ArenaStore
from .config import Settings
from .launch import ArenaLaunchResponse, RunLauncher
from .run_store import RunStore


TERMINAL_STATES = {"succeeded", "failed", "provider_interrupted", "launch_failed"}
AGGREGATE_TERMINAL_STATES = TERMINAL_STATES | {"missing"}


class ArenaService:
    def __init__(
        self,
        settings: Settings,
        launcher: RunLauncher | None = None,
        arena_store: ArenaStore | None = None,
        run_store: RunStore | None = None,
    ) -> None:
        self.settings = settings
        self.run_store = run_store or RunStore(settings.run_root)
        self.launcher = launcher or RunLauncher(settings)
        self.arena_store = arena_store or ArenaStore(settings.arena_root, self.run_store)

    async def create_arena(self, fixture_name: str, task_override: str, lane_count: int = 4) -> ArenaLaunchResponse:
        return await self.launcher.start_arena_run(
            fixture_name=fixture_name,
            task=task_override,
            lane_count=lane_count,
        )

    def list_arenas(self) -> list[ArenaListItem]:
        items: list[ArenaListItem] = []
        for arena_id in self.arena_store.list_arena_ids():
            try:
                projection = self.get_arena(arena_id)
            except (FileNotFoundError, ValueError):
                continue
            items.append(
                ArenaListItem(
                    arena_id=projection.metadata.arena_id,
                    fixture_name=projection.metadata.fixture_name,
                    state=projection.status.state,
                    total_lanes=projection.status.total_lanes,
                    completed_lanes=projection.status.completed_lanes,
                )
            )
        return items

    def get_arena(self, arena_id: str) -> ArenaProjection:
        metadata = self.arena_store.load_metadata(arena_id)
        lanes = self.arena_store.load_lane_summaries(arena_id)
        status = self._reconciled_status(metadata.arena_id, metadata.lanes, lanes)

        persisted_status = self.arena_store.load_status(arena_id)
        if persisted_status is None:
            recorder = ArenaRecorder.open_existing(self.settings.arena_root, arena_id)
            recorder.initialize_status(status)
        elif self._status_changed(persisted_status, status):
            recorder = ArenaRecorder.open_existing(self.settings.arena_root, arena_id)
            recorder.update_status(
                state=status.state,
                total_lanes=status.total_lanes,
                completed_lanes=status.completed_lanes,
                lane_states=status.lane_states,
            )
            reloaded_status = self.arena_store.load_status(arena_id)
            if reloaded_status is None:
                raise FileNotFoundError(f"Arena status not found: {arena_id}")
            status = reloaded_status
        else:
            status = persisted_status

        return ArenaProjection(
            metadata=metadata,
            status=status,
            demo_summary=self._demo_summary(lanes, status.state),
            recommended_lane_id=self._recommended_lane_id(lanes),
            lanes=lanes,
        )

    def _reconciled_status(self, arena_id: str, lane_records, lanes) -> ArenaStatus:  # type: ignore[no-untyped-def]
        lane_states = {lane.lane_id: lane.state for lane in lanes}
        total_lanes = len(lane_records)
        completed_lanes = sum(1 for lane in lanes if lane.state in AGGREGATE_TERMINAL_STATES)
        if completed_lanes < total_lanes:
            state = "running"
        elif all(lane.state == "succeeded" for lane in lanes):
            state = "succeeded"
        else:
            state = "completed_with_failures"
        return ArenaStatus(
            arena_id=arena_id,
            state=state,
            total_lanes=total_lanes,
            completed_lanes=completed_lanes,
            lane_states=lane_states,
        )

    def _status_changed(self, persisted: ArenaStatus, next_status: ArenaStatus) -> bool:
        return (
            persisted.state != next_status.state
            or persisted.total_lanes != next_status.total_lanes
            or persisted.completed_lanes != next_status.completed_lanes
            or persisted.lane_states != next_status.lane_states
        )

    def _recommended_lane_id(self, lanes) -> str | None:  # type: ignore[no-untyped-def]
        if not lanes:
            return None
        for lane in lanes:
            if lane.state == "succeeded" and lane.preview_url:
                return lane.lane_id
        for lane in lanes:
            if lane.state == "succeeded":
                return lane.lane_id
        for lane in lanes:
            if lane.preview_url:
                return lane.lane_id
        for lane in lanes:
            if lane.failure_reason:
                return lane.lane_id
        return lanes[0].lane_id

    def _demo_summary(self, lanes, arena_state: str) -> str:  # type: ignore[no-untyped-def]
        if not lanes:
            return "No lanes have been recorded for this arena yet."
        succeeded = sum(1 for lane in lanes if lane.state == "succeeded")
        with_preview = sum(
            1 for lane in lanes if lane.preview_state in {"live", "retained"} and lane.preview_url
        )
        failures = sum(1 for lane in lanes if lane.state not in {"running", "succeeded"})
        if arena_state == "running":
            return f"{succeeded} lane(s) have already finished while the rest are still running."
        if failures == 0:
            return f"All {len(lanes)} lanes finished successfully; {with_preview} published a preview."
        return (
            f"{succeeded} lane(s) succeeded, {failures} lane(s) failed or stalled, "
            f"and {with_preview} produced preview evidence."
        )
