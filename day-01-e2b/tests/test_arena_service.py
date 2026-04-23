from __future__ import annotations

import json

from agent_black_box.arena import ArenaMetadata, ArenaLaneRecord, ArenaRecorder, ArenaStatus
from agent_black_box.arena_service import ArenaService
from agent_black_box.config import Settings
from agent_black_box.recorder import RunMetadata, RunStatus, RunSummary, prepare_run_directory


def _write_run(tmp_path, run_id: str, state: str) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_root.mkdir(exist_ok=True)
    run_dir = prepare_run_directory(
        run_root,
        run_id,
        RunMetadata(
            run_id=run_id,
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id=f"sbx-{run_id}",
        ),
        RunStatus(run_id=run_id, state=state, current_model_name="kimi-k2.6:cloud"),
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id=run_id,
            status=state,
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id=f"sbx-{run_id}",
            failure_reason="boom" if state != "succeeded" else None,
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )


def test_arena_service_reconciles_terminal_lane_state(tmp_path) -> None:  # noqa: ANN001
    arena_root = tmp_path / "arenas"
    arena_root.mkdir()
    _write_run(tmp_path, "run-1", "succeeded")
    _write_run(tmp_path, "run-2", "failed")

    recorder = ArenaRecorder(
        arena_root,
        "arena-1",
        ArenaMetadata(
            arena_id="arena-1",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[
                ArenaLaneRecord(lane_id="lane-1", run_id="run-1"),
                ArenaLaneRecord(lane_id="lane-2", run_id="run-2"),
            ],
        ),
    )
    recorder.initialize_status(
        ArenaStatus(
            arena_id="arena-1",
            state="running",
            total_lanes=2,
            lane_states={"lane-1": "running", "lane-2": "running"},
        )
    )

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=arena_root,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = ArenaService(settings)

    projection = service.get_arena("arena-1")
    persisted = json.loads((arena_root / "arena-1" / "status.json").read_text(encoding="utf-8"))

    assert projection.status.state == "completed_with_failures"
    assert projection.status.completed_lanes == 2
    assert projection.demo_summary is not None
    assert projection.recommended_lane_id in {"lane-1", "lane-2"}
    assert persisted["state"] == "completed_with_failures"


def test_arena_service_does_not_rewrite_status_when_nothing_changed(tmp_path) -> None:  # noqa: ANN001
    arena_root = tmp_path / "arenas"
    arena_root.mkdir()
    _write_run(tmp_path, "run-1", "succeeded")

    recorder = ArenaRecorder(
        arena_root,
        "arena-1",
        ArenaMetadata(
            arena_id="arena-1",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[ArenaLaneRecord(lane_id="lane-1", run_id="run-1")],
        ),
    )
    recorder.initialize_status(
        ArenaStatus(
            arena_id="arena-1",
            state="succeeded",
            total_lanes=1,
            completed_lanes=1,
            lane_states={"lane-1": "succeeded"},
            updated_at="2026-04-21T22:00:00+00:00",
        )
    )

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=arena_root,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = ArenaService(settings)

    projection = service.get_arena("arena-1")
    persisted = json.loads((arena_root / "arena-1" / "status.json").read_text(encoding="utf-8"))

    assert projection.status.updated_at == "2026-04-21T22:00:00+00:00"
    assert persisted["updated_at"] == "2026-04-21T22:00:00+00:00"


def test_arena_service_treats_missing_lanes_as_terminal_failures(tmp_path) -> None:  # noqa: ANN001
    arena_root = tmp_path / "arenas"
    arena_root.mkdir()
    _write_run(tmp_path, "run-1", "succeeded")

    recorder = ArenaRecorder(
        arena_root,
        "arena-1",
        ArenaMetadata(
            arena_id="arena-1",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[
                ArenaLaneRecord(lane_id="lane-1", run_id="run-1"),
                ArenaLaneRecord(lane_id="lane-2", run_id="missing-run"),
            ],
        ),
    )
    recorder.initialize_status(
        ArenaStatus(
            arena_id="arena-1",
            state="running",
            total_lanes=2,
            lane_states={"lane-1": "running", "lane-2": "running"},
        )
    )

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=arena_root,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = ArenaService(settings)

    projection = service.get_arena("arena-1")

    assert projection.status.state == "completed_with_failures"
    assert projection.status.completed_lanes == 2
    assert projection.status.lane_states["lane-2"] == "missing"


def test_arena_service_skips_corrupt_arenas_when_listing(tmp_path) -> None:  # noqa: ANN001
    arena_root = tmp_path / "arenas"
    arena_root.mkdir()
    _write_run(tmp_path, "run-1", "succeeded")

    recorder = ArenaRecorder(
        arena_root,
        "arena-1",
        ArenaMetadata(
            arena_id="arena-1",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[ArenaLaneRecord(lane_id="lane-1", run_id="run-1")],
        ),
    )
    recorder.initialize_status(
        ArenaStatus(
            arena_id="arena-1",
            state="succeeded",
            total_lanes=1,
            completed_lanes=1,
            lane_states={"lane-1": "succeeded"},
        )
    )
    corrupt_dir = arena_root / "arena-bad"
    corrupt_dir.mkdir()
    (corrupt_dir / "metadata.json").write_text('{"arena_id":', encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=arena_root,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = ArenaService(settings)

    assert [item.arena_id for item in service.list_arenas()] == ["arena-1"]
