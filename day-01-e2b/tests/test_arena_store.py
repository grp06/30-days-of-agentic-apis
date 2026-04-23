from __future__ import annotations

import json

from agent_black_box.arena import ArenaMetadata, ArenaLaneRecord, ArenaRecorder, ArenaStatus
from agent_black_box.arena_store import ArenaStore
from agent_black_box.recorder import RunMetadata, RunStatus, RunSummary, prepare_run_directory
from agent_black_box.run_store import RunStore


def test_arena_store_loads_concise_lane_summaries(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    arena_root = tmp_path / "arenas"
    run_root.mkdir()
    arena_root.mkdir()

    for lane_id, run_id, state in [("lane-1", "run-1", "running"), ("lane-2", "run-2", "succeeded")]:
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
        (run_dir / "checkpoints" / "0007.json").write_text(
            json.dumps({"snapshot_id": f"snap-{run_id}", "note": lane_id}, indent=2) + "\n",
            encoding="utf-8",
        )
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
        ArenaStatus(arena_id="arena-1", state="running", total_lanes=2, lane_states={"lane-1": "running", "lane-2": "running"})
    )

    store = ArenaStore(arena_root, RunStore(run_root))
    lanes = store.load_lane_summaries("arena-1")

    assert [lane.lane_id for lane in lanes] == ["lane-1", "lane-2"]
    assert lanes[0].checkpoint_count == 1
    assert lanes[0].latest_checkpoint_sequence == 7
    assert lanes[0].lifecycle_steps[0].label == "Sandbox"
    assert lanes[1].state == "succeeded"


def test_arena_store_prefers_richer_summary_preview_truth_when_status_is_stale(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    arena_root = tmp_path / "arenas"
    run_root.mkdir()
    arena_root.mkdir()

    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-run-1",
        ),
        RunStatus(
            run_id="run-1",
            state="succeeded",
            current_model_name="kimi-k2.6:cloud",
            preview_state="unavailable",
            sandbox_retained=False,
        ),
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-run-1",
            preview_url="https://preview.example",
            preview_state="retained",
            sandbox_retained=True,
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

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
        ArenaStatus(arena_id="arena-1", state="succeeded", total_lanes=1, completed_lanes=1, lane_states={"lane-1": "succeeded"})
    )

    store = ArenaStore(arena_root, RunStore(run_root))
    lane = store.load_lane_summaries("arena-1")[0]

    assert lane.preview_state == "retained"
    assert lane.preview_url == "https://preview.example"
    assert lane.sandbox_retained is True


def test_arena_store_rejects_arena_dir_path_traversal(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    arena_root = tmp_path / "arenas"
    run_root.mkdir()
    arena_root.mkdir()
    outside_dir = tmp_path / "outside-arena"
    outside_dir.mkdir()

    store = ArenaStore(arena_root, RunStore(run_root))

    try:
        store.get_arena_dir("../outside-arena")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected get_arena_dir to reject path traversal")


def test_arena_store_list_arena_ids_skips_symlinked_dirs_outside_arena_root(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    arena_root = tmp_path / "arenas"
    run_root.mkdir()
    arena_root.mkdir()

    recorder = ArenaRecorder(
        arena_root,
        "arena-1",
        ArenaMetadata(
            arena_id="arena-1",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[],
        ),
    )
    recorder.initialize_status(ArenaStatus(arena_id="arena-1", state="running", total_lanes=0))

    outside_dir = tmp_path / "outside-arena"
    outside_dir.mkdir()
    (outside_dir / "metadata.json").write_text(
        ArenaMetadata(
            arena_id="outside-arena",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[],
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (arena_root / "symlink-arena").symlink_to(outside_dir, target_is_directory=True)

    store = ArenaStore(arena_root, RunStore(run_root))

    assert store.list_arena_ids() == ["arena-1"]


def test_arena_store_rejects_metadata_symlink_outside_arena_root(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    arena_root = tmp_path / "arenas"
    run_root.mkdir()
    arena_root.mkdir()

    recorder = ArenaRecorder(
        arena_root,
        "arena-1",
        ArenaMetadata(
            arena_id="arena-1",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[],
        ),
    )
    recorder.initialize_status(ArenaStatus(arena_id="arena-1", state="running", total_lanes=0))

    outside_metadata = tmp_path / "outside-arena-metadata.json"
    outside_metadata.write_text(
        ArenaMetadata(
            arena_id="outside-arena",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[],
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    metadata_path = arena_root / "arena-1" / "metadata.json"
    metadata_path.unlink()
    metadata_path.symlink_to(outside_metadata)

    store = ArenaStore(arena_root, RunStore(run_root))

    try:
        store.load_metadata("arena-1")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected load_metadata to reject metadata symlink traversal")


def test_arena_store_rejects_status_symlink_outside_arena_root(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    arena_root = tmp_path / "arenas"
    run_root.mkdir()
    arena_root.mkdir()

    recorder = ArenaRecorder(
        arena_root,
        "arena-1",
        ArenaMetadata(
            arena_id="arena-1",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[],
        ),
    )
    recorder.initialize_status(ArenaStatus(arena_id="arena-1", state="running", total_lanes=0))

    outside_status = tmp_path / "outside-arena-status.json"
    outside_status.write_text(
        ArenaStatus(arena_id="outside-arena", state="succeeded", total_lanes=1, completed_lanes=1).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    status_path = arena_root / "arena-1" / "status.json"
    status_path.unlink()
    status_path.symlink_to(outside_status)

    store = ArenaStore(arena_root, RunStore(run_root))

    assert store.load_status("arena-1") is None


def test_arena_store_treats_malformed_optional_status_as_missing(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    arena_root = tmp_path / "arenas"
    run_root.mkdir()
    arena_root.mkdir()

    recorder = ArenaRecorder(
        arena_root,
        "arena-1",
        ArenaMetadata(
            arena_id="arena-1",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[],
        ),
    )
    recorder.initialize_status(ArenaStatus(arena_id="arena-1", state="running", total_lanes=0))
    (arena_root / "arena-1" / "status.json").write_text('{"arena_id":', encoding="utf-8")

    store = ArenaStore(arena_root, RunStore(run_root))

    assert store.load_status("arena-1") is None


def test_arena_store_treats_unreadable_optional_status_as_missing(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    arena_root = tmp_path / "arenas"
    run_root.mkdir()
    arena_root.mkdir()

    recorder = ArenaRecorder(
        arena_root,
        "arena-1",
        ArenaMetadata(
            arena_id="arena-1",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[],
        ),
    )
    recorder.initialize_status(ArenaStatus(arena_id="arena-1", state="running", total_lanes=0))
    status_path = arena_root / "arena-1" / "status.json"
    status_path.unlink()
    status_path.mkdir()

    store = ArenaStore(arena_root, RunStore(run_root))

    assert store.load_status("arena-1") is None


def test_arena_recorder_update_status_rejects_status_symlink_outside_arena_root(tmp_path) -> None:  # noqa: ANN001
    arena_root = tmp_path / "arenas"
    arena_root.mkdir()

    recorder = ArenaRecorder(
        arena_root,
        "arena-1",
        ArenaMetadata(
            arena_id="arena-1",
            fixture_name="sample_frontend_task",
            task="shared task",
            lanes=[],
        ),
    )
    recorder.initialize_status(ArenaStatus(arena_id="arena-1", state="running", total_lanes=0))

    outside_status = tmp_path / "outside-arena-status.json"
    outside_status.write_text(
        ArenaStatus(arena_id="outside-arena", state="succeeded", total_lanes=1, completed_lanes=1).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    status_path = arena_root / "arena-1" / "status.json"
    status_path.unlink()
    status_path.symlink_to(outside_status)

    try:
        recorder.update_status(state="succeeded")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected update_status to reject status symlink traversal")
