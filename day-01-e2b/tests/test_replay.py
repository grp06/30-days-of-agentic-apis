from __future__ import annotations

import json
from pathlib import Path

from agent_black_box.recorder import (
    RunMetadata,
    RunStatus,
    RunSummary,
    prepare_run_directory,
)
from agent_black_box.replay import load_run_projection


def test_replay_projection_loads_real_run_fixture() -> None:
    run_dir = Path(
        "/Users/georgepickett/devrel-jobs/30-days-of-agentic-apis/day-01-e2b/runs/20260421T212322Z-444b18b7"
    )
    projection = load_run_projection(run_dir)

    assert projection.metadata.run_id == "20260421T212322Z-444b18b7"
    assert projection.status.state == "succeeded"
    assert any(card.kind == "command" for card in projection.timeline)
    assert any(card.kind == "diff" for card in projection.timeline)
    assert any(card.kind == "checkpoint" for card in projection.timeline)
    checkpoint_card = next(
        card for card in projection.timeline if card.kind == "checkpoint"
    )
    assert checkpoint_card.detail is not None
    assert checkpoint_card.detail["checkpoint_sequence"] in {70, 82}


def test_replay_projection_tolerates_missing_lineage_and_status(tmp_path) -> None:  # noqa: ANN001
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "task": "task",
                "model_name": "kimi-k2.6:cloud",
                "fixture_name": "fixture",
                "started_at": "2026-04-21T00:00:00+00:00",
                "sandbox_id": "sbx-1",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "succeeded",
                "model_name": "kimi-k2.6:cloud",
                "fixture_name": "fixture",
                "sandbox_id": "sbx-1",
                "command_count": 0,
                "diff_count": 0,
                "tool_call_count": 0,
                "completed_at": "2026-04-21T00:05:00+00:00",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    projection = load_run_projection(run_dir)

    assert projection.status.state == "succeeded"
    assert projection.metadata.parent_run_id is None
    assert projection.timeline == []


def test_replay_projection_infers_retained_preview_for_legacy_terminal_run(
    tmp_path,
) -> None:  # noqa: ANN001
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "task": "task",
                "model_name": "kimi-k2.6:cloud",
                "fixture_name": "fixture",
                "started_at": "2026-04-22T00:00:00+00:00",
                "sandbox_id": "sbx-1",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "state": "succeeded",
                "current_model_name": "kimi-k2.6:cloud",
                "latest_sequence": 80,
                "preview_url": "https://preview.example",
                "checkpoint_id": "snap-1",
                "is_fork": False,
                "updated_at": "2026-04-22T00:35:04.200912+00:00",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "succeeded",
                "model_name": "kimi-k2.6:cloud",
                "fixture_name": "fixture",
                "sandbox_id": "sbx-1",
                "preview_url": "https://preview.example",
                "checkpoint_id": "snap-1",
                "failure_reason": None,
                "command_count": 5,
                "diff_count": 1,
                "tool_call_count": 10,
                "completed_at": "2026-04-22T00:35:04.200550+00:00",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    projection = load_run_projection(run_dir)

    assert projection.status.preview_url == "https://preview.example"
    assert projection.status.preview_state == "retained"
    assert projection.status.sandbox_retained is True
    assert projection.summary is not None
    assert projection.summary.preview_state == "retained"
    assert projection.summary.sandbox_retained is True


def test_replay_projection_uses_reconciled_summary_when_status_is_stale(
    tmp_path,
) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1",
            state="running",
            current_model_name="kimi-k2.6:cloud",
            preview_state="unavailable",
            sandbox_retained=False,
        ),
    )
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="glm-5.1:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            preview_url="https://preview.example",
            preview_state="retained",
            sandbox_retained=True,
            checkpoint_id="snap-1",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    projection = load_run_projection(run_dir)

    assert projection.status.state == "succeeded"
    assert projection.summary is not None
    assert projection.summary.status == "succeeded"
    assert projection.summary.model_name == "glm-5.1:cloud"
    assert projection.summary.preview_url == "https://preview.example"
    assert projection.summary.preview_state == "retained"
    assert projection.summary.sandbox_retained is True
    assert projection.summary.checkpoint_id == "snap-1"


def test_replay_projection_recovers_checkpoint_from_file_without_event(
    tmp_path,
) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"
        ),
    )
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            checkpoint_id="snap-1",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "checkpoints" / "0007.json").write_text(
        json.dumps({"snapshot_id": "snap-1", "note": "checkpoint"}, indent=2) + "\n",
        encoding="utf-8",
    )

    projection = load_run_projection(run_dir)

    assert len(projection.checkpoints) == 1
    assert projection.checkpoints[0].snapshot_id == "snap-1"
    checkpoint_card = next(
        card for card in projection.timeline if card.kind == "checkpoint"
    )
    assert checkpoint_card.detail is not None
    assert checkpoint_card.detail["checkpoint_sequence"] == 7
    assert checkpoint_card.detail["recovered_from_file"] is True


def test_replay_projection_labels_checkpoint_card_with_checkpoint_sequence(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"
        ),
    )
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            checkpoint_id="snap-1",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "0016.json").write_text(
        json.dumps({"snapshot_id": "snap-1", "note": "first workspace diff"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "checkpoint_created",
                "run_id": "run-1",
                "lane_id": "lane-1",
                "sequence": 17,
                "timestamp": "2026-04-22T00:00:00+00:00",
                "snapshot_id": "snap-1",
                "note": "first workspace diff",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    projection = load_run_projection(run_dir)

    checkpoint_card = next(
        card for card in projection.timeline if card.kind == "checkpoint"
    )
    assert checkpoint_card.sequence == 16
    assert checkpoint_card.detail is not None
    assert checkpoint_card.detail["checkpoint_sequence"] == 16
    assert checkpoint_card.detail["event_sequence"] == 17


def test_replay_projection_ignores_non_numeric_checkpoint_files(
    tmp_path,
) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"
        ),
    )
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            checkpoint_id="snap-1",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "checkpoints" / "0007.json").write_text(
        json.dumps({"snapshot_id": "snap-1", "note": "checkpoint"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "checkpoints" / "notes.json").write_text(
        json.dumps({"snapshot_id": "not-a-checkpoint"}, indent=2) + "\n",
        encoding="utf-8",
    )

    projection = load_run_projection(run_dir)

    assert len(projection.checkpoints) == 1
    assert projection.checkpoints[0].sequence == 7
    assert projection.checkpoints[0].snapshot_id == "snap-1"


def test_replay_projection_ignores_malformed_event_lines(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"
        ),
    )
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        "not json\n"
        + json.dumps(
            {
                "type": "command_completed",
                "run_id": "run-1",
                "lane_id": "lane-1",
                "sequence": 2,
                "timestamp": "2026-04-22T00:00:00+00:00",
                "command": "echo ok",
                "exit_code": 0,
                "stdout": "ok\n",
                "stderr": "",
                "background": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    projection = load_run_projection(run_dir)

    assert [card.kind for card in projection.timeline] == ["command"]


def test_replay_projection_merges_run_command_tool_call_into_command_card(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"
        ),
    )
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "tool_call",
                "run_id": "run-1",
                "lane_id": "lane-1",
                "sequence": 1,
                "timestamp": "2026-04-22T00:00:00+00:00",
                "tool_name": "run_command",
                "arguments": {"command": "pnpm build", "timeout_seconds": 120},
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "command_completed",
                "run_id": "run-1",
                "lane_id": "lane-1",
                "sequence": 2,
                "timestamp": "2026-04-22T00:00:01+00:00",
                "command": "pnpm build",
                "exit_code": 0,
                "stdout": "built\n",
                "stderr": "",
                "background": False,
                "pid": 123,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    projection = load_run_projection(run_dir)

    assert [card.kind for card in projection.timeline] == ["command"]
    command_card = projection.timeline[0]
    assert command_card.title == "pnpm build"
    assert command_card.subtitle == "exit 0 · foreground"
    assert command_card.detail is not None
    assert command_card.detail["command"] == "pnpm build"


def test_replay_projection_redacts_large_tool_arguments(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"
        ),
    )
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "tool_call",
                "run_id": "run-1",
                "lane_id": "lane-1",
                "sequence": 1,
                "timestamp": "2026-04-22T00:00:00+00:00",
                "tool_name": "write_file",
                "arguments": {"path": "index.html", "content": "<h1>Hello</h1>"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    projection = load_run_projection(run_dir)

    assert [card.kind for card in projection.timeline] == ["tool_call"]
    tool_card = projection.timeline[0]
    assert tool_card.title == "Write file"
    assert tool_card.subtitle == "index.html"
    assert tool_card.detail is not None
    assert tool_card.detail["arguments"] == {
        "path": "index.html",
        "content": "<14 chars>",
    }


def test_replay_projection_ignores_events_symlink_outside_run_root(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"
        ),
    )
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    events_path = run_dir / "events.jsonl"

    outside_events = tmp_path / "outside-events.jsonl"
    outside_events.write_text(
        json.dumps(
            {
                "type": "command_completed",
                "sequence": 1,
                "timestamp": "2026-04-21T00:00:00+00:00",
                "command": "echo leaked",
                "exit_code": 0,
                "stdout": "leaked\n",
                "stderr": "",
                "background": False,
                "pid": 123,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events_path.symlink_to(outside_events)

    projection = load_run_projection(run_dir)

    assert projection.timeline == []
