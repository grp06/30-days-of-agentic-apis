from __future__ import annotations

import json
from pathlib import Path

from agent_black_box.events import RunStartedEvent
from agent_black_box.recorder import Recorder, RunMetadata, RunStatus, RunSummary


def test_recorder_writes_expected_artifact_layout(tmp_path: Path) -> None:
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    recorder = Recorder(tmp_path, "run-1", metadata)
    recorder.append(
        RunStartedEvent(
            run_id="run-1",
            lane_id="lane-1",
            sequence=1,
            task="task",
            fixture_name="fixture",
            model="kimi-k2.6:cloud",
        )
    )
    recorder.initialize_status(
        RunStatus(run_id="run-1", state="running", current_model_name="kimi-k2.6:cloud")
    )
    diff_path = recorder.write_diff(2, "workspace", "diff --git a/x b/x\n")
    checkpoint_path = recorder.write_checkpoint(3, "snap-1", {"note": "checkpoint"})
    recorder.write_artifact_text("preview-url.txt", "https://example.com\n")
    original_completed_at = "2026-04-21T00:00:00+00:00"
    recorder.finalize(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            preview_url="https://example.com",
            checkpoint_id="snap-1",
            completed_at=original_completed_at,
        )
    )

    assert (tmp_path / "run-1" / "events.jsonl").exists()
    assert diff_path.exists()
    assert checkpoint_path.exists()
    summary = json.loads((tmp_path / "run-1" / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "run-1" / "status.json").read_text(encoding="utf-8"))
    assert summary["preview_url"] == "https://example.com"
    assert summary["checkpoint_id"] == "snap-1"
    assert status["state"] == "succeeded"
    assert status["checkpoint_id"] == "snap-1"
    assert summary["completed_at"] != original_completed_at
    assert list((tmp_path / "run-1").glob(".*.tmp")) == []


def test_recorder_finalize_removes_stale_preview_artifact_when_final_preview_is_missing(tmp_path: Path) -> None:
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    recorder = Recorder(tmp_path, "run-1", metadata)
    recorder.initialize_status(
        RunStatus(run_id="run-1", state="running", current_model_name="kimi-k2.6:cloud")
    )
    recorder.write_artifact_text("preview-url.txt", "https://example.com\n")

    recorder.finalize(
        RunSummary(
            run_id="run-1",
            status="failed",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            preview_url=None,
            preview_state="expired",
            preview_last_error="Sandbox was not retained after run failure.",
            sandbox_retained=False,
        )
    )

    assert not (tmp_path / "run-1" / "artifacts" / "preview-url.txt").exists()
    summary = json.loads((tmp_path / "run-1" / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "run-1" / "status.json").read_text(encoding="utf-8"))
    assert summary["preview_url"] is None
    assert status["preview_url"] is None


def test_recorder_rejects_artifact_path_traversal(tmp_path: Path) -> None:
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    recorder = Recorder(tmp_path, "run-1", metadata)
    original_metadata = (tmp_path / "run-1" / "metadata.json").read_text(
        encoding="utf-8"
    )

    try:
        recorder.write_artifact_text("../metadata.json", "corrupt\n")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected artifact traversal to be rejected")

    assert (
        tmp_path / "run-1" / "metadata.json"
    ).read_text(encoding="utf-8") == original_metadata


def test_recorder_rejects_artifact_symlink_parent_outside_run(tmp_path: Path) -> None:
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    recorder = Recorder(tmp_path, "run-1", metadata)
    outside_dir = tmp_path / "outside-artifacts"
    outside_dir.mkdir()
    (tmp_path / "run-1" / "artifacts" / "linked").symlink_to(
        outside_dir,
        target_is_directory=True,
    )

    try:
        recorder.write_artifact_text("linked/leak.txt", "nope\n")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected artifact symlink traversal to be rejected")

    assert not (outside_dir / "leak.txt").exists()


def test_recorder_rejects_diff_slug_path_traversal(tmp_path: Path) -> None:
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    recorder = Recorder(tmp_path, "run-1", metadata)

    try:
        recorder.write_diff(1, "workspace/../../metadata", "diff --git a/x b/x\n")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected diff slug traversal to be rejected")

    assert not (tmp_path / "run-1" / "metadata.patch").exists()


def test_recorder_persists_explicit_terminal_status_without_rederiving_policy(tmp_path: Path) -> None:
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    recorder = Recorder(tmp_path, "run-1", metadata)
    recorder.initialize_status(
        RunStatus(
            run_id="run-1",
            state="running",
            current_model_name="kimi-k2.6:cloud",
            latest_sequence=9,
        )
    )

    recorder.persist_terminal_state(
        status=RunStatus(
            run_id="run-1",
            state="failed",
            current_model_name="glm-5.1:cloud",
            latest_sequence=9,
            preview_state="expired",
            preview_last_error="Sandbox was not retained after run failure.",
            checkpoint_id="snap-1",
        ),
        summary=RunSummary(
            run_id="run-1",
            status="failed",
            model_name="glm-5.1:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            preview_state="expired",
            preview_last_error="Sandbox was not retained after run failure.",
            checkpoint_id="snap-1",
        ),
    )

    summary = json.loads((tmp_path / "run-1" / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "run-1" / "status.json").read_text(encoding="utf-8"))

    assert summary["status"] == "failed"
    assert summary["checkpoint_id"] == "snap-1"
    assert status["state"] == "failed"
    assert status["current_model_name"] == "glm-5.1:cloud"
    assert status["latest_sequence"] == 9
    assert status["checkpoint_id"] == "snap-1"


def test_recorder_overwrites_existing_metadata_when_allow_existing(tmp_path: Path) -> None:
    initial_metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        started_at="2026-04-21T00:00:00+00:00",
        sandbox_id=None,
    )
    Recorder(tmp_path, "run-1", initial_metadata)

    updated_metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        started_at="2026-04-21T00:05:00+00:00",
        sandbox_id="sbx-1",
    )
    Recorder(tmp_path, "run-1", updated_metadata, allow_existing=True)

    metadata = json.loads((tmp_path / "run-1" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["sandbox_id"] == "sbx-1"
    assert metadata["started_at"] == "2026-04-21T00:00:00+00:00"


def test_recorder_open_existing_rejects_status_symlink_outside_run_root(tmp_path: Path) -> None:
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    recorder = Recorder(tmp_path, "run-1", metadata)
    recorder.initialize_status(
        RunStatus(run_id="run-1", state="running", current_model_name="kimi-k2.6:cloud")
    )

    outside_status = tmp_path / "outside-status.json"
    outside_status.write_text(
        RunStatus(run_id="outside-run", state="succeeded", current_model_name="kimi-k2.6:cloud").model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    status_path = tmp_path / "run-1" / "status.json"
    status_path.unlink()
    status_path.symlink_to(outside_status)

    try:
        Recorder.open_existing(tmp_path, "run-1")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected open_existing to reject status symlink traversal")
