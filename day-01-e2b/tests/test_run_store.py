from __future__ import annotations

import json

import pytest

import agent_black_box.fixture_policy as fixture_policy_module
from agent_black_box.fixture_policy import FixturePolicy, PreviewPolicy
from agent_black_box.recorder import RunMetadata, RunStatus, RunSummary, prepare_run_directory
from agent_black_box.run_store import RunStore


def _write_summary(run_dir, summary: RunSummary) -> None:  # noqa: ANN001
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _write_preview_tool_call(run_dir, command: str) -> None:  # noqa: ANN001
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "tool_call",
                "run_id": "run-1",
                "lane_id": "lane-1",
                "sequence": 1,
                "timestamp": "2026-04-22T00:00:00Z",
                "tool_name": "run_command",
                "arguments": {"command": command, "timeout_seconds": 10},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_store_lists_runs_and_children(tmp_path) -> None:  # noqa: ANN001
    parent_meta = RunMetadata(
        run_id="parent",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-parent",
    )
    child_meta = RunMetadata(
        run_id="child",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-child",
        parent_run_id="parent",
        source_snapshot_id="snap-1",
        source_checkpoint_sequence=82,
        instruction_override="Restyle it",
    )
    parent_dir = prepare_run_directory(
        tmp_path,
        "parent",
        parent_meta,
        RunStatus(run_id="parent", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    prepare_run_directory(
        tmp_path,
        "child",
        child_meta,
        RunStatus(run_id="child", state="running", current_model_name="glm-5.1:cloud", is_fork=True),
    )
    _write_summary(
        parent_dir,
        RunSummary(
            run_id="parent",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-parent",
        ),
    )
    (parent_dir / "diffs" / "0001-demo.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")

    store = RunStore(tmp_path)
    runs = {run.run_id: run for run in store.list_runs()}

    assert runs["parent"].child_run_ids == ["child"]
    assert runs["child"].parent_run_id == "parent"
    assert runs["parent"].demo_summary is not None
    assert store.resolve_diff("parent", "0001-demo.patch").name == "0001-demo.patch"
    assert store.list_artifacts("parent") == []


def test_run_store_infers_retained_preview_for_legacy_terminal_run(tmp_path) -> None:  # noqa: ANN001
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ).model_dump_json(indent=2)
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

    store = RunStore(tmp_path)
    status = store.load_status("run-1")
    summary = store.load_summary("run-1")
    runs = store.list_runs()

    assert status is not None
    assert status.preview_url == "https://preview.example"
    assert status.preview_state == "retained"
    assert status.sandbox_retained is True
    assert summary is not None
    assert summary.preview_state == "retained"
    assert summary.sandbox_retained is True
    assert runs[0].preview_state == "retained"
    assert runs[0].sandbox_retained is True


def test_run_store_preserves_optional_preview_miss_for_legacy_preview_attempt_without_url(tmp_path) -> None:  # noqa: ANN001
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "state": "succeeded",
                "current_model_name": "kimi-k2.6:cloud",
                "latest_sequence": 146,
                "preview_url": None,
                "checkpoint_id": "snap-1",
                "is_fork": False,
                "updated_at": "2026-04-22T00:53:12.971096+00:00",
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
                "preview_url": None,
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
    (run_dir / "events.jsonl").write_text(
        (
            json.dumps(
                {
                    "type": "tool_call",
                    "run_id": "run-1",
                    "lane_id": "lane-1",
                    "sequence": 88,
                    "timestamp": "2026-04-22T00:51:41Z",
                    "tool_name": "run_command",
                    "arguments": {"command": "pnpm preview --host --port 5173", "timeout_seconds": 10},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "command_started",
                    "run_id": "run-1",
                    "lane_id": "lane-1",
                    "sequence": 89,
                    "timestamp": "2026-04-22T00:51:42Z",
                    "command": "pnpm preview --host --port 5173",
                    "cwd": "/home/user/workspace",
                    "background": False,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "tool_result",
                    "run_id": "run-1",
                    "lane_id": "lane-1",
                    "sequence": 90,
                    "timestamp": "2026-04-22T00:51:51Z",
                    "tool_name": "run_command",
                    "ok": False,
                    "result": {"ok": False, "error": "context deadline exceeded"},
                }
            )
            + "\n"
        ),
        encoding="utf-8",
    )

    store = RunStore(tmp_path)
    status = store.load_status("run-1")
    summary = store.load_summary("run-1")

    assert status is not None
    assert status.preview_state == "unavailable"
    assert status.preview_expected is False
    assert (
        status.preview_failure_reason
        == "Preview command for port 5173 failed before publication: context deadline exceeded"
    )
    assert status.sandbox_retained is False
    assert summary is not None
    assert summary.preview_state == "unavailable"
    assert summary.preview_expected is False
    assert (
        summary.preview_failure_reason
        == "Preview command for port 5173 failed before publication: context deadline exceeded"
    )
    assert summary.sandbox_retained is False


def test_run_store_recovers_preview_url_from_preview_published_event(tmp_path) -> None:  # noqa: ANN001
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "state": "succeeded",
                "current_model_name": "kimi-k2.6:cloud",
                "latest_sequence": 146,
                "preview_url": None,
                "preview_state": "expired",
                "preview_last_error": "Paused sandbox sbx-1 not found",
                "preview_expected": False,
                "preview_failure_reason": None,
                "sandbox_retained": False,
                "checkpoint_id": "snap-1",
                "is_fork": False,
                "updated_at": "2026-04-22T00:53:12.971096+00:00",
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
                "preview_url": None,
                "preview_state": "expired",
                "preview_last_error": "Paused sandbox sbx-1 not found",
                "preview_expected": False,
                "preview_failure_reason": None,
                "sandbox_retained": False,
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
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "preview_published",
                "run_id": "run-1",
                "lane_id": "lane-1",
                "sequence": 82,
                "timestamp": "2026-04-22T00:53:00Z",
                "url": "https://4173-sbx-1.e2b.app",
                "port": 4173,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    store = RunStore(tmp_path)
    status = store.load_status("run-1")
    summary = store.load_summary("run-1")

    assert status is not None
    assert status.preview_url == "https://4173-sbx-1.e2b.app"
    assert status.preview_state == "expired"
    assert status.preview_expected is False
    assert summary is not None
    assert summary.preview_url == "https://4173-sbx-1.e2b.app"
    assert summary.preview_state == "expired"
    assert summary.preview_expected is False


def test_run_store_ignores_malformed_event_lines_during_projection(tmp_path) -> None:  # noqa: ANN001
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
    )
    (run_dir / "events.jsonl").write_text(
        "not json\n"
        + json.dumps(
            {
                "type": "preview_published",
                "run_id": "run-1",
                "lane_id": "lane-1",
                "sequence": 2,
                "timestamp": "2026-04-22T00:00:00Z",
                "url": "https://4173-sbx-1.e2b.app",
                "port": 4173,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    store = RunStore(tmp_path)
    projected = store.load_projected_run("run-1")

    assert projected is not None
    assert projected.status.preview_url == "https://4173-sbx-1.e2b.app"


def test_run_store_treats_unreadable_events_file_as_empty(tmp_path) -> None:  # noqa: ANN001
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
    )
    events_path = run_dir / "events.jsonl"
    events_path.unlink(missing_ok=True)
    events_path.mkdir()

    store = RunStore(tmp_path)

    assert store.load_event_lines("run-1") == []
    assert store.load_projected_run("run-1") is not None


def test_run_store_uses_sample_fixture_preview_port_as_history_default(tmp_path) -> None:  # noqa: ANN001
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    _write_preview_tool_call(run_dir, "pnpm preview --host")

    store = RunStore(tmp_path)

    assert store.infer_preview_port("run-1") == 4173


def test_run_store_uses_custom_fixture_preview_port_as_history_default(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # noqa: ANN001
    monkeypatch.setitem(
        fixture_policy_module._FIXTURE_POLICIES,
        "custom_preview_task",
        FixturePolicy(
            guardrails="Custom fixture guardrails.",
            preview=PreviewPolicy(
                command="pnpm preview --host",
                port=5179,
                cleanup_command="true",
            ),
        ),
    )
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="custom_preview_task",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    _write_preview_tool_call(run_dir, "pnpm preview --host")

    store = RunStore(tmp_path)

    assert store.infer_preview_port("run-1") == 5179


def test_run_store_rejects_diff_path_traversal(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        metadata,
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
    )
    (run_dir / "metadata.secret").write_text("nope", encoding="utf-8")

    store = RunStore(tmp_path)

    try:
        store.resolve_diff("run-1", "../metadata.secret")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected resolve_diff to reject path traversal")


def test_run_store_rejects_diff_dir_symlink_outside_run_root(tmp_path) -> None:  # noqa: ANN001
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
    )
    diff_dir = run_dir / "diffs"
    for child in diff_dir.iterdir():
        child.unlink()
    diff_dir.rmdir()

    outside_dir = tmp_path / "outside-diffs"
    outside_dir.mkdir()
    (outside_dir / "0001-demo.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    diff_dir.symlink_to(outside_dir, target_is_directory=True)

    store = RunStore(tmp_path)

    try:
        store.resolve_diff("run-1", "0001-demo.patch")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected resolve_diff to reject diff-dir symlink traversal")


def test_run_store_rejects_artifact_dir_symlink_outside_run_root(tmp_path) -> None:  # noqa: ANN001
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
    )
    artifact_dir = run_dir / "artifacts"
    artifact_dir.rmdir()
    outside_dir = tmp_path / "outside-artifacts"
    outside_dir.mkdir()
    (outside_dir / "preview-url.txt").write_text("https://preview.example\n", encoding="utf-8")
    artifact_dir.symlink_to(outside_dir, target_is_directory=True)

    store = RunStore(tmp_path)

    assert store.list_artifacts("run-1") == []
    try:
        store.resolve_artifact("run-1", "preview-url.txt")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected resolve_artifact to reject artifact-dir symlink traversal")


def test_run_store_rejects_run_dir_path_traversal(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_root.mkdir()
    outside_dir = tmp_path / "outside-run"
    outside_dir.mkdir()

    store = RunStore(run_root)

    try:
        store.get_run_dir("../outside-run")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected get_run_dir to reject path traversal")


def test_run_store_list_runs_skips_symlinked_dirs_outside_run_root(tmp_path) -> None:  # noqa: ANN001
    run_root = tmp_path / "runs"
    run_root.mkdir()

    valid_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    _write_summary(
        valid_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
    )

    outside_dir = tmp_path / "outside-run"
    outside_dir.mkdir()
    (outside_dir / "metadata.json").write_text(
        RunMetadata(
            run_id="outside-run",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-outside",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_root / "symlink-run").symlink_to(outside_dir, target_is_directory=True)

    store = RunStore(run_root)

    runs = store.list_runs()

    assert [run.run_id for run in runs] == ["run-1"]


def test_run_store_skips_runs_with_metadata_symlink_outside_run_root(tmp_path) -> None:  # noqa: ANN001
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    outside_metadata = tmp_path / "outside-metadata.json"
    outside_metadata.write_text(
        RunMetadata(
            run_id="outside-run",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-outside",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metadata.json").unlink()
    (run_dir / "metadata.json").symlink_to(outside_metadata)

    store = RunStore(tmp_path)

    assert store.list_runs() == []

    try:
        store.load_metadata("run-1")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected load_metadata to reject metadata symlink traversal")


def test_run_store_rejects_checkpoint_dir_symlink_outside_run_root(tmp_path) -> None:  # noqa: ANN001
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    checkpoint_dir = run_dir / "checkpoints"
    for child in checkpoint_dir.iterdir():
        child.unlink()
    checkpoint_dir.rmdir()

    outside_dir = tmp_path / "outside-checkpoints"
    outside_dir.mkdir()
    (outside_dir / "0007.json").write_text('{"snapshot_id":"snap-outside"}\n', encoding="utf-8")
    checkpoint_dir.symlink_to(outside_dir, target_is_directory=True)

    store = RunStore(tmp_path)

    try:
        store.resolve_checkpoint("run-1", 7)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected resolve_checkpoint to reject checkpoint-dir symlink traversal")

    assert store.list_checkpoints("run-1") == []


def test_run_store_ignores_non_numeric_checkpoint_files(tmp_path) -> None:  # noqa: ANN001
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    checkpoint_dir = run_dir / "checkpoints"
    (checkpoint_dir / "0007.json").write_text('{"snapshot_id":"snap-7"}\n', encoding="utf-8")
    (checkpoint_dir / "notes.json").write_text('{"snapshot_id":"not-a-checkpoint"}\n', encoding="utf-8")

    store = RunStore(tmp_path)
    checkpoints = store.list_checkpoints("run-1")
    projected = store.load_projected_run("run-1")

    assert [path.name for path in checkpoints] == ["0007.json"]
    assert projected is not None
    assert projected.checkpoint_count == 1
    assert projected.latest_checkpoint_sequence == 7


def test_run_store_ignores_invalid_checkpoint_payloads(tmp_path) -> None:  # noqa: ANN001
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    checkpoint_dir = run_dir / "checkpoints"
    (checkpoint_dir / "0007.json").write_text('{"snapshot_id":"snap-7"}\n', encoding="utf-8")
    (checkpoint_dir / "0008.json").write_text('{"note":"missing snapshot"}\n', encoding="utf-8")
    (checkpoint_dir / "0009.json").write_text('{"snapshot_id":', encoding="utf-8")

    store = RunStore(tmp_path)
    checkpoints = store.list_checkpoints("run-1")
    projected = store.load_projected_run("run-1")

    assert [path.name for path in checkpoints] == ["0007.json"]
    assert projected is not None
    assert projected.checkpoint_count == 1
    assert projected.latest_checkpoint_sequence == 7


def test_run_store_synthesizes_status_from_summary_when_missing(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            preview_url="https://preview.example",
            checkpoint_id="snap-1",
        ),
    )

    store = RunStore(tmp_path)
    status = store.load_status("run-1")

    assert status is not None
    assert status.state == "succeeded"
    assert status.preview_url == "https://preview.example"
    assert status.checkpoint_id == "snap-1"


def test_run_store_synthesizes_status_from_summary_when_status_json_is_malformed(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (run_dir / "status.json").write_text('{"run_id":', encoding="utf-8")
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            preview_url="https://preview.example",
            checkpoint_id="snap-1",
        ),
    )

    store = RunStore(tmp_path)
    status = store.load_status("run-1")

    assert status is not None
    assert status.state == "succeeded"
    assert status.preview_url == "https://preview.example"
    assert status.checkpoint_id == "snap-1"


def test_run_store_synthesizes_status_from_summary_when_status_json_is_unreadable(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(metadata.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (run_dir / "status.json").mkdir()
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            preview_url="https://preview.example",
            checkpoint_id="snap-1",
        ),
    )

    store = RunStore(tmp_path)
    status = store.load_status("run-1")

    assert status is not None
    assert status.state == "succeeded"
    assert status.preview_url == "https://preview.example"
    assert status.checkpoint_id == "snap-1"


def test_run_store_prefers_richer_summary_preview_truth_when_status_is_stale(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        metadata,
        RunStatus(
            run_id="run-1",
            state="succeeded",
            current_model_name="kimi-k2.6:cloud",
            preview_state="unavailable",
            sandbox_retained=False,
        ),
    )
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            preview_url="https://preview.example",
            preview_state="retained",
            sandbox_retained=True,
        ),
    )

    store = RunStore(tmp_path)
    item = store.list_runs()[0]

    assert item.preview_state == "retained"
    assert item.preview_url == "https://preview.example"
    assert item.sandbox_retained is True
    status = store.load_status("run-1")
    assert status is not None
    assert status.preview_state == "retained"
    assert status.preview_url == "https://preview.example"
    assert status.sandbox_retained is True


def test_run_store_prefers_terminal_summary_when_status_is_stale(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        metadata,
        RunStatus(
            run_id="run-1",
            state="running",
            current_model_name="kimi-k2.6:cloud",
            checkpoint_id=None,
        ),
    )
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="glm-5.1:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            checkpoint_id="snap-1",
        ),
    )

    store = RunStore(tmp_path)
    status = store.load_status("run-1")
    item = store.list_runs()[0]

    assert status is not None
    assert status.state == "succeeded"
    assert status.current_model_name == "glm-5.1:cloud"
    assert status.checkpoint_id == "snap-1"
    assert item.status == "succeeded"
    summary = store.load_summary("run-1")
    assert summary is not None
    assert summary.status == "succeeded"
    assert summary.model_name == "glm-5.1:cloud"
    assert summary.checkpoint_id == "snap-1"


def test_run_store_prefers_terminal_summary_preview_truth_when_status_is_stale(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        metadata,
        RunStatus(
            run_id="run-1",
            state="succeeded",
            current_model_name="kimi-k2.6:cloud",
            preview_url="https://old-preview.example",
            preview_state="live",
            preview_last_error=None,
            sandbox_retained=True,
        ),
    )
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            preview_url=None,
            preview_state="expired",
            preview_last_error="sandbox wasn't found",
            sandbox_retained=False,
        ),
    )

    store = RunStore(tmp_path)
    status = store.load_status("run-1")
    item = store.list_runs()[0]

    assert status is not None
    assert status.preview_url == "https://old-preview.example"
    assert status.preview_state == "expired"
    assert status.preview_last_error == "sandbox wasn't found"
    assert status.sandbox_retained is False
    assert item.preview_url == "https://old-preview.example"
    assert item.preview_state == "expired"
    assert item.preview_last_error == "sandbox wasn't found"
    assert item.sandbox_retained is False
    summary = store.load_summary("run-1")
    assert summary is not None
    assert summary.preview_url == "https://old-preview.example"
    assert summary.preview_state == "expired"
    assert summary.preview_last_error == "sandbox wasn't found"
    assert summary.sandbox_retained is False


def test_run_store_prefers_terminal_summary_state_even_when_status_is_also_terminal(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        metadata,
        RunStatus(
            run_id="run-1",
            state="succeeded",
            current_model_name="kimi-k2.6:cloud",
            checkpoint_id="snap-old",
        ),
    )
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="launch_failed",
            model_name="glm-5.1:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            checkpoint_id=None,
            failure_reason="sandbox bootstrap failed",
        ),
    )

    store = RunStore(tmp_path)
    status = store.load_status("run-1")
    item = store.list_runs()[0]

    assert status is not None
    assert status.state == "launch_failed"
    assert status.current_model_name == "glm-5.1:cloud"
    assert status.checkpoint_id is None
    assert item.status == "launch_failed"


def test_run_store_prefers_terminal_summary_metadata_even_when_status_state_matches(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        metadata,
        RunStatus(
            run_id="run-1",
            state="succeeded",
            current_model_name="kimi-k2.6:cloud",
            checkpoint_id="snap-old",
        ),
    )
    _write_summary(
        run_dir,
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="glm-5.1:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
            checkpoint_id="snap-new",
        ),
    )

    store = RunStore(tmp_path)
    status = store.load_status("run-1")
    item = store.list_runs()[0]

    assert status is not None
    assert status.state == "succeeded"
    assert status.current_model_name == "glm-5.1:cloud"
    assert status.checkpoint_id == "snap-new"
    assert item.status == "succeeded"
    assert item.model_name == "glm-5.1:cloud"
