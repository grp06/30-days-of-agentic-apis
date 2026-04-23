from __future__ import annotations

from agent_black_box.preview_execution import PreviewEvidence
from agent_black_box.run_lifecycle import (
    build_launch_failed_state,
    build_pending_run_state,
    build_running_run_state,
    build_terminal_run_state,
)


def test_build_pending_run_state_preserves_lineage_without_transport_shape() -> None:
    pending = build_pending_run_state(
        run_id="run-1",
        task="Parent task\n\nFork instruction override:\n- Restyle the hero\n",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
        is_fork=True,
        parent_run_id="parent-run",
        source_snapshot_id="snap-82",
        source_checkpoint_sequence=82,
        instruction_override="Restyle the hero",
    )

    assert pending.metadata.parent_run_id == "parent-run"
    assert pending.metadata.source_snapshot_id == "snap-82"
    assert pending.metadata.source_checkpoint_sequence == 82
    assert pending.metadata.instruction_override == "Restyle the hero"
    assert pending.status.state == "running"
    assert pending.status.is_fork is True


def test_build_running_run_state_sets_preview_expectation_from_task() -> None:
    running = build_running_run_state(
        run_id="run-1",
        task="Run the app locally and verify the preview before finishing.",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
        sandbox_id="sbx-1",
    )

    assert running.metadata.preview_expected is True
    assert running.summary.preview_expected is True
    assert running.status.preview_expected is True
    assert running.summary.status == "running"
    assert running.status.state == "running"


def test_build_running_run_state_sets_fixture_preview_expectation() -> None:
    running = build_running_run_state(
        run_id="run-1",
        task="Make a simple landing page.",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
        sandbox_id="sbx-1",
    )

    assert running.metadata.preview_expected is True
    assert running.summary.preview_expected is True
    assert running.status.preview_expected is True


def test_build_terminal_run_state_preserves_checkpoint_on_failure() -> None:
    running = build_running_run_state(
        run_id="run-1",
        task="Do the thing",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
        sandbox_id="sbx-1",
    )
    status = running.status.model_copy(
        update={
            "preview_url": "https://4173-preview.example",
            "preview_state": "live",
            "checkpoint_id": "snap-1",
        }
    )
    summary = running.summary.model_copy(
        update={
            "model_name": "glm-5.1:cloud",
            "preview_url": "https://4173-preview.example",
            "preview_state": "live",
            "checkpoint_id": "snap-1",
        }
    )

    terminal = build_terminal_run_state(
        metadata=running.metadata,
        status=status,
        summary=summary,
        outcome="failed",
        failure_reason="model crashed",
    )

    assert terminal.summary.status == "failed"
    assert terminal.summary.failure_reason == "model crashed"
    assert terminal.summary.checkpoint_id == "snap-1"
    assert terminal.status.state == "failed"
    assert terminal.status.current_model_name == "glm-5.1:cloud"
    assert terminal.status.checkpoint_id == "snap-1"


def test_build_terminal_run_state_finalizes_success_preview() -> None:
    running = build_running_run_state(
        run_id="run-1",
        task="Run the app locally and verify the preview before finishing.",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
        sandbox_id="sbx-1",
    )
    summary = running.summary.model_copy(
        update={"preview_failure_reason": "old failure"}
    )

    terminal = build_terminal_run_state(
        metadata=running.metadata,
        status=running.status,
        summary=summary,
        outcome="succeeded",
        preview_url="https://4173-preview.example",
    )

    assert terminal.summary.status == "succeeded"
    assert terminal.summary.preview_url == "https://4173-preview.example"
    assert terminal.summary.preview_state == "retained"
    assert terminal.summary.preview_failure_reason is None
    assert terminal.status.state == "succeeded"
    assert terminal.status.preview_url == "https://4173-preview.example"
    assert terminal.status.preview_state == "retained"


def test_build_launch_failed_state_uses_event_only_preview_evidence() -> None:
    pending = build_pending_run_state(
        run_id="run-1",
        task="Make a simple landing page.",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
    )

    terminal = build_launch_failed_state(
        run_id="run-1",
        fixture_name="sample_frontend_task",
        default_model_name="kimi-k2.6:cloud",
        failure_reason="sandbox bootstrap failed",
        metadata=pending.metadata,
        status=pending.status,
        summary=None,
        evidence=PreviewEvidence(preview_url="https://4173-preview.example"),
    )

    assert terminal.status.state == "launch_failed"
    assert terminal.status.preview_state == "expired"
    assert terminal.status.checkpoint_id is None
    assert terminal.summary.status == "launch_failed"
    assert terminal.summary.failure_reason == "sandbox bootstrap failed"
    assert terminal.summary.preview_state == "expired"
    assert terminal.summary.command_count == 0
