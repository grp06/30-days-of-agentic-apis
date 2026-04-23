from __future__ import annotations

import pytest

import agent_black_box.fixture_policy as fixture_policy_module
from agent_black_box.events import (
    CommandCompletedEvent,
    Event,
    FileDiffEvent,
    PreviewPublishedEvent,
    ProtocolRepairRequestedEvent,
    RunFailedEvent,
    dump_event,
)
from agent_black_box.fixture_policy import FixturePolicy
from agent_black_box.recorder import RunMetadata, RunStatus, RunSummary
from agent_black_box.run_projection import LoadedRunFacts, project_arena_lane, project_run


def _event_lines(*events: Event) -> list[str]:
    return [dump_event(event) for event in events]


def test_project_run_reconciles_terminal_summary_with_status_and_list_item() -> None:
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
            ),
            status=RunStatus(
                run_id="run-1",
                state="running",
                current_model_name="kimi-k2.6:cloud",
                preview_state="unavailable",
                sandbox_retained=False,
            ),
            summary=RunSummary(
                run_id="run-1",
                status="succeeded",
                model_name="glm-5.1:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                preview_url="https://preview.example",
                preview_state="retained",
                sandbox_retained=True,
                checkpoint_id="snap-1",
            ),
            child_run_ids=["child-1"],
            checkpoint_count=2,
        )
    )

    assert projected.status.state == "succeeded"
    assert projected.status.current_model_name == "glm-5.1:cloud"
    assert projected.status.preview_state == "retained"
    assert projected.status.sandbox_retained is True
    assert projected.summary is not None
    assert projected.summary.preview_url == "https://preview.example"
    assert projected.summary.preview_state == "retained"
    assert projected.list_item.status == "succeeded"
    assert projected.list_item.model_name == "glm-5.1:cloud"
    assert projected.list_item.child_run_ids == ["child-1"]


def test_project_arena_lane_uses_same_projected_run_meaning() -> None:
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
            ),
            status=RunStatus(
                run_id="run-1",
                state="failed",
                current_model_name="kimi-k2.6:cloud",
                preview_state="expired",
                preview_last_error="Paused sandbox sbx-1 not found",
                preview_expected=True,
                preview_failure_reason="Preview command failed before publication",
                sandbox_retained=False,
                preview_attempted=True,
            ),
            summary=RunSummary(
                run_id="run-1",
                status="failed",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                preview_state="expired",
                preview_last_error="Paused sandbox sbx-1 not found",
                preview_expected=True,
                preview_failure_reason="Preview command failed before publication",
                sandbox_retained=False,
                failure_reason="build failed",
            ),
            child_run_ids=["child-1", "child-2"],
            checkpoint_count=3,
            latest_checkpoint_sequence=9,
        )
    )

    lane = project_arena_lane(projected, lane_id="lane-1")

    assert lane.lane_id == "lane-1"
    assert lane.run_id == "run-1"
    assert lane.state == "failed"
    assert lane.preview_state == "expired"
    assert lane.preview_expected is True
    assert lane.preview_failure_reason == "Preview command failed before publication"
    assert lane.checkpoint_count == 3
    assert lane.child_run_ids == ["child-1", "child-2"]
    assert lane.failure_reason == "build failed"
    assert lane.latest_checkpoint_sequence == 9
    assert lane.preview_diagnostic == "Preview command failed before publication"
    assert [step.key for step in lane.lifecycle_steps] == [
        "sandbox",
        "agent",
        "build",
        "preview",
        "evidence",
        "snapshot",
    ]
    assert lane.lifecycle_steps[-1].status == "ok"
    assert lane.phase_key == "agent"
    assert lane.phase_label == "build failed"
    assert lane.phase_status == "error"


def test_running_preview_expected_lane_keeps_preview_pending_until_build_passes() -> None:
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                preview_expected=True,
            ),
            status=RunStatus(
                run_id="run-1",
                state="running",
                current_model_name="kimi-k2.6:cloud",
            ),
        )
    )

    lane = project_arena_lane(projected, lane_id="lane-1")
    steps = {step.key: step for step in lane.lifecycle_steps}

    assert steps["sandbox"].status == "ok"
    assert steps["agent"].status == "active"
    assert steps["preview"].status == "pending"
    assert lane.preview_diagnostic is None
    assert lane.phase_key == "agent"
    assert lane.phase_label == "Model is editing files"
    assert lane.phase_status == "active"


def test_agent_failure_marks_downstream_steps_not_reached() -> None:
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                preview_expected=True,
            ),
            status=RunStatus(
                run_id="run-1",
                state="failed",
                current_model_name="kimi-k2.6:cloud",
                preview_expected=True,
                preview_failure_reason="Preview was expected, but the run failed before it could be verified.",
            ),
            summary=RunSummary(
                run_id="run-1",
                status="failed",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                preview_expected=True,
                preview_failure_reason="Preview was expected, but the run failed before it could be verified.",
                failure_reason="Model provider timed out before producing a finishable workspace diff.",
            ),
        )
    )

    lane = project_arena_lane(projected, lane_id="lane-1")
    steps = {step.key: step for step in lane.lifecycle_steps}

    assert steps["agent"].status == "error"
    assert steps["agent"].detail == "Model provider timed out before producing a finishable workspace diff."
    assert steps["build"].status == "skipped"
    assert steps["build"].detail == "Not reached"
    assert steps["preview"].status == "skipped"
    assert steps["preview"].detail == "Not reached"
    assert steps["evidence"].status == "skipped"
    assert steps["snapshot"].status == "skipped"
    assert lane.phase_key == "agent"
    assert lane.phase_label == "Model provider timed out before producing a finishable workspace diff."


def test_typed_protocol_failure_event_drives_demo_summary() -> None:
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
            ),
            status=RunStatus(
                run_id="run-1",
                state="failed",
                current_model_name="kimi-k2.6:cloud",
            ),
            summary=RunSummary(
                run_id="run-1",
                status="failed",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                failure_reason="agent stopped without a valid action",
            ),
            event_lines=_event_lines(
                RunFailedEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=5,
                    error="agent stopped without a valid action",
                    failure_kind="completed_without_action",
                )
            ),
        )
    )

    assert (
        projected.demo_summary
        == "Model protocol incomplete: agent stopped without a valid action"
    )


def test_legacy_protocol_failure_string_still_drives_demo_summary() -> None:
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
            ),
            status=RunStatus(
                run_id="run-1",
                state="failed",
                current_model_name="kimi-k2.6:cloud",
            ),
            summary=RunSummary(
                run_id="run-1",
                status="failed",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                failure_reason=(
                    "Model protocol incomplete: model ended the turn without "
                    "calling a tool or finish_run."
                ),
            ),
        )
    )

    assert projected.demo_summary == (
        "Model protocol incomplete: model ended the turn without calling a tool "
        "or finish_run."
    )


def test_prior_protocol_repair_does_not_override_terminal_build_failure() -> None:
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
            ),
            status=RunStatus(
                run_id="run-1",
                state="failed",
                current_model_name="kimi-k2.6:cloud",
            ),
            summary=RunSummary(
                run_id="run-1",
                status="failed",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                failure_reason="Managed build command failed: sh: 1: vite: not found",
            ),
            event_lines=_event_lines(
                ProtocolRepairRequestedEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=2,
                    turn_number=1,
                    repair_attempt=1,
                    reason="Model protocol incomplete",
                    failure_kind="hit_generation_limit",
                    hit_generation_limit=True,
                    message="Call exactly one tool now.",
                ),
                RunFailedEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=7,
                    error="Managed build command failed: sh: 1: vite: not found",
                ),
            ),
        )
    )

    assert (
        projected.demo_summary
        == "Build failed: Managed build command failed: sh: 1: vite: not found"
    )


def test_build_failure_marks_preview_not_attempted() -> None:
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="qwen3.5:397b",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                preview_expected=True,
            ),
            status=RunStatus(
                run_id="run-1",
                state="failed",
                current_model_name="qwen3.5:397b",
                preview_expected=True,
                preview_failure_reason="Preview was not attempted because the managed build failed.",
            ),
            summary=RunSummary(
                run_id="run-1",
                status="failed",
                model_name="qwen3.5:397b",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                preview_expected=True,
                preview_failure_reason="Preview was not attempted because the managed build failed.",
                failure_reason="Managed build command failed: sh: 1: vite: not found",
            ),
            event_lines=_event_lines(
                FileDiffEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=1,
                    patch_path="diffs/0001.patch",
                    patch_summary="changed files",
                ),
                CommandCompletedEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=2,
                    command="pnpm build",
                    cwd="/workspace",
                    exit_code=1,
                    stdout="",
                    stderr="vite: not found",
                    background=False,
                ),
            ),
        )
    )

    lane = project_arena_lane(projected, lane_id="lane-1")
    steps = {step.key: step for step in lane.lifecycle_steps}

    assert steps["agent"].status == "ok"
    assert steps["build"].status == "error"
    assert steps["build"].detail == "Managed build command failed: sh: 1: vite: not found"
    assert steps["preview"].status == "skipped"
    assert steps["preview"].detail == "Not attempted because build failed"
    assert steps["evidence"].status == "skipped"
    assert steps["snapshot"].status == "skipped"
    assert lane.phase_key == "build"
    assert lane.phase_label == "Managed build command failed: sh: 1: vite: not found"


def test_preview_becomes_active_only_after_running_build_passes() -> None:
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                preview_expected=True,
            ),
            status=RunStatus(
                run_id="run-1",
                state="running",
                current_model_name="kimi-k2.6:cloud",
            ),
            event_lines=_event_lines(
                FileDiffEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=1,
                    patch_path="diffs/0001.patch",
                    patch_summary="changed files",
                ),
                CommandCompletedEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=2,
                    command="pnpm build",
                    cwd="/workspace",
                    exit_code=0,
                    stdout="ok",
                    stderr="",
                    background=False,
                ),
            ),
        )
    )

    lane = project_arena_lane(projected, lane_id="lane-1")
    steps = {step.key: step for step in lane.lifecycle_steps}

    assert steps["agent"].status == "ok"
    assert steps["build"].status == "ok"
    assert steps["preview"].status == "active"
    assert lane.phase_key == "preview"
    assert lane.phase_label == "Publishing preview"
    assert lane.phase_status == "active"


def test_configured_non_default_build_command_drives_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        fixture_policy_module._FIXTURE_POLICIES,
        "custom_build_task",
        FixturePolicy(
            guardrails="Custom fixture guardrails.",
            build_command="npm run compile",
        ),
    )
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="kimi-k2.6:cloud",
                fixture_name="custom_build_task",
                sandbox_id="sbx-1",
                preview_expected=True,
            ),
            status=RunStatus(
                run_id="run-1",
                state="running",
                current_model_name="kimi-k2.6:cloud",
            ),
            event_lines=_event_lines(
                FileDiffEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=1,
                    patch_path="diffs/0001.patch",
                    patch_summary="changed files",
                ),
                CommandCompletedEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=2,
                    command="npm run compile",
                    cwd="/workspace",
                    exit_code=0,
                    stdout="ok",
                    stderr="",
                    background=False,
                ),
            ),
        )
    )

    lane = project_arena_lane(projected, lane_id="lane-1")
    steps = {step.key: step for step in lane.lifecycle_steps}

    assert steps["build"].status == "ok"
    assert lane.phase_key == "preview"


def test_published_preview_moves_running_lane_to_snapshot_phase() -> None:
    projected = project_run(
        LoadedRunFacts(
            metadata=RunMetadata(
                run_id="run-1",
                task="task",
                model_name="kimi-k2.6:cloud",
                fixture_name="sample_frontend_task",
                sandbox_id="sbx-1",
                preview_expected=True,
            ),
            status=RunStatus(
                run_id="run-1",
                state="running",
                current_model_name="kimi-k2.6:cloud",
                preview_url="https://preview.example",
                preview_state="live",
            ),
            event_lines=_event_lines(
                FileDiffEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=1,
                    patch_path="diffs/0001.patch",
                    patch_summary="changed files",
                ),
                CommandCompletedEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=2,
                    command="pnpm build",
                    cwd="/workspace",
                    exit_code=0,
                    stdout="ok",
                    stderr="",
                    background=False,
                ),
                PreviewPublishedEvent(
                    run_id="run-1",
                    lane_id="lane-1",
                    sequence=3,
                    url="https://preview.example",
                    port=4173,
                ),
            ),
        )
    )

    lane = project_arena_lane(projected, lane_id="lane-1")
    steps = {step.key: step for step in lane.lifecycle_steps}

    assert steps["preview"].status == "ok"
    assert steps["evidence"].status == "ok"
    assert lane.phase_key == "snapshot"
    assert lane.phase_label == "Saving snapshot"
    assert lane.phase_status == "active"
