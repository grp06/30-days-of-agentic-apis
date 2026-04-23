from __future__ import annotations

from agent_black_box.preview_lifecycle import (
    finalize_failed_preview,
    finalize_finished_preview,
    finalize_refresh_preview,
    finalize_missing_sandbox_preview,
    preview_expected_for_task,
    project_preview,
    refresh_probe_request,
)
from agent_black_box.recorder import RunMetadata, RunStatus, RunSummary


def test_preview_lifecycle_projects_terminal_preview_from_summary_and_inference() -> (
    None
):
    metadata = RunMetadata(
        run_id="run-1",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    status = RunStatus(
        run_id="run-1",
        state="succeeded",
        current_model_name="kimi-k2.6:cloud",
        preview_state="unavailable",
        sandbox_retained=False,
    )
    summary = RunSummary(
        run_id="run-1",
        status="succeeded",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
        preview_url=None,
        preview_state="unavailable",
        sandbox_retained=False,
    )

    preview = project_preview(
        status=status,
        summary=summary,
        metadata=metadata,
        inferred_preview_url="https://4173-sbx-1.e2b.app",
        inferred_preview_failure_reason=None,
        inferred_preview_port=4173,
    )

    assert preview.preview_url == "https://4173-sbx-1.e2b.app"
    assert preview.preview_state == "retained"
    assert preview.preview_attempted is True
    assert preview.preview_refresh_allowed is True


def test_preview_lifecycle_keeps_optional_preview_attempts_optional() -> None:
    metadata = RunMetadata(
        run_id="run-1",
        task="make a landing page",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    summary = RunSummary(
        run_id="run-1",
        status="succeeded",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )

    preview = project_preview(
        status=None,
        summary=summary,
        metadata=metadata,
        inferred_preview_url=None,
        inferred_preview_failure_reason=(
            "Preview command for port 5173 failed before publication: context deadline exceeded"
        ),
        inferred_preview_port=5173,
    )

    assert preview.preview_expected is False
    assert preview.preview_attempted is True
    assert preview.preview_failure_reason is not None


def test_preview_lifecycle_refresh_request_requires_retained_or_refreshable_sandbox() -> (
    None
):
    metadata = RunMetadata(
        run_id="run-1",
        task="run the app and make preview live",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    status = RunStatus(
        run_id="run-1",
        state="succeeded",
        current_model_name="kimi-k2.6:cloud",
        preview_state="unavailable",
        sandbox_retained=False,
    )
    summary = RunSummary(
        run_id="run-1",
        status="succeeded",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
        preview_state="unavailable",
        sandbox_retained=False,
    )

    request = refresh_probe_request(
        metadata=metadata,
        status=status,
        summary=summary,
        inferred_preview_url=None,
        inferred_preview_failure_reason=None,
        inferred_preview_port=5173,
        default_preview_port=4173,
    )

    assert request.allowed is True
    assert request.preview_port == 5173


def test_preview_lifecycle_missing_sandbox_preserves_last_known_url_and_failure_reason() -> (
    None
):
    baseline = finalize_finished_preview(
        preview_expected=True,
        preview_url="https://4173-sbx-1.e2b.app",
        preview_failure_reason=None,
    ).model_copy(
        update={
            "preview_failure_reason": "Preview command for port 4173 failed before publication",
        }
    )

    preview = finalize_missing_sandbox_preview(
        baseline=baseline,
        preview_last_error="Paused sandbox sbx-1 not found",
    )

    assert preview.preview_state == "expired"
    assert preview.preview_url == "https://4173-sbx-1.e2b.app"
    assert (
        preview.preview_failure_reason
        == "Preview command for port 4173 failed before publication"
    )
    assert preview.preview_refresh_allowed is False


def test_preview_lifecycle_successful_refresh_clears_old_failure_reason() -> None:
    baseline = finalize_finished_preview(
        preview_expected=True,
        preview_url=None,
        preview_failure_reason="Managed preview command for port 4173 failed",
    )

    preview = finalize_refresh_preview(
        baseline=baseline,
        preview_state="live",
        preview_url="https://4173-sbx-1.e2b.app",
        preview_last_error=None,
        sandbox_retained=True,
    )

    assert preview.preview_state == "live"
    assert preview.preview_url == "https://4173-sbx-1.e2b.app"
    assert preview.preview_failure_reason is None


def test_preview_lifecycle_does_not_reoffer_refresh_for_known_expired_sandbox() -> None:
    metadata = RunMetadata(
        run_id="run-1",
        task="run the app locally and make sure the preview works",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
    )
    status = RunStatus(
        run_id="run-1",
        state="succeeded",
        current_model_name="kimi-k2.6:cloud",
        preview_state="expired",
        preview_last_error="Paused sandbox sbx-1 not found",
        sandbox_retained=False,
    )
    summary = RunSummary(
        run_id="run-1",
        status="succeeded",
        model_name="kimi-k2.6:cloud",
        fixture_name="fixture",
        sandbox_id="sbx-1",
        preview_state="expired",
        preview_last_error="Paused sandbox sbx-1 not found",
        sandbox_retained=False,
    )

    preview = project_preview(
        status=status,
        summary=summary,
        metadata=metadata,
        inferred_preview_url=None,
        inferred_preview_failure_reason=None,
        inferred_preview_port=4173,
    )

    assert preview.preview_state == "expired"
    assert preview.preview_refresh_allowed is False


def test_preview_lifecycle_failed_run_without_preview_stays_unavailable() -> None:
    preview = finalize_failed_preview(
        preview_expected=False,
        preview_failure_reason=None,
        preview_last_error="Sandbox was not retained after run failure.",
        default_preview_failure_reason="unused",
    )

    assert preview.preview_state == "unavailable"
    assert preview.preview_last_error is None
    assert preview.preview_attempted is False


def test_preview_lifecycle_failed_run_after_preview_stays_expired() -> None:
    preview = finalize_failed_preview(
        preview_expected=False,
        preview_failure_reason=None,
        preview_last_error="Sandbox was not retained after run failure.",
        default_preview_failure_reason="unused",
        had_preview=True,
    )

    assert preview.preview_state == "expired"
    assert preview.preview_last_error == "Sandbox was not retained after run failure."
    assert preview.preview_attempted is True


def test_preview_expected_for_task_matches_existing_prompt_contract() -> None:
    assert (
        preview_expected_for_task("run the app locally and make sure the preview works")
        is True
    )
    assert preview_expected_for_task("make a simple landing page") is False


def test_preview_expected_for_sample_frontend_fixture_is_not_tied_to_visible_brief() -> (
    None
):
    metadata = RunMetadata(
        run_id="run-1",
        task="make a simple landing page",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
    )

    assert preview_expected_for_task(metadata.task, metadata=metadata) is True
