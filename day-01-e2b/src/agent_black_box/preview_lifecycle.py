from __future__ import annotations

from pydantic import BaseModel

from .fixture_policy import fixture_requires_preview
from .recorder import RunMetadata, RunStatus, RunSummary


TERMINAL_RUN_STATES = {"succeeded", "failed", "provider_interrupted", "launch_failed"}


class PreviewView(BaseModel):
    preview_url: str | None = None
    preview_state: str = "unavailable"
    preview_last_error: str | None = None
    preview_expected: bool = False
    preview_failure_reason: str | None = None
    sandbox_retained: bool = False
    preview_attempted: bool = False
    preview_refresh_allowed: bool = False


class RefreshProbeRequest(BaseModel):
    allowed: bool
    preview_port: int | None = None
    baseline: PreviewView


def preview_expected_for_task(
    task: str,
    *,
    metadata: RunMetadata | None = None,
    summary: RunSummary | None = None,
) -> bool:
    lowered = task.lower()
    return (
        (summary.preview_expected if summary is not None else False)
        or (metadata.preview_expected if metadata is not None else False)
        or (metadata is not None and fixture_requires_preview(metadata.fixture_name))
        or any(
            token in lowered
            for token in (
                "preview",
                "dev server",
                "run the app",
                "run locally",
                "localhost",
                "start the app",
                "start a preview server",
            )
        )
    )


def project_preview(
    *,
    status: RunStatus | None,
    summary: RunSummary | None,
    metadata: RunMetadata | None,
    inferred_preview_url: str | None = None,
    inferred_preview_failure_reason: str | None = None,
    inferred_preview_port: int | None = None,
) -> PreviewView:
    preview_expected = preview_expected_for_task(
        metadata.task if metadata is not None else "",
        metadata=metadata,
        summary=summary,
    )
    preview_failure_reason = (
        status.preview_failure_reason
        if status is not None and status.preview_failure_reason is not None
        else summary.preview_failure_reason
        if summary is not None
        else None
    ) or inferred_preview_failure_reason

    run_state = (
        status.state
        if status is not None
        else summary.status
        if summary is not None
        else "unknown"
    )

    if summary is not None and summary.status in TERMINAL_RUN_STATES:
        preview_url = (
            summary.preview_url
            or (status.preview_url if status is not None else None)
            or inferred_preview_url
        )
        preview_state = summary.preview_state
        sandbox_retained = summary.sandbox_retained
        if preview_url is not None and preview_state == "unavailable":
            preview_state = "retained"
        if (
            preview_url is not None
            and not sandbox_retained
            and preview_state in {"live", "retained", "server_not_running"}
        ):
            sandbox_retained = True
        preview_last_error = summary.preview_last_error
    else:
        preview_url = (
            status.preview_url
            if status is not None and status.preview_url is not None
            else summary.preview_url
            if summary is not None
            else None
        ) or inferred_preview_url
        sandbox_retained = (
            status.sandbox_retained
            if status is not None and status.sandbox_retained
            else summary.sandbox_retained
            if summary is not None
            else False
        )
        preview_state = (
            status.preview_state
            if status is not None
            else summary.preview_state
            if summary is not None
            else "unavailable"
        )
        if preview_url is not None and preview_state == "unavailable":
            preview_state = "live" if run_state == "running" else "retained"
        if (
            preview_state == "unavailable"
            and summary is not None
            and summary.preview_state != "unavailable"
        ):
            preview_state = summary.preview_state
        if preview_url is not None and not sandbox_retained:
            sandbox_retained = preview_state in {
                "live",
                "retained",
                "server_not_running",
            }
        preview_last_error = (
            status.preview_last_error
            if status is not None and status.preview_last_error is not None
            else summary.preview_last_error
            if summary is not None
            else None
        )

    preview_attempted = bool(
        preview_url is not None
        or preview_failure_reason is not None
        or preview_state in {"live", "retained", "server_not_running"}
    )
    preview_refresh_allowed = run_state != "running" and (
        sandbox_retained
        or (
            metadata is not None
            and metadata.sandbox_id is not None
            and inferred_preview_port is not None
            and run_state == "succeeded"
            and preview_state != "expired"
        )
    )
    return PreviewView(
        preview_url=preview_url,
        preview_state=preview_state,
        preview_last_error=preview_last_error,
        preview_expected=preview_expected,
        preview_failure_reason=preview_failure_reason,
        sandbox_retained=sandbox_retained,
        preview_attempted=preview_attempted,
        preview_refresh_allowed=preview_refresh_allowed,
    )


def refresh_probe_request(
    *,
    metadata: RunMetadata,
    status: RunStatus,
    summary: RunSummary | None,
    inferred_preview_url: str | None,
    inferred_preview_failure_reason: str | None,
    inferred_preview_port: int | None,
    default_preview_port: int,
) -> RefreshProbeRequest:
    baseline = project_preview(
        status=status,
        summary=summary,
        metadata=metadata,
        inferred_preview_url=inferred_preview_url,
        inferred_preview_failure_reason=inferred_preview_failure_reason,
        inferred_preview_port=inferred_preview_port,
    )
    return RefreshProbeRequest(
        allowed=baseline.preview_refresh_allowed,
        preview_port=inferred_preview_port or default_preview_port,
        baseline=baseline,
    )


def apply_command_preview_result(
    *,
    current_failure_reason: str | None,
    preview_url: str | None,
    preview_failure_reason: str | None,
) -> PreviewView:
    failure_reason = preview_failure_reason or current_failure_reason
    if preview_url is not None:
        failure_reason = None
    return PreviewView(
        preview_url=preview_url,
        preview_state="live" if preview_url is not None else "unavailable",
        preview_last_error=None,
        preview_failure_reason=failure_reason,
        preview_attempted=preview_url is not None or failure_reason is not None,
    )


def finalize_finished_preview(
    *,
    preview_expected: bool,
    preview_url: str | None,
    preview_failure_reason: str | None,
) -> PreviewView:
    failure_reason = preview_failure_reason
    if preview_url is not None:
        failure_reason = None
    elif preview_expected and failure_reason is None:
        failure_reason = (
            "This run was expected to publish a preview, but none was captured."
        )
    return PreviewView(
        preview_url=preview_url,
        preview_state="retained" if preview_url else "unavailable",
        preview_last_error=None,
        preview_expected=preview_expected,
        preview_failure_reason=failure_reason,
        sandbox_retained=True,
        preview_attempted=preview_url is not None or failure_reason is not None,
        preview_refresh_allowed=True,
    )


def finalize_failed_preview(
    *,
    preview_expected: bool,
    preview_failure_reason: str | None,
    preview_last_error: str,
    default_preview_failure_reason: str,
    had_preview: bool = False,
) -> PreviewView:
    failure_reason = preview_failure_reason
    if preview_expected and failure_reason is None:
        failure_reason = default_preview_failure_reason
    preview_related = preview_expected or failure_reason is not None or had_preview
    return PreviewView(
        preview_url=None,
        preview_state="expired" if preview_related else "unavailable",
        preview_last_error=preview_last_error if preview_related else None,
        preview_expected=preview_expected,
        preview_failure_reason=failure_reason,
        sandbox_retained=False,
        preview_attempted=preview_related,
        preview_refresh_allowed=False,
    )


def finalize_refresh_preview(
    *,
    baseline: PreviewView,
    preview_state: str,
    preview_url: str | None,
    preview_last_error: str | None,
    sandbox_retained: bool,
) -> PreviewView:
    failure_reason = None if preview_url is not None else baseline.preview_failure_reason
    return PreviewView(
        preview_url=preview_url,
        preview_state=preview_state,
        preview_last_error=preview_last_error,
        preview_expected=baseline.preview_expected,
        preview_failure_reason=failure_reason,
        sandbox_retained=sandbox_retained,
        preview_attempted=preview_url is not None
        or baseline.preview_failure_reason is not None
        or preview_state in {"live", "retained", "server_not_running"},
        preview_refresh_allowed=sandbox_retained,
    )


def finalize_missing_sandbox_preview(
    *,
    baseline: PreviewView,
    preview_last_error: str,
) -> PreviewView:
    return PreviewView(
        preview_url=baseline.preview_url,
        preview_state="expired",
        preview_last_error=preview_last_error,
        preview_expected=baseline.preview_expected,
        preview_failure_reason=baseline.preview_failure_reason,
        sandbox_retained=False,
        preview_attempted=baseline.preview_attempted,
        preview_refresh_allowed=False,
    )


def finalize_transient_refresh_error(
    *,
    baseline: PreviewView,
    preview_last_error: str,
) -> PreviewView:
    preview_state = (
        "retained"
        if baseline.sandbox_retained and baseline.preview_state == "live"
        else baseline.preview_state
    )
    return PreviewView(
        preview_url=baseline.preview_url,
        preview_state=preview_state,
        preview_last_error=preview_last_error,
        preview_expected=baseline.preview_expected,
        preview_failure_reason=baseline.preview_failure_reason,
        sandbox_retained=baseline.sandbox_retained,
        preview_attempted=baseline.preview_attempted,
        preview_refresh_allowed=baseline.sandbox_retained,
    )
