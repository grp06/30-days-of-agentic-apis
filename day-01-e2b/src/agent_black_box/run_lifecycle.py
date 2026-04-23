from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .events import utc_now
from .preview_execution import (
    PreviewEvidence,
    finalize_execution_failure,
    finalize_execution_success,
)
from .preview_lifecycle import preview_expected_for_task
from .recorder import RunMetadata, RunStatus, RunSummary


TerminalOutcome = Literal["succeeded", "failed", "provider_interrupted"]


class PendingRunState(BaseModel):
    metadata: RunMetadata
    status: RunStatus


class RunningRunState(BaseModel):
    metadata: RunMetadata
    status: RunStatus
    summary: RunSummary


class TerminalRunState(BaseModel):
    status: RunStatus
    summary: RunSummary


def build_pending_run_state(
    *,
    run_id: str,
    task: str,
    model_name: str,
    fixture_name: str,
    is_fork: bool = False,
    parent_run_id: str | None = None,
    source_snapshot_id: str | None = None,
    source_checkpoint_sequence: int | None = None,
    instruction_override: str | None = None,
) -> PendingRunState:
    preview_expected = preview_expected_for_task(
        task,
        metadata=RunMetadata(
            run_id=run_id,
            task=task,
            model_name=model_name,
            fixture_name=fixture_name,
        ),
    )
    metadata = RunMetadata(
        run_id=run_id,
        task=task,
        model_name=model_name,
        fixture_name=fixture_name,
        preview_expected=preview_expected,
        parent_run_id=parent_run_id,
        source_snapshot_id=source_snapshot_id,
        source_checkpoint_sequence=source_checkpoint_sequence,
        instruction_override=instruction_override,
    )
    status = RunStatus(
        run_id=run_id,
        state="running",
        current_model_name=model_name,
        preview_expected=preview_expected,
        is_fork=is_fork,
    )
    return PendingRunState(metadata=metadata, status=status)


def build_running_run_state(
    *,
    run_id: str,
    task: str,
    model_name: str,
    fixture_name: str,
    sandbox_id: str,
    is_fork: bool = False,
    parent_run_id: str | None = None,
    source_snapshot_id: str | None = None,
    source_checkpoint_sequence: int | None = None,
    instruction_override: str | None = None,
) -> RunningRunState:
    preview_expected = preview_expected_for_task(
        task,
        metadata=RunMetadata(
            run_id=run_id,
            task=task,
            model_name=model_name,
            fixture_name=fixture_name,
        ),
    )
    metadata = RunMetadata(
        run_id=run_id,
        task=task,
        model_name=model_name,
        fixture_name=fixture_name,
        preview_expected=preview_expected,
        sandbox_id=sandbox_id,
        parent_run_id=parent_run_id,
        source_snapshot_id=source_snapshot_id,
        source_checkpoint_sequence=source_checkpoint_sequence,
        instruction_override=instruction_override,
    )
    status = RunStatus(
        run_id=run_id,
        state="running",
        current_model_name=model_name,
        preview_expected=preview_expected,
        is_fork=is_fork,
    )
    summary = RunSummary(
        run_id=run_id,
        status="running",
        model_name=model_name,
        fixture_name=fixture_name,
        sandbox_id=sandbox_id,
        preview_expected=preview_expected,
    )
    return RunningRunState(metadata=metadata, status=status, summary=summary)


def build_launch_failed_state(
    *,
    run_id: str,
    fixture_name: str,
    default_model_name: str,
    failure_reason: str,
    metadata: RunMetadata | None,
    status: RunStatus | None,
    summary: RunSummary | None,
    evidence: PreviewEvidence | None = None,
) -> TerminalRunState:
    failed_preview = finalize_execution_failure(
        metadata=metadata,
        status=status,
        summary=summary,
        evidence=evidence,
        preview_last_error="Sandbox was not retained after launch failure.",
        default_preview_failure_reason=(
            "Preview was expected, but launch failed before it could be verified."
        ),
    )
    next_status = (
        status or _default_launch_failed_status(run_id, default_model_name, metadata)
    ).model_copy(
        update={
            "state": "launch_failed",
            "latest_sequence": 0,
            "preview_url": failed_preview.preview_url,
            "preview_state": failed_preview.preview_state,
            "preview_last_error": failed_preview.preview_last_error,
            "preview_expected": failed_preview.preview_expected,
            "preview_failure_reason": failed_preview.preview_failure_reason,
            "sandbox_retained": failed_preview.sandbox_retained,
            "checkpoint_id": None,
            "updated_at": utc_now().isoformat(),
        }
    )
    next_summary = (
        summary
        or RunSummary(
            run_id=run_id,
            status="launch_failed",
            model_name=metadata.model_name
            if metadata is not None
            else default_model_name,
            fixture_name=metadata.fixture_name
            if metadata is not None
            else fixture_name,
        )
    ).model_copy(
        update={
            "status": "launch_failed",
            "failure_reason": failure_reason,
            "preview_url": failed_preview.preview_url,
            "preview_state": failed_preview.preview_state,
            "preview_last_error": failed_preview.preview_last_error,
            "preview_expected": failed_preview.preview_expected,
            "preview_failure_reason": failed_preview.preview_failure_reason,
            "sandbox_retained": failed_preview.sandbox_retained,
            "checkpoint_id": None,
            "command_count": 0,
            "diff_count": 0,
            "tool_call_count": 0,
        }
    )
    return TerminalRunState(status=next_status, summary=next_summary)


def build_terminal_run_state(
    *,
    metadata: RunMetadata,
    status: RunStatus,
    summary: RunSummary,
    outcome: TerminalOutcome,
    failure_reason: str | None = None,
    preview_url: str | None = None,
) -> TerminalRunState:
    if outcome == "succeeded":
        final_preview = finalize_execution_success(
            metadata=metadata,
            summary=summary,
            preview_url=preview_url,
        )
        next_summary = summary.model_copy(
            update={
                "status": "succeeded",
                "preview_url": final_preview.preview_url,
                "preview_state": final_preview.preview_state,
                "preview_last_error": final_preview.preview_last_error,
                "preview_failure_reason": final_preview.preview_failure_reason,
                "sandbox_retained": final_preview.sandbox_retained,
            }
        )
    else:
        if failure_reason is None:
            raise ValueError(
                f"failure_reason is required for terminal outcome {outcome}"
            )
        final_preview = finalize_execution_failure(
            metadata=metadata,
            status=status,
            summary=summary,
            preview_last_error=_preview_last_error(outcome),
            default_preview_failure_reason=_preview_failure_reason(outcome),
        )
        next_summary = summary.model_copy(
            update={
                "status": outcome,
                "failure_reason": failure_reason,
                "preview_url": final_preview.preview_url,
                "preview_state": final_preview.preview_state,
                "preview_last_error": final_preview.preview_last_error,
                "preview_failure_reason": final_preview.preview_failure_reason,
                "sandbox_retained": final_preview.sandbox_retained,
            }
        )

    next_status = status.model_copy(
        update={
            "state": next_summary.status,
            "current_model_name": next_summary.model_name,
            "preview_url": next_summary.preview_url,
            "preview_state": next_summary.preview_state,
            "preview_last_error": next_summary.preview_last_error,
            "preview_expected": next_summary.preview_expected,
            "preview_failure_reason": next_summary.preview_failure_reason,
            "sandbox_retained": next_summary.sandbox_retained,
            "checkpoint_id": next_summary.checkpoint_id,
            "updated_at": utc_now().isoformat(),
        }
    )
    return TerminalRunState(status=next_status, summary=next_summary)


def _default_launch_failed_status(
    run_id: str,
    model_name: str,
    metadata: RunMetadata | None,
) -> RunStatus:
    return RunStatus(
        run_id=run_id,
        state="launch_failed",
        current_model_name=model_name,
        is_fork=metadata.parent_run_id is not None if metadata is not None else False,
    )


def _preview_last_error(outcome: TerminalOutcome) -> str:
    if outcome == "provider_interrupted":
        return "Sandbox was not retained after provider interruption."
    return "Sandbox was not retained after run failure."


def _preview_failure_reason(outcome: TerminalOutcome) -> str:
    if outcome == "provider_interrupted":
        return "Preview was expected, but the provider interrupted the run before it could be verified."
    return "Preview was expected, but the run failed before it could be verified."
