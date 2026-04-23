from __future__ import annotations

from pydantic import BaseModel, Field

from .preview_execution import PreviewEvidence, project_preview_from_evidence
from .preview_lifecycle import TERMINAL_RUN_STATES, PreviewView
from .recorder import RunMetadata, RunStatus, RunSummary
from .run_evidence_projection import (
    ArenaLaneLifecycleStep,
    ArenaLanePhase,
    project_run_evidence,
)


class RunListItem(BaseModel):
    run_id: str
    status: str
    fixture_name: str
    model_name: str
    demo_summary: str | None = None
    preview_url: str | None = None
    preview_state: str = "unavailable"
    preview_last_error: str | None = None
    preview_expected: bool = False
    preview_failure_reason: str | None = None
    sandbox_retained: bool = False
    preview_attempted: bool = False
    preview_refresh_allowed: bool = False
    parent_run_id: str | None = None
    child_run_ids: list[str] = Field(default_factory=list)


class ArenaLaneSummary(BaseModel):
    lane_id: str
    run_id: str
    state: str
    model_name: str | None = None
    demo_summary: str | None = None
    preview_url: str | None = None
    preview_state: str = "unavailable"
    preview_last_error: str | None = None
    preview_expected: bool = False
    preview_failure_reason: str | None = None
    sandbox_retained: bool = False
    preview_attempted: bool = False
    preview_refresh_allowed: bool = False
    checkpoint_count: int = 0
    child_run_ids: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    started_at: str | None = None
    lifecycle_steps: list["ArenaLaneLifecycleStep"] = Field(default_factory=list)
    phase_key: str = "unknown"
    phase_label: str = "Unknown"
    phase_status: str = "pending"
    preview_diagnostic: str | None = None
    latest_checkpoint_sequence: int | None = None


class LoadedRunFacts(BaseModel):
    metadata: RunMetadata
    status: RunStatus | None = None
    summary: RunSummary | None = None
    preview_evidence: PreviewEvidence = Field(default_factory=PreviewEvidence)
    artifact_names: list[str] = Field(default_factory=list)
    child_run_ids: list[str] = Field(default_factory=list)
    checkpoint_count: int = 0
    latest_checkpoint_sequence: int | None = None
    event_lines: list[str] = Field(default_factory=list)


class ProjectedRun(BaseModel):
    metadata: RunMetadata
    status: RunStatus
    summary: RunSummary | None = None
    demo_summary: str | None = None
    list_item: RunListItem
    child_run_ids: list[str] = Field(default_factory=list)
    checkpoint_count: int = 0
    lifecycle_steps: list[ArenaLaneLifecycleStep] = Field(default_factory=list)
    phase: ArenaLanePhase
    preview_diagnostic: str | None = None
    latest_checkpoint_sequence: int | None = None


def project_run(facts: LoadedRunFacts) -> ProjectedRun:
    if facts.status is None and facts.summary is None:
        raise ValueError("run projection requires a status or summary")
    preview = project_preview_from_evidence(
        metadata=facts.metadata,
        status=facts.status,
        summary=facts.summary,
        evidence=facts.preview_evidence,
    )
    status = _project_status(facts, preview)
    summary = _project_summary(facts.summary, status)
    evidence = project_run_evidence(
        metadata=facts.metadata,
        status=status,
        summary=summary,
        checkpoint_count=facts.checkpoint_count,
        event_lines=facts.event_lines,
    )
    return ProjectedRun(
        metadata=facts.metadata,
        status=status,
        summary=summary,
        demo_summary=evidence.demo_summary,
        list_item=RunListItem(
            run_id=facts.metadata.run_id,
            status=status.state,
            fixture_name=facts.metadata.fixture_name,
            model_name=status.current_model_name,
            demo_summary=evidence.demo_summary,
            preview_url=preview.preview_url,
            preview_state=preview.preview_state,
            preview_last_error=preview.preview_last_error,
            preview_expected=preview.preview_expected,
            preview_failure_reason=preview.preview_failure_reason,
            sandbox_retained=preview.sandbox_retained,
            preview_attempted=preview.preview_attempted,
            preview_refresh_allowed=preview.preview_refresh_allowed,
            parent_run_id=facts.metadata.parent_run_id,
            child_run_ids=facts.child_run_ids,
        ),
        child_run_ids=facts.child_run_ids,
        checkpoint_count=facts.checkpoint_count,
        lifecycle_steps=evidence.lifecycle_steps,
        phase=evidence.phase,
        preview_diagnostic=evidence.preview_diagnostic,
        latest_checkpoint_sequence=facts.latest_checkpoint_sequence,
    )


def project_arena_lane(projected: ProjectedRun, *, lane_id: str) -> ArenaLaneSummary:
    summary = projected.summary
    status = projected.status
    return ArenaLaneSummary(
        lane_id=lane_id,
        run_id=projected.metadata.run_id,
        state=status.state,
        model_name=status.current_model_name,
        demo_summary=projected.demo_summary,
        preview_url=status.preview_url,
        preview_state=status.preview_state,
        preview_last_error=status.preview_last_error,
        preview_expected=status.preview_expected,
        preview_failure_reason=status.preview_failure_reason,
        sandbox_retained=status.sandbox_retained,
        preview_attempted=status.preview_attempted,
        preview_refresh_allowed=status.preview_refresh_allowed,
        checkpoint_count=projected.checkpoint_count,
        child_run_ids=projected.child_run_ids,
        failure_reason=summary.failure_reason if summary is not None else None,
        started_at=projected.metadata.started_at,
        lifecycle_steps=projected.lifecycle_steps,
        phase_key=projected.phase.key,
        phase_label=projected.phase.label,
        phase_status=projected.phase.status,
        preview_diagnostic=projected.preview_diagnostic,
        latest_checkpoint_sequence=projected.latest_checkpoint_sequence,
    )


def _project_status(facts: LoadedRunFacts, preview: PreviewView) -> RunStatus:
    status = facts.status
    summary = facts.summary
    if status is not None:
        updates: dict[str, str | bool | None] = {
            "preview_url": preview.preview_url,
            "preview_state": preview.preview_state,
            "preview_last_error": preview.preview_last_error,
            "preview_expected": preview.preview_expected,
            "preview_failure_reason": preview.preview_failure_reason,
            "sandbox_retained": preview.sandbox_retained,
            "preview_attempted": preview.preview_attempted,
            "preview_refresh_allowed": preview.preview_refresh_allowed,
        }
        if summary is not None and summary.status in TERMINAL_RUN_STATES:
            if status.state != summary.status:
                updates["state"] = summary.status
            if status.current_model_name != summary.model_name:
                updates["current_model_name"] = summary.model_name
            if status.checkpoint_id != summary.checkpoint_id:
                updates["checkpoint_id"] = summary.checkpoint_id
        elif (
            summary is not None
            and status.checkpoint_id is None
            and summary.checkpoint_id is not None
        ):
            updates["checkpoint_id"] = summary.checkpoint_id
        return status.model_copy(update=updates)

    if summary is None:
        raise ValueError("cannot synthesize run status without a summary")
    return RunStatus(
        run_id=summary.run_id,
        state=summary.status,
        current_model_name=summary.model_name,
        preview_url=preview.preview_url,
        preview_state=preview.preview_state,
        preview_last_error=preview.preview_last_error,
        preview_expected=preview.preview_expected,
        preview_failure_reason=preview.preview_failure_reason,
        sandbox_retained=preview.sandbox_retained,
        preview_attempted=preview.preview_attempted,
        preview_refresh_allowed=preview.preview_refresh_allowed,
        checkpoint_id=summary.checkpoint_id,
        is_fork=facts.metadata.parent_run_id is not None,
    )


def _project_summary(
    summary: RunSummary | None, status: RunStatus
) -> RunSummary | None:
    if summary is None:
        return None
    updates: dict[str, str | bool | None] = {
        "preview_url": status.preview_url,
        "preview_state": status.preview_state,
        "preview_last_error": status.preview_last_error,
        "preview_expected": status.preview_expected,
        "preview_failure_reason": status.preview_failure_reason,
        "sandbox_retained": status.sandbox_retained,
    }
    if summary.status != status.state:
        updates["status"] = status.state
    if summary.model_name != status.current_model_name:
        updates["model_name"] = status.current_model_name
    if summary.checkpoint_id != status.checkpoint_id:
        updates["checkpoint_id"] = status.checkpoint_id
    return summary.model_copy(update=updates)
