from __future__ import annotations

from pydantic import BaseModel, Field

from .events import CommandCompletedEvent, CommandStartedEvent, Event, load_event
from .fixture_policy import fixture_is_build_command
from .model_protocol import legacy_protocol_failure_from_reason
from .preview_lifecycle import TERMINAL_RUN_STATES
from .recorder import RunMetadata, RunStatus, RunSummary


class ArenaLaneLifecycleStep(BaseModel):
    key: str
    label: str
    status: str
    detail: str | None = None


class ArenaLanePhase(BaseModel):
    key: str
    label: str
    status: str


class RunEvidenceProjection(BaseModel):
    demo_summary: str
    lifecycle_steps: list[ArenaLaneLifecycleStep] = Field(default_factory=list)
    phase: ArenaLanePhase
    preview_diagnostic: str | None = None


class _LifecycleFacts(BaseModel):
    saw_model_turn: bool = False
    saw_command_started: bool = False
    saw_command: bool = False
    saw_diff: bool = False
    build_started: bool = False
    build_passed: bool = False
    build_failed: bool = False
    preview_published: bool = False
    protocol_failure_kind: str | None = None


def project_run_evidence(
    *,
    metadata: RunMetadata,
    status: RunStatus,
    summary: RunSummary | None,
    checkpoint_count: int,
    event_lines: list[str],
) -> RunEvidenceProjection:
    lifecycle = _lifecycle_facts(event_lines, fixture_name=metadata.fixture_name)
    preview_diagnostic = _preview_diagnostic(status)
    lifecycle_steps = _lifecycle_steps(
        metadata=metadata,
        status=status,
        summary=summary,
        lifecycle=lifecycle,
        checkpoint_count=checkpoint_count,
        preview_diagnostic=preview_diagnostic,
    )
    phase = _lane_phase(status=status, lifecycle_steps=lifecycle_steps)
    return RunEvidenceProjection(
        demo_summary=_demo_summary(status, summary, lifecycle),
        lifecycle_steps=lifecycle_steps,
        phase=phase,
        preview_diagnostic=preview_diagnostic,
    )


def _demo_summary(
    status: RunStatus,
    summary: RunSummary | None,
    lifecycle: _LifecycleFacts,
) -> str:
    if status.state == "succeeded":
        if summary is not None and summary.failure_reason:
            return "Finished with a model protocol warning; harness validation produced usable evidence."
        if (
            status.preview_state in {"live", "retained"}
            and status.preview_url is not None
        ):
            return "Finished successfully and produced a preview you can inspect."
        if status.preview_expected:
            return "Finished successfully, but the preview flow never stabilized."
        return "Finished successfully without a published preview."
    if status.state == "running":
        if status.preview_state == "live" and status.preview_url is not None:
            return "Still running, with a live preview already available."
        return "Still running through the task."
    if status.state == "launch_failed":
        return "Failed before the agent could take meaningful action."
    if status.state == "provider_interrupted":
        return "Stopped after the model provider interrupted the run."
    if summary is not None and summary.failure_reason:
        if "Configured Ollama model" in summary.failure_reason:
            return f"Invalid model configuration: {summary.failure_reason}"
        if (
            "timed out" in summary.failure_reason
            or "ReadTimeout" in summary.failure_reason
        ):
            return f"Model provider timeout: {summary.failure_reason}"
        if lifecycle.protocol_failure_kind is not None or (
            lifecycle.protocol_failure_kind is None
            and legacy_protocol_failure_from_reason(summary.failure_reason)
        ):
            if summary.failure_reason.startswith("Model protocol incomplete:"):
                return summary.failure_reason
            return f"Model protocol incomplete: {summary.failure_reason}"
        if "Managed build command failed" in summary.failure_reason:
            return f"Build failed: {summary.failure_reason}"
        if status.preview_state in {"expired", "retained"} and (
            status.preview_url or status.preview_expected
        ):
            return (
                "Failed after making visible progress, with preserved preview evidence."
            )
        return f"Failed: {summary.failure_reason}"
    return f"Ended in state: {status.state}."


def _lifecycle_facts(event_lines: list[str], *, fixture_name: str) -> _LifecycleFacts:
    facts = _LifecycleFacts()
    for raw in event_lines:
        if not raw.strip():
            continue
        try:
            event = load_event(raw)
        except Exception:  # noqa: BLE001
            continue
        _apply_lifecycle_event(facts, event, fixture_name=fixture_name)
    return facts


def _apply_lifecycle_event(
    facts: _LifecycleFacts, event: Event, *, fixture_name: str
) -> None:
    if event.type == "model_turn_started":
        facts.saw_model_turn = True
    elif event.type == "command_started":
        facts.saw_command_started = True
        if _is_build_command(event, fixture_name=fixture_name):
            facts.build_started = True
    elif event.type == "command_completed":
        facts.saw_command = True
        if _is_build_command(event, fixture_name=fixture_name):
            facts.build_started = True
            if event.exit_code == 0:
                facts.build_passed = True
            else:
                facts.build_failed = True
    elif event.type == "file_diff":
        facts.saw_diff = True
    elif event.type == "preview_published":
        facts.preview_published = True
    elif event.type == "run_failed":
        failure_kind = getattr(event, "failure_kind", None)
        if isinstance(failure_kind, str) and failure_kind:
            facts.protocol_failure_kind = failure_kind


def _is_build_command(
    event: CommandCompletedEvent | CommandStartedEvent, *, fixture_name: str
) -> bool:
    if fixture_is_build_command(fixture_name, event.command):
        return True
    command = event.command.lower()
    # Projection reads historical events; keep old build literals readable even if
    # the current configured fixture command changes later.
    return "pnpm build" in command or "npm run build" in command


def _lifecycle_steps(
    *,
    metadata: RunMetadata,
    status: RunStatus,
    summary: RunSummary | None,
    lifecycle: _LifecycleFacts,
    checkpoint_count: int,
    preview_diagnostic: str | None,
) -> list[ArenaLaneLifecycleStep]:
    failure_step = _failure_step(status, summary, lifecycle)
    return [
        ArenaLaneLifecycleStep(
            key="sandbox",
            label="Sandbox",
            status="ok" if metadata.sandbox_id else _pending_or_error(status),
            detail="E2B workspace created"
            if metadata.sandbox_id
            else "Waiting for sandbox",
        ),
        ArenaLaneLifecycleStep(
            key="agent",
            label="Agent work",
            status=_agent_step_status(
                status,
                lifecycle,
                failure_step=failure_step,
                sandbox_ready=metadata.sandbox_id is not None,
            ),
            detail=_agent_step_detail(summary, lifecycle, failure_step=failure_step),
        ),
        ArenaLaneLifecycleStep(
            key="build",
            label="Build",
            status=_build_step_status(status, lifecycle, failure_step=failure_step),
            detail=_build_step_detail(summary, lifecycle, failure_step=failure_step),
        ),
        ArenaLaneLifecycleStep(
            key="preview",
            label="Preview",
            status=_preview_step_status(status, lifecycle, failure_step=failure_step),
            detail=_preview_step_detail(
                preview_diagnostic=preview_diagnostic,
                failure_step=failure_step,
            ),
        ),
        ArenaLaneLifecycleStep(
            key="evidence",
            label="Evidence",
            status=_evidence_step_status(status, lifecycle, failure_step=failure_step),
            detail=_evidence_step_detail(lifecycle, failure_step=failure_step),
        ),
        ArenaLaneLifecycleStep(
            key="snapshot",
            label="Snapshot",
            status=_snapshot_step_status(
                status, lifecycle, checkpoint_count, failure_step=failure_step
            ),
            detail=_snapshot_step_detail(
                lifecycle, checkpoint_count, failure_step=failure_step
            ),
        ),
    ]


def _failure_step(
    status: RunStatus, summary: RunSummary | None, lifecycle: _LifecycleFacts
) -> str | None:
    if status.state not in {"failed", "launch_failed", "provider_interrupted"}:
        return None
    reason = (summary.failure_reason if summary is not None else None) or ""
    if lifecycle.build_failed or "Managed build command failed" in reason:
        return "build"
    if status.preview_failure_reason is not None and lifecycle.build_passed:
        return "preview"
    return "agent"


def _agent_step_status(
    status: RunStatus,
    lifecycle: _LifecycleFacts,
    *,
    failure_step: str | None,
    sandbox_ready: bool,
) -> str:
    if failure_step == "agent":
        return "error"
    if lifecycle.saw_diff:
        return "ok"
    if status.state == "running" and (
        sandbox_ready
        or lifecycle.saw_model_turn
        or lifecycle.saw_command_started
        or lifecycle.saw_command
    ):
        return "active"
    if status.state in TERMINAL_RUN_STATES:
        return "skipped"
    return "pending"


def _agent_step_detail(
    summary: RunSummary | None, lifecycle: _LifecycleFacts, *, failure_step: str | None
) -> str:
    if failure_step == "agent" and summary is not None and summary.failure_reason:
        return summary.failure_reason
    if lifecycle.saw_diff:
        return "Workspace diff recorded"
    if lifecycle.saw_command_started or lifecycle.saw_command:
        return "Commands are running"
    if lifecycle.saw_model_turn:
        return "Model is working"
    return "Waiting for first file change"


def _build_step_status(
    status: RunStatus, lifecycle: _LifecycleFacts, *, failure_step: str | None
) -> str:
    if failure_step == "agent":
        return "skipped"
    if lifecycle.build_passed:
        return "ok"
    if lifecycle.build_failed or failure_step == "build":
        return "error"
    if status.state == "running" and (lifecycle.build_started or lifecycle.saw_diff):
        return "active"
    return _terminal_missing_or_pending(status)


def _build_step_detail(
    summary: RunSummary | None, lifecycle: _LifecycleFacts, *, failure_step: str | None
) -> str:
    if failure_step == "agent":
        return "Not reached"
    if failure_step == "build" and summary is not None and summary.failure_reason:
        return summary.failure_reason
    if lifecycle.build_passed:
        return "Build command passed"
    if lifecycle.build_failed:
        return "Build command failed"
    if lifecycle.build_started:
        return "Build command is running"
    return "No passing build recorded"


def _preview_step_status(
    status: RunStatus, lifecycle: _LifecycleFacts, *, failure_step: str | None
) -> str:
    if failure_step in {"agent", "build"}:
        return "skipped"
    if (
        status.preview_url is not None
        and status.preview_state in {"live", "retained"}
        or lifecycle.preview_published
    ):
        return "ok"
    if (
        status.preview_failure_reason is not None
        or status.preview_last_error is not None
    ):
        return "warning" if status.state == "succeeded" else "error"
    if status.preview_expected and status.state == "running" and lifecycle.build_passed:
        return "active"
    return "pending"


def _preview_step_detail(
    *, preview_diagnostic: str | None, failure_step: str | None
) -> str | None:
    if failure_step == "agent":
        return "Not reached"
    if failure_step == "build":
        return "Not attempted because build failed"
    return preview_diagnostic


def _evidence_step_status(
    status: RunStatus, lifecycle: _LifecycleFacts, *, failure_step: str | None
) -> str:
    if lifecycle.preview_published or status.preview_url is not None:
        return "ok"
    if failure_step in {"agent", "build", "preview"}:
        return "skipped"
    if (
        status.state == "running"
        and status.preview_expected
        and (lifecycle.preview_published or status.preview_url is not None)
    ):
        return "active"
    return _terminal_missing_or_pending(status)


def _evidence_step_detail(
    lifecycle: _LifecycleFacts, *, failure_step: str | None
) -> str:
    if failure_step in {"agent", "build", "preview"}:
        return "Not reached"
    if lifecycle.preview_published:
        return "Preview URL recorded"
    return "No preview evidence recorded yet"


def _snapshot_step_status(
    status: RunStatus,
    lifecycle: _LifecycleFacts,
    checkpoint_count: int,
    *,
    failure_step: str | None,
) -> str:
    if checkpoint_count > 0:
        return "ok"
    if failure_step in {"agent", "build", "preview"}:
        return "skipped"
    if status.state == "running" and (lifecycle.preview_published or status.preview_url):
        return "active"
    return _terminal_missing_or_pending(status)


def _snapshot_step_detail(
    lifecycle: _LifecycleFacts, checkpoint_count: int, *, failure_step: str | None
) -> str:
    if failure_step in {"agent", "build", "preview"}:
        return "Not reached"
    if checkpoint_count > 0:
        return f"{checkpoint_count} checkpoint{'s' if checkpoint_count != 1 else ''}"
    return "No checkpoint yet"


def _lane_phase(
    *, status: RunStatus, lifecycle_steps: list[ArenaLaneLifecycleStep]
) -> ArenaLanePhase:
    if status.state == "succeeded":
        return ArenaLanePhase(key="succeeded", label="Succeeded", status="ok")
    if status.state in {"failed", "launch_failed"}:
        failed_step = next(
            (step for step in lifecycle_steps if step.status == "error"), None
        )
        if failed_step is not None and failed_step.detail:
            return ArenaLanePhase(
                key=failed_step.key,
                label=failed_step.detail,
                status="error",
            )
        return ArenaLanePhase(key="failed", label="Failed", status="error")
    if status.state == "provider_interrupted":
        return ArenaLanePhase(key="interrupted", label="Interrupted", status="warning")
    active_step = next(
        (step for step in lifecycle_steps if step.status == "active"), None
    )
    if active_step is not None:
        labels = {
            "sandbox": "Starting sandbox",
            "agent": "Model is editing files",
            "build": "Building",
            "preview": "Publishing preview",
            "evidence": "Capturing evidence",
            "snapshot": "Saving snapshot",
        }
        return ArenaLanePhase(
            key=active_step.key,
            label=labels.get(active_step.key, active_step.label),
            status="active",
        )
    if status.preview_url is not None and status.preview_state in {"live", "retained"}:
        return ArenaLanePhase(key="preview", label="Preview ready", status="ok")
    return ArenaLanePhase(key=status.state, label=status.state.replace("_", " ").title(), status="pending")


def _pending_or_error(status: RunStatus) -> str:
    return "error" if status.state in {"failed", "launch_failed"} else "pending"


def _terminal_missing_or_pending(status: RunStatus) -> str:
    if status.state == "running":
        return "pending"
    if status.state in TERMINAL_RUN_STATES:
        return "warning"
    return "pending"


def _preview_diagnostic(status: RunStatus) -> str | None:
    if status.preview_url is not None and status.preview_state in {"live", "retained"}:
        return "Preview published through E2B and can be opened from this lane."
    raw = status.preview_failure_reason or status.preview_last_error
    if raw is None:
        return None
    lowered = raw.lower()
    if (
        "not allowed" in lowered
        or "allowedhosts" in lowered
        or "allowed hosts" in lowered
    ):
        return "Vite rejected the E2B preview host. Add the .e2b.app host pattern to vite.config.js."
    if "did not become ready" in lowered or "server_not_running" in lowered:
        return "The expected preview port never served a ready HTTP response."
    if "context deadline exceeded" in lowered or "timeout" in lowered:
        return "Preview publication timed out before the server stabilized."
    if "not found" in lowered or "no longer available" in lowered:
        return "The preserved sandbox is no longer available."
    if "failed" in lowered:
        return raw
    return raw
