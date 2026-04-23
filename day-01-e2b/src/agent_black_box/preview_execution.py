from __future__ import annotations

import json
from json import JSONDecodeError
from typing import TYPE_CHECKING

from pydantic import BaseModel

from .preview_lifecycle import (
    PreviewView,
    RefreshProbeRequest,
    apply_command_preview_result,
    finalize_failed_preview,
    finalize_finished_preview,
    preview_expected_for_task,
    project_preview,
    refresh_probe_request,
)
from .recorder import RunMetadata, RunStatus, RunSummary
from .sandbox_controller import infer_preview_port, infer_preview_port_from_url

if TYPE_CHECKING:
    from .sandbox_controller import CommandResult, SandboxController


class PreviewEvidence(BaseModel):
    preview_url: str | None = None
    preview_port: int | None = None
    preview_failure_reason: str | None = None


class PreviewRefreshContext(BaseModel):
    evidence: PreviewEvidence
    request: RefreshProbeRequest


class CommandPreviewResult(BaseModel):
    preview_url: str | None = None
    preview_port: int | None = None
    preview_failure_reason: str | None = None


def build_preview_evidence(
    *,
    status: RunStatus | None,
    summary: RunSummary | None,
    event_lines: list[str],
    default_preview_port: int,
) -> PreviewEvidence:
    preview_url = _infer_preview_url(status=status, summary=summary, event_lines=event_lines)
    preview_port = _infer_preview_port(
        status=status,
        summary=summary,
        event_lines=event_lines,
        default_preview_port=default_preview_port,
    )
    preview_failure_reason = _infer_preview_failure_reason(
        event_lines=event_lines,
        default_preview_port=default_preview_port,
    )
    return PreviewEvidence(
        preview_url=preview_url,
        preview_port=preview_port,
        preview_failure_reason=preview_failure_reason,
    )


def project_preview_from_evidence(
    *,
    metadata: RunMetadata | None,
    status: RunStatus | None,
    summary: RunSummary | None,
    evidence: PreviewEvidence,
) -> PreviewView:
    return project_preview(
        status=status,
        summary=summary,
        metadata=metadata,
        inferred_preview_url=evidence.preview_url,
        inferred_preview_failure_reason=evidence.preview_failure_reason,
        inferred_preview_port=evidence.preview_port,
    )


def build_refresh_context(
    *,
    metadata: RunMetadata,
    status: RunStatus,
    summary: RunSummary | None,
    evidence: PreviewEvidence,
    default_preview_port: int,
) -> PreviewRefreshContext:
    request = refresh_probe_request(
        metadata=metadata,
        status=status,
        summary=summary,
        inferred_preview_url=evidence.preview_url,
        inferred_preview_failure_reason=evidence.preview_failure_reason,
        inferred_preview_port=evidence.preview_port,
        default_preview_port=default_preview_port,
    )
    return PreviewRefreshContext(evidence=evidence, request=request)


async def resolve_command_preview(
    *,
    sandbox_controller: SandboxController,
    command: str,
    command_result: CommandResult,
) -> CommandPreviewResult:
    preview_port = sandbox_controller.preview_port_for_command(
        command,
        stdout=command_result.stdout,
        stderr=command_result.stderr,
    )
    if preview_port is None:
        return CommandPreviewResult()
    if command_result.exit_code != 0:
        error = (command_result.stderr or command_result.stdout or "unknown error").strip()
        return CommandPreviewResult(
            preview_port=preview_port,
            preview_failure_reason=f"Preview command for port {preview_port} failed: {error}",
        )
    try:
        preview_url = await sandbox_controller.publish_preview(preview_port)
    except Exception as exc:  # noqa: BLE001
        return CommandPreviewResult(
            preview_port=preview_port,
            preview_failure_reason=(
                f"Preview server on port {preview_port} did not become publishable: {exc}"
            ),
        )
    return CommandPreviewResult(preview_url=preview_url, preview_port=preview_port)


def resolve_command_exception_preview(
    *,
    sandbox_controller: SandboxController,
    command: str,
    error: str,
) -> CommandPreviewResult:
    preview_port = sandbox_controller.preview_port_for_command(command)
    if preview_port is None:
        return CommandPreviewResult()
    return CommandPreviewResult(
        preview_port=preview_port,
        preview_failure_reason=f"Preview command for port {preview_port} failed before publication: {error}",
    )


def apply_command_preview(
    *,
    current_failure_reason: str | None,
    result: CommandPreviewResult,
) -> PreviewView:
    return apply_command_preview_result(
        current_failure_reason=current_failure_reason,
        preview_url=result.preview_url,
        preview_failure_reason=result.preview_failure_reason,
    )


def finalize_execution_success(
    *,
    metadata: RunMetadata | None,
    summary: RunSummary,
    preview_url: str | None,
) -> PreviewView:
    preview_expected = preview_expected_for_task(
        metadata.task if metadata is not None else "",
        metadata=metadata,
        summary=summary,
    )
    return finalize_finished_preview(
        preview_expected=preview_expected,
        preview_url=preview_url,
        preview_failure_reason=summary.preview_failure_reason,
    )


def finalize_execution_failure(
    *,
    metadata: RunMetadata | None,
    status: RunStatus | None,
    summary: RunSummary | None,
    evidence: PreviewEvidence | None = None,
    preview_last_error: str,
    default_preview_failure_reason: str,
) -> PreviewView:
    preview_expected = preview_expected_for_task(
        metadata.task if metadata is not None else "",
        metadata=metadata,
        summary=summary,
    )
    preview_failure_reason = None
    if summary is not None and summary.preview_failure_reason is not None:
        preview_failure_reason = summary.preview_failure_reason
    elif status is not None and status.preview_failure_reason is not None:
        preview_failure_reason = status.preview_failure_reason
    elif evidence is not None and evidence.preview_failure_reason is not None:
        preview_failure_reason = evidence.preview_failure_reason
    had_preview = (
        (summary.preview_url if summary is not None else None) is not None
        or (status.preview_url if status is not None else None) is not None
        or (evidence.preview_url if evidence is not None else None) is not None
    )
    return finalize_failed_preview(
        preview_expected=preview_expected,
        preview_failure_reason=preview_failure_reason,
        preview_last_error=preview_last_error,
        default_preview_failure_reason=default_preview_failure_reason,
        had_preview=had_preview,
    )


def _infer_preview_url(
    *,
    status: RunStatus | None,
    summary: RunSummary | None,
    event_lines: list[str],
) -> str | None:
    if status is not None and status.preview_url is not None:
        return status.preview_url
    if summary is not None and summary.preview_url is not None:
        return summary.preview_url
    for payload in _iter_payloads_reversed(event_lines):
        if payload.get("type") != "preview_published":
            continue
        url = payload.get("url")
        if isinstance(url, str) and url:
            return url
    return None


def _infer_preview_port(
    *,
    status: RunStatus | None,
    summary: RunSummary | None,
    event_lines: list[str],
    default_preview_port: int,
) -> int | None:
    preview_url = (
        status.preview_url
        if status is not None and status.preview_url is not None
        else summary.preview_url
        if summary is not None
        else None
    )
    if preview_url is not None:
        preview_port = infer_preview_port_from_url(preview_url)
        if preview_port is not None:
            return preview_port
    for payload in _iter_payloads_reversed(event_lines):
        payload_type = payload.get("type")
        if payload_type == "preview_published":
            preview_port = payload.get("port")
            if isinstance(preview_port, int):
                return preview_port
            continue
        if payload_type == "command_completed":
            exit_code = payload.get("exit_code")
            if not isinstance(exit_code, int) or exit_code != 0:
                continue
            preview_port = infer_preview_port(
                str(payload.get("command", "")),
                default_port=default_preview_port,
                stdout=str(payload.get("stdout", "")),
                stderr=str(payload.get("stderr", "")),
            )
            if preview_port is not None:
                return preview_port
            continue
        if payload_type == "tool_call" and payload.get("tool_name") == "run_command":
            arguments = payload.get("arguments", {})
            if isinstance(arguments, dict):
                preview_port = infer_preview_port(
                    str(arguments.get("command", "")),
                    default_port=default_preview_port,
                )
                if preview_port is not None:
                    return preview_port
    return None


def _infer_preview_failure_reason(
    *,
    event_lines: list[str],
    default_preview_port: int,
) -> str | None:
    for payload in _iter_payloads_reversed(event_lines):
        if payload.get("type") != "tool_result" or payload.get("tool_name") != "run_command":
            continue
        result = payload.get("result", {})
        if not isinstance(result, dict):
            continue
        failure_reason = result.get("preview_failure_reason")
        if isinstance(failure_reason, str) and failure_reason:
            return failure_reason
        if bool(payload.get("ok", False)):
            continue
        error = result.get("error")
        if not isinstance(error, str) or not error:
            continue
        preview_port = _preview_port_for_result(
            payload=payload,
            event_lines=event_lines,
            default_preview_port=default_preview_port,
        )
        if preview_port is None:
            continue
        return f"Preview command for port {preview_port} failed before publication: {error}"
    return None


def _preview_port_for_result(
    *,
    payload: dict[str, object],
    event_lines: list[str],
    default_preview_port: int,
) -> int | None:
    sequence = payload.get("sequence")
    if not isinstance(sequence, int):
        return None
    for candidate in _iter_payloads_reversed(event_lines):
        candidate_sequence = candidate.get("sequence")
        if not isinstance(candidate_sequence, int) or candidate_sequence >= sequence:
            continue
        if candidate.get("type") != "tool_call":
            continue
        if candidate.get("tool_name") != "run_command":
            return None
        arguments = candidate.get("arguments", {})
        if not isinstance(arguments, dict):
            return None
        return infer_preview_port(
            str(arguments.get("command", "")),
            default_port=default_preview_port,
        )
    return None


def _iter_payloads_reversed(event_lines: list[str]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for raw in reversed(event_lines):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        payloads.append(payload)
    return payloads
