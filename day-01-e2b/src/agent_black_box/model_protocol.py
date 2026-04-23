from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from .model_types import ModelDecision


ProtocolFailureKind = Literal[
    "hit_generation_limit",
    "plain_text_before_diff",
    "completed_without_action",
    "tool_call_without_name",
]

ProtocolActionKind = Literal[
    "tool_call",
    "terminal_summary",
    "repair",
    "fail",
]


class ProtocolContext(BaseModel):
    saw_diff: bool
    repair_attempts: int
    max_repair_attempts: int
    turn_number: int
    max_turns: int


class ProtocolAction(BaseModel):
    kind: ProtocolActionKind
    failure_kind: ProtocolFailureKind | None = None
    reason: str | None = None
    repair_message: str | None = None
    terminal_summary: str | None = None
    hit_generation_limit: bool = False


class ProtocolFailure(RuntimeError):
    def __init__(self, *, failure_kind: ProtocolFailureKind, reason: str) -> None:
        super().__init__(reason)
        self.failure_kind = failure_kind
        self.reason = reason


def classify_model_decision(
    decision: ModelDecision,
    context: ProtocolContext,
) -> ProtocolAction:
    if decision.finish_reason != "completed":
        if decision.tool_name is not None:
            return ProtocolAction(kind="tool_call")
        failure_kind: ProtocolFailureKind = "tool_call_without_name"
        return ProtocolAction(
            kind="fail",
            failure_kind=failure_kind,
            reason=protocol_failure_reason(failure_kind),
        )

    terminal_summary = terminal_summary_from_message(decision.message)
    if terminal_summary is not None and context.saw_diff:
        return ProtocolAction(
            kind="terminal_summary",
            terminal_summary=terminal_summary,
        )

    failure_kind = _completed_failure_kind(
        decision=decision,
        terminal_summary=terminal_summary,
        saw_diff=context.saw_diff,
    )
    reason = protocol_failure_reason(failure_kind)
    can_repair = (
        context.repair_attempts < context.max_repair_attempts
        and context.turn_number < context.max_turns
    )
    if can_repair:
        return ProtocolAction(
            kind="repair",
            failure_kind=failure_kind,
            reason=reason,
            repair_message=protocol_repair_message(
                reason=reason,
                failure_kind=failure_kind,
                saw_diff=context.saw_diff,
            ),
            hit_generation_limit=decision.hit_generation_limit,
        )
    return ProtocolAction(
        kind="fail",
        failure_kind=failure_kind,
        reason=reason,
        hit_generation_limit=decision.hit_generation_limit,
    )


def terminal_summary_from_message(message: str | None) -> str | None:
    if message is None:
        return None
    summary = re.sub(r"\s+", " ", message).strip()
    return summary or None


def protocol_failure_reason(kind: ProtocolFailureKind) -> str:
    if kind == "hit_generation_limit":
        return (
            "Model protocol incomplete: model hit the output token limit before "
            "calling a tool or finish_run."
        )
    if kind == "plain_text_before_diff":
        return (
            "Model protocol incomplete: model returned plain text before "
            "producing a workspace diff."
        )
    if kind == "tool_call_without_name":
        return "Model returned tool_call finish_reason without a tool name"
    return (
        "Model protocol incomplete: model ended the turn without calling a "
        "tool or finish_run."
    )


def protocol_repair_message(
    *,
    reason: str,
    failure_kind: ProtocolFailureKind,
    saw_diff: bool,
) -> str:
    parts = [
        f"{reason} No work can be accepted from that response.",
        "Call exactly one tool now.",
    ]
    if failure_kind == "hit_generation_limit":
        parts.append(
            "Your previous response appears to have hit the output token limit; "
            "keep this response short."
        )
    if saw_diff:
        parts.append("If the workspace is complete, call finish_run with a concise summary.")
    else:
        parts.append(
            "Use apply_patch or write_file to edit the workspace; read_file only "
            "if another file is truly required."
        )
    parts.append("Do not answer in plain text.")
    return " ".join(parts)


def protocol_failure_kind_from_exception(exc: Exception) -> str | None:
    if isinstance(exc, ProtocolFailure):
        return exc.failure_kind
    return None


def legacy_protocol_failure_from_reason(reason: str | None) -> bool:
    if reason is None:
        return False
    return "without calling finish_run" in reason or "Model protocol incomplete" in reason


def _completed_failure_kind(
    *,
    decision: ModelDecision,
    terminal_summary: str | None,
    saw_diff: bool,
) -> ProtocolFailureKind:
    if decision.hit_generation_limit:
        return "hit_generation_limit"
    if terminal_summary is not None and not saw_diff:
        return "plain_text_before_diff"
    return "completed_without_action"
