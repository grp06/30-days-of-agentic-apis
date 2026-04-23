from __future__ import annotations

from agent_black_box.model_types import ModelDecision
from agent_black_box.model_protocol import (
    ProtocolContext,
    ProtocolFailure,
    classify_model_decision,
    protocol_failure_kind_from_exception,
    terminal_summary_from_message,
)


def _context(
    *,
    saw_diff: bool = False,
    repair_attempts: int = 0,
    max_repair_attempts: int = 2,
    turn_number: int = 1,
    max_turns: int = 5,
) -> ProtocolContext:
    return ProtocolContext(
        saw_diff=saw_diff,
        repair_attempts=repair_attempts,
        max_repair_attempts=max_repair_attempts,
        turn_number=turn_number,
        max_turns=max_turns,
    )


def test_valid_tool_call_is_accepted() -> None:
    action = classify_model_decision(
        ModelDecision(
            finish_reason="tool_call",
            tool_name="read_file",
            tool_arguments={"path": "index.html"},
        ),
        _context(),
    )

    assert action.kind == "tool_call"
    assert action.failure_kind is None


def test_tool_call_without_name_fails() -> None:
    action = classify_model_decision(
        ModelDecision(finish_reason="tool_call"),
        _context(),
    )

    assert action.kind == "fail"
    assert action.failure_kind == "tool_call_without_name"
    assert action.reason == "Model returned tool_call finish_reason without a tool name"


def test_plain_text_completion_after_diff_becomes_terminal_summary() -> None:
    action = classify_model_decision(
        ModelDecision(
            finish_reason="completed",
            message="  Done\n\nand verified. ",
        ),
        _context(saw_diff=True),
    )

    assert action.kind == "terminal_summary"
    assert action.terminal_summary == "Done and verified."


def test_plain_text_completion_before_diff_requests_repair() -> None:
    action = classify_model_decision(
        ModelDecision(
            finish_reason="completed",
            message="Done",
        ),
        _context(saw_diff=False),
    )

    assert action.kind == "repair"
    assert action.failure_kind == "plain_text_before_diff"
    assert action.repair_message is not None
    assert "Call exactly one tool now" in action.repair_message
    assert "Use apply_patch or write_file" in action.repair_message


def test_generation_limit_completion_requests_short_repair() -> None:
    action = classify_model_decision(
        ModelDecision(
            finish_reason="completed",
            message="",
            hit_generation_limit=True,
        ),
        _context(),
    )

    assert action.kind == "repair"
    assert action.failure_kind == "hit_generation_limit"
    assert action.hit_generation_limit is True
    assert action.repair_message is not None
    assert "output token limit" in action.repair_message
    assert "Call exactly one tool now" in action.repair_message


def test_invalid_completion_fails_after_repair_budget_is_exhausted() -> None:
    action = classify_model_decision(
        ModelDecision(
            finish_reason="completed",
            message="",
        ),
        _context(repair_attempts=2, max_repair_attempts=2),
    )

    assert action.kind == "fail"
    assert action.failure_kind == "completed_without_action"
    assert action.reason == (
        "Model protocol incomplete: model ended the turn without calling a "
        "tool or finish_run."
    )


def test_invalid_completion_fails_on_final_turn() -> None:
    action = classify_model_decision(
        ModelDecision(
            finish_reason="completed",
            message="",
        ),
        _context(turn_number=5, max_turns=5),
    )

    assert action.kind == "fail"
    assert action.failure_kind == "completed_without_action"


def test_terminal_summary_from_message_collapses_whitespace() -> None:
    assert terminal_summary_from_message("  done\n\nnow\t") == "done now"
    assert terminal_summary_from_message("  \n\t") is None
    assert terminal_summary_from_message(None) is None


def test_protocol_failure_exception_exposes_kind() -> None:
    exc = ProtocolFailure(
        failure_kind="completed_without_action",
        reason="Model protocol incomplete",
    )

    assert str(exc) == "Model protocol incomplete"
    assert protocol_failure_kind_from_exception(exc) == "completed_without_action"
    assert protocol_failure_kind_from_exception(RuntimeError("boom")) is None
