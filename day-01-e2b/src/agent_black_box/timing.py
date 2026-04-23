from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from .events import load_event, utc_now
from .run_store import RunStore


class ProviderAttemptTiming(BaseModel):
    turn_number: int
    phase: str
    model_name: str | None = None
    attempt_number: int
    timeout_seconds: float
    is_fallback: bool = False
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    state: str
    outcome: str | None = None
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    ollama_total_duration_seconds: float | None = None
    ollama_load_duration_seconds: float | None = None
    ollama_prompt_eval_count: int | None = None
    ollama_prompt_eval_duration_seconds: float | None = None
    ollama_eval_count: int | None = None
    ollama_eval_duration_seconds: float | None = None


class ModelTurnTiming(BaseModel):
    turn_number: int
    started_at: datetime
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    state: str
    model_name: str | None = None
    message_count: int | None = None
    tool_schema_count: int | None = None
    conversation_chars: int | None = None
    last_message_chars: int | None = None
    last_tool_result_chars: int | None = None
    request_body_bytes: int | None = None
    finish_reason: str | None = None
    tool_name: str | None = None
    provider_attempts: list[ProviderAttemptTiming] = Field(default_factory=list)


class RunTimingReport(BaseModel):
    run_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    last_event_type: str | None = None
    last_event_at: datetime | None = None
    last_event_age_seconds: float | None = None
    model_turns: list[ModelTurnTiming] = Field(default_factory=list)


def build_run_timing_report(
    store: RunStore,
    run_id: str,
    *,
    now: datetime | None = None,
) -> RunTimingReport:
    store.get_run_dir(run_id)
    current_time = now or utc_now()
    model_turns_by_number: dict[int, ModelTurnTiming] = {}
    completed_turns: set[int] = set()
    provider_attempts: dict[
        tuple[int, str, str | None, int, bool],
        ProviderAttemptTiming,
    ] = {}
    last_event_type: str | None = None
    last_event_at: datetime | None = None

    for raw in store.load_event_lines(run_id):
        if not raw.strip():
            continue
        try:
            event = load_event(raw)
        except Exception:  # noqa: BLE001
            continue
        last_event_type = event.type
        last_event_at = event.timestamp
        if event.type == "model_turn_started":
            model_turns_by_number[event.turn_number] = ModelTurnTiming(
                turn_number=event.turn_number,
                started_at=event.timestamp,
                duration_seconds=(current_time - event.timestamp).total_seconds(),
                state="open",
                model_name=event.model_name,
                message_count=event.message_count,
                tool_schema_count=event.tool_schema_count,
                conversation_chars=event.conversation_chars,
                last_message_chars=event.last_message_chars,
                last_tool_result_chars=event.last_tool_result_chars,
                request_body_bytes=event.request_body_bytes,
            )
        elif event.type == "model_turn_completed":
            existing = model_turns_by_number.get(event.turn_number)
            started_at = (
                existing.started_at if existing is not None else event.timestamp
            )
            duration = (
                (event.timestamp - started_at).total_seconds()
                if started_at is not None
                else None
            )
            completed_turns.add(event.turn_number)
            model_turns_by_number[event.turn_number] = (
                existing
                or ModelTurnTiming(
                    turn_number=event.turn_number,
                    started_at=started_at,
                    state="open",
                )
            ).model_copy(
                update={
                    "completed_at": event.timestamp,
                    "duration_seconds": duration,
                    "state": "completed",
                    "finish_reason": event.finish_reason,
                    "tool_name": event.tool_name,
                }
            )
        elif event.type == "model_provider_attempt_started":
            key = _provider_attempt_key(
                turn_number=event.turn_number,
                phase=event.phase,
                model_name=event.model_name,
                attempt_number=event.attempt_number,
                is_fallback=event.is_fallback,
            )
            provider_attempts[key] = ProviderAttemptTiming(
                turn_number=event.turn_number,
                phase=event.phase,
                model_name=event.model_name,
                attempt_number=event.attempt_number,
                timeout_seconds=event.timeout_seconds,
                is_fallback=event.is_fallback,
                started_at=event.timestamp,
                duration_seconds=(current_time - event.timestamp).total_seconds(),
                state="open",
            )
        elif event.type == "model_provider_attempt_completed":
            key = _provider_attempt_key(
                turn_number=event.turn_number,
                phase=event.phase,
                model_name=event.model_name,
                attempt_number=event.attempt_number,
                is_fallback=event.is_fallback,
            )
            existing_attempt = provider_attempts.get(key)
            started_at = (
                existing_attempt.started_at
                if existing_attempt is not None
                else event.timestamp - timedelta(seconds=event.duration_seconds)
            )
            provider_attempts[key] = (
                existing_attempt
                or ProviderAttemptTiming(
                    turn_number=event.turn_number,
                    phase=event.phase,
                    model_name=event.model_name,
                    attempt_number=event.attempt_number,
                    timeout_seconds=event.timeout_seconds,
                    is_fallback=event.is_fallback,
                    started_at=started_at,
                    state="open",
                )
            ).model_copy(
                update={
                    "completed_at": event.timestamp,
                    "duration_seconds": event.duration_seconds,
                    "state": "completed",
                    "outcome": event.outcome,
                    "status_code": event.status_code,
                    "error_type": event.error_type,
                    "error_message": event.error_message,
                    "ollama_total_duration_seconds": event.ollama_total_duration_seconds,
                    "ollama_load_duration_seconds": event.ollama_load_duration_seconds,
                    "ollama_prompt_eval_count": event.ollama_prompt_eval_count,
                    "ollama_prompt_eval_duration_seconds": event.ollama_prompt_eval_duration_seconds,
                    "ollama_eval_count": event.ollama_eval_count,
                    "ollama_eval_duration_seconds": event.ollama_eval_duration_seconds,
                }
            )

    for attempt in provider_attempts.values():
        turn = model_turns_by_number.get(attempt.turn_number)
        if turn is None:
            turn = ModelTurnTiming(
                turn_number=attempt.turn_number,
                started_at=attempt.started_at,
                duration_seconds=(current_time - attempt.started_at).total_seconds(),
                state="open",
            )
            model_turns_by_number[attempt.turn_number] = turn
        turn.provider_attempts.append(attempt)

    model_turns = sorted(
        model_turns_by_number.values(),
        key=lambda turn: (turn.turn_number, turn.completed_at is None),
    )
    for turn in model_turns:
        turn.provider_attempts.sort(
            key=lambda attempt: (
                attempt.started_at,
                attempt.phase,
                attempt.is_fallback,
                attempt.attempt_number,
            )
        )
        if turn.turn_number in completed_turns:
            continue
        turn.duration_seconds = (current_time - turn.started_at).total_seconds()
        turn.state = "open"

    return RunTimingReport(
        run_id=run_id,
        generated_at=current_time,
        last_event_type=last_event_type,
        last_event_at=last_event_at,
        last_event_age_seconds=(
            (current_time - last_event_at).total_seconds()
            if last_event_at is not None
            else None
        ),
        model_turns=model_turns,
    )


def _provider_attempt_key(
    *,
    turn_number: int,
    phase: str,
    model_name: str | None,
    attempt_number: int,
    is_fallback: bool,
) -> tuple[int, str, str | None, int, bool]:
    return (turn_number, phase, model_name, attempt_number, is_fallback)
