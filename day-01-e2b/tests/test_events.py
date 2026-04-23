from __future__ import annotations

from agent_black_box.events import (
    ModelProviderAttemptCompletedEvent,
    ModelProviderAttemptStartedEvent,
    ModelTurnStartedEvent,
    ProtocolRepairRequestedEvent,
    RunFailedEvent,
    RunStartedEvent,
    ToolCallEvent,
    dump_event,
    load_event,
)


def test_event_round_trip_run_started() -> None:
    event = RunStartedEvent(
        run_id="run-1",
        lane_id="lane-1",
        sequence=1,
        task="hello",
        fixture_name="fixture",
        model="kimi-k2.6:cloud",
    )

    loaded = load_event(dump_event(event))
    assert loaded == event


def test_event_round_trip_tool_call() -> None:
    event = ToolCallEvent(
        run_id="run-1",
        lane_id="lane-1",
        sequence=2,
        tool_name="run_command",
        arguments={"command": "pwd", "timeout_seconds": 30},
    )

    loaded = load_event(dump_event(event))
    assert loaded == event


def test_event_round_trip_model_turn_request_metrics() -> None:
    event = ModelTurnStartedEvent(
        run_id="run-1",
        lane_id="lane-1",
        sequence=3,
        turn_number=4,
        model_name="kimi-k2.6:cloud",
        message_count=8,
        tool_schema_count=5,
        conversation_chars=1200,
        last_message_chars=250,
        last_tool_result_chars=250,
        request_body_bytes=2200,
    )

    loaded = load_event(dump_event(event))
    assert loaded == event


def test_event_round_trip_model_provider_attempts() -> None:
    started = ModelProviderAttemptStartedEvent(
        run_id="run-1",
        lane_id="lane-1",
        sequence=4,
        turn_number=4,
        phase="chat",
        model_name="kimi-k2.6",
        attempt_number=2,
        timeout_seconds=120.0,
    )
    completed = ModelProviderAttemptCompletedEvent(
        run_id="run-1",
        lane_id="lane-1",
        sequence=5,
        turn_number=4,
        phase="chat",
        model_name="kimi-k2.6",
        attempt_number=2,
        timeout_seconds=120.0,
        duration_seconds=120.1,
        outcome="timeout",
        error_type="ReadTimeout",
        error_message="timed out",
        ollama_total_duration_seconds=1.25,
        ollama_load_duration_seconds=0.05,
        ollama_prompt_eval_count=42,
        ollama_prompt_eval_duration_seconds=0.2,
        ollama_eval_count=12,
        ollama_eval_duration_seconds=0.9,
        response_done_reason="length",
        response_content_chars=0,
        response_tool_call_count=0,
        hit_generation_limit=True,
    )

    assert load_event(dump_event(started)) == started
    assert load_event(dump_event(completed)) == completed


def test_event_round_trip_protocol_repair_requested() -> None:
    event = ProtocolRepairRequestedEvent(
        run_id="run-1",
        lane_id="lane-1",
        sequence=6,
        turn_number=4,
        repair_attempt=1,
        reason="Model protocol incomplete",
        failure_kind="hit_generation_limit",
        hit_generation_limit=True,
        message="Call exactly one tool now.",
    )

    assert load_event(dump_event(event)) == event


def test_event_loads_legacy_protocol_repair_without_failure_kind() -> None:
    loaded = load_event(
        '{"type":"protocol_repair_requested","run_id":"run-1","lane_id":"lane-1",'
        '"sequence":6,"turn_number":4,"repair_attempt":1,'
        '"reason":"Model protocol incomplete","hit_generation_limit":true,'
        '"message":"Call exactly one tool now."}'
    )

    assert loaded.type == "protocol_repair_requested"
    assert loaded.failure_kind is None


def test_event_round_trip_run_failed_with_protocol_kind() -> None:
    event = RunFailedEvent(
        run_id="run-1",
        lane_id="lane-1",
        sequence=7,
        error="Model protocol incomplete",
        failure_kind="completed_without_action",
    )

    assert load_event(dump_event(event)) == event
