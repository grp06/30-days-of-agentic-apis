from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


class BaseEvent(BaseModel):
    type: str
    run_id: str
    lane_id: str
    sequence: int
    timestamp: datetime = Field(default_factory=utc_now)


class RunStartedEvent(BaseEvent):
    type: Literal["run_started"] = "run_started"
    task: str
    fixture_name: str
    model: str


class ModelTurnStartedEvent(BaseEvent):
    type: Literal["model_turn_started"] = "model_turn_started"
    turn_number: int
    model_name: str | None = None
    message_count: int | None = None
    tool_schema_count: int | None = None
    conversation_chars: int | None = None
    last_message_chars: int | None = None
    last_tool_result_chars: int | None = None
    request_body_bytes: int | None = None


class ModelTurnCompletedEvent(BaseEvent):
    type: Literal["model_turn_completed"] = "model_turn_completed"
    turn_number: int
    finish_reason: str
    content: str | None = None
    tool_name: str | None = None


class ProtocolRepairRequestedEvent(BaseEvent):
    type: Literal["protocol_repair_requested"] = "protocol_repair_requested"
    turn_number: int
    repair_attempt: int
    reason: str
    failure_kind: str | None = None
    hit_generation_limit: bool = False
    message: str


class ModelProviderAttemptStartedEvent(BaseEvent):
    type: Literal["model_provider_attempt_started"] = "model_provider_attempt_started"
    turn_number: int
    phase: Literal["list_models", "chat"]
    model_name: str | None = None
    attempt_number: int
    timeout_seconds: float
    is_fallback: bool = False


class ModelProviderAttemptCompletedEvent(BaseEvent):
    type: Literal["model_provider_attempt_completed"] = (
        "model_provider_attempt_completed"
    )
    turn_number: int
    phase: Literal["list_models", "chat"]
    model_name: str | None = None
    attempt_number: int
    timeout_seconds: float
    is_fallback: bool = False
    duration_seconds: float
    outcome: Literal[
        "success",
        "timeout",
        "http_error",
        "transport_error",
        "error",
    ]
    status_code: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    ollama_total_duration_seconds: float | None = None
    ollama_load_duration_seconds: float | None = None
    ollama_prompt_eval_count: int | None = None
    ollama_prompt_eval_duration_seconds: float | None = None
    ollama_eval_count: int | None = None
    ollama_eval_duration_seconds: float | None = None
    response_done_reason: str | None = None
    response_content_chars: int | None = None
    response_tool_call_count: int | None = None
    hit_generation_limit: bool = False


class ToolCallEvent(BaseEvent):
    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: dict[str, Any]


class ToolResultEvent(BaseEvent):
    type: Literal["tool_result"] = "tool_result"
    tool_name: str
    ok: bool
    result: dict[str, Any]


class CommandStartedEvent(BaseEvent):
    type: Literal["command_started"] = "command_started"
    command: str
    cwd: str
    background: bool


class CommandStreamEvent(BaseEvent):
    type: Literal["command_stream"] = "command_stream"
    command: str
    stream: Literal["stdout", "stderr"]
    chunk: str


class CommandCompletedEvent(BaseEvent):
    type: Literal["command_completed"] = "command_completed"
    command: str
    exit_code: int
    stdout: str
    stderr: str
    background: bool
    pid: int | None = None


class FileDiffEvent(BaseEvent):
    type: Literal["file_diff"] = "file_diff"
    patch_path: str
    patch_summary: str


class PreviewPublishedEvent(BaseEvent):
    type: Literal["preview_published"] = "preview_published"
    url: str
    port: int


class CheckpointCreatedEvent(BaseEvent):
    type: Literal["checkpoint_created"] = "checkpoint_created"
    snapshot_id: str
    note: str


class JudgeNoteEvent(BaseEvent):
    type: Literal["judge_note"] = "judge_note"
    note: str


class RunCompletedEvent(BaseEvent):
    type: Literal["run_completed"] = "run_completed"
    summary: str


class RunFailedEvent(BaseEvent):
    type: Literal["run_failed"] = "run_failed"
    error: str
    failure_kind: str | None = None


Event = Annotated[
    RunStartedEvent
    | ModelTurnStartedEvent
    | ModelTurnCompletedEvent
    | ProtocolRepairRequestedEvent
    | ModelProviderAttemptStartedEvent
    | ModelProviderAttemptCompletedEvent
    | ToolCallEvent
    | ToolResultEvent
    | CommandStartedEvent
    | CommandStreamEvent
    | CommandCompletedEvent
    | FileDiffEvent
    | PreviewPublishedEvent
    | CheckpointCreatedEvent
    | JudgeNoteEvent
    | RunCompletedEvent
    | RunFailedEvent,
    Field(discriminator="type"),
]

EVENT_ADAPTER = TypeAdapter(Event)


def dump_event(event: Event) -> str:
    return event.model_dump_json()


def load_event(raw: str) -> Event:
    return EVENT_ADAPTER.validate_json(raw)
