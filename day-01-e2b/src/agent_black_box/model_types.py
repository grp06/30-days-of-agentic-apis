from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from pydantic import BaseModel


class ModelDecision(BaseModel):
    finish_reason: str
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    message: str | None = None
    assistant_message: dict[str, Any] | None = None
    provider_eval_count: int | None = None
    provider_num_predict: int | None = None
    provider_done_reason: str | None = None
    provider_content_chars: int | None = None
    provider_tool_call_count: int | None = None
    hit_generation_limit: bool = False


class OllamaTiming(BaseModel):
    total_duration_seconds: float | None = None
    load_duration_seconds: float | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration_seconds: float | None = None
    eval_count: int | None = None
    eval_duration_seconds: float | None = None


class OllamaResponseMetadata(BaseModel):
    done_reason: str | None = None
    content_chars: int | None = None
    tool_call_count: int | None = None


def parse_chat_response(payload: dict[str, Any]) -> ModelDecision:
    message = _message_dict(payload.get("message"))
    content = _optional_str(message.get("content"))
    tool_calls = _tool_calls(message.get("tool_calls"))
    if tool_calls:
        call = tool_calls[0]
        function = call.get("function", {})
        if not isinstance(function, dict):
            function = {}
        return ModelDecision(
            finish_reason="tool_call",
            tool_name=_optional_str(function.get("name")),
            tool_arguments=_tool_arguments(function.get("arguments")),
            message=content or None,
            assistant_message=message,
        )
    return ModelDecision(
        finish_reason="completed",
        message=content or "",
        assistant_message=message,
    )


def extract_ollama_timing(payload: dict[str, Any]) -> OllamaTiming:
    return OllamaTiming(
        total_duration_seconds=_nanoseconds_to_seconds(payload.get("total_duration")),
        load_duration_seconds=_nanoseconds_to_seconds(payload.get("load_duration")),
        prompt_eval_count=_optional_int(payload.get("prompt_eval_count")),
        prompt_eval_duration_seconds=_nanoseconds_to_seconds(
            payload.get("prompt_eval_duration")
        ),
        eval_count=_optional_int(payload.get("eval_count")),
        eval_duration_seconds=_nanoseconds_to_seconds(payload.get("eval_duration")),
    )


def extract_ollama_response_metadata(payload: dict[str, Any]) -> OllamaResponseMetadata:
    message = _message_dict(payload.get("message"))
    content = message.get("content")
    tool_calls = _tool_calls(message.get("tool_calls"))
    return OllamaResponseMetadata(
        done_reason=_optional_str(payload.get("done_reason") or payload.get("done")),
        content_chars=len(content) if isinstance(content, str) else None,
        tool_call_count=len(tool_calls),
    )


def hit_generation_limit(*, eval_count: int | None, num_predict: object) -> bool:
    limit = _optional_int(num_predict)
    return limit is not None and eval_count is not None and eval_count >= limit


def _nanoseconds_to_seconds(value: object) -> float | None:
    if not isinstance(value, int | float):
        return None
    return float(value) / 1_000_000_000


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _message_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _tool_calls(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _tool_arguments(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
