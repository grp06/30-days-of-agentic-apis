from __future__ import annotations

from typing import Any, Callable, Literal, Protocol
import time

import httpx

from .config import Settings
from .events import (
    Event,
    ModelProviderAttemptCompletedEvent,
    ModelProviderAttemptStartedEvent,
)
from .model_types import (
    ModelDecision,
    OllamaResponseMetadata,
    OllamaTiming,
    extract_ollama_response_metadata,
    extract_ollama_timing,
    hit_generation_limit,
    parse_chat_response,
)


class ProviderTraceContext(Protocol):
    run_id: str
    lane_id: str
    turn_number: int
    next_sequence: Callable[[], int]
    append_event: Callable[[Event], None]


class OllamaChatAttemptRunner:
    def __init__(
        self,
        *,
        settings: Settings,
        headers: Callable[[], dict[str, str]],
        trace_context: Callable[[], ProviderTraceContext | None],
    ) -> None:
        self.settings = settings
        self._headers = headers
        self._trace_context = trace_context

    async def run_chat_attempt(
        self,
        *,
        model: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        attempt_number: int,
        is_fallback: bool,
    ) -> ModelDecision:
        body = {
            "model": model,
            "messages": conversation,
            "tools": tools,
            "stream": False,
        }
        think = self._request_think()
        if think is not None:
            body["think"] = think
        if self.settings.ollama_keep_alive:
            body["keep_alive"] = self.settings.ollama_keep_alive
        options = self._request_options()
        if options:
            body["options"] = options

        timeout_seconds = self.settings.ollama_chat_timeout_seconds
        self._emit_provider_attempt_started(
            model_name=model,
            attempt_number=attempt_number,
            timeout_seconds=timeout_seconds,
            is_fallback=is_fallback,
        )
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=timeout_seconds,
            ) as client:
                response = await client.post(
                    "/chat",
                    headers=self._headers(),
                    json=body,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError(
                        "Ollama /chat response must be a JSON object; "
                        f"got {type(payload).__name__}"
                    )
        except Exception as exc:  # noqa: BLE001
            self._emit_provider_attempt_completed(
                model_name=model,
                attempt_number=attempt_number,
                timeout_seconds=timeout_seconds,
                is_fallback=is_fallback,
                started=started,
                exc=exc,
            )
            raise

        timing = extract_ollama_timing(payload)
        response_metadata = extract_ollama_response_metadata(payload)
        reached_generation_limit = hit_generation_limit(
            eval_count=timing.eval_count,
            num_predict=options.get("num_predict"),
        )
        self._emit_provider_attempt_completed(
            model_name=model,
            attempt_number=attempt_number,
            timeout_seconds=timeout_seconds,
            is_fallback=is_fallback,
            started=started,
            exc=None,
            timing=timing,
            response_metadata=response_metadata,
            hit_generation_limit=reached_generation_limit,
        )

        decision = parse_chat_response(payload)
        decision.provider_eval_count = timing.eval_count
        num_predict = options.get("num_predict")
        decision.provider_num_predict = (
            num_predict if isinstance(num_predict, int) else None
        )
        decision.provider_done_reason = response_metadata.done_reason
        decision.provider_content_chars = response_metadata.content_chars
        decision.provider_tool_call_count = response_metadata.tool_call_count
        decision.hit_generation_limit = reached_generation_limit
        return decision

    def _request_options(self) -> dict[str, int | float]:
        options: dict[str, int | float] = {}
        if self.settings.ollama_num_predict is not None:
            options["num_predict"] = self.settings.ollama_num_predict
        if self.settings.ollama_num_ctx is not None:
            options["num_ctx"] = self.settings.ollama_num_ctx
        if self.settings.ollama_temperature is not None:
            options["temperature"] = self.settings.ollama_temperature
        return options

    def _request_think(self) -> bool | str | None:
        value = self.settings.ollama_think
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        return normalized

    def _emit_provider_attempt_started(
        self,
        *,
        model_name: str,
        attempt_number: int,
        timeout_seconds: float,
        is_fallback: bool,
    ) -> None:
        context = self._trace_context()
        if context is None:
            return
        context.append_event(
            ModelProviderAttemptStartedEvent(
                run_id=context.run_id,
                lane_id=context.lane_id,
                sequence=context.next_sequence(),
                turn_number=context.turn_number,
                phase="chat",
                model_name=model_name,
                attempt_number=attempt_number,
                timeout_seconds=timeout_seconds,
                is_fallback=is_fallback,
            )
        )

    def _emit_provider_attempt_completed(
        self,
        *,
        model_name: str,
        attempt_number: int,
        timeout_seconds: float,
        is_fallback: bool,
        started: float,
        exc: Exception | None,
        timing: OllamaTiming | None = None,
        response_metadata: OllamaResponseMetadata | None = None,
        hit_generation_limit: bool = False,
    ) -> None:
        context = self._trace_context()
        if context is None:
            return
        outcome, status_code, error_type, error_message = _classify_attempt_error(exc)
        context.append_event(
            ModelProviderAttemptCompletedEvent(
                run_id=context.run_id,
                lane_id=context.lane_id,
                sequence=context.next_sequence(),
                turn_number=context.turn_number,
                phase="chat",
                model_name=model_name,
                attempt_number=attempt_number,
                timeout_seconds=timeout_seconds,
                is_fallback=is_fallback,
                duration_seconds=time.monotonic() - started,
                outcome=outcome,
                status_code=status_code,
                error_type=error_type,
                error_message=error_message,
                ollama_total_duration_seconds=(
                    timing.total_duration_seconds if timing is not None else None
                ),
                ollama_load_duration_seconds=(
                    timing.load_duration_seconds if timing is not None else None
                ),
                ollama_prompt_eval_count=(
                    timing.prompt_eval_count if timing is not None else None
                ),
                ollama_prompt_eval_duration_seconds=(
                    timing.prompt_eval_duration_seconds if timing is not None else None
                ),
                ollama_eval_count=timing.eval_count if timing is not None else None,
                ollama_eval_duration_seconds=(
                    timing.eval_duration_seconds if timing is not None else None
                ),
                response_done_reason=(
                    response_metadata.done_reason
                    if response_metadata is not None
                    else None
                ),
                response_content_chars=(
                    response_metadata.content_chars
                    if response_metadata is not None
                    else None
                ),
                response_tool_call_count=(
                    response_metadata.tool_call_count
                    if response_metadata is not None
                    else None
                ),
                hit_generation_limit=hit_generation_limit,
            )
        )


def _classify_attempt_error(
    exc: Exception | None,
) -> tuple[
    Literal["success", "timeout", "http_error", "transport_error", "error"],
    int | None,
    str | None,
    str | None,
]:
    if exc is None:
        return "success", None, None, None
    if isinstance(exc, httpx.TimeoutException):
        return "timeout", None, type(exc).__name__, str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        return (
            "http_error",
            exc.response.status_code,
            type(exc).__name__,
            str(exc),
        )
    if isinstance(exc, httpx.TransportError):
        return "transport_error", None, type(exc).__name__, str(exc)
    return "error", None, type(exc).__name__, str(exc)
