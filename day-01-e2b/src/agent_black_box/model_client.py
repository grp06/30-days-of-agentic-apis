from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any, Callable, ClassVar, Literal, Protocol

import httpx

from .config import Settings
from .events import (
    Event,
    ModelProviderAttemptCompletedEvent,
    ModelProviderAttemptStartedEvent,
)
from .provider_attempts import OllamaChatAttemptRunner
from .model_types import (
    ModelDecision as ModelDecision,
    OllamaResponseMetadata as OllamaResponseMetadata,
    OllamaTiming as OllamaTiming,
    parse_chat_response as parse_chat_response,
)


class ModelClient(Protocol):
    def reset_run_state(self) -> None: ...

    def active_model_name(self) -> str: ...

    async def next_action(
        self,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelDecision: ...

    async def list_models(self) -> list[str]: ...


@dataclass(frozen=True)
class ModelTraceContext:
    run_id: str
    lane_id: str
    turn_number: int
    next_sequence: Callable[[], int]
    append_event: Callable[[Event], None]


@dataclass
class OllamaModelClient:
    _model_timeout_events: ClassVar[dict[str, list[float]]] = {}
    _global_demoted_until: ClassVar[dict[str, float]] = {}

    settings: Settings
    model_name: str | None = None
    fallback_enabled: bool = True
    _pinned_model: str | None = None
    _trace_context: ModelTraceContext | None = None
    _models_cache: list[str] | None = None
    _models_cache_expires_at: float = 0.0

    def reset_run_state(self) -> None:
        self._pinned_model = None
        self._trace_context = None

    def active_model_name(self) -> str:
        primary = self._primary_model_name()
        fallback = self._fallback_model_name()
        if (
            self._pinned_model is None
            and fallback is not None
            and self._is_globally_demoted(primary)
        ):
            return fallback
        return self._pinned_model or primary

    def set_trace_context(self, context: ModelTraceContext | None) -> None:
        self._trace_context = context

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.ollama_api_key}",
            "Content-Type": "application/json",
        }

    async def list_models(self) -> list[str]:
        now = time.monotonic()
        if self._models_cache is not None and now < self._models_cache_expires_at:
            return list(self._models_cache)

        timeout_seconds = self.settings.ollama_tags_timeout_seconds
        self._emit_provider_attempt_started(
            phase="list_models",
            model_name=None,
            attempt_number=1,
            timeout_seconds=timeout_seconds,
            is_fallback=False,
        )
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=timeout_seconds,
            ) as client:
                response = await client.get("/tags", headers=self._headers())
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:  # noqa: BLE001
            self._emit_provider_attempt_completed(
                phase="list_models",
                model_name=None,
                attempt_number=1,
                timeout_seconds=timeout_seconds,
                is_fallback=False,
                started=started,
                exc=exc,
            )
            raise
        self._emit_provider_attempt_completed(
            phase="list_models",
            model_name=None,
            attempt_number=1,
            timeout_seconds=timeout_seconds,
            is_fallback=False,
            started=started,
            exc=None,
        )
        models = _model_names_from_tags_payload(payload)
        cache_seconds = max(0.0, self.settings.ollama_tags_cache_seconds)
        self._models_cache = list(models)
        self._models_cache_expires_at = time.monotonic() + cache_seconds
        return models

    def has_model(self, requested: str, available: list[str]) -> bool:
        return self._resolve_model_name(requested, available) is not None

    def resolve_model_name(self, requested: str, available: list[str]) -> str | None:
        return self._resolve_model_name(requested, available)

    async def next_action(
        self,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelDecision:
        models = await self.list_models()
        primary = self._require_resolved_model(self.active_model_name(), models)
        fallback_name = self._fallback_model_name()
        fallback = (
            self._resolve_model_name(fallback_name, models)
            if fallback_name is not None
            else None
        )

        try:
            decision = await self._chat_with_model_with_retry(
                primary,
                conversation,
                tools,
                is_fallback=False,
            )
            self._pinned_model = primary if self._pinned_model else None
            return decision
        except Exception as primary_exc:  # noqa: BLE001
            if self._is_timeout_error(primary_exc):
                self._record_model_timeout(primary)
            if (
                fallback is None
                or fallback == primary
                or not self._is_transient_error(primary_exc)
            ):
                raise
            try:
                decision = await self._chat_with_model_with_retry(
                    fallback,
                    conversation,
                    tools,
                    is_fallback=True,
                )
            except Exception as fallback_exc:  # noqa: BLE001
                raise ProviderInterruptionError(
                    primary_model=primary,
                    fallback_model=fallback,
                    last_error=fallback_exc,
                ) from fallback_exc
            self._pinned_model = fallback
            return decision

    def _require_resolved_model(self, requested: str, available: list[str]) -> str:
        resolved_model = self._resolve_model_name(requested, available)
        if resolved_model is None:
            raise RuntimeError(
                f"Configured Ollama model {requested!r} was not found in /tags"
            )
        return resolved_model

    def _primary_model_name(self) -> str:
        return self.model_name or self.settings.ollama_model

    def _fallback_model_name(self) -> str | None:
        if not self.fallback_enabled:
            return None
        return self.settings.ollama_fallback_model

    async def _chat_with_model_with_retry(
        self,
        model: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        is_fallback: bool,
    ) -> ModelDecision:
        attempts = max(1, self.settings.ollama_max_attempts_per_model)
        delay = max(0.0, self.settings.ollama_retry_base_delay_seconds)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return await self._chat_with_model(
                    model,
                    conversation,
                    tools,
                    attempt_number=attempt,
                    is_fallback=is_fallback,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= attempts or not self._is_transient_error(exc):
                    raise
                await asyncio.sleep(delay * (2 ** (attempt - 1)))
        assert last_exc is not None
        raise last_exc

    async def _chat_with_model(
        self,
        model: str,
        conversation: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        attempt_number: int,
        is_fallback: bool,
    ) -> ModelDecision:
        return await self._chat_runner().run_chat_attempt(
            model=model,
            conversation=conversation,
            tools=tools,
            attempt_number=attempt_number,
            is_fallback=is_fallback,
        )

    def _chat_runner(self) -> OllamaChatAttemptRunner:
        return OllamaChatAttemptRunner(
            settings=self.settings,
            headers=self._headers,
            trace_context=lambda: self._trace_context,
        )

    def _resolve_model_name(self, requested: str, available: list[str]) -> str | None:
        if requested in available:
            return requested
        if requested.endswith(":cloud"):
            without_suffix = requested.removesuffix(":cloud")
            if without_suffix in available:
                return without_suffix
        cloud_variant = f"{requested}:cloud"
        if cloud_variant in available:
            return cloud_variant
        return None

    def _is_transient_error(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {429, 500, 502, 503, 504}
        return isinstance(exc, httpx.TransportError)

    def _is_timeout_error(self, exc: Exception) -> bool:
        return isinstance(exc, httpx.TimeoutException)

    def _is_globally_demoted(self, model: str) -> bool:
        now = time.monotonic()
        names = {model}
        if model.endswith(":cloud"):
            names.add(model.removesuffix(":cloud"))
        else:
            names.add(f"{model}:cloud")
        return any(self._global_demoted_until.get(name, 0.0) > now for name in names)

    def _record_model_timeout(self, model: str) -> None:
        now = time.monotonic()
        window = max(0.0, self.settings.ollama_timeout_demotion_window_seconds)
        recent = [
            timestamp
            for timestamp in self._model_timeout_events.get(model, [])
            if now - timestamp <= window
        ]
        recent.append(now)
        self._model_timeout_events[model] = recent
        threshold = max(1, self.settings.ollama_timeout_demotion_threshold)
        if len(recent) >= threshold:
            self._global_demoted_until[model] = (
                now + self.settings.ollama_timeout_demotion_seconds
            )

    def _emit_provider_attempt_started(
        self,
        *,
        phase: Literal["list_models", "chat"],
        model_name: str | None,
        attempt_number: int,
        timeout_seconds: float,
        is_fallback: bool,
    ) -> None:
        context = self._trace_context
        if context is None:
            return
        context.append_event(
            ModelProviderAttemptStartedEvent(
                run_id=context.run_id,
                lane_id=context.lane_id,
                sequence=context.next_sequence(),
                turn_number=context.turn_number,
                phase=phase,
                model_name=model_name,
                attempt_number=attempt_number,
                timeout_seconds=timeout_seconds,
                is_fallback=is_fallback,
            )
        )

    def _emit_provider_attempt_completed(
        self,
        *,
        phase: Literal["list_models", "chat"],
        model_name: str | None,
        attempt_number: int,
        timeout_seconds: float,
        is_fallback: bool,
        started: float,
        exc: Exception | None,
    ) -> None:
        context = self._trace_context
        if context is None:
            return
        outcome, status_code, error_type, error_message = self._classify_attempt_error(
            exc
        )
        context.append_event(
            ModelProviderAttemptCompletedEvent(
                run_id=context.run_id,
                lane_id=context.lane_id,
                sequence=context.next_sequence(),
                turn_number=context.turn_number,
                phase=phase,
                model_name=model_name,
                attempt_number=attempt_number,
                timeout_seconds=timeout_seconds,
                is_fallback=is_fallback,
                duration_seconds=time.monotonic() - started,
                outcome=outcome,
                status_code=status_code,
                error_type=error_type,
                error_message=error_message,
            )
        )

    def _classify_attempt_error(
        self,
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


class ProviderInterruptionError(RuntimeError):
    def __init__(
        self, primary_model: str, fallback_model: str, last_error: Exception
    ) -> None:
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.last_error = last_error
        super().__init__(
            "Provider interruption after exhausting transient retries for "
            f"{primary_model} and fallback {fallback_model}: {last_error}"
        )


def _model_names_from_tags_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    models = payload.get("models", [])
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names
