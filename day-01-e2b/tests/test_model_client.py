from __future__ import annotations

import httpx
import pytest

from agent_black_box.config import Settings
from agent_black_box import model_client as model_client_module
from agent_black_box.model_client import (
    ModelTraceContext,
    OllamaModelClient,
    ProviderInterruptionError,
    parse_chat_response,
)
from agent_black_box.model_types import extract_ollama_response_metadata


def test_parse_chat_response_tool_call() -> None:
    payload = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "run_command",
                        "arguments": {"command": "pwd", "timeout_seconds": 30},
                    }
                }
            ],
        }
    }

    decision = parse_chat_response(payload)
    assert decision.finish_reason == "tool_call"
    assert decision.tool_name == "run_command"
    assert decision.tool_arguments == {"command": "pwd", "timeout_seconds": 30}


def test_parse_chat_response_accepts_json_string_tool_arguments() -> None:
    payload = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "run_command",
                        "arguments": '{"command":"pwd","timeout_seconds":30}',
                    }
                }
            ],
        }
    }

    decision = parse_chat_response(payload)

    assert decision.finish_reason == "tool_call"
    assert decision.tool_name == "run_command"
    assert decision.tool_arguments == {"command": "pwd", "timeout_seconds": 30}


def test_parse_chat_response_completed() -> None:
    payload = {"message": {"role": "assistant", "content": "done"}}
    decision = parse_chat_response(payload)
    assert decision.finish_reason == "completed"
    assert decision.message == "done"


def test_parse_chat_response_ignores_malformed_tool_calls() -> None:
    payload = {
        "message": {
            "role": "assistant",
            "content": "done",
            "tool_calls": {"function": {"name": "finish_run"}},
        }
    }

    decision = parse_chat_response(payload)

    assert decision.finish_reason == "completed"
    assert decision.message == "done"


def test_parse_chat_response_ignores_malformed_tool_call_items() -> None:
    payload = {
        "message": {
            "role": "assistant",
            "content": "done",
            "tool_calls": [
                "not-an-object",
                {
                    "function": {
                        "name": "finish_run",
                        "arguments": {"summary": "done"},
                    }
                },
            ],
        }
    }

    decision = parse_chat_response(payload)

    assert decision.finish_reason == "tool_call"
    assert decision.tool_name == "finish_run"
    assert decision.tool_arguments == {"summary": "done"}


def test_parse_chat_response_tolerates_non_string_content() -> None:
    payload = {"message": {"role": "assistant", "content": ["done"]}}

    decision = parse_chat_response(payload)

    assert decision.finish_reason == "completed"
    assert decision.message == ""


def test_parse_chat_response_tolerates_non_string_tool_name() -> None:
    payload = {
        "message": {
            "role": "assistant",
            "content": "done",
            "tool_calls": [{"function": {"name": 123, "arguments": {}}}],
        }
    }

    decision = parse_chat_response(payload)

    assert decision.finish_reason == "tool_call"
    assert decision.tool_name is None
    assert decision.message == "done"


def test_extract_ollama_response_metadata_counts_only_valid_tool_calls() -> None:
    metadata = extract_ollama_response_metadata(
        {
            "done_reason": "stop",
            "message": {
                "content": "done",
                "tool_calls": [
                    "not-an-object",
                    {"function": {"name": "finish_run"}},
                ],
            },
        }
    )

    assert metadata.done_reason == "stop"
    assert metadata.content_chars == 4
    assert metadata.tool_call_count == 1


@pytest.mark.asyncio
async def test_list_models_ignores_malformed_tag_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")
    client = OllamaModelClient(settings)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "models": [
                    {"name": "kimi-k2.6"},
                    {"digest": "missing-name"},
                    "not-an-object",
                    {"name": ""},
                    {"name": "glm-5.1"},
                ],
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str]) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    assert await client.list_models() == ["kimi-k2.6", "glm-5.1"]


def test_model_resolution_accepts_direct_cloud_alias() -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")
    client = OllamaModelClient(settings)
    assert client.has_model("kimi-k2.6:cloud", ["kimi-k2.6"])


def test_model_resolution_accepts_qwen_without_cloud_suffix() -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")
    client = OllamaModelClient(settings)

    assert client.resolve_model_name("qwen3.5:397b", ["qwen3.5:397b"]) == "qwen3.5:397b"


def test_default_latency_settings_fail_fast_to_stable_model() -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")

    assert settings.ollama_model == "minimax-m2.7:cloud"
    assert settings.ollama_fallback_model == "glm-5.1:cloud"
    assert settings.ollama_max_attempts_per_model == 2
    assert settings.ollama_chat_timeout_seconds == 300.0
    assert settings.ollama_keep_alive == "10m"
    assert settings.ollama_think == "false"
    assert settings.ollama_num_predict is None
    assert settings.model_protocol_repair_attempts == 2
    assert settings.ollama_temperature == 0.0
    assert settings.ollama_arena_models == [
        "kimi-k2.6:cloud",
        "glm-5.1:cloud",
        "gemma4:31b",
        "qwen3.5:397b",
    ]


def test_settings_accepts_comma_separated_arena_models_from_dotenv(tmp_path) -> None:  # noqa: ANN001
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "E2B_API_KEY=e2b_test",
                "OLLAMA_API_KEY=ollama_test",
                "OLLAMA_ARENA_MODELS=kimi-k2.6:cloud,glm-5.1:cloud,gemma4:31b,qwen3.5:397b",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.ollama_arena_models == [
        "kimi-k2.6:cloud",
        "glm-5.1:cloud",
        "gemma4:31b",
        "qwen3.5:397b",
    ]


@pytest.mark.asyncio
async def test_next_action_falls_back_to_glm_on_primary_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_MODEL="kimi-k2.6:cloud",
        OLLAMA_FALLBACK_MODEL="glm-5.1:cloud",
        OLLAMA_MAX_ATTEMPTS_PER_MODEL=1,
    )
    client = OllamaModelClient(settings)
    posted_models: list[str] = []

    async def fake_list_models() -> list[str]:
        return ["kimi-k2.6", "glm-5.1"]

    class FakeResponse:
        def __init__(self, model: str) -> None:
            self.model = model

        def raise_for_status(self) -> None:
            if self.model == "kimi-k2.6":
                request = httpx.Request("POST", "https://ollama.com/api/chat")
                response = httpx.Response(503, request=request)
                raise httpx.HTTPStatusError(
                    "503 Service Unavailable", request=request, response=response
                )

        def json(self) -> dict:
            return {"message": {"role": "assistant", "content": "done"}}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self, url: str, headers: dict[str, str], json: dict
        ) -> FakeResponse:
            posted_models.append(json["model"])
            return FakeResponse(json["model"])

    monkeypatch.setattr(client, "list_models", fake_list_models)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    decision = await client.next_action(
        conversation=[{"role": "user", "content": "hi"}], tools=[]
    )

    assert decision.finish_reason == "completed"
    assert posted_models == ["kimi-k2.6", "glm-5.1"]
    assert client.active_model_name() == "glm-5.1"


@pytest.mark.asyncio
async def test_explicit_model_can_disable_fallback_for_model_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_MODEL="minimax-m2.7:cloud",
        OLLAMA_FALLBACK_MODEL="glm-5.1:cloud",
        OLLAMA_MAX_ATTEMPTS_PER_MODEL=1,
    )
    client = OllamaModelClient(
        settings,
        model_name="kimi-k2.6:cloud",
        fallback_enabled=False,
    )
    posted_models: list[str] = []

    async def fake_list_models() -> list[str]:
        return ["kimi-k2.6", "glm-5.1", "minimax-m2.7"]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, headers: dict[str, str], json: dict) -> None:
            posted_models.append(json["model"])
            request = httpx.Request("POST", "https://ollama.com/api/chat")
            raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setattr(client, "list_models", fake_list_models)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(httpx.ReadTimeout):
        await client.next_action(
            conversation=[{"role": "user", "content": "hi"}], tools=[]
        )

    assert posted_models == ["kimi-k2.6"]
    assert client.active_model_name() == "kimi-k2.6:cloud"


@pytest.mark.asyncio
async def test_next_action_retries_transient_error_before_succeeding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_MODEL="kimi-k2.6:cloud",
        OLLAMA_FALLBACK_MODEL="glm-5.1:cloud",
        OLLAMA_MAX_ATTEMPTS_PER_MODEL=3,
        OLLAMA_RETRY_BASE_DELAY_SECONDS=0,
    )
    client = OllamaModelClient(settings)
    call_count = 0

    async def fake_list_models() -> list[str]:
        return ["kimi-k2.6", "glm-5.1"]

    class FakeResponse:
        def raise_for_status(self) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                request = httpx.Request("POST", "https://ollama.com/api/chat")
                response = httpx.Response(503, request=request)
                raise httpx.HTTPStatusError(
                    "503 Service Unavailable", request=request, response=response
                )

        def json(self) -> dict:
            return {"message": {"role": "assistant", "content": "done"}}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self, url: str, headers: dict[str, str], json: dict
        ) -> FakeResponse:
            return FakeResponse()

    async def fake_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr(client, "list_models", fake_list_models)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(model_client_module.asyncio, "sleep", fake_sleep)

    decision = await client.next_action(
        conversation=[{"role": "user", "content": "hi"}], tools=[]
    )

    assert decision.finish_reason == "completed"
    assert call_count == 3
    assert client.active_model_name() == "kimi-k2.6:cloud"


@pytest.mark.asyncio
async def test_next_action_emits_provider_attempt_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_MODEL="kimi-k2.6:cloud",
        OLLAMA_FALLBACK_MODEL="glm-5.1:cloud",
        OLLAMA_MAX_ATTEMPTS_PER_MODEL=2,
        OLLAMA_RETRY_BASE_DELAY_SECONDS=0,
        OLLAMA_TAGS_TIMEOUT_SECONDS=7.0,
        OLLAMA_CHAT_TIMEOUT_SECONDS=11.0,
    )
    client = OllamaModelClient(settings)
    events = []
    sequence = 0
    post_count = 0
    posted_bodies: list[dict] = []

    def next_sequence() -> int:
        nonlocal sequence
        sequence += 1
        return sequence

    class FakeResponse:
        def __init__(self, status_code: int, payload: dict) -> None:
            request = httpx.Request("GET", "https://ollama.com/api/tags")
            self.response = httpx.Response(status_code, request=request)
            self.status_code = status_code
            self.payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(
                    "503 Service Unavailable",
                    request=self.response.request,
                    response=self.response,
                )

        def json(self) -> dict:
            return self.payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str]) -> FakeResponse:
            return FakeResponse(
                200,
                {"models": [{"name": "kimi-k2.6"}, {"name": "glm-5.1"}]},
            )

        async def post(
            self, url: str, headers: dict[str, str], json: dict
        ) -> FakeResponse:
            nonlocal post_count
            post_count += 1
            posted_bodies.append(json)
            if post_count == 1:
                return FakeResponse(503, {})
            return FakeResponse(
                200,
                {
                    "message": {"role": "assistant", "content": "done"},
                    "total_duration": 1_250_000_000,
                    "load_duration": 50_000_000,
                    "prompt_eval_count": 42,
                    "prompt_eval_duration": 200_000_000,
                    "eval_count": 12,
                    "eval_duration": 900_000_000,
                },
            )

    async def fake_sleep(delay: float) -> None:
        return None

    client.set_trace_context(
        ModelTraceContext(
            run_id="run-1",
            lane_id="lane-1",
            turn_number=4,
            next_sequence=next_sequence,
            append_event=events.append,
        )
    )
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(model_client_module.asyncio, "sleep", fake_sleep)

    decision = await client.next_action(
        conversation=[{"role": "user", "content": "hi"}],
        tools=[],
    )

    assert decision.finish_reason == "completed"
    assert [event.type for event in events] == [
        "model_provider_attempt_started",
        "model_provider_attempt_completed",
        "model_provider_attempt_started",
        "model_provider_attempt_completed",
        "model_provider_attempt_started",
        "model_provider_attempt_completed",
    ]
    assert events[0].phase == "list_models"
    assert events[0].timeout_seconds == 7.0
    assert events[1].outcome == "success"
    assert events[2].phase == "chat"
    assert events[2].model_name == "kimi-k2.6"
    assert events[2].attempt_number == 1
    assert events[2].timeout_seconds == 11.0
    assert events[3].outcome == "http_error"
    assert events[3].status_code == 503
    assert events[4].attempt_number == 2
    assert events[5].outcome == "success"
    assert events[5].ollama_total_duration_seconds == 1.25
    assert events[5].ollama_load_duration_seconds == 0.05
    assert events[5].ollama_prompt_eval_count == 42
    assert events[5].ollama_prompt_eval_duration_seconds == 0.2
    assert events[5].ollama_eval_count == 12
    assert events[5].ollama_eval_duration_seconds == 0.9
    assert events[5].response_content_chars == 4
    assert events[5].response_tool_call_count == 0
    assert events[5].hit_generation_limit is False
    assert decision.provider_eval_count == 12
    assert decision.provider_num_predict is None
    assert decision.provider_content_chars == 4
    assert decision.provider_tool_call_count == 0
    assert decision.hit_generation_limit is False
    assert posted_bodies[0]["think"] is False
    assert posted_bodies[0]["keep_alive"] == "10m"
    assert posted_bodies[0]["options"] == {"temperature": 0.0}


@pytest.mark.asyncio
async def test_next_action_marks_generation_limit_and_response_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_MODEL="kimi-k2.6:cloud",
        OLLAMA_NUM_PREDICT=2048,
    )
    client = OllamaModelClient(settings)
    events = []
    sequence = 0

    def next_sequence() -> int:
        nonlocal sequence
        sequence += 1
        return sequence

    async def fake_list_models() -> list[str]:
        return ["kimi-k2.6"]

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "message": {"role": "assistant", "content": ""},
                "done_reason": "length",
                "eval_count": 2048,
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, headers: dict[str, str], json: dict) -> FakeResponse:
            return FakeResponse()

    client.set_trace_context(
        ModelTraceContext(
            run_id="run-1",
            lane_id="lane-1",
            turn_number=4,
            next_sequence=next_sequence,
            append_event=events.append,
        )
    )
    monkeypatch.setattr(client, "list_models", fake_list_models)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    decision = await client.next_action(
        conversation=[{"role": "user", "content": "hi"}],
        tools=[],
    )

    assert decision.hit_generation_limit is True
    assert decision.provider_eval_count == 2048
    assert decision.provider_num_predict == 2048
    assert decision.provider_done_reason == "length"
    assert decision.provider_content_chars == 0
    assert decision.provider_tool_call_count == 0
    assert events[-1].hit_generation_limit is True
    assert events[-1].response_done_reason == "length"
    assert events[-1].response_content_chars == 0
    assert events[-1].response_tool_call_count == 0


@pytest.mark.asyncio
async def test_next_action_uses_cached_model_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_MODEL="kimi-k2.6:cloud",
        OLLAMA_FALLBACK_MODEL="glm-5.1:cloud",
        OLLAMA_TAGS_CACHE_SECONDS=600,
    )
    client = OllamaModelClient(settings)
    get_count = 0
    post_count = 0

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.payload

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str]) -> FakeResponse:
            nonlocal get_count
            get_count += 1
            return FakeResponse(
                {"models": [{"name": "kimi-k2.6"}, {"name": "glm-5.1"}]}
            )

        async def post(
            self, url: str, headers: dict[str, str], json: dict
        ) -> FakeResponse:
            nonlocal post_count
            post_count += 1
            return FakeResponse({"message": {"role": "assistant", "content": "done"}})

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    await client.next_action(conversation=[{"role": "user", "content": "hi"}], tools=[])
    await client.next_action(conversation=[{"role": "user", "content": "hi"}], tools=[])

    assert get_count == 1
    assert post_count == 2


@pytest.mark.asyncio
async def test_timeout_retries_primary_before_fallback_and_pins_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_MODEL="kimi-k2.6:cloud",
        OLLAMA_FALLBACK_MODEL="glm-5.1:cloud",
        OLLAMA_MAX_ATTEMPTS_PER_MODEL=3,
        OLLAMA_TIMEOUT_DEMOTION_THRESHOLD=99,
    )
    client = OllamaModelClient(settings)
    posted_models: list[str] = []

    async def fake_list_models() -> list[str]:
        return ["kimi-k2.6", "glm-5.1"]

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"role": "assistant", "content": "done"}}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self, url: str, headers: dict[str, str], json: dict
        ) -> FakeResponse:
            posted_models.append(json["model"])
            if json["model"] == "kimi-k2.6":
                request = httpx.Request("POST", "https://ollama.com/api/chat")
                raise httpx.ReadTimeout("timed out", request=request)
            return FakeResponse()

    monkeypatch.setattr(client, "list_models", fake_list_models)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    await client.next_action(conversation=[{"role": "user", "content": "hi"}], tools=[])
    await client.next_action(conversation=[{"role": "user", "content": "hi"}], tools=[])

    assert posted_models == [
        "kimi-k2.6",
        "kimi-k2.6",
        "kimi-k2.6",
        "glm-5.1",
        "glm-5.1",
    ]
    assert client.active_model_name() == "glm-5.1"


@pytest.mark.asyncio
async def test_next_action_records_timeout_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_MODEL="kimi-k2.6:cloud",
        OLLAMA_MAX_ATTEMPTS_PER_MODEL=1,
        OLLAMA_CHAT_TIMEOUT_SECONDS=0.5,
        OLLAMA_TIMEOUT_DEMOTION_THRESHOLD=99,
    )
    client = OllamaModelClient(settings)
    events = []
    sequence = 0

    def next_sequence() -> int:
        nonlocal sequence
        sequence += 1
        return sequence

    async def fake_list_models() -> list[str]:
        return ["kimi-k2.6"]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, headers: dict[str, str], json: dict) -> None:
            request = httpx.Request("POST", "https://ollama.com/api/chat")
            raise httpx.ReadTimeout("timed out", request=request)

    client.set_trace_context(
        ModelTraceContext(
            run_id="run-1",
            lane_id="lane-1",
            turn_number=4,
            next_sequence=next_sequence,
            append_event=events.append,
        )
    )
    monkeypatch.setattr(client, "list_models", fake_list_models)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(httpx.ReadTimeout):
        await client.next_action(
            conversation=[{"role": "user", "content": "hi"}],
            tools=[],
        )

    assert [event.type for event in events] == [
        "model_provider_attempt_started",
        "model_provider_attempt_completed",
    ]
    assert events[0].timeout_seconds == 0.5
    assert events[1].outcome == "timeout"
    assert events[1].error_type == "ReadTimeout"


@pytest.mark.asyncio
async def test_next_action_records_malformed_chat_payload_as_completed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_MODEL="kimi-k2.6:cloud",
        OLLAMA_MAX_ATTEMPTS_PER_MODEL=1,
    )
    client = OllamaModelClient(settings)
    events = []
    sequence = 0

    def next_sequence() -> int:
        nonlocal sequence
        sequence += 1
        return sequence

    async def fake_list_models() -> list[str]:
        return ["kimi-k2.6"]

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[str]:
            return ["not", "an", "object"]

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self, url: str, headers: dict[str, str], json: dict
        ) -> FakeResponse:
            return FakeResponse()

    client.set_trace_context(
        ModelTraceContext(
            run_id="run-1",
            lane_id="lane-1",
            turn_number=4,
            next_sequence=next_sequence,
            append_event=events.append,
        )
    )
    monkeypatch.setattr(client, "list_models", fake_list_models)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(ValueError, match="must be a JSON object"):
        await client.next_action(
            conversation=[{"role": "user", "content": "hi"}],
            tools=[],
        )

    assert [event.type for event in events] == [
        "model_provider_attempt_started",
        "model_provider_attempt_completed",
    ]
    assert events[1].outcome == "error"
    assert events[1].error_type == "ValueError"
    assert "must be a JSON object" in events[1].error_message


@pytest.mark.asyncio
async def test_next_action_raises_provider_interruption_when_primary_and_fallback_exhaust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_MODEL="kimi-k2.6:cloud",
        OLLAMA_FALLBACK_MODEL="glm-5.1:cloud",
        OLLAMA_MAX_ATTEMPTS_PER_MODEL=1,
    )
    client = OllamaModelClient(settings)

    async def fake_list_models() -> list[str]:
        return ["kimi-k2.6", "glm-5.1"]

    class FakeResponse:
        def raise_for_status(self) -> None:
            request = httpx.Request("POST", "https://ollama.com/api/chat")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError(
                "503 Service Unavailable", request=request, response=response
            )

        def json(self) -> dict:
            return {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self, url: str, headers: dict[str, str], json: dict
        ) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(client, "list_models", fake_list_models)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(ProviderInterruptionError):
        await client.next_action(
            conversation=[{"role": "user", "content": "hi"}], tools=[]
        )
