from __future__ import annotations

import asyncio
import json

from firecrawl_docs_auditor.codex_app_server import (
    CodexAccount,
    CodexAccountStatus,
    CodexLoginStatus,
    CodexSmokeTestResult,
    CodexStructuredJsonRequest,
    CodexStructuredJsonResult,
    ManagedCodexAppServerClient,
)
from firecrawl_docs_auditor.config import Settings


def test_account_status_model_serializes_safe_fields() -> None:
    status = CodexAccountStatus(
        status="available_signed_in",
        account=CodexAccount(type="chatgpt", email="dev@example.com", plan_type="team"),
        codex_bin_configured=True,
        codex_bin_detected="/Applications/Codex.app/Contents/Resources/codex",
    )

    assert status.model_dump(mode="json") == {
        "status": "available_signed_in",
        "requires_openai_auth": False,
        "account": {
            "type": "chatgpt",
            "email": "dev@example.com",
            "plan_type": "team",
        },
        "codex_bin_configured": True,
        "codex_bin_detected": "/Applications/Codex.app/Contents/Resources/codex",
        "message": None,
    }


def test_managed_client_reports_unavailable_when_codex_binary_is_missing() -> None:
    settings = Settings(CODEX_BIN="/definitely/missing/codex")
    client = ManagedCodexAppServerClient(settings)

    status = asyncio.run(client.read_account())

    assert status.status == "unavailable"
    assert status.codex_bin_configured is True
    assert status.codex_bin_detected is None
    assert status.message == "Codex binary was not found. Set CODEX_BIN if needed."


def test_smoke_result_model_allows_structured_response() -> None:
    result = CodexSmokeTestResult(
        status="pass",
        message="ok",
        response={"ok": True, "service": "codex_app_server"},
    )

    assert result.model_dump(mode="json")["response"]["ok"] is True


def test_structured_json_models_allow_generic_schema_response() -> None:
    request = CodexStructuredJsonRequest(
        base_instructions="Return JSON.",
        prompt="Return ok.",
        output_schema={"type": "object"},
    )
    result = CodexStructuredJsonResult(
        status="pass",
        message="ok",
        response={"ok": True},
    )

    assert request.output_schema["type"] == "object"
    assert result.response == {"ok": True}


def test_login_status_drains_app_server_completion_notification() -> None:
    settings = Settings(CODEX_BIN="/definitely/missing/codex")
    client = ManagedCodexAppServerClient(settings)
    client._process = _RunningProcess()
    client._pending_logins["login-1"] = CodexLoginStatus(
        status="pending",
        login_id="login-1",
    )
    messages = [
        json.dumps(
            {
                "method": "account/login/completed",
                "params": {
                    "loginId": "login-1",
                    "success": True,
                    "error": None,
                },
            }
        )
    ]

    async def fake_read_stdout_line() -> str:
        if messages:
            return messages.pop(0)
        await asyncio.sleep(1)
        return ""

    client._read_stdout_line = fake_read_stdout_line  # type: ignore[method-assign]

    status = asyncio.run(client.login_status("login-1"))

    assert status.status == "succeeded"
    assert status.message == "Login succeeded."


def test_start_login_rejects_malformed_app_server_response() -> None:
    settings = Settings(CODEX_BIN="/definitely/missing/codex")
    client = ManagedCodexAppServerClient(settings)

    async def fake_request(*args: object, **kwargs: object) -> dict[str, object]:
        return {"result": {"type": "chatgpt", "loginId": "login-1"}}

    client._request = fake_request  # type: ignore[method-assign]

    result = asyncio.run(client.start_login("browser"))

    assert result.status == "error"
    assert result.message == "Codex app-server did not return a browser auth URL."
    assert client._pending_logins == {}


class _RunningProcess:
    returncode = None
