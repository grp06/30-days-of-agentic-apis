from __future__ import annotations

import json

from fastapi.testclient import TestClient

from firecrawl_docs_auditor.codex_app_server import (
    CodexAccount,
    CodexAccountStatus,
    CodexCancelLoginResult,
    CodexLoginStartResult,
    CodexLoginStatus,
    CodexSmokeTestResult,
)
from firecrawl_docs_auditor.server import create_app


class FakeCodexClient:
    def __init__(self) -> None:
        self.account = CodexAccountStatus(
            status="available_login_required",
            requires_openai_auth=True,
            codex_bin_configured=False,
            codex_bin_detected="/fake/codex",
            message="Login required.",
        )
        self.logins: dict[str, CodexLoginStatus] = {}

    async def read_account(self, *, refresh_token: bool = False) -> CodexAccountStatus:
        return self.account

    async def start_login(self, mode: str) -> CodexLoginStartResult:
        login_id = f"{mode}-login"
        self.logins[login_id] = CodexLoginStatus(
            status="pending",
            login_id=login_id,
            message="Waiting for login.",
        )
        if mode == "device_code":
            return CodexLoginStartResult(
                status="started",
                mode="device_code",
                login_id=login_id,
                verification_url="https://example.com/device",
                user_code="CODE-123",
            )
        return CodexLoginStartResult(
            status="started",
            mode="browser",
            login_id=login_id,
            auth_url="https://example.com/auth",
        )

    async def login_status(self, login_id: str) -> CodexLoginStatus:
        return self.logins.get(
            login_id,
            CodexLoginStatus(status="unknown", login_id=login_id),
        )

    async def cancel_login(self, login_id: str) -> CodexCancelLoginResult:
        self.logins[login_id] = CodexLoginStatus(
            status="canceled",
            login_id=login_id,
        )
        return CodexCancelLoginResult(status="canceled", login_id=login_id)

    async def logout(self) -> CodexAccountStatus:
        self.account = CodexAccountStatus(
            status="available_login_required",
            requires_openai_auth=True,
            codex_bin_detected="/fake/codex",
            message="Logged out.",
        )
        return self.account

    async def run_prompt_json_smoke(self) -> CodexSmokeTestResult:
        if self.account.status != "available_signed_in":
            return CodexSmokeTestResult(
                status="login_required",
                message="Login required.",
            )
        return CodexSmokeTestResult(
            status="pass",
            message="Codex app-server returned structured JSON.",
            response={"ok": True, "service": "codex_app_server"},
        )


def test_codex_account_and_status_routes_use_fake_client() -> None:
    fake = FakeCodexClient()
    fake.account = CodexAccountStatus(
        status="available_signed_in",
        requires_openai_auth=False,
        account=CodexAccount(type="chatgpt", email="dev@example.com", plan_type="plus"),
        codex_bin_configured=True,
        codex_bin_detected="/fake/codex",
        message="Signed in.",
    )
    client = TestClient(create_app(codex_client=fake))

    status_response = client.get("/api/status")
    account_response = client.get("/api/codex/account")

    assert status_response.status_code == 200
    assert status_response.json()["codex_app_server"]["status"] == "available_signed_in"
    assert status_response.json()["codex_app_server"]["account"]["email"] == "dev@example.com"
    assert account_response.status_code == 200
    assert account_response.json()["account"]["plan_type"] == "plus"


def test_codex_login_start_status_cancel_and_logout_routes() -> None:
    fake = FakeCodexClient()
    client = TestClient(create_app(codex_client=fake))

    browser_response = client.post("/api/codex/login/start", json={"mode": "browser"})
    device_response = client.post("/api/codex/login/start", json={"mode": "device_code"})
    status_response = client.get("/api/codex/login/status/browser-login")
    cancel_response = client.post(
        "/api/codex/login/cancel",
        json={"login_id": "browser-login"},
    )
    logout_response = client.post("/api/codex/logout")

    assert browser_response.status_code == 200
    assert browser_response.json()["auth_url"] == "https://example.com/auth"
    assert device_response.status_code == 200
    assert device_response.json()["user_code"] == "CODE-123"
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "pending"
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "canceled"
    assert logout_response.status_code == 200
    assert logout_response.json()["status"] == "available_login_required"


def test_codex_smoke_test_success_and_login_required() -> None:
    fake = FakeCodexClient()
    client = TestClient(create_app(codex_client=fake))

    login_required_response = client.post("/api/codex/smoke-test")
    fake.account = CodexAccountStatus(
        status="available_signed_in",
        account=CodexAccount(type="chatgpt", email="dev@example.com"),
    )
    success_response = client.post("/api/codex/smoke-test")

    assert login_required_response.status_code == 200
    assert login_required_response.json()["status"] == "login_required"
    assert success_response.status_code == 200
    assert success_response.json()["status"] == "pass"
    assert success_response.json()["response"] == {
        "ok": True,
        "service": "codex_app_server",
    }


def test_codex_routes_do_not_expose_credential_shaped_fields() -> None:
    fake = FakeCodexClient()
    client = TestClient(create_app(codex_client=fake))

    payloads = [
        client.get("/api/codex/account").json(),
        client.post("/api/codex/login/start", json={"mode": "browser"}).json(),
        client.post("/api/codex/smoke-test").json(),
    ]
    serialized = json.dumps(payloads).lower()

    forbidden = ["accesstoken", "chatgptauthtokens", "api_key", "auth.json", "cookie", "secret"]
    assert not any(word in serialized for word in forbidden)
