from __future__ import annotations

import json

from fastapi.testclient import TestClient

from firecrawl_docs_auditor.codex_app_server import CodexAccountStatus
from firecrawl_docs_auditor.firecrawl_preflight import (
    DocsPreflightResult,
    FirecrawlKeyStatus,
    FirecrawlPreflightRequest,
    FirecrawlServiceStatus,
    PreflightCheck,
)
from firecrawl_docs_auditor.server import create_app


class FakeCodexClient:
    async def read_account(self, *, refresh_token: bool = False) -> CodexAccountStatus:
        return CodexAccountStatus(status="unavailable", message="Fake Codex client.")


class FakeFirecrawlClient:
    async def key_status(self) -> FirecrawlServiceStatus:
        return FirecrawlServiceStatus(
            status="configured",
            configured=True,
            message="Fake Firecrawl key configured.",
        )

    async def run_preflight(
        self,
        request: FirecrawlPreflightRequest,
    ) -> DocsPreflightResult:
        return DocsPreflightResult(
            verdict="warning",
            normalized_url=request.docs_url,
            allowed_hosts=["docs.example.com"],
            key_status=FirecrawlKeyStatus(
                status="valid",
                configured=True,
                message="Fake key validated.",
            ),
            checks=[
                PreflightCheck(
                    id="public_access",
                    status="pass",
                    severity="info",
                    message="Fake docs URL reachable.",
                    url=request.docs_url,
                ),
                PreflightCheck(
                    id="llms_txt",
                    status="warning",
                    severity="warning",
                    message="Fake llms.txt missing.",
                    url="https://docs.example.com/llms.txt",
                ),
            ],
        )


def test_firecrawl_status_and_preflight_routes_use_fake_client() -> None:
    client = TestClient(
        create_app(
            codex_client=FakeCodexClient(),
            firecrawl_client=FakeFirecrawlClient(),
        )
    )

    status_response = client.get("/api/firecrawl/status")
    preflight_response = client.post(
        "/api/firecrawl/preflight",
        json={
            "docs_url": "https://docs.example.com/docs/start",
            "integration_goal": "Build checkout",
            "max_pages": 50,
            "firecrawl_api_key": "fc-test-secret",
        },
    )

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "configured"
    assert preflight_response.status_code == 200
    assert preflight_response.json()["verdict"] == "warning"
    assert preflight_response.json()["checks"][0]["id"] == "public_access"
    serialized = json.dumps([status_response.json(), preflight_response.json()]).lower()
    assert "fc-test-secret" not in serialized
    assert "api_key" not in serialized
    assert "secret" not in serialized
    assert "token" not in serialized
    assert "cookie" not in serialized


def test_api_status_embeds_safe_firecrawl_status() -> None:
    client = TestClient(
        create_app(
            codex_client=FakeCodexClient(),
            firecrawl_client=FakeFirecrawlClient(),
        )
    )

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["firecrawl"] == {
        "status": "configured",
        "configured": True,
        "message": "Fake Firecrawl key configured.",
    }
