from __future__ import annotations

import json

from fastapi.testclient import TestClient

from firecrawl_docs_auditor.codex_app_server import CodexAccountStatus
from firecrawl_docs_auditor.config import Settings
from firecrawl_docs_auditor.server import create_app


class ScaffoldCodexClient:
    async def read_account(self, *, refresh_token: bool = False) -> CodexAccountStatus:
        return CodexAccountStatus(
            status="unavailable",
            codex_bin_configured=False,
            codex_bin_detected=None,
            message="Fake scaffold client.",
        )


def test_server_exposes_scaffold_endpoints() -> None:
    client = TestClient(create_app(codex_client=ScaffoldCodexClient()))

    health_response = client.get("/healthz")
    status_response = client.get("/api/status")
    request_response = client.get("/api/contracts/sample-request")
    report_response = client.get("/api/contracts/sample-report")

    assert health_response.status_code == 200
    assert health_response.json() == {"ok": True, "service": "firecrawl-docs-auditor"}

    assert status_response.status_code == 200
    status = status_response.json()
    assert status["codex_app_server"]["status"] == "unavailable"
    assert status["firecrawl"]["status"] in {"missing_key", "configured"}
    assert status["audit_engine"]["status"] == "report_generation_ready"
    assert status["contracts"]["fixtures_readable"] is True

    assert request_response.status_code == 200
    assert request_response.json()["mode"] in {"live", "cached"}

    assert report_response.status_code == 200
    report = report_response.json()
    assert "scorecard" in report
    assert "selected_sources" in report
    assert "warnings" in report
    assert "suggested_fixes" in report
    assert report["metadata"]["generated_by"] == "contract_fixture"


def test_sample_request_endpoint_excludes_credentials() -> None:
    client = TestClient(create_app(codex_client=ScaffoldCodexClient()))
    response = client.get("/api/contracts/sample-request")

    assert response.status_code == 200
    serialized = json.dumps(response.json()).lower()
    forbidden = ["api_key", "firecrawl_api_key", "token", "auth", "cookie", "secret"]
    assert not any(word in serialized for word in forbidden)


def test_status_uses_settings_and_cors_origin() -> None:
    settings = Settings(
        FIRECRAWL_DOCS_AUDITOR_HOST="127.0.0.1",
        FIRECRAWL_DOCS_AUDITOR_PORT=8123,
        FIRECRAWL_DOCS_AUDITOR_FRONTEND_ORIGINS="http://127.0.0.1:3999",
    )
    client = TestClient(create_app(settings, codex_client=ScaffoldCodexClient()))

    response = client.get(
        "/api/status",
        headers={
            "Origin": "http://127.0.0.1:3999",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.json()["api"]["port"] == 8123
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:3999"
