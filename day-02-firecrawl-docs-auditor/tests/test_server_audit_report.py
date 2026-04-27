from __future__ import annotations

import json

from fastapi.testclient import TestClient

from firecrawl_docs_auditor.audit_engine import (
    AuditReport,
    AuditReportMetadata,
    AuditReportRequest,
    AuditReportResult,
    EvidenceItem,
    ReportSource,
    ScorecardDimension,
    SmokeTestResult,
)
from firecrawl_docs_auditor.codex_app_server import CodexAccountStatus
from firecrawl_docs_auditor.firecrawl_preflight import FirecrawlServiceStatus
from firecrawl_docs_auditor.server import create_app


def test_audit_report_route_uses_fake_engine_and_serializes_safe_report() -> None:
    engine = FakeAuditEngine()
    client = TestClient(
        create_app(
            codex_client=FakeCodexClient(),
            firecrawl_client=FakePreflightClient(),
            audit_engine_client=engine,
        )
    )

    response = client.post(
        "/api/audit/report",
        json={
            "docs_url": "https://docs.example.com/docs/start",
            "integration_goal": "Build checkout redirect",
            "mode": "live",
            "max_pages": 2,
            "max_depth": 1,
            "allowed_hosts": ["docs.example.com"],
            "cache_key": "checkout-cache",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["report"]["metadata"]["generated_by"] == "codex_app_server"
    assert body["artifact_path"] == ".agent/cache/audit-reports/checkout-cache/report.json"
    assert engine.requests[0].cache_key == "checkout-cache"
    serialized = json.dumps(body).lower()
    for forbidden in ["fc-", "authorization", "api_key", "token", "cookie", "secret"]:
        assert forbidden not in serialized


def test_api_status_reports_report_generation_ready() -> None:
    client = TestClient(
        create_app(
            codex_client=FakeCodexClient(),
            firecrawl_client=FakePreflightClient(),
            audit_engine_client=FakeAuditEngine(),
        )
    )

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["audit_engine"]["status"] == "report_generation_ready"


def test_audit_report_route_returns_blocked_when_cache_key_is_missing() -> None:
    client = TestClient(
        create_app(
            codex_client=FakeCodexClient(),
            firecrawl_client=FakePreflightClient(),
        )
    )

    response = client.post(
        "/api/audit/report",
        json={
            "docs_url": "https://docs.example.com/docs/start",
            "integration_goal": "Build checkout redirect",
            "mode": "live",
            "max_pages": 2,
            "max_depth": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "blocked"
    assert response.json()["warnings"] == [
        "Source cache key is required before report generation."
    ]


class FakeAuditEngine:
    def __init__(self) -> None:
        self.requests: list[AuditReportRequest] = []

    async def generate_report(self, request: AuditReportRequest) -> AuditReportResult:
        self.requests.append(request)
        report = AuditReport(
            request=request,
            status="completed",
            summary="The docs can support the checkout task.",
            scorecard=[
                ScorecardDimension(
                    id="discoverability",
                    label="Discoverability",
                    score=4,
                    max_score=5,
                    rationale="The checkout guide is findable.",
                    source_refs=["src_1"],
                )
            ],
            selected_sources=[
                ReportSource(
                    id="src_1",
                    url="https://docs.example.com/docs/checkout",
                    title="Checkout guide",
                    reason_selected="Matched checkout.",
                    retrieved_via="firecrawl_scrape",
                )
            ],
            extracted_facts=[
                EvidenceItem(
                    id="fact_1",
                    message="Checkout redirects after a server call.",
                    basis="source_backed",
                    source_refs=["src_1"],
                )
            ],
            smoke_test=SmokeTestResult(
                result="pass",
                basis="source_backed",
                message="An agent can identify the main call.",
                source_refs=["src_1"],
            ),
            warnings=[
                EvidenceItem(
                    id="warning_1",
                    message="No major warning was found.",
                    basis="inferred",
                    severity="info",
                )
            ],
            suggested_fixes=[
                EvidenceItem(
                    id="fix_1",
                    message="Keep examples close to setup.",
                    basis="inferred",
                    severity="info",
                )
            ],
            metadata=AuditReportMetadata(
                mode=request.mode,
                generated_by="codex_app_server",
                cache_key=request.cache_key,
            ),
        )
        return AuditReportResult(
            status="completed",
            report=report,
            cache_key=request.cache_key,
            artifact_path=".agent/cache/audit-reports/checkout-cache/report.json",
        )


class FakeCodexClient:
    async def read_account(self, *, refresh_token: bool = False) -> CodexAccountStatus:
        return CodexAccountStatus(status="unavailable", message="Fake Codex client.")


class FakePreflightClient:
    async def key_status(self) -> FirecrawlServiceStatus:
        return FirecrawlServiceStatus(
            status="configured",
            configured=True,
            message="Fake Firecrawl key configured.",
        )
