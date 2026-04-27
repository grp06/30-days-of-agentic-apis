from __future__ import annotations

import json

from fastapi.testclient import TestClient

from firecrawl_docs_auditor.codex_app_server import CodexAccountStatus
from firecrawl_docs_auditor.firecrawl_ingestion import (
    FirecrawlCacheNotFound,
    FirecrawlFetchRequest,
    FirecrawlFetchResult,
    SelectedSource,
)
from firecrawl_docs_auditor.firecrawl_preflight import (
    DocsPreflightResult,
    FirecrawlKeyStatus,
    FirecrawlServiceStatus,
    PreflightCheck,
)
from firecrawl_docs_auditor.server import create_app


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

    async def run_preflight(self, request):  # pragma: no cover - route test should not call this
        raise AssertionError("Fetch route should use the ingestion client fake.")


class FakeFirecrawlIngestionClient:
    async def fetch_sources(
        self,
        request: FirecrawlFetchRequest,
    ) -> FirecrawlFetchResult:
        return _result(cache_key=request.cache_key or "fake-cache", docs_url=request.docs_url)

    async def read_cached_sources(self, cache_key: str) -> FirecrawlFetchResult:
        if cache_key == "missing":
            raise FirecrawlCacheNotFound
        return _result(cache_key=cache_key, docs_url="https://docs.example.com/docs/start")


def test_firecrawl_fetch_routes_use_fake_ingestion_client() -> None:
    client = TestClient(
        create_app(
            codex_client=FakeCodexClient(),
            firecrawl_client=FakePreflightClient(),
            firecrawl_ingestion_client=FakeFirecrawlIngestionClient(),
        )
    )

    response = client.post(
        "/api/firecrawl/fetch-sources",
        json={
            "docs_url": "https://docs.example.com/docs/start",
            "integration_goal": "Build checkout",
            "max_pages": 2,
            "firecrawl_api_key": "fc-test-secret",
            "cache_key": "fake-cache",
        },
    )
    replay_response = client.get("/api/firecrawl/fetch-sources/fake-cache")

    assert response.status_code == 200
    assert replay_response.status_code == 200
    assert response.json()["selected_sources"][0]["markdown_chars"] == 27
    assert "markdown" not in response.json()["selected_sources"][0]
    assert replay_response.json()["cache_key"] == "fake-cache"
    serialized = json.dumps([response.json(), replay_response.json()]).lower()
    assert "fc-test-secret" not in serialized
    for forbidden in ["api_key", "secret", "token", "cookie", "authorization"]:
        assert forbidden not in serialized


def test_firecrawl_fetch_replay_maps_missing_cache_to_404() -> None:
    client = TestClient(
        create_app(
            codex_client=FakeCodexClient(),
            firecrawl_client=FakePreflightClient(),
            firecrawl_ingestion_client=FakeFirecrawlIngestionClient(),
        )
    )

    response = client.get("/api/firecrawl/fetch-sources/missing")

    assert response.status_code == 404


def test_api_status_reports_report_generation_ready() -> None:
    client = TestClient(
        create_app(
            codex_client=FakeCodexClient(),
            firecrawl_client=FakePreflightClient(),
            firecrawl_ingestion_client=FakeFirecrawlIngestionClient(),
        )
    )

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["audit_engine"]["status"] == "report_generation_ready"


def _result(*, cache_key: str, docs_url: str) -> FirecrawlFetchResult:
    return FirecrawlFetchResult(
        status="completed",
        cache_key=cache_key,
        preflight=DocsPreflightResult(
            verdict="pass",
            normalized_url=docs_url,
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
                    url=docs_url,
                )
            ],
        ),
        selected_sources=[
            SelectedSource(
                id="src_1",
                url="https://docs.example.com/docs/checkout",
                title="Checkout guide",
                reason_selected="Matched goal term checkout.",
                retrieved_via="firecrawl_scrape",
                markdown_chars=27,
            )
        ],
        candidate_count=1,
        artifact_path=".agent/cache/firecrawl-runs/fake-cache/sources.json",
    )
