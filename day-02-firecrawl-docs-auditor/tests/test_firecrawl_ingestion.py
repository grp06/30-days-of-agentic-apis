from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from firecrawl_docs_auditor.config import Settings
from firecrawl_docs_auditor.firecrawl_ingestion import (
    CandidateSource,
    FirecrawlCacheNotFound,
    FirecrawlFetchRequest,
    HttpFirecrawlIngestionClient,
    SourcePlannerProbe,
    SourcePlannerResult,
    SourcePlannerSelection,
)
from firecrawl_docs_auditor.firecrawl_preflight import (
    DocsPreflightResult,
    FirecrawlKeyStatus,
    FirecrawlPreflightRequest,
    PreflightCheck,
)


class FakePreflightClient:
    def __init__(
        self,
        *,
        verdict: str = "pass",
        key_status: str = "valid",
        normalized_url: str | None = "https://docs.example.com/docs/start",
    ) -> None:
        self.verdict = verdict
        self.key_status = key_status
        self.normalized_url = normalized_url

    async def run_preflight(
        self,
        request: FirecrawlPreflightRequest,
    ) -> DocsPreflightResult:
        return DocsPreflightResult(
            verdict=self.verdict,
            normalized_url=self.normalized_url,
            allowed_hosts=["docs.example.com"],
            key_status=FirecrawlKeyStatus(
                status=self.key_status,
                configured=self.key_status != "missing",
                message="Fake key status.",
            ),
            checks=[
                PreflightCheck(
                    id="public_access",
                    status="pass" if self.verdict != "blocked" else "blocked",
                    severity="info" if self.verdict != "blocked" else "blocking",
                    message="Fake preflight.",
                    url=self.normalized_url,
                )
            ],
        )


class FakeSourcePlanner:
    def __init__(self, result_factory):
        self.result_factory = result_factory
        self.candidates: list[dict[str, object]] = []

    async def plan_sources(self, **kwargs) -> SourcePlannerResult:
        self.candidates = kwargs["candidates"]
        return self.result_factory(kwargs["candidates"])


def test_fetch_sources_selects_goal_pages_and_writes_safe_cache(tmp_path: Path) -> None:
    requested_urls: list[str] = []
    client = HttpFirecrawlIngestionClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        preflight_client=FakePreflightClient(),
        transport=httpx.MockTransport(
            lambda request: _response(request, requested_urls)
        ),
        cache_root=tmp_path,
    )

    result = asyncio.run(
        client.fetch_sources(
            FirecrawlFetchRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout redirect",
                max_pages=2,
                cache_key="checkout-test",
            )
        )
    )

    assert result.status == "completed"
    assert result.cache_key == "checkout-test"
    assert result.candidate_count >= 2
    assert len(result.selected_sources) == 2
    assert result.selected_sources[0].url == "https://docs.example.com/docs/checkout"
    assert all(
        "markdown" not in source.model_dump() for source in result.selected_sources
    )
    assert requested_urls == [
        "https://api.firecrawl.dev/v2/map",
        "https://api.firecrawl.dev/v2/scrape",
        "https://api.firecrawl.dev/v2/scrape",
    ]

    artifact = tmp_path / "checkout-test" / "sources.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    serialized = json.dumps(payload).lower()
    assert payload["selected_sources"][0]["markdown_chars"] > 0
    assert "markdown" not in result.model_dump(mode="json")["selected_sources"][0]
    assert payload["fetched_sources"][0]["markdown"].startswith("#")
    assert "fc-test-secret" not in serialized
    for forbidden in [
        "authorization",
        "api_key",
        "token",
        "cookie",
        "secret",
        "user:pass",
    ]:
        assert forbidden not in serialized

    replay = asyncio.run(client.read_cached_sources("checkout-test"))
    assert replay.cache_key == result.cache_key
    assert replay.selected_sources == result.selected_sources
    assert "fetched_sources" not in replay.model_dump(mode="json")
    assert '"markdown":' not in json.dumps(replay.model_dump(mode="json")).lower()

    backend_replay = asyncio.run(client.read_cached_fetched_sources("checkout-test"))
    assert backend_replay.cache_key == "checkout-test"
    assert backend_replay.result.selected_sources == result.selected_sources
    assert backend_replay.fetched_sources[0].markdown.startswith("#")


def test_fetch_sources_uses_planner_order_and_stores_role_metadata(
    tmp_path: Path,
) -> None:
    def plan(candidates: list[dict[str, object]]) -> SourcePlannerResult:
        quickstart = next(
            candidate
            for candidate in candidates
            if str(candidate["url"]).endswith("/docs/quickstart")
        )
        return SourcePlannerResult(
            selected_sources=[
                SourcePlannerSelection(
                    candidate_id=str(quickstart["id"]),
                    evidence_roles=["setup", "install"],
                    rationale="Quickstart covers setup and install.",
                    confidence="high",
                )
            ]
        )

    planner = FakeSourcePlanner(plan)
    client = HttpFirecrawlIngestionClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        preflight_client=FakePreflightClient(),
        source_planner=planner,
        transport=httpx.MockTransport(lambda request: _response(request, [])),
        cache_root=tmp_path,
    )

    result = asyncio.run(
        client.fetch_sources(
            FirecrawlFetchRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout redirect",
                max_pages=1,
                cache_key="planned-cache",
            )
        )
    )

    assert result.status == "completed"
    assert result.selected_sources[0].url == "https://docs.example.com/docs/quickstart"
    assert "Planner role(s): setup, install." in result.selected_sources[0].reason_selected
    artifact = json.loads(
        (tmp_path / "planned-cache" / "sources.json").read_text(encoding="utf-8")
    )
    assert artifact["planner_metadata"]["status"] == "planned"
    assert artifact["planner_metadata"]["source_roles"] == {
        "src_1": ["setup", "install"]
    }
    replay = asyncio.run(client.read_cached_fetched_sources("planned-cache"))
    assert replay.planner_metadata["source_roles"]["src_1"] == ["setup", "install"]


def test_fetch_sources_validates_planner_suggested_probe_urls(
    tmp_path: Path,
) -> None:
    def plan(candidates: list[dict[str, object]]) -> SourcePlannerResult:
        return SourcePlannerResult(
            selected_sources=[
                SourcePlannerSelection(
                    candidate_id="cand_missing",
                    evidence_roles=["auth"],
                    rationale="Unknown id should be ignored.",
                )
            ],
            suggested_probe_urls=[
                SourcePlannerProbe(
                    url="https://docs.example.com/docs/auth",
                    evidence_roles=["auth"],
                    rationale="Likely auth docs page.",
                ),
                SourcePlannerProbe(
                    url="https://evil.example.com/docs/auth",
                    evidence_roles=["auth"],
                    rationale="Off-host URL must be ignored.",
                ),
                SourcePlannerProbe(
                    url="https://docs.example.com/assets/logo.png",
                    evidence_roles=["reference"],
                    rationale="Asset URL must be ignored.",
                ),
            ],
        )

    client = HttpFirecrawlIngestionClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        preflight_client=FakePreflightClient(),
        source_planner=FakeSourcePlanner(plan),
        transport=httpx.MockTransport(lambda request: _response(request, [])),
        cache_root=tmp_path,
    )

    result = asyncio.run(
        client.fetch_sources(
            FirecrawlFetchRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout redirect",
                max_pages=1,
                cache_key="probe-cache",
            )
        )
    )

    assert result.status == "completed_with_warnings"
    assert result.selected_sources[0].url == "https://docs.example.com/docs/auth"
    assert "Source planner returned unknown candidate id" in " ".join(result.warnings)
    artifact = json.loads(
        (tmp_path / "probe-cache" / "sources.json").read_text(encoding="utf-8")
    )
    assert artifact["planner_metadata"]["source_roles"] == {"src_1": ["auth"]}
    assert artifact["planner_metadata"]["suggested_probe_urls"] == [
        "https://docs.example.com/docs/auth",
        "https://evil.example.com/docs/auth",
        "https://docs.example.com/assets/logo.png",
    ]


def test_fetch_sources_falls_back_when_planner_is_unavailable(tmp_path: Path) -> None:
    def plan(candidates: list[dict[str, object]]) -> SourcePlannerResult:
        return SourcePlannerResult(
            status="fallback",
            warnings=["Codex login required."],
        )

    client = HttpFirecrawlIngestionClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        preflight_client=FakePreflightClient(),
        source_planner=FakeSourcePlanner(plan),
        transport=httpx.MockTransport(lambda request: _response(request, [])),
        cache_root=tmp_path,
    )

    result = asyncio.run(
        client.fetch_sources(
            FirecrawlFetchRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout redirect",
                max_pages=1,
                cache_key="fallback-cache",
            )
        )
    )

    assert result.status == "completed_with_warnings"
    assert result.selected_sources[0].url == "https://docs.example.com/docs/checkout"
    assert "Source planner unavailable" in " ".join(result.warnings)


def test_fetch_filters_unsafe_candidates_before_scrape(tmp_path: Path) -> None:
    scraped_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/map":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "links": [
                        {"url": "https://docs.example.com/docs/checkout"},
                        {"url": "https://docs.example.com/docs/checkout?dup=1"},
                        {"url": "https://other.example.com/docs/checkout"},
                        {"url": "http://127.0.0.1/private"},
                        {"url": "https://user:pass@docs.example.com/docs/secret"},
                        {"url": "https://docs.example.com/assets/logo.png"},
                        {"url": "https://docs.example.com/blog/checkout"},
                    ],
                },
                request=request,
            )
        scraped_urls.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "data": {
                    "markdown": "# Checkout\nBody",
                    "metadata": {"title": "Checkout"},
                }
            },
            request=request,
        )

    client = HttpFirecrawlIngestionClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        preflight_client=FakePreflightClient(),
        transport=httpx.MockTransport(handler),
        cache_root=tmp_path,
    )

    result = asyncio.run(
        client.fetch_sources(
            FirecrawlFetchRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="checkout",
                max_pages=1,
            )
        )
    )

    assert result.status == "completed"
    assert scraped_urls == ["https://api.firecrawl.dev/v2/scrape"]
    assert result.selected_sources[0].url == "https://docs.example.com/docs/checkout"


def test_goal_ranking_ignores_stopwords_and_openclaw_host_token() -> None:
    client = HttpFirecrawlIngestionClient(Settings(_env_file=None))
    preflight = DocsPreflightResult(
        verdict="pass",
        normalized_url="https://docs.openclaw.ai/",
        allowed_hosts=["docs.openclaw.ai"],
        key_status=FirecrawlKeyStatus(
            status="valid",
            configured=True,
            message="Fake key status.",
        ),
        checks=[],
    )

    ranked = client._filter_and_rank_candidates(
        [
            CandidateSource(
                url="https://docs.openclaw.ai/channels/bluebubbles",
                title="BlueBubbles - OpenClaw",
                description="Channel setup.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.openclaw.ai/gateway/security",
                title="Security - OpenClaw",
                description="Secure the gateway.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.openclaw.ai/gateway/configuration-reference",
                title="Configuration reference - OpenClaw",
                description="Configure the gateway.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
        ],
        FirecrawlFetchRequest(
            docs_url="https://docs.openclaw.ai/",
            integration_goal="set up openclaw and make sure that it is secure",
        ),
        preflight,
    )

    assert [candidate.url for candidate in ranked] == [
        "https://docs.openclaw.ai/gateway/security",
        "https://docs.openclaw.ai/gateway/configuration-reference",
        "https://docs.openclaw.ai/channels/bluebubbles",
    ]
    assert "openclaw" not in ranked[0].reason_selected.lower()
    assert "that" not in ranked[0].reason_selected.lower()


def test_goal_ranking_preserves_exact_facets_and_prefers_canonical_pages() -> None:
    client = HttpFirecrawlIngestionClient(Settings(_env_file=None))
    preflight = DocsPreflightResult(
        verdict="pass",
        normalized_url="https://docs.example.com/",
        allowed_hosts=["docs.example.com"],
        key_status=FirecrawlKeyStatus(
            status="valid",
            configured=True,
            message="Fake key status.",
        ),
        checks=[],
    )

    ranked = client._filter_and_rank_candidates(
        [
            CandidateSource(
                url="https://docs.example.com/gateway/security",
                title="Security",
                description="Secure the gateway.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/zh-CN/gateway/configuration-reference",
                title="Configuration reference",
                description="Configure the gateway.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/gateway/configuration-reference.md",
                title="Configuration reference",
                description="Configure the gateway.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/channels/slack.md",
                title="Slack channel",
                description="Set up Slack bot tokens and routing.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/plugins/analytics",
                title="Analytics",
                description="Unrelated plugin.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
        ],
        FirecrawlFetchRequest(
            docs_url="https://docs.example.com/",
            integration_goal=(
                "Set up the gateway with Slack, configure agent routing, "
                "and verify security settings"
            ),
        ),
        preflight,
    )

    ranked_urls = [candidate.url for candidate in ranked]
    assert ranked_urls.index(
        "https://docs.example.com/channels/slack.md"
    ) < ranked_urls.index("https://docs.example.com/plugins/analytics")
    assert ranked_urls.index(
        "https://docs.example.com/gateway/configuration-reference.md"
    ) < ranked_urls.index(
        "https://docs.example.com/zh-CN/gateway/configuration-reference"
    )
    slack_candidate = next(
        candidate
        for candidate in ranked
        if candidate.url.endswith("/channels/slack.md")
    )
    assert "slack" in slack_candidate.reason_selected.lower()


def test_goal_ranking_fetches_default_language_before_localized_overflow() -> None:
    client = HttpFirecrawlIngestionClient(Settings(_env_file=None))
    preflight = DocsPreflightResult(
        verdict="pass",
        normalized_url="https://docs.example.com/",
        allowed_hosts=["docs.example.com"],
        key_status=FirecrawlKeyStatus(
            status="valid",
            configured=True,
            message="Fake key status.",
        ),
        checks=[],
    )

    ranked = client._filter_and_rank_candidates(
        [
            CandidateSource(
                url="https://docs.example.com/pt-BR/quickstarts/spring-boot",
                title="Guia de inicio rapido do Spring Boot",
                description="web sdk scrape quickstart",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/es/quickstarts/nodejs",
                title="Inicio rapido de Node.js",
                description="web sdk scrape quickstart",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/ja/quickstarts/nodejs",
                title="Node.js quickstart",
                description="web sdk scrape quickstart",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/quickstarts/rust",
                title="Rust Quickstart",
                description="web sdk scrape quickstart",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/en/quickstarts/nodejs",
                title="Node.js Quickstart",
                description="web sdk scrape quickstart",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/v2/sdks/python",
                title="Python SDK",
                description="python sdk api scrape markdown",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/advanced-scraping-guide",
                title="Advanced Scraping Guide",
                description="options scrape markdown api",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
        ],
        FirecrawlFetchRequest(
            docs_url="https://docs.example.com/",
            integration_goal=(
                "Set up a Python agent to search the web, scrape markdown, "
                "and explain SDK request options"
            ),
        ),
        preflight,
    )

    ranked_urls = [candidate.url for candidate in ranked]
    first_localized_index = min(
        index
        for index, url in enumerate(ranked_urls)
        if "/pt-BR/" in url or "/es/" in url or "/ja/" in url
    )
    assert all(
        "/pt-BR/" not in url and "/es/" not in url and "/ja/" not in url
        for url in ranked_urls[:first_localized_index]
    )
    assert (
        "https://docs.example.com/v2/sdks/python" in ranked_urls[:first_localized_index]
    )
    assert (
        "https://docs.example.com/advanced-scraping-guide"
        in ranked_urls[:first_localized_index]
    )
    assert (
        "https://docs.example.com/en/quickstarts/nodejs"
        in ranked_urls[:first_localized_index]
    )


def test_goal_ranking_prefers_current_docs_before_legacy_versions() -> None:
    client = HttpFirecrawlIngestionClient(Settings(_env_file=None))
    preflight = DocsPreflightResult(
        verdict="pass",
        normalized_url="https://docs.example.com/",
        allowed_hosts=["docs.example.com"],
        key_status=FirecrawlKeyStatus(
            status="valid",
            configured=True,
            message="Fake key status.",
        ),
        checks=[],
    )

    ranked = client._filter_and_rank_candidates(
        [
            CandidateSource(
                url="https://docs.example.com/v0/sdks/python",
                title="Python SDK",
                description="python sdk api web",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/sdks/python",
                title="Python SDK",
                description="python sdk api web",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/legacy/python",
                title="Legacy Python SDK",
                description="python sdk api web",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
        ],
        FirecrawlFetchRequest(
            docs_url="https://docs.example.com/",
            integration_goal="Set up Python SDK API access",
        ),
        preflight,
    )

    ranked_urls = [candidate.url for candidate in ranked]
    assert ranked_urls.index(
        "https://docs.example.com/sdks/python"
    ) < ranked_urls.index("https://docs.example.com/v0/sdks/python")
    assert ranked_urls.index(
        "https://docs.example.com/sdks/python"
    ) < ranked_urls.index("https://docs.example.com/legacy/python")


def test_goal_ranking_probes_current_python_workflow_facets_before_legacy() -> None:
    client = HttpFirecrawlIngestionClient(Settings(_env_file=None))
    preflight = DocsPreflightResult(
        verdict="pass",
        normalized_url="https://docs.example.com/",
        allowed_hosts=["docs.example.com"],
        key_status=FirecrawlKeyStatus(
            status="valid",
            configured=True,
            message="Fake key status.",
        ),
        checks=[],
    )

    ranked = client._filter_and_rank_candidates(
        [
            CandidateSource(
                url="https://docs.example.com/advanced-scraping-guide",
                title="Advanced Scraping Guide",
                description="Request options for scraping content.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/v0/sdks/python",
                title="Python SDK",
                description="Legacy Python SDK API examples.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/quickstarts/rust",
                title="Rust Quickstart",
                description="Search the web and scrape markdown with the SDK.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/api-reference/v1-endpoint/batch-scrape",
                title="Batch Scrape",
                description="API endpoint for batch scrape jobs.",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
        ],
        FirecrawlFetchRequest(
            docs_url="https://docs.example.com/",
            integration_goal=(
                "Set up Firecrawl in a Python agent to search the web, "
                "scrape the top result as markdown, and explain API key, "
                "SDK install, and request options."
            ),
        ),
        preflight,
    )

    ranked_urls = [candidate.url for candidate in ranked]
    expected_current_facets = {
        "https://docs.example.com/quickstarts/python",
        "https://docs.example.com/sdks/python",
        "https://docs.example.com/features/search",
        "https://docs.example.com/features/scrape",
    }
    assert expected_current_facets <= set(ranked_urls[:6])
    assert ranked_urls.index(
        "https://docs.example.com/quickstarts/python"
    ) < ranked_urls.index("https://docs.example.com/quickstarts/rust")
    assert ranked_urls.index(
        "https://docs.example.com/sdks/python"
    ) < ranked_urls.index("https://docs.example.com/v0/sdks/python")
    assert "https://docs.example.com/sdk/python" not in ranked_urls


def test_localized_entrypoint_keeps_that_locale_in_primary_pool() -> None:
    client = HttpFirecrawlIngestionClient(Settings(_env_file=None))
    preflight = DocsPreflightResult(
        verdict="pass",
        normalized_url="https://docs.example.com/es/",
        allowed_hosts=["docs.example.com"],
        key_status=FirecrawlKeyStatus(
            status="valid",
            configured=True,
            message="Fake key status.",
        ),
        checks=[],
    )

    ranked = client._filter_and_rank_candidates(
        [
            CandidateSource(
                url="https://docs.example.com/pt-BR/quickstarts/nodejs",
                title="Guia rapido Node.js",
                description="web sdk scrape quickstart",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/es/quickstarts/nodejs",
                title="Inicio rapido Node.js",
                description="web sdk scrape quickstart",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
            CandidateSource(
                url="https://docs.example.com/quickstarts/nodejs",
                title="Node.js Quickstart",
                description="web sdk scrape quickstart",
                score=0,
                reason_selected="Candidate returned by Firecrawl map.",
            ),
        ],
        FirecrawlFetchRequest(
            docs_url="https://docs.example.com/es/",
            integration_goal="Configurar Node.js quickstart",
        ),
        preflight,
    )

    ranked_urls = [candidate.url for candidate in ranked]
    assert ranked_urls.index(
        "https://docs.example.com/es/quickstarts/nodejs"
    ) < ranked_urls.index("https://docs.example.com/pt-BR/quickstarts/nodejs")


def test_blocked_preflight_and_invalid_key_do_not_call_firecrawl(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected network call to {request.url}")

    blocked_client = HttpFirecrawlIngestionClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        preflight_client=FakePreflightClient(verdict="blocked"),
        transport=httpx.MockTransport(handler),
        cache_root=tmp_path,
    )
    invalid_key_client = HttpFirecrawlIngestionClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        preflight_client=FakePreflightClient(key_status="invalid"),
        transport=httpx.MockTransport(handler),
        cache_root=tmp_path,
    )
    missing_key_client = HttpFirecrawlIngestionClient(
        Settings(_env_file=None),
        preflight_client=FakePreflightClient(key_status="valid"),
        transport=httpx.MockTransport(handler),
        cache_root=tmp_path,
    )

    request = FirecrawlFetchRequest(
        docs_url="https://docs.example.com/docs/start",
        integration_goal="Build checkout",
    )

    assert asyncio.run(blocked_client.fetch_sources(request)).status == "blocked"
    invalid_key_result = asyncio.run(invalid_key_client.fetch_sources(request))
    missing_key_result = asyncio.run(missing_key_client.fetch_sources(request))
    assert invalid_key_result.status == "blocked"
    assert invalid_key_result.artifact_path is None
    assert missing_key_result.status == "blocked"
    assert missing_key_result.artifact_path is None


def test_scrape_warning_continues_and_unsafe_cache_key_is_replaced(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/map":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "links": [
                        {"url": "https://docs.example.com/docs/missing-one"},
                        {"url": "https://docs.example.com/docs/missing-two"},
                        {"url": "https://docs.example.com/docs/checkout"},
                    ],
                },
                request=request,
            )
        requested_url = request.read().decode()
        if "missing" in requested_url:
            return httpx.Response(404, request=request)
        return httpx.Response(
            200,
            json={"data": {"markdown": "# Checkout\nBody"}},
            request=request,
        )

    client = HttpFirecrawlIngestionClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        preflight_client=FakePreflightClient(),
        transport=httpx.MockTransport(handler),
        cache_root=tmp_path,
    )

    result = asyncio.run(
        client.fetch_sources(
            FirecrawlFetchRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="checkout missing",
                max_pages=2,
                cache_key="..",
            )
        )
    )

    assert result.status == "completed_with_warnings"
    assert result.cache_key != ".."
    assert result.candidate_count >= 3
    assert "/" not in result.cache_key
    assert result.artifact_path
    assert not result.artifact_path.startswith("/Users/")
    artifact = tmp_path / result.cache_key / "sources.json"
    assert '"cache_key": ".."' not in artifact.read_text(encoding="utf-8")
    assert result.selected_sources
    assert result.selected_sources[0].id == "src_1"
    assert result.selected_sources[0].url == "https://docs.example.com/docs/checkout"
    assert result.warnings

    try:
        asyncio.run(client.read_cached_sources(".."))
    except FirecrawlCacheNotFound:
        pass
    else:
        raise AssertionError("Unsafe cache key should not be readable")

    try:
        asyncio.run(client.read_cached_fetched_sources(".."))
    except FirecrawlCacheNotFound:
        pass
    else:
        raise AssertionError(
            "Unsafe cache key should not expose fetched source markdown"
        )


def test_fetch_sources_rejects_rendered_not_found_pages_and_continues(
    tmp_path: Path,
) -> None:
    scraped_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/map":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "links": [
                        {
                            "url": "https://docs.example.com/docs/missing",
                            "title": "Missing checkout",
                            "description": "checkout missing guide",
                        },
                        {
                            "url": "https://docs.example.com/docs/checkout",
                            "title": "Checkout",
                            "description": "checkout guide",
                        },
                    ],
                },
                request=request,
            )
        requested_body = request.read().decode()
        scraped_urls.append(requested_body)
        if "missing" in requested_body:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "markdown": "# Page Not Found\nThis page could not be found.",
                        "metadata": {"title": "Page Not Found", "statusCode": 404},
                    }
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "markdown": "# Checkout\nUseful checkout docs.",
                    "metadata": {"title": "Checkout"},
                }
            },
            request=request,
        )

    client = HttpFirecrawlIngestionClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        preflight_client=FakePreflightClient(),
        transport=httpx.MockTransport(handler),
        cache_root=tmp_path,
    )

    result = asyncio.run(
        client.fetch_sources(
            FirecrawlFetchRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="missing checkout",
                max_pages=1,
                cache_key="reject-not-found",
            )
        )
    )

    assert result.status == "completed_with_warnings"
    assert len(scraped_urls) == 2
    assert result.selected_sources[0].url == "https://docs.example.com/docs/checkout"
    assert "Page Not Found" not in result.selected_sources[0].title
    assert "Rejected unusable docs page" in " ".join(result.warnings)


def test_cached_replay_rejects_payload_cache_key_mismatch(tmp_path: Path) -> None:
    client = HttpFirecrawlIngestionClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        preflight_client=FakePreflightClient(),
        transport=httpx.MockTransport(lambda request: _response(request, [])),
        cache_root=tmp_path,
    )
    result = asyncio.run(
        client.fetch_sources(
            FirecrawlFetchRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout redirect",
                max_pages=1,
                cache_key="checkout-test",
            )
        )
    )
    artifact = tmp_path / result.cache_key / "sources.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["cache_key"] = "../other"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    for read in [client.read_cached_sources, client.read_cached_fetched_sources]:
        try:
            asyncio.run(read("checkout-test"))
        except FirecrawlCacheNotFound:
            pass
        else:
            raise AssertionError("Mismatched cached payload key should not replay")


def _response(request: httpx.Request, requested_urls: list[str]) -> httpx.Response:
    requested_urls.append(str(request.url))
    if request.url.path == "/v2/map":
        return httpx.Response(
            200,
            json={
                "success": True,
                "links": [
                    {
                        "url": "https://docs.example.com/docs/checkout",
                        "title": "Checkout guide",
                        "description": "Create checkout sessions and redirects.",
                    },
                    {
                        "url": "https://docs.example.com/docs/quickstart",
                        "title": "Quickstart",
                        "description": "Install and configure the SDK.",
                    },
                    {
                        "url": "https://docs.example.com/blog/checkout",
                        "title": "Checkout launch",
                    },
                ],
            },
            request=request,
        )
    url = request.read().decode()
    title = "Checkout guide" if "checkout" in url else "Quickstart"
    return httpx.Response(
        200,
        json={
            "data": {
                "markdown": f"# {title}\nUseful docs.",
                "metadata": {"title": title},
            }
        },
        request=request,
    )
