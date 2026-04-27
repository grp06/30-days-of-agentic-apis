from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from firecrawl_docs_auditor.audit_engine import (
    AuditReportRequest,
    CodexAuditEngineClient,
    MAX_PROMPT_BYTES,
    ReportSource,
)
from firecrawl_docs_auditor.codex_app_server import (
    CodexStructuredJsonRequest,
    CodexStructuredJsonResult,
)
from firecrawl_docs_auditor.config import Settings
from firecrawl_docs_auditor.firecrawl_ingestion import (
    CachedFetchedSources,
    FetchedSource,
    FirecrawlCacheNotFound,
    FirecrawlFetchResult,
    SelectedSource,
)
from firecrawl_docs_auditor.firecrawl_preflight import (
    DocsPreflightResult,
    FirecrawlKeyStatus,
    PreflightCheck,
)
from test_contracts import assert_valid_audit_report


def test_generate_report_normalizes_codex_json_and_writes_safe_artifacts(
    tmp_path: Path,
) -> None:
    codex = FakeCodexClient(
        {
            "summary": "The docs explain the checkout flow but leave one setup step implicit.",
            "scorecard": [
                _score("discoverability", 4, ["src_1"]),
                _score("task_fit", 4, ["src_1", "unknown"]),
                _score("completeness", 3, ["src_1"]),
                _score("copy_pasteability", 5, ["src_1"]),
                _score("agent_friction", 2, ["src_1"]),
            ],
            "extracted_facts": [
                {
                    "id": "fact_checkout",
                    "message": "Checkout uses a server call followed by a redirect.",
                    "basis": "source_backed",
                    "source_refs": ["src_1"],
                }
            ],
            "smoke_test": {
                "result": "partial",
                "basis": "source_backed",
                "message": "An agent can identify the main API call.",
                "source_refs": ["src_1"],
                "missing_facts": [
                    "Webhook verification is not in the selected excerpt."
                ],
                "likely_next_steps": ["Open the API reference for webhook signatures."],
            },
            "warnings": [],
            "suggested_fixes": [
                {
                    "id": "fix_setup",
                    "message": "Add the required environment variables near the quickstart.",
                    "basis": "inferred",
                    "severity": "info",
                }
            ],
        }
    )
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=FakeSourceClient(
            _cached_sources(status="completed_with_warnings")
        ),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    result = asyncio.run(engine.generate_report(_request()))

    assert result.status == "completed_with_warnings"
    assert result.report is not None
    payload = result.report.model_dump(mode="json", exclude_none=True)
    assert_valid_audit_report(payload)
    assert result.report.selected_sources[0].id == "src_1"
    assert "markdown_chars" not in payload["selected_sources"][0]
    assert len(result.report.scorecard) == 5
    assert result.report.scorecard[1].source_refs == ["src_1"]
    assert any("Invalid source refs" in warning for warning in result.warnings)
    assert not any("excerpted" in warning.lower() for warning in result.warnings)
    assert result.report.warnings[0].message.startswith("No major warning")
    assert (
        result.artifact_path == ".agent/cache/audit-reports/checkout-cache/report.json"
    )
    assert codex.requests
    assert "Checkout markdown body" in codex.requests[0].prompt

    report_path = tmp_path / result.artifact_path
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    serialized = json.dumps(report_payload).lower()
    assert "checkout markdown body" not in serialized
    assert "markdown_chars" not in serialized
    assert "/users/" not in serialized
    for forbidden in ["fc-", "authorization", "api_key", "token", "cookie", "secret"]:
        assert forbidden not in serialized

    markdown = (report_path.parent / "report.md").read_text(encoding="utf-8")
    assert "src_1" in markdown
    assert "https://docs.example.com/docs/checkout" in markdown


def test_codex_prompt_bounds_absence_claims_to_selected_evidence(
    tmp_path: Path,
) -> None:
    codex = FakeCodexClient(
        {
            "summary": "The selected evidence supports only a partial answer.",
            "scorecard": [
                _score("discoverability", 3, ["src_1"]),
                _score("task_fit", 3, ["src_1"]),
                _score("completeness", 2, ["src_1"]),
                _score("copy_pasteability", 2, ["src_1"]),
                _score("agent_friction", 2, ["src_1"]),
            ],
            "extracted_facts": [
                {
                    "id": "fact_1",
                    "message": "The selected page describes checkout.",
                    "basis": "source_backed",
                    "source_refs": ["src_1"],
                    "severity": "info",
                }
            ],
            "smoke_test": {
                "result": "partial",
                "basis": "source_backed",
                "message": "The selected evidence is incomplete.",
                "source_refs": ["src_1"],
                "missing_facts": [],
                "likely_next_steps": [],
            },
            "warnings": [],
            "suggested_fixes": [
                {
                    "id": "fix_1",
                    "message": "Fetch the relevant implementation page.",
                    "basis": "inferred",
                    "source_refs": ["src_1"],
                    "severity": "warning",
                }
            ],
        }
    )
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=FakeSourceClient(_cached_sources()),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    result = asyncio.run(engine.generate_report(_request()))

    assert result.report is not None
    prompt_payload = json.loads(codex.requests[0].prompt)
    instructions = " ".join(prompt_payload["instructions"]).lower()
    assert "bounded evidence packet" in instructions
    assert "not the full docs corpus" in instructions
    assert "selected evidence does not show" in instructions
    assert prompt_payload["evidence_limits"]["fetched_source_count"] == 1


def test_prompt_includes_planner_roles_and_flags_contradicted_missing_facts(
    tmp_path: Path,
) -> None:
    codex = FakeCodexClient(
        {
            "summary": "The selected evidence is partially useful.",
            "scorecard": [
                _score("discoverability", 3, ["src_1"]),
                _score("task_fit", 3, ["src_1"]),
                _score("completeness", 2, ["src_1"]),
                _score("copy_pasteability", 2, ["src_1"]),
                _score("agent_friction", 2, ["src_1"]),
            ],
            "extracted_facts": [
                {
                    "id": "fact_1",
                    "message": "The packed current source covers setup.",
                    "basis": "source_backed",
                    "source_refs": ["src_1"],
                    "severity": "info",
                }
            ],
            "smoke_test": {
                "result": "partial",
                "basis": "source_backed",
                "message": "The selected evidence is incomplete.",
                "source_refs": ["src_1"],
                "missing_facts": ["Current SDK install command"],
                "likely_next_steps": [],
            },
            "warnings": [],
            "suggested_fixes": [
                {
                    "id": "fix_1",
                    "message": "Add the current SDK install command.",
                    "basis": "inferred",
                    "source_refs": ["src_1"],
                    "severity": "warning",
                }
            ],
        }
    )
    cached_sources = _cached_sources()
    cached_sources.fetched_sources[0].markdown = (
        "# SDK setup\n"
        "Install the SDK with `pip install example-sdk`.\n\n"
        "Set `EXAMPLE_API_KEY` in the environment before initializing the client.\n\n"
        "```python\n"
        "from example import Client\n"
        "client = Client(api_key=os.environ[\"EXAMPLE_API_KEY\"])\n"
        "```\n"
    )
    cached_sources.planner_metadata = {
        "source_roles": {"src_1": ["install", "setup"]},
        "roles_by_url": {
            "https://docs.example.com/docs/checkout": ["install", "setup"]
        },
    }
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=FakeSourceClient(cached_sources),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    request = _request().model_copy(
        update={"integration_goal": "Install the SDK for checkout redirect"}
    )
    result = asyncio.run(engine.generate_report(request))

    assert result.report is not None
    prompt_payload = json.loads(codex.requests[0].prompt)
    assert prompt_payload["sources"][0]["evidence_roles"] == ["install", "setup"]
    assert prompt_payload["sources"][0]["source_flags"]["legacy"] is False
    assert "setup_install" in prompt_payload["covered_evidence_slots"]
    assert (result.report.smoke_test.missing_facts or []) == []
    assert result.report.suggested_fixes[0].message.startswith(
        "No specific documentation fix"
    )
    assert any(
        "contradicted by packed evidence slot coverage" in warning
        for warning in result.warnings
    )


def test_prompt_slots_preserve_firecrawl_python_install_and_env_evidence(
    tmp_path: Path,
) -> None:
    codex = FakeCodexClient(
        {
            "summary": "The selected evidence is enough for a Python workflow.",
            "scorecard": [
                _score("discoverability", 4, ["src_1"]),
                _score("task_fit", 4, ["src_1", "src_2"]),
                _score("completeness", 4, ["src_1", "src_2"]),
                _score("copy_pasteability", 3, ["src_1"]),
                _score("agent_friction", 3, ["src_1"]),
            ],
            "extracted_facts": [
                {
                    "id": "fact_1",
                    "message": "Python setup and API key usage are present.",
                    "basis": "source_backed",
                    "source_refs": ["src_1"],
                    "severity": "info",
                }
            ],
            "smoke_test": {
                "result": "partial",
                "basis": "source_backed",
                "message": "The packet includes the workflow.",
                "source_refs": ["src_1", "src_2"],
                "missing_facts": [
                    "Current Python SDK install command",
                    "Environment variable setup for the API key",
                ],
                "likely_next_steps": [],
            },
            "warnings": [
                {
                    "id": "warning_install",
                    "message": "No current Python SDK install command is shown.",
                    "basis": "source_backed",
                    "source_refs": ["src_1"],
                    "severity": "warning",
                }
            ],
            "suggested_fixes": [
                {
                    "id": "fix_install",
                    "message": (
                        "Add the Python SDK install command and "
                        "FIRECRAWL_API_KEY setup."
                    ),
                    "basis": "inferred",
                    "source_refs": ["src_1"],
                    "severity": "warning",
                }
            ],
        }
    )
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=FakeSourceClient(_firecrawl_python_sources()),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    result = asyncio.run(engine.generate_report(_firecrawl_request()))

    assert result.report is not None
    prompt = codex.requests[0].prompt
    prompt_payload = json.loads(prompt)
    prompt_text = prompt.lower()
    snippet_text = "\n".join(
        entry["snippet"]
        for entry in prompt_payload["evidence_ledger"]
    ).lower()
    assert len(prompt.encode("utf-8")) <= MAX_PROMPT_BYTES
    assert max(len(line) for line in prompt.splitlines()) < 2_500
    assert "skip to main content" not in prompt_text
    assert "pip install firecrawl-py" in prompt_text
    assert "firecrawl_api_key" in prompt_text
    assert "results.web[0].url" in prompt_text
    assert 'formats=["markdown"]' in snippet_text
    assert {
        "setup_install",
        "auth_secret",
        "client_initialization",
        "search",
        "result_selection",
        "scrape",
        "output_format",
    } <= set(prompt_payload["covered_evidence_slots"])
    claim_status = {
        item["id"]: item["status"]
        for item in prompt_payload["claim_evidence_summary"]
    }
    assert claim_status["output_format"] == "supported"
    output_entries = [
        entry
        for entry in prompt_payload["evidence_ledger"]
        if "output_format" in entry["supported_claim_ids"]
    ]
    assert output_entries
    assert result.report.metadata.required_claims_supported == (
        result.report.metadata.required_claims_total
    )
    assert result.report.metadata.missing_required_claims == []
    assert result.report.smoke_test.result == "pass"
    assert (result.report.smoke_test.missing_facts or []) == []
    assert all(
        "install command" not in warning.message.lower()
        for warning in result.report.warnings
    )
    assert any(
        "contradicted by packed evidence slot coverage" in warning
        for warning in result.warnings
    )


def test_python_goal_prefers_canonical_same_language_sources(
    tmp_path: Path,
) -> None:
    codex = FakeCodexClient(
        {
            "summary": "Python docs can support the requested workflow.",
            "scorecard": [
                _score("discoverability", 4, ["src_1", "src_4"]),
                _score("task_fit", 4, ["src_1", "src_4", "src_5"]),
                _score("completeness", 4, ["src_1", "src_4", "src_5"]),
                _score("copy_pasteability", 4, ["src_1", "src_4"]),
                _score("agent_friction", 4, ["src_1", "src_4"]),
            ],
            "extracted_facts": [
                {
                    "id": "fact_python",
                    "message": "The Python source-of-truth page shows the workflow.",
                    "basis": "source_backed",
                    "source_refs": ["src_1", "src_4"],
                    "severity": "info",
                }
            ],
            "smoke_test": {
                "result": "pass",
                "basis": "source_backed",
                "message": "The Python agent workflow is supported.",
                "source_refs": ["src_1", "src_4"],
                "missing_facts": [],
                "likely_next_steps": [],
            },
            "warnings": [],
            "suggested_fixes": [
                {
                    "id": "fix_recipe",
                    "message": "Add one end-to-end Python recipe.",
                    "basis": "inferred",
                    "source_refs": ["src_1", "src_4", "src_5"],
                    "severity": "info",
                }
            ],
        }
    )
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=FakeSourceClient(_cross_language_python_sources()),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    request = _firecrawl_request().model_copy(
        update={"cache_key": "cross-language-cache"}
    )
    result = asyncio.run(engine.generate_report(request))

    assert result.report is not None
    prompt_payload = json.loads(codex.requests[0].prompt)
    prompt_source_ids = [source["id"] for source in prompt_payload["sources"]]
    assert prompt_source_ids.index("src_4") < prompt_source_ids.index("src_1")
    assert prompt_source_ids.index("src_5") < prompt_source_ids.index("src_1")
    python_source = next(
        source for source in prompt_payload["sources"] if source["id"] == "src_4"
    )
    assert python_source["source_flags"]["canonical"] is True
    assert result.report.smoke_test.source_refs == ["src_4", "src_1"]
    assert result.report.suggested_fixes[0].source_refs[:2] == ["src_4", "src_5"]


def test_openclaw_sized_cache_builds_bounded_ranked_evidence_packet(
    tmp_path: Path,
) -> None:
    codex = FakeCodexClient(
        {
            "summary": "OpenClaw setup and security docs are available.",
            "scorecard": [
                _score("discoverability", 4, ["src_3"]),
                _score("task_fit", 4, ["src_9"]),
                _score("completeness", 3, ["src_5"]),
                _score("copy_pasteability", 3, ["src_13"]),
                _score("agent_friction", 2, ["src_11"]),
            ],
            "extracted_facts": [
                {
                    "id": "fact_setup",
                    "message": "Setup and security pages are present.",
                    "basis": "source_backed",
                    "source_refs": ["src_3", "src_9"],
                    "severity": "info",
                }
            ],
            "smoke_test": {
                "result": "partial",
                "basis": "source_backed",
                "message": "An agent can identify setup and security references.",
                "source_refs": ["src_3", "src_9"],
                "missing_facts": [],
                "likely_next_steps": [],
            },
            "warnings": [],
            "suggested_fixes": [
                {
                    "id": "fix_1",
                    "message": "Keep setup and security docs cross-linked.",
                    "basis": "source_backed",
                    "source_refs": ["src_3", "src_9"],
                    "severity": "info",
                }
            ],
        }
    )
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=FakeSourceClient(_openclaw_sized_sources()),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    result = asyncio.run(engine.generate_report(_openclaw_request()))

    assert result.report is not None
    prompt = codex.requests[0].prompt
    prompt_payload = json.loads(prompt)
    prompt_source_ids = [source["id"] for source in prompt_payload["sources"]]
    assert len(prompt.encode("utf-8")) <= MAX_PROMPT_BYTES
    assert max(len(line) for line in prompt.splitlines()) < 2_500
    assert {"src_3", "src_9"} <= set(prompt_source_ids[:8])
    supported_claims = {
        claim["id"]
        for claim in prompt_payload["claim_evidence_summary"]
        if claim["status"] == "supported"
    }
    assert {"sdk_install", "api_key_setup", "client_initialization"} <= supported_claims
    assert "evidence_ledger" in prompt_payload
    assert "src_18" not in prompt_source_ids
    assert result.report.metadata.fetched_source_count == 20
    assert result.report.metadata.prompt_source_count == len(prompt_source_ids)
    assert result.report.metadata.omitted_source_count == 20 - len(prompt_source_ids)
    assert result.report.metadata.prompt_note
    assert result.report.selected_sources[-1].id == "src_20"
    assert result.report.scorecard[0].source_refs == ["src_3"]


def test_multifacet_docs_cache_preserves_goal_coverage_and_skips_chrome(
    tmp_path: Path,
) -> None:
    codex = FakeCodexClient(
        {
            "summary": "The selected excerpts cover setup, security, routing, and Slack.",
            "scorecard": [
                _score("discoverability", 4, ["src_8"]),
                _score("task_fit", 3, ["src_8"]),
                _score("completeness", 3, ["src_3", "src_8"]),
                _score("copy_pasteability", 3, ["src_8"]),
                _score("agent_friction", 2, ["src_5"]),
            ],
            "extracted_facts": [
                {
                    "id": "fact_slack",
                    "message": "Slack setup uses a bot token and routing config.",
                    "basis": "source_backed",
                    "source_refs": ["src_8"],
                    "severity": "info",
                }
            ],
            "smoke_test": {
                "result": "partial",
                "basis": "source_backed",
                "message": "The excerpts contain the requested facets.",
                "source_refs": ["src_3", "src_5", "src_8"],
                "missing_facts": [],
                "likely_next_steps": [],
            },
            "warnings": [],
            "suggested_fixes": [
                {
                    "id": "fix_crosslink",
                    "message": "Cross-link channel setup from security guidance.",
                    "basis": "source_backed",
                    "source_refs": ["src_8"],
                    "severity": "warning",
                }
            ],
        }
    )
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=FakeSourceClient(_multifacet_docs_sources()),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    result = asyncio.run(engine.generate_report(_multifacet_request()))

    assert result.report is not None
    prompt = codex.requests[0].prompt
    prompt_payload = json.loads(prompt)
    prompt_source_ids = [source["id"] for source in prompt_payload["sources"]]
    prompt_text = prompt.lower()
    assert len(prompt.encode("utf-8")) <= MAX_PROMPT_BYTES
    assert max(len(line) for line in prompt.splitlines()) < 2_500
    assert "src_8" in prompt_source_ids[:8]
    assert "slack bot token" in prompt_text
    assert "agent routing rule" in prompt_text
    assert "public exposure checklist" in prompt_text
    assert "deep index slack credential reference" in prompt_text
    assert "skip to main content" not in prompt_text
    assert "on this page" not in prompt_text
    assert not any("excerpted" in warning.lower() for warning in result.warnings)
    assert "slack" in result.report.metadata.represented_goal_terms
    assert "routing" in result.report.metadata.represented_goal_terms
    assert result.report.smoke_test.source_refs == ["src_3", "src_5", "src_8"]


def test_codex_output_schema_matches_strict_response_format_subset() -> None:
    schema = CodexAuditEngineClient._codex_output_schema()

    assert "uniqueItems" not in json.dumps(schema)
    _assert_strict_required_properties(schema)


def test_missing_source_cache_blocks_without_codex_call(tmp_path: Path) -> None:
    codex = FakeCodexClient({})
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=MissingSourceClient(),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    result = asyncio.run(engine.generate_report(_request()))

    assert result.status == "blocked"
    assert result.report is None
    assert result.warnings == ["Cached source fetch not found."]
    assert codex.requests == []


def test_codex_login_required_blocks_without_writing_report(tmp_path: Path) -> None:
    codex = FakeCodexClient({}, status="login_required", message="Login required.")
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=FakeSourceClient(_cached_sources()),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    result = asyncio.run(engine.generate_report(_request()))

    assert result.status == "blocked"
    assert result.report is None
    assert result.warnings == ["Login required."]
    assert not (tmp_path / ".agent" / "cache" / "audit-reports").exists()


def test_unsafe_cached_key_blocks_before_artifact_write(tmp_path: Path) -> None:
    cached_sources = _cached_sources()
    cached_sources.cache_key = "../other"
    codex = FakeCodexClient({})
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=FakeSourceClient(cached_sources),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    result = asyncio.run(engine.generate_report(_request()))

    assert result.status == "blocked"
    assert result.report is None
    assert result.warnings == ["Cached source fetch returned an unsafe cache key."]
    assert codex.requests == []
    assert not (tmp_path / ".agent" / "cache" / "audit-reports").exists()


def test_mismatched_cached_key_blocks_before_artifact_write(tmp_path: Path) -> None:
    cached_sources = _cached_sources()
    cached_sources.cache_key = "other-cache"
    codex = FakeCodexClient({})
    engine = CodexAuditEngineClient(
        Settings(project_root=tmp_path, _env_file=None),
        codex_client=codex,
        source_client=FakeSourceClient(cached_sources),
        artifact_root=tmp_path / ".agent" / "cache" / "audit-reports",
    )

    result = asyncio.run(engine.generate_report(_request()))

    assert result.status == "blocked"
    assert result.report is None
    assert result.cache_key == "checkout-cache"
    assert result.warnings == ["Cached source fetch returned a mismatched cache key."]
    assert codex.requests == []
    assert not (tmp_path / ".agent" / "cache" / "audit-reports").exists()


def test_report_models_reject_extra_contract_fields() -> None:
    with pytest.raises(ValidationError):
        ReportSource(
            id="src_1",
            url="https://docs.example.com",
            title="Docs",
            reason_selected="Selected.",
            retrieved_via="firecrawl_scrape",
            markdown_chars=10,
        )


class FakeCodexClient:
    def __init__(
        self,
        response: dict[str, object],
        *,
        status: str = "pass",
        message: str = "ok",
    ) -> None:
        self.response = response
        self.status = status
        self.message = message
        self.requests: list[CodexStructuredJsonRequest] = []

    async def run_structured_json(
        self,
        request: CodexStructuredJsonRequest,
    ) -> CodexStructuredJsonResult:
        self.requests.append(request)
        return CodexStructuredJsonResult(
            status=self.status,  # type: ignore[arg-type]
            message=self.message,
            response=self.response if self.status == "pass" else None,
        )


class FakeSourceClient:
    def __init__(self, cached_sources: CachedFetchedSources) -> None:
        self.cached_sources = cached_sources

    async def read_cached_fetched_sources(self, cache_key: str) -> CachedFetchedSources:
        return self.cached_sources


class MissingSourceClient:
    async def read_cached_fetched_sources(self, cache_key: str) -> CachedFetchedSources:
        raise FirecrawlCacheNotFound


def _request() -> AuditReportRequest:
    return AuditReportRequest(
        docs_url="https://docs.example.com/docs/start",
        integration_goal="Build checkout redirect",
        mode="live",
        max_pages=2,
        max_depth=1,
        allowed_hosts=["docs.example.com"],
        cache_key="checkout-cache",
    )


def _openclaw_request() -> AuditReportRequest:
    return AuditReportRequest(
        docs_url="https://docs.openclaw.ai/",
        integration_goal="set up openclaw and make sure that it is secure",
        mode="live",
        max_pages=20,
        max_depth=1,
        allowed_hosts=["docs.openclaw.ai"],
        cache_key="openclaw-cache",
    )


def _multifacet_request() -> AuditReportRequest:
    return AuditReportRequest(
        docs_url="https://docs.example.com/",
        integration_goal=(
            "Set up the gateway with Slack, configure agent routing, "
            "and verify security settings before public exposure"
        ),
        mode="live",
        max_pages=20,
        max_depth=1,
        allowed_hosts=["docs.example.com"],
        cache_key="multifacet-cache",
    )


def _firecrawl_request() -> AuditReportRequest:
    return AuditReportRequest(
        docs_url="https://docs.firecrawl.dev/",
        integration_goal=(
            "Set up Firecrawl in a Python agent to search the web, scrape the "
            "top result as markdown, and explain what API key, SDK install, "
            "and request options are needed to run it safely in production."
        ),
        mode="live",
        max_pages=20,
        max_depth=1,
        allowed_hosts=["docs.firecrawl.dev"],
        cache_key="firecrawl-cache",
    )


def _cached_sources(status: str = "completed") -> CachedFetchedSources:
    result = FirecrawlFetchResult(
        status=status,  # type: ignore[arg-type]
        cache_key="checkout-cache",
        preflight=DocsPreflightResult(
            verdict="pass",
            normalized_url="https://docs.example.com/docs/start",
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
                    url="https://docs.example.com/docs/start",
                )
            ],
        ),
        selected_sources=[
            SelectedSource(
                id="src_1",
                url="https://docs.example.com/docs/checkout",
                title="Checkout guide",
                reason_selected="Matched checkout.",
                retrieved_via="firecrawl_scrape",
                markdown_chars=22,
            )
        ],
        candidate_count=1,
        warnings=["Source fetch had a warning."]
        if status == "completed_with_warnings"
        else [],
        artifact_path=".agent/cache/firecrawl-runs/checkout-cache/sources.json",
    )
    return CachedFetchedSources(
        cache_key="checkout-cache",
        result=result,
        fetched_sources=[
            FetchedSource(
                id="src_1",
                url="https://docs.example.com/docs/checkout",
                title="Checkout guide",
                reason_selected="Matched checkout.",
                retrieved_via="firecrawl_scrape",
                markdown_chars=22,
                markdown="# Checkout\nCheckout markdown body.",
            )
        ],
    )


def _firecrawl_python_sources() -> CachedFetchedSources:
    source_specs = [
        (
            "src_1",
            "Python SDK | Firecrawl",
            "https://docs.firecrawl.dev/sdks/python",
            _chrome_markdown(
                "# Python SDK\n"
                "Install the current Python package:\n\n"
                "```bash\n"
                "pip install firecrawl-py\n"
                "export FIRECRAWL_API_KEY=fc-YOUR-API-KEY\n"
                "```\n\n"
                "```python\n"
                "import os\n"
                "from firecrawl import Firecrawl\n"
                "app = Firecrawl(api_key=os.environ[\"FIRECRAWL_API_KEY\"])\n"
                "results = app.search(\"agentic api docs\", limit=1)\n"
                "top_url = results.web[0].url\n"
                "page = app.scrape(top_url, formats=[\"markdown\"])\n"
                "print(page.markdown)\n"
                "```\n"
            ),
        ),
        (
            "src_2",
            "Search endpoint",
            "https://docs.firecrawl.dev/api-reference/v1-endpoint/search",
            "# Search\nUse `limit`, `timeout`, and `scrape_options` for search requests.",
        ),
        (
            "src_3",
            "Scrape endpoint",
            "https://docs.firecrawl.dev/api-reference/v1-endpoint/scrape",
            "# Scrape\nSet `formats` to `markdown`, `html`, or `json`.",
        ),
        (
            "src_4",
            "Navigation",
            "https://docs.firecrawl.dev/introduction",
            "Skip to main content\n\nSearch\n\nOn this page\n\nIntroduction",
        ),
    ]
    selected_sources = [
        SelectedSource(
            id=source_id,
            url=url,
            title=title,
            reason_selected="Matched Firecrawl Python workflow terms.",
            retrieved_via="firecrawl_scrape",
            markdown_chars=len(markdown),
        )
        for source_id, title, url, markdown in source_specs
    ]
    result = FirecrawlFetchResult(
        status="completed",
        cache_key="firecrawl-cache",
        preflight=DocsPreflightResult(
            verdict="pass",
            normalized_url="https://docs.firecrawl.dev/",
            allowed_hosts=["docs.firecrawl.dev"],
            key_status=FirecrawlKeyStatus(
                status="valid",
                configured=True,
                message="Fake key validated.",
            ),
            checks=[],
        ),
        selected_sources=selected_sources,
        candidate_count=len(selected_sources),
        artifact_path=".agent/cache/firecrawl-runs/firecrawl-cache/sources.json",
    )
    fetched_sources = [
        FetchedSource(
            id=source.id,
            url=source.url,
            title=source.title,
            reason_selected=source.reason_selected,
            retrieved_via=source.retrieved_via,
            markdown_chars=source.markdown_chars,
            markdown=markdown,
        )
        for source, (_, _, _, markdown) in zip(
            selected_sources, source_specs, strict=True
        )
    ]
    return CachedFetchedSources(
        cache_key="firecrawl-cache",
        result=result,
        fetched_sources=fetched_sources,
    )


def _cross_language_python_sources() -> CachedFetchedSources:
    python_workflow = (
        "```bash\n"
        "pip install firecrawl-py\n"
        "export FIRECRAWL_API_KEY=fc-YOUR-API-KEY\n"
        "```\n\n"
        "```python\n"
        "import os\n"
        "from firecrawl import Firecrawl\n"
        "client = Firecrawl(api_key=os.environ[\"FIRECRAWL_API_KEY\"])\n"
        "results = client.search(\"agentic api docs\", limit=1)\n"
        "top_url = results.web[0].url\n"
        "doc = client.scrape(top_url, formats=[\"markdown\"])\n"
        "print(doc.markdown)\n"
        "```\n\n"
        "Use timeout, cache, location, and retry handling for production runs.\n"
    )
    php_workflow = (
        "# PHP SDK\n"
        "Install a PHP package, set FIRECRAWL_API_KEY, call search, select the "
        "first web URL, scrape it with markdown formats, and handle retries.\n"
    )
    ruby_workflow = (
        "# Ruby SDK\n"
        "Install a Ruby gem, set FIRECRAWL_API_KEY, call search, select the "
        "first web URL, scrape it with markdown formats, and handle retries.\n"
    )
    source_specs = [
        (
            "src_1",
            "PHP SDK",
            "https://docs.example.com/sdks/php",
            php_workflow,
        ),
        (
            "src_2",
            "Ruby SDK",
            "https://docs.example.com/sdks/ruby",
            ruby_workflow,
        ),
        (
            "src_3",
            "Scrape API reference",
            "https://docs.example.com/api-reference/scrape",
            "# Scrape API\nUse `formats=[\"markdown\"]`, timeout, cache, and location options.",
        ),
        (
            "src_4",
            "Python Source of Truth",
            "https://docs.example.com/agent-source-of-truth/python",
            "# Python Source of Truth\n" + python_workflow,
        ),
        (
            "src_5",
            "Python Quickstart",
            "https://docs.example.com/quickstarts/python",
            "# Python Quickstart\n" + python_workflow,
        ),
    ]
    selected_sources = [
        SelectedSource(
            id=source_id,
            url=url,
            title=title,
            reason_selected="Matched Python search scrape markdown workflow terms.",
            retrieved_via="firecrawl_scrape",
            markdown_chars=len(markdown),
        )
        for source_id, title, url, markdown in source_specs
    ]
    result = FirecrawlFetchResult(
        status="completed",
        cache_key="cross-language-cache",
        preflight=DocsPreflightResult(
            verdict="pass",
            normalized_url="https://docs.example.com/",
            allowed_hosts=["docs.example.com"],
            key_status=FirecrawlKeyStatus(
                status="valid",
                configured=True,
                message="Fake key validated.",
            ),
            checks=[],
        ),
        selected_sources=selected_sources,
        candidate_count=len(selected_sources),
        artifact_path=".agent/cache/firecrawl-runs/cross-language-cache/sources.json",
    )
    fetched_sources = [
        FetchedSource(
            id=source.id,
            url=source.url,
            title=source.title,
            reason_selected=source.reason_selected,
            retrieved_via=source.retrieved_via,
            markdown_chars=source.markdown_chars,
            markdown=markdown,
        )
        for source, (_, _, _, markdown) in zip(
            selected_sources, source_specs, strict=True
        )
    ]
    return CachedFetchedSources(
        cache_key="cross-language-cache",
        result=result,
        fetched_sources=fetched_sources,
    )


def _multifacet_docs_sources() -> CachedFetchedSources:
    source_specs = [
        (
            "src_1",
            "Security overview",
            "https://docs.example.com/gateway/security",
            _chrome_markdown(
                "# Security overview\nSecurity headers and deployment posture."
            ),
        ),
        (
            "src_2",
            "Security audit checks",
            "https://docs.example.com/gateway/security/audit-checks",
            _chrome_markdown(
                "# Security audit checks\nAudit allowed origins and public endpoints."
            ),
        ),
        (
            "src_3",
            "Configuration reference",
            "https://docs.example.com/gateway/configuration-reference.md",
            "# Configuration reference\nUse `gateway.yaml` to configure the server.\n\n"
            "## Public exposure checklist\nVerify HTTPS, allowed origins, auth, and audit logging before launch.",
        ),
        (
            "src_4",
            "Configuration reference",
            "https://docs.example.com/zh-CN/gateway/configuration-reference",
            "# Configuration reference\nLocalized duplicate for gateway configuration.",
        ),
        (
            "src_5",
            "Agent routing",
            "https://docs.example.com/concepts/agent-routing",
            _chrome_markdown(
                "# Agent routing\nCreate an agent routing rule that sends Slack messages to the support agent."
            ),
        ),
        (
            "src_6",
            "Generic setup",
            "https://docs.example.com/start/setup",
            "# Generic setup\nInstall the CLI and start the gateway.",
        ),
        (
            "src_7",
            "Secrets",
            "https://docs.example.com/gateway/secrets",
            "# Secrets\nStore credentials through SecretRef values.",
        ),
        (
            "src_8",
            "Slack channel",
            "https://docs.example.com/channels/slack.md",
            _chrome_markdown(
                "# Slack channel\n"
                "Create a Slack bot token and save it as SLACK_BOT_TOKEN.\n\n"
                "## Gateway config\n"
                "Add a Slack connector route and reference the agent routing rule."
            ),
        ),
        (
            "src_9",
            "llms-full",
            "https://docs.example.com/llms-full.txt",
            "Introductory index text.\n\n"
            + ("Navigation filler.\n\n" * 700)
            + "## Slack credential index\nDeep index Slack credential reference for bot tokens and routing.",
        ),
        (
            "src_10",
            "Unrelated analytics",
            "https://docs.example.com/plugins/analytics",
            "# Analytics\nUnrelated plugin details.",
        ),
    ]
    selected_sources = [
        SelectedSource(
            id=source_id,
            url=url,
            title=title,
            reason_selected="Selected from generic docs candidates.",
            retrieved_via="firecrawl_scrape",
            markdown_chars=len(markdown),
        )
        for source_id, title, url, markdown in source_specs
    ]
    result = FirecrawlFetchResult(
        status="completed",
        cache_key="multifacet-cache",
        preflight=DocsPreflightResult(
            verdict="pass",
            normalized_url="https://docs.example.com/",
            allowed_hosts=["docs.example.com"],
            key_status=FirecrawlKeyStatus(
                status="valid",
                configured=True,
                message="Fake key validated.",
            ),
            checks=[],
        ),
        selected_sources=selected_sources,
        candidate_count=len(selected_sources),
        artifact_path=".agent/cache/firecrawl-runs/multifacet-cache/sources.json",
    )
    fetched_sources = [
        FetchedSource(
            id=source.id,
            url=source.url,
            title=source.title,
            reason_selected=source.reason_selected,
            retrieved_via=source.retrieved_via,
            markdown_chars=source.markdown_chars,
            markdown=markdown,
        )
        for source, (_, _, _, markdown) in zip(
            selected_sources, source_specs, strict=True
        )
    ]
    return CachedFetchedSources(
        cache_key="multifacet-cache",
        result=result,
        fetched_sources=fetched_sources,
    )


def _chrome_markdown(content: str) -> str:
    return f"Skip to main content\n\nSearch\n\nEnglish\n\nOn this page\n\n{content}"


def _openclaw_sized_sources() -> CachedFetchedSources:
    titles = [
        (
            "src_1",
            "Plugin entry points - OpenClaw",
            "https://docs.openclaw.ai/plugins/sdk-entrypoints",
            11_615,
        ),
        (
            "src_2",
            "Tool-loop detection - OpenClaw",
            "https://docs.openclaw.ai/tools/loop-detection",
            5_117,
        ),
        ("src_3", "Setup - OpenClaw", "https://docs.openclaw.ai/start/setup", 9_925),
        (
            "src_4",
            "Active memory - OpenClaw",
            "https://docs.openclaw.ai/concepts/active-memory",
            22_871,
        ),
        (
            "src_5",
            "Configuration reference - OpenClaw",
            "https://docs.openclaw.ai/gateway/configuration-reference",
            56_692,
        ),
        (
            "src_6",
            "Slash commands - OpenClaw",
            "https://docs.openclaw.ai/tools/slash-commands",
            27_036,
        ),
        (
            "src_7",
            "Community plugins - OpenClaw",
            "https://docs.openclaw.ai/plugins/community",
            8_473,
        ),
        (
            "src_8",
            "Multi-agent sandbox & tools - OpenClaw",
            "https://docs.openclaw.ai/tools/multi-agent-sandbox-tools",
            15_725,
        ),
        (
            "src_9",
            "Security - OpenClaw",
            "https://docs.openclaw.ai/gateway/security",
            80_000,
        ),
        (
            "src_10",
            "Groups - OpenClaw",
            "https://docs.openclaw.ai/channels/groups",
            18_985,
        ),
        (
            "src_11",
            "Security - OpenClaw",
            "https://docs.openclaw.ai/cli/security",
            7_287,
        ),
        (
            "src_12",
            "Synology Chat - OpenClaw",
            "https://docs.openclaw.ai/channels/synology-chat",
            9_859,
        ),
        (
            "src_13",
            "Configuration - agents - OpenClaw",
            "https://docs.openclaw.ai/gateway/config-agents",
            60_949,
        ),
        (
            "src_14",
            "Browser control API - OpenClaw",
            "https://docs.openclaw.ai/tools/browser-control",
            16_650,
        ),
        (
            "src_15",
            "Trusted Proxy Auth - OpenClaw",
            "https://docs.openclaw.ai/gateway/trusted-proxy-auth",
            20_650,
        ),
        (
            "src_16",
            "Exec approvals - OpenClaw",
            "https://docs.openclaw.ai/tools/exec-approvals",
            34_803,
        ),
        (
            "src_17",
            "Local Models - OpenClaw",
            "https://docs.openclaw.ai/gateway/local-models",
            10_077,
        ),
        (
            "src_18",
            "BlueBubbles - OpenClaw",
            "https://docs.openclaw.ai/channels/bluebubbles",
            33_964,
        ),
        (
            "src_19",
            "Plugin SDK subpaths - OpenClaw",
            "https://docs.openclaw.ai/plugins/sdk-subpaths",
            24_732,
        ),
        (
            "src_20",
            "Mattermost - OpenClaw",
            "https://docs.openclaw.ai/channels/mattermost",
            23_577,
        ),
    ]
    selected_sources = [
        SelectedSource(
            id=source_id,
            url=url,
            title=title,
            reason_selected="Matched goal term(s): openclaw.",
            retrieved_via="firecrawl_scrape",
            markdown_chars=chars,
        )
        for source_id, title, url, chars in titles
    ]
    result = FirecrawlFetchResult(
        status="completed",
        cache_key="openclaw-cache",
        preflight=DocsPreflightResult(
            verdict="pass",
            normalized_url="https://docs.openclaw.ai/",
            allowed_hosts=["docs.openclaw.ai"],
            key_status=FirecrawlKeyStatus(
                status="valid",
                configured=True,
                message="Fake key validated.",
            ),
            checks=[],
        ),
        selected_sources=selected_sources,
        candidate_count=len(selected_sources),
        artifact_path=".agent/cache/firecrawl-runs/openclaw-cache/sources.json",
    )
    fetched_sources = [
        FetchedSource(
            id=source.id,
            url=source.url,
            title=source.title,
            reason_selected=source.reason_selected,
            retrieved_via=source.retrieved_via,
            markdown_chars=source.markdown_chars,
            markdown=_markdown_for_source(source.title, source.markdown_chars),
        )
        for source in selected_sources
    ]
    return CachedFetchedSources(
        cache_key="openclaw-cache",
        result=result,
        fetched_sources=fetched_sources,
    )


def _markdown_for_source(title: str, chars: int) -> str:
    lowered = title.lower()
    if "setup" in lowered:
        paragraph = (
            f"# {title}\n"
            "Install the SDK with `pip install openclaw` and initialize the "
            "client with `client = OpenClaw()` before running the gateway.\n\n"
        )
        return (paragraph * ((chars // len(paragraph)) + 1))[:chars]
    if "security" in lowered or "trusted" in lowered:
        paragraph = (
            f"# {title}\n"
            "Store credentials in environment variables such as "
            "`OPENCLAW_API_KEY`, require bearer authentication, and audit "
            "trusted proxy access before public exposure.\n\n"
        )
        return (paragraph * ((chars // len(paragraph)) + 1))[:chars]
    if "configuration" in lowered:
        paragraph = (
            f"# {title}\n"
            "Configure the gateway with allowed origins, auth settings, "
            "approval policies, timeout values, and agent routing options.\n\n"
        )
        return (paragraph * ((chars // len(paragraph)) + 1))[:chars]
    paragraph = (
        f"# {title}\n"
        "This page describes setup, configuration, security, authentication, "
        "credentials, approvals, and operational details for an agent.\n\n"
    )
    return (paragraph * ((chars // len(paragraph)) + 1))[:chars]


def _score(dimension_id: str, score: int, refs: list[str]) -> dict[str, object]:
    return {
        "id": dimension_id,
        "label": dimension_id.replace("_", " ").title(),
        "score": score,
        "max_score": 5,
        "rationale": f"{dimension_id} rationale.",
        "source_refs": refs,
    }


def _assert_strict_required_properties(value: object) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict) and value.get("additionalProperties") is False:
            assert set(value.get("required", [])) == set(properties)
        for child in value.values():
            _assert_strict_required_properties(child)
    elif isinstance(value, list):
        for child in value:
            _assert_strict_required_properties(child)
