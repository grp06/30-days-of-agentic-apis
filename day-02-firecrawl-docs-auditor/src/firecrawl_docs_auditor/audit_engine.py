from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from .codex_app_server import (
    CodexAppServerClient,
    CodexStructuredJsonRequest,
)
from .config import Settings
from .firecrawl_ingestion import (
    CachedFetchedSources,
    FirecrawlCacheNotFound,
    FirecrawlIngestionClient,
)
from .firecrawl_preflight import MAX_AUDIT_PAGES


ReportStatus = Literal["completed", "completed_with_warnings", "blocked"]
Basis = Literal["source_backed", "inferred", "uncertain"]
Severity = Literal["info", "warning", "blocking"]
SmokeResult = Literal["pass", "partial", "fail"]

SCORECARD_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("discoverability", "Discoverability"),
    ("task_fit", "Task fit"),
    ("completeness", "Completeness"),
    ("copy_pasteability", "Copy/pasteability"),
    ("agent_friction", "Agent friction"),
)
SCORECARD_IDS = {dimension_id for dimension_id, _ in SCORECARD_DIMENSIONS}
MAX_PROMPT_BYTES = 45_000
MAX_TOTAL_SNIPPET_CHARS = 30_000
MAX_EVIDENCE_SOURCES = 8
MAX_SNIPPETS_PER_SOURCE = 4
MAX_SNIPPET_CHARS = 1_500
MAX_LEDGER_ENTRIES_PER_CLAIM = 3
MAX_LEDGER_ENTRY_CHARS = 900
MAX_LEDGER_TOTAL_CHARS = 24_000
SAFE_REPORT_CACHE_KEY = re.compile(r"^[A-Za-z0-9._-]+$")
GOAL_STOPWORDS = {
    "a",
    "and",
    "before",
    "can",
    "for",
    "how",
    "it",
    "make",
    "needed",
    "of",
    "or",
    "set",
    "sure",
    "that",
    "the",
    "this",
    "to",
    "with",
}
SETUP_SECURITY_TERMS = {
    "approval",
    "auth",
    "config",
    "configuration",
    "credential",
    "install",
    "quickstart",
    "security",
    "setup",
    "start",
    "trusted",
}
LEGACY_PATH_SEGMENTS = {"deprecated", "legacy", "v0"}
LANGUAGE_ALIASES: dict[str, set[str]] = {
    "python": {"python", "py"},
    "javascript": {"javascript", "js"},
    "typescript": {"typescript", "ts"},
    "node": {"node", "nodejs", "node.js"},
    "ruby": {"ruby", "rb"},
    "php": {"php"},
    "go": {"go", "golang"},
    "rust": {"rust", "rs"},
    "java": {"java"},
    "elixir": {"elixir", "ex"},
    "dotnet": {"dotnet", ".net", "csharp", "c#"},
}
CHROME_LINE_PATTERNS = (
    re.compile(r"^skip to (main )?content\.?$", re.IGNORECASE),
    re.compile(
        r"^(search|menu|navigation|on this page|table of contents)$", re.IGNORECASE
    ),
    re.compile(r"^(english|language|languages)$", re.IGNORECASE),
    re.compile(r"^(previous|next|copyright|all rights reserved)$", re.IGNORECASE),
    re.compile(r"^!\[[^\]]*\]\([^)]*\)$"),
    re.compile(r"^(ctrl\+? ?k|ctrl\+? ?i|chat widget|loading\.\.\.)$", re.IGNORECASE),
    re.compile(r"^(suggest edits|raise issue)$", re.IGNORECASE),
)
LOCALE_SEGMENT = re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$", re.IGNORECASE)
COMMAND_OR_CONFIG_PATTERNS = (
    re.compile(r"```"),
    re.compile(
        r"\b(?:curl|npm|pnpm|yarn|uv|pip|python|docker|npx|openclaw|export)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b"),
    re.compile(
        r"\b(?:token|secret|api[_-]?key|config|configuration|credential|auth|install|setup)\b",
        re.IGNORECASE,
    ),
)
EVIDENCE_SLOT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "setup_install": (
        re.compile(r"\b(?:pip|uv|poetry|npm|pnpm|yarn|bun|cargo|gem)\s+(?:install|add)\b", re.IGNORECASE),
        re.compile(r"\binstall(?:ing|ation)?\b.{0,80}\b(?:sdk|package|client|library)\b", re.IGNORECASE),
        re.compile(r"\b(?:requirements\.txt|pyproject\.toml|package\.json|mix\.exs)\b", re.IGNORECASE),
    ),
    "auth_secret": (
        re.compile(r"\b(?:api[_ -]?key|authorization|bearer|secret|credential)\b", re.IGNORECASE),
        re.compile(r"\b[A-Z][A-Z0-9_]*API[A-Z0-9_]*KEY\b"),
        re.compile(r"\b(?:os\.environ|process\.env|System\.get_env|std::env::var|export)\b", re.IGNORECASE),
    ),
    "client_initialization": (
        re.compile(r"\bfrom\s+\w+\s+import\b"),
        re.compile(r"\b(?:client|app|firecrawl)\s*=\s*[A-Z][A-Za-z0-9_]*\(", re.IGNORECASE),
        re.compile(r"\bnew\s+[A-Z][A-Za-z0-9_]*\("),
    ),
    "search": (
        re.compile(r"\.search\s*\(", re.IGNORECASE),
        re.compile(r"\bsearch\s*\(", re.IGNORECASE),
        re.compile(r"/search\b", re.IGNORECASE),
    ),
    "result_selection": (
        re.compile(r"\bresults?\.web\b", re.IGNORECASE),
        re.compile(r"\bdata\[?['\"]?web['\"]?\]?", re.IGNORECASE),
        re.compile(r"\b(?:first|top)\b.{0,80}\b(?:result|url)\b", re.IGNORECASE),
        re.compile(r"\b(?:result|item|entry)\.(?:url|link)\b", re.IGNORECASE),
    ),
    "scrape": (
        re.compile(r"\.scrape\s*\(", re.IGNORECASE),
        re.compile(r"\bscrape\s*\(", re.IGNORECASE),
        re.compile(r"/scrape\b", re.IGNORECASE),
    ),
    "output_format": (
        re.compile(r"\bmarkdown\b", re.IGNORECASE),
        re.compile(r"\bformats?\s*[:=].{0,80}\b(?:markdown|html|json|links)\b", re.IGNORECASE),
        re.compile(r"\b(?:format|output)\b", re.IGNORECASE),
    ),
    "configuration_options": (
        re.compile(r"\b(?:limit|timeout|headers|options|parameters|scrape[_-]?options|scrapeOptions|max_age|maxAge)\b", re.IGNORECASE),
    ),
    "error_handling": (
        re.compile(r"\b(?:error|exception|try:|except|catch|retry|retries|backoff|429|408|5xx|timeout)\b", re.IGNORECASE),
        re.compile(r"\brate limit", re.IGNORECASE),
    ),
    "production_safety": (
        re.compile(r"\b(?:production|rate limit|retry|backoff|timeout|secret|environment|env var|zdr|zero data retention|cache)\b", re.IGNORECASE),
    ),
}


@dataclass(frozen=True)
class EvidencePlan:
    source: Any
    score: int
    facets: set[str]
    canonical_key: str


@dataclass(frozen=True)
class PromptEvidenceSummary:
    fetched_source_count: int
    prompt_source_count: int
    omitted_source_count: int
    represented_goal_terms: list[str]
    missing_goal_terms: list[str]
    covered_evidence_slots: list[str]
    missing_evidence_slots: list[str]
    supported_claims: list[str]
    missing_claims: list[str]
    required_claims_supported: int
    required_claims_total: int
    optional_claims_supported: int
    optional_claims_total: int
    missing_required_claims: list[str]
    missing_optional_claims: list[str]
    note: str


@dataclass(frozen=True)
class EvidenceSnippetPlan:
    source: Any
    index: int
    score: int
    slots: set[str]
    snippet: str


@dataclass(frozen=True)
class ClaimRequirement:
    id: str
    label: str
    description: str
    evidence_slots: set[str]
    keywords: set[str]
    required_for_pass: bool


@dataclass(frozen=True)
class EvidenceLedgerEntry:
    claim_id: str
    source: Any
    snippet: str
    evidence_slots: set[str]
    score: int
    confidence: Literal["high", "medium", "low"]
    reason: str


class AuditReportRequest(BaseModel):
    docs_url: str = Field(min_length=1)
    integration_goal: str = Field(min_length=1)
    mode: Literal["live", "cached"] = "live"
    max_pages: int = Field(default=20, ge=1, le=MAX_AUDIT_PAGES)
    max_depth: int = Field(default=1, ge=0, le=3)
    allowed_hosts: list[str] | None = None
    cache_key: str = ""

    model_config = ConfigDict(extra="forbid")


class ReportSource(BaseModel):
    id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    reason_selected: str = Field(min_length=1)
    retrieved_via: Literal[
        "firecrawl_map",
        "firecrawl_scrape",
        "llms_txt",
        "sitemap",
        "cached_fixture",
    ]

    model_config = ConfigDict(extra="forbid")


class ScorecardDimension(BaseModel):
    id: Literal[
        "discoverability",
        "task_fit",
        "completeness",
        "copy_pasteability",
        "agent_friction",
    ]
    label: str = Field(min_length=1)
    score: int = Field(ge=0)
    max_score: int = Field(default=5, ge=1)
    rationale: str = Field(min_length=1)
    source_refs: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class EvidenceItem(BaseModel):
    id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    basis: Basis
    source_refs: list[str] | None = None
    severity: Severity | None = None

    model_config = ConfigDict(extra="forbid")


class SmokeTestResult(BaseModel):
    result: SmokeResult
    basis: Basis
    message: str = Field(min_length=1)
    source_refs: list[str] | None = None
    missing_facts: list[str] | None = None
    likely_next_steps: list[str] | None = None

    model_config = ConfigDict(extra="forbid")


class AuditReportMetadata(BaseModel):
    mode: Literal["live", "cached"]
    generated_by: str = Field(min_length=1)
    cache_key: str | None = None
    notes: str | None = None
    fetched_source_count: int = 0
    prompt_source_count: int = 0
    omitted_source_count: int = 0
    represented_goal_terms: list[str] = Field(default_factory=list)
    missing_goal_terms: list[str] = Field(default_factory=list)
    supported_claims: list[str] = Field(default_factory=list)
    missing_claims: list[str] = Field(default_factory=list)
    required_claims_supported: int = 0
    required_claims_total: int = 0
    optional_claims_supported: int = 0
    optional_claims_total: int = 0
    missing_required_claims: list[str] = Field(default_factory=list)
    missing_optional_claims: list[str] = Field(default_factory=list)
    prompt_note: str | None = None

    model_config = ConfigDict(extra="forbid")


class AuditReport(BaseModel):
    request: AuditReportRequest
    status: ReportStatus
    summary: str = Field(min_length=1)
    scorecard: list[ScorecardDimension] = Field(min_length=1)
    selected_sources: list[ReportSource] = Field(min_length=1)
    extracted_facts: list[EvidenceItem] = Field(min_length=1)
    smoke_test: SmokeTestResult
    warnings: list[EvidenceItem] = Field(min_length=1)
    suggested_fixes: list[EvidenceItem] = Field(min_length=1)
    metadata: AuditReportMetadata

    model_config = ConfigDict(extra="forbid")


class AuditReportResult(BaseModel):
    status: ReportStatus
    report: AuditReport | None = None
    cache_key: str
    warnings: list[str] = Field(default_factory=list)
    artifact_path: str | None = None


class AuditEngineClient(Protocol):
    async def generate_report(
        self, request: AuditReportRequest
    ) -> AuditReportResult: ...


class CodexAuditEngineClient:
    def __init__(
        self,
        settings: Settings,
        *,
        codex_client: CodexAppServerClient,
        source_client: FirecrawlIngestionClient,
        artifact_root: Path | None = None,
    ) -> None:
        self._settings = settings
        self._codex_client = codex_client
        self._source_client = source_client
        self._artifact_root = artifact_root or (
            settings.project_root / ".agent" / "cache" / "audit-reports"
        )

    async def generate_report(self, request: AuditReportRequest) -> AuditReportResult:
        if not request.cache_key:
            return AuditReportResult(
                status="blocked",
                cache_key="",
                warnings=["Source cache key is required before report generation."],
            )
        try:
            cached = await self._source_client.read_cached_fetched_sources(
                request.cache_key
            )
        except FirecrawlCacheNotFound:
            return AuditReportResult(
                status="blocked",
                cache_key=request.cache_key,
                warnings=["Cached source fetch not found."],
            )
        if not cached.fetched_sources or not cached.result.selected_sources:
            return AuditReportResult(
                status="blocked",
                cache_key=request.cache_key,
                warnings=["Cached source fetch contains no selected sources."],
            )
        if not self._is_safe_report_cache_key(cached.cache_key):
            return AuditReportResult(
                status="blocked",
                cache_key=request.cache_key,
                warnings=["Cached source fetch returned an unsafe cache key."],
            )
        if cached.cache_key != request.cache_key:
            return AuditReportResult(
                status="blocked",
                cache_key=request.cache_key,
                warnings=["Cached source fetch returned a mismatched cache key."],
            )
        effective_request = request.model_copy(update={"cache_key": cached.cache_key})

        prompt, prompt_warnings, evidence_source_ids, prompt_summary = (
            self._build_prompt(
                effective_request,
                cached,
            )
        )
        codex_result = await self._codex_client.run_structured_json(
            CodexStructuredJsonRequest(
                base_instructions=(
                    "You are auditing public developer documentation for whether "
                    "an AI coding agent can complete one integration goal. Return "
                    "only JSON matching the supplied schema."
                ),
                prompt=prompt,
                output_schema=self._codex_output_schema(),
                timeout_seconds=self._settings.codex_app_server_timeout_seconds,
            )
        )
        if codex_result.status != "pass" or codex_result.response is None:
            return AuditReportResult(
                status="blocked",
                cache_key=request.cache_key,
                warnings=[codex_result.message],
            )

        report, normalization_warnings = self._normalize_report(
            effective_request,
            cached,
            codex_result.response,
            prompt_warnings,
            evidence_source_ids,
            prompt_summary,
        )
        artifact_path = self._write_artifacts(report)
        return AuditReportResult(
            status=report.status,
            report=report,
            cache_key=effective_request.cache_key,
            warnings=normalization_warnings,
            artifact_path=artifact_path,
        )

    def _build_prompt(
        self,
        request: AuditReportRequest,
        cached: CachedFetchedSources,
    ) -> tuple[str, list[str], set[str], PromptEvidenceSummary]:
        warnings: list[str] = []
        safe_request = request.model_dump(mode="json", exclude_none=True)
        facets = self._meaningful_goal_tokens(
            request.integration_goal, request.docs_url
        )
        required_slots = self._required_evidence_slots(request.integration_goal)
        ranked_sources = self._rank_evidence_sources(request, cached.fetched_sources)
        claim_requirements = self._claim_requirements(
            request.integration_goal,
            facets,
            required_slots,
        )
        ledger_entries = self._build_evidence_ledger(
            ranked_sources,
            facets,
            claim_requirements,
        )
        packed_sources, packed_ledger_entries = self._pack_ledger_evidence_sources(
            ledger_entries,
            ranked_sources,
            cached.planner_metadata,
        )
        prompt_ledger_entries = list(packed_ledger_entries)
        prompt_package = {
            "audit_request": safe_request,
            "allowed_scorecard_ids": [item[0] for item in SCORECARD_DIMENSIONS],
            "allowed_basis_values": ["source_backed", "inferred", "uncertain"],
            "claim_checklist": [
                {
                    "id": claim.id,
                    "label": claim.label,
                    "description": claim.description,
                    "evidence_slots": sorted(claim.evidence_slots),
                    "required_for_pass": claim.required_for_pass,
                }
                for claim in claim_requirements
            ],
            "claim_evidence_summary": [],
            "evidence_ledger": [],
            "required_evidence_slots": sorted(required_slots),
            "covered_evidence_slots": [],
            "missing_evidence_slots": [],
            "instructions": [
                "Treat sources as a bounded evidence packet, not the full docs corpus.",
                "Treat claim_evidence_summary and evidence_ledger as the proof record for this audit.",
                "Use sources only as supporting excerpt context for ledger entries.",
                "A claim with status='supported' is present in fetched evidence; do not report it as missing.",
                "A claim with status='not_found_in_fetched_sources' was searched in fetched sources but not supported.",
                "Required missing claims may block completion; optional missing claims are improvement areas.",
                "Do not say the full docs lack a fact unless the claim status says not_found_in_fetched_sources.",
                "When a relevant claim is unsupported, say the selected evidence does not show the fact.",
                "Cite only selected source ids in source_refs.",
                "Prefer canonical same-language docs for source_refs when they support the claim.",
                "Use basis='source_backed' only when the claim is supported by cited excerpts.",
                "Use basis='inferred' or basis='uncertain' for claims without direct source support.",
                "If every required claim is supported, the smoke_test.result should usually be pass.",
                "Return concise analysis for the requested integration goal.",
                "Keep summary and rationale fields under 500 characters.",
                "Return at most 5 extracted facts, 5 warnings, and 5 suggested fixes.",
            ],
            "evidence_limits": {
                "fetched_source_count": len(cached.fetched_sources),
                "max_prompt_bytes": MAX_PROMPT_BYTES,
                "max_evidence_sources": MAX_EVIDENCE_SOURCES,
                "max_snippets_per_source": MAX_SNIPPETS_PER_SOURCE,
                "max_snippet_chars": MAX_SNIPPET_CHARS,
                "max_ledger_entries_per_claim": MAX_LEDGER_ENTRIES_PER_CLAIM,
            },
            "sources": self._source_headers_for_prompt(packed_sources),
        }
        claim_statuses = self._refresh_prompt_claim_evidence(
            prompt_package,
            claim_requirements,
            prompt_ledger_entries,
            required_slots,
        )
        prompt = json.dumps(prompt_package, indent=2)
        while len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES and prompt_ledger_entries:
            prompt_ledger_entries.pop()
            claim_statuses = self._refresh_prompt_claim_evidence(
                prompt_package,
                claim_requirements,
                prompt_ledger_entries,
                required_slots,
            )
            prompt = json.dumps(prompt_package, indent=2)
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            prompt_ledger_entries = []
            claim_statuses = self._refresh_prompt_claim_evidence(
                prompt_package,
                claim_requirements,
                prompt_ledger_entries,
                required_slots,
            )
            prompt = json.dumps(prompt_package, indent=2)
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            warnings.append("Evidence packet exceeded the Codex prompt budget.")
        evidence_source_ids = {str(entry.source.id) for entry in prompt_ledger_entries}
        covered_slots = list(prompt_package["covered_evidence_slots"])
        missing_slots = list(prompt_package["missing_evidence_slots"])
        claim_counts = self._claim_counts(claim_statuses)
        supported_claims = [
            str(status["id"])
            for status in claim_statuses
            if status["status"] == "supported"
        ]
        missing_claims = [
            str(status["id"])
            for status in claim_statuses
            if status["status"] == "not_found_in_fetched_sources"
        ]
        represented_terms = sorted(
            {
                term
                for source in ranked_sources
                if str(source.id) in evidence_source_ids
                for term in self._matched_facets(source, facets)
            }
        )
        missing_terms = sorted(facets - set(represented_terms))
        if missing_terms:
            warnings.append(
                "Prompt evidence omitted goal term(s): "
                + ", ".join(missing_terms[:8])
                + "."
            )
        if missing_slots:
            warnings.append(
                "Prompt evidence omitted required evidence slot(s): "
                + ", ".join(missing_slots[:8])
                + "."
            )
        summary = PromptEvidenceSummary(
            fetched_source_count=len(cached.fetched_sources),
            prompt_source_count=len(evidence_source_ids),
            omitted_source_count=max(
                0, len(cached.fetched_sources) - len(evidence_source_ids)
            ),
            represented_goal_terms=represented_terms,
            missing_goal_terms=missing_terms,
            covered_evidence_slots=covered_slots,
            missing_evidence_slots=missing_slots,
            supported_claims=supported_claims,
            missing_claims=missing_claims,
            required_claims_supported=claim_counts["required_supported"],
            required_claims_total=claim_counts["required_total"],
            optional_claims_supported=claim_counts["optional_supported"],
            optional_claims_total=claim_counts["optional_total"],
            missing_required_claims=[
                str(status["id"])
                for status in claim_statuses
                if status["required_for_pass"]
                and status["status"] == "not_found_in_fetched_sources"
            ],
            missing_optional_claims=[
                str(status["id"])
                for status in claim_statuses
                if not status["required_for_pass"]
                and status["status"] == "not_found_in_fetched_sources"
            ],
            note=(
                f"Codex prompt used a claim evidence ledger from "
                f"{len(evidence_source_ids)} of {len(cached.fetched_sources)} "
                "fetched sources."
            ),
        )
        return prompt, warnings, evidence_source_ids, summary

    def _rank_evidence_sources(
        self,
        request: AuditReportRequest,
        sources: list[Any],
    ) -> list[Any]:
        goal_tokens = self._meaningful_goal_tokens(
            request.integration_goal, request.docs_url
        )
        priority_terms = self._priority_terms(request.integration_goal)
        plans = [
            EvidencePlan(
                source=source,
                score=self._evidence_source_score(source, goal_tokens, priority_terms),
                facets=self._matched_facets(source, goal_tokens),
                canonical_key=self._canonical_source_key(source.url),
            )
            for source in sources
        ]
        selected: list[EvidencePlan] = []
        selected_ids: set[str] = set()
        selected_keys: set[str] = set()
        available_facets = sorted({facet for plan in plans for facet in plan.facets})

        for facet in available_facets:
            candidates = [
                plan
                for plan in plans
                if facet in plan.facets and str(plan.source.id) not in selected_ids
            ]
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda plan: self._coverage_sort_key(plan, selected_keys),
            )
            selected.append(best)
            selected_ids.add(str(best.source.id))
            selected_keys.add(best.canonical_key)

        for plan in sorted(
            plans,
            key=lambda item: self._coverage_sort_key(item, selected_keys),
            reverse=True,
        ):
            if str(plan.source.id) in selected_ids:
                continue
            selected.append(plan)
            selected_ids.add(str(plan.source.id))
            selected_keys.add(plan.canonical_key)

        return [plan.source for plan in selected]

    def _pack_evidence_sources(
        self,
        ranked_sources: list[Any],
        facets: set[str],
        required_slots: set[str],
        planner_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        source_payloads: dict[str, dict[str, Any]] = {}
        snippets_by_source: dict[str, list[dict[str, Any]]] = {}
        total_snippet_chars = 0
        covered_slots: set[str] = set()
        selected_identities: set[tuple[str, str]] = set()

        snippet_plans = self._rank_snippet_plans(ranked_sources, facets)
        for slot in sorted(required_slots):
            candidates = [
                plan
                for plan in snippet_plans
                if slot in plan.slots
                and self._snippet_identity(plan) not in selected_identities
                and self._can_add_snippet(
                    plan,
                    source_payloads,
                    snippets_by_source,
                    total_snippet_chars,
                )
            ]
            best = self._best_slot_plan(
                candidates,
                covered_slots,
            )
            if best is None:
                continue
            total_snippet_chars += self._add_snippet_plan(
                best,
                source_payloads,
                snippets_by_source,
                planner_metadata,
            )
            selected_identities.add(self._snippet_identity(best))
            covered_slots.update(best.slots)

        for plan in snippet_plans:
            if not self._can_add_snippet(
                plan,
                source_payloads,
                snippets_by_source,
                total_snippet_chars,
            ):
                continue
            if self._snippet_identity(plan) in selected_identities:
                continue
            total_snippet_chars += self._add_snippet_plan(
                plan,
                source_payloads,
                snippets_by_source,
                planner_metadata,
            )
            selected_identities.add(self._snippet_identity(plan))
            if (
                len(source_payloads) >= MAX_EVIDENCE_SOURCES
                and all(
                    len(snippets) >= MAX_SNIPPETS_PER_SOURCE
                    for snippets in snippets_by_source.values()
                )
            ):
                break

        return [
            source_payloads[str(source.id)]
            for source in ranked_sources
            if str(source.id) in source_payloads
        ]

    def _build_evidence_ledger(
        self,
        ranked_sources: list[Any],
        facets: set[str],
        claims: list[ClaimRequirement],
    ) -> list[EvidenceLedgerEntry]:
        snippet_plans = self._rank_snippet_plans(ranked_sources, facets)
        ledger: list[EvidenceLedgerEntry] = []
        total_chars = 0
        used_identities: set[tuple[str, str, str]] = set()
        for claim in claims:
            candidates = [
                (self._claim_evidence_score(plan, claim), plan)
                for plan in snippet_plans
            ]
            ranked_candidates = sorted(
                [(score, plan) for score, plan in candidates if score > 0],
                key=lambda item: (
                    item[0],
                    self._source_quality_score(item[1].source),
                    -self._source_order(item[1].source.id),
                ),
                reverse=True,
            )
            selected_for_claim = 0
            for score, plan in ranked_candidates:
                if selected_for_claim >= MAX_LEDGER_ENTRIES_PER_CLAIM:
                    break
                identity = (claim.id, str(plan.source.id), plan.snippet)
                if identity in used_identities:
                    continue
                snippet = self._bounded_snippet(
                    plan.snippet,
                    claim.keywords | facets,
                    MAX_LEDGER_ENTRY_CHARS,
                )
                if not snippet:
                    continue
                if total_chars + len(snippet) > MAX_LEDGER_TOTAL_CHARS:
                    break
                used_identities.add(identity)
                selected_for_claim += 1
                total_chars += len(snippet)
                ledger.append(
                    EvidenceLedgerEntry(
                        claim_id=claim.id,
                        source=plan.source,
                        snippet=snippet,
                        evidence_slots=plan.slots,
                        score=score,
                        confidence=self._claim_confidence(score),
                        reason=self._claim_evidence_reason(plan, claim),
                    )
                )
        return ledger

    def _pack_ledger_evidence_sources(
        self,
        ledger_entries: list[EvidenceLedgerEntry],
        ranked_sources: list[Any],
        planner_metadata: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[EvidenceLedgerEntry]]:
        source_payloads: dict[str, dict[str, Any]] = {}
        snippets_by_source: dict[str, list[dict[str, Any]]] = {}
        packed_entries: list[EvidenceLedgerEntry] = []
        seen_snippets: set[tuple[str, str]] = set()
        for entry in ledger_entries:
            source_id = str(entry.source.id)
            source_is_new = source_id not in source_payloads
            if source_is_new and len(source_payloads) >= MAX_EVIDENCE_SOURCES:
                continue
            if len(snippets_by_source.get(source_id, [])) >= MAX_SNIPPETS_PER_SOURCE:
                continue
            if source_id not in source_payloads:
                snippets_by_source[source_id] = []
                source_payloads[source_id] = {
                    "id": entry.source.id,
                    "url": entry.source.url,
                    "title": entry.source.title,
                    "reason_selected": entry.source.reason_selected,
                    "retrieved_via": entry.source.retrieved_via,
                    "evidence_roles": self._planner_source_roles(
                        entry.source,
                        planner_metadata,
                    ),
                    "source_flags": self._prompt_source_flags(entry.source.url),
                    "snippets": snippets_by_source[source_id],
                }
            snippet_identity = (source_id, entry.snippet)
            if snippet_identity not in seen_snippets:
                snippets_by_source[source_id].append(
                    {
                        "text": entry.snippet,
                        "claim_ids": [entry.claim_id],
                        "evidence_slots": sorted(entry.evidence_slots),
                    }
                )
                seen_snippets.add(snippet_identity)
            else:
                for snippet in snippets_by_source[source_id]:
                    if snippet.get("text") == entry.snippet:
                        claim_ids = snippet.setdefault("claim_ids", [])
                        if isinstance(claim_ids, list) and entry.claim_id not in claim_ids:
                            claim_ids.append(entry.claim_id)
                        break
            packed_entries.append(entry)
        if not packed_entries:
            fallback = self._pack_evidence_sources(
                ranked_sources,
                set(),
                set(),
                planner_metadata,
            )
            return fallback, []
        return (
            [
                source_payloads[str(source.id)]
                for source in ranked_sources
                if str(source.id) in source_payloads
            ],
            packed_entries,
        )

    @staticmethod
    def _ledger_prompt_payload(
        ledger_entries: list[EvidenceLedgerEntry],
        claims: list[ClaimRequirement],
    ) -> list[dict[str, Any]]:
        return [
            {
                "claim_id": entry.claim_id,
                "supported_claim_ids": [
                    claim.id
                    for claim in claims
                    if CodexAuditEngineClient._entry_supports_claim(entry, claim)
                ],
                "source_id": entry.source.id,
                "title": entry.source.title,
                "url": entry.source.url,
                "confidence": entry.confidence,
                "reason": entry.reason,
                "evidence_slots": sorted(entry.evidence_slots),
                "snippet": entry.snippet,
            }
            for entry in ledger_entries
        ]

    @classmethod
    def _refresh_prompt_claim_evidence(
        cls,
        prompt_package: dict[str, Any],
        claims: list[ClaimRequirement],
        ledger_entries: list[EvidenceLedgerEntry],
        required_slots: set[str],
    ) -> list[dict[str, Any]]:
        claim_statuses = cls._reconciled_claim_statuses(claims, ledger_entries)
        covered_slots = sorted(
            {
                slot
                for entry in ledger_entries
                for slot in entry.evidence_slots
            }
        )
        prompt_package["claim_evidence_summary"] = claim_statuses
        prompt_package["evidence_ledger"] = cls._ledger_prompt_payload(
            ledger_entries,
            claims,
        )
        prompt_package["covered_evidence_slots"] = covered_slots
        prompt_package["missing_evidence_slots"] = sorted(
            required_slots - set(covered_slots)
        )
        return claim_statuses

    @staticmethod
    def _source_headers_for_prompt(
        packed_sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": source.get("id"),
                "url": source.get("url"),
                "title": source.get("title"),
                "reason_selected": source.get("reason_selected"),
                "retrieved_via": source.get("retrieved_via"),
                "evidence_roles": source.get("evidence_roles", []),
                "source_flags": source.get("source_flags", {}),
            }
            for source in packed_sources
        ]

    @classmethod
    def _reconciled_claim_statuses(
        cls,
        claims: list[ClaimRequirement],
        ledger_entries: list[EvidenceLedgerEntry],
    ) -> list[dict[str, Any]]:
        entries_by_claim: dict[str, list[EvidenceLedgerEntry]] = {}
        for claim in claims:
            for entry in ledger_entries:
                if cls._entry_supports_claim(entry, claim):
                    entries_by_claim.setdefault(claim.id, []).append(entry)
        statuses: list[dict[str, Any]] = []
        for claim in claims:
            entries = entries_by_claim.get(claim.id, [])
            statuses.append(
                {
                    "id": claim.id,
                    "label": claim.label,
                    "required_for_pass": claim.required_for_pass,
                    "status": "supported"
                    if entries
                    else "not_found_in_fetched_sources",
                    "evidence_count": len(entries),
                    "source_ids": sorted({str(entry.source.id) for entry in entries}),
                }
            )
        return statuses

    @staticmethod
    def _claim_counts(claim_statuses: list[dict[str, Any]]) -> dict[str, int]:
        required = [
            status
            for status in claim_statuses
            if bool(status.get("required_for_pass"))
        ]
        optional = [
            status
            for status in claim_statuses
            if not bool(status.get("required_for_pass"))
        ]
        return {
            "required_total": len(required),
            "required_supported": sum(
                1 for status in required if status.get("status") == "supported"
            ),
            "optional_total": len(optional),
            "optional_supported": sum(
                1 for status in optional if status.get("status") == "supported"
            ),
        }

    @staticmethod
    def _entry_supports_claim(
        entry: EvidenceLedgerEntry,
        claim: ClaimRequirement,
    ) -> bool:
        if entry.evidence_slots & claim.evidence_slots:
            return True
        lowered = entry.snippet.lower()
        if claim.id == "output_format":
            return any(
                pattern.search(entry.snippet)
                for pattern in (
                    re.compile(r"\bformats?\s*[:=]\s*\[?", re.IGNORECASE),
                    re.compile(r"\b(?:result|response|data|page|doc)\.markdown\b", re.IGNORECASE),
                    re.compile(r"\b(?:json|html|markdown)\s+(?:response|output|field|format)\b", re.IGNORECASE),
                    re.compile(r"\bcontent[-_ ]?type\b", re.IGNORECASE),
                    re.compile(r"\boutput schema\b", re.IGNORECASE),
                )
            )
        keyword_hits = sum(1 for keyword in claim.keywords if keyword in lowered)
        if claim.evidence_slots:
            return keyword_hits >= 2
        return keyword_hits >= 1

    @staticmethod
    def _claim_evidence_score(
        plan: EvidenceSnippetPlan,
        claim: ClaimRequirement,
    ) -> int:
        lowered = plan.snippet.lower()
        slot_overlap = len(plan.slots & claim.evidence_slots)
        keyword_hits = sum(1 for keyword in claim.keywords if keyword in lowered)
        if slot_overlap == 0 and keyword_hits == 0:
            return 0
        command_density = sum(
            1 for pattern in COMMAND_OR_CONFIG_PATTERNS if pattern.search(plan.snippet)
        )
        role_text = str(getattr(plan.source, "reason_selected", "")).lower()
        role_boost = 0
        if claim.id == "sdk_install" and (
            "planner role(s): setup" in role_text
            or "planner role(s): install" in role_text
            or "install" in role_text
        ):
            role_boost = 45
        elif claim.id in {"search_request", "scrape_request", "output_format"} and (
            claim.label.lower().split()[0] in role_text
        ):
            role_boost = 30
        return (
            plan.score // 3
            + slot_overlap * 70
            + keyword_hits * 35
            + command_density * 8
            + role_boost
        )

    @staticmethod
    def _claim_confidence(score: int) -> Literal["high", "medium", "low"]:
        if score >= 120:
            return "high"
        if score >= 70:
            return "medium"
        return "low"

    @staticmethod
    def _claim_evidence_reason(
        plan: EvidenceSnippetPlan,
        claim: ClaimRequirement,
    ) -> str:
        matched_slots = sorted(plan.slots & claim.evidence_slots)
        if matched_slots:
            return "Matched evidence slot(s): " + ", ".join(matched_slots) + "."
        matched_keywords = sorted(
            keyword for keyword in claim.keywords if keyword in plan.snippet.lower()
        )
        if matched_keywords:
            return "Matched claim keyword(s): " + ", ".join(matched_keywords[:4]) + "."
        return "Selected by source relevance."

    def _rank_snippet_plans(
        self,
        ranked_sources: list[Any],
        facets: set[str],
    ) -> list[EvidenceSnippetPlan]:
        plans: list[EvidenceSnippetPlan] = []
        for source in ranked_sources:
            sections = self._clean_markdown_sections(source.markdown)
            if not sections and source.markdown.strip():
                sections = [self._clean_snippet(source.markdown)]
            for index, section in enumerate(sections):
                snippet = self._bounded_snippet(
                    section,
                    facets,
                    MAX_SNIPPET_CHARS,
                )
                if not snippet:
                    continue
                slots = self._section_evidence_slots(section)
                score = self._snippet_score(section, source, facets)
                score += len(slots) * 25
                if self._is_chrome_section(section):
                    score -= 100
                plans.append(
                    EvidenceSnippetPlan(
                        source=source,
                        index=index,
                        score=score,
                        slots=slots,
                        snippet=snippet,
                    )
                )
        return sorted(
            plans,
            key=lambda plan: (
                plan.score,
                len(plan.slots),
                self._source_quality_score(plan.source),
                -self._source_order(plan.source.id),
                -plan.index,
            ),
            reverse=True,
        )

    def _can_add_snippet(
        self,
        plan: EvidenceSnippetPlan,
        source_payloads: dict[str, dict[str, Any]],
        snippets_by_source: dict[str, list[dict[str, Any]]],
        total_snippet_chars: int,
    ) -> bool:
        source_id = str(plan.source.id)
        if (
            source_id not in source_payloads
            and len(source_payloads) >= MAX_EVIDENCE_SOURCES
        ):
            return False
        if len(snippets_by_source.get(source_id, [])) >= MAX_SNIPPETS_PER_SOURCE:
            return False
        return total_snippet_chars + len(plan.snippet) <= MAX_TOTAL_SNIPPET_CHARS

    @staticmethod
    def _best_slot_plan(
        candidates: list[EvidenceSnippetPlan],
        covered_slots: set[str],
    ) -> EvidenceSnippetPlan | None:
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda plan: (
                len(plan.slots - covered_slots),
                plan.score,
                CodexAuditEngineClient._source_quality_score(plan.source),
                -CodexAuditEngineClient._source_order(plan.source.id),
                -plan.index,
            ),
        )

    def _add_snippet_plan(
        self,
        plan: EvidenceSnippetPlan,
        source_payloads: dict[str, dict[str, Any]],
        snippets_by_source: dict[str, list[dict[str, Any]]],
        planner_metadata: dict[str, Any],
    ) -> int:
        source_id = str(plan.source.id)
        if source_id not in source_payloads:
            snippets_by_source[source_id] = []
            source_payloads[source_id] = {
                "id": plan.source.id,
                "url": plan.source.url,
                "title": plan.source.title,
                "reason_selected": plan.source.reason_selected,
                "retrieved_via": plan.source.retrieved_via,
                "evidence_roles": self._planner_source_roles(
                    plan.source,
                    planner_metadata,
                ),
                "source_flags": self._prompt_source_flags(plan.source.url),
                "snippets": snippets_by_source[source_id],
            }
        snippets_by_source[source_id].append(
            {
                "text": plan.snippet,
                "evidence_slots": sorted(plan.slots),
            }
        )
        return len(plan.snippet)

    @staticmethod
    def _snippet_identity(plan: EvidenceSnippetPlan) -> tuple[str, str]:
        return str(plan.source.id), plan.snippet

    @staticmethod
    def _section_evidence_slots(section: str) -> set[str]:
        return {
            slot
            for slot, patterns in EVIDENCE_SLOT_PATTERNS.items()
            if any(pattern.search(section) for pattern in patterns)
        }

    def _source_snippets(
        self,
        source: Any,
        facets: set[str],
        remaining_budget: int,
    ) -> list[dict[str, Any]]:
        snippets: list[str] = []
        sections = self._clean_markdown_sections(source.markdown)
        if not sections and source.markdown.strip():
            sections = [self._clean_snippet(source.markdown)]
        scored_sections = [
            (
                index,
                self._snippet_score(section, source, facets),
                self._bounded_snippet(section, facets, remaining_budget),
            )
            for index, section in enumerate(sections)
        ]
        ranked_sections = sorted(
            [item for item in scored_sections if item[2]],
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
        selected = sorted(
            ranked_sections[:MAX_SNIPPETS_PER_SOURCE], key=lambda item: item[0]
        )
        for _, _, snippet in selected:
            if remaining_budget <= 0:
                break
            chunk = snippet[: min(MAX_SNIPPET_CHARS, remaining_budget)].strip()
            if not chunk:
                continue
            snippets.append(chunk)
            remaining_budget -= len(chunk)
        return [
            {
                "text": snippet,
                "evidence_slots": sorted(self._section_evidence_slots(snippet)),
            }
            for snippet in snippets
        ]

    @staticmethod
    def _planner_source_roles(
        source: Any,
        planner_metadata: dict[str, Any],
    ) -> list[str]:
        source_roles = planner_metadata.get("source_roles")
        if isinstance(source_roles, dict):
            roles = source_roles.get(str(source.id))
            if isinstance(roles, list):
                return [role for role in roles if isinstance(role, str)]
        roles_by_url = planner_metadata.get("roles_by_url")
        if isinstance(roles_by_url, dict):
            roles = roles_by_url.get(str(source.url))
            if isinstance(roles, list):
                return [role for role in roles if isinstance(role, str)]
        return []

    @staticmethod
    def _prompt_source_flags(url: str) -> dict[str, bool]:
        parsed = urlparse(url)
        path = parsed.path.lower()
        segments = [segment for segment in path.split("/") if segment]
        return {
            "legacy": bool(set(segments) & LEGACY_PATH_SEGMENTS),
            "localized": bool(segments and LOCALE_SEGMENT.fullmatch(segments[0])),
            "quickstart": bool({"quickstart", "quickstarts"} & set(segments)),
            "sdk": bool({"sdk", "sdks"} & set(segments)),
            "api_reference": "api-reference" in segments
            or bool({"api", "reference"} & set(segments)),
            "canonical": "source-of-truth" in path
            or bool({"quickstart", "quickstarts", "sdk", "sdks"} & set(segments)),
            "source_like": path.endswith((".md", ".mdx", "llms.txt")),
        }

    @staticmethod
    def _drop_lowest_priority_snippet(packed_sources: list[dict[str, Any]]) -> bool:
        for source in reversed(packed_sources):
            snippets = source.get("snippets")
            if isinstance(snippets, list) and snippets:
                snippets.pop()
                if not snippets:
                    packed_sources.remove(source)
                return True
        return False

    @staticmethod
    def _meaningful_goal_tokens(goal: str, docs_url: str) -> set[str]:
        host_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", docs_url.lower())
            if len(token) > 2
        }
        return {
            token
            for token in re.findall(r"[a-z0-9]+", goal.lower())
            if len(token) > 2
            and token not in GOAL_STOPWORDS
            and token not in host_tokens
        }

    @staticmethod
    def _claim_requirements(
        goal: str,
        facets: set[str],
        required_slots: set[str],
    ) -> list[ClaimRequirement]:
        definitions = {
            "setup_install": ClaimRequirement(
                id="sdk_install",
                label="SDK install",
                description="The docs show how to install the required package or SDK.",
                evidence_slots={"setup_install"},
                keywords={"install", "sdk", "package", "pip", "npm", "pnpm", "yarn"},
                required_for_pass=True,
            ),
            "auth_secret": ClaimRequirement(
                id="api_key_setup",
                label="API key setup",
                description="The docs show how credentials or API keys are provided.",
                evidence_slots={"auth_secret"},
                keywords={"api key", "apikey", "credential", "secret", "env", "environment"},
                required_for_pass=True,
            ),
            "client_initialization": ClaimRequirement(
                id="client_initialization",
                label="Client initialization",
                description="The docs show how to initialize the SDK or client.",
                evidence_slots={"client_initialization"},
                keywords={"client", "initialize", "initialization", "from ", "import"},
                required_for_pass=True,
            ),
            "search": ClaimRequirement(
                id="search_request",
                label="Search request",
                description="The docs show how to run the requested search/query operation.",
                evidence_slots={"search"},
                keywords={"search", "query"},
                required_for_pass=True,
            ),
            "result_selection": ClaimRequirement(
                id="result_selection",
                label="Result selection",
                description="The docs show how an agent can select the requested result URL or item.",
                evidence_slots={"result_selection"},
                keywords={"result", "results", "url", "link", "top", "first"},
                required_for_pass=True,
            ),
            "scrape": ClaimRequirement(
                id="scrape_request",
                label="Scrape request",
                description="The docs show how to run the requested scrape/crawl operation.",
                evidence_slots={"scrape"},
                keywords={"scrape", "crawl"},
                required_for_pass=True,
            ),
            "output_format": ClaimRequirement(
                id="output_format",
                label="Output format",
                description="The docs show how to request or read the requested output format.",
                evidence_slots={"output_format"},
                keywords={"markdown", "html", "json", "format", "formats", "output"},
                required_for_pass=True,
            ),
            "configuration_options": ClaimRequirement(
                id="request_options",
                label="Request options",
                description="The docs explain relevant request options or parameters.",
                evidence_slots={"configuration_options"},
                keywords={"option", "options", "parameter", "timeout", "limit", "headers"},
                required_for_pass="option" in goal.lower()
                or "parameter" in goal.lower()
                or "production" in goal.lower(),
            ),
            "error_handling": ClaimRequirement(
                id="error_handling",
                label="Error handling",
                description="The docs explain common failures, retries, or error responses.",
                evidence_slots={"error_handling"},
                keywords={"error", "exception", "retry", "retries", "timeout", "429"},
                required_for_pass=False,
            ),
            "production_safety": ClaimRequirement(
                id="production_safety",
                label="Production safety",
                description="The docs explain production concerns such as rate limits, caching, or secret handling.",
                evidence_slots={"production_safety"},
                keywords={"production", "rate limit", "cache", "secret", "timeout", "retry"},
                required_for_pass=False,
            ),
        }
        claim_order = [
            "setup_install",
            "auth_secret",
            "client_initialization",
            "search",
            "result_selection",
            "scrape",
            "output_format",
            "configuration_options",
            "production_safety",
            "error_handling",
        ]
        claims = [
            definitions[slot]
            for slot in claim_order
            if slot in required_slots and slot in definitions
        ]
        covered_keywords = {
            keyword
            for claim in claims
            for keyword in claim.keywords
        }
        facet_claims = [
            ClaimRequirement(
                id="goal_" + re.sub(r"[^a-z0-9]+", "_", facet.lower()).strip("_"),
                label=facet,
                description=(
                    "The fetched docs include evidence for the user-mentioned "
                    f"goal term '{facet}'."
                ),
                evidence_slots=set(),
                keywords={"secure", "security"} if facet == "secure" else {facet},
                required_for_pass=False,
            )
            for facet in sorted(facets)
            if facet not in covered_keywords
            and facet
            not in {
                "agent",
                "api",
                "docs",
                "explain",
                "install",
                "key",
                "markdown",
                "make",
                "options",
                "production",
                "python",
                "request",
                "result",
                "scrape",
                "sdk",
                "search",
                "top",
                "web",
            }
        ][:10]
        if claims or facet_claims:
            return [*claims, *facet_claims]
        return [
            ClaimRequirement(
                id="goal_fit",
                label="Goal fit",
                description="The fetched docs contain evidence relevant to the requested integration goal.",
                evidence_slots=set(),
                keywords=facets,
                required_for_pass=True,
            )
        ]

    @staticmethod
    def _required_evidence_slots(goal: str) -> set[str]:
        lowered = goal.lower()
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", lowered)
            if len(token) > 2 and token not in GOAL_STOPWORDS
        }
        slots: set[str] = set()
        if "set up" in lowered or tokens & {"setup", "install", "sdk", "package"}:
            slots.update({"setup_install", "auth_secret", "client_initialization"})
        if tokens & {"api", "key", "auth", "credential", "secret"}:
            slots.add("auth_secret")
        if tokens & {"search", "query"}:
            slots.add("search")
        if tokens & {"top", "first", "result", "url", "link"}:
            slots.add("result_selection")
        if tokens & {"scrape", "scraping", "crawl", "crawling"}:
            slots.add("scrape")
        if tokens & {"markdown", "html", "json", "format", "output"}:
            slots.add("output_format")
        if tokens & {"options", "configuration", "config", "parameters"}:
            slots.add("configuration_options")
        if tokens & {"production", "timeout", "retry", "retries", "rate", "error"}:
            slots.update({"configuration_options", "error_handling", "production_safety"})
        return slots

    @staticmethod
    def _evidence_source_score(
        source: Any,
        goal_tokens: set[str],
        priority_terms: set[str],
    ) -> int:
        locator_haystack = " ".join(
            [source.url, source.title, source.reason_selected]
        ).lower()
        body_haystack = source.markdown[:20_000].lower()
        matched_locator_terms = [
            token for token in goal_tokens if token in locator_haystack
        ]
        matched_body_terms = [
            token
            for token in goal_tokens
            if token not in matched_locator_terms and token in body_haystack
        ]
        matched_priority_terms = [
            token for token in priority_terms if token in locator_haystack
        ]
        score = len(matched_locator_terms) * 24
        score += len(matched_body_terms) * 10
        score += len(matched_priority_terms) * 6
        score += CodexAuditEngineClient._source_quality_score(source)
        score += CodexAuditEngineClient._language_source_score(source, goal_tokens)
        return score

    @staticmethod
    def _matched_facets(source: Any, goal_tokens: set[str]) -> set[str]:
        locator_haystack = " ".join(
            [source.url, source.title, source.reason_selected]
        ).lower()
        body_haystack = source.markdown[:20_000].lower()
        return {
            token
            for token in goal_tokens
            if token in locator_haystack or token in body_haystack
        }

    @staticmethod
    def _coverage_sort_key(
        plan: EvidencePlan, selected_keys: set[str]
    ) -> tuple[int, int, int, int]:
        duplicate_penalty = -18 if plan.canonical_key in selected_keys else 0
        return (
            plan.score + duplicate_penalty,
            len(plan.facets),
            CodexAuditEngineClient._source_quality_score(plan.source),
            -CodexAuditEngineClient._source_order(plan.source.id),
        )

    @staticmethod
    def _source_quality_score(source: Any) -> int:
        parsed = urlparse(source.url)
        path = parsed.path.lower()
        score = 0
        if path.endswith((".md", ".mdx")):
            score += 8
        if "source-of-truth" in path:
            score += 28
        if "quickstart" in path or "quickstarts" in path:
            score += 18
        if "/sdks/" in path or "/sdk/" in path:
            score += 16
        if "api-reference" in path:
            score += 10
        if "llms-full" in path:
            score -= 6
        elif path.endswith("/llms.txt") or path.endswith("llms.txt"):
            score += 4
        segments = [segment for segment in path.split("/") if segment]
        if segments and LOCALE_SEGMENT.fullmatch(segments[0]):
            score -= 10
        if segments and segments[0] in LEGACY_PATH_SEGMENTS:
            score -= 20
        if any(segment in {"legacy", "deprecated"} for segment in segments):
            score -= 20
        first_sections = CodexAuditEngineClient._raw_sections(source.markdown)[:3]
        if first_sections and all(
            CodexAuditEngineClient._is_chrome_section(section)
            for section in first_sections
        ):
            score -= 8
        return score

    @staticmethod
    def _language_source_score(source: Any, goal_tokens: set[str]) -> int:
        requested_languages = CodexAuditEngineClient._requested_languages(goal_tokens)
        if not requested_languages:
            return 0
        source_languages = CodexAuditEngineClient._source_languages(source)
        if not source_languages:
            return 0
        if requested_languages & source_languages:
            return 30
        path = urlparse(source.url).path.lower()
        if "/sdks/" in path or "/sdk/" in path or "quickstart" in path:
            return -24
        return -8

    @staticmethod
    def _requested_languages(goal_tokens: set[str]) -> set[str]:
        requested: set[str] = set()
        for canonical, aliases in LANGUAGE_ALIASES.items():
            if goal_tokens & aliases:
                requested.add(canonical)
        return requested

    @staticmethod
    def _source_languages(source: Any) -> set[str]:
        parsed = urlparse(source.url)
        haystack = " ".join(
            [
                parsed.path.replace("/", " "),
                str(source.title),
                str(source.reason_selected),
            ]
        ).lower()
        tokens = set(re.findall(r"[a-z0-9.+#]+", haystack))
        languages: set[str] = set()
        for canonical, aliases in LANGUAGE_ALIASES.items():
            if tokens & aliases:
                languages.add(canonical)
        return languages

    @staticmethod
    def _canonical_source_key(url: str) -> str:
        parsed = urlparse(url)
        segments = [segment for segment in parsed.path.lower().split("/") if segment]
        if segments and LOCALE_SEGMENT.fullmatch(segments[0]):
            segments = segments[1:]
        if segments and segments[-1] in {"index", "index.md", "index.mdx"}:
            segments = segments[:-1]
        if segments:
            segments[-1] = re.sub(r"\.(md|mdx|html?)$", "", segments[-1])
        return "/".join(segments).rstrip("/") or "/"

    @staticmethod
    def _source_order(source_id: str) -> int:
        match = re.search(r"(\d+)$", source_id)
        return int(match.group(1)) if match else 10_000

    @staticmethod
    def _clean_markdown_sections(markdown: str) -> list[str]:
        cleaned_lines = []
        for line in markdown.splitlines():
            stripped = line.strip()
            if not stripped or CodexAuditEngineClient._is_chrome_line(stripped):
                cleaned_lines.append("")
                continue
            cleaned_lines.append(stripped)
        cleaned = "\n".join(cleaned_lines)
        sections = [
            CodexAuditEngineClient._clean_snippet(section)
            for section in CodexAuditEngineClient._raw_sections(cleaned)
        ]
        return [
            section
            for section in sections
            if section and not CodexAuditEngineClient._is_chrome_section(section)
        ]

    @staticmethod
    def _raw_sections(markdown: str) -> list[str]:
        heading_sections = re.split(r"(?=\n?#{1,6}\s+)", markdown)
        sections = [section.strip() for section in heading_sections if section.strip()]
        if len(sections) <= 1:
            sections = [
                paragraph.strip()
                for paragraph in re.split(r"\n\s*\n", markdown)
                if paragraph.strip()
            ]
        return sections

    @staticmethod
    def _is_chrome_line(line: str) -> bool:
        normalized = re.sub(r"\s+", " ", line.strip())
        if len(normalized) <= 2:
            return True
        return any(pattern.search(normalized) for pattern in CHROME_LINE_PATTERNS)

    @staticmethod
    def _is_chrome_section(section: str) -> bool:
        lines = [
            line.strip("#-*` >").strip()
            for line in section.splitlines()
            if line.strip()
        ]
        if not lines:
            return True
        lowered = section.lower()
        if any(
            marker in lowered
            for marker in [
                "suggest edits",
                "raise issue",
                "chat widget",
                "ctrl+i",
                "ctrl k",
            ]
        ):
            non_link_lines = [
                line
                for line in lines
                if not re.search(r"\[[^\]]+\]\([^)]*\)", line)
                and not CodexAuditEngineClient._is_chrome_line(line)
            ]
            if len(non_link_lines) <= 4:
                return True
        if any(pattern.search(section) for pattern in COMMAND_OR_CONFIG_PATTERNS):
            return False
        chrome_lines = sum(
            1 for line in lines if CodexAuditEngineClient._is_chrome_line(line)
        )
        short_nav_lines = sum(1 for line in lines if len(line.split()) <= 3)
        link_lines = sum(1 for line in lines if re.search(r"\[[^\]]+\]\([^)]*\)", line))
        return chrome_lines == len(lines) or (
            len(lines) >= 3 and short_nav_lines / len(lines) > 0.8
        ) or (
            len(lines) >= 5 and link_lines / len(lines) > 0.65
        ) or (
            len(lines) >= 8 and link_lines / len(lines) > 0.35 and "```" not in section
        )

    @staticmethod
    def _clean_snippet(value: str) -> str:
        lines = [line.rstrip() for line in value.strip().splitlines()]
        compact = "\n".join(line for line in lines if line.strip())
        return re.sub(r"\n{3,}", "\n\n", compact).strip()

    @staticmethod
    def _snippet_score(section: str, source: Any, facets: set[str]) -> int:
        lowered = section.lower()
        title_terms = {
            token
            for token in re.findall(
                r"[a-z0-9]+", f"{source.title} {source.url}".lower()
            )
            if len(token) > 2 and token not in GOAL_STOPWORDS
        }
        score = sum(18 for facet in facets if facet in lowered)
        score += sum(5 for token in title_terms if token in lowered)
        score += sum(
            6 for pattern in COMMAND_OR_CONFIG_PATTERNS if pattern.search(section)
        )
        score += CodexAuditEngineClient._source_quality_score(source) // 2
        score += CodexAuditEngineClient._language_source_score(source, facets)
        if section.lstrip().startswith("#"):
            score += 2
        if len(section) > MAX_SNIPPET_CHARS * 3:
            score -= 4
        return score

    @staticmethod
    def _bounded_snippet(section: str, facets: set[str], remaining_budget: int) -> str:
        if remaining_budget <= 0:
            return ""
        limit = min(MAX_SNIPPET_CHARS, remaining_budget)
        if len(section) <= limit:
            return section.strip()
        lowered = section.lower()
        match_positions = [
            lowered.find(facet)
            for facet in facets
            if facet in lowered and lowered.find(facet) >= 0
        ]
        center = min(match_positions) if match_positions else 0
        start = max(0, center - limit // 3)
        end = min(len(section), start + limit)
        start = max(0, end - limit)
        chunk = section[start:end].strip()
        if start > 0:
            chunk = "..." + chunk
        if end < len(section):
            chunk = chunk + "..."
        return chunk.strip()

    @staticmethod
    def _priority_terms(goal: str) -> set[str]:
        lowered = goal.lower()
        goal_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", lowered)
            if len(token) > 2 and token not in GOAL_STOPWORDS
        }
        if "set up" in lowered or goal_tokens & {
            "approval",
            "auth",
            "configure",
            "configuration",
            "credential",
            "install",
            "secure",
            "security",
            "setup",
            "trusted",
        }:
            return SETUP_SECURITY_TERMS
        return set()

    @staticmethod
    def _codex_output_schema() -> dict[str, Any]:
        evidence_item = {
            "type": "object",
            "additionalProperties": False,
            "required": ["id", "message", "basis", "source_refs", "severity"],
            "properties": {
                "id": {"type": "string", "maxLength": 80},
                "message": {"type": "string", "maxLength": 500},
                "basis": {
                    "type": "string",
                    "enum": ["source_backed", "inferred", "uncertain"],
                },
                "source_refs": {
                    "type": "array",
                    "maxItems": MAX_EVIDENCE_SOURCES,
                    "items": {"type": "string"},
                },
                "severity": {
                    "type": "string",
                    "enum": ["info", "warning", "blocking"],
                },
            },
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "summary",
                "scorecard",
                "extracted_facts",
                "smoke_test",
                "warnings",
                "suggested_fixes",
            ],
            "properties": {
                "summary": {"type": "string", "maxLength": 500},
                "scorecard": {
                    "type": "array",
                    "maxItems": len(SCORECARD_DIMENSIONS),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "id",
                            "label",
                            "score",
                            "max_score",
                            "rationale",
                            "source_refs",
                        ],
                        "properties": {
                            "id": {
                                "type": "string",
                                "enum": [item[0] for item in SCORECARD_DIMENSIONS],
                            },
                            "label": {"type": "string", "maxLength": 80},
                            "score": {"type": "integer"},
                            "max_score": {"type": "integer"},
                            "rationale": {"type": "string", "maxLength": 500},
                            "source_refs": {
                                "type": "array",
                                "maxItems": MAX_EVIDENCE_SOURCES,
                                "items": {"type": "string"},
                            },
                        },
                    },
                },
                "extracted_facts": {
                    "type": "array",
                    "maxItems": 5,
                    "items": evidence_item,
                },
                "smoke_test": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "result",
                        "basis",
                        "message",
                        "source_refs",
                        "missing_facts",
                        "likely_next_steps",
                    ],
                    "properties": {
                        "result": {
                            "type": "string",
                            "enum": ["pass", "partial", "fail"],
                        },
                        "basis": {
                            "type": "string",
                            "enum": ["source_backed", "inferred", "uncertain"],
                        },
                        "message": {"type": "string", "maxLength": 500},
                        "source_refs": {
                            "type": "array",
                            "maxItems": MAX_EVIDENCE_SOURCES,
                            "items": {"type": "string"},
                        },
                        "missing_facts": {
                            "type": "array",
                            "maxItems": 5,
                            "items": {"type": "string"},
                        },
                        "likely_next_steps": {
                            "type": "array",
                            "maxItems": 5,
                            "items": {"type": "string"},
                        },
                    },
                },
                "warnings": {
                    "type": "array",
                    "maxItems": 5,
                    "items": evidence_item,
                },
                "suggested_fixes": {
                    "type": "array",
                    "maxItems": 5,
                    "items": evidence_item,
                },
            },
        }

    def _normalize_report(
        self,
        request: AuditReportRequest,
        cached: CachedFetchedSources,
        response: dict[str, Any],
        prompt_warnings: list[str],
        evidence_source_ids: set[str],
        prompt_summary: PromptEvidenceSummary,
    ) -> tuple[AuditReport, list[str]]:
        warnings = list(prompt_warnings)
        selected_sources = [
            ReportSource(
                id=source.id,
                url=source.url,
                title=source.title,
                reason_selected=source.reason_selected,
                retrieved_via=source.retrieved_via,
            )
            for source in cached.result.selected_sources
        ]
        valid_source_ids = evidence_source_ids
        source_ref_rank = self._source_ref_rank(
            cached.fetched_sources,
            self._meaningful_goal_tokens(request.integration_goal, request.docs_url),
            valid_source_ids,
        )
        scorecard = self._normalize_scorecard(
            response.get("scorecard"), valid_source_ids, warnings, source_ref_rank
        )
        extracted_facts = self._normalize_evidence_list(
            response.get("extracted_facts"),
            "fact",
            valid_source_ids,
            warnings,
            source_ref_rank,
        ) or [
            EvidenceItem(
                id="fact_1",
                message="Codex did not return extracted facts.",
                basis="uncertain",
                severity="warning",
            )
        ]
        warning_items = self._normalize_evidence_list(
            response.get("warnings"),
            "warning",
            valid_source_ids,
            warnings,
            source_ref_rank,
        ) or [
            EvidenceItem(
                id="warning_1",
                message="No major warning was found from the selected sources.",
                basis="inferred",
                severity="info",
            )
        ]
        fix_items = self._normalize_evidence_list(
            response.get("suggested_fixes"),
            "fix",
            valid_source_ids,
            warnings,
            source_ref_rank,
        ) or [
            EvidenceItem(
                id="fix_1",
                message="No specific documentation fix was found from the selected sources.",
                basis="inferred",
                severity="info",
            )
        ]
        smoke_test = self._normalize_smoke_test(
            response.get("smoke_test"),
            valid_source_ids,
            warnings,
            source_ref_rank,
        )
        contradicted_slots = self._contradicted_missing_slots(
            prompt_summary.covered_evidence_slots,
            smoke_test.missing_facts or [],
        )
        if contradicted_slots:
            smoke_test = self._remove_contradicted_missing_facts(
                smoke_test,
                contradicted_slots,
            )
            warning_items = self._remove_contradicted_items(
                warning_items,
                contradicted_slots,
                "warning_1",
                "No major warning was found from the selected sources.",
            )
            fix_items = self._remove_contradicted_items(
                fix_items,
                contradicted_slots,
                "fix_1",
                "No specific documentation fix was found from the selected sources.",
            )
            warnings.append(
                "Removed Codex missing-fact claim(s) contradicted by packed "
                "evidence slot coverage: "
                + ", ".join(sorted(contradicted_slots))
                + "."
            )
        smoke_test, warning_items, fix_items = self._apply_claim_consistency(
            smoke_test,
            warning_items,
            fix_items,
            prompt_summary,
            warnings,
        )
        has_report_warnings = any(
            item.severity in {"warning", "blocking"} for item in warning_items
        )
        status: ReportStatus = (
            "completed_with_warnings"
            if warnings
            or has_report_warnings
            or cached.result.status == "completed_with_warnings"
            else "completed"
        )
        notes = "; ".join(warnings[:3]) if status == "completed_with_warnings" else None
        report = AuditReport(
            request=request,
            status=status,
            summary=str(response.get("summary") or "Documentation audit completed."),
            scorecard=scorecard,
            selected_sources=selected_sources,
            extracted_facts=extracted_facts,
            smoke_test=smoke_test,
            warnings=warning_items,
            suggested_fixes=fix_items,
            metadata=AuditReportMetadata(
                mode=request.mode,
                generated_by="codex_app_server",
                cache_key=request.cache_key,
                notes=notes,
                fetched_source_count=prompt_summary.fetched_source_count,
                prompt_source_count=prompt_summary.prompt_source_count,
                omitted_source_count=prompt_summary.omitted_source_count,
                represented_goal_terms=prompt_summary.represented_goal_terms,
                missing_goal_terms=prompt_summary.missing_goal_terms,
                supported_claims=prompt_summary.supported_claims,
                missing_claims=prompt_summary.missing_claims,
                required_claims_supported=prompt_summary.required_claims_supported,
                required_claims_total=prompt_summary.required_claims_total,
                optional_claims_supported=prompt_summary.optional_claims_supported,
                optional_claims_total=prompt_summary.optional_claims_total,
                missing_required_claims=prompt_summary.missing_required_claims,
                missing_optional_claims=prompt_summary.missing_optional_claims,
                prompt_note=prompt_summary.note,
            ),
        )
        return report, warnings

    @classmethod
    def _apply_claim_consistency(
        cls,
        smoke_test: SmokeTestResult,
        warning_items: list[EvidenceItem],
        fix_items: list[EvidenceItem],
        prompt_summary: PromptEvidenceSummary,
        warnings: list[str],
    ) -> tuple[SmokeTestResult, list[EvidenceItem], list[EvidenceItem]]:
        missing_required = prompt_summary.missing_required_claims
        if not missing_required:
            if smoke_test.result != "pass":
                smoke_test = smoke_test.model_copy(
                    update={
                        "result": "pass",
                        "message": (
                            "All required claims are supported by the selected "
                            "evidence; optional improvement checks may remain."
                        ),
                        "missing_facts": None,
                    }
                )
                warnings.append(
                    "Adjusted smoke test to pass because all required claims "
                    "were supported by reconciled evidence."
                )
            return smoke_test, warning_items, fix_items

        missing_labels = [
            cls._claim_label_from_id(claim_id) for claim_id in missing_required
        ]
        next_missing_facts = list(smoke_test.missing_facts or [])
        for label in missing_labels:
            fact = f"Required claim not found in fetched evidence: {label}."
            if fact not in next_missing_facts:
                next_missing_facts.append(fact)
        if smoke_test.result == "pass":
            smoke_test = smoke_test.model_copy(update={"result": "partial"})
            warnings.append(
                "Adjusted smoke test from pass because required claim(s) were missing."
            )
        smoke_test = smoke_test.model_copy(update={"missing_facts": next_missing_facts})
        if not any(item.severity == "blocking" for item in warning_items):
            warning_items = [
                EvidenceItem(
                    id="missing_required_claims",
                    message=(
                        "Required claim(s) were not found in fetched evidence: "
                        + ", ".join(missing_labels)
                        + "."
                    ),
                    basis="source_backed",
                    severity="blocking",
                ),
                *warning_items,
            ]
        if not any(item.severity in {"blocking", "warning"} for item in fix_items):
            fix_items = [
                EvidenceItem(
                    id="fix_missing_required_claims",
                    message=(
                        "Add or expose documentation for the missing required claim(s): "
                        + ", ".join(missing_labels)
                        + "."
                    ),
                    basis="inferred",
                    severity="warning",
                ),
                *fix_items,
            ]
        return smoke_test, warning_items, fix_items

    @staticmethod
    def _claim_label_from_id(claim_id: str) -> str:
        return claim_id.removeprefix("goal_").replace("_", " ")

    @staticmethod
    def _contradicted_missing_slots(
        covered_slots: list[str],
        missing_facts: list[str],
    ) -> set[str]:
        missing_text = " ".join(missing_facts).lower()
        if not missing_text:
            return set()
        covered = set(covered_slots)
        contradiction_groups = {
            "setup_install": {"install", "installation", "sdk command", "package"},
            "auth_secret": {"auth", "api key", "environment", "env-var", "env var"},
            "client_initialization": {"client", "initialize", "initialization"},
            "search": {"search"},
            "result_selection": {"top", "first", "result url", "result"},
            "scrape": {"scrape"},
            "output_format": {"markdown", "format", "output"},
            "response_shape": {"response", "result", "shape"},
            "configuration_options": {"option", "parameter", "timeout", "limit"},
            "error_handling": {"error", "retry", "rate limit", "timeout"},
            "production_safety": {"production", "secret", "rate limit", "retry"},
        }
        return {
            role
            for role, keywords in contradiction_groups.items()
            if role in covered and any(keyword in missing_text for keyword in keywords)
        }

    @classmethod
    def _remove_contradicted_missing_facts(
        cls,
        smoke_test: SmokeTestResult,
        contradicted_slots: set[str],
    ) -> SmokeTestResult:
        missing_facts = [
            fact
            for fact in smoke_test.missing_facts or []
            if not cls._text_matches_slots(fact, contradicted_slots)
        ]
        return smoke_test.model_copy(
            update={"missing_facts": missing_facts or None}
        )

    @classmethod
    def _remove_contradicted_items(
        cls,
        items: list[EvidenceItem],
        contradicted_slots: set[str],
        fallback_id: str,
        fallback_message: str,
    ) -> list[EvidenceItem]:
        filtered = [
            item
            for item in items
            if not cls._text_matches_slots(item.message, contradicted_slots)
        ]
        return filtered or [
            EvidenceItem(
                id=fallback_id,
                message=fallback_message,
                basis="inferred",
                severity="info",
            )
        ]

    @staticmethod
    def _text_matches_slots(text: str, slots: set[str]) -> bool:
        lowered = text.lower()
        slot_keywords = {
            "setup_install": {"install", "installation", "sdk command", "package"},
            "auth_secret": {"auth", "api key", "environment", "env-var", "env var"},
            "client_initialization": {"client", "initialize", "initialization"},
            "search": {"search"},
            "result_selection": {"top", "first", "result url", "result"},
            "scrape": {"scrape"},
            "output_format": {"markdown", "format", "output"},
            "configuration_options": {"option", "parameter", "timeout", "limit"},
            "error_handling": {"error", "retry", "rate limit", "timeout"},
            "production_safety": {"production", "secret", "rate limit", "retry"},
        }
        return any(
            slot in slots and any(keyword in lowered for keyword in keywords)
            for slot, keywords in slot_keywords.items()
        )

    def _normalize_scorecard(
        self,
        raw_scorecard: object,
        valid_source_ids: set[str],
        warnings: list[str],
        source_ref_rank: dict[str, int],
    ) -> list[ScorecardDimension]:
        raw_items = raw_scorecard if isinstance(raw_scorecard, list) else []
        by_id: dict[str, dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            dimension_id = item.get("id")
            if isinstance(dimension_id, str) and dimension_id in SCORECARD_IDS:
                by_id.setdefault(dimension_id, item)

        normalized: list[ScorecardDimension] = []
        for dimension_id, label in SCORECARD_DIMENSIONS:
            item = by_id.get(dimension_id)
            if item is None:
                warnings.append(f"Codex omitted scorecard dimension {dimension_id}.")
                normalized.append(
                    ScorecardDimension(
                        id=dimension_id,  # type: ignore[arg-type]
                        label=label,
                        score=0,
                        max_score=5,
                        rationale="Codex omitted this dimension, so it is marked uncertain.",
                        source_refs=[],
                    )
                )
                continue
            max_score = self._coerce_int(item.get("max_score"), default=5, minimum=1)
            score = self._coerce_int(
                item.get("score"), default=0, minimum=0, maximum=max_score
            )
            refs, changed = self._valid_source_refs(
                item.get("source_refs"), valid_source_ids, source_ref_rank
            )
            if changed:
                warnings.append(
                    f"Invalid source refs were removed from {dimension_id}."
                )
            normalized.append(
                ScorecardDimension(
                    id=dimension_id,  # type: ignore[arg-type]
                    label=str(item.get("label") or label),
                    score=score,
                    max_score=max_score,
                    rationale=str(item.get("rationale") or "No rationale returned."),
                    source_refs=refs,
                )
            )
        return normalized

    def _normalize_evidence_list(
        self,
        raw_items: object,
        prefix: str,
        valid_source_ids: set[str],
        warnings: list[str],
        source_ref_rank: dict[str, int],
    ) -> list[EvidenceItem]:
        if not isinstance(raw_items, list):
            return []
        normalized: list[EvidenceItem] = []
        for index, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                continue
            refs, changed = self._valid_source_refs(
                raw_item.get("source_refs"), valid_source_ids, source_ref_rank
            )
            basis = self._coerce_basis(raw_item.get("basis"))
            if changed:
                basis = "uncertain"
                warnings.append(
                    f"Invalid source refs were removed from {prefix}_{index}."
                )
            severity = raw_item.get("severity")
            normalized.append(
                EvidenceItem(
                    id=str(raw_item.get("id") or f"{prefix}_{index}"),
                    message=str(raw_item.get("message") or "No message returned."),
                    basis=basis,
                    source_refs=refs or None,
                    severity=severity
                    if severity in {"info", "warning", "blocking"}
                    else None,
                )
            )
        return normalized

    def _normalize_smoke_test(
        self,
        raw_smoke_test: object,
        valid_source_ids: set[str],
        warnings: list[str],
        source_ref_rank: dict[str, int],
    ) -> SmokeTestResult:
        if not isinstance(raw_smoke_test, dict):
            warnings.append("Codex omitted smoke_test.")
            return SmokeTestResult(
                result="partial",
                basis="uncertain",
                message="Codex did not return a smoke-test result.",
            )
        refs, changed = self._valid_source_refs(
            raw_smoke_test.get("source_refs"),
            valid_source_ids,
            source_ref_rank,
        )
        if changed:
            warnings.append("Invalid source refs were removed from smoke_test.")
        result = raw_smoke_test.get("result")
        return SmokeTestResult(
            result=result if result in {"pass", "partial", "fail"} else "partial",
            basis="uncertain"
            if changed
            else self._coerce_basis(raw_smoke_test.get("basis")),
            message=str(
                raw_smoke_test.get("message") or "No smoke-test message returned."
            ),
            source_refs=refs or None,
            missing_facts=self._string_list(raw_smoke_test.get("missing_facts"))
            or None,
            likely_next_steps=self._string_list(raw_smoke_test.get("likely_next_steps"))
            or None,
        )

    def _write_artifacts(self, report: AuditReport) -> str:
        report_dir = self._artifact_root / report.metadata.cache_key
        report_dir.mkdir(parents=True, exist_ok=True)
        report_json = report.model_dump(mode="json", exclude_none=True)
        json_path = report_dir / "report.json"
        markdown_path = report_dir / "report.md"
        json_tmp = report_dir / "report.tmp.json"
        markdown_tmp = report_dir / "report.tmp.md"
        json_tmp.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
        markdown_tmp.write_text(self._render_markdown(report), encoding="utf-8")
        json_tmp.replace(json_path)
        markdown_tmp.replace(markdown_path)
        return self._relative_artifact_path(json_path)

    def _relative_artifact_path(self, artifact_path: Path) -> str:
        try:
            return artifact_path.relative_to(self._settings.project_root).as_posix()
        except ValueError:
            return artifact_path.relative_to(self._artifact_root).as_posix()

    @staticmethod
    def _render_markdown(report: AuditReport) -> str:
        lines = [
            f"# Audit report for {report.request.docs_url}",
            "",
            report.summary,
            "",
            "## Sources",
        ]
        for source in report.selected_sources:
            lines.append(f"- `{source.id}` [{source.title}]({source.url})")
        lines.extend(["", "## Scorecard"])
        for dimension in report.scorecard:
            lines.append(
                f"- {dimension.label}: {dimension.score}/{dimension.max_score} - {dimension.rationale}"
            )
        lines.extend(
            [
                "",
                "## Smoke test",
                f"{report.smoke_test.result}: {report.smoke_test.message}",
            ]
        )
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {item.message}" for item in report.warnings)
        lines.extend(["", "## Suggested fixes"])
        lines.extend(f"- {item.message}" for item in report.suggested_fixes)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _valid_source_refs(
        raw_refs: object,
        valid_source_ids: set[str],
        source_ref_rank: dict[str, int] | None = None,
    ) -> tuple[list[str], bool]:
        if not isinstance(raw_refs, list):
            return [], False
        refs: list[str] = []
        changed = False
        for ref in raw_refs:
            if isinstance(ref, str) and ref in valid_source_ids and ref not in refs:
                refs.append(ref)
            else:
                changed = True
        if source_ref_rank:
            refs.sort(key=lambda ref: source_ref_rank.get(ref, 0), reverse=True)
        return refs, changed

    @staticmethod
    def _source_ref_rank(
        sources: list[Any],
        goal_tokens: set[str],
        valid_source_ids: set[str],
    ) -> dict[str, int]:
        if not CodexAuditEngineClient._requested_languages(goal_tokens):
            return {}
        rank: dict[str, int] = {}
        for source in sources:
            source_id = str(source.id)
            if source_id not in valid_source_ids:
                continue
            rank[source_id] = (
                CodexAuditEngineClient._source_quality_score(source)
                + CodexAuditEngineClient._language_source_score(source, goal_tokens)
                + len(CodexAuditEngineClient._matched_facets(source, goal_tokens)) * 4
                - CodexAuditEngineClient._source_order(source_id)
            )
        return rank

    @staticmethod
    def _coerce_basis(value: object) -> Basis:
        if value in {"source_backed", "inferred", "uncertain"}:
            return value  # type: ignore[return-value]
        return "uncertain"

    @staticmethod
    def _coerce_int(
        value: object,
        *,
        default: int,
        minimum: int,
        maximum: int | None = None,
    ) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        number = max(number, minimum)
        if maximum is not None:
            number = min(number, maximum)
        return number

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item]

    @staticmethod
    def _is_safe_report_cache_key(cache_key: str) -> bool:
        return bool(SAFE_REPORT_CACHE_KEY.fullmatch(cache_key)) and cache_key not in {
            ".",
            "..",
        }
