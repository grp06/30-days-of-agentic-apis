from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlparse, urlunparse

import httpx
from pydantic import BaseModel, Field

from .config import Settings
from .firecrawl_preflight import (
    DocsPreflightResult,
    FirecrawlPreflightClient,
    FirecrawlPreflightRequest,
    HttpFirecrawlPreflightClient,
    MAX_AUDIT_PAGES,
    is_private_host,
)


FetchStatus = Literal["completed", "completed_with_warnings", "blocked"]
EvidenceRole = Literal[
    "setup",
    "install",
    "auth",
    "search",
    "response_shape",
    "scrape",
    "output_format",
    "production",
    "reference",
]
RetrievedVia = Literal[
    "firecrawl_map",
    "firecrawl_scrape",
    "llms_txt",
    "sitemap",
    "cached_fixture",
]

ASSET_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".css",
    ".js",
    ".zip",
    ".tar",
    ".gz",
    ".mp4",
    ".mov",
    ".webm",
    ".pdf",
}
DOC_PATH_SEGMENTS = {
    "docs",
    "guides",
    "reference",
    "api",
    "quickstart",
    "sdk",
    "tutorial",
    "examples",
}
MARKETING_PATH_SEGMENTS = {
    "blog",
    "pricing",
    "careers",
    "customers",
    "contact",
    "privacy",
    "terms",
}
SAFE_CACHE_KEY = re.compile(r"^[A-Za-z0-9._-]+$")
MAX_MARKDOWN_CHARS = 80_000
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
LOCALE_SEGMENT = re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$", re.IGNORECASE)
LEGACY_PATH_SEGMENTS = {"deprecated", "legacy", "v0"}
LANGUAGE_ALIASES: dict[str, set[str]] = {
    "python": {"python", "py"},
    "nodejs": {"node", "nodejs", "javascript", "js", "typescript", "ts"},
    "ruby": {"ruby", "rails"},
    "php": {"php"},
    "go": {"go", "golang"},
    "rust": {"rust"},
    "java": {"java"},
    "dotnet": {"dotnet", "net", "csharp", "c#"},
    "elixir": {"elixir"},
}
LANGUAGE_PATH_ALIASES: dict[str, set[str]] = {
    "python": {"python", "py"},
    "nodejs": {"node", "nodejs", "javascript", "js", "typescript", "ts", "nestjs"},
    "ruby": {"ruby", "rails"},
    "php": {"php"},
    "go": {"go", "golang"},
    "rust": {"rust"},
    "java": {"java", "spring", "spring-boot"},
    "dotnet": {"dotnet", "net", "csharp"},
    "elixir": {"elixir"},
}
OPERATION_ALIASES: dict[str, set[str]] = {
    "search": {"search", "query"},
    "scrape": {"scrape", "scraping", "markdown"},
    "crawl": {"crawl", "crawling"},
    "extract": {"extract", "extraction", "json"},
    "map": {"map", "sitemap"},
    "auth": {"auth", "login", "signin", "sign", "session"},
}
SETUP_FACET_TERMS = {"api", "api_key", "key", "sdk", "install", "env", "environment", "setup"}


@dataclass(frozen=True)
class GoalProfile:
    tokens: set[str]
    languages: set[str]
    operations: set[str]
    setup_terms: set[str]
    output_terms: set[str]


class FirecrawlFetchRequest(BaseModel):
    docs_url: str = Field(min_length=1)
    integration_goal: str = Field(min_length=1)
    max_pages: int = Field(default=20, ge=1, le=MAX_AUDIT_PAGES)
    max_depth: int = Field(default=1, ge=0, le=3)
    allowed_hosts: list[str] | None = None
    firecrawl_api_key: str | None = None
    cache_key: str | None = None


class CandidateSource(BaseModel):
    url: str
    title: str | None = None
    description: str | None = None
    score: int
    reason_selected: str


class SourcePlannerSelection(BaseModel):
    candidate_id: str
    evidence_roles: list[EvidenceRole] = Field(default_factory=list)
    rationale: str = ""
    confidence: Literal["high", "medium", "low"] = "medium"


class SourcePlannerRejection(BaseModel):
    candidate_id: str
    reason: str


class SourcePlannerProbe(BaseModel):
    url: str
    evidence_roles: list[EvidenceRole] = Field(default_factory=list)
    rationale: str = ""


class SourcePlannerResult(BaseModel):
    status: Literal["planned", "fallback"] = "planned"
    selected_sources: list[SourcePlannerSelection] = Field(default_factory=list)
    rejected_sources: list[SourcePlannerRejection] = Field(default_factory=list)
    suggested_probe_urls: list[SourcePlannerProbe] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SelectedSource(BaseModel):
    id: str
    url: str
    title: str
    reason_selected: str
    retrieved_via: RetrievedVia
    markdown_chars: int


class FetchedSource(SelectedSource):
    markdown: str


class FirecrawlFetchResult(BaseModel):
    status: FetchStatus
    cache_key: str
    preflight: DocsPreflightResult
    selected_sources: list[SelectedSource] = Field(default_factory=list)
    candidate_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    artifact_path: str | None = None


class CachedFetchedSources(BaseModel):
    cache_key: str
    result: FirecrawlFetchResult
    fetched_sources: list[FetchedSource]
    planner_metadata: dict[str, Any] = Field(default_factory=dict)


class FirecrawlCacheNotFound(Exception):
    pass


class FirecrawlIngestionClient(Protocol):
    async def fetch_sources(
        self,
        request: FirecrawlFetchRequest,
    ) -> FirecrawlFetchResult: ...

    async def read_cached_sources(self, cache_key: str) -> FirecrawlFetchResult: ...

    async def read_cached_fetched_sources(
        self,
        cache_key: str,
    ) -> CachedFetchedSources: ...


class SourcePlanner(Protocol):
    async def plan_sources(
        self,
        *,
        docs_url: str,
        integration_goal: str,
        max_pages: int,
        allowed_hosts: list[str],
        candidates: list[dict[str, Any]],
    ) -> SourcePlannerResult: ...


class HttpFirecrawlIngestionClient:
    def __init__(
        self,
        settings: Settings,
        *,
        preflight_client: FirecrawlPreflightClient | None = None,
        source_planner: SourcePlanner | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        cache_root: Path | None = None,
    ) -> None:
        self._settings = settings
        self._preflight_client = preflight_client or HttpFirecrawlPreflightClient(
            settings,
            transport=transport,
        )
        self._source_planner = source_planner
        self._transport = transport
        self._timeout = httpx.Timeout(settings.firecrawl_timeout_seconds)
        self._cache_root = cache_root or (
            settings.project_root / ".agent" / "cache" / "firecrawl-runs"
        )

    async def fetch_sources(
        self,
        request: FirecrawlFetchRequest,
    ) -> FirecrawlFetchResult:
        cache_key = self._safe_cache_key(request.cache_key, request)
        artifact_path = self._artifact_path(cache_key)
        relative_artifact_path = self._relative_artifact_path(artifact_path)
        api_key = self._resolve_key(request.firecrawl_api_key)
        preflight = await self._preflight_client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url=request.docs_url,
                integration_goal=request.integration_goal,
                max_pages=request.max_pages,
                max_depth=request.max_depth,
                allowed_hosts=request.allowed_hosts,
                firecrawl_api_key=request.firecrawl_api_key,
            )
        )

        if preflight.verdict == "blocked":
            return self._blocked_result(
                cache_key,
                preflight,
                "Preflight blocked the docs URL before source fetch.",
            )
        if not api_key:
            return self._blocked_result(
                cache_key,
                preflight,
                "Firecrawl key is required for live source fetch.",
            )
        if preflight.key_status.status != "valid":
            return self._blocked_result(
                cache_key,
                preflight,
                "Firecrawl key must validate before live source fetch.",
            )
        if not preflight.normalized_url:
            return self._blocked_result(
                cache_key,
                preflight,
                "Preflight did not produce a normalized docs URL.",
            )

        warnings: list[str] = []
        candidates, map_warnings = await self._map_candidates(
            request,
            preflight,
            api_key,
        )
        warnings.extend(map_warnings)
        ranked_candidates = self._filter_and_rank_candidates(
            candidates,
            request,
            preflight,
        )
        ranked_candidates, planner_metadata, planner_warnings = (
            await self._apply_source_planner(request, preflight, ranked_candidates)
        )
        warnings.extend(planner_warnings)
        candidate_count = len(ranked_candidates)

        fetched_sources: list[FetchedSource] = []
        async with self._http_client() as client:
            for candidate in ranked_candidates:
                if len(fetched_sources) >= request.max_pages:
                    break
                fetched, warning, fatal = await self._scrape_source(
                    client,
                    candidate,
                    len(fetched_sources) + 1,
                    api_key,
                )
                if warning:
                    warnings.append(warning)
                if fetched:
                    fetched_sources.append(fetched)
                if fatal:
                    break

        if not fetched_sources:
            return FirecrawlFetchResult(
                status="blocked",
                cache_key=cache_key,
                preflight=preflight,
                selected_sources=[],
                candidate_count=candidate_count,
                warnings=warnings or ["No source pages could be fetched."],
                artifact_path=None,
            )

        selected_sources = [
            self._selected_summary(source) for source in fetched_sources
        ]
        planner_metadata = self._finalize_planner_metadata(
            planner_metadata,
            fetched_sources,
        )
        status: FetchStatus = "completed_with_warnings" if warnings else "completed"
        result = FirecrawlFetchResult(
            status=status,
            cache_key=cache_key,
            preflight=preflight,
            selected_sources=selected_sources,
            candidate_count=candidate_count,
            warnings=warnings,
            artifact_path=relative_artifact_path,
        )
        self._write_cache_artifact(
            artifact_path,
            request,
            result,
            fetched_sources,
            planner_metadata,
        )
        return result

    async def read_cached_sources(self, cache_key: str) -> FirecrawlFetchResult:
        if not self._is_safe_cache_key(cache_key):
            raise FirecrawlCacheNotFound
        payload = self._read_cache_payload(cache_key)
        try:
            result = FirecrawlFetchResult.model_validate(payload)
        except ValueError:
            raise FirecrawlCacheNotFound from None
        if result.cache_key != cache_key:
            raise FirecrawlCacheNotFound
        return result

    async def read_cached_fetched_sources(
        self,
        cache_key: str,
    ) -> CachedFetchedSources:
        if not self._is_safe_cache_key(cache_key):
            raise FirecrawlCacheNotFound
        payload = self._read_cache_payload(cache_key)
        fetched_payload = payload.get("fetched_sources")
        if not isinstance(fetched_payload, list):
            raise FirecrawlCacheNotFound
        try:
            result = FirecrawlFetchResult.model_validate(payload)
            fetched_sources = [
                FetchedSource.model_validate(source) for source in fetched_payload
            ]
        except ValueError:
            raise FirecrawlCacheNotFound from None
        if result.cache_key != cache_key:
            raise FirecrawlCacheNotFound
        if not fetched_sources:
            raise FirecrawlCacheNotFound
        return CachedFetchedSources(
            cache_key=result.cache_key,
            result=result,
            fetched_sources=fetched_sources,
            planner_metadata=payload.get("planner_metadata")
            if isinstance(payload.get("planner_metadata"), dict)
            else {},
        )

    def _read_cache_payload(self, cache_key: str) -> dict[str, Any]:
        artifact_path = self._artifact_path(cache_key)
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            raise FirecrawlCacheNotFound from None
        if not isinstance(payload, dict):
            raise FirecrawlCacheNotFound
        return payload

    def _http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=False,
            headers={"User-Agent": "FirecrawlDocsAuditor/0.1"},
            transport=self._transport,
        )

    def _resolve_key(self, request_key: str | None) -> str | None:
        if request_key and request_key.strip():
            return request_key.strip()
        if (
            self._settings.firecrawl_api_key
            and self._settings.firecrawl_api_key.strip()
        ):
            return self._settings.firecrawl_api_key.strip()
        return None

    def _blocked_result(
        self,
        cache_key: str,
        preflight: DocsPreflightResult,
        warning: str,
    ) -> FirecrawlFetchResult:
        return FirecrawlFetchResult(
            status="blocked",
            cache_key=cache_key,
            preflight=preflight,
            selected_sources=[],
            candidate_count=0,
            warnings=[warning],
            artifact_path=None,
        )

    async def _map_candidates(
        self,
        request: FirecrawlFetchRequest,
        preflight: DocsPreflightResult,
        api_key: str,
    ) -> tuple[list[CandidateSource], list[str]]:
        warnings: list[str] = []
        url = f"{self._settings.firecrawl_api_base_url.rstrip('/')}/map"
        payload = {
            "url": preflight.normalized_url,
            "search": request.integration_goal,
            "sitemap": "include",
            "includeSubdomains": False,
            "ignoreQueryParameters": True,
            "limit": min(max(request.max_pages * 5, request.max_pages), 100),
            "timeout": int(self._settings.firecrawl_timeout_seconds * 1000),
        }
        try:
            async with self._http_client() as client:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
        except httpx.HTTPError:
            return [
                CandidateSource(
                    url=preflight.normalized_url or request.docs_url,
                    score=0,
                    reason_selected="Fallback to the normalized docs entry point after map failed.",
                )
            ], [
                "Firecrawl map could not be reached; falling back to the docs entry point."
            ]

        if response.status_code in {401, 402, 429}:
            return [], [self._firecrawl_status_message(response.status_code)]
        if response.status_code >= 500:
            return [
                CandidateSource(
                    url=preflight.normalized_url or request.docs_url,
                    score=0,
                    reason_selected="Fallback to the normalized docs entry point after map failed.",
                )
            ], [
                f"Firecrawl map returned HTTP {response.status_code}; using fallback entry point."
            ]
        try:
            body = response.json()
        except ValueError:
            return [
                CandidateSource(
                    url=preflight.normalized_url or request.docs_url,
                    score=0,
                    reason_selected="Fallback to the normalized docs entry point after map returned malformed JSON.",
                )
            ], ["Firecrawl map returned malformed JSON; using fallback entry point."]
        links = body.get("links") if isinstance(body, dict) else None
        if not isinstance(links, list):
            return [
                CandidateSource(
                    url=preflight.normalized_url or request.docs_url,
                    score=0,
                    reason_selected="Fallback to the normalized docs entry point because map returned no links.",
                )
            ], [
                "Firecrawl map returned no candidate links; using fallback entry point."
            ]

        candidates = []
        for link in links:
            if isinstance(link, str):
                candidates.append(
                    CandidateSource(
                        url=link,
                        score=0,
                        reason_selected="Candidate returned by Firecrawl map.",
                    )
                )
            elif isinstance(link, dict) and isinstance(link.get("url"), str):
                candidates.append(
                    CandidateSource(
                        url=link["url"],
                        title=link.get("title")
                        if isinstance(link.get("title"), str)
                        else None,
                        description=(
                            link.get("description")
                            if isinstance(link.get("description"), str)
                            else None
                        ),
                        score=0,
                        reason_selected="Candidate returned by Firecrawl map.",
                    )
                )
        if preflight.normalized_url and all(
            candidate.url != preflight.normalized_url for candidate in candidates
        ):
            candidates.insert(
                0,
                CandidateSource(
                    url=preflight.normalized_url,
                    score=0,
                    reason_selected="Included the normalized docs entry point as a fallback.",
                ),
            )
        return candidates, warnings

    def _filter_and_rank_candidates(
        self,
        candidates: list[CandidateSource],
        request: FirecrawlFetchRequest,
        preflight: DocsPreflightResult,
    ) -> list[CandidateSource]:
        allowed_hosts = set(preflight.allowed_hosts)
        goal_profile = self._goal_profile(request.integration_goal, request.docs_url)
        candidates = [
            *candidates,
            *self._canonical_probe_candidates(request, preflight, goal_profile),
        ]
        priority_terms = self._priority_terms(request.integration_goal)
        seen_urls: set[str] = set()
        ranked: list[tuple[CandidateSource, str]] = []
        for candidate in candidates:
            parsed = urlparse(candidate.url.strip())
            host = parsed.hostname.lower() if parsed.hostname else ""
            if (
                parsed.scheme not in {"http", "https"}
                or not host
                or parsed.username
                or parsed.password
                or is_private_host(host)
                or host not in allowed_hosts
                or self._is_asset_path(parsed.path)
                or self._is_marketing_path(parsed.path)
            ):
                continue
            normalized_url = urlunparse(
                (
                    parsed.scheme.lower(),
                    parsed.netloc.lower(),
                    parsed.path or "/",
                    "",
                    "",
                    "",
                )
            )
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            score, reason = self._score_candidate(
                candidate,
                normalized_url,
                goal_profile,
                priority_terms,
                preflight.normalized_url,
            )
            ranked.append(
                (
                    CandidateSource(
                        url=normalized_url,
                        title=candidate.title,
                        description=candidate.description,
                        score=score,
                        reason_selected=reason,
                    ),
                    self._canonical_source_key(normalized_url),
                )
            )
        preferred_locale = self._locale_prefix(urlparse(request.docs_url).path)
        primary_ranked = [
            item
            for item in ranked
            if self._is_primary_language_candidate(item[0].url, preferred_locale)
        ]
        overflow_ranked = [
            item
            for item in ranked
            if not self._is_primary_language_candidate(item[0].url, preferred_locale)
        ]
        return self._select_ranked_candidates(
            primary_ranked,
            overflow_ranked,
            goal_profile,
        )

    async def _apply_source_planner(
        self,
        request: FirecrawlFetchRequest,
        preflight: DocsPreflightResult,
        ranked_candidates: list[CandidateSource],
    ) -> tuple[list[CandidateSource], dict[str, Any], list[str]]:
        metadata: dict[str, Any] = {
            "status": "not_configured",
            "selected_candidate_ids": [],
            "rejected_candidate_ids": [],
            "suggested_probe_urls": [],
            "roles_by_url": {},
            "warnings": [],
        }
        if self._source_planner is None or not ranked_candidates:
            return ranked_candidates, metadata, []

        catalog = self._planner_candidate_catalog(
            ranked_candidates,
            request,
        )
        candidate_by_id = {
            str(item["id"]): candidate
            for item, candidate in zip(catalog, ranked_candidates, strict=False)
        }
        try:
            plan = await self._source_planner.plan_sources(
                docs_url=request.docs_url,
                integration_goal=request.integration_goal,
                max_pages=request.max_pages,
                allowed_hosts=preflight.allowed_hosts,
                candidates=catalog,
            )
        except Exception:
            metadata.update(
                {
                    "status": "fallback",
                    "fallback_reason": "Source planner raised an exception.",
                }
            )
            return ranked_candidates, metadata, [
                "Source planner failed; using deterministic source ranking."
            ]

        metadata["status"] = plan.status
        metadata["selected_candidate_ids"] = [
            selection.candidate_id for selection in plan.selected_sources
        ]
        metadata["rejected_candidate_ids"] = [
            rejection.candidate_id for rejection in plan.rejected_sources
        ]
        metadata["suggested_probe_urls"] = [
            probe.url for probe in plan.suggested_probe_urls
        ]
        metadata["warnings"] = plan.warnings

        warnings = [f"Source planner: {warning}" for warning in plan.warnings]
        if plan.status != "planned":
            metadata["fallback_reason"] = "Source planner reported fallback status."
            warnings.append("Source planner unavailable; using deterministic source ranking.")
            return ranked_candidates, metadata, warnings

        selected: list[CandidateSource] = []
        selected_urls: set[str] = set()
        roles_by_url: dict[str, list[str]] = {}
        invalid_ids: list[str] = []
        for selection in plan.selected_sources:
            candidate = candidate_by_id.get(selection.candidate_id)
            if candidate is None:
                invalid_ids.append(selection.candidate_id)
                continue
            roles = self._clean_roles(selection.evidence_roles)
            roles_by_url[candidate.url] = roles
            selected.append(
                self._planner_decorated_candidate(
                    candidate,
                    roles,
                    selection.rationale,
                )
            )
            selected_urls.add(candidate.url)

        suggested_candidates = self._validated_suggested_probe_candidates(
            plan.suggested_probe_urls,
            request,
            preflight,
            roles_by_url,
        )
        for candidate in suggested_candidates:
            if candidate.url in selected_urls:
                continue
            selected.append(candidate)
            selected_urls.add(candidate.url)

        if invalid_ids:
            warnings.append(
                "Source planner returned unknown candidate id(s): "
                + ", ".join(invalid_ids[:5])
                + "."
            )
        if not selected:
            metadata["status"] = "fallback"
            metadata["fallback_reason"] = "Source planner returned no valid candidates."
            warnings.append("Source planner returned no valid candidates; using deterministic source ranking.")
            return ranked_candidates, metadata, warnings

        for candidate in ranked_candidates:
            if candidate.url not in selected_urls:
                selected.append(candidate)
                selected_urls.add(candidate.url)

        metadata["roles_by_url"] = roles_by_url
        return selected, metadata, warnings

    def _planner_candidate_catalog(
        self,
        candidates: list[CandidateSource],
        request: FirecrawlFetchRequest,
    ) -> list[dict[str, Any]]:
        goal_tokens = self._goal_tokens(request.integration_goal, request.docs_url)
        return [
            {
                "id": f"cand_{index}",
                "url": candidate.url,
                "title": candidate.title or self._title_from_url(candidate.url),
                "description": self._compact_planner_text(candidate.description or "", 160),
                "deterministic_score": candidate.score,
                "reason_selected": self._compact_planner_text(
                    candidate.reason_selected,
                    180,
                ),
                "matched_goal_terms": sorted(
                    token
                    for token in goal_tokens
                    if token in self._candidate_haystack(candidate)
                ),
                "path_flags": self._planner_path_flags(candidate.url),
            }
            for index, candidate in enumerate(candidates[:60], start=1)
        ]

    @staticmethod
    def _compact_planner_text(value: str, limit: int) -> str:
        compact = re.sub(r"\s+", " ", value).strip()
        return compact[:limit]

    @staticmethod
    def _planner_path_flags(url: str) -> dict[str, bool]:
        parsed = urlparse(url)
        path = parsed.path.lower()
        segments = {segment for segment in path.split("/") if segment}
        return {
            "legacy": HttpFirecrawlIngestionClient._path_has_legacy_segment(path),
            "localized": HttpFirecrawlIngestionClient._path_has_locale_prefix(path),
            "source_like": path.endswith((".md", ".mdx", "llms.txt")),
            "quickstart": bool(segments & {"quickstart", "quickstarts"}),
            "sdk": bool(segments & {"sdk", "sdks"}),
            "api_reference": "api-reference" in segments
            or bool(segments & {"api", "reference"}),
        }

    def _validated_suggested_probe_candidates(
        self,
        probes: list[SourcePlannerProbe],
        request: FirecrawlFetchRequest,
        preflight: DocsPreflightResult,
        roles_by_url: dict[str, list[str]],
    ) -> list[CandidateSource]:
        if not probes:
            return []
        raw_candidates = [
            CandidateSource(
                url=probe.url,
                title=None,
                description=probe.rationale,
                score=0,
                reason_selected="Suggested by Codex source planner.",
            )
            for probe in probes
        ]
        normalized_probe_urls = {
            self._normalize_candidate_url(probe.url) for probe in probes
        }
        normalized_probe_urls.discard(None)
        ranked = self._filter_and_rank_candidates(raw_candidates, request, preflight)
        by_url = {
            self._normalize_candidate_url(probe.url): probe
            for probe in probes
            if self._normalize_candidate_url(probe.url)
        }
        validated: list[CandidateSource] = []
        for candidate in ranked:
            if candidate.url not in normalized_probe_urls:
                continue
            probe = by_url.get(candidate.url)
            roles = self._clean_roles(probe.evidence_roles if probe else [])
            roles_by_url[candidate.url] = roles
            validated.append(
                self._planner_decorated_candidate(
                    candidate,
                    roles,
                    probe.rationale if probe else "",
                )
            )
        return validated

    @staticmethod
    def _normalize_candidate_url(url: str) -> str | None:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return None
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path or "/",
                "",
                "",
                "",
            )
        )

    @staticmethod
    def _planner_decorated_candidate(
        candidate: CandidateSource,
        roles: list[str],
        rationale: str,
    ) -> CandidateSource:
        reason_parts = [candidate.reason_selected]
        if roles:
            reason_parts.append("Planner role(s): " + ", ".join(roles) + ".")
        if rationale:
            reason_parts.append("Planner rationale: " + rationale[:220])
        return candidate.model_copy(update={"reason_selected": " ".join(reason_parts)})

    @staticmethod
    def _clean_roles(roles: list[EvidenceRole]) -> list[str]:
        cleaned: list[str] = []
        for role in roles:
            if role not in cleaned:
                cleaned.append(role)
        return cleaned

    @staticmethod
    def _finalize_planner_metadata(
        metadata: dict[str, Any],
        fetched_sources: list[FetchedSource],
    ) -> dict[str, Any]:
        roles_by_url = metadata.get("roles_by_url")
        if not isinstance(roles_by_url, dict):
            roles_by_url = {}
        source_roles = {
            source.id: roles_by_url.get(source.url, [])
            for source in fetched_sources
            if isinstance(roles_by_url.get(source.url), list)
        }
        next_metadata = dict(metadata)
        next_metadata["source_roles"] = source_roles
        next_metadata["fetched_urls"] = [source.url for source in fetched_sources]
        return next_metadata

    def _select_ranked_candidates(
        self,
        primary_ranked: list[tuple[CandidateSource, str]],
        overflow_ranked: list[tuple[CandidateSource, str]],
        goal_profile: GoalProfile,
    ) -> list[CandidateSource]:
        selected: list[CandidateSource] = []
        selected_urls: set[str] = set()
        selected_keys: set[str] = set()
        required_facets = sorted(self._required_facets(goal_profile))
        for facet in required_facets:
            candidates_for_facet = [
                (candidate, key)
                for candidate, key in primary_ranked
                if candidate.url not in selected_urls
                and facet in self._candidate_facets(candidate, goal_profile)
            ]
            if not candidates_for_facet:
                continue
            candidate, key = max(
                candidates_for_facet,
                key=lambda item: self._rank_sort_key(item[0], item[1], selected_keys),
            )
            selected.append(candidate)
            selected_urls.add(candidate.url)
            selected_keys.add(key)
        available_tokens = sorted(
            {
                token
                for candidate, _ in primary_ranked
                for token in goal_profile.tokens
                if token in self._candidate_haystack(candidate)
            }
        )
        for token in available_tokens:
            candidates_for_token = [
                (candidate, key)
                for candidate, key in primary_ranked
                if candidate.url not in selected_urls
                and token in self._candidate_haystack(candidate)
            ]
            if not candidates_for_token:
                continue
            candidate, key = max(
                candidates_for_token,
                key=lambda item: self._rank_sort_key(item[0], item[1], selected_keys),
            )
            selected.append(candidate)
            selected_urls.add(candidate.url)
            selected_keys.add(key)
        for candidate_pool in (primary_ranked, overflow_ranked):
            for candidate, key in sorted(
                candidate_pool,
                key=lambda item: self._rank_sort_key(item[0], item[1], selected_keys),
                reverse=True,
            ):
                if candidate.url in selected_urls:
                    continue
                selected.append(candidate)
                selected_urls.add(candidate.url)
                selected_keys.add(key)
        return selected

    async def _scrape_source(
        self,
        client: httpx.AsyncClient,
        candidate: CandidateSource,
        index: int,
        api_key: str,
    ) -> tuple[FetchedSource | None, str | None, bool]:
        url = f"{self._settings.firecrawl_api_base_url.rstrip('/')}/scrape"
        try:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "url": candidate.url,
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                    "removeBase64Images": True,
                    "timeout": int(self._settings.firecrawl_timeout_seconds * 1000),
                },
            )
        except httpx.HTTPError:
            return None, f"Could not fetch {candidate.url}; continuing.", False

        if response.status_code in {401, 402, 429}:
            return None, self._firecrawl_status_message(response.status_code), True
        if response.status_code == 404:
            return (
                None,
                f"Firecrawl scrape returned HTTP 404 for {candidate.url}.",
                False,
            )
        if response.status_code >= 400:
            return (
                None,
                f"Firecrawl scrape returned HTTP {response.status_code} for one source.",
                False,
            )
        try:
            body = response.json()
        except ValueError:
            return (
                None,
                f"Firecrawl scrape returned malformed JSON for {candidate.url}.",
                False,
            )
        markdown = self._extract_markdown(body)
        if not markdown.strip():
            return (
                None,
                f"Firecrawl scrape returned no markdown for {candidate.url}.",
                False,
            )
        markdown = markdown[:MAX_MARKDOWN_CHARS]
        title = (
            self._extract_title(body)
            or candidate.title
            or self._title_from_url(candidate.url)
        )
        if self._is_unusable_scraped_page(body, title, markdown):
            return (
                None,
                f"Rejected unusable docs page {candidate.url}; continuing.",
                False,
            )
        return (
            FetchedSource(
                id=f"src_{index}",
                url=candidate.url,
                title=title,
                reason_selected=candidate.reason_selected,
                retrieved_via="firecrawl_scrape",
                markdown=markdown,
                markdown_chars=len(markdown),
            ),
            None,
            False,
        )

    @staticmethod
    def _extract_markdown(body: Any) -> str:
        if not isinstance(body, dict):
            return ""
        if isinstance(body.get("markdown"), str):
            return body["markdown"]
        data = body.get("data")
        if isinstance(data, dict):
            if isinstance(data.get("markdown"), str):
                return data["markdown"]
            for value in data.values():
                if isinstance(value, str) and value.lstrip().startswith("#"):
                    return value
        return ""

    @staticmethod
    def _extract_title(body: Any) -> str | None:
        if not isinstance(body, dict):
            return None
        data = body.get("data")
        metadata = (
            data.get("metadata") if isinstance(data, dict) else body.get("metadata")
        )
        if isinstance(metadata, dict) and isinstance(metadata.get("title"), str):
            return metadata["title"]
        return None

    @staticmethod
    def _is_unusable_scraped_page(body: Any, title: str, markdown: str) -> bool:
        status_code = HttpFirecrawlIngestionClient._extract_scraped_status_code(body)
        if status_code is not None and status_code >= 400:
            return True
        normalized_title = re.sub(r"\s+", " ", title.strip()).lower()
        if normalized_title in {"404", "404 not found", "not found", "page not found"}:
            return True
        first_text = re.sub(r"\s+", " ", markdown.strip()[:1200]).lower()
        not_found_markers = (
            "404 not found",
            "page not found",
            "this page could not be found",
            "we couldn't find this page",
            "the page you are looking for does not exist",
        )
        if any(marker in first_text for marker in not_found_markers):
            useful_lines = [
                line
                for line in markdown.splitlines()
                if len(line.strip()) > 40
                and not re.search(r"\b(?:404|not found|search|navigation)\b", line, re.IGNORECASE)
            ]
            return len(useful_lines) <= 2
        return False

    @staticmethod
    def _extract_scraped_status_code(body: Any) -> int | None:
        if not isinstance(body, dict):
            return None
        candidates: list[Any] = []
        data = body.get("data")
        if isinstance(data, dict):
            metadata = data.get("metadata")
            if isinstance(metadata, dict):
                candidates.extend(
                    [
                        metadata.get("statusCode"),
                        metadata.get("status_code"),
                        metadata.get("sourceStatusCode"),
                    ]
                )
            candidates.extend([data.get("statusCode"), data.get("status_code")])
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            candidates.extend(
                [
                    metadata.get("statusCode"),
                    metadata.get("status_code"),
                    metadata.get("sourceStatusCode"),
                ]
            )
        for value in candidates:
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    @staticmethod
    def _selected_summary(source: FetchedSource) -> SelectedSource:
        return SelectedSource(
            id=source.id,
            url=source.url,
            title=source.title,
            reason_selected=source.reason_selected,
            retrieved_via=source.retrieved_via,
            markdown_chars=source.markdown_chars,
        )

    def _write_cache_artifact(
        self,
        artifact_path: Path,
        request: FirecrawlFetchRequest,
        result: FirecrawlFetchResult,
        fetched_sources: list[FetchedSource],
        planner_metadata: dict[str, Any],
    ) -> None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = result.model_dump(mode="json")
        payload.update(
            {
                "request": self._safe_request_payload(
                    request,
                    result.preflight.allowed_hosts,
                    result.cache_key,
                ),
                "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "note": "Local demo cache; runtime credentials are intentionally excluded.",
                "fetched_sources": [
                    source.model_dump(mode="json") for source in fetched_sources
                ],
                "planner_metadata": planner_metadata,
            }
        )
        temp_path = artifact_path.with_name("sources.tmp.json")
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(artifact_path)

    @staticmethod
    def _safe_request_payload(
        request: FirecrawlFetchRequest,
        allowed_hosts: list[str],
        cache_key: str,
    ) -> dict[str, Any]:
        return {
            "docs_url": request.docs_url,
            "integration_goal": request.integration_goal,
            "max_pages": request.max_pages,
            "max_depth": request.max_depth,
            "allowed_hosts": allowed_hosts,
            "cache_key": cache_key,
        }

    def _safe_cache_key(
        self,
        requested_cache_key: str | None,
        request: FirecrawlFetchRequest,
    ) -> str:
        if requested_cache_key and self._is_safe_cache_key(requested_cache_key):
            return requested_cache_key
        host = urlparse(request.docs_url).hostname or "docs"
        goal_slug = self._slug(request.integration_goal) or "goal"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{self._slug(host)}-{goal_slug}-{timestamp}"

    @staticmethod
    def _is_safe_cache_key(cache_key: str) -> bool:
        return bool(SAFE_CACHE_KEY.fullmatch(cache_key)) and cache_key not in {
            ".",
            "..",
        }

    def _artifact_path(self, cache_key: str) -> Path:
        return self._cache_root / cache_key / "sources.json"

    def _relative_artifact_path(self, artifact_path: Path) -> str:
        try:
            return artifact_path.relative_to(self._settings.project_root).as_posix()
        except ValueError:
            return artifact_path.relative_to(self._cache_root).as_posix()

    @staticmethod
    def _goal_tokens(goal: str, docs_url: str = "") -> set[str]:
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

    @classmethod
    def _goal_profile(cls, goal: str, docs_url: str = "") -> GoalProfile:
        tokens = cls._goal_tokens(goal, docs_url)
        languages = {
            language
            for language, aliases in LANGUAGE_ALIASES.items()
            if tokens & aliases
        }
        operations = {
            operation
            for operation, aliases in OPERATION_ALIASES.items()
            if tokens & aliases
        }
        setup_terms = tokens & SETUP_FACET_TERMS
        output_terms = tokens & {"markdown", "json", "html", "links", "top", "result"}
        return GoalProfile(
            tokens=tokens,
            languages=languages,
            operations=operations,
            setup_terms=setup_terms,
            output_terms=output_terms,
        )

    @staticmethod
    def _canonical_probe_candidates(
        request: FirecrawlFetchRequest,
        preflight: DocsPreflightResult,
        goal_profile: GoalProfile,
    ) -> list[CandidateSource]:
        base_url = preflight.normalized_url or request.docs_url
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return []
        origin = urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))
        paths: list[str] = []
        for language in sorted(goal_profile.languages):
            for prefix in ("quickstarts", "sdks"):
                paths.append(f"/{prefix}/{language}")
        for operation in sorted(goal_profile.operations):
            paths.append(f"/features/{operation}")
            paths.append(f"/api-reference/endpoint/{operation}")
        seen: set[str] = set()
        probes: list[CandidateSource] = []
        for path in paths:
            url = f"{origin}{path}"
            if url in seen:
                continue
            seen.add(url)
            probes.append(
                CandidateSource(
                    url=url,
                    score=0,
                    reason_selected="Added canonical docs-path probe for requested facets.",
                )
            )
        return probes

    @staticmethod
    def _is_asset_path(path: str) -> bool:
        lowered = path.lower()
        return any(lowered.endswith(extension) for extension in ASSET_EXTENSIONS)

    @staticmethod
    def _is_marketing_path(path: str) -> bool:
        segments = {segment for segment in path.lower().split("/") if segment}
        return bool(segments & MARKETING_PATH_SEGMENTS)

    @staticmethod
    def _score_candidate(
        candidate: CandidateSource,
        normalized_url: str,
        goal_profile: GoalProfile,
        priority_terms: set[str],
        entry_url: str | None,
    ) -> tuple[int, str]:
        haystack = " ".join(
            [
                normalized_url.lower(),
                candidate.title or "",
                candidate.description or "",
            ]
        ).lower()
        priority_haystack = " ".join(
            [
                normalized_url.lower(),
                candidate.title or "",
            ]
        ).lower()
        parsed = urlparse(normalized_url)
        path_segments = [segment for segment in parsed.path.lower().split("/") if segment]
        path_segment_set = set(path_segments)
        matched_tokens = [token for token in goal_profile.tokens if token in haystack]
        exact_locator_tokens = [
            token for token in goal_profile.tokens if token in priority_haystack
        ]
        matched_priority_terms = [
            token for token in priority_terms if token in priority_haystack
        ]
        matched_facets = HttpFirecrawlIngestionClient._candidate_facets(
            candidate.model_copy(update={"url": normalized_url}),
            goal_profile,
        )
        score = len(matched_tokens) * 10
        score += len(exact_locator_tokens) * 10
        score += len(matched_priority_terms) * 6
        score += len(matched_facets) * 14
        score += HttpFirecrawlIngestionClient._intent_score(
            normalized_url,
            priority_haystack,
            goal_profile,
        )
        if path_segment_set & DOC_PATH_SEGMENTS:
            score += 5
        if normalized_url.lower().endswith((".md", ".mdx")):
            score += 6
        if HttpFirecrawlIngestionClient._path_has_locale_prefix(parsed.path):
            score -= 6
        if HttpFirecrawlIngestionClient._path_has_legacy_segment(parsed.path):
            score -= 120
        if entry_url and normalized_url == entry_url:
            score += 4
        reason_parts = []
        if matched_tokens:
            reason_parts.append(
                f"Matched goal term(s): {', '.join(matched_tokens[:4])}."
            )
        if matched_facets:
            reason_parts.append(
                f"Matched required facet(s): {', '.join(sorted(matched_facets)[:4])}."
            )
        if matched_priority_terms:
            reason_parts.append(
                f"Matched setup/security term(s): {', '.join(matched_priority_terms[:4])}."
            )
        if path_segment_set & DOC_PATH_SEGMENTS:
            reason_parts.append("URL path looks documentation-oriented.")
        if entry_url and normalized_url == entry_url:
            reason_parts.append("Included the supplied docs entry point.")
        if not reason_parts:
            reason_parts.append("Selected from Firecrawl map candidates.")
        return score, " ".join(reason_parts)

    @staticmethod
    def _rank_sort_key(
        candidate: CandidateSource,
        canonical_key: str,
        selected_keys: set[str],
    ) -> tuple[int, int, int]:
        duplicate_penalty = -12 if canonical_key in selected_keys else 0
        return (
            candidate.score + duplicate_penalty,
            HttpFirecrawlIngestionClient._source_quality_score(candidate.url),
            -len(candidate.url),
        )

    @staticmethod
    def _required_facets(goal_profile: GoalProfile) -> set[str]:
        facets = {f"language:{language}" for language in goal_profile.languages}
        facets.update(f"operation:{operation}" for operation in goal_profile.operations)
        if goal_profile.languages and goal_profile.setup_terms:
            facets.add("setup")
        if goal_profile.output_terms:
            facets.update(f"output:{term}" for term in goal_profile.output_terms)
        return facets

    @staticmethod
    def _candidate_facets(
        candidate: CandidateSource,
        goal_profile: GoalProfile,
    ) -> set[str]:
        haystack = HttpFirecrawlIngestionClient._candidate_haystack(candidate)
        parsed = urlparse(candidate.url)
        path_segments = [segment.lower() for segment in parsed.path.split("/") if segment]
        facets: set[str] = set()
        for language in goal_profile.languages:
            aliases = LANGUAGE_PATH_ALIASES[language]
            if aliases & set(path_segments) or any(alias in haystack for alias in aliases):
                facets.add(f"language:{language}")
        for operation in goal_profile.operations:
            aliases = OPERATION_ALIASES[operation]
            if aliases & set(path_segments) or any(alias in haystack for alias in aliases):
                facets.add(f"operation:{operation}")
        if goal_profile.setup_terms and (
            {"quickstarts", "quickstart", "sdks", "sdk"} & set(path_segments)
            or goal_profile.setup_terms & set(re.findall(r"[a-z0-9_]+", haystack))
        ):
            facets.add("setup")
        for output_term in goal_profile.output_terms:
            if output_term in haystack or (
                output_term == "markdown"
                and (
                    {"scrape", "scraping"} & set(path_segments)
                    or "operation:scrape" in facets
                )
            ):
                facets.add(f"output:{output_term}")
        return facets

    @staticmethod
    def _intent_score(
        normalized_url: str,
        priority_haystack: str,
        goal_profile: GoalProfile,
    ) -> int:
        parsed = urlparse(normalized_url)
        segments = [segment.lower() for segment in parsed.path.split("/") if segment]
        segment_set = set(segments)
        score = 0
        for language in goal_profile.languages:
            aliases = LANGUAGE_PATH_ALIASES[language]
            has_language = bool(aliases & segment_set) or any(
                alias in priority_haystack for alias in aliases
            )
            if not has_language:
                continue
            if "quickstarts" in segment_set or "quickstart" in segment_set:
                score += 70
            if "sdks" in segment_set:
                score += 65
            elif "sdk" in segment_set:
                score += 55
        if goal_profile.languages and (
            "quickstarts" in segment_set or "quickstart" in segment_set
        ):
            if not any(
                LANGUAGE_PATH_ALIASES[language] & segment_set
                for language in goal_profile.languages
            ):
                score -= 70
        for operation in goal_profile.operations:
            if len(segments) >= 2 and segments[-2] == "features" and segments[-1] == operation:
                score += 55
            if "api-reference" in segment_set and segments and segments[-1] == operation:
                score += 18
        if goal_profile.setup_terms and (
            "quickstarts" in segment_set or "quickstart" in segment_set
        ):
            score += 20
        if "developer-guides" in segment_set and not any(
            token in priority_haystack for token in goal_profile.tokens
        ):
            score -= 20
        return score

    @staticmethod
    def _candidate_haystack(candidate: CandidateSource) -> str:
        return " ".join(
            [
                candidate.url,
                candidate.title or "",
                candidate.description or "",
                candidate.reason_selected,
            ]
        ).lower()

    @staticmethod
    def _source_quality_score(url: str) -> int:
        parsed = urlparse(url)
        score = 0
        if parsed.path.lower().endswith((".md", ".mdx")):
            score += 8
        if parsed.path.lower().endswith("llms.txt"):
            score += 4
        if HttpFirecrawlIngestionClient._path_has_locale_prefix(parsed.path):
            score -= 8
        if HttpFirecrawlIngestionClient._path_has_legacy_segment(parsed.path):
            score -= 60
        return score

    @staticmethod
    def _is_primary_language_candidate(url: str, preferred_locale: str | None) -> bool:
        candidate_locale = HttpFirecrawlIngestionClient._locale_prefix(
            urlparse(url).path
        )
        if candidate_locale is None:
            return True
        if candidate_locale.startswith("en"):
            return True
        if preferred_locale and candidate_locale == preferred_locale:
            return True
        return False

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
    def _path_has_locale_prefix(path: str) -> bool:
        return HttpFirecrawlIngestionClient._locale_prefix(path) is not None

    @staticmethod
    def _locale_prefix(path: str) -> str | None:
        segments = [segment.lower() for segment in path.split("/") if segment]
        if not segments or not LOCALE_SEGMENT.fullmatch(segments[0]):
            return None
        return segments[0]

    @staticmethod
    def _path_has_legacy_segment(path: str) -> bool:
        segments = {segment.lower() for segment in path.split("/") if segment}
        return bool(segments & LEGACY_PATH_SEGMENTS)

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
    def _title_from_url(url: str) -> str:
        path = urlparse(url).path.strip("/")
        if not path:
            return urlparse(url).hostname or "Documentation page"
        return path.rsplit("/", 1)[-1].replace("-", " ").replace("_", " ").title()

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")
        return slug[:60]

    @staticmethod
    def _firecrawl_status_message(status_code: int) -> str:
        if status_code == 401:
            return "Firecrawl rejected the key."
        if status_code == 402:
            return "Firecrawl account requires payment or credits."
        if status_code == 429:
            return "Firecrawl rate limit was reached."
        return f"Firecrawl returned HTTP {status_code}."
