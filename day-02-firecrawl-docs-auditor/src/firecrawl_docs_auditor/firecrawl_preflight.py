from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from pydantic import BaseModel, Field

from .config import Settings


Verdict = Literal["pass", "warning", "blocked"]
CheckStatus = Literal["pass", "warning", "blocked", "skipped"]
CheckSeverity = Literal["info", "warning", "blocking"]
MAX_AUDIT_PAGES = 50


class FirecrawlServiceStatus(BaseModel):
    status: Literal["missing_key", "configured"]
    configured: bool
    message: str


class FirecrawlKeyStatus(BaseModel):
    status: Literal[
        "missing",
        "configured",
        "valid",
        "invalid",
        "rate_limited",
        "payment_required",
        "error",
    ]
    configured: bool
    message: str


class FirecrawlPreflightRequest(BaseModel):
    docs_url: str = Field(min_length=1)
    integration_goal: str = Field(min_length=1)
    max_pages: int = Field(default=20, ge=1, le=MAX_AUDIT_PAGES)
    max_depth: int = Field(default=1, ge=0, le=3)
    allowed_hosts: list[str] | None = None
    firecrawl_api_key: str | None = None


class PreflightCheck(BaseModel):
    id: str
    status: CheckStatus
    severity: CheckSeverity
    message: str
    url: str | None = None


class DocsPreflightResult(BaseModel):
    verdict: Verdict
    normalized_url: str | None = None
    allowed_hosts: list[str] = Field(default_factory=list)
    key_status: FirecrawlKeyStatus
    checks: list[PreflightCheck] = Field(default_factory=list)


class FirecrawlPreflightClient(Protocol):
    async def key_status(self) -> FirecrawlServiceStatus:
        ...

    async def run_preflight(
        self,
        request: FirecrawlPreflightRequest,
    ) -> DocsPreflightResult:
        ...


@dataclass(frozen=True)
class _NormalizedUrl:
    original: str
    normalized: str
    scheme: str
    host: str
    path: str
    origin: str
    allowed_hosts: list[str]


def normalize_allowed_host_value(value: str) -> tuple[str, str | None]:
    raw_host = value.strip().lower()
    if "/" in raw_host or "@" in raw_host:
        parsed = urlparse(raw_host)
        if parsed.hostname and not (parsed.username or parsed.password):
            return parsed.hostname.lower(), None
        return "", raw_host
    if ":" in raw_host and raw_host.count(":") == 1:
        raw_host = raw_host.split(":", 1)[0]
    return raw_host.strip("[]"), None


def is_private_host(host: str) -> bool:
    if host in {"localhost", "0.0.0.0"}:
        return True
    if host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return any(
        [
            address.is_private,
            address.is_loopback,
            address.is_link_local,
            address.is_multicast,
            address.is_unspecified,
            address.is_reserved,
        ]
    )


class HttpFirecrawlPreflightClient:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._timeout = httpx.Timeout(settings.firecrawl_timeout_seconds)

    async def key_status(self) -> FirecrawlServiceStatus:
        if self._configured_key(self._settings.firecrawl_api_key):
            return FirecrawlServiceStatus(
                status="configured",
                configured=True,
                message="Firecrawl key configured; live validation runs during preflight.",
            )
        return FirecrawlServiceStatus(
            status="missing_key",
            configured=False,
            message="Firecrawl key missing; enter one in the UI or set FIRECRAWL_API_KEY.",
        )

    async def run_preflight(
        self,
        request: FirecrawlPreflightRequest,
    ) -> DocsPreflightResult:
        api_key = self._resolve_key(request.firecrawl_api_key)
        normalized, normalize_check = self._normalize_url(request.docs_url, request.allowed_hosts)
        checks = [normalize_check]
        if normalized is None:
            return DocsPreflightResult(
                verdict="blocked",
                key_status=self._skipped_key_status(api_key),
                checks=checks,
            )

        key_status = await self._validate_key(api_key)
        if key_status.status == "missing":
            checks.append(
                PreflightCheck(
                    id="firecrawl_key",
                    status="warning",
                    severity="warning",
                    message="Firecrawl key is missing; URL preflight can continue but live fetching cannot.",
                )
            )
        elif key_status.status in {"invalid", "payment_required", "rate_limited", "error"}:
            checks.append(
                PreflightCheck(
                    id="firecrawl_key",
                    status="warning",
                    severity="warning",
                    message=key_status.message,
                )
            )

        async with self._http_client(follow_redirects=False) as client:
            robots_checks, sitemap_candidates = await self._check_robots_txt(
                client,
                normalized,
            )
            checks.extend(
                [
                    await self._check_public_access(client, normalized),
                    *robots_checks,
                    await self._check_sitemap(client, normalized, sitemap_candidates),
                    *await self._check_llms_txt(client, normalized),
                ]
            )

        return DocsPreflightResult(
            verdict=self._compute_verdict(checks),
            normalized_url=normalized.normalized,
            allowed_hosts=normalized.allowed_hosts,
            key_status=key_status,
            checks=checks,
        )

    def _http_client(self, *, follow_redirects: bool) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=follow_redirects,
            headers={"User-Agent": "FirecrawlDocsAuditor/0.1"},
            transport=self._transport,
        )

    def _resolve_key(self, request_key: str | None) -> str | None:
        if self._configured_key(request_key):
            return request_key.strip()
        if self._configured_key(self._settings.firecrawl_api_key):
            return self._settings.firecrawl_api_key.strip()
        return None

    @staticmethod
    def _configured_key(key: str | None) -> bool:
        return bool(key and key.strip())

    @staticmethod
    def _skipped_key_status(api_key: str | None) -> FirecrawlKeyStatus:
        if api_key:
            return FirecrawlKeyStatus(
                status="configured",
                configured=True,
                message="Firecrawl key configured; live validation skipped because URL preflight is blocked.",
            )
        return FirecrawlKeyStatus(
            status="missing",
            configured=False,
            message="Firecrawl key missing.",
        )

    async def _validate_key(self, api_key: str | None) -> FirecrawlKeyStatus:
        if not api_key:
            return FirecrawlKeyStatus(
                status="missing",
                configured=False,
                message="Firecrawl key missing.",
            )

        url = f"{self._settings.firecrawl_api_base_url.rstrip('/')}/scrape"
        try:
            async with self._http_client(follow_redirects=False) as client:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"url": "https://docs.firecrawl.dev", "formats": ["markdown"]},
                )
        except httpx.HTTPError:
            return FirecrawlKeyStatus(
                status="error",
                configured=True,
                message="Firecrawl key validation failed due to a network error.",
            )

        if 200 <= response.status_code < 300:
            return FirecrawlKeyStatus(
                status="valid",
                configured=True,
                message="Firecrawl key validated.",
            )
        if response.status_code == 401:
            status = "invalid"
            message = "Firecrawl rejected the key."
        elif response.status_code == 402:
            status = "payment_required"
            message = "Firecrawl account requires payment or credits."
        elif response.status_code == 429:
            status = "rate_limited"
            message = "Firecrawl key is rate limited."
        else:
            status = "error"
            message = f"Firecrawl returned HTTP {response.status_code} during key validation."
        return FirecrawlKeyStatus(status=status, configured=True, message=message)

    def _normalize_url(
        self,
        docs_url: str,
        allowed_hosts: list[str] | None,
    ) -> tuple[_NormalizedUrl | None, PreflightCheck]:
        parsed = urlparse(docs_url.strip())
        if parsed.scheme not in {"http", "https"}:
            return None, PreflightCheck(
                id="url",
                status="blocked",
                severity="blocking",
                message="Docs URL must use http or https.",
                url=docs_url,
            )
        if not parsed.hostname:
            return None, PreflightCheck(
                id="url",
                status="blocked",
                severity="blocking",
                message="Docs URL must include a host.",
                url=docs_url,
            )
        if parsed.username or parsed.password:
            return None, PreflightCheck(
                id="url",
                status="blocked",
                severity="blocking",
                message="Docs URL must not include embedded credentials.",
                url=None,
            )

        host = parsed.hostname.lower()
        if is_private_host(host):
            return None, PreflightCheck(
                id="url",
                status="blocked",
                severity="blocking",
                message="Docs URL points to a local or private host.",
                url=docs_url,
            )

        normalized_hosts, blocked_allowed_host = self._normalize_allowed_hosts(
            allowed_hosts,
            host,
        )
        if blocked_allowed_host:
            return None, PreflightCheck(
                id="url",
                status="blocked",
                severity="blocking",
                message="Allowed host is invalid or private.",
                url=docs_url,
            )
        if host not in normalized_hosts:
            normalized_hosts.insert(0, host)

        normalized = urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path or "/",
                "",
                parsed.query,
                "",
            )
        )
        return _NormalizedUrl(
            original=docs_url,
            normalized=normalized,
            scheme=parsed.scheme.lower(),
            host=host,
            path=parsed.path or "/",
            origin=f"{parsed.scheme.lower()}://{parsed.netloc.lower()}",
            allowed_hosts=normalized_hosts,
        ), PreflightCheck(
            id="url",
            status="pass",
            severity="info",
            message=f"Docs URL normalized with host boundary {host}.",
            url=normalized,
        )

    def _normalize_allowed_hosts(
        self,
        allowed_hosts: list[str] | None,
        fallback: str,
    ) -> tuple[list[str], str | None]:
        hosts = []
        for value in allowed_hosts or [fallback]:
            host, invalid_host = normalize_allowed_host_value(value)
            if invalid_host:
                return [], invalid_host
            if is_private_host(host):
                return [], host
            if host and host not in hosts:
                hosts.append(host)
        return hosts, None

    async def _check_public_access(
        self,
        client: httpx.AsyncClient,
        normalized: _NormalizedUrl,
    ) -> PreflightCheck:
        try:
            response = await client.get(normalized.normalized)
        except httpx.HTTPError:
            return PreflightCheck(
                id="public_access",
                status="warning",
                severity="warning",
                message="Docs URL could not be reached during preflight.",
                url=normalized.normalized,
            )

        redirect_host = self._redirect_host(response)
        if redirect_host and redirect_host != normalized.host:
            return PreflightCheck(
                id="public_access",
                status="warning",
                severity="warning",
                message=f"Docs URL redirects to a different host: {redirect_host}.",
                url=normalized.normalized,
            )
        if response.status_code == 200:
            if self._looks_like_captcha(response.text):
                return PreflightCheck(
                    id="public_access",
                    status="blocked",
                    severity="blocking",
                    message="Docs URL appears to present a bot challenge or CAPTCHA.",
                    url=normalized.normalized,
                )
            return PreflightCheck(
                id="public_access",
                status="pass",
                severity="info",
                message="Docs URL is publicly accessible.",
                url=normalized.normalized,
            )
        if response.status_code in {401, 403}:
            return PreflightCheck(
                id="public_access",
                status="blocked",
                severity="blocking",
                message="Docs URL requires authentication or denies access.",
                url=normalized.normalized,
            )
        if response.status_code == 404:
            return PreflightCheck(
                id="public_access",
                status="blocked",
                severity="blocking",
                message="Docs URL was not found.",
                url=normalized.normalized,
            )
        if response.status_code == 429:
            return PreflightCheck(
                id="public_access",
                status="warning",
                severity="warning",
                message="Docs URL is rate limited.",
                url=normalized.normalized,
            )
        return PreflightCheck(
            id="public_access",
            status="warning",
            severity="warning",
            message=f"Docs URL returned HTTP {response.status_code}.",
            url=normalized.normalized,
        )

    async def _check_robots_txt(
        self,
        client: httpx.AsyncClient,
        normalized: _NormalizedUrl,
    ) -> tuple[list[PreflightCheck], list[str]]:
        robots_url = urljoin(normalized.origin, "/robots.txt")
        try:
            response = await client.get(robots_url)
        except httpx.HTTPError:
            return [
                PreflightCheck(
                    id="robots_txt",
                    status="warning",
                    severity="warning",
                    message="robots.txt could not be reached.",
                    url=robots_url,
                )
            ], []
        if response.status_code == 404:
            return [
                PreflightCheck(
                    id="robots_txt",
                    status="warning",
                    severity="warning",
                    message="robots.txt is missing.",
                    url=robots_url,
                )
            ], []
        if response.status_code != 200:
            return [
                PreflightCheck(
                    id="robots_txt",
                    status="warning",
                    severity="warning",
                    message=f"robots.txt returned HTTP {response.status_code}.",
                    url=robots_url,
                )
            ], []

        policy = self._parse_robots_txt(response.text, normalized.path)
        checks = [
            PreflightCheck(
                id="robots_txt",
                status="blocked" if policy["blocked"] else "pass",
                severity="blocking" if policy["blocked"] else "info",
                message=(
                    "robots.txt disallows the requested docs path."
                    if policy["blocked"]
                    else "robots.txt is present and does not clearly block the docs path."
                ),
                url=robots_url,
            )
        ]
        if policy["crawl_delay"]:
            checks.append(
                PreflightCheck(
                    id="robots_crawl_delay",
                    status="warning",
                    severity="warning",
                    message=f"robots.txt declares Crawl-delay: {policy['crawl_delay']}.",
                    url=robots_url,
                )
            )
        safe_sitemaps = self._safe_sitemap_candidates(
            normalized,
            policy["sitemaps"],
            include_origin_default=False,
        )
        if safe_sitemaps:
            checks.append(
                PreflightCheck(
                    id="robots_sitemap",
                    status="pass",
                    severity="info",
                    message=f"robots.txt lists {len(safe_sitemaps)} sitemap candidate(s).",
                    url=safe_sitemaps[0],
                )
            )
        return checks, policy["sitemaps"]

    async def _check_sitemap(
        self,
        client: httpx.AsyncClient,
        normalized: _NormalizedUrl,
        sitemap_candidates: list[str],
    ) -> PreflightCheck:
        candidates = self._safe_sitemap_candidates(
            normalized,
            sitemap_candidates,
            include_origin_default=True,
        )
        seen_candidates = []
        for candidate in candidates:
            if candidate not in seen_candidates:
                seen_candidates.append(candidate)
        last_check: PreflightCheck | None = None
        for sitemap_url in seen_candidates:
            check = await self._check_sitemap_candidate(client, sitemap_url)
            if check.status == "pass":
                return check
            last_check = check
        return last_check or PreflightCheck(
            id="sitemap",
            status="warning",
            severity="warning",
            message="Sitemap could not be reached.",
            url=urljoin(normalized.origin, "/sitemap.xml"),
        )

    def _safe_sitemap_candidates(
        self,
        normalized: _NormalizedUrl,
        sitemap_candidates: list[str],
        *,
        include_origin_default: bool,
    ) -> list[str]:
        candidates = []
        candidate_urls = [*sitemap_candidates]
        if include_origin_default:
            candidate_urls.append(urljoin(normalized.origin, "/sitemap.xml"))
        for candidate in candidate_urls:
            resolved = urljoin(normalized.origin, candidate)
            parsed = urlparse(resolved)
            host = parsed.hostname.lower() if parsed.hostname else ""
            if (
                parsed.scheme in {"http", "https"}
                and not (parsed.username or parsed.password)
                and host in normalized.allowed_hosts
                and not is_private_host(host)
            ):
                candidates.append(
                    urlunparse(
                        (
                            parsed.scheme,
                            parsed.netloc.lower(),
                            parsed.path or "/",
                            "",
                            parsed.query,
                            "",
                        )
                    )
                )
        return candidates

    async def _check_sitemap_candidate(
        self,
        client: httpx.AsyncClient,
        sitemap_url: str,
    ) -> PreflightCheck:
        try:
            response = await client.get(sitemap_url)
        except httpx.HTTPError:
            return PreflightCheck(
                id="sitemap",
                status="warning",
                severity="warning",
                message="Sitemap could not be reached.",
                url=sitemap_url,
            )
        content_type = response.headers.get("content-type", "")
        if response.status_code == 200 and (
            "xml" in content_type or "text" in content_type or response.text.lstrip().startswith("<")
        ):
            return PreflightCheck(
                id="sitemap",
                status="pass",
                severity="info",
                message="Sitemap candidate is reachable.",
                url=sitemap_url,
            )
        if response.status_code == 404:
            return PreflightCheck(
                id="sitemap",
                status="warning",
                severity="warning",
                message="Sitemap candidate is missing.",
                url=sitemap_url,
            )
        return PreflightCheck(
            id="sitemap",
            status="warning",
            severity="warning",
            message=f"Sitemap candidate returned HTTP {response.status_code}.",
            url=sitemap_url,
        )

    async def _check_llms_txt(
        self,
        client: httpx.AsyncClient,
        normalized: _NormalizedUrl,
    ) -> list[PreflightCheck]:
        candidates = [urljoin(normalized.origin, "/llms.txt")]
        first_segment = normalized.path.strip("/").split("/", 1)[0]
        if first_segment == "docs":
            docs_candidate = urljoin(normalized.origin, "/docs/llms.txt")
            if docs_candidate not in candidates:
                candidates.append(docs_candidate)

        checks = []
        for candidate in candidates:
            try:
                response = await client.get(candidate)
            except httpx.HTTPError:
                checks.append(
                    PreflightCheck(
                        id="llms_txt",
                        status="warning",
                        severity="warning",
                        message="llms.txt could not be reached.",
                        url=candidate,
                    )
                )
                continue
            if response.status_code == 200:
                checks.append(
                    PreflightCheck(
                        id="llms_txt",
                        status="pass",
                        severity="info",
                        message="llms.txt is present.",
                        url=candidate,
                    )
                )
            elif response.status_code == 404:
                checks.append(
                    PreflightCheck(
                        id="llms_txt",
                        status="warning",
                        severity="warning",
                        message="llms.txt is missing.",
                        url=candidate,
                    )
                )
            else:
                checks.append(
                    PreflightCheck(
                        id="llms_txt",
                        status="warning",
                        severity="warning",
                        message=f"llms.txt returned HTTP {response.status_code}.",
                        url=candidate,
                    )
                )
        return checks

    @staticmethod
    def _parse_robots_txt(text: str, requested_path: str) -> dict[str, Any]:
        active_star_group = False
        disallows: list[str] = []
        allows: list[str] = []
        crawl_delay: str | None = None
        sitemaps: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            key = key.lower()
            if key == "sitemap" and value:
                sitemaps.append(value)
                continue
            if key == "user-agent":
                active_star_group = value == "*"
                continue
            if not active_star_group:
                continue
            if key == "disallow" and value:
                disallows.append(value)
            elif key == "allow" and value:
                allows.append(value)
            elif key == "crawl-delay" and value and crawl_delay is None:
                crawl_delay = value

        matched_disallow = _best_prefix(disallows, requested_path)
        matched_allow = _best_prefix(allows, requested_path)
        blocked = bool(matched_disallow) and len(matched_disallow) > len(matched_allow or "")
        return {"blocked": blocked, "crawl_delay": crawl_delay, "sitemaps": sitemaps}

    @staticmethod
    def _redirect_host(response: httpx.Response) -> str | None:
        if response.status_code not in {301, 302, 303, 307, 308}:
            return None
        location = response.headers.get("location")
        if not location:
            return None
        parsed = urlparse(location)
        return parsed.hostname.lower() if parsed.hostname else None

    @staticmethod
    def _looks_like_captcha(text: str) -> bool:
        lowered = text.lower()
        challenge_patterns = [
            r"\bcf-challenge\b",
            r"\bbot challenge\b",
            r"\bcaptcha\b.{0,120}\b(required|verification|verify|challenge|protected)\b",
            r"\b(required|verification|verify|challenge|protected)\b.{0,120}\bcaptcha\b",
            r"\bverify you are human\b",
            r"\bchecking your browser\b",
        ]
        return any(re.search(pattern, lowered) for pattern in challenge_patterns)

    @staticmethod
    def _compute_verdict(checks: list[PreflightCheck]) -> Verdict:
        if any(check.status == "blocked" for check in checks):
            return "blocked"
        if any(check.status == "warning" for check in checks):
            return "warning"
        return "pass"


def _best_prefix(prefixes: list[str], path: str) -> str | None:
    matches = [prefix for prefix in prefixes if path.startswith(prefix)]
    if not matches:
        return None
    return max(matches, key=len)
