from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from firecrawl_docs_auditor.config import Settings
from firecrawl_docs_auditor.firecrawl_preflight import (
    FirecrawlPreflightRequest,
    HttpFirecrawlPreflightClient,
)


def test_settings_accept_firecrawl_key_alias() -> None:
    settings = Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None)

    assert settings.firecrawl_api_key == "fc-test-secret"


def test_key_status_is_local_only_for_missing_and_configured_keys() -> None:
    missing = HttpFirecrawlPreflightClient(Settings(_env_file=None))
    configured = HttpFirecrawlPreflightClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None)
    )

    assert asyncio.run(missing.key_status()).model_dump(mode="json") == {
        "status": "missing_key",
        "configured": False,
        "message": "Firecrawl key missing; enter one in the UI or set FIRECRAWL_API_KEY.",
    }
    assert asyncio.run(configured.key_status()).status == "configured"


def test_preflight_request_accepts_50_pages_but_rejects_51() -> None:
    request = FirecrawlPreflightRequest(
        docs_url="https://docs.example.com/docs/start",
        integration_goal="Build checkout",
        max_pages=50,
    )

    assert request.max_pages == 50
    with pytest.raises(ValidationError):
        FirecrawlPreflightRequest(
            docs_url="https://docs.example.com/docs/start",
            integration_goal="Build checkout",
            max_pages=51,
        )


def test_preflight_without_key_warns_but_checks_public_url() -> None:
    client = HttpFirecrawlPreflightClient(Settings(_env_file=None), transport=_transport())

    result = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout",
            )
        )
    )

    assert result.verdict == "warning"
    assert result.key_status.status == "missing"
    assert result.normalized_url == "https://docs.example.com/docs/start"
    assert result.allowed_hosts == ["docs.example.com"]
    assert _check(result, "public_access").status == "pass"
    assert _check(result, "llms_txt").status == "warning"


def test_private_and_non_http_urls_are_blocked_without_network() -> None:
    client = HttpFirecrawlPreflightClient(Settings(_env_file=None), transport=_transport())

    localhost = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="http://127.0.0.1:8000/docs",
                integration_goal="Build checkout",
            )
        )
    )
    file_url = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="file:///tmp/docs",
                integration_goal="Build checkout",
            )
        )
    )

    assert localhost.verdict == "blocked"
    assert _check(localhost, "url").message == "Docs URL points to a local or private host."
    assert file_url.verdict == "blocked"
    assert _check(file_url, "url").message == "Docs URL must use http or https."


def test_embedded_credentials_and_private_allowed_hosts_are_blocked() -> None:
    client = HttpFirecrawlPreflightClient(Settings(_env_file=None), transport=_transport())

    with_credentials = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://user:pass@docs.example.com/docs",
                integration_goal="Build checkout",
            )
        )
    )
    private_allowed_host = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/docs",
                integration_goal="Build checkout",
                allowed_hosts=["docs.example.com", "127.0.0.1"],
            )
        )
    )
    credential_allowed_host = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/docs",
                integration_goal="Build checkout",
                allowed_hosts=["https://user:pass@docs.example.com"],
            )
        )
    )

    assert with_credentials.verdict == "blocked"
    assert _check(with_credentials, "url").message == "Docs URL must not include embedded credentials."
    assert "user:pass" not in json.dumps(with_credentials.model_dump(mode="json")).lower()
    assert private_allowed_host.verdict == "blocked"
    assert _check(private_allowed_host, "url").message == "Allowed host is invalid or private."
    assert credential_allowed_host.verdict == "blocked"
    assert _check(credential_allowed_host, "url").message == "Allowed host is invalid or private."
    assert "user:pass" not in json.dumps(credential_allowed_host.model_dump(mode="json")).lower()


def test_blocked_url_does_not_validate_firecrawl_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected network call to {request.url}")

    client = HttpFirecrawlPreflightClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/docs",
                integration_goal="Build checkout",
                allowed_hosts=["127.0.0.1"],
            )
        )
    )

    assert result.verdict == "blocked"
    assert result.key_status.status == "configured"


def test_robots_disallow_blocks_requested_path() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /docs\nCrawl-delay: 2\n",
                request=request,
            )
        return _default_response(request)

    client = HttpFirecrawlPreflightClient(
        Settings(_env_file=None),
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout",
            )
        )
    )

    assert result.verdict == "blocked"
    assert _check(result, "robots_txt").status == "blocked"
    assert _check(result, "robots_crawl_delay").status == "warning"


def test_allow_rule_can_override_shorter_disallow() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /docs\nAllow: /docs/public\n",
                request=request,
            )
        return _default_response(request)

    client = HttpFirecrawlPreflightClient(
        Settings(_env_file=None),
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/docs/public/start",
                integration_goal="Build checkout",
            )
        )
    )

    assert _check(result, "robots_txt").status == "pass"


def test_sitemap_check_uses_robots_candidate_before_origin_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nSitemap: https://docs.example.com/custom-sitemap.xml\n",
                request=request,
            )
        if request.url.path == "/custom-sitemap.xml":
            return httpx.Response(
                200,
                text="<urlset></urlset>",
                headers={"content-type": "application/xml"},
                request=request,
            )
        if request.url.path == "/sitemap.xml":
            return httpx.Response(404, request=request)
        return _default_response(request)

    client = HttpFirecrawlPreflightClient(
        Settings(_env_file=None),
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout",
            )
        )
    )

    sitemap_check = _check(result, "sitemap")
    assert sitemap_check.status == "pass"
    assert sitemap_check.url == "https://docs.example.com/custom-sitemap.xml"


def test_sitemap_check_ignores_private_robots_candidate() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nSitemap: http://127.0.0.1/private-sitemap.xml\n",
                request=request,
            )
        if request.url.path == "/sitemap.xml":
            return httpx.Response(404, request=request)
        return _default_response(request)

    client = HttpFirecrawlPreflightClient(
        Settings(_env_file=None),
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout",
            )
        )
    )

    assert "http://127.0.0.1/private-sitemap.xml" not in requested_paths
    sitemap_check = _check(result, "sitemap")
    assert sitemap_check.status == "warning"
    assert sitemap_check.url == "https://docs.example.com/sitemap.xml"


def test_sitemap_check_ignores_credentialed_robots_candidate() -> None:
    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nSitemap: https://user:pass@docs.example.com/sitemap.xml\n",
                request=request,
            )
        if request.url.path == "/sitemap.xml":
            return httpx.Response(404, request=request)
        return _default_response(request)

    client = HttpFirecrawlPreflightClient(
        Settings(_env_file=None),
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout",
            )
        )
    )

    assert "https://user:pass@docs.example.com/sitemap.xml" not in requested_urls
    serialized = json.dumps(result.model_dump(mode="json")).lower()
    assert "user:pass" not in serialized
    assert not any(check.id == "robots_sitemap" for check in result.checks)
    assert _check(result, "sitemap").url == "https://docs.example.com/sitemap.xml"


def test_valid_key_and_clean_entrypoints_can_pass() -> None:
    client = HttpFirecrawlPreflightClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        transport=_transport(llms_status=200),
    )

    result = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/docs/start",
                integration_goal="Build checkout",
                allowed_hosts=["docs.example.com"],
            )
        )
    )

    assert result.verdict == "pass"
    assert result.key_status.status == "valid"
    serialized = json.dumps(result.model_dump(mode="json")).lower()
    assert "fc-test-secret" not in serialized
    assert "api_key" not in serialized
    assert "secret" not in serialized


def test_component_name_captcha_provider_does_not_block_public_access() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.firecrawl.dev" and request.url.path == "/v2/scrape":
            return httpx.Response(200, json={"success": True}, request=request)
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(404, request=request)
        if request.url.path.endswith("/llms.txt"):
            return httpx.Response(404, request=request)
        return httpx.Response(
            200,
            text='<script>self.__next_f.push(["CaptchaProvider"])</script><main>Docs</main>',
            request=request,
        )

    client = HttpFirecrawlPreflightClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/",
                integration_goal="Build checkout",
            )
        )
    )

    assert _check(result, "public_access").status == "pass"
    assert result.verdict == "warning"


def test_real_captcha_challenge_still_blocks_public_access() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.firecrawl.dev" and request.url.path == "/v2/scrape":
            return httpx.Response(200, json={"success": True}, request=request)
        if request.url.path == "/robots.txt":
            return httpx.Response(404, request=request)
        if request.url.path == "/sitemap.xml":
            return httpx.Response(404, request=request)
        if request.url.path.endswith("/llms.txt"):
            return httpx.Response(404, request=request)
        return httpx.Response(
            200,
            text="<html><title>Captcha required</title><p>Verify you are human.</p></html>",
            request=request,
        )

    client = HttpFirecrawlPreflightClient(
        Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.run_preflight(
            FirecrawlPreflightRequest(
                docs_url="https://docs.example.com/",
                integration_goal="Build checkout",
            )
        )
    )

    assert _check(result, "public_access").status == "blocked"
    assert result.verdict == "blocked"


def test_firecrawl_validation_maps_common_http_errors() -> None:
    statuses = {401: "invalid", 402: "payment_required", 429: "rate_limited", 500: "error"}
    for http_status, expected_status in statuses.items():
        client = HttpFirecrawlPreflightClient(
            Settings(FIRECRAWL_API_KEY="fc-test-secret", _env_file=None),
            transport=_transport(scrape_status=http_status),
        )

        result = asyncio.run(
            client.run_preflight(
                FirecrawlPreflightRequest(
                    docs_url="https://docs.example.com/docs/start",
                    integration_goal="Build checkout",
                )
            )
        )

        assert result.key_status.status == expected_status


def _check(result, check_id: str):
    return next(check for check in result.checks if check.id == check_id)


def _transport(scrape_status: int = 200, llms_status: int = 404) -> httpx.MockTransport:
    return httpx.MockTransport(
        lambda request: _response_for_request(
            request,
            scrape_status=scrape_status,
            llms_status=llms_status,
        )
    )


def _response_for_request(
    request: httpx.Request,
    *,
    scrape_status: int,
    llms_status: int,
) -> httpx.Response:
    if request.url.host == "api.firecrawl.dev" and request.url.path == "/v2/scrape":
        return httpx.Response(scrape_status, json={"success": scrape_status < 400}, request=request)
    if request.url.path == "/robots.txt":
        return httpx.Response(
            200,
            text="User-agent: *\nSitemap: https://docs.example.com/sitemap.xml\n",
            request=request,
        )
    if request.url.path == "/sitemap.xml":
        return httpx.Response(
            200,
            text="<urlset></urlset>",
            headers={"content-type": "application/xml"},
            request=request,
        )
    if request.url.path.endswith("/llms.txt"):
        return httpx.Response(llms_status, text="# Docs" if llms_status == 200 else "", request=request)
    return _default_response(request)


def _default_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="<html>Docs</html>", request=request)
