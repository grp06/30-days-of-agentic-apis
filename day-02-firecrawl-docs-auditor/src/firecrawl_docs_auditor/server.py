from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .audit_engine import (
    AuditEngineClient,
    AuditReportRequest,
    CodexAuditEngineClient,
)
from .codex_app_server import (
    CodexAppServerClient,
    CodexLoginCancelRequest,
    CodexLoginStartRequest,
    ManagedCodexAppServerClient,
)
from .config import Settings, get_settings
from .contracts import contracts_readable, load_sample_report, load_sample_request
from .firecrawl_preflight import (
    FirecrawlPreflightClient,
    FirecrawlPreflightRequest,
    HttpFirecrawlPreflightClient,
)
from .firecrawl_ingestion import (
    FirecrawlCacheNotFound,
    FirecrawlFetchRequest,
    FirecrawlIngestionClient,
    HttpFirecrawlIngestionClient,
)
from .source_planner import CodexSourcePlanner


SERVICE_NAME = "firecrawl-docs-auditor"
NOT_WIRED = "not_wired_in_this_slice"


def create_app(
    settings: Settings | None = None,
    codex_client: CodexAppServerClient | None = None,
    firecrawl_client: FirecrawlPreflightClient | None = None,
    firecrawl_ingestion_client: FirecrawlIngestionClient | None = None,
    audit_engine_client: AuditEngineClient | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_codex_client = codex_client or ManagedCodexAppServerClient(resolved_settings)
    resolved_firecrawl_client = firecrawl_client or HttpFirecrawlPreflightClient(
        resolved_settings
    )
    resolved_firecrawl_ingestion_client = (
        firecrawl_ingestion_client
        or HttpFirecrawlIngestionClient(
            resolved_settings,
            preflight_client=resolved_firecrawl_client,
            source_planner=CodexSourcePlanner(resolved_codex_client),
        )
    )
    resolved_audit_engine_client = audit_engine_client or CodexAuditEngineClient(
        resolved_settings,
        codex_client=resolved_codex_client,
        source_client=resolved_firecrawl_ingestion_client,
    )
    app = FastAPI(title="Firecrawl Docs Auditor API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.frontend_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"ok": True, "service": SERVICE_NAME}

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        codex_status = await resolved_codex_client.read_account()
        firecrawl_status = await resolved_firecrawl_client.key_status()
        return {
            "service": SERVICE_NAME,
            "api": {
                "host": resolved_settings.host,
                "port": resolved_settings.port,
            },
            "frontend": {
                "allowed_origins": resolved_settings.frontend_origins,
            },
            "contracts": {
                "fixtures_readable": contracts_readable(
                    project_root=resolved_settings.project_root
                ),
            },
            "codex_app_server": {
                "status": codex_status.status,
                "codex_bin_configured": codex_status.codex_bin_configured,
                "codex_bin_detected": codex_status.codex_bin_detected,
                "requires_openai_auth": codex_status.requires_openai_auth,
                "account": (
                    codex_status.account.model_dump(mode="json")
                    if codex_status.account
                    else None
                ),
                "message": codex_status.message,
            },
            "firecrawl": firecrawl_status.model_dump(mode="json"),
            "audit_engine": {
                "status": "report_generation_ready",
                "message": "Report generation is wired through Codex app-server.",
            },
        }

    @app.get("/api/contracts/sample-request")
    async def sample_request() -> dict[str, Any]:
        return load_sample_request(project_root=resolved_settings.project_root)

    @app.get("/api/contracts/sample-report")
    async def sample_report() -> dict[str, Any]:
        return load_sample_report(project_root=resolved_settings.project_root)

    @app.get("/api/firecrawl/status")
    async def firecrawl_status() -> dict[str, Any]:
        return (await resolved_firecrawl_client.key_status()).model_dump(mode="json")

    @app.post("/api/firecrawl/preflight")
    async def firecrawl_preflight(
        request: FirecrawlPreflightRequest,
    ) -> dict[str, Any]:
        return (await resolved_firecrawl_client.run_preflight(request)).model_dump(
            mode="json"
        )

    @app.post("/api/firecrawl/fetch-sources")
    async def firecrawl_fetch_sources(
        request: FirecrawlFetchRequest,
    ) -> dict[str, Any]:
        return (await resolved_firecrawl_ingestion_client.fetch_sources(request)).model_dump(
            mode="json"
        )

    @app.get("/api/firecrawl/fetch-sources/{cache_key}")
    async def firecrawl_cached_sources(cache_key: str) -> dict[str, Any]:
        try:
            result = await resolved_firecrawl_ingestion_client.read_cached_sources(cache_key)
        except FirecrawlCacheNotFound as exc:
            raise HTTPException(status_code=404, detail="Cached source fetch not found.") from exc
        return result.model_dump(mode="json")

    @app.post("/api/audit/report")
    async def audit_report(request: AuditReportRequest) -> dict[str, Any]:
        result = await resolved_audit_engine_client.generate_report(request)
        return result.model_dump(mode="json", exclude_none=True)

    @app.get("/api/codex/account")
    async def codex_account() -> dict[str, Any]:
        return (await resolved_codex_client.read_account()).model_dump(mode="json")

    @app.post("/api/codex/login/start")
    async def codex_login_start(request: CodexLoginStartRequest) -> dict[str, Any]:
        return (await resolved_codex_client.start_login(request.mode)).model_dump(
            mode="json"
        )

    @app.get("/api/codex/login/status/{login_id}")
    async def codex_login_status(login_id: str) -> dict[str, Any]:
        return (await resolved_codex_client.login_status(login_id)).model_dump(mode="json")

    @app.post("/api/codex/login/cancel")
    async def codex_login_cancel(request: CodexLoginCancelRequest) -> dict[str, Any]:
        return (await resolved_codex_client.cancel_login(request.login_id)).model_dump(
            mode="json"
        )

    @app.post("/api/codex/logout")
    async def codex_logout() -> dict[str, Any]:
        return (await resolved_codex_client.logout()).model_dump(mode="json")

    @app.post("/api/codex/smoke-test")
    async def codex_smoke_test() -> dict[str, Any]:
        return (await resolved_codex_client.run_prompt_json_smoke()).model_dump(
            mode="json"
        )

    return app
