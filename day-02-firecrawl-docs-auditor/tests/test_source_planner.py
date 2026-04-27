from __future__ import annotations

import asyncio

from firecrawl_docs_auditor.codex_app_server import (
    CodexStructuredJsonRequest,
    CodexStructuredJsonResult,
)
from firecrawl_docs_auditor.source_planner import CodexSourcePlanner


class FakeCodexPlannerClient:
    def __init__(
        self,
        response: dict[str, object] | None,
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
            response=self.response,
        )


def test_codex_source_planner_normalizes_valid_response() -> None:
    codex = FakeCodexPlannerClient(
        {
            "selected_sources": [
                {
                    "candidate_id": "cand_2",
                    "evidence_roles": ["install", "auth"],
                    "rationale": "Current setup page.",
                    "confidence": "high",
                }
            ],
            "rejected_sources": [
                {"candidate_id": "cand_3", "reason": "Legacy duplicate."}
            ],
            "suggested_probe_urls": [
                {
                    "url": "https://docs.example.com/quickstarts/python",
                    "evidence_roles": ["setup"],
                    "rationale": "Likely canonical quickstart.",
                }
            ],
            "warnings": ["One candidate looked stale."],
        }
    )
    planner = CodexSourcePlanner(codex)

    result = asyncio.run(
        planner.plan_sources(
            docs_url="https://docs.example.com/",
            integration_goal="Set up Python auth",
            max_pages=5,
            allowed_hosts=["docs.example.com"],
            candidates=[
                {
                    "id": "cand_1",
                    "url": "https://docs.example.com/",
                    "title": "Docs",
                    "path_flags": {},
                },
                {
                    "id": "cand_2",
                    "url": "https://docs.example.com/sdks/python",
                    "title": "Python SDK",
                    "path_flags": {},
                },
            ],
        )
    )

    assert result.status == "planned"
    assert result.selected_sources[0].candidate_id == "cand_2"
    assert result.selected_sources[0].evidence_roles == ["install", "auth"]
    assert result.suggested_probe_urls[0].url.endswith("/quickstarts/python")
    assert codex.requests
    assert "role-balanced" in codex.requests[0].prompt


def test_codex_source_planner_falls_back_on_unavailable_codex() -> None:
    planner = CodexSourcePlanner(
        FakeCodexPlannerClient(
            None,
            status="login_required",
            message="Login required.",
        )
    )

    result = asyncio.run(
        planner.plan_sources(
            docs_url="https://docs.example.com/",
            integration_goal="Set up Python auth",
            max_pages=5,
            allowed_hosts=["docs.example.com"],
            candidates=[{"id": "cand_1", "url": "https://docs.example.com/"}],
        )
    )

    assert result.status == "fallback"
    assert "Login required." in result.warnings[0]
