from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .codex_app_server import CodexAppServerClient, CodexStructuredJsonRequest
from .firecrawl_ingestion import (
    EvidenceRole,
    SourcePlannerProbe,
    SourcePlannerRejection,
    SourcePlannerResult,
    SourcePlannerSelection,
)


class _PlannerResponse(BaseModel):
    selected_sources: list[SourcePlannerSelection] = Field(default_factory=list)
    rejected_sources: list[SourcePlannerRejection] = Field(default_factory=list)
    suggested_probe_urls: list[SourcePlannerProbe] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CodexSourcePlanner:
    def __init__(self, codex_client: CodexAppServerClient) -> None:
        self._codex_client = codex_client

    async def plan_sources(
        self,
        *,
        docs_url: str,
        integration_goal: str,
        max_pages: int,
        allowed_hosts: list[str],
        candidates: list[dict[str, Any]],
    ) -> SourcePlannerResult:
        if not candidates:
            return SourcePlannerResult(
                status="fallback",
                warnings=["No candidate URLs were available for source planning."],
            )
        prompt = json.dumps(
            {
                "audit_request": {
                    "docs_url": docs_url,
                    "integration_goal": integration_goal,
                    "max_pages": max_pages,
                    "allowed_hosts": allowed_hosts,
                },
                "instructions": [
                    "Select the smallest role-balanced set of candidate IDs needed to audit the integration goal.",
                    "Prefer current canonical quickstarts, SDK pages, feature guides, and API references.",
                    "Avoid legacy, deprecated, migration, localized duplicate, marketing, or weakly related pages when current alternatives exist.",
                    "Do not invent candidate IDs.",
                    "Suggest probe URLs only when a likely same-host docs page is missing from the catalog.",
                    "Assign evidence roles that explain why each URL is useful.",
                ],
                "allowed_evidence_roles": list(EvidenceRole.__args__),
                "candidates": candidates[:100],
            },
            separators=(",", ":"),
        )
        result = await self._codex_client.run_structured_json(
            CodexStructuredJsonRequest(
                base_instructions=(
                    "You choose public documentation URLs for a docs-readiness audit. "
                    "Return only JSON matching the supplied schema."
                ),
                prompt=prompt,
                output_schema=self._output_schema(max_pages),
            )
        )
        if result.status != "pass" or result.response is None:
            return SourcePlannerResult(
                status="fallback",
                warnings=[f"Codex source planner unavailable: {result.message}"],
            )
        try:
            parsed = _PlannerResponse.model_validate(result.response)
        except ValidationError:
            return SourcePlannerResult(
                status="fallback",
                warnings=["Codex source planner returned invalid JSON shape."],
            )
        return SourcePlannerResult(
            status="planned",
            selected_sources=parsed.selected_sources[:max_pages],
            rejected_sources=parsed.rejected_sources,
            suggested_probe_urls=parsed.suggested_probe_urls,
            warnings=parsed.warnings,
        )

    @staticmethod
    def _output_schema(max_pages: int) -> dict[str, Any]:
        role_schema = {
            "type": "array",
            "maxItems": 4,
            "items": {"type": "string", "enum": list(EvidenceRole.__args__)},
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "selected_sources",
                "rejected_sources",
                "suggested_probe_urls",
                "warnings",
            ],
            "properties": {
                "selected_sources": {
                    "type": "array",
                    "maxItems": max_pages,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "candidate_id",
                            "evidence_roles",
                            "rationale",
                            "confidence",
                        ],
                        "properties": {
                            "candidate_id": {"type": "string", "maxLength": 40},
                            "evidence_roles": role_schema,
                            "rationale": {"type": "string", "maxLength": 240},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                    },
                },
                "rejected_sources": {
                    "type": "array",
                    "maxItems": 50,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["candidate_id", "reason"],
                        "properties": {
                            "candidate_id": {"type": "string", "maxLength": 40},
                            "reason": {"type": "string", "maxLength": 240},
                        },
                    },
                },
                "suggested_probe_urls": {
                    "type": "array",
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["url", "evidence_roles", "rationale"],
                        "properties": {
                            "url": {"type": "string", "maxLength": 500},
                            "evidence_roles": role_schema,
                            "rationale": {"type": "string", "maxLength": 240},
                        },
                    },
                },
                "warnings": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {"type": "string", "maxLength": 240},
                },
            },
        }
