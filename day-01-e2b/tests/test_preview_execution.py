from __future__ import annotations

import pytest

from agent_black_box.preview_execution import (
    build_preview_evidence,
    build_refresh_context,
    resolve_command_preview,
)
from agent_black_box.recorder import RunMetadata, RunStatus, RunSummary
from agent_black_box.sandbox_controller import CommandResult


class FakeSandboxController:
    def __init__(self) -> None:
        self.published_ports: list[int] = []

    def preview_port_for_command(self, command: str, *, stdout: str = "", stderr: str = "") -> int | None:
        if "--port 5173" in command or "localhost:5173" in stdout or "localhost:5173" in stderr:
            return 5173
        if "preview" in command or "dev" in command:
            return 4173
        return None

    async def publish_preview(
        self, port: int, *, readiness_timeout_seconds: int | None = None
    ) -> str:
        self.published_ports.append(port)
        return f"https://{port}-preview.example"


def test_build_preview_evidence_recovers_url_port_and_failure_from_history() -> None:
    summary = RunSummary(
        run_id="run-1",
        status="failed",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
        sandbox_id="sbx-1",
        preview_url=None,
        preview_state="expired",
        sandbox_retained=False,
    )
    event_lines = [
        '{"type":"tool_call","tool_name":"run_command","sequence":4,"arguments":{"command":"pnpm dev --host 0.0.0.0 --port 5173"}}',
        '{"type":"tool_result","tool_name":"run_command","sequence":5,"ok":false,"result":{"error":"context deadline exceeded"}}',
        '{"type":"preview_published","sequence":6,"url":"https://5173-preview.example","port":5173}',
    ]

    evidence = build_preview_evidence(
        status=None,
        summary=summary,
        event_lines=event_lines,
        default_preview_port=4173,
    )

    assert evidence.preview_url == "https://5173-preview.example"
    assert evidence.preview_port == 5173
    assert evidence.preview_failure_reason == "Preview command for port 5173 failed before publication: context deadline exceeded"


def test_build_refresh_context_uses_unified_preview_evidence() -> None:
    metadata = RunMetadata(
        run_id="run-1",
        task="Make the page and keep the preview working.",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
        sandbox_id="sbx-1",
    )
    status = RunStatus(
        run_id="run-1",
        state="succeeded",
        current_model_name="kimi-k2.6:cloud",
        preview_state="unavailable",
        sandbox_retained=False,
    )
    summary = RunSummary(
        run_id="run-1",
        status="succeeded",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
        sandbox_id="sbx-1",
        preview_url="https://5173-preview.example",
        preview_state="retained",
        sandbox_retained=True,
    )

    evidence = build_preview_evidence(
        status=status,
        summary=summary,
        event_lines=[],
        default_preview_port=4173,
    )
    refresh_context = build_refresh_context(
        metadata=metadata,
        status=status,
        summary=summary,
        evidence=evidence,
        default_preview_port=4173,
    )

    assert refresh_context.evidence.preview_port == 5173
    assert refresh_context.request.allowed is True
    assert refresh_context.request.preview_port == 5173
    assert refresh_context.request.baseline.preview_state == "retained"


@pytest.mark.asyncio
async def test_resolve_command_preview_publishes_preview_when_command_succeeds() -> None:
    controller = FakeSandboxController()

    result = await resolve_command_preview(
        sandbox_controller=controller,  # type: ignore[arg-type]
        command="pnpm dev --host 0.0.0.0 --port 5173",
        command_result=CommandResult(stdout="ready on http://localhost:5173", stderr="", exit_code=0),
    )

    assert result.preview_port == 5173
    assert result.preview_url == "https://5173-preview.example"
    assert result.preview_failure_reason is None
    assert controller.published_ports == [5173]
