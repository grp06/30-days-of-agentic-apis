from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_black_box import fixture_policy as fixture_policy_module
from agent_black_box.config import Settings
from agent_black_box.fixture_policy import FixturePolicy, PreviewPolicy
from agent_black_box.preview_service import PreviewService
from agent_black_box.recorder import RunMetadata, RunStatus, RunSummary, prepare_run_directory
from agent_black_box.run_store import RunStore


class FakeSandboxController:
    def __init__(self, result: tuple[str, str | None, str | None] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, int]] = []

    async def refresh_preview(self, sandbox_id: str, port: int) -> tuple[str, str | None, str | None]:
        self.calls.append((sandbox_id, port))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _write_run(
    tmp_path: Path,
    *,
    preview_url: str | None = None,
    sandbox_id: str = "sbx-1",
    fixture_name: str = "sample_frontend_task",
) -> None:
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name=fixture_name,
            sandbox_id=sandbox_id,
        ),
        RunStatus(run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    if preview_url is not None:
        (run_dir / "artifacts" / "preview-url.txt").write_text(preview_url + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name=fixture_name,
            sandbox_id=sandbox_id,
            preview_url=preview_url,
            preview_state="retained" if preview_url else "unavailable",
            sandbox_retained=True,
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_preview_service_marks_live_preview_when_refresh_succeeds(tmp_path: Path) -> None:
    _write_run(tmp_path)
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = PreviewService(
        settings,
        run_store=RunStore(tmp_path),
        sandbox_controller=FakeSandboxController(("live", "https://4173-sbx-1.e2b.app", None)),  # type: ignore[arg-type]
    )

    await service.refresh_preview("run-1")

    summary = json.loads((tmp_path / "run-1" / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "run-1" / "status.json").read_text(encoding="utf-8"))
    assert summary["preview_state"] == "live"
    assert summary["preview_url"] == "https://4173-sbx-1.e2b.app"
    assert status["preview_state"] == "live"
    assert status["sandbox_retained"] is True


@pytest.mark.asyncio
async def test_preview_service_marks_expired_preview_when_refresh_fails(tmp_path: Path) -> None:
    _write_run(tmp_path, preview_url="https://4173-old.e2b.app")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = PreviewService(
        settings,
        run_store=RunStore(tmp_path),
        sandbox_controller=FakeSandboxController(RuntimeError("sandbox wasn't found")),  # type: ignore[arg-type]
    )

    await service.refresh_preview("run-1")

    summary = json.loads((tmp_path / "run-1" / "summary.json").read_text(encoding="utf-8"))
    assert summary["preview_state"] == "expired"
    assert summary["preview_url"] == "https://4173-old.e2b.app"
    assert summary["sandbox_retained"] is False
    assert "wasn't found" in summary["preview_last_error"]
    assert (tmp_path / "run-1" / "artifacts" / "preview-url.txt").exists()


@pytest.mark.asyncio
async def test_preview_service_preserves_preview_failure_reason_when_sandbox_is_missing(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path)
    summary_path = tmp_path / "run-1" / "summary.json"
    status_path = tmp_path / "run-1" / "status.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    summary["preview_failure_reason"] = "Preview command for port 4173 failed before publication"
    status["preview_failure_reason"] = "Preview command for port 4173 failed before publication"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = PreviewService(
        settings,
        run_store=RunStore(tmp_path),
        sandbox_controller=FakeSandboxController(RuntimeError("sandbox wasn't found")),  # type: ignore[arg-type]
    )

    await service.refresh_preview("run-1")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert summary["preview_state"] == "expired"
    assert summary["preview_failure_reason"] == "Preview command for port 4173 failed before publication"
    assert status["preview_state"] == "expired"
    assert status["preview_failure_reason"] == "Preview command for port 4173 failed before publication"


@pytest.mark.asyncio
async def test_preview_service_preserves_retained_preview_on_transient_refresh_error(
    tmp_path: Path,
) -> None:
    _write_run(tmp_path, preview_url="https://4173-old.e2b.app")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = PreviewService(
        settings,
        run_store=RunStore(tmp_path),
        sandbox_controller=FakeSandboxController(RuntimeError("temporary network timeout")),  # type: ignore[arg-type]
    )

    await service.refresh_preview("run-1")

    summary = json.loads((tmp_path / "run-1" / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "run-1" / "status.json").read_text(encoding="utf-8"))
    assert summary["preview_state"] == "retained"
    assert summary["preview_url"] == "https://4173-old.e2b.app"
    assert summary["sandbox_retained"] is True
    assert "temporary network timeout" in summary["preview_last_error"]
    assert status["preview_state"] == "retained"
    assert status["preview_url"] == "https://4173-old.e2b.app"
    assert status["sandbox_retained"] is True
    assert (tmp_path / "run-1" / "artifacts" / "preview-url.txt").exists()


@pytest.mark.asyncio
async def test_preview_service_rejects_refresh_for_running_run(tmp_path: Path) -> None:
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
        ),
        RunStatus(run_id="run-1", state="running", current_model_name="kimi-k2.6:cloud"),
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = PreviewService(
        settings,
        run_store=RunStore(tmp_path),
        sandbox_controller=FakeSandboxController(("live", "https://4173-sbx-1.e2b.app", None)),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="only available after a run has finished"):
        await service.refresh_preview("run-1")


@pytest.mark.asyncio
async def test_preview_service_rejects_refresh_for_non_retained_run(tmp_path: Path) -> None:
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1",
            state="failed",
            current_model_name="kimi-k2.6:cloud",
            preview_state="expired",
            sandbox_retained=False,
        ),
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="failed",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
            preview_state="expired",
            preview_last_error="Sandbox was not retained after run failure.",
            sandbox_retained=False,
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = PreviewService(
        settings,
        run_store=RunStore(tmp_path),
        sandbox_controller=FakeSandboxController(("live", "https://4173-sbx-1.e2b.app", None)),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="only available for runs with retained sandboxes"):
        await service.refresh_preview("run-1")


@pytest.mark.asyncio
async def test_preview_service_infers_non_default_preview_port_from_run_history(tmp_path: Path) -> None:
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1",
            state="succeeded",
            current_model_name="kimi-k2.6:cloud",
            preview_state="unavailable",
            sandbox_retained=False,
        ),
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "tool_call",
                "run_id": "run-1",
                "lane_id": "lane-1",
                "sequence": 1,
                "timestamp": "2026-04-22T00:00:00Z",
                "tool_name": "run_command",
                "arguments": {"command": "pnpm preview --host --port 5173", "timeout_seconds": 10},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="run-1",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
            preview_state="unavailable",
            sandbox_retained=False,
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )

    sandbox = FakeSandboxController(("live", "https://5173-sbx-1.e2b.app", None))  # type: ignore[arg-type]
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = PreviewService(
        settings,
        run_store=RunStore(tmp_path),
        sandbox_controller=sandbox,
    )

    await service.refresh_preview("run-1")

    assert sandbox.calls == [("sbx-1", 5173)]
    summary = json.loads((tmp_path / "run-1" / "summary.json").read_text(encoding="utf-8"))
    assert summary["preview_url"] == "https://5173-sbx-1.e2b.app"
    assert summary["preview_state"] == "live"


@pytest.mark.asyncio
async def test_preview_service_uses_fixture_preview_port_when_history_has_no_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        fixture_policy_module._FIXTURE_POLICIES,
        "custom_preview_task",
        FixturePolicy(
            guardrails="Custom fixture guardrails.",
            preview=PreviewPolicy(
                command="pnpm preview --host",
                port=5179,
                cleanup_command="true",
            ),
        ),
    )
    _write_run(tmp_path, fixture_name="custom_preview_task")
    sandbox = FakeSandboxController(("live", "https://5179-sbx-1.e2b.app", None))  # type: ignore[arg-type]
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    service = PreviewService(
        settings,
        run_store=RunStore(tmp_path),
        sandbox_controller=sandbox,
    )

    await service.refresh_preview("run-1")

    assert sandbox.calls == [("sbx-1", 5179)]
