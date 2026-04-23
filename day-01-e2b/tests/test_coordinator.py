from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agent_black_box.config import Settings
from agent_black_box import coordinator as coordinator_module
from agent_black_box.coordinator import (
    FixtureRunSource,
    RunCoordinator,
    SnapshotRunSource,
)
from agent_black_box.events import load_event
from agent_black_box.model_client import ModelDecision, ProviderInterruptionError
from agent_black_box.sandbox_controller import CommandResult, SandboxRunContext


class FakeModelClient:
    def __init__(self) -> None:
        self.calls = 0
        self.current_model = "kimi-k2.6:cloud"

    def reset_run_state(self) -> None:
        self.calls = 0
        self.current_model = "kimi-k2.6:cloud"

    def active_model_name(self) -> str:
        return self.current_model

    async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls == 1:
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="run_command",
                tool_arguments={"command": "echo hello", "timeout_seconds": 30},
                assistant_message={"role": "assistant", "tool_calls": []},
            )
        return ModelDecision(
            finish_reason="tool_call",
            tool_name="finish_run",
            tool_arguments={"summary": "done"},
            assistant_message={"role": "assistant", "tool_calls": []},
        )

    async def list_models(self):  # noqa: ANN201
        return ["kimi-k2.6:cloud"]


class RecordingOllamaModelClient(FakeModelClient):
    created: list["RecordingOllamaModelClient"] = []

    def __init__(
        self,
        settings: Settings,
        model_name: str | None = None,
        fallback_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.current_model = model_name or settings.ollama_model
        self.initial_model = self.current_model
        self.fallback_enabled = fallback_enabled
        type(self).created.append(self)

    def reset_run_state(self) -> None:
        self.calls = 0
        self.current_model = self.initial_model

    @classmethod
    def reset_created(cls) -> None:
        cls.created = []


class ProviderInterruptedModelClient(FakeModelClient):
    async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
        raise ProviderInterruptionError(
            primary_model="kimi-k2.6",
            fallback_model="glm-5.1",
            last_error=RuntimeError("503 Service Unavailable"),
        )


class PublishPreviewThenFailModelClient(FakeModelClient):
    async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls == 1:
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="run_command",
                tool_arguments={
                    "command": "pnpm dev --host 0.0.0.0 --port 4173",
                    "timeout_seconds": 30,
                },
                assistant_message={"role": "assistant", "tool_calls": []},
            )
        raise RuntimeError("model crashed after preview")


class PreviewThenFinishModelClient(FakeModelClient):
    async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls == 1:
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="run_command",
                tool_arguments={
                    "command": "pnpm dev --host 0.0.0.0 --port 4173",
                    "timeout_seconds": 30,
                },
                assistant_message={"role": "assistant", "tool_calls": []},
            )
        return ModelDecision(
            finish_reason="tool_call",
            tool_name="finish_run",
            tool_arguments={"summary": "preview ready"},
            assistant_message={"role": "assistant", "tool_calls": []},
        )


class BuildThenFinishModelClient(FakeModelClient):
    async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls == 1:
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="run_command",
                tool_arguments={"command": "pnpm build", "timeout_seconds": 30},
                assistant_message={"role": "assistant", "tool_calls": []},
            )
        return ModelDecision(
            finish_reason="tool_call",
            tool_name="finish_run",
            tool_arguments={"summary": "build passed"},
            assistant_message={"role": "assistant", "tool_calls": []},
        )


class PlainTextCompletionModelClient(FakeModelClient):
    async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls == 1:
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="run_command",
                tool_arguments={"command": "echo hello", "timeout_seconds": 30},
                assistant_message={"role": "assistant", "tool_calls": []},
            )
        return ModelDecision(
            finish_reason="completed",
            message="Implemented the requested change and verified the command output.",
            assistant_message={
                "role": "assistant",
                "content": "Implemented the requested change and verified the command output.",
            },
        )


class TimeoutAfterDiffModelClient(FakeModelClient):
    async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls == 1:
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="write_file",
                tool_arguments={"path": "index.html", "content": "<h1>Done</h1>"},
                assistant_message={"role": "assistant", "tool_calls": []},
            )
        raise httpx.ReadTimeout("model turn timed out")


class EmptyCompletionModelClient(FakeModelClient):
    async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
        return ModelDecision(
            finish_reason="completed",
            message="   ",
            assistant_message={"role": "assistant", "content": "   "},
        )


class EmptyThenWriteThenFinishModelClient(FakeModelClient):
    def __init__(self) -> None:
        super().__init__()
        self.repair_prompt: str | None = None

    async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls == 1:
            return ModelDecision(
                finish_reason="completed",
                message="",
                assistant_message={"role": "assistant", "content": ""},
                provider_eval_count=512,
                provider_num_predict=512,
                provider_content_chars=0,
                provider_tool_call_count=0,
                hit_generation_limit=True,
            )
        if self.calls == 2:
            self.repair_prompt = conversation[-1]["content"]
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="write_file",
                tool_arguments={"path": "index.html", "content": "<svg></svg>"},
                assistant_message={"role": "assistant", "tool_calls": []},
            )
        return ModelDecision(
            finish_reason="tool_call",
            tool_name="finish_run",
            tool_arguments={"summary": "SVG written"},
            assistant_message={"role": "assistant", "tool_calls": []},
        )


class InvalidFinishThenValidFinishModelClient(FakeModelClient):
    async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
        self.calls += 1
        if self.calls == 1:
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="write_file",
                tool_arguments={"path": "index.html", "content": "<h1>Done</h1>"},
                assistant_message={"role": "assistant", "tool_calls": []},
            )
        if self.calls == 2:
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="finish_run",
                tool_arguments={},
                assistant_message={"role": "assistant", "tool_calls": []},
            )
        assert conversation[-1]["role"] == "tool"
        assert "summary" in conversation[-1]["content"]
        return ModelDecision(
            finish_reason="tool_call",
            tool_name="finish_run",
            tool_arguments={"summary": "done after valid finish"},
            assistant_message={"role": "assistant", "tool_calls": []},
        )


class FakeSandboxController:
    def __init__(self) -> None:
        self.workspace_dir = "/tmp/workspace"
        self.closed = False
        self.commands: list[str] = []
        self.preview_readiness_timeouts: list[int | None] = []
        self.preview_ready = False

    async def start_run_from_fixture(self, fixture_path: Path) -> SandboxRunContext:
        return SandboxRunContext(sandbox_id="sbx-1", workspace_dir=self.workspace_dir)

    async def start_run_from_snapshot(self, snapshot_id: str) -> SandboxRunContext:
        return SandboxRunContext(
            sandbox_id="sbx-snapshot", workspace_dir=self.workspace_dir
        )

    async def read_file(self, path: str) -> str:
        return "contents"

    async def write_file(self, path: str, content: str) -> None:
        return None

    async def apply_patch(self, patch_text: str) -> None:
        return None

    async def run_command(
        self,
        command: str,
        timeout_seconds: int,
        *,
        on_stdout=None,
        on_stderr=None,
    ) -> CommandResult:
        self.commands.append(command)
        if on_stdout is not None:
            await on_stdout("hello\n")
        if "pnpm dev --host 0.0.0.0 --port 4173 --strictPort" in command:
            self.preview_ready = True
        return CommandResult(stdout="hello\n", stderr="", exit_code=0)

    async def collect_git_diff(self) -> str:
        return "diff --git a/src/main.js b/src/main.js\n+++ b/src/main.js\n@@\n+hello\n"

    async def publish_preview(
        self, port: int, *, readiness_timeout_seconds: int | None = None
    ) -> str:
        self.preview_readiness_timeouts.append(readiness_timeout_seconds)
        if not self.preview_ready:
            raise RuntimeError(f"Preview server on port {port} did not become ready")
        return f"https://{port}-preview.example"

    async def create_checkpoint(self, note: str) -> str:
        return "snap-1"

    async def close(self, *, retain: bool = False) -> None:
        self.closed = True

    def is_background_command(self, command: str) -> bool:
        return self._is_background_command(command)

    def _is_background_command(self, command: str) -> bool:
        return False

    def preview_port_for_command(
        self, command: str, *, stdout: str = "", stderr: str = ""
    ) -> int | None:
        if "--port 5173" in command or "localhost:5173" in command:
            return 5173
        if "preview" in command or "dev" in command:
            return 4173
        return None


class MissingFileSandboxController(FakeSandboxController):
    async def read_file(self, path: str) -> str:
        raise FileNotFoundError(f"path {path!r} does not exist")


class PublishPreviewThenCheckpointFailSandboxController(FakeSandboxController):
    async def create_checkpoint(self, note: str) -> str:
        raise RuntimeError("snapshot creation failed")


class PublishPreviewFailSandboxController(FakeSandboxController):
    async def publish_preview(
        self, port: int, *, readiness_timeout_seconds: int | None = None
    ) -> str:
        self.preview_readiness_timeouts.append(readiness_timeout_seconds)
        raise RuntimeError("Preview server on port 4173 did not become ready")


class PreviewPortAlreadyInUseSandboxController(FakeSandboxController):
    async def run_command(
        self,
        command: str,
        timeout_seconds: int,
        *,
        on_stdout=None,
        on_stderr=None,
    ) -> CommandResult:
        self.commands.append(command)
        if command == "pnpm dev --host 0.0.0.0 --port 4173 --strictPort":
            return CommandResult(
                stdout="",
                stderr="Error: Port 4173 is already in use\n",
                exit_code=1,
                background=True,
            )
        if on_stdout is not None:
            await on_stdout("hello\n")
        return CommandResult(stdout="hello\n", stderr="", exit_code=0)

    async def publish_preview(
        self, port: int, *, readiness_timeout_seconds: int | None = None
    ) -> str:
        self.preview_readiness_timeouts.append(readiness_timeout_seconds)
        if readiness_timeout_seconds == 2:
            raise RuntimeError(f"Preview server on port {port} did not become ready")
        return f"https://{port}-preview.example"


class PreviewCommandRaisesSandboxController(FakeSandboxController):
    async def run_command(
        self,
        command: str,
        timeout_seconds: int,
        *,
        on_stdout=None,
        on_stderr=None,
    ) -> CommandResult:
        if command == "pnpm preview --host --port 5173":
            self.commands.append(command)
            raise RuntimeError("agent preview failed")
        return await super().run_command(
            command,
            timeout_seconds,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )


class BuildFailSandboxController(FakeSandboxController):
    async def run_command(
        self,
        command: str,
        timeout_seconds: int,
        *,
        on_stdout=None,
        on_stderr=None,
    ) -> CommandResult:
        self.commands.append(command)
        if command == "pnpm build":
            if on_stderr is not None:
                await on_stderr("build exploded\n")
            return CommandResult(stdout="", stderr="build exploded\n", exit_code=1)
        return await super().run_command(
            command,
            timeout_seconds,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )


@pytest.mark.asyncio
async def test_coordinator_records_mocked_run(tmp_path: Path) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    sandbox = FakeSandboxController()
    coordinator = RunCoordinator(
        settings=settings,
        model_client=FakeModelClient(),
        sandbox_controller=sandbox,
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    event_payloads = [
        json.loads(raw)
        for raw in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    events = "\n".join(payload["type"] for payload in event_payloads)

    assert summary["status"] == "succeeded"
    assert summary["preview_url"] == "https://4173-preview.example"
    assert summary["preview_failure_reason"] is None
    assert summary["sandbox_retained"] is True
    assert any("fuser -k 4173/tcp" in command for command in sandbox.commands)
    assert "pnpm dev --host 0.0.0.0 --port 4173 --strictPort" in sandbox.commands
    assert sandbox.preview_readiness_timeouts == [2, 30]
    assert "run_started" in events
    assert "tool_call" in events
    assert "command_stream" in events
    assert "command_completed" in events
    assert "preview_published" in events
    first_turn = next(
        payload for payload in event_payloads if payload["type"] == "model_turn_started"
    )
    assert first_turn["model_name"] == "kimi-k2.6:cloud"
    assert first_turn["message_count"] == 2
    assert first_turn["tool_schema_count"] > 0
    assert first_turn["conversation_chars"] > 0
    assert first_turn["request_body_bytes"] > first_turn["conversation_chars"]


@pytest.mark.asyncio
async def test_coordinator_uses_source_model_without_fallback_for_arena_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    RecordingOllamaModelClient.reset_created()
    monkeypatch.setattr(
        coordinator_module,
        "OllamaModelClient",
        RecordingOllamaModelClient,
    )
    coordinator = RunCoordinator(
        settings=settings,
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(
            fixture_name="sample_frontend_task",
            model_name="gemma4:31b",
        )
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["model_name"] == "gemma4:31b"
    assert len(RecordingOllamaModelClient.created) == 2
    assert RecordingOllamaModelClient.created[-1].current_model == "gemma4:31b"
    assert RecordingOllamaModelClient.created[-1].fallback_enabled is False


@pytest.mark.asyncio
async def test_coordinator_records_preview_without_extra_artifact_capture(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=PreviewThenFinishModelClient(),
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    event_payloads = [
        json.loads(raw)
        for raw in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]

    assert any(payload["type"] == "preview_published" for payload in event_payloads)
    assert sorted(path.name for path in (run_dir / "artifacts").iterdir()) == [
        "preview-url.txt"
    ]


@pytest.mark.asyncio
async def test_coordinator_records_checkpoint_for_successful_build_command(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "plain_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Build the workspace", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=BuildThenFinishModelClient(),
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(FixtureRunSource(fixture_name="plain_task"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    event_payloads = [
        json.loads(raw)
        for raw in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    checkpoint_event = next(
        payload
        for payload in event_payloads
        if payload["type"] == "checkpoint_created"
    )

    assert summary["status"] == "succeeded"
    assert summary["checkpoint_id"] == "snap-1"
    assert checkpoint_event["snapshot_id"] == "snap-1"
    assert checkpoint_event["note"] == "successful milestone"
    assert not any(
        payload["type"] == "preview_published" for payload in event_payloads
    )


@pytest.mark.asyncio
async def test_coordinator_marks_provider_interrupted_runs_honestly(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=ProviderInterruptedModelClient(),
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["status"] == "provider_interrupted"
    assert "Provider interruption" in summary["failure_reason"]
    assert summary["preview_url"] is None
    assert summary["sandbox_retained"] is False
    assert summary["preview_state"] == "expired"
    assert (
        summary["preview_last_error"]
        == "Sandbox was not retained after provider interruption."
    )
    assert (
        summary["preview_failure_reason"]
        == "Preview was expected, but the provider interrupted the run before it could be verified."
    )


@pytest.mark.asyncio
async def test_coordinator_clears_preview_when_run_fails_after_publish(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=PublishPreviewThenFailModelClient(),
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

    assert summary["status"] == "failed"
    assert summary["preview_url"] is None
    assert summary["preview_state"] == "expired"
    assert summary["sandbox_retained"] is False
    assert status["preview_url"] is None
    assert status["preview_state"] == "expired"


@pytest.mark.asyncio
async def test_coordinator_preserves_preview_when_followup_checkpointing_fails(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    class PublishPreviewThenFinishModelClient(FakeModelClient):
        async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
            self.calls += 1
            if self.calls == 1:
                return ModelDecision(
                    finish_reason="tool_call",
                    tool_name="run_command",
                    tool_arguments={
                        "command": "pnpm dev --host 0.0.0.0 --port 4173",
                        "timeout_seconds": 30,
                    },
                    assistant_message={"role": "assistant", "tool_calls": []},
                )
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="finish_run",
                tool_arguments={"summary": "done"},
                assistant_message={"role": "assistant", "tool_calls": []},
            )

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=PublishPreviewThenFinishModelClient(),
        sandbox_controller=PublishPreviewThenCheckpointFailSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")

    assert summary["status"] == "succeeded"
    assert summary["preview_url"] == "https://4173-preview.example"
    assert summary["preview_state"] == "retained"
    assert summary["sandbox_retained"] is True
    assert status["preview_url"] == "https://4173-preview.example"
    assert status["preview_state"] == "retained"
    assert "preview_published" in events


@pytest.mark.asyncio
async def test_coordinator_records_managed_preview_failure_without_failing_run(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=FakeModelClient(),
        sandbox_controller=PublishPreviewFailSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")

    assert summary["status"] == "succeeded"
    assert summary["preview_url"] is None
    assert summary["preview_state"] == "unavailable"
    assert (
        summary["preview_failure_reason"]
        == "Managed preview on port 4173 did not become publishable: Preview server on port 4173 did not become ready"
    )
    assert status["preview_attempted"] is True
    assert "pnpm dev --host 0.0.0.0 --port 4173 --strictPort" in events


@pytest.mark.asyncio
async def test_coordinator_publishes_existing_preview_when_managed_port_is_in_use(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    sandbox = PreviewPortAlreadyInUseSandboxController()
    coordinator = RunCoordinator(
        settings=settings,
        model_client=FakeModelClient(),
        sandbox_controller=sandbox,
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")

    assert summary["status"] == "succeeded"
    assert summary["preview_url"] == "https://4173-preview.example"
    assert summary["preview_failure_reason"] is None
    assert status["preview_url"] == "https://4173-preview.example"
    assert sandbox.preview_readiness_timeouts == [2, 30]
    assert "Error: Port 4173 is already in use" in events
    assert "preview_published" in events


@pytest.mark.asyncio
async def test_coordinator_uses_managed_fixture_preview_over_agent_port(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    class PreviewOn5173ModelClient(FakeModelClient):
        async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
            self.calls += 1
            if self.calls == 1:
                return ModelDecision(
                    finish_reason="tool_call",
                    tool_name="run_command",
                    tool_arguments={
                        "command": "pnpm preview --host --port 5173",
                        "timeout_seconds": 30,
                    },
                    assistant_message={"role": "assistant", "tool_calls": []},
                )
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="finish_run",
                tool_arguments={"summary": "done"},
                assistant_message={"role": "assistant", "tool_calls": []},
            )

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=PreviewOn5173ModelClient(),
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    event_payloads = [
        json.loads(raw)
        for raw in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    preview_events = [
        payload for payload in event_payloads if payload["type"] == "preview_published"
    ]

    assert summary["preview_url"] == "https://4173-preview.example"
    assert [payload["port"] for payload in preview_events] == [4173]
    assert any(
        payload["type"] == "command_completed"
        and payload["command"] == "pnpm preview --host --port 5173"
        for payload in event_payloads
    )


@pytest.mark.asyncio
async def test_coordinator_suppresses_agent_preview_exception_for_managed_fixture(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    class PreviewOn5173ThenFinishModelClient(FakeModelClient):
        async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
            self.calls += 1
            if self.calls == 1:
                return ModelDecision(
                    finish_reason="tool_call",
                    tool_name="run_command",
                    tool_arguments={
                        "command": "pnpm preview --host --port 5173",
                        "timeout_seconds": 30,
                    },
                    assistant_message={"role": "assistant", "tool_calls": []},
                )
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="finish_run",
                tool_arguments={"summary": "done"},
                assistant_message={"role": "assistant", "tool_calls": []},
            )

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=PreviewOn5173ThenFinishModelClient(),
        sandbox_controller=PreviewCommandRaisesSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    event_payloads = [
        json.loads(raw)
        for raw in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    failed_tool_result = next(
        payload
        for payload in event_payloads
        if payload["type"] == "tool_result"
        and payload["tool_name"] == "run_command"
        and payload["result"].get("error") == "agent preview failed"
    )
    preview_events = [
        payload for payload in event_payloads if payload["type"] == "preview_published"
    ]

    assert summary["status"] == "succeeded"
    assert summary["preview_url"] == "https://4173-preview.example"
    assert summary["preview_failure_reason"] is None
    assert "preview_failure_reason" not in failed_tool_result["result"]
    assert [payload["port"] for payload in preview_events] == [4173]


@pytest.mark.asyncio
async def test_coordinator_returns_recoverable_tool_error_for_missing_files(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    class ReadMissingFileThenFinishModelClient(FakeModelClient):
        async def next_action(self, conversation, tools):  # noqa: ANN001, ANN201
            self.calls += 1
            if self.calls == 1:
                return ModelDecision(
                    finish_reason="tool_call",
                    tool_name="read_file",
                    tool_arguments={"path": "src/main.tsx"},
                    assistant_message={"role": "assistant", "tool_calls": []},
                )
            return ModelDecision(
                finish_reason="tool_call",
                tool_name="finish_run",
                tool_arguments={"summary": "done after recoverable file miss"},
                assistant_message={"role": "assistant", "tool_calls": []},
            )

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=ReadMissingFileThenFinishModelClient(),
        sandbox_controller=MissingFileSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")

    assert summary["status"] == "succeeded"
    assert '"tool_name":"read_file","ok":false' in events
    assert "src/main.tsx" in events


@pytest.mark.asyncio
async def test_coordinator_synthesizes_finish_run_from_plain_text_completion(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=PlainTextCompletionModelClient(),
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")

    assert summary["status"] == "succeeded"
    assert '"tool_name":"finish_run"' not in events
    assert "Implemented the requested change and verified the command output." in events
    assert "plain text instead of finish_run" in summary["failure_reason"]


@pytest.mark.asyncio
async def test_coordinator_finalizes_after_timeout_when_diff_exists(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=TimeoutAfterDiffModelClient(),
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")

    assert summary["status"] == "succeeded"
    assert "ReadTimeout" in summary["failure_reason"]
    assert "pnpm install" in events
    assert "pnpm build" in events
    assert "preview_published" in events


@pytest.mark.asyncio
async def test_coordinator_recovers_from_invalid_finish_run_arguments(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=InvalidFinishThenValidFinishModelClient(),
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    event_payloads = [
        json.loads(raw)
        for raw in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    invalid_finish_result = next(
        payload
        for payload in event_payloads
        if payload["type"] == "tool_result"
        and payload["tool_name"] == "finish_run"
        and payload["ok"] is False
    )

    assert summary["status"] == "succeeded"
    assert invalid_finish_result["result"]["error"] == "'summary'"
    assert sum(1 for payload in event_payloads if payload["type"] == "run_completed") == 1


@pytest.mark.asyncio
async def test_coordinator_reports_harness_build_failure_after_plain_text_completion(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=PlainTextCompletionModelClient(),
        sandbox_controller=BuildFailSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))

    assert summary["status"] == "failed"
    assert status["state"] == "failed"
    assert "Managed build command failed" in summary["failure_reason"]
    assert (
        summary["preview_failure_reason"]
        == "Preview was not attempted because the managed build failed."
    )
    assert summary["preview_url"] is None


@pytest.mark.asyncio
async def test_coordinator_repairs_empty_completion_at_generation_limit(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    model = EmptyThenWriteThenFinishModelClient()
    coordinator = RunCoordinator(
        settings=settings,
        model_client=model,
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    event_lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [load_event(line) for line in event_lines]
    raw_events = "\n".join(event_lines)

    assert summary["status"] == "succeeded"
    assert model.repair_prompt is not None
    assert "output token limit" in model.repair_prompt
    assert "Call exactly one tool now" in model.repair_prompt
    repair_event = next(
        event for event in events if event.type == "protocol_repair_requested"
    )
    assert repair_event.failure_kind == "hit_generation_limit"
    assert repair_event.hit_generation_limit is True
    assert '"tool_name":"write_file"' in raw_events


@pytest.mark.asyncio
async def test_coordinator_fails_after_repeated_empty_plain_text_completions(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    coordinator = RunCoordinator(
        settings=settings,
        model_client=EmptyCompletionModelClient(),
        sandbox_controller=FakeSandboxController(),
    )

    run_dir = await coordinator.run_once(
        FixtureRunSource(fixture_name="sample_frontend_task")
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    event_lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [load_event(line) for line in event_lines]

    assert summary["status"] == "failed"
    assert "Model protocol incomplete" in summary["failure_reason"]
    assert sum(1 for event in events if event.type == "protocol_repair_requested") == 2
    run_failed = next(event for event in events if event.type == "run_failed")
    assert run_failed.failure_kind == "completed_without_action"


@pytest.mark.asyncio
async def test_coordinator_uses_snapshot_source_without_fixture_upload(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixtures"
    fixture_dir = fixture_root / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")

    class SnapshotAwareSandboxController(FakeSandboxController):
        def __init__(self) -> None:
            super().__init__()
            self.started_from_snapshot: str | None = None
            self.fixture_started = False

        async def start_run_from_fixture(self, fixture_path: Path) -> SandboxRunContext:
            self.fixture_started = True
            return await super().start_run_from_fixture(fixture_path)

        async def start_run_from_snapshot(self, snapshot_id: str) -> SandboxRunContext:
            self.started_from_snapshot = snapshot_id
            return await super().start_run_from_snapshot(snapshot_id)

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        FIXTURE_ROOT=fixture_root,
    )
    sandbox = SnapshotAwareSandboxController()
    coordinator = RunCoordinator(
        settings=settings,
        model_client=FakeModelClient(),
        sandbox_controller=sandbox,
    )

    run_dir = await coordinator.run_once(
        SnapshotRunSource(
            fixture_name="sample_frontend_task",
            parent_run_id="parent-run",
            source_snapshot_id="snap-82",
            source_checkpoint_sequence=82,
            instruction_override="Restyle the hero",
            parent_task="Do the thing",
        )
    )
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    assert sandbox.started_from_snapshot == "snap-82"
    assert sandbox.fixture_started is False
    assert metadata["parent_run_id"] == "parent-run"
    assert metadata["source_snapshot_id"] == "snap-82"
