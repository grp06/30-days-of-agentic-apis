from __future__ import annotations

from pathlib import Path

import pytest
from agent_black_box.config import Settings
from agent_black_box.sandbox_controller import (
    SandboxController,
    infer_preview_port,
    infer_preview_port_from_url,
)
from e2b.sandbox.commands.command_handle import CommandExitException


def test_should_skip_upload_filters_generated_directories(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    path = root / "node_modules" / "vite" / "index.js"
    path.parent.mkdir(parents=True)
    path.write_text("x", encoding="utf-8")

    assert SandboxController._should_skip_upload(path, root) is True
    assert SandboxController._should_skip_upload(root / "src" / "main.js", root) is False


def test_should_skip_upload_filters_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    root.mkdir()
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    symlink = root / "linked-secret.txt"
    symlink.symlink_to(outside)

    assert SandboxController._should_skip_upload(symlink, root) is True


def test_workspace_path_accepts_relative_and_absolute_workspace_paths() -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")
    controller = SandboxController(settings)

    assert controller._workspace_path("package.json") == "/home/user/workspace/package.json"
    assert controller._workspace_path("./src/../package.json") == "/home/user/workspace/package.json"
    assert (
        controller._workspace_path("/home/user/workspace/package.json")
        == "/home/user/workspace/package.json"
    )


def test_workspace_path_rejects_absolute_paths_outside_workspace() -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")
    controller = SandboxController(settings)

    try:
        controller._workspace_path("/etc/passwd")
    except ValueError as exc:
        assert "outside the workspace root" in str(exc)
    else:
        raise AssertionError("expected absolute paths outside the workspace to be rejected")


def test_workspace_path_rejects_relative_traversal_outside_workspace() -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")
    controller = SandboxController(settings)

    with pytest.raises(ValueError, match="outside the workspace root"):
        controller._workspace_path("../package.json")

    with pytest.raises(ValueError, match="outside the workspace root"):
        controller._workspace_path("src/../../package.json")


@pytest.mark.asyncio
async def test_run_command_returns_nonzero_results_without_crashing() -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")
    controller = SandboxController(settings)

    class FakeCommands:
        async def run(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            raise CommandExitException(
                stderr="bad",
                stdout="partial",
                exit_code=2,
                error="boom",
            )

    class FakeSandbox:
        commands = FakeCommands()

    controller.sandbox = FakeSandbox()  # type: ignore[assignment]
    result = await controller.run_command("false", timeout_seconds=5)

    assert result.exit_code == 2
    assert result.stdout == "partial"
    assert result.stderr == "bad"


@pytest.mark.asyncio
async def test_run_command_reports_immediate_background_failures(tmp_path: Path) -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")
    controller = SandboxController(settings)

    class FakeHandle:
        pid = 4321
        stdout = "booting...\n"
        stderr = "address already in use\n"

        async def wait(self):  # noqa: ANN202
            raise CommandExitException(
                stderr=self.stderr,
                stdout=self.stdout,
                exit_code=1,
                error="boom",
            )

        async def disconnect(self) -> None:
            raise AssertionError("disconnect should not be called when the process exits immediately")

    class FakeCommands:
        async def run(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            assert kwargs["background"] is True
            return FakeHandle()

    class FakeSandbox:
        commands = FakeCommands()

    controller.sandbox = FakeSandbox()  # type: ignore[assignment]
    result = await controller.run_command("pnpm dev --host 0.0.0.0 --port 4173", timeout_seconds=5)

    assert result.background is True
    assert result.exit_code == 1
    assert "booting" in result.stdout
    assert "address already in use" in result.stderr


def test_infer_preview_port_prefers_explicit_command_port() -> None:
    assert (
        infer_preview_port(
            "cd /home/user/workspace && pnpm preview --host --port 5173",
            default_port=4173,
        )
        == 5173
    )


def test_infer_preview_port_uses_actual_dev_server_port_from_output() -> None:
    assert (
        infer_preview_port(
            "cd /home/user/workspace && pnpm dev --host 0.0.0.0 --port 4173",
            default_port=4173,
            stdout=(
                "Port 4173 is in use, trying another one...\n"
                "  ➜  Local: http://localhost:4176/\n"
            ),
        )
        == 4176
    )


def test_infer_preview_port_uses_python_http_server_default_port() -> None:
    assert infer_preview_port("python -m http.server", default_port=4173) == 8000
    assert infer_preview_port("python3 -m http.server", default_port=4173) == 8000


def test_background_command_detection_allows_workspace_cd_prefix() -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")
    controller = SandboxController(settings)

    assert (
        controller.is_background_command(
            "cd /home/user/workspace && pnpm dev --host 0.0.0.0 --port 4173"
        )
        is True
    )
    assert (
        controller.is_background_command(
            "cd /home/user/workspace && python3 -m http.server"
        )
        is True
    )
    assert controller.is_background_command("cd /tmp && pnpm dev --port 4173") is False
    assert (
        controller.is_background_command("pnpm install && pnpm dev --port 4173")
        is False
    )


def test_background_command_detection_rejects_shell_control_operators() -> None:
    settings = Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test")
    controller = SandboxController(settings)

    assert controller.is_background_command("pnpm dev --port 4173& echo done") is False
    assert controller.is_background_command("pnpm dev --port 4173 | cat") is False
    assert controller.is_background_command("pnpm dev --port 4173; echo done") is False
    assert (
        controller.is_background_command(
            "cd /home/user/workspace && pnpm dev --port 4173 && echo done"
        )
        is False
    )


def test_infer_preview_port_uses_localhost_probe_when_present() -> None:
    assert (
        infer_preview_port(
            "curl -s http://localhost:5173 | head -n 20",
            default_port=4173,
        )
        == 5173
    )


def test_infer_preview_port_from_url_reads_e2b_host_prefix() -> None:
    assert infer_preview_port_from_url("https://5173-sbx-1.e2b.app") == 5173
