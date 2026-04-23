from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Awaitable, Callable
from urllib.parse import urlparse

import anyio
from e2b import AsyncSandbox
from e2b.sandbox.commands.command_handle import CommandExitException
from e2b.sandbox.commands.command_handle import CommandResult as E2BCommandResult

from .config import Settings


@dataclass
class SandboxRunContext:
    sandbox_id: str
    workspace_dir: str


@dataclass
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    pid: int | None = None
    background: bool = False


class SandboxController:
    SKIP_UPLOAD_NAMES = {".git", ".DS_Store", "__pycache__", "node_modules", "dist"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.sandbox: AsyncSandbox | None = None
        self.workspace_dir = "/home/user/workspace"

    async def start_run_from_fixture(self, fixture_path: Path) -> SandboxRunContext:
        self.sandbox = await AsyncSandbox.create(
            template=self.settings.e2b_template,
            timeout=self.settings.sandbox_timeout_seconds,
            metadata={"project": "agent-black-box", "fixture": fixture_path.name},
            lifecycle={"on_timeout": "pause", "auto_resume": True},
            api_key=self.settings.e2b_api_key,
        )
        await self.sandbox.files.make_dir(self.workspace_dir)
        await self._upload_tree(fixture_path, self.workspace_dir)
        await self._require_success("git init", timeout_seconds=60)
        await self._require_success(
            'git config user.email "agent-black-box@example.com"',
            timeout_seconds=60,
        )
        await self._require_success('git config user.name "Agent Black Box"', timeout_seconds=60)
        await self._require_success("git add .", timeout_seconds=60)
        await self._require_success('git commit -m "Initial fixture"', timeout_seconds=60)
        return SandboxRunContext(sandbox_id=self.sandbox.sandbox_id, workspace_dir=self.workspace_dir)

    async def start_run_from_snapshot(self, snapshot_id: str) -> SandboxRunContext:
        self.sandbox = await AsyncSandbox.create(
            snapshot_id,
            timeout=self.settings.sandbox_timeout_seconds,
            metadata={"project": "agent-black-box", "source": "snapshot"},
            lifecycle={"on_timeout": "pause", "auto_resume": True},
            api_key=self.settings.e2b_api_key,
        )
        return SandboxRunContext(sandbox_id=self.sandbox.sandbox_id, workspace_dir=self.workspace_dir)

    async def _upload_tree(self, local_root: Path, remote_root: str) -> None:
        assert self.sandbox is not None
        for path in sorted(local_root.rglob("*")):
            if self._should_skip_upload(path, local_root):
                continue
            if path.is_dir():
                rel = path.relative_to(local_root).as_posix()
                await self.sandbox.files.make_dir(f"{remote_root}/{rel}")
                continue
            rel = path.relative_to(local_root).as_posix()
            await self.sandbox.files.write(f"{remote_root}/{rel}", path.read_bytes())

    async def read_file(self, path: str) -> str:
        assert self.sandbox is not None
        return await self.sandbox.files.read(self._workspace_path(path), format="text")

    async def write_file(self, path: str, content: str) -> None:
        assert self.sandbox is not None
        target = self._workspace_path(path)
        parent = str(Path(target).parent)
        await self.sandbox.files.make_dir(parent)
        await self.sandbox.files.write(target, content)

    async def apply_patch(self, patch_text: str) -> None:
        assert self.sandbox is not None
        patch_path = f"{self.workspace_dir}/.agent-black-box.patch"
        await self.sandbox.files.write(patch_path, patch_text)
        try:
            result = await self.run_command(
                f"git apply --recount --whitespace=nowarn {shlex.quote(patch_path)}",
                timeout_seconds=120,
            )
            if result.exit_code != 0:
                raise RuntimeError(f"git apply failed: {result.stderr or result.stdout}")
        finally:
            await self.sandbox.files.remove(patch_path)

    async def run_command(
        self,
        command: str,
        timeout_seconds: int,
        *,
        on_stdout: Callable[[str], Awaitable[None] | None] | None = None,
        on_stderr: Callable[[str], Awaitable[None] | None] | None = None,
    ) -> CommandResult:
        assert self.sandbox is not None
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        background = self._is_background_command(command)

        async def stdout_handler(chunk: str) -> None:
            stdout_chunks.append(chunk)
            if on_stdout is not None:
                maybe = on_stdout(chunk)
                if maybe is not None:
                    await maybe

        async def stderr_handler(chunk: str) -> None:
            stderr_chunks.append(chunk)
            if on_stderr is not None:
                maybe = on_stderr(chunk)
                if maybe is not None:
                    await maybe

        try:
            handle_or_result = await self.sandbox.commands.run(
                command,
                cwd=self.workspace_dir,
                timeout=timeout_seconds,
                background=background,
                on_stdout=stdout_handler,
                on_stderr=stderr_handler,
            )
        except CommandExitException as exc:
            return CommandResult(
                stdout=exc.stdout,
                stderr=exc.stderr,
                exit_code=exc.exit_code,
                background=False,
            )
        if background:
            handle = handle_or_result
            with anyio.move_on_after(2):
                try:
                    result = await handle.wait()
                    return CommandResult(
                        stdout=result.stdout,
                        stderr=result.stderr,
                        exit_code=result.exit_code,
                        pid=handle.pid,
                        background=True,
                    )
                except CommandExitException as exc:
                    return CommandResult(
                        stdout=exc.stdout,
                        stderr=exc.stderr,
                        exit_code=exc.exit_code,
                        pid=handle.pid,
                        background=True,
                    )
            await handle.disconnect()
            return CommandResult(
                stdout=handle.stdout or "".join(stdout_chunks),
                stderr=handle.stderr or "".join(stderr_chunks),
                exit_code=0,
                pid=handle.pid,
                background=True,
            )
        result: E2BCommandResult = handle_or_result
        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            background=False,
        )

    async def collect_git_diff(self) -> str:
        result = await self.run_command(
            "git diff --binary --no-ext-diff --src-prefix=a/ --dst-prefix=b/",
            timeout_seconds=60,
        )
        if result.exit_code != 0:
            raise RuntimeError(f"git diff failed: {result.stderr or result.stdout}")
        return result.stdout

    async def publish_preview(
        self, port: int, *, readiness_timeout_seconds: int | None = None
    ) -> str:
        assert self.sandbox is not None
        await self._require_preview_ready(
            self.sandbox,
            port,
            timeout_seconds=readiness_timeout_seconds or 20,
        )
        host = self.sandbox.get_host(port)
        if host.startswith("http://") or host.startswith("https://"):
            return host
        return f"https://{host}"

    async def refresh_preview(self, sandbox_id: str, port: int) -> tuple[str, str | None, str | None]:
        sandbox = await AsyncSandbox.connect(
            sandbox_id,
            timeout=self.settings.sandbox_timeout_seconds,
            api_key=self.settings.e2b_api_key,
        )
        try:
            try:
                await self._require_preview_ready(sandbox, port, timeout_seconds=20)
            except RuntimeError as exc:
                return ("server_not_running", None, str(exc))
            host = sandbox.get_host(port)
            url = host if host.startswith(("http://", "https://")) else f"https://{host}"
            return ("live", url, None)
        finally:
            # Do not kill resumed sandboxes; leave lifecycle policy intact.
            self.sandbox = None

    async def create_checkpoint(self, note: str) -> str:
        assert self.sandbox is not None
        snapshot = await self.sandbox.create_snapshot()
        return snapshot.snapshot_id

    async def close(self, *, retain: bool = False) -> None:
        if self.sandbox is not None:
            if not retain:
                await self.sandbox.kill()
            self.sandbox = None

    def preview_port_for_command(
        self,
        command: str,
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> int | None:
        return infer_preview_port(
            command,
            default_port=self.settings.preview_port,
            stdout=stdout,
            stderr=stderr,
        )

    def _workspace_path(self, path: str) -> str:
        workspace = PurePosixPath(self.workspace_dir)
        candidate = PurePosixPath(path)
        if candidate.is_absolute():
            resolved = self._normalize_posix_path(candidate)
        else:
            resolved = self._normalize_posix_path(workspace / candidate)
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                f"path {path!r} is outside the workspace root {self.workspace_dir!r}"
            ) from exc
        return str(resolved)

    @staticmethod
    def _normalize_posix_path(path: PurePosixPath) -> PurePosixPath:
        parts: list[str] = []
        for part in path.parts:
            if part in {"", ".", "/"}:
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        if path.is_absolute():
            return PurePosixPath("/", *parts)
        return PurePosixPath(*parts)

    def is_background_command(self, command: str) -> bool:
        return self._is_background_command(command)

    def _is_background_command(self, command: str) -> bool:
        candidate = self._background_command_candidate(command)
        if candidate is None:
            return False
        lowered = candidate.lower()
        return (
            lowered.strip().startswith("pnpm dev")
            or lowered.strip().startswith("npm run dev")
            or lowered.strip().startswith("vite preview")
            or lowered.strip().startswith("pnpm preview")
            or lowered.strip().startswith("npm run preview")
            or lowered.strip().startswith("python -m http.server")
            or lowered.strip().startswith("python3 -m http.server")
        )

    def _background_command_candidate(self, command: str) -> str | None:
        stripped = command.strip()
        if not stripped or "\n" in stripped or ";" in stripped or "|" in stripped:
            return None
        if "&&" not in stripped:
            if "&" in stripped:
                return None
            return stripped
        if stripped.count("&&") != 1:
            return None
        prefix, suffix = stripped.split("&&", 1)
        if "&" in prefix or "&" in suffix or ";" in suffix or "|" in suffix:
            return None
        try:
            prefix_parts = shlex.split(prefix)
        except ValueError:
            return None
        if prefix_parts not in (["cd", self.workspace_dir], ["cd", "."]):
            return None
        return suffix.strip()

    def may_publish_preview(self, command: str) -> bool:
        return self.preview_port_for_command(command) is not None

    async def _require_success(self, command: str, timeout_seconds: int) -> None:
        result = await self.run_command(command, timeout_seconds=timeout_seconds)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Sandbox bootstrap command failed: {command}\n{result.stderr or result.stdout}"
            )

    async def _require_preview_ready(
        self, sandbox: AsyncSandbox, port: int, *, timeout_seconds: int
    ) -> None:
        probe_seconds = max(1, int(timeout_seconds))
        command = (
            f"for i in $(seq 1 {probe_seconds}); do "
            f"curl -fsS http://127.0.0.1:{port}/ >/dev/null && exit 0; "
            "sleep 1; "
            "done; "
            "exit 1"
        )
        try:
            result = await sandbox.commands.run(
                command,
                cwd=self.workspace_dir,
                timeout=probe_seconds + 5,
                background=False,
            )
            exit_code = result.exit_code
        except CommandExitException as exc:
            exit_code = exc.exit_code
        if exit_code != 0:
            raise RuntimeError(f"Preview server on port {port} did not become ready")

    @classmethod
    def _should_skip_upload(cls, path: Path, local_root: Path) -> bool:
        if path.is_symlink():
            return True
        rel_parts = path.relative_to(local_root).parts
        return any(part in cls.SKIP_UPLOAD_NAMES for part in rel_parts)


def infer_preview_port(
    command: str,
    *,
    default_port: int,
    stdout: str = "",
    stderr: str = "",
) -> int | None:
    lower_command = command.lower()
    if any(
        token in lower_command
        for token in (
            "pnpm dev",
            "npm run dev",
            "vite preview",
            "pnpm preview",
            "npm run preview",
        )
    ):
        port = _localhost_port(stdout)
        if port is not None:
            return port
        port = _localhost_port(stderr)
        if port is not None:
            return port

    port = _explicit_port(command)
    if port is not None:
        return port

    port = _localhost_port(lower_command)
    if port is not None:
        return port

    if "python3 -m http.server" in lower_command or "python -m http.server" in lower_command:
        return 8000

    if any(
        token in lower_command
        for token in (
            "pnpm dev",
            "npm run dev",
            "vite preview",
            "pnpm preview",
            "npm run preview",
        )
    ):
        return default_port

    port = _localhost_port(stdout)
    if port is not None:
        return port
    return _localhost_port(stderr)


def infer_preview_port_from_url(url: str) -> int | None:
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path
    match = re.match(r"(?P<port>\d+)-", host)
    if match is None:
        return None
    return int(match.group("port"))


def _explicit_port(command: str) -> int | None:
    patterns = (
        r"(?:^|\s)--port(?:=|\s+)(?P<port>\d+)(?:\s|$)",
        r"(?:^|\s)-p\s+(?P<port>\d+)(?:\s|$)",
        r"http\.server\s+(?P<port>\d+)(?:\s|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, command)
        if match is not None:
            return int(match.group("port"))
    return None


def _localhost_port(text: str) -> int | None:
    match = re.search(r"localhost:(?P<port>\d+)", text)
    if match is not None:
        return int(match.group("port"))
    match = re.search(r"127\.0\.0\.1:(?P<port>\d+)", text)
    if match is not None:
        return int(match.group("port"))
    return None
