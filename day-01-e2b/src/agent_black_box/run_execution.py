from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from .config import Settings
from .events import (
    CommandCompletedEvent,
    CommandStartedEvent,
    CommandStreamEvent,
    RunCompletedEvent,
    RunFailedEvent,
    ToolResultEvent,
)
from .fixture_policy import (
    fixture_policy,
    fixture_should_checkpoint_after_command,
    fixture_suppresses_command_preview_detection,
)
from .preview_execution import (
    CommandPreviewResult,
    resolve_command_exception_preview,
    resolve_command_preview,
)
from .recorder import Recorder, RunMetadata, RunSummary
from .run_evidence_milestones import RunEvidenceMilestones
from .run_lifecycle import build_terminal_run_state
from .sandbox_controller import CommandResult, SandboxController


@dataclass(frozen=True)
class RunEventSequence:
    next_value: Callable[[], int]
    current_value: Callable[[], int]

    def next(self) -> int:
        return self.next_value()

    def current(self) -> int:
        return self.current_value()


class ToolExecutionResult(BaseModel):
    payload: dict[str, Any]
    diff_recorded: bool = False


class RunExecution:
    def __init__(
        self,
        *,
        settings: Settings,
        sandbox_controller: SandboxController,
        sequence: RunEventSequence,
    ) -> None:
        self.settings = settings
        self.sandbox_controller = sandbox_controller
        self.sequence = sequence
        self.milestones = RunEvidenceMilestones(
            settings=settings,
            sandbox_controller=sandbox_controller,
            sequence=sequence,
        )

    async def execute_tool(
        self,
        *,
        run_id: str,
        recorder: Recorder,
        summary: RunSummary,
        tool_name: str,
        tool_args: dict[str, object],
    ) -> ToolExecutionResult:
        preview_url_for_result: str | None = None
        checkpoint_id_for_result: str | None = None
        try:
            if tool_name == "read_file":
                content = await self.sandbox_controller.read_file(
                    str(tool_args["path"])
                )
                result: dict[str, Any] = {"ok": True, "content": content}
            elif tool_name == "apply_patch":
                await self.sandbox_controller.apply_patch(str(tool_args["patch_text"]))
                result = {"ok": True, "applied": True}
                diff_result = await self.milestones.record_diff(run_id, recorder)
                if diff_result.diff_recorded:
                    result["diff_recorded"] = True
                if diff_result.checkpoint_id is not None:
                    checkpoint_id_for_result = diff_result.checkpoint_id
                    result["checkpoint_id"] = diff_result.checkpoint_id
            elif tool_name == "write_file":
                await self.sandbox_controller.write_file(
                    str(tool_args["path"]),
                    str(tool_args["content"]),
                )
                result = {"ok": True, "written": True}
                diff_result = await self.milestones.record_diff(run_id, recorder)
                if diff_result.diff_recorded:
                    result["diff_recorded"] = True
                if diff_result.checkpoint_id is not None:
                    checkpoint_id_for_result = diff_result.checkpoint_id
                    result["checkpoint_id"] = diff_result.checkpoint_id
            elif tool_name == "run_command":
                command = str(tool_args["command"])
                timeout_seconds = int(tool_args.get("timeout_seconds", 120))
                command_result = await self._run_command_with_events(
                    run_id=run_id,
                    recorder=recorder,
                    command=command,
                    timeout_seconds=timeout_seconds,
                )
                result = {
                    "ok": command_result.exit_code == 0,
                    "stdout": command_result.stdout,
                    "stderr": command_result.stderr,
                    "exit_code": command_result.exit_code,
                    "pid": command_result.pid,
                    "background": command_result.background,
                }
                if not fixture_suppresses_command_preview_detection(
                    summary.fixture_name
                ):
                    maybe_preview = await resolve_command_preview(
                        sandbox_controller=self.sandbox_controller,
                        command=command,
                        command_result=command_result,
                    )
                else:
                    maybe_preview = CommandPreviewResult()
                if maybe_preview.preview_url is not None:
                    preview_url_for_result = maybe_preview.preview_url
                if maybe_preview.preview_failure_reason is not None:
                    result["preview_failure_reason"] = (
                        maybe_preview.preview_failure_reason
                    )
                command_will_checkpoint = (
                    command_result.exit_code == 0
                    and fixture_should_checkpoint_after_command(
                        summary.fixture_name,
                        command,
                        preview_published=maybe_preview.preview_url is not None,
                    )
                )
                diff_result = await self.milestones.record_diff(
                    run_id,
                    recorder,
                    checkpoint_allowed=not command_will_checkpoint,
                )
                if diff_result.diff_recorded:
                    result["diff_recorded"] = True
                if diff_result.checkpoint_id is not None:
                    checkpoint_id_for_result = diff_result.checkpoint_id
                    result["checkpoint_id"] = diff_result.checkpoint_id
                milestone_result = await self.milestones.record_command_milestones(
                    run_id=run_id,
                    recorder=recorder,
                    fixture_name=summary.fixture_name,
                    command=command,
                    command_result=command_result,
                    preview=maybe_preview,
                )
                if milestone_result.preview_url is not None:
                    result["preview_url"] = milestone_result.preview_url
                if milestone_result.checkpoint_id is not None:
                    checkpoint_id_for_result = milestone_result.checkpoint_id
                    result["checkpoint_id"] = milestone_result.checkpoint_id
            elif tool_name == "finish_run":
                result = {"ok": True, "summary": str(tool_args["summary"])}
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": str(exc)}
            if tool_name == "run_command" and not fixture_suppresses_command_preview_detection(
                summary.fixture_name
            ):
                preview_error = resolve_command_exception_preview(
                    sandbox_controller=self.sandbox_controller,
                    command=str(tool_args.get("command", "")),
                    error=str(exc),
                )
                if preview_error.preview_failure_reason is not None:
                    result["preview_failure_reason"] = (
                        preview_error.preview_failure_reason
                    )
            if preview_url_for_result is not None:
                result["preview_url"] = preview_url_for_result
            if checkpoint_id_for_result is not None:
                result["checkpoint_id"] = checkpoint_id_for_result

        recorder.append(
            ToolResultEvent(
                run_id=run_id,
                lane_id="lane-1",
                sequence=self.sequence.next(),
                tool_name=tool_name,
                ok=bool(result.get("ok", False)),
                result=result,
            )
        )
        self.milestones.sync_tool_result_preview_state(
            recorder=recorder,
            summary=summary,
            result=result,
        )
        return ToolExecutionResult(
            payload=result,
            diff_recorded=bool(result.get("diff_recorded", False)),
        )

    async def finalize_harness_owned_run(
        self,
        *,
        run_id: str,
        recorder: Recorder,
        metadata: RunMetadata,
        summary: RunSummary,
        completion_summary: str,
        warning: str | None = None,
    ) -> bool:
        summary.failure_reason = warning
        build_failure = await self._run_managed_build(
            run_id=run_id,
            recorder=recorder,
            summary=summary,
        )
        if build_failure is not None:
            summary.preview_failure_reason = (
                "Preview was not attempted because the managed build failed."
            )
            current_status = recorder.current_status
            if current_status is None:
                raise RuntimeError("status must exist before finalizing a run")
            terminal = build_terminal_run_state(
                metadata=metadata,
                status=current_status,
                summary=summary,
                outcome="failed",
                failure_reason=build_failure,
            )
            recorder.append(
                RunFailedEvent(
                    run_id=run_id,
                    lane_id="lane-1",
                    sequence=self.sequence.next(),
                    error=build_failure,
                )
            )
            recorder.persist_terminal_state(
                status=terminal.status,
                summary=terminal.summary,
            )
            return False

        await self._ensure_managed_preview(
            run_id=run_id,
            recorder=recorder,
            summary=summary,
        )
        current_status = recorder.current_status
        if current_status is None:
            raise RuntimeError("status must exist before finalizing a run")
        terminal = build_terminal_run_state(
            metadata=metadata,
            status=current_status,
            summary=summary,
            outcome="succeeded",
            preview_url=summary.preview_url,
        )
        recorder.append(
            RunCompletedEvent(
                run_id=run_id,
                lane_id="lane-1",
                sequence=self.sequence.next(),
                summary=completion_summary,
            )
        )
        recorder.persist_terminal_state(
            status=terminal.status, summary=terminal.summary
        )
        return True

    async def _ensure_managed_preview(
        self,
        *,
        run_id: str,
        recorder: Recorder,
        summary: RunSummary,
    ) -> None:
        policy = fixture_policy(summary.fixture_name)
        if policy is None or policy.preview is None or summary.preview_url is not None:
            return

        preview = policy.preview
        if await self._try_record_managed_preview(
            run_id=run_id,
            recorder=recorder,
            summary=summary,
            port=preview.port,
            readiness_timeout_seconds=2,
        ):
            return

        await self._run_command_with_events(
            run_id=run_id,
            recorder=recorder,
            command=preview.cleanup_command,
            timeout_seconds=10,
        )
        command_result = await self._run_command_with_events(
            run_id=run_id,
            recorder=recorder,
            command=preview.command,
            timeout_seconds=preview.start_timeout_seconds,
        )
        if command_result.exit_code != 0:
            error = (
                command_result.stderr
                or command_result.stdout
                or "preview command failed without output"
            ).strip()
            if "port" in error.lower() and "in use" in error.lower():
                if await self._try_record_managed_preview(
                    run_id=run_id,
                    recorder=recorder,
                    summary=summary,
                    port=preview.port,
                    readiness_timeout_seconds=preview.start_timeout_seconds,
                ):
                    return
            self._record_managed_preview_failure(
                recorder=recorder,
                summary=summary,
                reason=f"Managed preview command for port {preview.port} failed: {error}",
            )
            return

        if await self._try_record_managed_preview(
            run_id=run_id,
            recorder=recorder,
            summary=summary,
            port=preview.port,
            readiness_timeout_seconds=preview.start_timeout_seconds,
        ):
            return

        self._record_managed_preview_failure(
            recorder=recorder,
            summary=summary,
            reason=(
                f"Managed preview on port {preview.port} did not become publishable: "
                f"{summary.preview_last_error or 'preview server did not become ready'}"
            ),
        )

    async def _try_record_managed_preview(
        self,
        *,
        run_id: str,
        recorder: Recorder,
        summary: RunSummary,
        port: int,
        readiness_timeout_seconds: int,
    ) -> bool:
        try:
            preview_url = await self.sandbox_controller.publish_preview(
                port,
                readiness_timeout_seconds=readiness_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            summary.preview_last_error = str(exc)
            return False

        await self.milestones.record_managed_preview(
            run_id=run_id,
            recorder=recorder,
            summary=summary,
            preview=CommandPreviewResult(
                preview_url=preview_url,
                preview_port=port,
            ),
        )
        return True

    async def _run_managed_build(
        self,
        *,
        run_id: str,
        recorder: Recorder,
        summary: RunSummary,
    ) -> str | None:
        policy = fixture_policy(summary.fixture_name)
        if policy is None or policy.build_command is None:
            return None
        if policy.setup_command is not None:
            setup_result = await self._run_command_with_events(
                run_id=run_id,
                recorder=recorder,
                command=policy.setup_command,
                timeout_seconds=policy.setup_timeout_seconds,
            )
            if setup_result.exit_code != 0:
                error = (
                    setup_result.stderr
                    or setup_result.stdout
                    or "setup command failed without output"
                ).strip()
                return f"Managed setup command failed: {error}"
        command_result = await self._run_command_with_events(
            run_id=run_id,
            recorder=recorder,
            command=policy.build_command,
            timeout_seconds=policy.build_timeout_seconds,
        )
        if command_result.exit_code == 0:
            return None
        error = (
            command_result.stderr
            or command_result.stdout
            or "build command failed without output"
        ).strip()
        return f"Managed build command failed: {error}"

    async def _run_command_with_events(
        self,
        *,
        run_id: str,
        recorder: Recorder,
        command: str,
        timeout_seconds: int,
    ) -> CommandResult:
        recorder.append(
            CommandStartedEvent(
                run_id=run_id,
                lane_id="lane-1",
                sequence=self.sequence.next(),
                command=command,
                cwd=self.sandbox_controller.workspace_dir,
                background=self.sandbox_controller.is_background_command(command),
            )
        )

        async def on_stdout(chunk: str) -> None:
            recorder.append(
                CommandStreamEvent(
                    run_id=run_id,
                    lane_id="lane-1",
                    sequence=self.sequence.next(),
                    command=command,
                    stream="stdout",
                    chunk=chunk,
                )
            )

        async def on_stderr(chunk: str) -> None:
            recorder.append(
                CommandStreamEvent(
                    run_id=run_id,
                    lane_id="lane-1",
                    sequence=self.sequence.next(),
                    command=command,
                    stream="stderr",
                    chunk=chunk,
                )
            )

        command_result = await self.sandbox_controller.run_command(
            command,
            timeout_seconds,
            on_stdout=on_stdout,
            on_stderr=on_stderr,
        )
        recorder.append(
            CommandCompletedEvent(
                run_id=run_id,
                lane_id="lane-1",
                sequence=self.sequence.next(),
                command=command,
                exit_code=command_result.exit_code,
                stdout=command_result.stdout,
                stderr=command_result.stderr,
                background=command_result.background,
                pid=command_result.pid,
            )
        )
        return command_result

    def _record_managed_preview_failure(
        self,
        *,
        recorder: Recorder,
        summary: RunSummary,
        reason: str,
    ) -> None:
        summary.preview_state = "unavailable"
        summary.preview_last_error = None
        summary.preview_failure_reason = reason
        recorder.update_status(
            preview_url=summary.preview_url,
            preview_state=summary.preview_state,
            preview_last_error=summary.preview_last_error,
            preview_expected=summary.preview_expected,
            preview_failure_reason=summary.preview_failure_reason,
            preview_attempted=True,
            checkpoint_id=summary.checkpoint_id,
        )
