from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import httpx
from pydantic import BaseModel

from .config import Settings
from .events import (
    ModelTurnCompletedEvent,
    ModelTurnStartedEvent,
    ProtocolRepairRequestedEvent,
    RunFailedEvent,
    RunStartedEvent,
    ToolCallEvent,
)
from .model_client import (
    ModelClient,
    ModelTraceContext,
    OllamaModelClient,
    ProviderInterruptionError,
)
from .model_protocol import (
    ProtocolContext,
    ProtocolFailure,
    classify_model_decision,
    protocol_failure_kind_from_exception,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .recorder import Recorder
from .run_execution import RunEventSequence, RunExecution
from .run_lifecycle import build_running_run_state, build_terminal_run_state
from .sandbox_controller import SandboxController
from .tools import tool_schemas


class FixtureRunSource(BaseModel):
    fixture_name: str
    task_override: str | None = None
    model_name: str | None = None


class SnapshotRunSource(BaseModel):
    fixture_name: str
    parent_run_id: str
    source_snapshot_id: str
    source_checkpoint_sequence: int
    instruction_override: str
    parent_task: str
    model_name: str | None = None


RunSource = FixtureRunSource | SnapshotRunSource


class RunCoordinator:
    def __init__(
        self,
        settings: Settings,
        model_client: ModelClient | None = None,
        sandbox_controller: SandboxController | None = None,
    ) -> None:
        self.settings = settings
        self.model_client = model_client or OllamaModelClient(settings)
        self._owns_model_client = model_client is None
        self.sandbox_controller = sandbox_controller or SandboxController(settings)
        self._sequence = 0

    async def run_once(self, source: RunSource, run_id: str | None = None) -> Path:
        self._sequence = 0
        model_name = self._model_name_for_source(source)
        if self._owns_model_client:
            self.model_client = OllamaModelClient(
                self.settings,
                model_name=model_name,
                fallback_enabled=source.model_name is None,
            )
        self.model_client.reset_run_state()
        fixture_path = self.settings.fixture_root / source.fixture_name
        if not fixture_path.exists():
            raise FileNotFoundError(f"Fixture not found: {fixture_path}")

        task = self._task_for_source(source, fixture_path)
        run_id = run_id or self._build_run_id()
        context = await self._start_context(source, fixture_path)
        running = build_running_run_state(
            run_id=run_id,
            task=task,
            model_name=model_name,
            fixture_name=source.fixture_name,
            sandbox_id=context.sandbox_id,
            is_fork=isinstance(source, SnapshotRunSource),
            parent_run_id=source.parent_run_id
            if isinstance(source, SnapshotRunSource)
            else None,
            source_snapshot_id=(
                source.source_snapshot_id
                if isinstance(source, SnapshotRunSource)
                else None
            ),
            source_checkpoint_sequence=(
                source.source_checkpoint_sequence
                if isinstance(source, SnapshotRunSource)
                else None
            ),
            instruction_override=(
                source.instruction_override
                if isinstance(source, SnapshotRunSource)
                else None
            ),
        )
        metadata = running.metadata
        recorder = Recorder(
            self.settings.run_root,
            run_id,
            metadata,
            allow_existing=True,
        )
        summary = running.summary
        recorder.initialize_status(running.status)
        execution = RunExecution(
            settings=self.settings,
            sandbox_controller=self.sandbox_controller,
            sequence=RunEventSequence(
                next_value=self._next_sequence,
                current_value=lambda: self._sequence,
            ),
        )
        saw_diff = False

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    workspace_dir=context.workspace_dir,
                    fixture_name=source.fixture_name,
                    task=task,
                ),
            },
        ]

        recorder.append(
            RunStartedEvent(
                run_id=run_id,
                lane_id="lane-1",
                sequence=self._next_sequence(),
                task=task,
                fixture_name=source.fixture_name,
                model=model_name,
            )
        )

        retain_sandbox = False
        protocol_repair_attempts = 0
        try:
            for turn in range(1, self.settings.max_turns + 1):
                schemas = tool_schemas()
                model_name = self.model_client.active_model_name()
                request_metrics = self._model_turn_request_metrics(
                    model_name=model_name,
                    conversation=conversation,
                    schemas=schemas,
                )
                recorder.append(
                    ModelTurnStartedEvent(
                        run_id=run_id,
                        lane_id="lane-1",
                        sequence=self._next_sequence(),
                        turn_number=turn,
                        model_name=model_name,
                        **request_metrics,
                    )
                )
                summary.model_name = model_name
                recorder.update_status(current_model_name=summary.model_name)
                self._set_model_trace_context(
                    ModelTraceContext(
                        run_id=run_id,
                        lane_id="lane-1",
                        turn_number=turn,
                        next_sequence=self._next_sequence,
                        append_event=recorder.append,
                    )
                )
                try:
                    decision = await self.model_client.next_action(
                        conversation, schemas
                    )
                finally:
                    self._set_model_trace_context(None)
                summary.model_name = self.model_client.active_model_name()
                recorder.update_status(current_model_name=summary.model_name)
                recorder.append(
                    ModelTurnCompletedEvent(
                        run_id=run_id,
                        lane_id="lane-1",
                        sequence=self._next_sequence(),
                        turn_number=turn,
                        finish_reason=decision.finish_reason,
                        content=decision.message,
                        tool_name=decision.tool_name,
                    )
                )
                assistant_message = decision.assistant_message or {
                    "role": "assistant",
                    "content": decision.message or "",
                }
                conversation.append(assistant_message)

                action = classify_model_decision(
                    decision,
                    ProtocolContext(
                        saw_diff=saw_diff,
                        repair_attempts=protocol_repair_attempts,
                        max_repair_attempts=(
                            self.settings.model_protocol_repair_attempts
                        ),
                        turn_number=turn,
                        max_turns=self.settings.max_turns,
                    ),
                )

                if action.kind == "repair":
                    if (
                        action.reason is None
                        or action.repair_message is None
                        or action.failure_kind is None
                    ):
                        raise RuntimeError(
                            "protocol repair action missing reason or failure kind"
                        )
                    protocol_repair_attempts += 1
                    recorder.append(
                        ProtocolRepairRequestedEvent(
                            run_id=run_id,
                            lane_id="lane-1",
                            sequence=self._next_sequence(),
                            turn_number=turn,
                            repair_attempt=protocol_repair_attempts,
                            reason=action.reason,
                            failure_kind=action.failure_kind,
                            hit_generation_limit=action.hit_generation_limit,
                            message=action.repair_message,
                        )
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": action.repair_message,
                        }
                    )
                    continue

                if action.kind == "terminal_summary":
                    if action.terminal_summary is None:
                        raise RuntimeError("terminal protocol action missing summary")
                    finalized = await execution.finalize_harness_owned_run(
                        run_id=run_id,
                        recorder=recorder,
                        metadata=metadata,
                        summary=summary,
                        completion_summary=action.terminal_summary,
                        warning=(
                            "Model ended with plain text instead of finish_run; "
                            "the harness validated the recorded workspace diff."
                        ),
                    )
                    retain_sandbox = finalized
                    return recorder.run_dir

                if action.kind == "fail":
                    if action.failure_kind is None or action.reason is None:
                        raise RuntimeError(
                            "protocol failure action missing reason or failure kind"
                        )
                    raise ProtocolFailure(
                        failure_kind=action.failure_kind,
                        reason=action.reason,
                    )

                if action.kind != "tool_call":
                    raise RuntimeError(f"unknown protocol action: {action.kind}")
                if decision.tool_name is None:
                    raise RuntimeError("protocol tool action missing tool name")

                tool_args = decision.tool_arguments or {}
                recorder.append(
                    ToolCallEvent(
                        run_id=run_id,
                        lane_id="lane-1",
                        sequence=self._next_sequence(),
                        tool_name=decision.tool_name,
                        arguments=tool_args,
                    )
                )
                tool_execution = await execution.execute_tool(
                    run_id=run_id,
                    recorder=recorder,
                    summary=summary,
                    tool_name=decision.tool_name,
                    tool_args=tool_args,
                )
                if tool_execution.diff_recorded:
                    saw_diff = True
                conversation.append(
                    {
                        "role": "tool",
                        "tool_name": decision.tool_name,
                        "content": self._tool_result_content(tool_execution.payload),
                    }
                )

                if decision.tool_name == "finish_run":
                    completion_summary = self._finish_summary_from_tool_payload(
                        tool_execution.payload
                    )
                    if completion_summary is None:
                        continue
                    finalized = await execution.finalize_harness_owned_run(
                        run_id=run_id,
                        recorder=recorder,
                        metadata=metadata,
                        summary=summary,
                        completion_summary=completion_summary,
                    )
                    retain_sandbox = finalized
                    return recorder.run_dir

            raise RuntimeError(
                f"Reached max turns ({self.settings.max_turns}) without finish_run"
            )
        except ProviderInterruptionError as exc:
            if saw_diff:
                finalized = await execution.finalize_harness_owned_run(
                    run_id=run_id,
                    recorder=recorder,
                    metadata=metadata,
                    summary=summary,
                    completion_summary="Model provider interrupted after producing workspace changes.",
                    warning=(
                        "Model provider interrupted after producing workspace changes; "
                        "the harness validated the recorded diff."
                    ),
                )
                retain_sandbox = finalized
                return recorder.run_dir
            summary.model_name = self.model_client.active_model_name()
            current_status = recorder.current_status
            if current_status is None:
                raise RuntimeError("status must exist before finalizing a run")
            terminal = build_terminal_run_state(
                metadata=metadata,
                status=current_status,
                summary=summary,
                outcome="provider_interrupted",
                failure_reason=str(exc),
            )
            summary = terminal.summary
            recorder.append(
                RunFailedEvent(
                    run_id=run_id,
                    lane_id="lane-1",
                    sequence=self._next_sequence(),
                    error=str(exc),
                )
            )
            recorder.persist_terminal_state(status=terminal.status, summary=summary)
            return recorder.run_dir
        except Exception as exc:  # noqa: BLE001
            if saw_diff and self._can_finalize_after_exception(exc):
                finalized = await execution.finalize_harness_owned_run(
                    run_id=run_id,
                    recorder=recorder,
                    metadata=metadata,
                    summary=summary,
                    completion_summary=(
                        "Model stopped after producing workspace changes."
                    ),
                    warning=(
                        f"Model stopped after producing workspace changes ({type(exc).__name__}); "
                        "the harness validated the recorded diff."
                    ),
                )
                retain_sandbox = finalized
                return recorder.run_dir
            failure_kind = protocol_failure_kind_from_exception(exc)
            failure_reason = self._failure_reason_for_exception(exc)
            summary.model_name = self.model_client.active_model_name()
            current_status = recorder.current_status
            if current_status is None:
                raise RuntimeError("status must exist before finalizing a run")
            terminal = build_terminal_run_state(
                metadata=metadata,
                status=current_status,
                summary=summary,
                outcome="failed",
                failure_reason=failure_reason,
            )
            summary = terminal.summary
            recorder.append(
                RunFailedEvent(
                    run_id=run_id,
                    lane_id="lane-1",
                    sequence=self._next_sequence(),
                    error=failure_reason,
                    failure_kind=failure_kind,
                )
            )
            recorder.persist_terminal_state(status=terminal.status, summary=summary)
            return recorder.run_dir
        finally:
            await self.sandbox_controller.close(retain=retain_sandbox)

    async def _start_context(
        self,
        source: RunSource,
        fixture_path: Path,
    ):
        if isinstance(source, SnapshotRunSource):
            return await self.sandbox_controller.start_run_from_snapshot(
                source.source_snapshot_id
            )
        return await self.sandbox_controller.start_run_from_fixture(fixture_path)

    def _task_for_source(self, source: RunSource, fixture_path: Path) -> str:
        if isinstance(source, SnapshotRunSource):
            return (
                f"{source.parent_task}\n\n"
                "Fork instruction override:\n"
                f"- {source.instruction_override}\n"
            )
        return source.task_override or (fixture_path / "TASK.md").read_text(
            encoding="utf-8"
        )

    def _model_name_for_source(self, source: RunSource) -> str:
        return source.model_name or self.settings.ollama_model

    def _can_finalize_after_exception(self, exc: Exception) -> bool:
        if isinstance(exc, ProtocolFailure):
            return False
        if isinstance(exc, httpx.TimeoutException):
            return True
        message = str(exc)
        return "without calling finish_run" in message

    def _failure_reason_for_exception(self, exc: Exception) -> str:
        if isinstance(exc, httpx.TimeoutException):
            return (
                "Model provider timed out before producing a finishable workspace diff."
            )
        return str(exc)

    def _tool_result_content(self, payload: dict[str, object]) -> str:
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)

    def _finish_summary_from_tool_payload(
        self, payload: dict[str, object]
    ) -> str | None:
        if payload.get("ok") is not True:
            return None
        summary = payload.get("summary")
        return summary if isinstance(summary, str) and summary.strip() else None

    def _set_model_trace_context(self, context: ModelTraceContext | None) -> None:
        set_trace_context = getattr(self.model_client, "set_trace_context", None)
        if set_trace_context is not None:
            set_trace_context(context)

    def _model_turn_request_metrics(
        self,
        *,
        model_name: str,
        conversation: list[dict[str, object]],
        schemas: list[dict[str, object]],
    ) -> dict[str, int | None]:
        last_message = conversation[-1] if conversation else {}
        last_content = self._message_content_text(last_message.get("content"))
        last_tool_result_chars = (
            len(last_content) if last_message.get("role") == "tool" else None
        )
        body = {
            "model": model_name,
            "messages": conversation,
            "tools": schemas,
            "stream": False,
        }
        return {
            "message_count": len(conversation),
            "tool_schema_count": len(schemas),
            "conversation_chars": sum(
                len(self._message_content_text(message.get("content")))
                for message in conversation
            ),
            "last_message_chars": len(last_content),
            "last_tool_result_chars": last_tool_result_chars,
            "request_body_bytes": len(
                json.dumps(body, ensure_ascii=True).encode("utf-8")
            ),
        }

    def _message_content_text(self, content: object) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True, sort_keys=True)

    def _build_run_id(self) -> str:
        return utc_filename()

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence


def utc_filename() -> str:
    from datetime import UTC, datetime

    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
