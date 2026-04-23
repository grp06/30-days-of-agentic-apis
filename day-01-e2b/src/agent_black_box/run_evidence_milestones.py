from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel

from .config import Settings
from .events import (
    CheckpointCreatedEvent,
    FileDiffEvent,
    PreviewPublishedEvent,
)
from .fixture_policy import fixture_should_checkpoint_after_command
from .preview_execution import CommandPreviewResult, apply_command_preview
from .recorder import Recorder, RunSummary
from .sandbox_controller import CommandResult, SandboxController


class SequenceCounter(Protocol):
    def next(self) -> int: ...

    def current(self) -> int: ...


class CommandMilestoneResult(BaseModel):
    preview_url: str | None = None
    checkpoint_id: str | None = None


class DiffMilestoneResult(BaseModel):
    diff_recorded: bool = False
    checkpoint_id: str | None = None


class RunEvidenceMilestones:
    def __init__(
        self,
        *,
        settings: Settings,
        sandbox_controller: SandboxController,
        sequence: SequenceCounter,
    ) -> None:
        self.settings = settings
        self.sandbox_controller = sandbox_controller
        self.sequence = sequence
        self._last_patch = ""
        self._distinct_diff_count = 0

    async def record_command_milestones(
        self,
        *,
        run_id: str,
        recorder: Recorder,
        fixture_name: str,
        command: str,
        command_result: CommandResult,
        preview: CommandPreviewResult,
    ) -> CommandMilestoneResult:
        if preview.preview_url is not None:
            self._record_preview_publication(
                run_id=run_id,
                recorder=recorder,
                preview_url=preview.preview_url,
                preview_port=preview.preview_port or self.settings.preview_port,
            )

        checkpoint_id: str | None = None
        if command_result.exit_code == 0 and fixture_should_checkpoint_after_command(
            fixture_name,
            command,
            preview_published=preview.preview_url is not None,
        ):
            checkpoint_id = await self._record_checkpoint(
                run_id=run_id,
                recorder=recorder,
                note="successful milestone",
            )

        return CommandMilestoneResult(
            preview_url=preview.preview_url,
            checkpoint_id=checkpoint_id,
        )

    def sync_tool_result_preview_state(
        self,
        *,
        recorder: Recorder,
        summary: RunSummary,
        result: dict[str, object],
    ) -> None:
        next_preview = apply_command_preview(
            current_failure_reason=summary.preview_failure_reason,
            result=CommandPreviewResult(
                preview_url=_optional_str(result.get("preview_url")),
                preview_failure_reason=_optional_str(
                    result.get("preview_failure_reason")
                ),
            ),
        )
        summary.preview_url = summary.preview_url or next_preview.preview_url
        if next_preview.preview_url is not None:
            summary.preview_state = next_preview.preview_state
            summary.preview_last_error = next_preview.preview_last_error
            summary.preview_failure_reason = next_preview.preview_failure_reason
        elif result.get("preview_failure_reason"):
            summary.preview_failure_reason = next_preview.preview_failure_reason
        summary.checkpoint_id = summary.checkpoint_id or _optional_str(
            result.get("checkpoint_id")
        )
        recorder.update_status(
            preview_url=summary.preview_url,
            preview_state=summary.preview_state,
            preview_last_error=summary.preview_last_error,
            preview_expected=summary.preview_expected,
            preview_failure_reason=summary.preview_failure_reason,
            preview_attempted=summary.preview_url is not None
            or summary.preview_failure_reason is not None,
            checkpoint_id=summary.checkpoint_id,
        )

    async def record_managed_preview(
        self,
        *,
        run_id: str,
        recorder: Recorder,
        summary: RunSummary,
        preview: CommandPreviewResult,
    ) -> None:
        if preview.preview_url is None:
            raise ValueError("managed preview milestone requires a preview URL")

        next_preview = apply_command_preview(
            current_failure_reason=summary.preview_failure_reason,
            result=preview,
        )
        summary.preview_url = summary.preview_url or next_preview.preview_url
        summary.preview_state = next_preview.preview_state
        summary.preview_last_error = next_preview.preview_last_error
        summary.preview_failure_reason = next_preview.preview_failure_reason
        self._record_preview_publication(
            run_id=run_id,
            recorder=recorder,
            preview_url=preview.preview_url,
            preview_port=preview.preview_port or self.settings.preview_port,
        )
        recorder.update_status(
            preview_url=summary.preview_url,
            preview_state=summary.preview_state,
            preview_last_error=summary.preview_last_error,
            preview_expected=summary.preview_expected,
            preview_failure_reason=summary.preview_failure_reason,
            preview_attempted=True,
            checkpoint_id=summary.checkpoint_id,
        )

        try:
            checkpoint_id = await self._record_checkpoint(
                run_id=run_id,
                recorder=recorder,
                note="successful preview",
            )
            summary.checkpoint_id = summary.checkpoint_id or checkpoint_id
        except Exception:  # noqa: BLE001
            pass

    async def record_diff(
        self,
        run_id: str,
        recorder: Recorder,
        *,
        checkpoint_allowed: bool = True,
    ) -> DiffMilestoneResult:
        patch = await self.sandbox_controller.collect_git_diff()
        if not patch or patch == self._last_patch:
            return DiffMilestoneResult()
        self._last_patch = patch
        self._distinct_diff_count += 1
        path = recorder.write_diff(self.sequence.next(), "workspace", patch)
        recorder.append(
            FileDiffEvent(
                run_id=run_id,
                lane_id="lane-1",
                sequence=self.sequence.next(),
                patch_path=str(path),
                patch_summary=self._summarize_patch(patch),
            )
        )
        checkpoint_id: str | None = None
        if checkpoint_allowed and self._should_checkpoint_diff():
            try:
                checkpoint_id = await self._record_checkpoint(
                    run_id=run_id,
                    recorder=recorder,
                    note=self._diff_checkpoint_note(),
                )
            except Exception:  # noqa: BLE001
                checkpoint_id = None
        return DiffMilestoneResult(diff_recorded=True, checkpoint_id=checkpoint_id)

    def _record_preview_publication(
        self,
        *,
        run_id: str,
        recorder: Recorder,
        preview_url: str,
        preview_port: int,
    ) -> None:
        recorder.write_artifact_text("preview-url.txt", preview_url + "\n")
        recorder.append(
            PreviewPublishedEvent(
                run_id=run_id,
                lane_id="lane-1",
                sequence=self.sequence.next(),
                url=preview_url,
                port=preview_port,
            )
        )

    async def _record_checkpoint(
        self,
        *,
        run_id: str,
        recorder: Recorder,
        note: str,
    ) -> str:
        checkpoint_id = await self.sandbox_controller.create_checkpoint(note=note)
        recorder.write_checkpoint(
            self.sequence.current(),
            checkpoint_id,
            {"note": note},
        )
        recorder.append(
            CheckpointCreatedEvent(
                run_id=run_id,
                lane_id="lane-1",
                sequence=self.sequence.next(),
                snapshot_id=checkpoint_id,
                note=note,
            )
        )
        return checkpoint_id

    def _summarize_patch(self, patch: str) -> str:
        filenames = sorted(
            set(re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.MULTILINE))
        )
        if filenames:
            return ", ".join(filenames[:5])
        return "workspace diff updated"

    def _should_checkpoint_diff(self) -> bool:
        return self._distinct_diff_count == 1 or (
            self._distinct_diff_count > 1
            and (self._distinct_diff_count - 1) % 3 == 0
        )

    def _diff_checkpoint_note(self) -> str:
        if self._distinct_diff_count == 1:
            return "first workspace diff"
        return f"workspace diff milestone {self._distinct_diff_count}"


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None
