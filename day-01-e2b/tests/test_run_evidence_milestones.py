from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_black_box.config import Settings
from agent_black_box.recorder import Recorder, RunMetadata, RunStatus
from agent_black_box.run_evidence_milestones import RunEvidenceMilestones


class FakeSequence:
    def __init__(self) -> None:
        self.value = 0

    def next(self) -> int:
        self.value += 1
        return self.value

    def current(self) -> int:
        return self.value


class DiffSandbox:
    def __init__(self, diffs: list[str], *, fail_checkpoint: bool = False) -> None:
        self.diffs = diffs
        self.fail_checkpoint = fail_checkpoint
        self.checkpoint_notes: list[str] = []

    async def collect_git_diff(self) -> str:
        return self.diffs.pop(0)

    async def create_checkpoint(self, note: str) -> str:
        if self.fail_checkpoint:
            raise RuntimeError("checkpoint failed")
        self.checkpoint_notes.append(note)
        return f"snap-{len(self.checkpoint_notes)}"


def _recorder(tmp_path: Path) -> Recorder:
    recorder = Recorder(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
        ),
    )
    recorder.initialize_status(
        RunStatus(
            run_id="run-1",
            state="running",
            current_model_name="kimi-k2.6:cloud",
        )
    )
    return recorder


@pytest.mark.asyncio
async def test_diff_milestones_checkpoint_first_diff_and_every_third_after(
    tmp_path: Path,
) -> None:
    sandbox = DiffSandbox(
        [
            "diff --git a/index.html b/index.html\n+++ b/index.html\n+one\n",
            "diff --git a/index.html b/index.html\n+++ b/index.html\n+two\n",
            "diff --git a/index.css b/index.css\n+++ b/index.css\n+three\n",
            "diff --git a/index.js b/index.js\n+++ b/index.js\n+four\n",
            "diff --git a/index.js b/index.js\n+++ b/index.js\n+five\n",
        ]
    )
    milestones = RunEvidenceMilestones(
        settings=Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test"),
        sandbox_controller=sandbox,  # type: ignore[arg-type]
        sequence=FakeSequence(),
    )
    recorder = _recorder(tmp_path)

    results = [
        await milestones.record_diff("run-1", recorder),
        await milestones.record_diff("run-1", recorder),
        await milestones.record_diff("run-1", recorder),
        await milestones.record_diff("run-1", recorder),
        await milestones.record_diff("run-1", recorder),
    ]

    assert [result.diff_recorded for result in results] == [True] * 5
    assert [result.checkpoint_id for result in results] == [
        "snap-1",
        None,
        None,
        "snap-2",
        None,
    ]
    assert sandbox.checkpoint_notes == [
        "first workspace diff",
        "workspace diff milestone 4",
    ]
    checkpoints = sorted((tmp_path / "run-1" / "checkpoints").glob("*.json"))
    assert len(checkpoints) == 2
    checkpoint_payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in checkpoints
    ]
    assert [payload["snapshot_id"] for payload in checkpoint_payloads] == [
        "snap-1",
        "snap-2",
    ]


@pytest.mark.asyncio
async def test_diff_milestones_skip_duplicate_and_disallowed_checkpoints(
    tmp_path: Path,
) -> None:
    sandbox = DiffSandbox(
        [
            "diff --git a/index.html b/index.html\n+++ b/index.html\n+one\n",
            "diff --git a/index.html b/index.html\n+++ b/index.html\n+one\n",
            "diff --git a/index.html b/index.html\n+++ b/index.html\n+two\n",
        ]
    )
    milestones = RunEvidenceMilestones(
        settings=Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test"),
        sandbox_controller=sandbox,  # type: ignore[arg-type]
        sequence=FakeSequence(),
    )
    recorder = _recorder(tmp_path)

    first = await milestones.record_diff(
        "run-1",
        recorder,
        checkpoint_allowed=False,
    )
    duplicate = await milestones.record_diff("run-1", recorder)
    second_distinct = await milestones.record_diff("run-1", recorder)

    assert first.diff_recorded is True
    assert first.checkpoint_id is None
    assert duplicate.diff_recorded is False
    assert duplicate.checkpoint_id is None
    assert second_distinct.diff_recorded is True
    assert second_distinct.checkpoint_id is None
    assert sandbox.checkpoint_notes == []
    assert list((tmp_path / "run-1" / "checkpoints").glob("*.json")) == []


@pytest.mark.asyncio
async def test_optional_diff_checkpoint_failure_still_records_diff(
    tmp_path: Path,
) -> None:
    sandbox = DiffSandbox(
        ["diff --git a/index.html b/index.html\n+++ b/index.html\n+one\n"],
        fail_checkpoint=True,
    )
    milestones = RunEvidenceMilestones(
        settings=Settings(E2B_API_KEY="e2b_test", OLLAMA_API_KEY="ollama_test"),
        sandbox_controller=sandbox,  # type: ignore[arg-type]
        sequence=FakeSequence(),
    )
    recorder = _recorder(tmp_path)

    result = await milestones.record_diff("run-1", recorder)

    assert result.diff_recorded is True
    assert result.checkpoint_id is None
    assert sandbox.checkpoint_notes == []
    assert list((tmp_path / "run-1" / "diffs").glob("*.patch"))
    events = [
        json.loads(line)
        for line in (tmp_path / "run-1" / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [event["type"] for event in events] == ["file_diff"]
