from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_black_box.launch as launch_module
from agent_black_box.arena import ArenaMetadata, ArenaStatus
from agent_black_box.config import Settings
from agent_black_box.launch import RunLauncher
from agent_black_box.recorder import RunMetadata, RunStatus, prepare_run_directory


class FakeCoordinator:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str | None]] = []

    async def run_once(self, source, run_id: str | None = None):  # noqa: ANN001, ANN201
        self.calls.append((source, run_id))
        await asyncio.sleep(0)
        return Path("/tmp/fake")


class FakeModelCatalog:
    def __init__(self, models: list[str] | None = None) -> None:
        self.models = models or [
            "kimi-k2.6",
            "glm-5.1",
            "gemma4:31b",
            "qwen3.5:397b",
        ]

    async def list_models(self) -> list[str]:
        return self.models

    def resolve_model_name(self, requested: str, available: list[str]) -> str | None:
        if requested in available:
            return requested
        if requested.endswith(":cloud"):
            without_suffix = requested.removesuffix(":cloud")
            if without_suffix in available:
                return without_suffix
        cloud_variant = f"{requested}:cloud"
        if cloud_variant in available:
            return cloud_variant
        return None


class FailingCoordinator(FakeCoordinator):
    async def run_once(self, source, run_id: str | None = None):  # noqa: ANN001, ANN201
        self.calls.append((source, run_id))
        raise RuntimeError("sandbox bootstrap failed")


class PartiallyWritingFailingCoordinator(FakeCoordinator):
    def __init__(self, run_root: Path) -> None:
        super().__init__()
        self.run_root = run_root

    async def run_once(self, source, run_id: str | None = None):  # noqa: ANN001, ANN201
        self.calls.append((source, run_id))
        assert run_id is not None
        run_dir = self.run_root / run_id
        (run_dir / "artifacts").mkdir(exist_ok=True)
        (run_dir / "diffs").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "artifacts" / "preview-url.txt").write_text(
            "https://preview.example\n",
            encoding="utf-8",
        )
        (run_dir / "events.jsonl").write_text(
            json.dumps(
                {
                    "type": "command_completed",
                    "sequence": 7,
                    "timestamp": "2026-04-21T00:00:00+00:00",
                    "command": "pnpm preview",
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "background": True,
                    "pid": 123,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "diffs" / "0001-demo.patch").write_text(
            "diff --git a/x b/x\n",
            encoding="utf-8",
        )
        (run_dir / "checkpoints" / "0007.json").write_text(
            json.dumps({"snapshot_id": "snap-1", "note": "checkpoint"}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": "running",
                    "model_name": "kimi-k2.6:cloud",
                    "fixture_name": source.fixture_name,
                    "preview_url": "https://preview.example",
                    "preview_state": "live",
                    "preview_last_error": None,
                    "sandbox_retained": True,
                    "checkpoint_id": "snap-1",
                    "command_count": 3,
                    "diff_count": 2,
                    "tool_call_count": 5,
                    "completed_at": "2026-04-21T00:00:00+00:00",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "status.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "state": "running",
                    "current_model_name": "kimi-k2.6:cloud",
                    "latest_sequence": 11,
                    "preview_url": "https://preview.example",
                    "preview_state": "live",
                    "preview_last_error": None,
                    "sandbox_retained": True,
                    "checkpoint_id": "snap-1",
                    "is_fork": False,
                    "updated_at": "2026-04-21T00:00:00+00:00",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("sandbox bootstrap failed")


class CorruptStatusFailingCoordinator(FakeCoordinator):
    def __init__(self, run_root: Path) -> None:
        super().__init__()
        self.run_root = run_root

    async def run_once(self, source, run_id: str | None = None):  # noqa: ANN001, ANN201
        self.calls.append((source, run_id))
        assert run_id is not None
        run_dir = self.run_root / run_id
        (run_dir / "status.json").write_text('{"run_id":', encoding="utf-8")
        (run_dir / "summary.json").write_text('{"run_id":', encoding="utf-8")
        raise RuntimeError("sandbox bootstrap failed after partial json")


class CountingCoordinator(FakeCoordinator):
    created = 0
    run_ids_by_instance: dict[int, list[str | None]] = {}
    sources_by_instance: dict[int, list[object]] = {}

    def __init__(self) -> None:
        super().__init__()
        type(self).created += 1
        type(self).run_ids_by_instance[id(self)] = []
        type(self).sources_by_instance[id(self)] = []

    async def run_once(self, source, run_id: str | None = None):  # noqa: ANN001, ANN201
        type(self).run_ids_by_instance[id(self)].append(run_id)
        type(self).sources_by_instance[id(self)].append(source)
        return await super().run_once(source, run_id)

    @classmethod
    def reset(cls) -> None:
        cls.created = 0
        cls.run_ids_by_instance = {}
        cls.sources_by_instance = {}


class FailOnSecondPendingLauncher(RunLauncher):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.pending_calls = 0

    def _register_pending_run(self, *, run_id: str, source):  # noqa: ANN001, ANN201
        self.pending_calls += 1
        pending = super()._register_pending_run(run_id=run_id, source=source)
        if self.pending_calls == 2:
            raise RuntimeError("second lane registration failed")
        return pending


@pytest.mark.asyncio
async def test_launcher_records_lineage_immediately(tmp_path) -> None:  # noqa: ANN001
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    (parent_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "parent",
                "task": "original task",
                "model_name": "gemma4:31b",
                "fixture_name": "sample_frontend_task",
                "started_at": "2026-04-21T00:00:00+00:00",
                "sandbox_id": "sbx-parent",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (parent_dir / "checkpoints").mkdir()
    (parent_dir / "checkpoints" / "0082.json").write_text(
        json.dumps({"snapshot_id": "snap-82", "note": "checkpoint"}, indent=2) + "\n",
        encoding="utf-8",
    )
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(
        settings,
        coordinator_factory=FakeCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    response = await launcher.start_fork_run("parent", 82, "Restyle the hero")

    child_dir = tmp_path / response.run_id
    metadata = json.loads((child_dir / "metadata.json").read_text(encoding="utf-8"))
    status = json.loads((child_dir / "status.json").read_text(encoding="utf-8"))

    assert response.parent_run_id == "parent"
    assert metadata["parent_run_id"] == "parent"
    assert metadata["source_snapshot_id"] == "snap-82"
    assert metadata["instruction_override"] == "Restyle the hero"
    assert metadata["model_name"] == "gemma4:31b"
    assert status["state"] == "running"
    assert status["current_model_name"] == "gemma4:31b"
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_launcher_rejects_fork_from_checkpoint_dir_symlink_outside_run_root(
    tmp_path,
) -> None:  # noqa: ANN001
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    (parent_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "parent",
                "task": "original task",
                "model_name": "kimi-k2.6:cloud",
                "fixture_name": "sample_frontend_task",
                "started_at": "2026-04-21T00:00:00+00:00",
                "sandbox_id": "sbx-parent",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    outside_dir = tmp_path / "outside-checkpoints"
    outside_dir.mkdir()
    (outside_dir / "0082.json").write_text(
        json.dumps({"snapshot_id": "snap-82", "note": "checkpoint"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (parent_dir / "checkpoints").symlink_to(outside_dir, target_is_directory=True)

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(
        settings,
        coordinator_factory=FakeCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    with pytest.raises(
        FileNotFoundError, match="Checkpoint 82 not found for run parent"
    ):
        await launcher.start_fork_run("parent", 82, "Restyle the hero")


@pytest.mark.asyncio
async def test_launcher_rejects_fork_from_invalid_checkpoint_payload(
    tmp_path,
) -> None:  # noqa: ANN001
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    (parent_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": "parent",
                "task": "original task",
                "model_name": "kimi-k2.6:cloud",
                "fixture_name": "sample_frontend_task",
                "started_at": "2026-04-21T00:00:00+00:00",
                "sandbox_id": "sbx-parent",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (parent_dir / "checkpoints").mkdir()
    (parent_dir / "checkpoints" / "0082.json").write_text(
        '{"note":"missing snapshot id"}\n',
        encoding="utf-8",
    )

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(
        settings,
        coordinator_factory=FakeCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    with pytest.raises(FileNotFoundError, match="missing snapshot_id"):
        await launcher.start_fork_run("parent", 82, "Restyle the hero")


@pytest.mark.asyncio
async def test_launcher_marks_failed_launches_honestly(tmp_path) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(settings, coordinator_factory=FailingCoordinator)  # type: ignore[arg-type]

    response = await launcher.start_fixture_run("sample_frontend_task")
    await asyncio.sleep(0)

    status = json.loads(
        (tmp_path / response.run_id / "status.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (tmp_path / response.run_id / "summary.json").read_text(encoding="utf-8")
    )

    assert status["state"] == "launch_failed"
    assert status["preview_url"] is None
    assert status["preview_state"] == "expired"
    assert (
        status["preview_last_error"] == "Sandbox was not retained after launch failure."
    )
    assert (
        status["preview_failure_reason"]
        == "Preview was expected, but launch failed before it could be verified."
    )
    assert status["sandbox_retained"] is False
    assert summary["status"] == "launch_failed"
    assert summary["preview_url"] is None
    assert summary["preview_state"] == "expired"
    assert (
        summary["preview_last_error"]
        == "Sandbox was not retained after launch failure."
    )
    assert (
        summary["preview_failure_reason"]
        == "Preview was expected, but launch failed before it could be verified."
    )
    assert summary["sandbox_retained"] is False
    assert "sandbox bootstrap failed" in summary["failure_reason"]


@pytest.mark.asyncio
async def test_launcher_preserves_preview_expectation_on_early_launch_failure(
    tmp_path,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text(
        "Run the app locally and make sure the preview works before finishing.",
        encoding="utf-8",
    )
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(settings, coordinator_factory=FailingCoordinator)  # type: ignore[arg-type]

    response = await launcher.start_fixture_run("sample_frontend_task")
    await asyncio.sleep(0)

    status = json.loads(
        (tmp_path / response.run_id / "status.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (tmp_path / response.run_id / "summary.json").read_text(encoding="utf-8")
    )

    assert status["state"] == "launch_failed"
    assert status["preview_expected"] is True
    assert status["preview_state"] == "expired"
    assert status["preview_failure_reason"] == (
        "Preview was expected, but launch failed before it could be verified."
    )
    assert (
        status["preview_last_error"] == "Sandbox was not retained after launch failure."
    )
    assert summary["status"] == "launch_failed"
    assert summary["preview_expected"] is True
    assert summary["preview_state"] == "expired"
    assert summary["preview_failure_reason"] == (
        "Preview was expected, but launch failed before it could be verified."
    )
    assert (
        summary["preview_last_error"]
        == "Sandbox was not retained after launch failure."
    )


def test_launcher_preserves_event_only_preview_publish_on_launch_failure(
    tmp_path,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(settings, coordinator_factory=FailingCoordinator)  # type: ignore[arg-type]

    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="Make a simple landing page.",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="running", current_model_name="kimi-k2.6:cloud"
        ),
    )
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "type": "preview_published",
                "sequence": 3,
                "url": "https://4173-preview.example",
                "port": 4173,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    launcher._mark_launch_failed(
        "run-1", SimpleNamespace(fixture_name="sample_frontend_task"), "boom"
    )

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert status["preview_state"] == "expired"
    assert status["preview_expected"] is True
    assert (
        status["preview_last_error"] == "Sandbox was not retained after launch failure."
    )
    assert (
        status["preview_failure_reason"]
        == "Preview was expected, but launch failed before it could be verified."
    )
    assert summary["preview_state"] == "expired"
    assert summary["preview_expected"] is True
    assert (
        summary["preview_last_error"]
        == "Sandbox was not retained after launch failure."
    )
    assert (
        summary["preview_failure_reason"]
        == "Preview was expected, but launch failed before it could be verified."
    )
    assert not (run_dir / "events.jsonl").exists()


def test_launcher_preserves_event_only_preview_failure_reason_on_launch_failure(
    tmp_path,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(settings, coordinator_factory=FailingCoordinator)  # type: ignore[arg-type]

    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="Make a simple landing page.",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="running", current_model_name="kimi-k2.6:cloud"
        ),
    )
    (run_dir / "events.jsonl").write_text(
        (
            json.dumps(
                {
                    "type": "tool_call",
                    "sequence": 4,
                    "tool_name": "run_command",
                    "arguments": {"command": "pnpm preview --host --port 5173"},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "tool_result",
                    "sequence": 5,
                    "tool_name": "run_command",
                    "ok": False,
                    "result": {"error": "context deadline exceeded"},
                }
            )
            + "\n"
        ),
        encoding="utf-8",
    )

    launcher._mark_launch_failed(
        "run-1", SimpleNamespace(fixture_name="sample_frontend_task"), "boom"
    )

    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    expected_reason = "Preview command for port 5173 failed before publication: context deadline exceeded"
    assert status["preview_state"] == "expired"
    assert status["preview_failure_reason"] == expected_reason
    assert (
        status["preview_last_error"] == "Sandbox was not retained after launch failure."
    )
    assert summary["preview_state"] == "expired"
    assert summary["preview_failure_reason"] == expected_reason
    assert (
        summary["preview_last_error"]
        == "Sandbox was not retained after launch failure."
    )
    assert not (run_dir / "events.jsonl").exists()


def test_launcher_launch_failure_cleanup_does_not_follow_output_dir_symlinks(
    tmp_path,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(settings, coordinator_factory=FailingCoordinator)  # type: ignore[arg-type]
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="Make a simple landing page.",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="running", current_model_name="kimi-k2.6:cloud"
        ),
    )
    for name, filename in (
        ("artifacts", "preview-url.txt"),
        ("diffs", "0001-demo.patch"),
        ("checkpoints", "0001.json"),
    ):
        real_dir = run_dir / name
        real_dir.rmdir()
        outside_dir = tmp_path / f"outside-{name}"
        outside_dir.mkdir()
        (outside_dir / filename).write_text("keep me\n", encoding="utf-8")
        real_dir.symlink_to(outside_dir, target_is_directory=True)

    launcher._mark_launch_failed(
        "run-1", SimpleNamespace(fixture_name="sample_frontend_task"), "boom"
    )

    assert (tmp_path / "outside-artifacts" / "preview-url.txt").read_text(
        encoding="utf-8"
    ) == "keep me\n"
    assert (tmp_path / "outside-diffs" / "0001-demo.patch").read_text(
        encoding="utf-8"
    ) == "keep me\n"
    assert (tmp_path / "outside-checkpoints" / "0001.json").read_text(
        encoding="utf-8"
    ) == "keep me\n"
    assert not (run_dir / "artifacts").exists()
    assert not (run_dir / "diffs").exists()
    assert not (run_dir / "checkpoints").exists()


def test_launcher_launch_failure_cleanup_unlinks_events_symlink_only(
    tmp_path,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(settings, coordinator_factory=FailingCoordinator)  # type: ignore[arg-type]
    run_dir = prepare_run_directory(
        tmp_path,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="Make a simple landing page.",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="running", current_model_name="kimi-k2.6:cloud"
        ),
    )
    outside_events = tmp_path / "outside-events.jsonl"
    outside_events.write_text("keep me\n", encoding="utf-8")
    events_path = run_dir / "events.jsonl"
    events_path.symlink_to(outside_events)

    launcher._mark_launch_failed(
        "run-1", SimpleNamespace(fixture_name="sample_frontend_task"), "boom"
    )

    assert outside_events.read_text(encoding="utf-8") == "keep me\n"
    assert not events_path.exists()


@pytest.mark.asyncio
async def test_launcher_creates_arena_and_lane_run_artifacts(tmp_path) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=tmp_path / "arenas",
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(
        settings,
        coordinator_factory=FakeCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    response = await launcher.start_arena_run(
        "sample_frontend_task", "Shared task", lane_count=4
    )
    await asyncio.sleep(0)

    arena_dir = tmp_path / "arenas" / response.arena_id
    metadata = ArenaMetadata.model_validate_json(
        (arena_dir / "metadata.json").read_text(encoding="utf-8")
    )
    status = ArenaStatus.model_validate_json(
        (arena_dir / "status.json").read_text(encoding="utf-8")
    )

    assert len(response.lanes) == 4
    assert len(metadata.lanes) == 4
    assert [lane.model_name for lane in response.lanes] == [
        "kimi-k2.6:cloud",
        "glm-5.1:cloud",
        "gemma4:31b",
        "qwen3.5:397b",
    ]
    assert [lane.model_name for lane in metadata.lanes] == [
        "kimi-k2.6:cloud",
        "glm-5.1:cloud",
        "gemma4:31b",
        "qwen3.5:397b",
    ]
    assert status.state == "running"
    assert status.lane_states["lane-1"] == "running"
    for lane in response.lanes:
        metadata_path = tmp_path / "runs" / lane.run_id / "metadata.json"
        run_status_path = tmp_path / "runs" / lane.run_id / "status.json"
        assert metadata_path.exists()
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
        assert run_metadata["model_name"] == lane.model_name
        assert run_status["current_model_name"] == lane.model_name


@pytest.mark.asyncio
async def test_launcher_uses_fresh_coordinator_per_lane(tmp_path) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=tmp_path / "arenas",
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    CountingCoordinator.reset()
    launcher = RunLauncher(
        settings,
        coordinator_factory=CountingCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    await launcher.start_arena_run("sample_frontend_task", "Shared task", lane_count=4)
    await asyncio.sleep(0)

    assert CountingCoordinator.created == 4
    assert sorted(
        len(run_ids) for run_ids in CountingCoordinator.run_ids_by_instance.values()
    ) == [1, 1, 1, 1]
    assert sorted(
        source.model_name
        for sources in CountingCoordinator.sources_by_instance.values()
        for source in sources
    ) == [
        "gemma4:31b",
        "glm-5.1:cloud",
        "kimi-k2.6:cloud",
        "qwen3.5:397b",
    ]


@pytest.mark.asyncio
async def test_launcher_requires_enough_arena_models(tmp_path) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_ARENA_MODELS="kimi-k2.6:cloud,glm-5.1:cloud",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=tmp_path / "arenas",
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    CountingCoordinator.reset()
    launcher = RunLauncher(
        settings,
        coordinator_factory=CountingCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    with pytest.raises(ValueError, match="OLLAMA_ARENA_MODELS"):
        await launcher.start_arena_run(
            "sample_frontend_task", "Shared task", lane_count=4
        )

    assert CountingCoordinator.created == 0
    assert list((tmp_path / "runs").glob("*")) == []
    assert list((tmp_path / "arenas").glob("*")) == []


@pytest.mark.asyncio
async def test_launcher_requires_distinct_resolved_arena_models(tmp_path) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_ARENA_MODELS="kimi-k2.6:cloud,kimi-k2.6",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=tmp_path / "arenas",
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    CountingCoordinator.reset()
    launcher = RunLauncher(
        settings,
        coordinator_factory=CountingCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    with pytest.raises(ValueError, match="distinct models"):
        await launcher.start_arena_run(
            "sample_frontend_task", "Shared task", lane_count=2
        )

    assert CountingCoordinator.created == 0
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "arenas").exists()


@pytest.mark.asyncio
async def test_launcher_rejects_unresolved_arena_models_before_allocating_runs(
    tmp_path,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        OLLAMA_ARENA_MODELS="kimi-k2.6:cloud,qwen3.6:cloud",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=tmp_path / "arenas",
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    CountingCoordinator.reset()
    launcher = RunLauncher(
        settings,
        coordinator_factory=CountingCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=lambda: FakeModelCatalog(
            models=["kimi-k2.6", "qwen3.5:397b"]
        ),
    )

    with pytest.raises(ValueError, match="qwen3.6:cloud"):
        await launcher.start_arena_run(
            "sample_frontend_task", "Shared task", lane_count=2
        )

    assert CountingCoordinator.created == 0
    assert not (tmp_path / "runs").exists()
    assert not (tmp_path / "arenas").exists()


@pytest.mark.asyncio
async def test_launcher_does_not_spawn_lane_tasks_before_arena_record_exists(
    tmp_path,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    broken_arena_root = tmp_path / "arenas-file"
    broken_arena_root.write_text("not a directory", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=broken_arena_root,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    CountingCoordinator.reset()
    launcher = RunLauncher(
        settings,
        coordinator_factory=CountingCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    with pytest.raises(NotADirectoryError):
        await launcher.start_arena_run(
            "sample_frontend_task", "Shared task", lane_count=2
        )

    assert CountingCoordinator.created == 0
    assert launcher._tasks == {}
    assert list((tmp_path / "runs").glob("*")) == []


@pytest.mark.asyncio
async def test_launcher_cleans_partial_arena_record_when_status_initialization_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    real_arena_recorder = launch_module.ArenaRecorder

    class FailingStatusArenaRecorder(real_arena_recorder):
        def initialize_status(self, status: ArenaStatus) -> None:
            raise RuntimeError("arena status write failed")

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=tmp_path / "arenas",
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    CountingCoordinator.reset()
    monkeypatch.setattr(launch_module, "ArenaRecorder", FailingStatusArenaRecorder)
    launcher = RunLauncher(
        settings,
        coordinator_factory=CountingCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    with pytest.raises(RuntimeError, match="arena status write failed"):
        await launcher.start_arena_run(
            "sample_frontend_task", "Shared task", lane_count=2
        )

    assert CountingCoordinator.created == 0
    assert launcher._tasks == {}
    assert list((tmp_path / "runs").glob("*")) == []
    assert list((tmp_path / "arenas").glob("*")) == []


@pytest.mark.asyncio
async def test_launcher_does_not_delete_existing_arena_on_id_collision(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    arena_root = tmp_path / "arenas"
    existing_arena = arena_root / "existing-arena"
    existing_arena.mkdir(parents=True)
    (existing_arena / "marker.txt").write_text("keep me", encoding="utf-8")
    ids = iter(["existing-arena", "run-1", "run-2"])
    monkeypatch.setattr(launch_module, "utc_filename", lambda: next(ids))
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=arena_root,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    CountingCoordinator.reset()
    launcher = RunLauncher(
        settings,
        coordinator_factory=CountingCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    with pytest.raises(FileExistsError):
        await launcher.start_arena_run(
            "sample_frontend_task", "Shared task", lane_count=2
        )

    assert CountingCoordinator.created == 0
    assert launcher._tasks == {}
    assert list((tmp_path / "runs").glob("*")) == []
    assert (existing_arena / "marker.txt").read_text(encoding="utf-8") == "keep me"


@pytest.mark.asyncio
async def test_launcher_cleans_partial_lane_registration_before_arena_exists(
    tmp_path,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path / "runs",
        ARENA_ROOT=tmp_path / "arenas",
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    CountingCoordinator.reset()
    launcher = FailOnSecondPendingLauncher(
        settings,
        coordinator_factory=CountingCoordinator,  # type: ignore[arg-type]
        model_catalog_factory=FakeModelCatalog,
    )

    with pytest.raises(RuntimeError, match="second lane registration failed"):
        await launcher.start_arena_run(
            "sample_frontend_task", "Shared task", lane_count=2
        )

    assert CountingCoordinator.created == 0
    assert launcher._tasks == {}
    assert list((tmp_path / "runs").glob("*")) == []
    assert list((tmp_path / "arenas").glob("*")) == []


@pytest.mark.asyncio
async def test_launcher_overwrites_partial_summary_and_status_on_launch_failure(
    tmp_path,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(
        settings,
        coordinator_factory=lambda: PartiallyWritingFailingCoordinator(tmp_path),  # type: ignore[arg-type]
    )

    response = await launcher.start_fixture_run("sample_frontend_task")
    await asyncio.sleep(0)

    status = json.loads(
        (tmp_path / response.run_id / "status.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (tmp_path / response.run_id / "summary.json").read_text(encoding="utf-8")
    )

    assert status["state"] == "launch_failed"
    assert status["preview_url"] is None
    assert status["preview_state"] == "expired"
    assert (
        status["preview_last_error"] == "Sandbox was not retained after launch failure."
    )
    assert status["sandbox_retained"] is False
    assert status["checkpoint_id"] is None
    assert status["latest_sequence"] == 0
    assert summary["status"] == "launch_failed"
    assert summary["preview_url"] is None
    assert summary["preview_state"] == "expired"
    assert (
        summary["preview_last_error"]
        == "Sandbox was not retained after launch failure."
    )
    assert summary["sandbox_retained"] is False
    assert summary["checkpoint_id"] is None
    assert summary["command_count"] == 0
    assert summary["diff_count"] == 0
    assert summary["tool_call_count"] == 0
    assert "sandbox bootstrap failed" in summary["failure_reason"]
    assert not (tmp_path / response.run_id / "artifacts" / "preview-url.txt").exists()
    assert not (tmp_path / response.run_id / "events.jsonl").exists()
    assert list((tmp_path / response.run_id / "diffs").glob("*.patch")) == []
    assert list((tmp_path / response.run_id / "checkpoints").glob("*.json")) == []


@pytest.mark.asyncio
async def test_launcher_marks_launch_failed_when_existing_status_is_malformed(
    tmp_path,
) -> None:  # noqa: ANN001
    fixture_dir = tmp_path / "fixtures" / "sample_frontend_task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Do the thing", encoding="utf-8")
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = RunLauncher(
        settings,
        coordinator_factory=lambda: CorruptStatusFailingCoordinator(tmp_path),  # type: ignore[arg-type]
    )

    response = await launcher.start_fixture_run("sample_frontend_task")
    await asyncio.sleep(0)

    status = json.loads(
        (tmp_path / response.run_id / "status.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (tmp_path / response.run_id / "summary.json").read_text(encoding="utf-8")
    )

    assert status["state"] == "launch_failed"
    assert summary["status"] == "launch_failed"
    assert "partial json" in summary["failure_reason"]
