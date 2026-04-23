from __future__ import annotations

import json

from agent_black_box.arena import ArenaMetadata, ArenaLaneRecord, ArenaStatus
from agent_black_box.arena_store import ArenaListItem, ArenaProjection, ArenaLaneSummary
from fastapi.testclient import TestClient

from agent_black_box.config import Settings
from agent_black_box.launch import LaunchResponse
from agent_black_box.recorder import RunMetadata, RunStatus, RunSummary, prepare_run_directory
from agent_black_box.run_projection import ProjectedRun
from agent_black_box.run_store import RunStore
from agent_black_box.server import create_app


class FakeLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    async def start_fork_run(
        self,
        parent_run_id: str,
        checkpoint_sequence: int,
        instruction_override: str,
    ) -> LaunchResponse:
        self.calls.append((parent_run_id, checkpoint_sequence, instruction_override))
        return LaunchResponse(run_id="child-run", parent_run_id=parent_run_id, status="running")

    def get_launch_hint(self, run_id: str):  # noqa: ANN201
        if run_id == "pending-run":
            return type(
                "Hint",
                (),
                {
                    "model_dump": lambda self, mode="json": {
                        "run_id": "pending-run",
                        "state": "running",
                        "current_model_name": "kimi-k2.6:cloud",
                        "latest_sequence": 0,
                        "preview_url": None,
                        "checkpoint_id": None,
                        "is_fork": True,
                    }
                },
            )()
        return None


class FakeArenaService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def create_arena(self, fixture_name: str, task_override: str):  # noqa: ANN201
        self.calls.append((fixture_name, task_override))
        return type(
            "ArenaLaunch",
            (),
            {
                "model_dump": lambda self, mode="json": {
                    "arena_id": "arena-1",
                    "status": "running",
                    "lanes": [
                        {"lane_id": "lane-1", "run_id": "run-1"},
                        {"lane_id": "lane-2", "run_id": "run-2"},
                        {"lane_id": "lane-3", "run_id": "run-3"},
                        {"lane_id": "lane-4", "run_id": "run-4"},
                    ],
                }
            },
        )()

    def list_arenas(self) -> list[ArenaListItem]:
        return [
            ArenaListItem(
                arena_id="arena-1",
                fixture_name="sample_frontend_task",
                state="running",
                total_lanes=4,
                completed_lanes=0,
            )
        ]

    def get_arena(self, arena_id: str) -> ArenaProjection:
        return ArenaProjection(
            metadata=ArenaMetadata(
                arena_id=arena_id,
                fixture_name="sample_frontend_task",
                task="shared task",
                lanes=[
                    ArenaLaneRecord(lane_id="lane-1", run_id="run-1"),
                    ArenaLaneRecord(lane_id="lane-2", run_id="run-2"),
                ],
            ),
            status=ArenaStatus(
                arena_id=arena_id,
                state="running",
                total_lanes=2,
                completed_lanes=0,
                lane_states={"lane-1": "running", "lane-2": "running"},
            ),
            lanes=[
                ArenaLaneSummary(lane_id="lane-1", run_id="run-1", state="running", model_name="kimi-k2.6:cloud"),
                ArenaLaneSummary(lane_id="lane-2", run_id="run-2", state="running", model_name="kimi-k2.6:cloud"),
            ],
        )


class FailingArenaService(FakeArenaService):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    async def create_arena(self, fixture_name: str, task_override: str):  # noqa: ANN201
        raise self.error


class FakePreviewService:
    def __init__(self, error_by_run_id: dict[str, Exception] | None = None) -> None:
        self.calls: list[str] = []
        self.error_by_run_id = error_by_run_id or {}

    async def refresh_preview(self, run_id: str) -> None:
        self.calls.append(run_id)
        if run_id in self.error_by_run_id:
            raise self.error_by_run_id[run_id]


class OverrideProjectedRunStore(RunStore):
    def load_projected_run(self, run_id: str) -> ProjectedRun | None:
        projected = super().load_projected_run(run_id)
        if projected is None:
            return None
        status = projected.status.model_copy(update={"preview_last_error": "custom-store"})
        return projected.model_copy(update={"status": status})


def _write_parent_run(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="parent-run",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
        sandbox_id="sbx-1",
    )
    run_dir = prepare_run_directory(
        tmp_path,
        "parent-run",
        metadata,
        RunStatus(run_id="parent-run", state="succeeded", current_model_name="kimi-k2.6:cloud"),
    )
    (run_dir / "checkpoints" / "0082.json").write_text(
        json.dumps({"snapshot_id": "snap-82", "note": "checkpoint"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (run_dir / "diffs" / "0001-demo.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (run_dir / "summary.json").write_text(
        RunSummary(
            run_id="parent-run",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-1",
            sandbox_retained=True,
            preview_state="retained",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )


def test_server_exposes_run_endpoints(tmp_path) -> None:  # noqa: ANN001
    _write_parent_run(tmp_path)
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    launcher = FakeLauncher()
    preview_service = FakePreviewService()
    app = create_app(
        settings=settings,
        launcher=launcher,
        run_store=RunStore(tmp_path),
        arena_service=FakeArenaService(),
        preview_service=preview_service,  # type: ignore[arg-type]
    )
    client = TestClient(app)

    arenas_response = client.get("/api/arenas")
    demos_response = client.get("/api/demo-catalog")
    create_arena_response = client.post(
        "/api/arenas",
        json={"fixture_name": "sample_frontend_task", "task_override": "Repair the app"},
    )
    arena_detail_response = client.get("/api/arenas/arena-1")
    list_response = client.get("/api/runs")
    detail_response = client.get("/api/runs/parent-run")
    refresh_preview_response = client.post("/api/runs/parent-run/preview/refresh")
    diff_response = client.get("/api/runs/parent-run/diffs/0001-demo.patch")
    status_response = client.get("/api/runs/parent-run/status")
    fork_response = client.post(
        "/api/runs/parent-run/fork",
        json={"checkpoint_sequence": 82, "instruction_override": "Restyle the hero"},
    )
    pending_response = client.get("/api/runs/pending-run/status")

    assert arenas_response.status_code == 200
    assert arenas_response.json()["arenas"][0]["arena_id"] == "arena-1"
    assert demos_response.status_code == 200
    assert [demo["demo_id"] for demo in demos_response.json()["demos"]] == [
        "hello-world-static"
    ]
    assert create_arena_response.status_code == 200
    assert create_arena_response.json()["arena_id"] == "arena-1"
    assert arena_detail_response.status_code == 200
    assert arena_detail_response.json()["metadata"]["arena_id"] == "arena-1"
    assert list_response.status_code == 200
    assert list_response.json()["runs"][0]["run_id"] == "parent-run"
    assert list_response.json()["runs"][0]["demo_summary"] is not None
    assert detail_response.status_code == 200
    assert detail_response.json()["metadata"]["run_id"] == "parent-run"
    assert detail_response.json()["demo_summary"] is not None
    assert refresh_preview_response.status_code == 200
    assert refresh_preview_response.json()["metadata"]["run_id"] == "parent-run"
    assert preview_service.calls == ["parent-run"]
    assert diff_response.text.startswith("diff --git")
    assert status_response.json()["state"] == "succeeded"
    assert fork_response.json()["run_id"] == "child-run"
    assert pending_response.json()["state"] == "running"


def test_server_uses_configured_frontend_origins(tmp_path) -> None:  # noqa: ANN001
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
        FRONTEND_ALLOWED_ORIGINS="https://demo.example,https://www.demo.example",
    )
    app = create_app(
        settings=settings,
        launcher=FakeLauncher(),
        run_store=RunStore(tmp_path),
        arena_service=FakeArenaService(),
        preview_service=FakePreviewService(),  # type: ignore[arg-type]
    )

    cors_middleware = next(middleware for middleware in app.user_middleware if middleware.cls.__name__ == "CORSMiddleware")

    assert cors_middleware.kwargs["allow_origins"] == [
        "https://demo.example",
        "https://www.demo.example",
    ]


def test_server_returns_client_error_for_invalid_arena_launch(tmp_path) -> None:  # noqa: ANN001
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    app = create_app(
        settings=settings,
        launcher=FakeLauncher(),
        run_store=RunStore(tmp_path),
        arena_service=FailingArenaService(ValueError("bad model config")),
        preview_service=FakePreviewService(),  # type: ignore[arg-type]
    )
    client = TestClient(app)

    response = client.post(
        "/api/arenas",
        json={"fixture_name": "sample_frontend_task", "task_override": "Repair the app"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad model config"


def test_server_reuses_injected_run_store_for_run_projection_routes(tmp_path) -> None:  # noqa: ANN001
    _write_parent_run(tmp_path)
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    preview_service = FakePreviewService()
    store = OverrideProjectedRunStore(tmp_path)
    app = create_app(
        settings=settings,
        launcher=FakeLauncher(),
        run_store=store,
        arena_service=FakeArenaService(),
        preview_service=preview_service,  # type: ignore[arg-type]
    )
    client = TestClient(app)

    detail_response = client.get("/api/runs/parent-run")
    refresh_preview_response = client.post("/api/runs/parent-run/preview/refresh")

    assert detail_response.status_code == 200
    assert detail_response.json()["status"]["preview_last_error"] == "custom-store"
    assert refresh_preview_response.status_code == 200
    assert refresh_preview_response.json()["status"]["preview_last_error"] == "custom-store"


def test_server_rejects_diff_path_traversal(tmp_path) -> None:  # noqa: ANN001
    _write_parent_run(tmp_path)
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    (tmp_path / "parent-run" / "metadata.secret").write_text("nope", encoding="utf-8")
    app = create_app(
        settings=settings,
        launcher=FakeLauncher(),
        run_store=RunStore(tmp_path),
        arena_service=FakeArenaService(),
        preview_service=FakePreviewService(),  # type: ignore[arg-type]
    )
    client = TestClient(app)

    response = client.get("/api/runs/parent-run/diffs/../metadata.secret")

    assert response.status_code == 404


def test_server_rejects_diff_dir_symlink_outside_run_root(tmp_path) -> None:  # noqa: ANN001
    _write_parent_run(tmp_path)
    diff_dir = tmp_path / "parent-run" / "diffs"
    for child in diff_dir.iterdir():
        child.unlink()
    diff_dir.rmdir()
    outside_dir = tmp_path / "outside-diffs"
    outside_dir.mkdir()
    (outside_dir / "0001-demo.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    diff_dir.symlink_to(outside_dir, target_is_directory=True)

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    app = create_app(
        settings=settings,
        launcher=FakeLauncher(),
        run_store=RunStore(tmp_path),
        arena_service=FakeArenaService(),
        preview_service=FakePreviewService(),  # type: ignore[arg-type]
    )
    client = TestClient(app)

    response = client.get("/api/runs/parent-run/diffs/0001-demo.patch")

    assert response.status_code == 404


def test_server_rejects_artifact_path_traversal(tmp_path) -> None:  # noqa: ANN001
    _write_parent_run(tmp_path)
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    app = create_app(
        settings=settings,
        launcher=FakeLauncher(),
        run_store=RunStore(tmp_path),
        arena_service=FakeArenaService(),
        preview_service=FakePreviewService(),  # type: ignore[arg-type]
    )
    client = TestClient(app)

    response = client.get("/api/runs/parent-run/artifacts/../metadata.json")

    assert response.status_code == 404


def test_server_rejects_preview_refresh_for_running_runs(tmp_path) -> None:  # noqa: ANN001
    metadata = RunMetadata(
        run_id="running-run",
        task="task",
        model_name="kimi-k2.6:cloud",
        fixture_name="sample_frontend_task",
        sandbox_id="sbx-1",
    )
    prepare_run_directory(
        tmp_path,
        "running-run",
        metadata,
        RunStatus(run_id="running-run", state="running", current_model_name="kimi-k2.6:cloud"),
    )
    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    app = create_app(
        settings=settings,
        launcher=FakeLauncher(),
        run_store=RunStore(tmp_path),
        arena_service=FakeArenaService(),
        preview_service=FakePreviewService(
            error_by_run_id={
                "running-run": RuntimeError("Preview refresh is only available after a run has finished.")
            }
        ),  # type: ignore[arg-type]
    )
    client = TestClient(app)

    response = client.post("/api/runs/running-run/preview/refresh")

    assert response.status_code == 409


def test_server_rejects_run_detail_when_metadata_symlink_escapes_run_root(tmp_path) -> None:  # noqa: ANN001
    _write_parent_run(tmp_path)
    run_dir = tmp_path / "parent-run"
    outside_metadata = tmp_path / "outside-metadata.json"
    outside_metadata.write_text(
        RunMetadata(
            run_id="outside-run",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="sample_frontend_task",
            sandbox_id="sbx-outside",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "metadata.json").unlink()
    (run_dir / "metadata.json").symlink_to(outside_metadata)

    settings = Settings(
        E2B_API_KEY="e2b_test",
        OLLAMA_API_KEY="ollama_test",
        RUN_ROOT=tmp_path,
        FIXTURE_ROOT=tmp_path / "fixtures",
    )
    app = create_app(
        settings=settings,
        launcher=FakeLauncher(),
        run_store=RunStore(tmp_path),
        arena_service=FakeArenaService(),
        preview_service=FakePreviewService(),  # type: ignore[arg-type]
    )
    client = TestClient(app)

    response = client.get("/api/runs/parent-run")

    assert response.status_code == 404
