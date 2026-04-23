from __future__ import annotations

import mimetypes

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .arena_service import ArenaService
from .arena_store import ArenaListItem, ArenaProjection
from .config import Settings, get_settings
from .demo_catalog import DemoCatalogEntry, load_demo_catalog
from .launch import RunLauncher
from .preview_service import PreviewService
from .replay import RunProjection, load_run_projection
from .run_projection import RunListItem
from .run_store import RunStore


class RunsResponse(BaseModel):
    runs: list[RunListItem]


class ForkRequest(BaseModel):
    checkpoint_sequence: int
    instruction_override: str


class ArenasResponse(BaseModel):
    arenas: list[ArenaListItem]


class DemoCatalogResponse(BaseModel):
    demos: list[DemoCatalogEntry]


class CreateArenaRequest(BaseModel):
    fixture_name: str
    task_override: str


def create_app(
    settings: Settings | None = None,
    launcher: RunLauncher | None = None,
    run_store: RunStore | None = None,
    arena_service: ArenaService | None = None,
    preview_service: PreviewService | None = None,
) -> FastAPI:
    app = FastAPI(title="Agent Black Box API")
    resolved_settings = settings or get_settings()
    resolved_launcher = launcher or RunLauncher(resolved_settings)
    resolved_store = run_store or RunStore(resolved_settings.run_root)
    resolved_arena_service = arena_service or ArenaService(
        resolved_settings,
        launcher=resolved_launcher,
        run_store=resolved_store,
    )
    resolved_preview_service = preview_service or PreviewService(
        resolved_settings,
        run_store=resolved_store,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.frontend_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/demo-catalog", response_model=DemoCatalogResponse)
    async def get_demo_catalog() -> DemoCatalogResponse:
        return DemoCatalogResponse(demos=load_demo_catalog())

    @app.get("/api/runs", response_model=RunsResponse)
    async def list_runs() -> RunsResponse:
        return RunsResponse(runs=resolved_store.list_runs())

    @app.get("/api/arenas", response_model=ArenasResponse)
    async def list_arenas() -> ArenasResponse:
        return ArenasResponse(arenas=resolved_arena_service.list_arenas())

    @app.post("/api/arenas")
    async def create_arena(request: CreateArenaRequest) -> dict[str, object]:
        try:
            response = await resolved_arena_service.create_arena(
                fixture_name=request.fixture_name,
                task_override=request.task_override,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return response.model_dump(mode="json")

    @app.get("/api/arenas/{arena_id}", response_model=ArenaProjection)
    async def get_arena(arena_id: str) -> ArenaProjection:
        try:
            return resolved_arena_service.get_arena(arena_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}", response_model=RunProjection)
    async def get_run(run_id: str) -> RunProjection:
        try:
            return load_run_projection(resolved_store.get_run_dir(run_id), store=resolved_store)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/runs/{run_id}/preview/refresh", response_model=RunProjection)
    async def refresh_preview(run_id: str) -> RunProjection:
        try:
            await resolved_preview_service.refresh_preview(run_id)
            return load_run_projection(resolved_store.get_run_dir(run_id), store=resolved_store)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/diffs/{diff_id}", response_class=PlainTextResponse)
    async def get_diff(run_id: str, diff_id: str) -> str:
        try:
            path = resolved_store.resolve_diff(run_id, diff_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return path.read_text(encoding="utf-8")

    @app.get("/api/runs/{run_id}/artifacts/{artifact_name:path}")
    async def get_artifact(run_id: str, artifact_name: str) -> FileResponse:
        try:
            path = resolved_store.resolve_artifact(run_id, artifact_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(path, media_type=media_type or "application/octet-stream")

    @app.get("/api/runs/{run_id}/status")
    async def get_status(run_id: str) -> dict[str, object]:
        try:
            status = resolved_store.load_status(run_id)
        except FileNotFoundError:
            status = None
        if status is not None:
            return status.model_dump(mode="json")
        hint = resolved_launcher.get_launch_hint(run_id)
        if hint is not None:
            return hint.model_dump(mode="json")
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    @app.post("/api/runs/{run_id}/fork")
    async def fork_run(run_id: str, request: ForkRequest) -> dict[str, object]:
        try:
            response = await resolved_launcher.start_fork_run(
                parent_run_id=run_id,
                checkpoint_sequence=request.checkpoint_sequence,
                instruction_override=request.instruction_override,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return response.model_dump(mode="json")

    return app
