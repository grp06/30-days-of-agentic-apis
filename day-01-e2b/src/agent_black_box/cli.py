from __future__ import annotations

import anyio
import typer
from rich.console import Console
import uvicorn
from pydantic import ValidationError

from .arena_service import ArenaService
from .config import get_settings
from .coordinator import FixtureRunSource, RunCoordinator
from .doctor import run_doctor
from .run_store import RunStore
from .server import create_app
from .timing import build_run_timing_report

app = typer.Typer(help="Agent Black Box CLI")
console = Console()


@app.command("doctor")
def doctor_command() -> None:
    """Validate Ollama and E2B assumptions before a full run."""

    settings = get_settings()
    report = anyio.run(run_doctor, settings)
    console.print("Doctor checks")
    console.print(f"ollama_auth: {'ok' if report.ollama_auth_ok else 'failed'}")
    model_status = "ok" if report.ollama_model_ok else "failed"
    console.print(f"ollama_model: {model_status} ({settings.ollama_model})")
    console.print(f"e2b_auth: {'ok' if report.e2b_auth_ok else 'failed'}")
    console.print(f"e2b_template: {'ok' if report.e2b_template_ok else 'failed'}")
    for note in report.notes:
        console.print(f"note: {note}")
    if not all(
        [
            report.ollama_auth_ok,
            report.ollama_model_ok,
            report.e2b_auth_ok,
            report.e2b_template_ok,
        ]
    ):
        raise typer.Exit(code=1)


@app.command("run-fixture")
def run_fixture(
    fixture: str = typer.Option(
        ..., "--fixture", help="Fixture directory name under fixtures/"
    ),
    task_override: str | None = typer.Option(
        None, "--task-override", help="Override the fixture task."
    ),
) -> None:
    """Run one fixture task through the single-agent harness."""

    settings = get_settings()
    coordinator = RunCoordinator(settings)
    source = FixtureRunSource(fixture_name=fixture, task_override=task_override)
    run_dir = anyio.run(coordinator.run_once, source)
    store = RunStore(settings.run_root)
    summary = store.load_summary(run_dir.name)
    if summary is None:
        raise typer.BadParameter(f"Run summary not found: {run_dir / 'summary.json'}")
    console.print("Run complete")
    console.print(f"artifacts: {run_dir}")
    console.print(summary.model_dump_json(indent=2))


@app.command("show-run")
def show_run(
    run_id: str = typer.Option(..., "--run-id", help="Recorded run identifier."),
) -> None:
    """Print a concise human-readable synopsis for a recorded run."""

    settings = get_settings()
    store = RunStore(settings.run_root)
    try:
        summary = store.load_summary(run_id)
    except (FileNotFoundError, ValidationError) as exc:
        raise typer.BadParameter(
            f"Run summary not found: {settings.run_root / run_id / 'summary.json'}"
        ) from exc
    if summary is None:
        raise typer.BadParameter(
            f"Run summary not found: {settings.run_root / run_id / 'summary.json'}"
        )
    console.print(summary.model_dump_json(indent=2))


@app.command("show-run-timing")
def show_run_timing(
    run_id: str = typer.Option(..., "--run-id", help="Recorded run identifier."),
) -> None:
    """Print model-turn timing from backend event timestamps."""

    settings = get_settings()
    store = RunStore(settings.run_root)
    try:
        report = build_run_timing_report(store, run_id)
    except (FileNotFoundError, ValidationError) as exc:
        raise typer.BadParameter(
            f"Run events not found: {settings.run_root / run_id / 'events.jsonl'}"
        ) from exc
    console.print_json(data=report.model_dump(mode="json"))


@app.command("run-arena")
def run_arena(
    fixture: str = typer.Option(
        ..., "--fixture", help="Fixture directory name under fixtures/"
    ),
    task_override: str = typer.Option(
        ..., "--task-override", help="Visible page brief for all four lanes."
    ),
) -> None:
    """Launch one four-lane arena from the CLI."""

    async def _run() -> object:
        settings = get_settings()
        service = ArenaService(settings)
        response = await service.create_arena(fixture, task_override)
        console.print(f"arena_id: {response.arena_id}")
        for lane in response.lanes:
            console.print(f"{lane.lane_id}: {lane.run_id} ({lane.model_name})")
        while True:
            projection = service.get_arena(response.arena_id)
            if projection.status.state != "running":
                return projection
            await anyio.sleep(1)

    projection = anyio.run(_run)
    console.print("final_status:")
    console.print_json(data=projection.model_dump(mode="json"))


@app.command("show-arena")
def show_arena(
    arena_id: str = typer.Option(..., "--arena-id", help="Recorded arena identifier."),
) -> None:
    """Print a concise synopsis for a recorded arena."""

    settings = get_settings()
    service = ArenaService(settings)
    projection = service.get_arena(arena_id)
    console.print_json(data=projection.model_dump(mode="json"))


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8011, "--port"),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for local development.",
    ),
) -> None:
    """Serve the replay and fork API locally."""

    if reload:
        # Uvicorn reload only works with an import string, not a pre-built app object.
        uvicorn.run(
            "agent_black_box.server:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
        return
    uvicorn.run(create_app(), host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
