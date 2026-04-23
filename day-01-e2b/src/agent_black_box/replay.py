from __future__ import annotations
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .events import load_event
from .recorder import RunMetadata, RunStatus, RunSummary
from .run_store import RunStore


class CheckpointRef(BaseModel):
    sequence: int
    snapshot_id: str
    note: str | None = None


class TimelineCard(BaseModel):
    id: str
    kind: str
    sequence: int
    title: str
    subtitle: str | None = None
    status: str | None = None
    artifact_url: str | None = None
    detail_ref: str | None = None
    detail: dict[str, Any] | None = None


class RunProjection(BaseModel):
    metadata: RunMetadata
    status: RunStatus
    summary: RunSummary | None
    demo_summary: str | None = None
    timeline: list[TimelineCard]
    checkpoints: list[CheckpointRef]
    children: list[str]


def load_run_projection(run_dir: Path, *, store: RunStore | None = None) -> RunProjection:
    store = store or RunStore(run_dir.parent)
    run_id = run_dir.name
    projected = store.load_projected_run(run_id)
    if projected is None:
        raise FileNotFoundError(f"Run status not found for {run_id}")
    timeline: list[TimelineCard] = []
    checkpoints: list[CheckpointRef] = []
    checkpoint_lookup = _checkpoint_lookup(store, run_id)
    seen_checkpoint_ids: set[str] = set()
    events = []

    for raw in store.load_event_lines(run_id):
        if not raw.strip():
            continue
        try:
            event = load_event(raw)
        except Exception:  # noqa: BLE001
            continue
        events.append(event)

    completed_commands = {
        event.command for event in events if event.type == "command_completed"
    }

    for event in events:
        card = _event_to_card(event, checkpoint_lookup, completed_commands)
        if card is not None:
            timeline.append(card)
        if event.type == "checkpoint_created":
            checkpoint_ref = checkpoint_lookup.get(
                event.snapshot_id,
                CheckpointRef(
                    sequence=event.sequence,
                    snapshot_id=event.snapshot_id,
                    note=event.note,
                ),
            )
            checkpoints.append(checkpoint_ref)
            seen_checkpoint_ids.add(checkpoint_ref.snapshot_id)

    for checkpoint_ref in checkpoint_lookup.values():
        if checkpoint_ref.snapshot_id in seen_checkpoint_ids:
            continue
        checkpoints.append(checkpoint_ref)
        timeline.append(
            TimelineCard(
                id=f"{checkpoint_ref.sequence}-checkpoint-file",
                kind="checkpoint",
                sequence=checkpoint_ref.sequence,
                title="Checkpoint created",
                subtitle="Snapshot saved",
                status="ok",
                detail={
                    "snapshot_id": checkpoint_ref.snapshot_id,
                    "note": checkpoint_ref.note,
                    "checkpoint_sequence": checkpoint_ref.sequence,
                    "recovered_from_file": True,
                },
            )
        )

    timeline.sort(key=lambda card: card.sequence)
    checkpoints.sort(key=lambda checkpoint: checkpoint.sequence)

    return RunProjection(
        metadata=projected.metadata,
        status=projected.status,
        summary=projected.summary,
        demo_summary=projected.demo_summary,
        timeline=timeline,
        checkpoints=checkpoints,
        children=projected.child_run_ids,
    )

def _event_to_card(
    event,  # type: ignore[no-untyped-def]
    checkpoint_lookup: dict[str, CheckpointRef],
    completed_commands: set[str],
) -> TimelineCard | None:
    if event.type == "tool_call":
        if _tool_command(event) in completed_commands:
            return None
        target = _tool_target(event.tool_name, event.arguments)
        return TimelineCard(
            id=f"{event.sequence}-tool-call",
            kind="tool_call",
            sequence=event.sequence,
            title=_humanize_tool_name(event.tool_name),
            subtitle=target,
            status=None,
            detail={
                "tool_name": event.tool_name,
                "target": target,
                "arguments": _redacted_arguments(event.arguments),
            },
        )
    if event.type == "command_completed":
        return TimelineCard(
            id=f"{event.sequence}-command",
            kind="command",
            sequence=event.sequence,
            title=event.command,
            subtitle=f"exit {event.exit_code} · {'background' if event.background else 'foreground'}",
            status="ok" if event.exit_code == 0 else "error",
            detail={
                "command": event.command,
                "exit_code": event.exit_code,
                "stdout": event.stdout,
                "stderr": event.stderr,
                "background": event.background,
                "pid": event.pid,
            },
        )
    if event.type == "file_diff":
        return TimelineCard(
            id=f"{event.sequence}-diff",
            kind="diff",
            sequence=event.sequence,
            title=event.patch_summary,
            subtitle="Workspace diff",
            detail_ref=Path(event.patch_path).name,
            detail={"patch_path": event.patch_path},
        )
    if event.type == "preview_published":
        return TimelineCard(
            id=f"{event.sequence}-preview",
            kind="preview",
            sequence=event.sequence,
            title="Preview published",
            subtitle=f"port {event.port}",
            status="ok",
            detail={"url": event.url, "port": event.port},
        )
    if event.type == "checkpoint_created":
        checkpoint_ref = checkpoint_lookup.get(
            event.snapshot_id,
            CheckpointRef(
                sequence=event.sequence,
                snapshot_id=event.snapshot_id,
                note=event.note,
            ),
        )
        return TimelineCard(
            id=f"{event.sequence}-checkpoint",
            kind="checkpoint",
            sequence=checkpoint_ref.sequence,
            title="Checkpoint created",
            subtitle="Snapshot saved",
            status="ok",
            detail={
                "snapshot_id": event.snapshot_id,
                "note": event.note,
                "checkpoint_sequence": checkpoint_ref.sequence,
                "event_sequence": event.sequence,
            },
        )
    if event.type == "run_completed":
        return TimelineCard(
            id=f"{event.sequence}-completed",
            kind="run_completed",
            sequence=event.sequence,
            title="Run completed",
            subtitle=event.summary,
            status="ok",
            detail={"summary": event.summary},
        )
    if event.type == "run_failed":
        return TimelineCard(
            id=f"{event.sequence}-failed",
            kind="run_failed",
            sequence=event.sequence,
            title="Run failed",
            subtitle=event.error,
            status="error",
            detail={"error": event.error},
        )
    return None


def _tool_command(event) -> str | None:  # type: ignore[no-untyped-def]
    if event.tool_name != "run_command":
        return None
    command = event.arguments.get("command")
    return command if isinstance(command, str) else None


def _tool_target(tool_name: str, arguments: dict[str, Any]) -> str | None:
    if tool_name == "run_command":
        command = arguments.get("command")
        return _truncate(command) if isinstance(command, str) else None
    if tool_name in {"read_file", "write_file"}:
        path = arguments.get("path")
        return _truncate(path) if isinstance(path, str) else None
    if tool_name == "finish_run":
        summary = arguments.get("summary")
        return _truncate(summary) if isinstance(summary, str) else None
    if tool_name == "apply_patch":
        patch_text = arguments.get("patch_text")
        if isinstance(patch_text, str):
            return f"Patch text · {len(patch_text)} chars"
    return None


def _humanize_tool_name(tool_name: str) -> str:
    labels = {
        "apply_patch": "Apply patch",
        "finish_run": "Finish run",
        "read_file": "Read file",
        "run_command": "Run command",
        "write_file": "Write file",
    }
    return labels.get(tool_name, tool_name.replace("_", " ").title())


def _redacted_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in {"content", "patch_text"} and isinstance(value, str):
            redacted[key] = f"<{len(value)} chars>"
        else:
            redacted[key] = value
    return redacted


def _truncate(value: str, limit: int = 140) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}..."


def _checkpoint_lookup(store: RunStore, run_id: str) -> dict[str, CheckpointRef]:
    lookup: dict[str, CheckpointRef] = {}
    for checkpoint_path in store.list_checkpoints(run_id):
        payload = store.load_checkpoint_payload(checkpoint_path)
        lookup[str(payload["snapshot_id"])] = CheckpointRef(
            sequence=int(checkpoint_path.stem),
            snapshot_id=str(payload["snapshot_id"]),
            note=payload.get("note"),
        )
    return lookup
