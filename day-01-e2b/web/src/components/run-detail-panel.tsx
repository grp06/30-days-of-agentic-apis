"use client";

import type { RunProjection, TimelineCard } from "@/lib/api";

import { DiffViewer } from "./diff-viewer";
import { ForkDialog } from "./fork-dialog";

type RunDetailPanelProps = {
  projection: RunProjection;
  selectedCard: TimelineCard | null;
  diffText: string | null;
};

export function RunDetailPanel({
  projection,
  selectedCard,
  diffText,
}: RunDetailPanelProps) {
  if (!selectedCard) {
    return (
      <section className="detail-panel">
        <div className="detail-header">
          <span className="eyebrow">Replay</span>
          <h2>Select a moment</h2>
          <p>Choose any command, diff, checkpoint, or failure marker from the timeline.</p>
        </div>
        <div className="detail-grid">
          <article className="detail-card">
            <h3>Run summary</h3>
            <p>Status: {projection.status.state}</p>
            <p>Fixture: {projection.metadata.fixture_name}</p>
            <p>Model: {projection.status.current_model_name}</p>
            <p>Last checked: {new Date(projection.status.updated_at).toLocaleString()}</p>
          </article>
        </div>
      </section>
    );
  }

  const detail = selectedCard.detail ?? {};
  const selectedCheckpointSequence =
    typeof detail.checkpoint_sequence === "number"
      ? detail.checkpoint_sequence
      : selectedCard.sequence;

  return (
    <section className="detail-panel">
      <div className="detail-header">
        <span className="eyebrow">
          {formatCardEyebrow(selectedCard)}
        </span>
        <h2>{selectedCard.title}</h2>
        {selectedCard.subtitle ? <p>{selectedCard.subtitle}</p> : null}
      </div>
      <div className="detail-grid">
        <article className="detail-card">
          {renderDetailContent(selectedCard, diffText, projection)}
        </article>
        {selectedCard.kind === "checkpoint" ? (
          <article className="detail-card">
            <h3>Branch from this checkpoint</h3>
            <p className="muted">
              Start a child run from the saved E2B snapshot at this moment.
            </p>
            <ForkDialog
              runId={projection.metadata.run_id}
              checkpointSequence={selectedCheckpointSequence}
            />
          </article>
        ) : null}
      </div>
    </section>
  );
}

function renderDetailContent(
  selectedCard: TimelineCard,
  diffText: string | null,
  projection: RunProjection,
) {
  const detail = selectedCard.detail ?? {};
  if (selectedCard.kind === "diff") {
    return (
      <>
        <h3>Diff</h3>
        <DiffViewer diffText={diffText} />
      </>
    );
  }
  if (selectedCard.kind === "command") {
    return <CommandDetail detail={detail} />;
  }
  if (selectedCard.kind === "tool_call") {
    return <ToolDetail detail={detail} />;
  }
  if (selectedCard.kind === "checkpoint") {
    return <CheckpointDetail detail={detail} />;
  }
  if (selectedCard.kind === "preview") {
    return <PreviewDetail detail={detail} projection={projection} />;
  }
  if (selectedCard.kind === "run_completed") {
    return <MessageDetail heading="Completion" label="Summary" value={readString(detail, "summary")} />;
  }
  if (selectedCard.kind === "run_failed") {
    return <MessageDetail heading="Failure" label="Error" value={readString(detail, "error")} />;
  }
  return (
    <>
      <h3>Event</h3>
      <RawDetails detail={detail} />
    </>
  );
}

function CommandDetail({ detail }: { detail: Record<string, unknown> }) {
  const stdout = readString(detail, "stdout");
  const stderr = readString(detail, "stderr");
  return (
    <>
      <h3>Command</h3>
      <dl className="event-fields">
        <Field label="Command" value={readString(detail, "command")} />
        <Field label="Exit" value={readNumber(detail, "exit_code")?.toString()} />
        <Field label="Mode" value={readBoolean(detail, "background") ? "background" : "foreground"} />
        <Field label="PID" value={readNumber(detail, "pid")?.toString()} />
      </dl>
      <OutputBlock label="stdout" value={stdout} />
      <OutputBlock label="stderr" value={stderr} />
    </>
  );
}

function ToolDetail({ detail }: { detail: Record<string, unknown> }) {
  const argumentsValue = detail.arguments;
  return (
    <>
      <h3>Tool call</h3>
      <dl className="event-fields">
        <Field label="Tool" value={readString(detail, "tool_name")} />
        <Field label="Target" value={readString(detail, "target")} />
      </dl>
      {isRecord(argumentsValue) ? <RawDetails label="Arguments" detail={argumentsValue} /> : null}
    </>
  );
}

function CheckpointDetail({ detail }: { detail: Record<string, unknown> }) {
  return (
    <>
      <h3>Checkpoint</h3>
      <dl className="event-fields">
        <Field label="Snapshot" value={readString(detail, "snapshot_id")} />
        <Field label="Note" value={readString(detail, "note")} />
        <Field label="Checkpoint" value={formatCheckpointNumber(detail)} />
      </dl>
    </>
  );
}

function PreviewDetail({
  detail,
  projection,
}: {
  detail: Record<string, unknown>;
  projection: RunProjection;
}) {
  return (
    <>
      <h3>Preview</h3>
      <dl className="event-fields">
        <Field label="URL" value={readString(detail, "url") ?? projection.status.preview_url} />
        <Field label="Port" value={readNumber(detail, "port")?.toString()} />
        <Field label="State" value={projection.status.preview_state} />
      </dl>
    </>
  );
}

function MessageDetail({
  heading,
  label,
  value,
}: {
  heading: string;
  label: string;
  value: string | null;
}) {
  return (
    <>
      <h3>{heading}</h3>
      <dl className="event-fields">
        <Field label={label} value={value} />
      </dl>
    </>
  );
}

function Field({ label, value }: { label: string; value: string | null | undefined }) {
  if (!value) {
    return null;
  }
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function OutputBlock({ label, value }: { label: string; value: string | null }) {
  if (!value) {
    return null;
  }
  return (
    <details className="output-block">
      <summary>{label}</summary>
      <pre className="detail-pre">{value}</pre>
    </details>
  );
}

function RawDetails({
  detail,
  label = "Raw event",
}: {
  detail: Record<string, unknown>;
  label?: string;
}) {
  return (
    <details className="raw-event">
      <summary>{label}</summary>
      <pre className="detail-pre">{JSON.stringify(detail, null, 2)}</pre>
    </details>
  );
}

function formatKind(kind: string): string {
  const labels: Record<string, string> = {
    command: "Command",
    checkpoint: "Checkpoint",
    diff: "Diff",
    preview: "Preview",
    run_completed: "Run completed",
    run_failed: "Run failed",
    tool_call: "Tool",
  };
  return labels[kind] ?? kind.replaceAll("_", " ");
}

function formatCardEyebrow(card: TimelineCard): string {
  if (card.kind === "checkpoint") {
    return `Checkpoint #${card.sequence}`;
  }
  return `#${card.sequence.toString().padStart(2, "0")} · ${formatKind(card.kind)}`;
}

function formatCheckpointNumber(detail: Record<string, unknown>): string | null {
  const sequence = readNumber(detail, "checkpoint_sequence");
  return sequence === null ? null : `#${sequence}`;
}

function readString(detail: Record<string, unknown>, key: string): string | null {
  const value = detail[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function readNumber(detail: Record<string, unknown>, key: string): number | null {
  const value = detail[key];
  return typeof value === "number" ? value : null;
}

function readBoolean(detail: Record<string, unknown>, key: string): boolean {
  return detail[key] === true;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
