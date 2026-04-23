"use client";

import type { ArenaLaneSummary, ArenaProjection } from "@/lib/api";
import { API_BASE_URL } from "@/lib/api";
import Link from "next/link";
import { useEffect, useState } from "react";

type ArenaViewProps = {
  projection: ArenaProjection;
};

export function ArenaView({ projection }: ArenaViewProps) {
  const [currentProjection, setCurrentProjection] = useState(projection);
  const previewCount = currentProjection.lanes.filter(hasOpenablePreview).length;
  const checkpointCount = currentProjection.lanes.reduce(
    (total, lane) => total + lane.checkpoint_count,
    0,
  );
  const forkCount = currentProjection.lanes.reduce(
    (total, lane) => total + lane.child_run_ids.length,
    0,
  );

  useEffect(() => {
    if (currentProjection.status.state !== "running") {
      return;
    }
    const intervalId = window.setInterval(async () => {
      const response = await fetch(
        `${API_BASE_URL}/api/arenas/${currentProjection.metadata.arena_id}`,
        { cache: "no-store" },
      );
      if (!response.ok) {
        return;
      }
      const nextProjection = (await response.json()) as ArenaProjection;
      setCurrentProjection(nextProjection);
    }, 4000);
    return () => window.clearInterval(intervalId);
  }, [currentProjection.metadata.arena_id, currentProjection.status.state]);

  return (
    <section className="arena-shell">
      <div className="arena-overview">
        <div className="arena-task-block">
          <span className="eyebrow">Task</span>
          <p>{currentProjection.metadata.task}</p>
        </div>
        <dl className="arena-overview-stats" aria-label="Comparison summary">
          <div>
            <dt>Models</dt>
            <dd>
              {currentProjection.status.completed_lanes}/{currentProjection.status.total_lanes}
            </dd>
          </div>
          <div>
            <dt>Previews</dt>
            <dd>{previewCount}</dd>
          </div>
          <div>
            <dt>Checkpoints</dt>
            <dd>{checkpointCount}</dd>
          </div>
          <div>
            <dt>Forks</dt>
            <dd>{forkCount}</dd>
          </div>
        </dl>
      </div>

      <div className="arena-grid">
        {currentProjection.lanes.map((lane) => {
          const previewReady = hasOpenablePreview(lane);
          return (
            <article className="run-list-card arena-lane-card" key={lane.run_id}>
              <div className="lane-card-header">
                <div className="lane-title-block">
                  <h2>{formatModelName(lane.model_name)}</h2>
                  <span className="muted">{lane.lane_id}</span>
                </div>
                <span className={`pill pill-${lane.state}`}>{formatStateLabel(lane.state)}</span>
              </div>
              <p className={`lane-phase is-${lane.phase_status}`}>
                {formatLaneOutcome(lane)}
              </p>
              <div className="lane-actions" aria-label={`${lane.lane_id} actions`}>
                {previewReady ? (
                  <a className="preview-link" href={lane.preview_url ?? undefined} target="_blank" rel="noreferrer">
                    {lane.preview_state === "live" ? "Open preview" : "Open last preview"}
                  </a>
                ) : null}
                <Link className="secondary-button" href={`/runs/${lane.run_id}`}>
                  Inspect replay
                </Link>
                {lane.latest_checkpoint_sequence !== null ? (
                  <Link
                    className="secondary-button"
                    href={`/runs/${lane.run_id}?checkpoint=${lane.latest_checkpoint_sequence}`}
                  >
                    Branch from checkpoint
                  </Link>
                ) : null}
              </div>
              <dl className="lane-evidence-strip" aria-label={`${lane.lane_id} evidence summary`}>
                <div>
                  <dt>Build</dt>
                  <dd>{summarizeStep(lane, "build")}</dd>
                </div>
                <div>
                  <dt>Preview</dt>
                  <dd>{previewReady ? "Ready" : summarizeStep(lane, "preview")}</dd>
                </div>
                <div>
                  <dt>Checkpoints</dt>
                  <dd>{lane.checkpoint_count}</dd>
                </div>
                <div>
                  <dt>Forks</dt>
                  <dd>{lane.child_run_ids.length}</dd>
                </div>
              </dl>
              <div className="lane-evidence-details">
                <p className="lane-evidence-heading">Evidence details</p>
                <ol className="lane-lifecycle" aria-label={`${lane.lane_id} E2B lifecycle`}>
                  {lane.lifecycle_steps.map((step) => (
                    <li className={`lane-lifecycle-step is-${step.status}`} key={step.key}>
                      <span className="lane-step-marker" aria-hidden="true" />
                      <div>
                        <span>{step.label}</span>
                        {step.detail ? <small>{step.detail}</small> : null}
                      </div>
                    </li>
                  ))}
                </ol>
                <p className="lane-run-id">Run {lane.run_id}</p>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function hasOpenablePreview(lane: ArenaLaneSummary): boolean {
  return Boolean(
    lane.preview_url && (lane.preview_state === "live" || lane.preview_state === "retained"),
  );
}

function formatLaneOutcome(lane: ArenaLaneSummary): string {
  if (lane.state === "succeeded" && hasOpenablePreview(lane)) {
    return joinSummaryParts([
      "Preview ready",
      checkpointLabel(lane.checkpoint_count),
      forkLabel(lane.child_run_ids.length),
    ]);
  }
  if (lane.state === "succeeded") {
    return joinSummaryParts(["Completed", checkpointLabel(lane.checkpoint_count)]);
  }
  if (lane.failure_reason) {
    return lane.failure_reason;
  }
  return lane.phase_label;
}

function formatModelName(modelName: string | null): string {
  if (modelName === null) {
    return "Model unavailable";
  }
  return modelName.replace(":cloud", "").replace(":397b", " 397B").replace(":31b", " 31B");
}

function formatStateLabel(state: string): string {
  return state.replaceAll("_", " ");
}

function checkpointLabel(count: number): string | null {
  if (count === 0) {
    return null;
  }
  return `${count} checkpoint${count === 1 ? "" : "s"}`;
}

function forkLabel(count: number): string | null {
  if (count === 0) {
    return null;
  }
  return `${count} fork${count === 1 ? "" : "s"}`;
}

function joinSummaryParts(parts: Array<string | null>): string {
  return parts.filter(Boolean).join(" · ");
}

function summarizeStep(lane: ArenaLaneSummary, key: string): string {
  const step = lane.lifecycle_steps.find((candidate) => candidate.key === key);
  if (!step) {
    return "Not reached";
  }
  if (step.status === "ok") {
    return "Passed";
  }
  if (step.status === "active") {
    return "Running";
  }
  if (step.status === "error") {
    return "Failed";
  }
  if (step.status === "warning") {
    return "Needs review";
  }
  return "Not reached";
}
