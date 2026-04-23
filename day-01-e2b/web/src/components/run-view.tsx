"use client";

import { API_BASE_URL, getDiff, refreshRunPreview, type RunProjection } from "@/lib/api";
import { useEffect, useMemo, useRef, useState } from "react";

import { RunDetailPanel } from "./run-detail-panel";
import { RunTimeline } from "./run-timeline";

type RunViewProps = {
  projection: RunProjection;
  initialCheckpointSequence?: number | null;
};

export function RunView({ projection, initialCheckpointSequence = null }: RunViewProps) {
  const [currentProjection, setCurrentProjection] = useState(projection);
  const [selectedCardId, setSelectedCardId] = useState<string | null>(
    selectInitialCardId(projection, initialCheckpointSequence),
  );
  const [diffText, setDiffText] = useState<string | null>(null);
  const previewRefreshAttempts = useRef<Set<string>>(new Set());

  const selectedCard = useMemo(
    () => currentProjection.timeline.find((card) => card.id === selectedCardId) ?? null,
    [currentProjection.timeline, selectedCardId],
  );
  function summarizePreviewIssue(message: string | null): string | null {
    if (!message) {
      return null;
    }
    const lowered = message.toLowerCase();
    if (lowered.includes("context deadline exceeded") || lowered.includes("timeout")) {
      return "Preview launch timed out before publication.";
    }
    if (lowered.includes("not found")) {
      return "The preserved sandbox is no longer available.";
    }
    if (lowered.includes("server is not running")) {
      return "The sandbox is still available, but the preview server is not running.";
    }
    return message;
  }

  function renderPreviewStatus() {
    const {
      preview_state: previewState,
      preview_url: previewUrl,
      preview_last_error: previewLastError,
      preview_expected: previewExpected,
      preview_failure_reason: previewFailureReason,
      preview_attempted: previewAttempted,
    } =
      currentProjection.status;
    const previewIssueSummary =
      summarizePreviewIssue(previewFailureReason) ?? summarizePreviewIssue(previewLastError);
    if (currentProjection.status.state === "running" && !previewUrl) {
      return (
        <div className="preview-note">
          <p className="muted">
            {previewExpected
              ? previewAttempted
                ? "Preview is still being prepared. This run is active and will update when the next preview event arrives."
                : "Preview is pending. The agent is still working; the preview will appear after build and publication."
              : "This run is active. A preview may appear if the agent starts a browser-visible app."}
          </p>
          {previewAttempted && previewIssueSummary ? (
            <p className="muted">{previewIssueSummary}</p>
          ) : null}
        </div>
      );
    }
    if (previewState === "live" && previewUrl) {
      return (
        <>
          <a className="preview-link" href={previewUrl} target="_blank" rel="noreferrer">
            Open live preview
          </a>
          <p className="muted">Preview verified against the current sandbox state.</p>
        </>
      );
    }
    if (previewState === "retained" && previewUrl) {
      return (
        <div className="preview-note">
          <a className="preview-link" href={previewUrl} target="_blank" rel="noreferrer">
            Open last known preview
          </a>
          <p className="muted">
            This run produced a preview. The app checks whether retained previews are still live when the replay opens.
          </p>
        </div>
      );
    }
    if (previewState === "server_not_running") {
      return (
        <div className="preview-note">
          <p className="muted">The sandbox still exists, but the preview server is not running.</p>
          {previewIssueSummary ? <p className="muted">{previewIssueSummary}</p> : null}
        </div>
      );
    }
    if (previewState === "expired") {
      return (
        <div className="preview-note">
          {previewUrl ? (
            <a className="preview-link" href={previewUrl} target="_blank" rel="noreferrer">
              Open last known preview
            </a>
          ) : null}
          <p className="muted">
            The preserved sandbox is no longer available.
            {previewUrl ? " This was the last known preview URL for the run." : ""}
          </p>
          {previewIssueSummary ? <p className="muted">{previewIssueSummary}</p> : null}
        </div>
      );
    }
    return (
      <div className="preview-note">
        <p className="muted">
          {previewExpected
            ? currentProjection.status.sandbox_retained
              ? "This run was expected to publish a preview, but the first attempt failed before publication. The app will check retained sandboxes automatically when possible."
              : "This run was expected to publish a preview, but none was captured."
            : previewAttempted
              ? currentProjection.status.sandbox_retained
                ? "A preview was attempted for this run, but the first attempt failed before publication. The app will check retained sandboxes automatically when possible."
                : "A preview was attempted for this run, but none was captured."
            : currentProjection.status.sandbox_retained
              ? "This run finished without a recorded preview, but the sandbox was preserved. The app will check whether it is serving when possible."
              : "No preview was expected for this run."}
        </p>
        {previewIssueSummary ? <p className="muted">{previewIssueSummary}</p> : null}
      </div>
    );
  }

  useEffect(() => {
    if (currentProjection.status.state !== "running") {
      return;
    }
    const intervalId = window.setInterval(async () => {
      const response = await fetch(
        `${API_BASE_URL}/api/runs/${currentProjection.metadata.run_id}`,
        { cache: "no-store" },
      );
      if (!response.ok) {
        return;
      }
      const nextProjection = (await response.json()) as RunProjection;
      setCurrentProjection(nextProjection);
      setSelectedCardId((currentSelected) => {
        if (
          currentSelected &&
          nextProjection.timeline.some((card) => card.id === currentSelected)
        ) {
          return currentSelected;
        }
        return selectInitialCardId(nextProjection);
      });
    }, 4000);
    return () => window.clearInterval(intervalId);
  }, [currentProjection.metadata.run_id, currentProjection.status.state]);

  useEffect(() => {
    const runId = currentProjection.metadata.run_id;
    if (
      currentProjection.status.state === "running" ||
      !currentProjection.status.preview_refresh_allowed ||
      previewRefreshAttempts.current.has(runId)
    ) {
      return;
    }
    previewRefreshAttempts.current.add(runId);
    let cancelled = false;
    async function refreshPreview() {
      try {
        const nextProjection = await refreshRunPreview(runId);
        if (!cancelled) {
          setCurrentProjection(nextProjection);
        }
      } catch {
        // Keep the recorded preview state if the liveness probe itself fails.
      }
    }
    void refreshPreview();
    return () => {
      cancelled = true;
    };
  }, [
    currentProjection.metadata.run_id,
    currentProjection.status.preview_refresh_allowed,
    currentProjection.status.state,
  ]);

  useEffect(() => {
    let cancelled = false;
    async function loadDiff() {
      if (!selectedCard?.detail_ref || selectedCard.kind !== "diff") {
        setDiffText(null);
        return;
      }
      const nextDiff = await getDiff(currentProjection.metadata.run_id, selectedCard.detail_ref);
      if (!cancelled) {
        setDiffText(nextDiff);
      }
    }
    void loadDiff();
    return () => {
      cancelled = true;
    };
  }, [currentProjection.metadata.run_id, selectedCard]);

  return (
    <div className="run-shell">
      <section className="run-sidebar">
        <div className="run-summary-card">
          <span className="eyebrow">Single-lane replay</span>
          <h1>Replay</h1>
          <p>
            {currentProjection.status.current_model_name} · {currentProjection.status.state}
          </p>
          {currentProjection.demo_summary ? <p>{currentProjection.demo_summary}</p> : null}
          <dl className="meta-list">
            <div>
              <dt>Fixture</dt>
              <dd>{currentProjection.metadata.fixture_name}</dd>
            </div>
            <div>
              <dt>Checkpoints</dt>
              <dd>{currentProjection.checkpoints.length}</dd>
            </div>
            <div>
              <dt>Forks</dt>
              <dd>{currentProjection.children.length}</dd>
            </div>
            <div>
              <dt>Preview</dt>
              <dd>{currentProjection.status.preview_expected ? "expected" : "optional"}</dd>
            </div>
          </dl>
          {renderPreviewStatus()}
        </div>
        <RunTimeline
          cards={currentProjection.timeline}
          selectedCardId={selectedCardId}
          onSelect={setSelectedCardId}
        />
      </section>
      <RunDetailPanel projection={currentProjection} selectedCard={selectedCard} diffText={diffText} />
    </div>
  );
}

function selectInitialCardId(
  projection: RunProjection,
  checkpointSequence: number | null = null,
): string | null {
  if (checkpointSequence !== null) {
    const checkpointCard = projection.timeline.find((card) => {
      if (card.kind !== "checkpoint") {
        return false;
      }
      const sequence = card.detail?.checkpoint_sequence;
      return sequence === checkpointSequence || card.sequence === checkpointSequence;
    });
    if (checkpointCard) {
      return checkpointCard.id;
    }
  }
  const preferredKinds = ["diff", "preview", "checkpoint", "run_failed", "run_completed"];
  for (const kind of preferredKinds) {
    const match = projection.timeline.find((card) => card.kind === kind);
    if (match) {
      return match.id;
    }
  }
  return projection.timeline[0]?.id ?? null;
}
