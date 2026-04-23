"use client";

import type { TimelineCard } from "@/lib/api";

type RunTimelineProps = {
  cards: TimelineCard[];
  selectedCardId: string | null;
  onSelect: (cardId: string) => void;
};

export function RunTimeline({
  cards,
  selectedCardId,
  onSelect,
}: RunTimelineProps) {
  return (
    <div className="timeline">
      {cards.map((card) => {
        const selected = card.id === selectedCardId;
        const statusClass = card.status ? ` status-${card.status}` : "";
        return (
          <button
            key={card.id}
            className={`timeline-card${selected ? " is-selected" : ""}${statusClass}`}
            onClick={() => onSelect(card.id)}
            type="button"
          >
            <span className="timeline-sequence">{formatSequence(card)}</span>
            <span className="timeline-kind">{formatKind(card.kind)}</span>
            <span className="timeline-title">{card.title}</span>
            {card.subtitle ? <span className="timeline-subtitle">{card.subtitle}</span> : null}
            {card.kind === "checkpoint" ? <span className="timeline-badge">forkable</span> : null}
          </button>
        );
      })}
    </div>
  );
}

function formatSequence(card: TimelineCard): string {
  if (card.kind === "checkpoint") {
    return `#${card.sequence}`;
  }
  return `#${card.sequence.toString().padStart(2, "0")}`;
}

function formatKind(kind: string): string {
  const labels: Record<string, string> = {
    command: "Command",
    checkpoint: "Checkpoint",
    diff: "Diff",
    preview: "Preview",
    run_completed: "Done",
    run_failed: "Failed",
    tool_call: "Tool",
  };
  return labels[kind] ?? kind.replaceAll("_", " ");
}
