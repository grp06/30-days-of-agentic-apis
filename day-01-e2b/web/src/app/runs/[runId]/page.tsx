import Link from "next/link";

import { RunView } from "@/components/run-view";
import { getRunProjection } from "@/lib/api";

export const dynamic = "force-dynamic";

type RunPageProps = {
  params: Promise<{ runId: string }>;
  searchParams: Promise<{ checkpoint?: string }>;
};

export default async function RunPage({ params, searchParams }: RunPageProps) {
  const { runId } = await params;
  const { checkpoint } = await searchParams;
  const projection = await getRunProjection(runId);
  const initialCheckpointSequence = resolveCheckpointSequence(checkpoint, projection);

  return (
    <main className="run-page">
      <div className="run-page-header">
        <Link className="back-link" href="/">
          Back to arenas
        </Link>
        <div>
          <span className="eyebrow">Single-lane replay</span>
          <h1>Inspect one model run.</h1>
          <p>
            Review the commands, diffs, preview events, and forkable E2B checkpoints
            behind this lane.
          </p>
          <p className="run-page-meta">
            {projection.status.current_model_name} · {projection.status.state} ·{" "}
            {projection.checkpoints.length}{" "}
            {projection.checkpoints.length === 1 ? "checkpoint" : "checkpoints"}
          </p>
        </div>
      </div>
      <RunView
        projection={projection}
        initialCheckpointSequence={initialCheckpointSequence}
      />
    </main>
  );
}

function resolveCheckpointSequence(
  checkpoint: string | undefined,
  projection: Awaited<ReturnType<typeof getRunProjection>>,
): number | null {
  if (!checkpoint) {
    return null;
  }
  if (checkpoint === "latest") {
    return projection.checkpoints.at(-1)?.sequence ?? null;
  }
  const parsed = Number.parseInt(checkpoint, 10);
  return Number.isFinite(parsed) ? parsed : null;
}
