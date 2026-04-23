import Link from "next/link";

import { ArenaView } from "@/components/arena-view";
import { getArenaProjection } from "@/lib/api";

export const dynamic = "force-dynamic";

type ArenaPageProps = {
  params: Promise<{ arenaId: string }>;
};

export default async function ArenaPage({ params }: ArenaPageProps) {
  const { arenaId } = await params;
  const projection = await getArenaProjection(arenaId);
  const previewCount = projection.lanes.filter(
    (lane) => lane.preview_url && (lane.preview_state === "live" || lane.preview_state === "retained"),
  ).length;

  return (
    <main className="run-page">
      <div className="run-page-header">
        <Link className="back-link" href="/">
          Back to comparisons
        </Link>
        <div>
          <span className="eyebrow">Model comparison</span>
          <h1>{formatArenaHeadline(projection.status.completed_lanes, projection.status.total_lanes)}</h1>
          <p>
            {projection.demo_summary ??
              "One task ran across four isolated E2B sandboxes. Open a preview, inspect the replay, or continue from a checkpoint."}
          </p>
          <p className="run-page-meta">
            Run {projection.metadata.arena_id} · {previewCount}{" "}
            {previewCount === 1 ? "preview" : "previews"} · Updated{" "}
            {new Date(projection.status.updated_at).toLocaleString()}
          </p>
        </div>
      </div>
      <ArenaView projection={projection} />
    </main>
  );
}

function formatArenaHeadline(completed: number, total: number): string {
  if (total === 0) {
    return "No model runs yet.";
  }
  if (completed === total) {
    return `All ${total} models completed.`;
  }
  return `${completed} of ${total} models completed.`;
}
