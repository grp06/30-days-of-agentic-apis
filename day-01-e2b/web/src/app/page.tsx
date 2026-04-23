import Link from "next/link";

import { ArenaLaunchForm } from "@/components/arena-launch-form";
import { listArenas, listDemoCatalog } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Home() {
  const [arenas, demos] = await Promise.all([listArenas(), listDemoCatalog()]);
  const primaryDemo = demos[0] ?? null;
  const completedArena =
    arenas.find((arena) => arena.completed_lanes === arena.total_lanes) ?? arenas[0] ?? null;
  const primaryArenaId = primaryDemo?.canonical_arena_id ?? completedArena?.arena_id ?? null;
  const primaryAction = primaryDemo?.canonical_arena_id ? "View canonical run" : "View latest run";

  return (
    <main className="landing-shell">
      <section className="landing-hero hero-doorway">
        <div className="hero-copy">
          <span className="eyebrow">Agent Black Box</span>
          <h1>Compare four AI models on the same coding task.</h1>
          <p>
            Run one prompt across isolated E2B sandboxes, then inspect the preview,
            diff, command log, and replay.
          </p>
        </div>
        <div className="hero-launch" aria-label="Start a model comparison">
          <ArenaLaunchForm
            demos={demos}
            secondaryActionHref={primaryArenaId ? `/arenas/${primaryArenaId}` : null}
            secondaryActionLabel={primaryArenaId ? primaryAction : null}
          />
        </div>
      </section>

      <section className="recent-comparisons" id="recent" aria-label="Recent model comparisons">
        <div className="section-header">
          <div>
            <h2>Previous comparisons</h2>
            <p>Reopen an arena, inspect model lanes, or fork from saved checkpoints.</p>
          </div>
          <span className="section-count">
            {arenas.length} {arenas.length === 1 ? "arena" : "arenas"}
          </span>
        </div>
        <div className="arena-activity-list">
          {arenas.slice(0, 6).map((arena) => (
            <Link
              className="arena-activity-row"
              href={`/arenas/${arena.arena_id}`}
              key={arena.arena_id}
            >
              <div className="arena-activity-main">
                <h3>{formatArenaTitle(arena.arena_id)}</h3>
                <span className="muted">{arena.arena_id}</span>
              </div>
              <div className="arena-activity-meta">
                <span className={`pill pill-${arena.state}`}>{arena.state}</span>
                <span>
                  {arena.completed_lanes}/{arena.total_lanes} models
                </span>
              </div>
            </Link>
          ))}
          {arenas.length === 0 ? (
            <div className="empty-state">
              <h2>No comparisons yet</h2>
              <p className="muted">Start the first model comparison above.</p>
            </div>
          ) : null}
        </div>
      </section>
    </main>
  );
}

function formatArenaTitle(arenaId: string): string {
  const match = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z/.exec(arenaId);
  if (!match) {
    return "Model comparison";
  }
  const [, year, month, day, hour, minute] = match;
  return `Comparison ${year}-${month}-${day} ${hour}:${minute} UTC`;
}
