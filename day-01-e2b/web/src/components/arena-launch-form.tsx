"use client";

import type { DemoCatalogEntry } from "@/lib/api";
import { createArena } from "@/lib/api";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

type ArenaLaunchFormProps = {
  demos: DemoCatalogEntry[];
  secondaryActionHref?: string | null;
  secondaryActionLabel?: string | null;
};

export function ArenaLaunchForm({
  demos,
  secondaryActionHref = null,
  secondaryActionLabel = null,
}: ArenaLaunchFormProps) {
  const router = useRouter();
  const selectedDemo = demos[0] ?? null;
  const [task, setTask] = useState(selectedDemo?.default_task ?? "");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedDemo) {
      return;
    }
    setIsSubmitting(true);
    setError(null);
    try {
      const response = await createArena(selectedDemo.fixture_name, task);
      router.push(`/arenas/${response.arena_id}`);
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Failed to launch arena");
      setIsSubmitting(false);
    }
  }

  return (
    <form className="launch-panel" onSubmit={handleSubmit}>
      <div className="launch-panel-header">
        <div className="launch-panel-copy">
          <h2>Enter one coding task</h2>
          <p>Kimi K2.6 · GLM 5.1 · Gemma 4 · Qwen 3.5</p>
        </div>
        {secondaryActionHref && secondaryActionLabel ? (
          <Link className="secondary-action launch-panel-link" href={secondaryActionHref}>
            {secondaryActionLabel}
          </Link>
        ) : null}
      </div>
      <div className="prompt-block">
        <label className="fork-label" htmlFor="arena-task">
          Coding task
        </label>
        <textarea
          id="arena-task"
          className="fork-textarea"
          rows={6}
          value={task}
          onChange={(event) => setTask(event.target.value)}
        />
      </div>
      {error ? <p className="error-text">{error}</p> : null}
      <button
        className="fork-button"
        type="submit"
        disabled={isSubmitting || task.trim().length === 0 || selectedDemo === null}
      >
        {isSubmitting ? "Starting comparison..." : "Run comparison"}
      </button>
    </form>
  );
}
