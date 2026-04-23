"use client";

import { API_BASE_URL } from "@/lib/api";
import { useRouter } from "next/navigation";
import { useState } from "react";

type ForkDialogProps = {
  runId: string;
  checkpointSequence: number;
};

export function ForkDialog({ runId, checkpointSequence }: ForkDialogProps) {
  const router = useRouter();
  const [instruction, setInstruction] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submitFork() {
    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/runs/${runId}/fork`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          checkpoint_sequence: checkpointSequence,
          instruction_override: instruction,
        }),
      });
      if (!response.ok) {
        throw new Error(`Fork failed: ${response.status}`);
      }
      const payload = (await response.json()) as { run_id: string };
      router.push(`/runs/${payload.run_id}`);
      router.refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to fork run.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fork-panel">
      <label className="fork-label" htmlFor={`fork-${checkpointSequence}`}>
        Fork from this checkpoint
      </label>
      <textarea
        id={`fork-${checkpointSequence}`}
        className="fork-textarea"
        value={instruction}
        onChange={(event) => setInstruction(event.target.value)}
        placeholder="Keep the working fix, but restyle the hero to feel more premium and editorial."
      />
      {error ? <p className="error-text">{error}</p> : null}
      <button
        className="fork-button"
        disabled={submitting || instruction.trim().length === 0}
        onClick={submitFork}
        type="button"
      >
        {submitting ? "Starting fork..." : "Start fork"}
      </button>
    </div>
  );
}
