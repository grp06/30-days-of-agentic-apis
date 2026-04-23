export const API_BASE_URL =
  process.env.NEXT_PUBLIC_AGENT_BLACK_BOX_API_URL ?? "http://127.0.0.1:8011";

export interface DemoCatalogEntry {
  demo_id: string;
  title: string;
  summary: string;
  what_to_notice: string;
  fixture_name: string;
  default_task: string;
  canonical_arena_id: string | null;
}

export interface ArenaListItem {
  arena_id: string;
  fixture_name: string;
  state: string;
  total_lanes: number;
  completed_lanes: number;
}

export interface ArenaLaneRecord {
  lane_id: string;
  run_id: string;
  model_name: string | null;
}

export interface ArenaLaneLifecycleStep {
  key: string;
  label: string;
  status: string;
  detail: string | null;
}

export interface ArenaLaneSummary {
  lane_id: string;
  run_id: string;
  state: string;
  model_name: string | null;
  demo_summary: string | null;
  preview_url: string | null;
  preview_state: string;
  preview_last_error: string | null;
  preview_expected: boolean;
  preview_failure_reason: string | null;
  sandbox_retained: boolean;
  preview_attempted: boolean;
  preview_refresh_allowed: boolean;
  checkpoint_count: number;
  child_run_ids: string[];
  failure_reason: string | null;
  started_at: string | null;
  lifecycle_steps: ArenaLaneLifecycleStep[];
  phase_key: string;
  phase_label: string;
  phase_status: string;
  preview_diagnostic: string | null;
  latest_checkpoint_sequence: number | null;
}

export interface ArenaProjection {
  metadata: {
    arena_id: string;
    fixture_name: string;
    task: string;
    started_at: string;
    lanes: ArenaLaneRecord[];
  };
  status: {
    arena_id: string;
    state: string;
    total_lanes: number;
    completed_lanes: number;
    lane_states: Record<string, string>;
    updated_at: string;
  };
  demo_summary: string | null;
  recommended_lane_id: string | null;
  lanes: ArenaLaneSummary[];
}

export interface ArenaLaunchResponse {
  arena_id: string;
  status: string;
  lanes: ArenaLaneRecord[];
}

export interface TimelineCard {
  id: string;
  kind: string;
  sequence: number;
  title: string;
  subtitle: string | null;
  status: string | null;
  artifact_url: string | null;
  detail_ref: string | null;
  detail: Record<string, unknown> | null;
}

export interface CheckpointRef {
  sequence: number;
  snapshot_id: string;
  note: string | null;
}

export interface RunProjection {
  metadata: {
    run_id: string;
    task: string;
    model_name: string;
    fixture_name: string;
    sandbox_id: string | null;
    parent_run_id: string | null;
    source_snapshot_id: string | null;
    source_checkpoint_sequence: number | null;
    instruction_override: string | null;
    started_at: string;
  };
  status: {
    run_id: string;
    state: string;
    current_model_name: string;
    latest_sequence: number;
    preview_url: string | null;
    preview_state: string;
    preview_last_error: string | null;
    preview_expected: boolean;
    preview_failure_reason: string | null;
    sandbox_retained: boolean;
    preview_attempted: boolean;
    preview_refresh_allowed: boolean;
    checkpoint_id: string | null;
    is_fork: boolean;
    updated_at: string;
  };
  summary: {
    run_id: string;
    status: string;
    model_name: string;
    fixture_name: string;
    sandbox_id: string | null;
    preview_url: string | null;
    preview_state: string;
    preview_last_error: string | null;
    preview_expected: boolean;
    preview_failure_reason: string | null;
    sandbox_retained: boolean;
    checkpoint_id: string | null;
    failure_reason: string | null;
    command_count: number;
    diff_count: number;
    tool_call_count: number;
    completed_at: string;
  } | null;
  demo_summary: string | null;
  timeline: TimelineCard[];
  checkpoints: CheckpointRef[];
  children: string[];
}

export async function listDemoCatalog(): Promise<DemoCatalogEntry[]> {
  const response = await fetch(`${API_BASE_URL}/api/demo-catalog`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load demo catalog: ${response.status}`);
  }
  const payload = (await response.json()) as { demos: DemoCatalogEntry[] };
  return payload.demos;
}

export async function listArenas(): Promise<ArenaListItem[]> {
  const response = await fetch(`${API_BASE_URL}/api/arenas`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load arenas: ${response.status}`);
  }
  const payload = (await response.json()) as { arenas: ArenaListItem[] };
  return payload.arenas;
}

export async function getArenaProjection(arenaId: string): Promise<ArenaProjection> {
  const response = await fetch(`${API_BASE_URL}/api/arenas/${arenaId}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load arena ${arenaId}: ${response.status}`);
  }
  return (await response.json()) as ArenaProjection;
}

export async function createArena(
  fixtureName: string,
  taskOverride: string,
): Promise<ArenaLaunchResponse> {
  const response = await fetch(`${API_BASE_URL}/api/arenas`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fixture_name: fixtureName, task_override: taskOverride }),
  });
  if (!response.ok) {
    throw new Error(`Failed to create arena: ${response.status}`);
  }
  return (await response.json()) as ArenaLaunchResponse;
}

export async function getRunProjection(runId: string): Promise<RunProjection> {
  const response = await fetch(`${API_BASE_URL}/api/runs/${runId}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load run ${runId}: ${response.status}`);
  }
  return (await response.json()) as RunProjection;
}

export async function refreshRunPreview(runId: string): Promise<RunProjection> {
  const response = await fetch(`${API_BASE_URL}/api/runs/${runId}/preview/refresh`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(`Failed to refresh preview for ${runId}: ${response.status}`);
  }
  return (await response.json()) as RunProjection;
}

export async function getDiff(runId: string, diffId: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/runs/${runId}/diffs/${diffId}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Failed to load diff ${diffId}: ${response.status}`);
  }
  return response.text();
}
