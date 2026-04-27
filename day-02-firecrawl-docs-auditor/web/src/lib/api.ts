export const API_BASE_URL =
  process.env.NEXT_PUBLIC_FIRECRAWL_DOCS_AUDITOR_API_URL ?? "http://127.0.0.1:8122";

export interface ServiceStatus {
  status: string;
  codex_bin_configured?: boolean;
  codex_bin_detected?: string | null;
  requires_openai_auth?: boolean;
  account?: CodexAccount | null;
  message?: string | null;
}

export interface AuditorStatus {
  service: string;
  api: {
    host: string;
    port: number;
  };
  frontend: {
    allowed_origins: string[];
  };
  contracts: {
    fixtures_readable: boolean;
  };
  codex_app_server: ServiceStatus;
  firecrawl: ServiceStatus;
  audit_engine: ServiceStatus;
}

export interface AuditRequest {
  docs_url: string;
  integration_goal: string;
  mode: "live" | "cached";
  max_pages?: number;
  max_depth?: number;
  allowed_hosts?: string[];
  cache_key?: string;
}

export interface FirecrawlServiceStatus {
  status: "missing_key" | "configured";
  configured: boolean;
  message: string;
}

export interface FirecrawlKeyStatus {
  status:
    | "missing"
    | "configured"
    | "valid"
    | "invalid"
    | "rate_limited"
    | "payment_required"
    | "error";
  configured: boolean;
  message: string;
}

export interface FirecrawlPreflightRequest {
  docs_url: string;
  integration_goal: string;
  max_pages: number;
  max_depth: number;
  allowed_hosts?: string[];
  firecrawl_api_key?: string;
}

export interface PreflightCheck {
  id: string;
  status: "pass" | "warning" | "blocked" | "skipped";
  severity: "info" | "warning" | "blocking";
  message: string;
  url?: string | null;
}

export interface DocsPreflightResult {
  verdict: "pass" | "warning" | "blocked";
  normalized_url?: string | null;
  allowed_hosts: string[];
  key_status: FirecrawlKeyStatus;
  checks: PreflightCheck[];
}

export interface FirecrawlFetchRequest {
  docs_url: string;
  integration_goal: string;
  max_pages: number;
  max_depth: number;
  allowed_hosts?: string[];
  firecrawl_api_key?: string;
  cache_key?: string;
}

export interface SelectedSource {
  id: string;
  url: string;
  title: string;
  reason_selected: string;
  retrieved_via: "firecrawl_map" | "firecrawl_scrape" | "llms_txt" | "sitemap" | "cached_fixture";
  markdown_chars: number;
}

export interface FirecrawlFetchResult {
  status: "completed" | "completed_with_warnings" | "blocked";
  cache_key: string;
  preflight: DocsPreflightResult;
  selected_sources: SelectedSource[];
  candidate_count: number;
  warnings: string[];
  artifact_path?: string | null;
}

export interface Source {
  id: string;
  url: string;
  title: string;
  reason_selected: string;
  retrieved_via: string;
}

export interface ScorecardDimension {
  id: string;
  label: string;
  score: number;
  max_score: number;
  rationale: string;
  source_refs: string[];
}

export interface EvidenceItem {
  id: string;
  message: string;
  basis: "source_backed" | "inferred" | "uncertain";
  source_refs?: string[];
  severity?: "info" | "warning" | "blocking";
}

export interface SmokeTest {
  result: "pass" | "partial" | "fail";
  basis: "source_backed" | "inferred" | "uncertain";
  message: string;
  source_refs?: string[];
  missing_facts?: string[];
  likely_next_steps?: string[];
}

export interface AuditReport {
  request: AuditRequest;
  status: "completed" | "completed_with_warnings" | "blocked";
  summary: string;
  scorecard: ScorecardDimension[];
  selected_sources: Source[];
  extracted_facts: EvidenceItem[];
  smoke_test: SmokeTest;
  warnings: EvidenceItem[];
  suggested_fixes: EvidenceItem[];
  metadata: {
    mode: "live" | "cached";
    generated_by: string;
    cache_key?: string;
    notes?: string;
    fetched_source_count?: number;
    prompt_source_count?: number;
    omitted_source_count?: number;
    represented_goal_terms?: string[];
    missing_goal_terms?: string[];
    supported_claims?: string[];
    missing_claims?: string[];
    required_claims_supported?: number;
    required_claims_total?: number;
    optional_claims_supported?: number;
    optional_claims_total?: number;
    missing_required_claims?: string[];
    missing_optional_claims?: string[];
    prompt_note?: string;
  };
}

export interface AuditReportRequest extends AuditRequest {
  cache_key: string;
}

export interface AuditReportResult {
  status: "completed" | "completed_with_warnings" | "blocked";
  report?: AuditReport | null;
  cache_key: string;
  warnings: string[];
  artifact_path?: string | null;
}

export interface CodexAccount {
  type: string;
  email?: string | null;
  plan_type?: string | null;
}

export interface CodexAccountStatus {
  status: "available_signed_in" | "available_login_required" | "unavailable" | "error";
  requires_openai_auth: boolean;
  account?: CodexAccount | null;
  codex_bin_configured: boolean;
  codex_bin_detected?: string | null;
  message?: string | null;
}

export interface CodexLoginStartResult {
  status: "started" | "unavailable" | "error";
  mode: "browser" | "device_code";
  login_id?: string | null;
  auth_url?: string | null;
  verification_url?: string | null;
  user_code?: string | null;
  message?: string | null;
}

export interface CodexLoginStatus {
  status: "pending" | "succeeded" | "failed" | "canceled" | "unknown";
  login_id: string;
  message?: string | null;
}

export interface CodexCancelLoginResult {
  status: "canceled" | "not_found" | "error";
  login_id: string;
  message?: string | null;
}

export interface CodexSmokeTestResult {
  status: "pass" | "login_required" | "unavailable" | "error";
  message: string;
  response?: Record<string, unknown> | null;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, { cache: "no-store", ...init });
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status}`);
  }
  return (await response.json()) as T;
}

async function postJson<T>(path: string, body?: object): Promise<T> {
  return fetchJson<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
}

export async function getStatus(): Promise<AuditorStatus> {
  return fetchJson<AuditorStatus>("/api/status");
}

export async function getFirecrawlStatus(): Promise<FirecrawlServiceStatus> {
  return fetchJson<FirecrawlServiceStatus>("/api/firecrawl/status");
}

export async function runFirecrawlPreflight(
  request: FirecrawlPreflightRequest,
): Promise<DocsPreflightResult> {
  return postJson<DocsPreflightResult>("/api/firecrawl/preflight", request);
}

export async function runFirecrawlFetchSources(
  request: FirecrawlFetchRequest,
): Promise<FirecrawlFetchResult> {
  return postJson<FirecrawlFetchResult>("/api/firecrawl/fetch-sources", request);
}

export async function getCachedFirecrawlFetchSources(
  cacheKey: string,
): Promise<FirecrawlFetchResult> {
  return fetchJson<FirecrawlFetchResult>(
    `/api/firecrawl/fetch-sources/${encodeURIComponent(cacheKey)}`,
  );
}

export async function generateAuditReport(
  request: AuditReportRequest,
): Promise<AuditReportResult> {
  return postJson<AuditReportResult>("/api/audit/report", request);
}

export async function getCodexAccount(): Promise<CodexAccountStatus> {
  return fetchJson<CodexAccountStatus>("/api/codex/account");
}

export async function startCodexLogin(mode: "browser" | "device_code"): Promise<CodexLoginStartResult> {
  return postJson<CodexLoginStartResult>("/api/codex/login/start", { mode });
}

export async function getCodexLoginStatus(loginId: string): Promise<CodexLoginStatus> {
  return fetchJson<CodexLoginStatus>(`/api/codex/login/status/${encodeURIComponent(loginId)}`);
}

export async function cancelCodexLogin(loginId: string): Promise<CodexCancelLoginResult> {
  return postJson<CodexCancelLoginResult>("/api/codex/login/cancel", { login_id: loginId });
}

export async function logoutCodex(): Promise<CodexAccountStatus> {
  return postJson<CodexAccountStatus>("/api/codex/logout");
}

export async function runCodexSmokeTest(): Promise<CodexSmokeTestResult> {
  return postJson<CodexSmokeTestResult>("/api/codex/smoke-test");
}
