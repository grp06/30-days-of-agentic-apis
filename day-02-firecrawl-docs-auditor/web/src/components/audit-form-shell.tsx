"use client";

import { useState } from "react";

import type {
  AuditReport,
  AuditRequest,
  AuditReportResult,
  DocsPreflightResult,
  FirecrawlFetchResult,
} from "@/lib/api";
import { generateAuditReport, runFirecrawlFetchSources, runFirecrawlPreflight } from "@/lib/api";

interface AuditFormShellProps {
  request: AuditRequest | null;
  serverFirecrawlConfigured?: boolean;
  onInputChanged?: () => void;
  onPreflightChanged?: (result: DocsPreflightResult | null) => void;
  onSourceFetchChanged?: (result: FirecrawlFetchResult | null) => void;
  onReportResultChanged?: (result: AuditReportResult | null) => void;
  onReportGenerated?: (report: AuditReport | null, artifactPath: string | null) => void;
  onReportError?: (message: string | null) => void;
}

export function AuditFormShell({
  request,
  serverFirecrawlConfigured = false,
  onInputChanged,
  onPreflightChanged,
  onSourceFetchChanged,
  onReportResultChanged,
  onReportGenerated,
  onReportError,
}: AuditFormShellProps) {
  const [firecrawlKey, setFirecrawlKey] = useState("");
  const [docsUrl, setDocsUrl] = useState(request?.docs_url ?? "");
  const [integrationGoal, setIntegrationGoal] = useState(
    request?.integration_goal ?? "",
  );
  const [maxPages, setMaxPages] = useState(request?.max_pages ?? 20);
  const [maxDepth, setMaxDepth] = useState(request?.max_depth ?? 1);
  const [result, setResult] = useState<DocsPreflightResult | null>(null);
  const [fetchResult, setFetchResult] = useState<FirecrawlFetchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [reportError, setReportError] = useState<string | null>(null);
  const [isPreflightRunning, setIsPreflightRunning] = useState(false);
  const [isFetchRunning, setIsFetchRunning] = useState(false);
  const [isReportRunning, setIsReportRunning] = useState(false);
  const [analysisState, setAnalysisState] = useState<AnalysisState>("idle");

  function updateFirecrawlKey(value: string) {
    setFirecrawlKey(value);
    clearDownstreamState();
  }

  function updateDocsUrl(value: string) {
    setDocsUrl(value);
    clearDownstreamState();
  }

  function updateIntegrationGoal(value: string) {
    setIntegrationGoal(value);
    clearDownstreamState();
  }

  function updateMaxPages(value: number) {
    setMaxPages(value);
    clearDownstreamState();
  }

  function updateMaxDepth(value: number) {
    setMaxDepth(value);
    clearDownstreamState();
  }

  function clearDownstreamState() {
    setResult(null);
    setError(null);
    setFetchResult(null);
    setFetchError(null);
    setReportError(null);
    setAnalysisState("idle");
    onPreflightChanged?.(null);
    onSourceFetchChanged?.(null);
    onReportResultChanged?.(null);
    onReportError?.(null);
    onInputChanged?.();
  }

  async function runPreflightStep(): Promise<DocsPreflightResult | null> {
    setIsPreflightRunning(true);
    setError(null);
    setFetchError(null);
    try {
      const nextResult = await runFirecrawlPreflight({
        docs_url: docsUrl,
        integration_goal: integrationGoal,
        max_pages: maxPages,
        max_depth: maxDepth,
        allowed_hosts: currentRequestHints(request, docsUrl, integrationGoal, maxPages, maxDepth)
          .allowedHosts,
        firecrawl_api_key: firecrawlKey.trim() || undefined,
      });
      setResult(nextResult);
      setFetchResult(null);
      onPreflightChanged?.(nextResult);
      onSourceFetchChanged?.(null);
      onReportResultChanged?.(null);
      return nextResult;
    } catch (preflightError) {
      setResult(null);
      onPreflightChanged?.(null);
      setError(
        preflightError instanceof Error
          ? preflightError.message
          : "Unable to run Firecrawl preflight.",
      );
      return null;
    } finally {
      setIsPreflightRunning(false);
    }
  }

  async function runFetchStep(): Promise<FirecrawlFetchResult | null> {
    setIsFetchRunning(true);
    setFetchError(null);
    try {
      const nextResult = await runFirecrawlFetchSources({
        docs_url: docsUrl,
        integration_goal: integrationGoal,
        max_pages: maxPages,
        max_depth: maxDepth,
        allowed_hosts: currentRequestHints(request, docsUrl, integrationGoal, maxPages, maxDepth)
          .allowedHosts,
        firecrawl_api_key: firecrawlKey.trim() || undefined,
        cache_key: currentRequestHints(request, docsUrl, integrationGoal, maxPages, maxDepth)
          .cacheKey,
      });
      setFetchResult(nextResult);
      onSourceFetchChanged?.(nextResult);
      onReportResultChanged?.(null);
      setReportError(null);
      onReportError?.(null);
      return nextResult;
    } catch (sourceError) {
      setFetchResult(null);
      onSourceFetchChanged?.(null);
      setFetchError(
        sourceError instanceof Error
          ? sourceError.message
          : "Unable to fetch Firecrawl sources.",
      );
      return null;
    } finally {
      setIsFetchRunning(false);
    }
  }

  async function runReportStep(sourceResult: FirecrawlFetchResult): Promise<boolean> {
    setIsReportRunning(true);
    setReportError(null);
    onReportError?.(null);
    try {
      const nextResult = await generateAuditReport({
        docs_url: docsUrl,
        integration_goal: integrationGoal,
        mode: "live",
        max_pages: maxPages,
        max_depth: maxDepth,
        allowed_hosts: currentRequestHints(request, docsUrl, integrationGoal, maxPages, maxDepth)
          .allowedHosts,
        cache_key: sourceResult.cache_key,
      });
      onReportResultChanged?.(nextResult);
      if (nextResult.report) {
        onReportGenerated?.(nextResult.report, nextResult.artifact_path ?? null);
        return true;
      } else {
        const message =
          nextResult.warnings[0] ?? "Report generation was blocked before Codex returned a report.";
        setReportError(message);
        onReportError?.(message);
        return false;
      }
    } catch (generateError) {
      const message =
        generateError instanceof Error
          ? generateError.message
          : "Unable to generate audit report.";
      setReportError(message);
      onReportResultChanged?.(null);
      onReportError?.(message);
      return false;
    } finally {
      setIsReportRunning(false);
    }
  }

  async function handleRunAnalysis() {
    setAnalysisState("checking");
    setResult(null);
    setFetchResult(null);
    setError(null);
    setFetchError(null);
    setReportError(null);
    onPreflightChanged?.(null);
    onSourceFetchChanged?.(null);
    onReportResultChanged?.(null);
    onReportError?.(null);

    const nextPreflight = await runPreflightStep();
    if (!nextPreflight) {
      setAnalysisState("failed");
      return;
    }
    if (nextPreflight.verdict === "blocked") {
      setAnalysisState("blocked");
      return;
    }

    setAnalysisState("fetching");
    const nextFetchResult = await runFetchStep();
    if (!nextFetchResult?.cache_key || nextFetchResult.selected_sources.length === 0) {
      setAnalysisState("failed");
      return;
    }

    setAnalysisState("reporting");
    const reportGenerated = await runReportStep(nextFetchResult);
    setAnalysisState(reportGenerated ? "completed" : "failed");
  }

  const isAnalysisRunning =
    isPreflightRunning || isFetchRunning || isReportRunning ||
    analysisState === "checking" || analysisState === "fetching" || analysisState === "reporting";
  const canRunAnalysis = Boolean(docsUrl.trim() && integrationGoal.trim()) && !isAnalysisRunning;
  const ctaLabel = analysisButtonLabel(analysisState);

  function applyPreset(preset: AuditPreset) {
    setDocsUrl(preset.docsUrl);
    setIntegrationGoal(preset.integrationGoal);
    clearDownstreamState();
  }

  return (
    <section className="panel input-panel" aria-label="Audit request">
        <div className="request-heading">
          <div>
            <p className="request-intro">
              Enter a docs site and the task an agent should complete. We will check access,
              fetch relevant pages, and generate a docs-readiness report.
            </p>
          </div>
        </div>
        <div className="request-grid">
          <label>
            <span>Docs URL</span>
            <input
              value={docsUrl}
              placeholder="https://docs.example.com"
              aria-label="Documentation URL"
              disabled={isAnalysisRunning}
              onChange={(event) => updateDocsUrl(event.target.value)}
            />
          </label>
        </div>
        <label>
          <span>Integration goal</span>
          <textarea
            value={integrationGoal}
            placeholder="Describe the integration task an agent should complete from these docs."
            aria-label="Integration goal"
            disabled={isAnalysisRunning}
            rows={3}
            onChange={(event) => updateIntegrationGoal(event.target.value)}
          />
        </label>
        <div className="preset-strip" aria-label="Example audit presets">
          <span>Try one of these</span>
          <div>
            {AUDIT_PRESETS.map((preset) => (
              <button
                className="preset-chip"
                key={preset.name}
                type="button"
                disabled={isAnalysisRunning}
                onClick={() => applyPreset(preset)}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>
        <details className="advanced-settings" open={!serverFirecrawlConfigured}>
          <summary>Advanced settings</summary>
          <div className="advanced-grid">
            <label>
              <span>Max pages</span>
              <input
                type="number"
                min={1}
                max={50}
                value={maxPages}
                aria-label="Maximum pages"
                disabled={isAnalysisRunning}
                onChange={(event) => updateMaxPages(Number(event.target.value))}
              />
            </label>
            <label>
              <span>Depth</span>
              <input
                type="number"
                min={0}
                max={3}
                value={maxDepth}
                aria-label="Maximum depth"
                disabled={isAnalysisRunning}
                onChange={(event) => updateMaxDepth(Number(event.target.value))}
              />
            </label>
            <label className="key-override">
              <span>{serverFirecrawlConfigured ? "Firecrawl key override" : "Firecrawl API key"}</span>
              <input
                type="password"
                value={firecrawlKey}
                autoComplete="off"
                aria-label="Firecrawl API key"
                disabled={isAnalysisRunning}
                onChange={(event) => updateFirecrawlKey(event.target.value)}
              />
              <small>
                {serverFirecrawlConfigured
                  ? "Optional. Leave blank to use the backend-configured key."
                  : "Required for live Firecrawl checks unless a key is configured on the backend."}
              </small>
            </label>
          </div>
        </details>
        <div className="analysis-cta">
          <button
            type="button"
            className="run-analysis-button"
            disabled={!canRunAnalysis}
            onClick={handleRunAnalysis}
          >
            {ctaLabel}
          </button>
        </div>
        <AnalysisProgress
          analysisState={analysisState}
          preflightResult={result}
          fetchResult={fetchResult}
          reportError={reportError}
        />
        {error ? <p className="preflight-error">{error}</p> : null}
        {fetchError ? <p className="preflight-error">{fetchError}</p> : null}
      {reportError ? <p className="preflight-error">{reportError}</p> : null}
    </section>
  );
}

type AnalysisState = "idle" | "checking" | "fetching" | "reporting" | "completed" | "blocked" | "failed";

function analysisButtonLabel(state: AnalysisState): string {
  if (state === "checking" || state === "fetching" || state === "reporting") {
    return "Running analysis...";
  }
  if (state === "completed") {
    return "Run analysis again";
  }
  if (state === "blocked" || state === "failed") {
    return "Try again";
  }
  return "Run analysis";
}

function AnalysisProgress({
  analysisState,
  preflightResult,
  fetchResult,
  reportError,
}: {
  analysisState: AnalysisState;
  preflightResult: DocsPreflightResult | null;
  fetchResult: FirecrawlFetchResult | null;
  reportError: string | null;
}) {
  const preflightTone = stepTone({
    active: analysisState === "checking",
    complete: Boolean(preflightResult && preflightResult.verdict !== "blocked"),
    warning: preflightResult?.verdict === "warning",
    blocked: preflightResult?.verdict === "blocked",
    failed: analysisState === "failed" && !preflightResult,
  });
  const fetchTone = stepTone({
    active: analysisState === "fetching",
    complete: Boolean(fetchResult?.selected_sources.length),
    warning: Boolean(fetchResult && fetchResult.selected_sources.length < fetchResult.candidate_count),
    failed: analysisState === "failed" && Boolean(preflightResult) && !fetchResult,
  });
  const reportTone = stepTone({
    active: analysisState === "reporting",
    complete: analysisState === "completed",
    failed: Boolean(reportError) || (analysisState === "failed" && Boolean(fetchResult)),
  });

  return (
    <ol className="analysis-progress" aria-label="Analysis progress">
      <AnalysisStep label="Check access" tone={preflightTone} />
      <AnalysisStep label="Fetch evidence" tone={fetchTone} detail={fetchResult ? `${fetchResult.selected_sources.length} pages` : null} />
      <AnalysisStep label="Generate report" tone={reportTone} />
    </ol>
  );
}

type AnalysisStepTone = "pending" | "active" | "complete" | "warning" | "blocked" | "failed";

function stepTone({
  active = false,
  complete = false,
  warning = false,
  blocked = false,
  failed = false,
}: {
  active?: boolean;
  complete?: boolean;
  warning?: boolean;
  blocked?: boolean;
  failed?: boolean;
}): AnalysisStepTone {
  if (active) {
    return "active";
  }
  if (blocked) {
    return "blocked";
  }
  if (failed) {
    return "failed";
  }
  if (warning) {
    return "warning";
  }
  if (complete) {
    return "complete";
  }
  return "pending";
}

function AnalysisStep({
  label,
  tone,
  detail,
}: {
  label: string;
  tone: AnalysisStepTone;
  detail?: string | null;
}) {
  return (
    <li className={`analysis-step ${tone}`}>
      <span aria-hidden="true" />
      <div>
        <strong>{label}</strong>
        {detail ? <small>{detail}</small> : null}
      </div>
    </li>
  );
}

interface AuditPreset {
  name: string;
  docsUrl: string;
  integrationGoal: string;
}

const AUDIT_PRESETS: AuditPreset[] = [
  {
    name: "Stripe",
    docsUrl: "https://docs.stripe.com/checkout/quickstart",
    integrationGoal:
      "Create a Stripe Checkout Session for a one-time payment, redirect the customer to Checkout, and explain the success and cancel URL setup.",
  },
  {
    name: "OpenAI",
    docsUrl: "https://platform.openai.com/docs/quickstart",
    integrationGoal:
      "Build a small Node.js script that calls the OpenAI API, sends one user prompt, prints the model response, and explains the required API key setup.",
  },
  {
    name: "Supabase",
    docsUrl: "https://supabase.com/docs/guides/auth",
    integrationGoal:
      "Add email sign-up and sign-in with Supabase Auth, read the current user session, and explain the environment variables needed in a web app.",
  },
  {
    name: "Twilio",
    docsUrl: "https://www.twilio.com/docs/messaging/quickstart",
    integrationGoal:
      "Send one SMS message with Twilio from a server-side script, including the account credentials, sender number, recipient number, and safe production defaults.",
  },
  {
    name: "Firecrawl",
    docsUrl: "https://docs.firecrawl.dev/",
    integrationGoal:
      "Set up Firecrawl in a Python agent to search the web, scrape the top result as markdown, and explain the API key, SDK install, and request options needed for production.",
  },
];

export function PreflightResult({ result }: { result: DocsPreflightResult }) {
  return (
    <div className={`preflight-result ${result.verdict}`}>
      <div className="preflight-verdict">
        <span>Verdict</span>
        <strong>{formatStatus(result.verdict)}</strong>
      </div>
      <div className="preflight-meta">
        <span>{result.normalized_url ?? "No normalized URL"}</span>
        <span>{result.allowed_hosts.join(", ") || "No allowed host"}</span>
      </div>
      <div className="preflight-key">
        <span>Key</span>
        <strong>{formatStatus(result.key_status.status)}</strong>
      </div>
      <div className="preflight-checks">
        {result.checks.map((check, index) => (
          <div className={`preflight-check ${check.status}`} key={`${check.id}-${index}`}>
            <div>
              <span>{formatStatus(check.id)}</span>
              <strong>{formatStatus(check.status)}</strong>
            </div>
            <p>{check.message}</p>
            {check.url ? <small>{check.url}</small> : null}
          </div>
        ))}
      </div>
    </div>
  );
}

function formatStatus(status: string): string {
  return status.replaceAll("_", " ");
}

function currentRequestHints(
  request: AuditRequest | null,
  docsUrl: string,
  integrationGoal: string,
  maxPages: number,
  maxDepth: number,
): { allowedHosts?: string[]; cacheKey?: string } {
  if (!request) {
    return {};
  }
  const matchesInitialRequest =
    docsUrl === request.docs_url &&
    integrationGoal === request.integration_goal &&
    maxPages === (request.max_pages ?? 20) &&
    maxDepth === (request.max_depth ?? 1);
  if (!matchesInitialRequest) {
    return {};
  }
  return {
    allowedHosts: request.allowed_hosts,
    cacheKey: request.cache_key,
  };
}

export function ReportGenerationResult({ result }: { result: AuditReportResult }) {
  return (
    <div className={`report-generation-result ${result.status}`}>
      <div className="preflight-verdict">
        <span>Report</span>
        <strong>{formatStatus(result.status)}</strong>
      </div>
      <div className="source-fetch-meta">
        <span>{result.cache_key}</span>
        {result.artifact_path ? <span>{result.artifact_path}</span> : null}
      </div>
      {result.warnings.length ? (
        <div className="source-warnings">
          {result.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}
    </div>
  );
}
