"use client";

import { useState } from "react";

import {
  AuditFormShell,
  PreflightResult,
  ReportGenerationResult,
} from "@/components/audit-form-shell";
import { ReportPreview } from "@/components/report-preview";
import { ServiceStatusPanel } from "@/components/service-status";
import { SourceFetchPreview } from "@/components/source-fetch-preview";
import {
  type AuditReport,
  type AuditRequest,
  type AuditReportResult,
  type AuditorStatus,
  type DocsPreflightResult,
  type FirecrawlFetchResult,
} from "@/lib/api";

interface AuditWorkspaceProps {
  initialStatus: AuditorStatus | null;
  initialRequest: AuditRequest | null;
  initialReport: AuditReport | null;
  initialError: string | null;
}

export function AuditWorkspace({
  initialStatus,
  initialRequest,
  initialReport,
  initialError,
}: AuditWorkspaceProps) {
  const [status, setStatus] = useState(initialStatus);
  const [preflightResult, setPreflightResult] = useState<DocsPreflightResult | null>(null);
  const [sourceFetchResult, setSourceFetchResult] = useState<FirecrawlFetchResult | null>(null);
  const [reportResult, setReportResult] = useState<AuditReportResult | null>(null);
  const [report, setReport] = useState(initialReport);
  const [reportError, setReportError] = useState(initialError);
  const [artifactPath, setArtifactPath] = useState<string | null>(null);
  const [runLabel, setRunLabel] = useState<string | null>(null);

  return (
    <section className="workspace">
      <section className="workflow-shell" aria-label="Audit workflow">
        <AuditFormShell
          request={initialRequest}
          serverFirecrawlConfigured={status?.firecrawl.status === "configured"}
          onInputChanged={() => {
            setPreflightResult(null);
            setSourceFetchResult(null);
            setReportResult(null);
            setReport(null);
            setReportError(null);
            setArtifactPath(null);
            setRunLabel(null);
          }}
          onPreflightChanged={setPreflightResult}
          onSourceFetchChanged={(nextResult) => {
            setSourceFetchResult(nextResult);
            setReportResult(null);
            setReport(null);
            setReportError(null);
            setArtifactPath(null);
            setRunLabel(nextResult ? "live run" : null);
          }}
          onReportResultChanged={setReportResult}
          onReportGenerated={(nextReport, nextArtifactPath) => {
            if (nextReport) {
              setReport(nextReport);
              setArtifactPath(nextArtifactPath);
              setReportError(null);
              setRunLabel("live run");
            }
          }}
          onReportError={setReportError}
        />

      </section>

      <HowItWorksPanel />

      <ReportPreview
        report={report}
        error={reportError}
      />

      <section className="run-details">
        <h2>Run details</h2>
        <details>
          <summary>Preflight checks</summary>
          {preflightResult ? <PreflightResult result={preflightResult} /> : <p>No preflight run yet.</p>}
        </details>
        <details>
          <summary>Fetched sources</summary>
          <SourceFetchPreview result={sourceFetchResult} title="Fetch result" runLabel={runLabel} />
          {!sourceFetchResult ? <p>No sources fetched yet.</p> : null}
        </details>
        <details>
          <summary>Report artifacts</summary>
          {reportResult ? <ReportGenerationResult result={reportResult} /> : null}
          {artifactPath ? <p className="artifact-path">{artifactPath}</p> : null}
          {!reportResult && !artifactPath ? <p>No report generated yet.</p> : null}
        </details>
        <details>
          <summary>Service status</summary>
          <ServiceStatusPanel status={status} onStatusChanged={setStatus} showHeader={false} />
        </details>
      </section>
    </section>
  );
}

function HowItWorksPanel() {
  return (
    <aside className="overview-panel" aria-label="How it works">
      <p className="eyebrow">How it works</p>
      <div className="overview-steps">
        <OverviewStep
          icon="search"
          title="Check access"
          body="Firecrawl validates the docs site."
        />
        <OverviewStep
          icon="pages"
          title="Fetch evidence"
          body="Firecrawl finds relevant pages."
        />
        <OverviewStep
          icon="spark"
          title="Infer readiness"
          body="Codex judges agent readiness."
        />
      </div>
    </aside>
  );
}

function OverviewStep({
  icon,
  title,
  body,
}: {
  icon: "search" | "pages" | "spark";
  title: string;
  body: string;
}) {
  return (
    <article className="overview-step">
      <span className="overview-icon" aria-hidden="true">
        <OverviewIcon icon={icon} />
      </span>
      <div>
        <strong>{title}</strong>
        <p>{body}</p>
      </div>
    </article>
  );
}

function OverviewIcon({ icon }: { icon: "search" | "pages" | "spark" }) {
  if (icon === "search") {
    return (
      <svg viewBox="0 0 24 24" focusable="false">
        <circle cx="10.5" cy="10.5" r="5.5" />
        <path d="m15 15 4 4" />
      </svg>
    );
  }
  if (icon === "pages") {
    return (
      <svg viewBox="0 0 24 24" focusable="false">
        <path d="M7 4h7l3 3v13H7z" />
        <path d="M14 4v4h4" />
        <path d="M10 12h5M10 15h5" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" focusable="false">
      <path d="M12 3v5M12 16v5M4 12h5M15 12h5" />
      <path d="m7 7 2 2M15 15l2 2M17 7l-2 2M9 15l-2 2" />
    </svg>
  );
}
