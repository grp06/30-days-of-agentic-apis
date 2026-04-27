import type { AuditReport } from "@/lib/api";

interface ReportPreviewProps {
  report: AuditReport | null;
  error: string | null;
}

export function ReportPreview({ report, error }: ReportPreviewProps) {
  if (error) {
    return (
      <section className="panel report-panel" aria-label="Report preview">
        <p className="eyebrow">Report preview</p>
        <div className="empty-state">
          <h2>Backend unavailable</h2>
          <p>{error}</p>
        </div>
      </section>
    );
  }

  if (!report) {
    return (
      <section className="panel report-panel" aria-label="Report preview">
        <div className="empty-state">
          <p className="eyebrow">Generated report</p>
          <h2>Your docs-readiness answer will appear here.</h2>
          <p>Codex turns the fetched docs evidence into a concise judgment for the integration goal.</p>
          <div className="empty-report-grid" aria-label="Report preview placeholders">
            <span>Verdict</span>
            <span>Top blockers</span>
            <span>Recommended fix</span>
          </div>
        </div>
      </section>
    );
  }

  const primaryFix = selectPrimaryFix(report.suggested_fixes);
  const sourceMap = buildSourceMap(report);
  const fixLabel = primaryFixLabel(report, primaryFix);
  const passWithImprovement =
    report.smoke_test.result === "pass" && primaryFix?.severity === "info";

  return (
    <section className="panel report-panel decision-report" aria-label="Report preview">
      <div className="decision-hero">
        <h2>{report.smoke_test.message}</h2>
        <p>{report.summary}</p>
        {passWithImprovement ? (
          <p>The docs appear sufficient for this task. The item below is an improvement opportunity, not a blocker.</p>
        ) : null}
      </div>

      <div className="decision-grid">
        <section className="decision-section recommended-fix-card">
          <p className="eyebrow">{fixLabel}</p>
          {primaryFix ? (
            <>
              <strong>{primaryFix.message}</strong>
              <SourceRefs label="Sources" refs={primaryFix.source_refs} sources={sourceMap} />
            </>
          ) : (
            <p>No suggested fix returned.</p>
          )}
        </section>
        <section className="decision-section blockers-card">
          <p className="eyebrow">Blocking gaps</p>
          <BulletList items={report.smoke_test.missing_facts ?? []} empty="No required claim is blocked." />
        </section>
      </div>

      <PromptEvidenceSummary report={report} sources={sourceMap} />

      <ScorecardSummary report={report} sources={sourceMap} />

      <details className="report-detail">
        <summary>Evidence details</summary>
        <EvidenceReview report={report} sources={sourceMap} />
      </details>

      <details className="report-detail">
        <summary>Selected sources</summary>
        <div className="report-source-list">
          {report.selected_sources.map((source) => (
            <article className="report-source" key={source.id}>
              <div>
                <strong>{sourceNumber(source.id)}</strong>
              </div>
              <h4>{source.title}</h4>
              <a href={source.url} target="_blank" rel="noreferrer">
                {source.url}
              </a>
              <p>{source.reason_selected}</p>
            </article>
          ))}
        </div>
      </details>
    </section>
  );
}

function ScorecardSummary({
  report,
  sources,
}: {
  report: AuditReport;
  sources: SourceLookup;
}) {
  return (
    <div className="score-grid" aria-label="Scorecard">
      {report.scorecard.map((dimension) => (
        <article className="score-card" key={dimension.id}>
          <div>
            <span>{dimension.label}</span>
            <strong className={`score-badge ${scoreTone(dimension.score, dimension.max_score)}`}>
              {dimension.score}/{dimension.max_score}
            </strong>
          </div>
          <p>{dimension.rationale}</p>
          <SourceRefs refs={dimension.source_refs} sources={sources} />
        </article>
      ))}
    </div>
  );
}

function PromptEvidenceSummary({
  report,
  sources,
}: {
  report: AuditReport;
  sources: SourceLookup;
}) {
  const metadata = report.metadata;
  const selectedSourceCount = report.selected_sources.length;
  const promptSourceCount = metadata.prompt_source_count ?? selectedSourceCount;
  const fetchedSourceCount = metadata.fetched_source_count ?? selectedSourceCount;
  const supportedClaims = metadata.supported_claims ?? [];
  const missingClaims = metadata.missing_claims ?? [];
  const requiredSupported = metadata.required_claims_supported ?? supportedClaims.length;
  const requiredTotal =
    metadata.required_claims_total ?? supportedClaims.length + missingClaims.length;
  const optionalSupported = metadata.optional_claims_supported ?? 0;
  const optionalTotal = metadata.optional_claims_total ?? 0;
  const missingRequired = metadata.missing_required_claims ?? [];
  const missingOptional = metadata.missing_optional_claims ?? [];
  const primaryEvidence = primaryEvidenceSources(report, sources);

  return (
    <section className="evidence-summary" aria-label="Evidence packet">
      <strong>
        {requiredSupported}/{requiredTotal || requiredSupported} required claims supported
      </strong>
      {metadata.prompt_note ? <p>{metadata.prompt_note}</p> : null}
      {optionalTotal ? (
        <small>{optionalSupported}/{optionalTotal} improvement checks supported</small>
      ) : null}
      <InlineList label="Missing required claims" items={missingRequired.map(formatClaimId)} />
      <InlineList label="Improvement areas" items={missingOptional.map(formatClaimId)} />
      <InlineList label="Supported claims" items={supportedClaims.map(formatClaimId)} />
      <InlineList label="Primary evidence" items={primaryEvidence} />
      <small>{promptSourceCount}/{fetchedSourceCount} fetched sources contributed to the proof ledger.</small>
      <InlineList label="Covered terms" items={metadata.represented_goal_terms} />
      <InlineList label="Missing terms" items={metadata.missing_goal_terms} />
      {metadata.omitted_source_count ? (
        <small>{metadata.omitted_source_count} fetched sources had no stronger claim evidence in the final packet.</small>
      ) : null}
    </section>
  );
}

function EvidenceReview({
  report,
  sources,
}: {
  report: AuditReport;
  sources: SourceLookup;
}) {
  const sections = [
    { kind: "Risks", title: "What may block an agent", items: report.warnings },
    { kind: "Suggested fixes", title: "What to improve", items: report.suggested_fixes },
    { kind: "Supporting facts", title: "What the docs already prove", items: report.extracted_facts },
  ];

  return (
    <div className="evidence-review">
      {sections.map((section) => (
        <section className="evidence-group" key={section.kind}>
          <div className="evidence-group-heading">
            <span>{section.kind}</span>
            <h3>{section.title}</h3>
          </div>
          <div className="evidence-items">
            {section.items.map((item) => (
              <article className="evidence-item compact-citations" key={item.id}>
                <div>
                  <strong>{formatLabels(item.severity, item.basis)}</strong>
                </div>
                <p>
                  {item.message}
                  <SourceRefs refs={item.source_refs} sources={sources} />
                </p>
              </article>
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function BulletList({ items, empty }: { items: string[]; empty: string }) {
  if (!items.length) {
    return <p>{empty}</p>;
  }
  return (
    <ul>
      {items.map((item) => (
        <li key={item}>{item}</li>
      ))}
    </ul>
  );
}

function InlineList({ label, items }: { label: string; items?: string[] }) {
  if (!items?.length) {
    return null;
  }
  return (
    <small>
      {label}: {items.join(", ")}
    </small>
  );
}

type SourceLookup = Map<string, AuditReport["selected_sources"][number]>;

function SourceRefs({
  label,
  refs,
  sources,
}: {
  label?: string;
  refs?: string[];
  sources: SourceLookup;
}) {
  if (!refs?.length) {
    return null;
  }
  return (
    <small className="source-refs">
      {label ? <span>{label}: </span> : null}
      {refs.map((ref, index) => {
        const source = sources.get(ref);
        const text = sourceNumber(ref);
        return (
          <span key={`${ref}-${index}`}>
            {source ? (
              <a
                className="source-ref-chip"
                href={source.url}
                title={source.title}
                target="_blank"
                rel="noreferrer"
              >
                {text}
              </a>
            ) : (
              <span className="source-ref-chip">{text}</span>
            )}
          </span>
        );
      })}
    </small>
  );
}

function buildSourceMap(report: AuditReport): SourceLookup {
  return new Map(report.selected_sources.map((source) => [source.id, source]));
}

function sourceNumber(sourceId: string): string {
  const match = sourceId.match(/\d+$/);
  return match?.[0] ?? sourceId;
}

function selectPrimaryFix(items: AuditReport["suggested_fixes"]) {
  const rank = { blocking: 0, warning: 1, info: 2 };
  return (
    [...items].sort(
      (left, right) =>
        rank[left.severity ?? "info"] - rank[right.severity ?? "info"],
    )[0] ?? null
  );
}

function primaryFixLabel(
  report: AuditReport,
  fix: AuditReport["suggested_fixes"][number] | null,
): string {
  if (report.smoke_test.result === "pass" && fix?.severity === "info") {
    return "Best docs improvement";
  }
  if (fix?.severity === "blocking" || fix?.severity === "warning") {
    return "Highest-priority fix";
  }
  return "Recommended fix";
}

function primaryEvidenceSources(report: AuditReport, sources: SourceLookup): string[] {
  const refs = [
    ...(report.smoke_test.source_refs ?? []),
    ...report.scorecard.flatMap((dimension) => dimension.source_refs ?? []),
    ...report.suggested_fixes.flatMap((fix) => fix.source_refs ?? []),
  ];
  const names: string[] = [];
  for (const ref of refs) {
    const source = sources.get(ref);
    if (!source) {
      continue;
    }
    const name = source.title.replace(/\s*[-|]\s*[^-|]+$/, "").trim();
    if (name && !names.includes(name)) {
      names.push(name);
    }
    if (names.length >= 4) {
      break;
    }
  }
  return names;
}

function scoreTone(score: number, maxScore: number): "strong" | "mixed" | "weak" {
  const ratio = maxScore > 0 ? score / maxScore : 0;
  if (ratio >= 0.8) {
    return "strong";
  }
  if (ratio >= 0.55) {
    return "mixed";
  }
  return "weak";
}

function formatStatus(status: string): string {
  return status.replaceAll("_", " ");
}

function formatClaimId(claimId: string): string {
  return claimId
    .replace(/^goal_/, "")
    .replaceAll("_", " ");
}

function formatLabels(...labels: Array<string | undefined>): string {
  return labels
    .filter((label): label is string => Boolean(label))
    .map(formatStatus)
    .join(" / ");
}
