import type { FirecrawlFetchResult } from "@/lib/api";

interface SourceFetchPreviewProps {
  result: FirecrawlFetchResult | null;
  title?: string;
  runLabel?: string | null;
}

export function SourceFetchPreview({
  result,
  title = "Sources",
  runLabel,
}: SourceFetchPreviewProps) {
  if (!result) {
    return null;
  }

  return (
    <div className={`source-fetch-result ${result.status}`}>
      <div className="preflight-verdict">
        <span>{title}</span>
        <strong>{formatStatus(result.status)}</strong>
      </div>
      <div className="source-fetch-meta">
        <span>{result.selected_sources.length} fetched</span>
        <span>{result.candidate_count} candidates considered</span>
        {runLabel ? <span>{runLabel}</span> : null}
      </div>
      {result.warnings.length ? (
        <div className="source-warnings">
          {result.warnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      ) : null}
      <div className="source-list">
        {result.selected_sources.map((source, index) => (
          <article className="source-card" key={source.id}>
            <span className="source-index">{index + 1}</span>
            <div>
              <h3>{source.title}</h3>
              <a href={source.url} target="_blank" rel="noreferrer">
                {source.url}
              </a>
              <p>{source.reason_selected}</p>
            </div>
            <small>{source.markdown_chars.toLocaleString()} chars</small>
          </article>
        ))}
      </div>
      {result.artifact_path ? (
        <details className="source-artifact">
          <summary>Artifact</summary>
          <span>{result.artifact_path}</span>
        </details>
      ) : null}
    </div>
  );
}

function formatStatus(status: string): string {
  return status.replaceAll("_", " ");
}
