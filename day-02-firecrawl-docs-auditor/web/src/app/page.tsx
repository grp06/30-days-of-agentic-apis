import { AuditWorkspace } from "@/components/audit-workspace";
import { CodexAuthPanel } from "@/components/codex-auth-panel";
import { getStatus } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function Home() {
  const data = await loadScaffoldData();

  return (
    <main className="app-shell">
      <section className="top-bar">
        <div>
          <p className="eyebrow brand-kicker">
            <span className="flame-mark" aria-hidden="true" />
            Day 2 / Firecrawl
          </p>
          <h1>Agent-native docs auditor</h1>
        </div>
        <div className="top-actions">
          <CodexAuthPanel variant="header" />
        </div>
      </section>

      <AuditWorkspace
        initialStatus={data.status}
        initialRequest={data.request}
        initialReport={data.report}
        initialError={data.error}
      />
    </main>
  );
}

async function loadScaffoldData() {
  const statusResult = await Promise.allSettled([getStatus()]).then((results) => results[0]);
  const status = statusResult.status === "fulfilled" ? statusResult.value : null;
  const message =
    statusResult.status === "rejected" && statusResult.reason instanceof Error
      ? statusResult.reason.message
      : null;
  const error =
    message && !status
      ? `${message} Start the backend and refresh.`
      : message
        ? `Some scaffold data could not load: ${message}`
        : null;
  return {
    status,
    request: null,
    report: null,
    error,
  };
}
