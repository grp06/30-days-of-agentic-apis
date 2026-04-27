# Day 2: Firecrawl Agent-Native Docs Auditor

Paste a public docs URL and an integration goal. The app will use Firecrawl to
discover and fetch a bounded set of relevant docs pages, then use Codex app
server inference to produce an agent-readiness report with source-backed
claims, warnings, and suggested docs fixes.

This folder now has a runnable local app with a Codex app-server auth layer, a
Firecrawl preflight path, bounded goal-directed source fetching, and
Codex-backed audit report generation.

## Contract Status

The first implemented slice defines the product contract. Read
[`contracts/README.md`](./contracts/README.md) for the local user journey,
request/report schemas, provenance rules, and non-goals.

The backend owns managed Codex auth through `codex app-server`, so the UI can
show account state, start browser or device-code login, cancel/logout, and run
a tiny authenticated smoke test without reading local credential files. The
Firecrawl preflight endpoint checks runtime key status and direct public docs
entrypoints before any multi-page map, crawl, or scrape work. The source-fetch
endpoint then uses Firecrawl map/scrape plus source planning and deterministic
ranking to select and fetch a bounded same-host page set for the requested
goal, writing replay artifacts under `.agent/cache/firecrawl-runs/`. After
source fetch succeeds, the audit report endpoint sends a compact evidence
packet through Codex app-server, normalizes the structured response into the
public report schema, and writes report artifacts under
`.agent/cache/audit-reports/`.

## Run Locally

Backend:

```bash
uv sync
uv run pytest
uv run uvicorn firecrawl_docs_auditor.server:create_app --factory --host 127.0.0.1 --port 8122
```

Frontend:

```bash
pnpm --dir web install
pnpm --dir web lint
pnpm --dir web build
pnpm --dir web dev --port 3122
```

Open `http://127.0.0.1:3122` with the backend running. The Codex auth panel is
live when `codex app-server` can be launched. Enter a docs URL and an
integration goal, then click `Run analysis`. The app checks access, fetches
selected source pages with Firecrawl, and generates a source-backed report. If
Codex is not signed in, report generation returns a safe blocked status instead
of fabricating a completed report.

## Live Audit

Live source fetching requires a Firecrawl key. You can provide it through
`FIRECRAWL_API_KEY` or paste it into the local UI. Live report generation also
requires Codex app-server sign-in through the Codex auth panel. If either
credential is missing or unavailable, the app should return a visible blocked
state instead of pretending a completed audit happened.

## Environment

Copy placeholders from `.env.example` when needed. Keep real Firecrawl keys and
Codex account state in local runtime inputs only; they should not appear in
fixtures, schemas, or report artifacts.

`FIRECRAWL_API_KEY` is optional for URL preflight and required later for live
fetching. You can also paste the key into the local UI for one preflight run.
The backend reports only configured/missing/validation status and never returns
the key.

`FIRECRAWL_API_BASE_URL` defaults to `https://api.firecrawl.dev/v2`.
`FIRECRAWL_TIMEOUT_SECONDS` controls the bounded Firecrawl smoke request and
direct docs entrypoint checks.

Live source fetch writes local replay artifacts to `.agent/cache/firecrawl-runs/`.
Those artifacts are ignored and may contain public docs Markdown, but they must
not contain Firecrawl keys, authorization headers, cookies, tokens, or Codex
account state.

Generated audit reports write local replay artifacts to
`.agent/cache/audit-reports/`. `report.json` is the same safe report shape
returned by the API; `report.md` is a human-readable summary with source ids,
source URLs, scorecard, smoke test, warnings, and suggested fixes. These files
are ignored and safe to keep locally, but they may include source URLs and
Codex-written analysis.

`CODEX_BIN` is optional. Set it when the `codex` executable is not on `PATH`,
for example:

```bash
CODEX_BIN=/Applications/Codex.app/Contents/Resources/codex
```

`CODEX_APP_SERVER_TIMEOUT_SECONDS` controls the backend timeout for managed
app-server requests. Report generation can take longer than auth/status checks,
so the default is 30 seconds for thread setup and up to three times that for the
report turn. The app uses managed app-server login only; it does not copy
`auth.json`, tokens, cookies, or other Codex credential files.

## Local API

The auth UI calls these local backend routes:

- `GET /api/firecrawl/status`
- `POST /api/firecrawl/preflight`
- `POST /api/firecrawl/fetch-sources`
- `GET /api/firecrawl/fetch-sources/{cache_key}`
- `POST /api/audit/report`
- `GET /api/codex/account`
- `POST /api/codex/login/start`
- `GET /api/codex/login/status/{login_id}`
- `POST /api/codex/login/cancel`
- `POST /api/codex/logout`
- `POST /api/codex/smoke-test`
