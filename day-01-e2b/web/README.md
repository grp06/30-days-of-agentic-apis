## Agent Black Box Frontend

This directory contains the Next.js frontend for the Agent Black Box hosted demo.
The homepage is a small doorway into one model-comparison flow: start an arena,
open an existing arena, then inspect individual replay pages.

### Local development

From the repo root:

```bash
pnpm --dir web install
printf 'NEXT_PUBLIC_AGENT_BLACK_BOX_API_URL=http://127.0.0.1:8011\n' > web/.env.local
pnpm --dir web dev
```

Or from inside `web/`:

```bash
pnpm install
printf 'NEXT_PUBLIC_AGENT_BLACK_BOX_API_URL=http://127.0.0.1:8011\n' > .env.local
pnpm dev
```

The frontend expects the FastAPI backend to be running at
`NEXT_PUBLIC_AGENT_BLACK_BOX_API_URL`. For a hosted deployment, point that env var
at the deployed backend origin instead of localhost.

For the product overview, seeded demo context, and backend setup, see the
repo-root [README.md](../README.md).
