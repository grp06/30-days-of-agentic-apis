# Agent Black Box

Same task. Four E2B sandboxes. Inspectable trajectories.

Agent Black Box compares how coding agents diverge when they start from the same tiny app. Each lane runs in its own E2B sandbox, then the app records the preview state, commands, diffs, checkpoints, and replay evidence so the result is explainable after the agent finishes.

## What to Notice

- Every lane starts from the same three-file Hello World fixture.
- E2B isolates each run while still giving the agent a real filesystem, shell, package install, build, and preview surface.
- The arena page compares outcomes at a glance.
- The replay page shows the evidence behind a lane instead of only the final app.
- Local `runs/` and `arenas/` are development artifacts and are intentionally not committed.

## Environment

Copy `.env.example` to `.env` and add real keys:

```sh
E2B_API_KEY=...
OLLAMA_API_KEY=...
```

Optional runtime settings live in `.env.example`. For the web app, set:

```sh
NEXT_PUBLIC_AGENT_BLACK_BOX_API_URL=http://127.0.0.1:8011
```

Put that in `web/.env.local` for local frontend work.

## Run Locally

Install Python dependencies:

```sh
uv sync
```

Build the E2B template before the first real sandbox run:

```sh
e2b template create agent-black-box
```

Check credentials and runtime assumptions:

```sh
uv run python -m agent_black_box.cli doctor
```

Run the backend:

```sh
uv run python -m agent_black_box.cli serve-api --host 127.0.0.1 --port 8011 --reload
```

Run the frontend:

```sh
pnpm --dir web install
printf 'NEXT_PUBLIC_AGENT_BLACK_BOX_API_URL=http://127.0.0.1:8011\n' > web/.env.local
pnpm --dir web dev
```

## Demo Flow

Use the homepage to open a prepared or recent arena instead of waiting on a fresh live run.

1. Open the homepage and state the premise: one prompt, one starting app, four isolated E2B sandboxes.
2. Open an arena and compare the lane summaries, preview states, checkpoints, and model names.
3. Open one strong preview.
4. Open that lane's replay and show the diff, commands, and checkpoint evidence.
5. Close on why E2B matters: the agents can act in real dev environments, and the work remains inspectable afterward.

## Useful Commands

Run tests:

```sh
uv run pytest
```

Lint Python:

```sh
uv run ruff check .
```

Launch a four-lane arena from the CLI:

```sh
uv run python -m agent_black_box.cli run-arena --fixture sample_frontend_task --task-override "Turn this tiny static Hello World app into a focused Agent Black Box landing page."
```

Inspect a recorded arena from the CLI when you need raw JSON:

```sh
uv run python -m agent_black_box.cli show-arena --arena-id <arena-id>
```

Validate the frontend:

```sh
pnpm --dir web lint
pnpm --dir web build
```

## Artifact Policy

The source repo should contain app code, fixtures, tests, docs, and examples. It should not contain local E2B evidence from active development:

- `runs/`
- `arenas/`
- `web/node_modules/`
- `web/.next/`
- fixture `node_modules/` and `dist/`
- `.env` and other local secret files

If a public demo needs canonical evidence later, add a small sanitized fixture under a named examples directory instead of committing live `runs/` or `arenas/` output.
