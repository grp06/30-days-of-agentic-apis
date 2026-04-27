# Firecrawl Docs Auditor Product Contract

This folder defines the Day 2 product contract before implementation. The app is a local demo that answers one question: can an AI coding agent use a public documentation site to complete a specific integration goal?

The first version is goal-directed. It does not build a whole-site search engine, exhaustive crawler, or generated MCP server. Later code should treat this contract as the product boundary for request data, report data, provenance, and non-goals.

## Local User Journey

1. Start the local app.
2. Sign in with ChatGPT/Codex through managed `codex app-server` auth.
3. Enter a Firecrawl API key as a runtime credential.
4. Enter a public developer-docs URL and one integration goal.
5. Run a bounded audit.
6. Inspect a report with source-backed claims, warnings, and suggested fixes.

The Firecrawl key and Codex account state are runtime inputs. They must not be stored in request fixtures, report artifacts, or schema examples.

## Audit Request

An audit request is the safe-to-store input for one audit run. It describes what docs to inspect and what the agent is trying to accomplish.

Required fields:

- `docs_url`: a public `http://` or `https://` documentation entry point.
- `integration_goal`: a human-readable goal, such as "Create a checkout session and redirect the user."
- `mode`: either `live` or `cached`.

Optional bounded controls:

- `max_pages`: maximum pages to inspect. Default: 20. Allowed range: 1-50.
- `max_depth`: link depth from the entry point. Default: 1. Allowed range: 0-3.
- `allowed_hosts`: host allowlist for later fetch logic. This is a safety boundary, not permission to crawl broadly.
- `cache_key`: a stable local fixture key when `mode` is `cached`.

An audit request must not include API keys, Codex auth tokens, cookies, account ids, or private endpoints.

## Audit Report

An audit report is the safe-to-store result of one run. The first version uses these top-level sections:

- `request`: the safe audit request data.
- `status`: `completed`, `completed_with_warnings`, or `blocked`.
- `summary`: a short human-readable result.
- `scorecard`: scored dimensions for agent readiness.
- `selected_sources`: the source pages used as evidence.
- `extracted_facts`: facts the audit found in the docs.
- `smoke_test`: a reasoning-only check of whether the docs are enough for an agent to attempt the goal.
- `warnings`: missing or confusing documentation signals.
- `suggested_fixes`: concrete docs changes that would help agents.
- `metadata`: local run metadata that contains no secrets.

The scorecard dimensions for the initial demo are:

- `discoverability`: can an agent find the right starting pages?
- `task_fit`: do the selected pages match the integration goal?
- `completeness`: are required steps, parameters, and outcomes documented?
- `copy_pasteability`: can an agent safely reuse examples or snippets?
- `agent_friction`: how much extra inference, guessing, or cross-page assembly is required?

## Provenance Rule

Every important claim must either point to selected source ids or be marked as inferred or uncertain.

Use `source_refs` when a claim is backed by selected documentation pages. Use `basis: "inferred"` when the claim is a reasonable conclusion from the available sources but not directly stated. Use `basis: "uncertain"` when the report is intentionally calling out a gap.

This rule applies to scorecard rationales, extracted facts, warnings, suggested fixes, and the smoke-test result.

## Non-Goals

The first version must not:

- crawl an entire documentation site;
- index private, authenticated, CAPTCHA-protected, dashboard-only, or rate-limited pages;
- bypass robots, crawl-delay signals, auth walls, or terms of service;
- redistribute bulk documentation content beyond local demo cache and generated report artifacts;
- generate a complete MCP server for arbitrary APIs;
- add hosted multi-user account management;
- automate a browser as the docs ingestion path;
- crawl across multiple hosts by default;
- store Firecrawl keys, Codex auth state, cookies, tokens, or account ids in report artifacts.

## Later Slice Handoff

Later implementation slices should use this contract as the common vocabulary. Backend code can translate these schemas into Pydantic models. Frontend code can use the same field names for UI state and report rendering. Firecrawl and Codex integration code should produce reports that satisfy the schema instead of inventing separate section names or provenance rules.
