from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreviewPolicy:
    command: str
    port: int
    cleanup_command: str
    start_timeout_seconds: int = 30


@dataclass(frozen=True)
class FixturePolicy:
    guardrails: str
    setup_command: str | None = None
    setup_timeout_seconds: int = 120
    build_command: str | None = None
    build_timeout_seconds: int = 120
    preview: PreviewPolicy | None = None

    def is_build_command(self, command: str) -> bool:
        return (
            self.build_command is not None
            and _normalize_command(command) == _normalize_command(self.build_command)
        )

    def should_checkpoint_after_command(
        self, command: str, *, preview_published: bool
    ) -> bool:
        return preview_published or self.is_build_command(command)

    def preview_port(self, default: int) -> int:
        if self.preview is None:
            return default
        return self.preview.port

    def suppresses_command_preview_detection(self) -> bool:
        return self.preview is not None


_FIXTURE_POLICIES: dict[str, FixturePolicy] = {
    "sample_frontend_task": FixturePolicy(
        guardrails="""Fixture guardrails:
- Treat the user request as the content brief for the page.
- Keep edits inside index.html, index.css, and index.js unless a truly necessary fix requires another file.
- Make sure the browser-visible result matches the request; do not leave JavaScript that overwrites your intended page after load.
- Keep the result compact and focused; avoid extra sections, new libraries, and framework churn.
- Use pnpm install before building if dependencies are missing.
- The harness will run the final build; run it yourself only if you need the output while debugging.
- Do not start or manage preview servers; the harness publishes the preview after the model finishes.
- When the build is verified, call finish_run with a concise summary.""",
        setup_command="test -d node_modules || pnpm install",
        build_command="pnpm build",
        preview=PreviewPolicy(
            command="pnpm dev --host 0.0.0.0 --port 4173 --strictPort",
            port=4173,
            cleanup_command=(
                "(fuser -k 4173/tcp 2>/dev/null || true); "
                "for pid in $(lsof -ti tcp:4173 2>/dev/null || true); do "
                'kill -TERM "$pid" 2>/dev/null || true; '
                "done; "
                'pkill -f "[v]ite.*--port 4173" 2>/dev/null || true; '
                'pkill -f "pnpm.*[d]ev.*--port 4173" 2>/dev/null || true; '
                "sleep 1"
            ),
        ),
    ),
}


def fixture_policy(fixture_name: str) -> FixturePolicy | None:
    return _FIXTURE_POLICIES.get(fixture_name)


def fixture_is_build_command(fixture_name: str, command: str) -> bool:
    policy = fixture_policy(fixture_name)
    if policy is None:
        return _looks_like_generic_build_command(command)
    return policy.is_build_command(command)


def fixture_should_checkpoint_after_command(
    fixture_name: str, command: str, *, preview_published: bool
) -> bool:
    policy = fixture_policy(fixture_name)
    if policy is None:
        return preview_published or _looks_like_generic_build_command(command)
    return policy.should_checkpoint_after_command(
        command,
        preview_published=preview_published,
    )


def fixture_preview_port(fixture_name: str, default: int) -> int:
    policy = fixture_policy(fixture_name)
    return policy.preview_port(default) if policy is not None else default


def fixture_suppresses_command_preview_detection(fixture_name: str) -> bool:
    policy = fixture_policy(fixture_name)
    return policy is not None and policy.suppresses_command_preview_detection()


def fixture_guardrails(fixture_name: str) -> str | None:
    policy = fixture_policy(fixture_name)
    return policy.guardrails if policy is not None else None


def fixture_requires_preview(fixture_name: str) -> bool:
    policy = fixture_policy(fixture_name)
    return policy is not None and policy.preview is not None


def _normalize_command(command: str) -> str:
    return " ".join(command.split())


def _looks_like_generic_build_command(command: str) -> bool:
    normalized = _normalize_command(command).lower()
    return "pnpm build" in normalized or "npm run build" in normalized
