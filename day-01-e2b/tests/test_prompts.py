from __future__ import annotations

from agent_black_box.prompts import build_user_prompt


def test_build_user_prompt_hides_sample_frontend_guardrails_from_visible_task() -> None:
    prompt = build_user_prompt(
        workspace_dir="/tmp/workspace",
        fixture_name="sample_frontend_task",
        task="Create a simple portfolio page for Alex.",
    )

    assert "User request:\nCreate a simple portfolio page for Alex." in prompt
    assert "Fixture guardrails:" in prompt
    assert "Keep edits inside index.html, index.css, and index.js" in prompt
    assert "harness publishes the preview after the model finishes" in prompt
    assert "pnpm dev --host 0.0.0.0 --port 4173" not in prompt


def test_build_user_prompt_leaves_unknown_fixtures_without_guardrails() -> None:
    prompt = build_user_prompt(
        workspace_dir="/tmp/workspace",
        fixture_name="unknown_fixture",
        task="Do the thing.",
    )

    assert (
        prompt == "You are working in /tmp/workspace.\n\nUser request:\nDo the thing."
    )
