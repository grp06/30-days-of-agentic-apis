from __future__ import annotations

from agent_black_box.fixture_policy import (
    fixture_is_build_command,
    fixture_preview_port,
    fixture_should_checkpoint_after_command,
    fixture_suppresses_command_preview_detection,
)


def test_sample_fixture_policy_answers_runtime_questions() -> None:
    assert fixture_is_build_command("sample_frontend_task", "pnpm build") is True
    assert fixture_is_build_command("sample_frontend_task", "echo hello") is False
    assert fixture_preview_port("sample_frontend_task", default=5173) == 4173
    assert fixture_suppresses_command_preview_detection("sample_frontend_task") is True
    assert (
        fixture_should_checkpoint_after_command(
            "sample_frontend_task",
            "pnpm build",
            preview_published=False,
        )
        is True
    )
    assert (
        fixture_should_checkpoint_after_command(
            "sample_frontend_task",
            "echo hello",
            preview_published=True,
        )
        is True
    )
    assert (
        fixture_should_checkpoint_after_command(
            "sample_frontend_task",
            "echo hello",
            preview_published=False,
        )
        is False
    )


def test_unknown_fixture_keeps_generic_build_defaults() -> None:
    assert fixture_is_build_command("unknown_task", "pnpm build") is True
    assert fixture_is_build_command("unknown_task", "npm run build") is True
    assert fixture_is_build_command("unknown_task", "echo hello") is False
    assert fixture_preview_port("unknown_task", default=5173) == 5173
    assert fixture_suppresses_command_preview_detection("unknown_task") is False
    assert (
        fixture_should_checkpoint_after_command(
            "unknown_task",
            "pnpm build",
            preview_published=False,
        )
        is True
    )
    assert (
        fixture_should_checkpoint_after_command(
            "unknown_task",
            "echo hello",
            preview_published=True,
        )
        is True
    )
    assert (
        fixture_should_checkpoint_after_command(
            "unknown_task",
            "echo hello",
            preview_published=False,
        )
        is False
    )
