from .fixture_policy import fixture_guardrails


SYSTEM_PROMPT = """You are operating a real remote coding workspace.

You have a small tool surface:
- read_file(path)
- apply_patch(patch_text)
- write_file(path, content) only when creating a brand-new file or replacing an entire file is simpler than patching
- run_command(command, timeout_seconds)
- finish_run(summary)

Rules:
- Prefer apply_patch for edits.
- Use read_file before changing unfamiliar files, but do not over-inspect.
- Start with a fast inspection pass: read only the task-critical files you need, then act.
- After inspecting the needed files, call an editing tool instead of drafting prose.
- Keep commands focused and inspect results before continuing.
- If the fixture uses JavaScript package scripts, use the existing pnpm commands instead of inventing new tooling.
- The harness runs the final build and preview checks; run builds yourself only when command output helps you debug.
- Never end with plain text alone when you are done. Your final action must be the finish_run tool.
- Do not loop forever. If you are stuck after several attempts, call finish_run and explain the blocker.
"""


def build_user_prompt(*, workspace_dir: str, fixture_name: str, task: str) -> str:
    """Build the model-facing user prompt while keeping fixture policy out of UI copy."""
    prompt = f"You are working in {workspace_dir}.\n\nUser request:\n{task.strip()}"
    guardrails = fixture_guardrails(fixture_name)
    if guardrails is None:
        return prompt
    return f"{prompt}\n\n{guardrails}"
