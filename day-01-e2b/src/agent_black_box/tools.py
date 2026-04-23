from __future__ import annotations

from typing import Any


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a text file from the remote workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path relative to the workspace root."}
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": "Apply a unified diff patch inside the remote workspace. Prefer this for normal edits.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patch_text": {"type": "string", "description": "Unified diff patch text."}
                    },
                    "required": ["patch_text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a full file when creating a new file or replacing the entire contents is simpler than patching.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Run a shell command in the remote workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout_seconds": {"type": "integer", "minimum": 1, "default": 120},
                    },
                    "required": ["command", "timeout_seconds"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish_run",
                "description": "Finish the run with a concise summary of what succeeded or what remains broken.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                    },
                    "required": ["summary"],
                },
            },
        },
    ]
