from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_black_box.cli import app
from agent_black_box.config import get_settings
from agent_black_box.recorder import (
    RunMetadata,
    RunStatus,
    RunSummary,
    prepare_run_directory,
)


runner = CliRunner()


def test_show_run_rejects_summary_symlink_outside_run_root(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "runs"
    run_root.mkdir()
    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="succeeded", current_model_name="kimi-k2.6:cloud"
        ),
    )
    outside_summary = tmp_path / "outside-summary.json"
    outside_summary.write_text(
        RunSummary(
            run_id="outside-run",
            status="succeeded",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-outside",
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    summary_path = run_dir / "summary.json"
    summary_path.symlink_to(outside_summary)

    monkeypatch.setenv("RUN_ROOT", str(run_root))
    monkeypatch.setenv("FIXTURE_ROOT", str(tmp_path / "fixtures"))
    monkeypatch.setenv("E2B_API_KEY", "e2b_test")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama_test")
    get_settings.cache_clear()

    result = runner.invoke(app, ["show-run", "--run-id", "run-1"])

    assert result.exit_code == 2
    assert "Run summary not found" in result.output


def test_show_run_timing_reports_completed_and_open_model_turns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "runs"
    run_dir = prepare_run_directory(
        run_root,
        "run-1",
        RunMetadata(
            run_id="run-1",
            task="task",
            model_name="kimi-k2.6:cloud",
            fixture_name="fixture",
            sandbox_id="sbx-1",
        ),
        RunStatus(
            run_id="run-1", state="running", current_model_name="kimi-k2.6:cloud"
        ),
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(
            [
                '{"type":"model_turn_started","run_id":"run-1","lane_id":"lane-1","sequence":1,"timestamp":"2026-04-22T00:00:00Z","turn_number":1,"model_name":"kimi-k2.6:cloud","message_count":4,"tool_schema_count":5,"conversation_chars":1000,"last_message_chars":200,"last_tool_result_chars":200,"request_body_bytes":1400}',
                '{"type":"model_provider_attempt_started","run_id":"run-1","lane_id":"lane-1","sequence":2,"timestamp":"2026-04-22T00:00:00.100000Z","turn_number":1,"phase":"list_models","model_name":null,"attempt_number":1,"timeout_seconds":7.0,"is_fallback":false}',
                '{"type":"model_provider_attempt_completed","run_id":"run-1","lane_id":"lane-1","sequence":3,"timestamp":"2026-04-22T00:00:00.200000Z","turn_number":1,"phase":"list_models","model_name":null,"attempt_number":1,"timeout_seconds":7.0,"is_fallback":false,"duration_seconds":0.1,"outcome":"success","status_code":null,"error_type":null,"error_message":null}',
                '{"type":"model_provider_attempt_started","run_id":"run-1","lane_id":"lane-1","sequence":4,"timestamp":"2026-04-22T00:00:00.300000Z","turn_number":1,"phase":"chat","model_name":"kimi-k2.6","attempt_number":1,"timeout_seconds":11.0,"is_fallback":false}',
                '{"type":"model_provider_attempt_completed","run_id":"run-1","lane_id":"lane-1","sequence":5,"timestamp":"2026-04-22T00:00:03.400000Z","turn_number":1,"phase":"chat","model_name":"kimi-k2.6","attempt_number":1,"timeout_seconds":11.0,"is_fallback":false,"duration_seconds":3.1,"outcome":"success","status_code":null,"error_type":null,"error_message":null,"ollama_total_duration_seconds":2.9,"ollama_load_duration_seconds":0.1,"ollama_prompt_eval_count":42,"ollama_prompt_eval_duration_seconds":0.2,"ollama_eval_count":12,"ollama_eval_duration_seconds":2.4}',
                "not json",
                '{"type":"model_turn_completed","run_id":"run-1","lane_id":"lane-1","sequence":6,"timestamp":"2026-04-22T00:00:03.500000Z","turn_number":1,"finish_reason":"tool_call","content":null,"tool_name":"read_file"}',
                '{"type":"model_turn_started","run_id":"run-1","lane_id":"lane-1","sequence":7,"timestamp":"2026-04-22T00:00:04Z","turn_number":2,"model_name":"kimi-k2.6:cloud","message_count":6,"tool_schema_count":5,"conversation_chars":1500,"last_message_chars":300,"last_tool_result_chars":300,"request_body_bytes":2100}',
                '{"type":"model_provider_attempt_started","run_id":"run-1","lane_id":"lane-1","sequence":8,"timestamp":"2026-04-22T00:00:04.100000Z","turn_number":2,"phase":"chat","model_name":"kimi-k2.6","attempt_number":1,"timeout_seconds":11.0,"is_fallback":false}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("RUN_ROOT", str(run_root))
    monkeypatch.setenv("FIXTURE_ROOT", str(tmp_path / "fixtures"))
    monkeypatch.setenv("E2B_API_KEY", "e2b_test")
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama_test")
    get_settings.cache_clear()

    result = runner.invoke(app, ["show-run-timing", "--run-id", "run-1"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["last_event_type"] == "model_provider_attempt_started"
    assert payload["model_turns"][0]["turn_number"] == 1
    assert payload["model_turns"][0]["duration_seconds"] == 3.5
    assert payload["model_turns"][0]["state"] == "completed"
    assert payload["model_turns"][0]["model_name"] == "kimi-k2.6:cloud"
    assert payload["model_turns"][0]["request_body_bytes"] == 1400
    assert payload["model_turns"][0]["provider_attempts"][0]["phase"] == "list_models"
    assert payload["model_turns"][0]["provider_attempts"][0]["outcome"] == "success"
    assert payload["model_turns"][0]["provider_attempts"][1]["phase"] == "chat"
    assert payload["model_turns"][0]["provider_attempts"][1]["duration_seconds"] == 3.1
    assert (
        payload["model_turns"][0]["provider_attempts"][1][
            "ollama_total_duration_seconds"
        ]
        == 2.9
    )
    assert payload["model_turns"][0]["provider_attempts"][1]["ollama_eval_count"] == 12
    assert payload["model_turns"][1]["turn_number"] == 2
    assert payload["model_turns"][1]["state"] == "open"
    assert payload["model_turns"][1]["duration_seconds"] is not None
    assert payload["model_turns"][1]["provider_attempts"][0]["state"] == "open"
