from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REQUEST_PATH = Path("contracts/examples/sample_request.json")
SAMPLE_REPORT_PATH = Path("contracts/examples/sample_report.json")
SAMPLE_SOURCE_FETCH_PATH = Path("contracts/examples/sample_source_fetch.json")


def load_json_contract(
    relative_path: str | Path,
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, Any]:
    path = project_root / relative_path
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Contract JSON must be an object: {path}")
    return payload


def load_sample_request(*, project_root: Path = DEFAULT_PROJECT_ROOT) -> dict[str, Any]:
    return load_json_contract(SAMPLE_REQUEST_PATH, project_root=project_root)


def load_sample_report(*, project_root: Path = DEFAULT_PROJECT_ROOT) -> dict[str, Any]:
    return load_json_contract(SAMPLE_REPORT_PATH, project_root=project_root)


def load_sample_source_fetch(
    *,
    project_root: Path = DEFAULT_PROJECT_ROOT,
) -> dict[str, Any]:
    return load_json_contract(SAMPLE_SOURCE_FETCH_PATH, project_root=project_root)


def contracts_readable(*, project_root: Path = DEFAULT_PROJECT_ROOT) -> bool:
    try:
        load_sample_request(project_root=project_root)
        load_sample_report(project_root=project_root)
        load_sample_source_fetch(project_root=project_root)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return False
    return True
