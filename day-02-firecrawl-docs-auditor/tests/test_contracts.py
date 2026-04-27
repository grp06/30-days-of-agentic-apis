from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from firecrawl_docs_auditor.contracts import (
    contracts_readable,
    load_sample_report,
    load_sample_request,
    load_sample_source_fetch,
)
from firecrawl_docs_auditor.firecrawl_ingestion import FirecrawlFetchResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def assert_valid_audit_report(report: dict[str, object]) -> None:
    request_schema = json.loads(
        (PROJECT_ROOT / "contracts" / "audit_request.schema.json").read_text(
            encoding="utf-8"
        )
    )
    report_schema = json.loads(
        (PROJECT_ROOT / "contracts" / "audit_report.schema.json").read_text(
            encoding="utf-8"
        )
    )
    registry = Registry().with_resources(
        [("audit-request.schema.json", Resource.from_contents(request_schema))]
    )
    Draft202012Validator(report_schema, registry=registry).validate(report)


def test_contract_fixtures_load() -> None:
    request = load_sample_request()
    report = load_sample_report()
    source_fetch = load_sample_source_fetch()

    assert request["mode"] in {"live", "cached"}
    assert report["metadata"]["generated_by"] == "contract_fixture"
    assert {"scorecard", "selected_sources", "warnings", "suggested_fixes"} <= set(report)
    assert source_fetch["cache_key"] == request["cache_key"]
    assert FirecrawlFetchResult.model_validate(source_fetch).cache_key == request["cache_key"]
    assert_valid_audit_report(report)


def test_audit_request_contract_accepts_50_pages_but_rejects_51() -> None:
    schema = json.loads(
        (PROJECT_ROOT / "contracts" / "audit_request.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = Draft202012Validator(schema)
    valid = {
        "docs_url": "https://docs.example.com/docs/start",
        "integration_goal": "Build checkout",
        "mode": "live",
        "max_pages": 50,
    }
    invalid = {**valid, "max_pages": 51}

    validator.validate(valid)
    assert list(validator.iter_errors(invalid))


def test_contracts_readable_reports_missing_root(tmp_path) -> None:
    assert contracts_readable(project_root=tmp_path) is False


def test_sample_request_has_no_credential_shaped_keys() -> None:
    payloads = [
        load_sample_request(),
        load_sample_source_fetch(),
        load_sample_report(),
    ]
    serialized = json.dumps(payloads).lower()

    _assert_no_forbidden_keys(payloads)
    forbidden = [
        "api_key",
        "firecrawl_api_key",
        "token",
        "auth.json",
        "cookie",
        "secret",
        "fetched_sources",
        "/users/",
    ]
    assert not any(word in serialized for word in forbidden)


def _assert_no_forbidden_keys(payload: object) -> None:
    forbidden = {
        "api_key",
        "firecrawl_api_key",
        "token",
        "authorization",
        "cookie",
        "secret",
        "markdown",
        "fetched_sources",
    }
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert str(key).lower() not in forbidden
            _assert_no_forbidden_keys(value)
    elif isinstance(payload, list):
        for item in payload:
            _assert_no_forbidden_keys(item)
