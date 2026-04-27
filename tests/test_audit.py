"""Tests for workflow.audit."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from workflow.audit import (
    AuditLogger,
    AuditRecord,
    JSONLBackend,
    MultiBackend,
    StdoutBackend,
)


def _sample_record(**overrides: Any) -> AuditRecord:
    base: Dict[str, Any] = dict(
        model_version="1.0.0",
        applicant_features={"LIMIT_BAL": 200000, "PAY_0": 0},
        probability=0.42,
        decision="APPROVE",
        shap_factors=[{"feature": "PAY_0", "shap_value": 0.12, "direction": "increases risk"}],
        llm_prompt="prompt body",
        llm_response="memo body",
        safety_results={"protected_attr_hits": []},
        final_output={"memo": "memo body", "adverse_action": None},
        processing_time_ms=512.5,
    )
    base.update(overrides)
    return AuditRecord.new(**base)


def test_record_autofills_id_and_timestamp() -> None:
    rec = _sample_record()
    assert rec.decision_id and len(rec.decision_id) == 36
    assert rec.timestamp_utc.endswith("+00:00") or "T" in rec.timestamp_utc


def test_record_is_immutable() -> None:
    rec = _sample_record()
    with pytest.raises(Exception):
        rec.probability = 0.99  # type: ignore[misc]


def test_jsonl_backend_writes_valid_json(tmp_path: Path) -> None:
    backend = JSONLBackend(tmp_path)
    logger = AuditLogger(backend)
    rec = _sample_record()
    logger.log(rec)
    files = list(tmp_path.glob("audit-*.jsonl"))
    assert len(files) == 1
    line = files[0].read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert parsed["decision_id"] == rec.decision_id
    assert parsed["model_version"] == "1.0.0"
    assert parsed["applicant_features"]["LIMIT_BAL"] == 200000


def test_jsonl_backend_appends(tmp_path: Path) -> None:
    backend = JSONLBackend(tmp_path)
    logger = AuditLogger(backend)
    logger.log(_sample_record())
    logger.log(_sample_record())
    files = list(tmp_path.glob("audit-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_stdout_backend_emits_one_line() -> None:
    buf = io.StringIO()
    backend = StdoutBackend(stream=buf)
    AuditLogger(backend).log(_sample_record())
    out = buf.getvalue()
    assert out.endswith("\n")
    parsed = json.loads(out.strip())
    assert parsed["decision"] == "APPROVE"


def test_multi_backend_fans_out(tmp_path: Path) -> None:
    buf = io.StringIO()
    multi = MultiBackend(JSONLBackend(tmp_path), StdoutBackend(stream=buf))
    AuditLogger(multi).log(_sample_record())
    assert buf.getvalue().strip()
    assert any(tmp_path.glob("audit-*.jsonl"))


def test_logger_rejects_record_missing_field() -> None:
    """Hand-construct a malformed record by skipping field validation."""
    class Broken:
        decision_id = "x"  # intentionally missing most fields

    with pytest.raises(ValueError):
        AuditLogger(StdoutBackend(stream=io.StringIO())).log(Broken())  # type: ignore[arg-type]


def test_no_pii_appears_when_features_pre_scrubbed() -> None:
    """Audit logger faithfully records what the workflow sends it.

    Phase-2 PII scrubbing happens upstream — this test verifies that when the
    workflow passes already-scrubbed features, the audit log preserves them.
    """
    rec = _sample_record(
        applicant_features={"LIMIT_BAL": 100000, "AGE": 30},
        scrubbed_fields=["name", "ssn", "address"],
    )
    buf = io.StringIO()
    AuditLogger(StdoutBackend(stream=buf)).log(rec)
    parsed = json.loads(buf.getvalue().strip())
    assert "name" not in parsed["applicant_features"]
    assert "ssn" not in parsed["applicant_features"]
    assert parsed["scrubbed_fields"] == ["name", "ssn", "address"]
