"""Tests for workflow.escalation."""
from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from workflow.escalation import (
    EscalationRecord,
    EscalationRouter,
    QueueBackend,
    WebhookBackend,
)


# --------------------------- helpers ---------------------------------------


class _RecordingBackend:
    """In-memory backend that just records what it was sent."""

    def __init__(self) -> None:
        self.records: List[EscalationRecord] = []

    def send(self, record: EscalationRecord) -> None:
        self.records.append(record)


def _fake_response(status: int) -> MagicMock:
    """Build a context-manager-shaped fake response with the given HTTP status."""
    resp = MagicMock()
    resp.status = status
    resp.getcode.return_value = status
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# --------------------------- QueueBackend ----------------------------------


def test_queue_backend_appends_jsonl(tmp_path: Path) -> None:
    backend = QueueBackend(tmp_path)
    rec1 = EscalationRecord.new("dec-1", "BORDERLINE", "MEDIUM", 24, {"k": 1})
    rec2 = EscalationRecord.new("dec-2", "HIGH_VALUE_LOAN", "HIGH", 8, {"loan": 1_000_000})
    backend.send(rec1)
    backend.send(rec2)

    files = list(tmp_path.glob("escalations-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    parsed_first = json.loads(lines[0])
    parsed_second = json.loads(lines[1])
    assert parsed_first["decision_id"] == "dec-1"
    assert parsed_first["priority"] == "MEDIUM"
    assert parsed_first["suggested_sla_hours"] == 24
    assert parsed_second["decision_id"] == "dec-2"
    assert parsed_second["context"]["loan"] == 1_000_000


def test_queue_backend_creates_root_directory(tmp_path: Path) -> None:
    nested = tmp_path / "does" / "not" / "exist"
    backend = QueueBackend(nested)
    backend.send(EscalationRecord.new("d", "BORDERLINE", "MEDIUM", 24))
    assert nested.exists()
    assert any(nested.glob("escalations-*.jsonl"))


# --------------------------- WebhookBackend --------------------------------


def test_webhook_retries_on_503_then_succeeds() -> None:
    sleeps: List[float] = []
    backend = WebhookBackend(
        "https://example.test/hook",
        max_retries=3,
        sleep=lambda s: sleeps.append(s),
    )
    rec = EscalationRecord.new("dec-x", "BORDERLINE", "MEDIUM", 24)

    responses = [_fake_response(503), _fake_response(503), _fake_response(200)]
    with patch("urllib.request.urlopen", side_effect=responses) as mock_open:
        backend.send(rec)

    assert mock_open.call_count == 3
    # Two backoff sleeps between three attempts.
    assert sleeps == [1.0, 2.0]


def test_webhook_raises_after_exhausting_retries() -> None:
    sleeps: List[float] = []
    backend = WebhookBackend(
        "https://example.test/hook",
        max_retries=3,
        sleep=lambda s: sleeps.append(s),
    )
    rec = EscalationRecord.new("dec-x", "BORDERLINE", "MEDIUM", 24)

    responses = [_fake_response(503), _fake_response(503), _fake_response(503)]
    with patch("urllib.request.urlopen", side_effect=responses) as mock_open:
        with pytest.raises(urllib.error.HTTPError):
            backend.send(rec)

    assert mock_open.call_count == 3
    assert sleeps == [1.0, 2.0]


def test_webhook_retries_on_timeout_then_raises() -> None:
    sleeps: List[float] = []
    backend = WebhookBackend(
        "https://example.test/hook",
        max_retries=3,
        sleep=lambda s: sleeps.append(s),
    )
    rec = EscalationRecord.new("dec-x", "BORDERLINE", "MEDIUM", 24)

    with patch("urllib.request.urlopen", side_effect=TimeoutError("slow")):
        with pytest.raises(TimeoutError):
            backend.send(rec)

    assert sleeps == [1.0, 2.0]


def test_webhook_does_not_retry_on_4xx() -> None:
    sleeps: List[float] = []
    backend = WebhookBackend(
        "https://example.test/hook",
        max_retries=3,
        sleep=lambda s: sleeps.append(s),
    )
    rec = EscalationRecord.new("dec-x", "BORDERLINE", "MEDIUM", 24)
    err = urllib.error.HTTPError("u", 404, "not found", hdrs=None, fp=None)  # type: ignore[arg-type]

    with patch("urllib.request.urlopen", side_effect=err) as mock_open:
        with pytest.raises(urllib.error.HTTPError):
            backend.send(rec)

    assert mock_open.call_count == 1
    assert sleeps == []


# --------------------------- EscalationRouter ------------------------------


def test_router_fires_borderline_on_flag() -> None:
    backend = _RecordingBackend()
    router = EscalationRouter(backend)
    trigger = router.should_escalate(flags={"BORDERLINE"})
    assert trigger == "BORDERLINE"


def test_router_does_not_fire_on_clean_approve() -> None:
    backend = _RecordingBackend()
    router = EscalationRouter(backend)
    assert router.should_escalate(
        flags=set(),
        probability=0.10,
        loan_amount=10_000,
        confidence_threshold=0.05,
        high_value_threshold=500_000,
    ) is None


def test_router_high_priority_for_protected_attr() -> None:
    backend = _RecordingBackend()
    router = EscalationRouter(backend)
    rec = router.route(
        decision_id="dec-7",
        reason="PROTECTED_ATTR_DETECTED",
        flags={"PROTECTED_ATTR_DETECTED"},
        context={"hits": ["race"]},
    )
    assert rec.priority == "HIGH"
    assert rec.suggested_sla_hours == 4
    assert rec.reason == "PROTECTED_ATTR_DETECTED"


def test_router_high_value_loan_trigger_fires() -> None:
    backend = _RecordingBackend()
    router = EscalationRouter(backend)
    trigger = router.should_escalate(loan_amount=750_000, high_value_threshold=500_000)
    assert trigger == "HIGH_VALUE_LOAN"


def test_router_high_value_loan_below_threshold_no_fire() -> None:
    backend = _RecordingBackend()
    router = EscalationRouter(backend)
    trigger = router.should_escalate(loan_amount=400_000, high_value_threshold=500_000)
    assert trigger is None


def test_router_low_confidence_trigger_fires() -> None:
    backend = _RecordingBackend()
    router = EscalationRouter(backend)
    trigger = router.should_escalate(probability=0.51, confidence_threshold=0.05)
    assert trigger == "LOW_CONFIDENCE"


def test_route_writes_one_record_via_backend() -> None:
    backend = _RecordingBackend()
    router = EscalationRouter(backend)
    rec = router.route(
        decision_id="dec-9",
        reason="BORDERLINE",
        flags={"BORDERLINE"},
        context={"probability": 0.49},
    )
    assert len(backend.records) == 1
    assert backend.records[0] is rec
    assert rec.priority == "MEDIUM"
    assert rec.suggested_sla_hours == 24
    assert rec.context["probability"] == 0.49
    assert rec.context["flags"] == ["BORDERLINE"]


def test_route_persists_through_queue_backend(tmp_path: Path) -> None:
    backend = QueueBackend(tmp_path)
    router = EscalationRouter(backend)
    router.route(
        decision_id="dec-10",
        reason="HIGH_VALUE_LOAN",
        flags={"HIGH_VALUE_LOAN"},
        context={"loan_amount": 900_000},
    )
    files = list(tmp_path.glob("escalations-*.jsonl"))
    assert len(files) == 1
    parsed = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert parsed["decision_id"] == "dec-10"
    assert parsed["priority"] == "HIGH"
    assert parsed["suggested_sla_hours"] == 8


def test_router_first_match_wins_when_multiple_triggers_apply() -> None:
    backend = _RecordingBackend()
    router = EscalationRouter(backend)
    # BORDERLINE comes before HIGH_VALUE_LOAN in DEFAULT_TRIGGERS insertion order.
    trigger = router.should_escalate(
        flags={"BORDERLINE"},
        loan_amount=900_000,
        high_value_threshold=500_000,
    )
    assert trigger == "BORDERLINE"


def test_router_custom_triggers_override_defaults() -> None:
    backend = _RecordingBackend()
    router = EscalationRouter(
        backend,
        triggers={"MANUAL_REVIEW": "LOW"},
        slas={"MANUAL_REVIEW": 72},
    )
    assert router.should_escalate(flags={"MANUAL_REVIEW"}) == "MANUAL_REVIEW"
    rec = router.route("dec-z", "MANUAL_REVIEW", flags={"MANUAL_REVIEW"}, context={})
    assert rec.priority == "LOW"
    assert rec.suggested_sla_hours == 72
