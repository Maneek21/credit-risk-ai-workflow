"""Escalation routing for credit decisions that warrant human review.

Some decisions cannot be auto-resolved by the model + LLM pipeline alone:
borderline probabilities, hits on protected-attribute detectors, and
high-value loans all require a human in the loop. This module routes those
decisions to an escalation backend (a queue file or a webhook to a ticketing
system) with a priority and an SLA.

Two storage backends:
  - ``QueueBackend``   - append-only ``.jsonl`` files, one per UTC date
  - ``WebhookBackend`` - POSTs JSON to a URL with retry-with-backoff

Records are append-only - there is no public update or delete API.

Example
-------
    from workflow.escalation import EscalationRouter, QueueBackend
    router = EscalationRouter(QueueBackend("./escalations"))
    if router.should_escalate(flags=result.flags, probability=result.probability):
        router.route(result.decision_id, "borderline", result.flags, {...})
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol


DEFAULT_TRIGGERS: Dict[str, str] = {
    "BORDERLINE": "MEDIUM",
    "PROTECTED_ATTR_DETECTED": "HIGH",
    "LOW_CONFIDENCE": "MEDIUM",
    "HIGH_VALUE_LOAN": "HIGH",
}

DEFAULT_SLAS: Dict[str, int] = {
    "BORDERLINE": 24,
    "PROTECTED_ATTR_DETECTED": 4,
    "LOW_CONFIDENCE": 24,
    "HIGH_VALUE_LOAN": 8,
}


@dataclass(frozen=True)
class EscalationRecord:
    """Immutable escalation record sent to a backend.

    Attributes:
        decision_id: UUID of the originating credit decision.
        reason: Trigger name (e.g. ``"BORDERLINE"``) or human-readable cause.
        priority: One of ``"HIGH"``, ``"MEDIUM"``, ``"LOW"``.
        suggested_sla_hours: Wall-clock hours by which review should complete.
        context: Free-form context dict (probability, loan amount, flags, ...).
        created_utc: ISO-8601 UTC timestamp; auto-filled by ``new``.
    """

    decision_id: str
    reason: str
    priority: str
    suggested_sla_hours: int
    context: Dict[str, Any]
    created_utc: str

    @classmethod
    def new(
        cls,
        decision_id: str,
        reason: str,
        priority: str,
        suggested_sla_hours: int,
        context: Optional[Dict[str, Any]] = None,
        created_utc: Optional[str] = None,
    ) -> "EscalationRecord":
        """Build a record, auto-filling ``created_utc`` if not supplied.

        Args:
            decision_id: UUID of the originating credit decision.
            reason: Trigger name or human-readable cause.
            priority: ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"``.
            suggested_sla_hours: SLA in hours.
            context: Free-form context dict.
            created_utc: ISO-8601 UTC timestamp; defaults to now.

        Returns:
            A new ``EscalationRecord``.
        """
        return cls(
            decision_id=decision_id,
            reason=reason,
            priority=priority,
            suggested_sla_hours=suggested_sla_hours,
            context=dict(context or {}),
            created_utc=created_utc or datetime.now(timezone.utc).isoformat(),
        )


class EscalationBackend(Protocol):
    """Interface every escalation backend must implement."""

    def send(self, record: EscalationRecord) -> None: ...


class QueueBackend:
    """Append-only JSON Lines backend, one file per UTC date.

    File layout: ``<root>/escalations-YYYY-MM-DD.jsonl``. The handle is opened
    in append mode and flushed (+ fsync) after every write so a crash never
    loses the just-completed record. Thread-safe via an internal lock.
    """

    def __init__(self, root: str | os.PathLike[str] = "./escalations") -> None:
        """Initialise.

        Args:
            root: Directory to write JSONL files into. Created if missing.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, ts_iso: str) -> Path:
        """Resolve the JSONL path for a given ISO timestamp.

        Args:
            ts_iso: ISO-8601 timestamp string. The first 10 chars are used
                as the YYYY-MM-DD bucket key.

        Returns:
            Path to the JSONL file for that date.
        """
        date = ts_iso[:10]
        return self.root / f"escalations-{date}.jsonl"

    def send(self, record: EscalationRecord) -> None:
        """Append one record. Raises any I/O error - callers must not swallow.

        Args:
            record: Escalation record to persist.
        """
        line = json.dumps(asdict(record), default=str, ensure_ascii=False)
        path = self._path_for(record.created_utc)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())


class WebhookBackend:
    """POST escalation records to a URL with retry-on-failure.

    Retries up to ``max_retries`` times on HTTP 5xx and request timeouts using
    exponential backoff (1s, 2s, 4s, ...). After the final attempt fails, the
    underlying error is raised so the caller can react.
    """

    def __init__(
        self,
        url: str,
        timeout: float = 10.0,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        sleep: Any = time.sleep,
    ) -> None:
        """Initialise.

        Args:
            url: Webhook endpoint to POST records to.
            timeout: Per-request timeout in seconds.
            max_retries: Total attempts before giving up (default 3).
            backoff_base: Base for exponential backoff in seconds; sleeps
                ``backoff_base * 2**attempt`` between attempts.
            sleep: Sleep function (override for tests to avoid real delays).
        """
        self.url = url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._sleep = sleep

    def _build_request(self, record: EscalationRecord) -> urllib.request.Request:
        """Build the POST request body and headers.

        Args:
            record: Record to serialise.

        Returns:
            A configured ``urllib.request.Request``.
        """
        body = json.dumps(asdict(record), default=str, ensure_ascii=False).encode("utf-8")
        return urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

    def send(self, record: EscalationRecord) -> None:
        """POST the record, retrying on 5xx and timeouts.

        Args:
            record: Escalation record to deliver.

        Raises:
            urllib.error.HTTPError: After exhausting retries on a 5xx.
            urllib.error.URLError: After exhausting retries on a network error.
            TimeoutError: After exhausting retries on a timeout.
        """
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                req = self._build_request(record)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = getattr(resp, "status", None)
                    if status is None:
                        status = resp.getcode()
                    if status is not None and 500 <= int(status) < 600:
                        last_error = urllib.error.HTTPError(
                            self.url, int(status), f"HTTP {status}", hdrs=None, fp=None
                        )
                    else:
                        return
            except urllib.error.HTTPError as e:
                if 500 <= e.code < 600:
                    last_error = e
                else:
                    raise
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_error = e

            if attempt < self.max_retries - 1:
                self._sleep(self.backoff_base * (2 ** attempt))

        assert last_error is not None
        raise last_error


class EscalationRouter:
    """Decide whether a decision should escalate, and route it if so.

    Triggers are evaluated in the order they appear in the ``triggers`` dict.
    The first matching trigger wins, and its priority + SLA are stamped onto
    the record.
    """

    def __init__(
        self,
        backend: EscalationBackend,
        triggers: Optional[Dict[str, str]] = None,
        slas: Optional[Dict[str, int]] = None,
    ) -> None:
        """Initialise.

        Args:
            backend: Where to send escalation records.
            triggers: Map of trigger name to priority. Defaults to
                ``DEFAULT_TRIGGERS``.
            slas: Map of trigger name to SLA in hours. Defaults to
                ``DEFAULT_SLAS``. Trigger names not in this map fall back
                to a priority-keyed default (HIGH=4, MEDIUM=24, LOW=72).
        """
        self.backend = backend
        self.triggers = dict(triggers) if triggers is not None else dict(DEFAULT_TRIGGERS)
        self.slas = dict(slas) if slas is not None else dict(DEFAULT_SLAS)

    def _sla_for(self, trigger: str, priority: str) -> int:
        """Resolve the SLA hours for a given trigger.

        Args:
            trigger: Trigger name (key in the ``triggers`` dict).
            priority: Trigger priority - used as a fallback if the trigger
                itself is not in the ``slas`` map.

        Returns:
            SLA in hours.
        """
        if trigger in self.slas:
            return self.slas[trigger]
        # Priority-keyed defaults if the slas dict is keyed by priority instead
        # of trigger - supports the example signature in the spec.
        if priority in self.slas:
            return self.slas[priority]
        return {"HIGH": 4, "MEDIUM": 24, "LOW": 72}.get(priority, 24)

    def should_escalate(
        self,
        flags: Optional[Iterable[str]] = None,
        probability: Optional[float] = None,
        loan_amount: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        high_value_threshold: Optional[float] = None,
    ) -> Optional[str]:
        """Return the first matching trigger name, or ``None``.

        Triggers are checked in dict-insertion order. ``BORDERLINE`` and
        ``PROTECTED_ATTR_DETECTED`` consult the ``flags`` set; ``LOW_CONFIDENCE``
        consults ``probability`` and ``confidence_threshold``; ``HIGH_VALUE_LOAN``
        consults ``loan_amount`` and ``high_value_threshold``.

        Args:
            flags: Iterable of flag names raised during pipeline execution.
            probability: Model default probability (0.0-1.0).
            loan_amount: Requested loan amount (numeric).
            confidence_threshold: Distance from 0.5 below which probability
                is considered low-confidence (e.g. 0.05 means 0.45-0.55).
            high_value_threshold: Loan amount above which the ``HIGH_VALUE_LOAN``
                trigger fires.

        Returns:
            Trigger name (e.g. ``"BORDERLINE"``) or ``None`` if no trigger
            applies.
        """
        flag_set = set(flags or [])
        for trigger in self.triggers:
            if trigger == "BORDERLINE" and "BORDERLINE" in flag_set:
                return trigger
            if trigger == "PROTECTED_ATTR_DETECTED" and "PROTECTED_ATTR_DETECTED" in flag_set:
                return trigger
            if trigger == "LOW_CONFIDENCE":
                if (
                    probability is not None
                    and confidence_threshold is not None
                    and abs(probability - 0.5) <= confidence_threshold
                ):
                    return trigger
            if trigger == "HIGH_VALUE_LOAN":
                if (
                    loan_amount is not None
                    and high_value_threshold is not None
                    and loan_amount > high_value_threshold
                ):
                    return trigger
            # Custom triggers: treat the trigger name as a flag name.
            if trigger not in DEFAULT_TRIGGERS and trigger in flag_set:
                return trigger
        return None

    def route(
        self,
        decision_id: str,
        reason: str,
        flags: Optional[Iterable[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> EscalationRecord:
        """Build a record and dispatch it to the backend.

        Priority and SLA are looked up from ``self.triggers`` / ``self.slas``
        using ``reason`` as the key. If the reason is not a known trigger,
        priority defaults to ``"MEDIUM"`` and SLA to 24 hours.

        Args:
            decision_id: UUID of the originating credit decision.
            reason: Trigger name (preferred) or free-form cause string.
            flags: Iterable of flags raised during pipeline execution; merged
                into the record's context as ``context["flags"]``.
            context: Additional free-form context.

        Returns:
            The dispatched ``EscalationRecord``.
        """
        priority = self.triggers.get(reason, "MEDIUM")
        sla = self._sla_for(reason, priority)
        merged: Dict[str, Any] = dict(context or {})
        merged.setdefault("flags", sorted(set(flags or [])))
        record = EscalationRecord.new(
            decision_id=decision_id,
            reason=reason,
            priority=priority,
            suggested_sla_hours=sla,
            context=merged,
        )
        self.backend.send(record)
        return record
