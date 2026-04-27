"""Audit trail for credit decisions.

Every call to ``CreditWorkflow.process_application`` produces one immutable
audit record. Examiners (OCC / Fed under SR 11-7 §IV) can use these records
to reconstruct the full decision chain for any applicant on any date.

Two storage backends:
  - ``JSONLBackend``  — append-only ``.jsonl`` files, one per UTC date
  - ``StdoutBackend`` — line-delimited JSON to stdout (forward to SIEM)

Records are append-only — there is no public update or delete API.

Example
-------
    from workflow.audit import AuditLogger, JSONLBackend
    logger = AuditLogger(JSONLBackend("./audit_logs"))
    logger.log({...record...})
"""
from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


REQUIRED_FIELDS = (
    "decision_id", "timestamp_utc", "model_version", "applicant_features",
    "probability", "decision", "shap_factors", "llm_prompt", "llm_response",
    "safety_results", "final_output", "processing_time_ms",
)


@dataclass(frozen=True)
class AuditRecord:
    """Immutable audit record for one credit decision.

    Field names match SR 11-7 §IV ("Outcomes Analysis") expectations: every
    decision should be reconstructable from inputs to delivered output.

    Attributes:
        decision_id: UUID4 identifier for the decision.
        timestamp_utc: ISO-8601 UTC timestamp.
        model_version: Identifier from the model registry (e.g. "1.0.0").
        applicant_features: Full feature dict that entered the model.
        probability: Raw default probability.
        decision: "APPROVE" or "DENY".
        shap_factors: Top-5 SHAP factors (feature, shap_value, direction).
        llm_prompt: Full prompt sent to the LLM (after PII scrubbing).
        llm_response: Full LLM completion.
        safety_results: Which safety layers fired and what they caught.
        final_output: Final memo + adverse-action text delivered downstream.
        processing_time_ms: End-to-end latency.
        scrubbed_fields: Names of PII fields removed before LLM call (if any).
    """

    decision_id: str
    timestamp_utc: str
    model_version: str
    applicant_features: Dict[str, Any]
    probability: float
    decision: str
    shap_factors: List[Dict[str, Any]]
    llm_prompt: str
    llm_response: str
    safety_results: Dict[str, Any]
    final_output: Dict[str, Any]
    processing_time_ms: float
    scrubbed_fields: List[str] = field(default_factory=list)

    @classmethod
    def new(cls, **kwargs: Any) -> "AuditRecord":
        """Build a record, auto-filling decision_id and timestamp_utc if absent."""
        kwargs.setdefault("decision_id", str(uuid.uuid4()))
        kwargs.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        return cls(**kwargs)


class AuditBackend(Protocol):
    """Interface every storage backend must implement."""

    def write(self, record: AuditRecord) -> None: ...


class JSONLBackend:
    """Append-only JSON Lines backend, one file per UTC date.

    File layout: ``<root>/audit-YYYY-MM-DD.jsonl``. The handle is opened in
    append mode and flushed after every write so a crash never loses the
    just-completed record.
    """

    def __init__(self, root: str | os.PathLike[str] = "./audit_logs") -> None:
        """Initialise.

        Args:
            root: Directory to write JSONL files into. Created if missing.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, ts_iso: str) -> Path:
        date = ts_iso[:10]  # 2026-04-27
        return self.root / f"audit-{date}.jsonl"

    def write(self, record: AuditRecord) -> None:
        """Append one record. Raises any I/O error — callers should not swallow."""
        line = json.dumps(asdict(record), default=str, ensure_ascii=False)
        path = self._path_for(record.timestamp_utc)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())


class StdoutBackend:
    """Emit one JSON object per line to stdout.

    Production deployments forward stdout into the bank's SIEM (Splunk,
    Elastic, Sumo, etc.) which provides immutability and retention.
    """

    def __init__(self, stream: Any = None) -> None:
        """Initialise.

        Args:
            stream: File-like object to write to. Default: sys.stdout.
        """
        self.stream = stream or sys.stdout
        self._lock = threading.Lock()

    def write(self, record: AuditRecord) -> None:
        line = json.dumps(asdict(record), default=str, ensure_ascii=False)
        with self._lock:
            self.stream.write(line + "\n")
            self.stream.flush()


class MultiBackend:
    """Fan out one record to multiple backends (e.g. JSONL + SIEM)."""

    def __init__(self, *backends: AuditBackend) -> None:
        if not backends:
            raise ValueError("MultiBackend requires at least one backend")
        self.backends = backends

    def write(self, record: AuditRecord) -> None:
        for b in self.backends:
            b.write(record)


class AuditLogger:
    """Public facade. Validates records and dispatches to a backend.

    The logger fails loudly: any exception raised by the backend propagates,
    because losing audit records silently is a regulatory breach.
    """

    def __init__(self, backend: Optional[AuditBackend] = None) -> None:
        """Initialise.

        Args:
            backend: Storage backend. Default: ``JSONLBackend("./audit_logs")``.
        """
        self.backend = backend or JSONLBackend()

    def log(self, record: AuditRecord) -> None:
        """Validate and append one record."""
        for f_ in REQUIRED_FIELDS:
            if not hasattr(record, f_):
                raise ValueError(f"AuditRecord missing required field: {f_}")
        self.backend.write(record)
