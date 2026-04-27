"""Tests for :mod:`workflow.batch`.

Exercises :class:`workflow.batch.BatchProcessor` end-to-end with a mock
workflow object that mimics the duck-typed
``process_application(applicant: dict) -> WorkflowResult`` contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import pytest

from workflow.batch import BatchProcessor, BatchResult


@dataclass
class _FakeResult:
    """Minimal duck-typed stand-in for ``WorkflowResult``."""

    decision: str
    probability: float
    memo: str
    adverse_action: Optional[str]
    shap_factors: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockWorkflow:
    """Successful mock workflow.

    Returns a deterministic synthetic ``WorkflowResult`` derived from the
    incoming applicant dict so tests can verify per-row content.
    """

    def process_application(self, applicant: Dict[str, Any]) -> _FakeResult:
        rid = int(applicant.get("id", -1))
        prob = (rid % 100) / 100.0
        decision = "DENY" if prob >= 0.5 else "APPROVE"
        return _FakeResult(
            decision=decision,
            probability=prob,
            memo=f"memo for row {rid}",
            adverse_action=("denial " + str(rid)) if decision == "DENY" else None,
            flags=["BORDERLINE"] if 0.4 <= prob <= 0.6 else [],
        )


class FlakyWorkflow(MockWorkflow):
    """Mock workflow that raises on a configurable set of row ids."""

    def __init__(self, fail_ids: Set[int]) -> None:
        self.fail_ids = set(fail_ids)

    def process_application(self, applicant: Dict[str, Any]) -> _FakeResult:
        rid = int(applicant.get("id", -1))
        if rid in self.fail_ids:
            raise RuntimeError(f"synthetic failure for row {rid}")
        return super().process_application(applicant)


class PartialWorkflow(MockWorkflow):
    """Mock workflow that aborts after processing N rows (simulating a crash).

    Raises :class:`SystemExit`, which is *not* caught by ``BatchProcessor``
    (it only catches :class:`Exception`), so processing halts mid-batch and
    leaves a checkpoint behind for the resume test.
    """

    def __init__(self, halt_after: int) -> None:
        self.halt_after = halt_after
        self._count = 0
        self._lock_count = 0

    def process_application(self, applicant: Dict[str, Any]) -> _FakeResult:
        # Count how many we've handled. Using an attribute is racy under
        # threads, but the test runs single-threaded (max_workers=1).
        self._count += 1
        if self._count > self.halt_after:
            raise SystemExit("simulated crash")
        return super().process_application(applicant)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_df(n: int) -> pd.DataFrame:
    """Build a deterministic DataFrame of ``n`` synthetic applicants."""
    return pd.DataFrame(
        [
            {
                "id": i,
                "LIMIT_BAL": 100_000 + i * 1000,
                "PAY_0": (i % 5) - 2,
                "AGE": 25 + (i % 30),
            }
            for i in range(n)
        ]
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_process_50_applications_all_succeed(tmp_path: Path) -> None:
    """50 mock applications produce 50 output rows and zero failures."""
    df = _make_df(50)
    out_csv = tmp_path / "out.csv"
    bp = BatchProcessor(
        workflow=MockWorkflow(),
        output_csv=str(out_csv),
        max_workers=4,
        checkpoint_interval=10,
    )
    result = bp.process_dataframe(df)

    assert isinstance(result, BatchResult)
    assert result.total == 50
    assert result.succeeded == 50
    assert result.failed == 0

    out_df = pd.read_csv(out_csv)
    assert len(out_df) == 50
    assert sorted(out_df["row_index"].tolist()) == list(range(50))
    assert set(out_df.columns) == {
        "row_index",
        "decision",
        "probability",
        "memo",
        "adverse_action",
        "flags_json",
        "error",
    }
    # Decisions are valid
    assert set(out_df["decision"].unique()).issubset({"APPROVE", "DENY"})
    # flags_json column parses
    for raw in out_df["flags_json"]:
        json.loads(raw)

    failures_csv = Path(bp.failures_csv)
    assert failures_csv.exists()
    failures_df = pd.read_csv(failures_csv)
    assert len(failures_df) == 0


def test_flaky_workflow_isolates_failures(tmp_path: Path) -> None:
    """Failures on rows 5/15/25 yield 47 successes + 3 failures, batch continues."""
    df = _make_df(50)
    out_csv = tmp_path / "flaky.csv"
    fail_ids = {5, 15, 25}
    bp = BatchProcessor(
        workflow=FlakyWorkflow(fail_ids),
        output_csv=str(out_csv),
        max_workers=4,
        checkpoint_interval=10,
    )
    result = bp.process_dataframe(df)

    assert result.total == 50
    assert result.succeeded == 47
    assert result.failed == 3

    out_df = pd.read_csv(out_csv)
    assert len(out_df) == 47
    assert set(out_df["row_index"]).isdisjoint(fail_ids)

    failures_df = pd.read_csv(bp.failures_csv)
    assert len(failures_df) == 3
    assert sorted(failures_df["row_index"].tolist()) == sorted(fail_ids)
    assert set(failures_df["error_class"].unique()) == {"RuntimeError"}
    for msg in failures_df["error_message"]:
        assert "synthetic failure" in msg


def test_checkpoint_file_written_after_partial_run(tmp_path: Path) -> None:
    """A partial run leaves a checkpoint sidecar with the last flushed index."""
    df = _make_df(50)
    out_csv = tmp_path / "ckpt.csv"
    # Halt after 20 rows so we expect at least one checkpoint flush at idx 9
    # (interval=10, so flushes happen at 10, 20).
    bp = BatchProcessor(
        workflow=PartialWorkflow(halt_after=20),
        output_csv=str(out_csv),
        max_workers=1,  # deterministic ordering
        checkpoint_interval=10,
    )

    with pytest.raises(SystemExit):
        bp.process_dataframe(df)

    ckpt = Path(bp.checkpoint_path)
    assert ckpt.exists(), "checkpoint file should be written after a flush"
    last_idx = int(ckpt.read_text(encoding="utf-8").strip())
    # We flushed in batches of 10; after 20 successes the highest emitted
    # row_index is 19.
    assert last_idx >= 9
    assert last_idx <= 19

    # Output CSV should also contain the flushed rows.
    out_df = pd.read_csv(out_csv)
    assert len(out_df) >= 10
    assert (out_df["row_index"] <= last_idx).all()


def test_resume_skips_processed_rows(tmp_path: Path) -> None:
    """A second run with the same paths skips rows at or below the checkpoint."""
    df = _make_df(50)
    out_csv = tmp_path / "resume.csv"

    # First, simulate a partial run by running on rows 0..29 with a healthy
    # workflow then writing a checkpoint manually at index 29. This mimics
    # what would be on disk after a real crash + flush.
    bp1 = BatchProcessor(
        workflow=MockWorkflow(),
        output_csv=str(out_csv),
        max_workers=1,
        checkpoint_interval=10,
    )
    first_chunk = df.iloc[:30].copy()
    res1 = bp1.process_dataframe(first_chunk)
    assert res1.succeeded == 30
    # Checkpoint should reflect the last emitted index (29).
    assert int(Path(bp1.checkpoint_path).read_text(encoding="utf-8").strip()) == 29

    # Sanity: failures CSV exists but has only its header.
    failures_first = pd.read_csv(bp1.failures_csv)
    assert len(failures_first) == 0

    # Now resume with the full DataFrame. Rows 0..29 should be skipped
    # because their indices are <= the checkpoint.
    seen_calls: List[int] = []

    class TrackingWorkflow(MockWorkflow):
        def process_application(self, applicant: Dict[str, Any]) -> _FakeResult:
            seen_calls.append(int(applicant["id"]))
            return super().process_application(applicant)

    bp2 = BatchProcessor(
        workflow=TrackingWorkflow(),
        output_csv=str(out_csv),
        max_workers=1,
        checkpoint_interval=10,
    )
    res2 = bp2.process_dataframe(df)

    # On the second run only rows 30..49 should have been dispatched.
    assert sorted(seen_calls) == list(range(30, 50))
    assert res2.succeeded == 20
    assert res2.failed == 0

    # The output CSV should now hold all 50 unique row indices.
    out_df = pd.read_csv(out_csv)
    assert sorted(out_df["row_index"].tolist()) == list(range(50))


def test_progress_callback_invoked(tmp_path: Path) -> None:
    """``on_progress`` fires after every flush with monotonically growing counts."""
    df = _make_df(25)
    out_csv = tmp_path / "progress.csv"
    calls: List[tuple] = []

    bp = BatchProcessor(
        workflow=MockWorkflow(),
        output_csv=str(out_csv),
        max_workers=2,
        checkpoint_interval=10,
        on_progress=lambda completed, total: calls.append((completed, total)),
    )
    result = bp.process_dataframe(df)
    assert result.succeeded == 25
    assert calls, "on_progress should be invoked at least once"
    # Totals are constant; completed counts are strictly increasing.
    completed_seq = [c for c, _ in calls]
    assert completed_seq == sorted(completed_seq)
    assert all(t == 25 for _, t in calls)
    # Final flush should report all rows done.
    assert calls[-1][0] == 25


def test_process_csv_reads_input_file(tmp_path: Path) -> None:
    """``process_csv`` reads from disk and produces the same shape of output."""
    df = _make_df(15)
    in_csv = tmp_path / "in.csv"
    df.to_csv(in_csv, index=False)
    out_csv = tmp_path / "csv_out.csv"
    bp = BatchProcessor(
        workflow=MockWorkflow(),
        output_csv=str(out_csv),
        max_workers=2,
        checkpoint_interval=5,
    )
    result = bp.process_csv(str(in_csv))
    assert result.total == 15
    assert result.succeeded == 15
    assert result.failed == 0
    out_df = pd.read_csv(out_csv)
    assert sorted(out_df["row_index"].tolist()) == list(range(15))
