"""Batch processing for credit-risk workflow applications.

Provides :class:`BatchProcessor`, a concurrent CSV-driven runner that calls a
``process_application(applicant: dict) -> WorkflowResult``-shaped object on
each row of a :class:`pandas.DataFrame` (or input CSV), persists results,
isolates failures, and supports checkpoint-based resume.

Designed so a single bad row never halts a long-running batch: any exception
raised by the underlying workflow is captured to a sibling failures CSV, the
failure count is incremented, and processing continues.
"""

from __future__ import annotations

import csv
import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    pass


OUTPUT_COLUMNS: List[str] = [
    "row_index",
    "decision",
    "probability",
    "memo",
    "adverse_action",
    "flags_json",
    "error",
]

FAILURE_COLUMNS: List[str] = [
    "row_index",
    "error_class",
    "error_message",
]


@dataclass
class BatchResult:
    """Summary of a completed batch run.

    Attributes:
        total: Total number of rows iterated (including resumed-skipped rows).
        succeeded: Number of rows whose workflow call returned without error.
        failed: Number of rows whose workflow call raised an exception.
        output_csv: Absolute path to the output CSV containing successful rows.
        failures_csv: Absolute path to the failures CSV containing error rows.
        elapsed_sec: Wall-clock seconds spent in :meth:`BatchProcessor.process_dataframe`.
    """

    total: int
    succeeded: int
    failed: int
    output_csv: str
    failures_csv: str
    elapsed_sec: float


class BatchProcessor:
    """Concurrent batch runner over a workflow with checkpoint/resume.

    Args:
        workflow: Any object exposing a ``process_application(applicant: dict)``
            method that returns a ``WorkflowResult``-shaped value (duck-typed).
        output_csv: Path where successful rows are written.
        max_workers: Thread-pool worker count. Defaults to ``4``.
        checkpoint_interval: Flush results and write the checkpoint file every
            N completed rows. Defaults to ``10``.
        on_progress: Optional callable invoked as
            ``on_progress(completed, total)`` after each flush.

    Notes:
        The processor writes a ``<output_csv>.checkpoint`` sidecar containing
        the highest ``row_index`` flushed so far. If both the checkpoint file
        and the output CSV exist on a subsequent run, rows whose ``row_index``
        is less than or equal to the checkpoint are skipped, allowing the
        batch to be resumed safely.
    """

    def __init__(
        self,
        workflow: Any,
        output_csv: str,
        max_workers: int = 4,
        checkpoint_interval: int = 10,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.workflow = workflow
        self.output_csv = str(output_csv)
        self.failures_csv = self.output_csv.replace(".csv", "_failures.csv")
        if self.failures_csv == self.output_csv:
            # Defensive: ensure failures csv is distinct even if extension
            # didn't match. Append suffix.
            self.failures_csv = self.output_csv + "_failures.csv"
        self.checkpoint_path = self.output_csv + ".checkpoint"
        self.max_workers = max(1, int(max_workers))
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        self.on_progress = on_progress

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_csv(self, input_csv: str) -> BatchResult:
        """Read ``input_csv`` and process every row.

        Args:
            input_csv: Path to a CSV file readable by :func:`pandas.read_csv`.

        Returns:
            A :class:`BatchResult` summarising the run.
        """
        df = pd.read_csv(input_csv)
        return self.process_dataframe(df)

    def process_dataframe(self, df: pd.DataFrame) -> BatchResult:
        """Process every row of ``df`` concurrently and persist results.

        Rows are dispatched to a :class:`concurrent.futures.ThreadPoolExecutor`
        and collected as they finish. Each row's outcome (success or failure)
        is written deterministically to either the output CSV or the failures
        CSV. After every ``checkpoint_interval`` completed rows the buffered
        results are flushed and the checkpoint file is updated.

        Args:
            df: DataFrame whose rows are converted to dicts via ``to_dict``.

        Returns:
            A :class:`BatchResult` summarising the run.
        """
        start = time.perf_counter()

        already_done = self._load_checkpoint_skip_set()

        # Ensure parent directories for output paths exist.
        Path(self.output_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(self.failures_csv).parent.mkdir(parents=True, exist_ok=True)

        # Open output and failures CSVs in append mode if resuming, else
        # truncate. We append when we already have a checkpoint AND the
        # output CSV exists.
        resuming = bool(already_done) and Path(self.output_csv).exists()
        out_mode = "a" if resuming else "w"
        fail_mode = "a" if resuming and Path(self.failures_csv).exists() else "w"

        succeeded = 0
        failed = 0
        total = len(df)

        applicants: List[Tuple[int, Dict[str, Any]]] = []
        for idx, row in df.iterrows():
            row_index = int(idx)
            if row_index in already_done:
                continue
            applicants.append((row_index, row.to_dict()))

        # Buffers flushed every checkpoint_interval.
        success_buffer: List[Dict[str, Any]] = []
        failure_buffer: List[Dict[str, Any]] = []
        completed_indices: List[int] = []

        with open(self.output_csv, out_mode, newline="", encoding="utf-8") as out_f, open(
            self.failures_csv, fail_mode, newline="", encoding="utf-8"
        ) as fail_f:
            out_writer = csv.DictWriter(out_f, fieldnames=OUTPUT_COLUMNS)
            fail_writer = csv.DictWriter(fail_f, fieldnames=FAILURE_COLUMNS)
            if out_mode == "w":
                out_writer.writeheader()
            if fail_mode == "w":
                fail_writer.writeheader()

            results_by_index: Dict[int, Dict[str, Any]] = {}
            failures_by_index: Dict[int, Dict[str, Any]] = {}

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_to_index = {
                    pool.submit(self._process_one, ridx, applicant): ridx
                    for ridx, applicant in applicants
                }

                next_to_emit = 0  # pointer into applicants (preserves order)
                applicant_order = [ridx for ridx, _ in applicants]

                done_set: set[int] = set()
                for future in as_completed(future_to_index):
                    ridx = future_to_index[future]
                    success_row, failure_row = future.result()
                    if success_row is not None:
                        results_by_index[ridx] = success_row
                    if failure_row is not None:
                        failures_by_index[ridx] = failure_row
                    done_set.add(ridx)

                    # Emit any contiguous prefix of finished rows in input order
                    while (
                        next_to_emit < len(applicant_order)
                        and applicant_order[next_to_emit] in done_set
                    ):
                        emit_idx = applicant_order[next_to_emit]
                        if emit_idx in results_by_index:
                            success_buffer.append(results_by_index.pop(emit_idx))
                            succeeded += 1
                        if emit_idx in failures_by_index:
                            failure_buffer.append(failures_by_index.pop(emit_idx))
                            failed += 1
                        completed_indices.append(emit_idx)
                        next_to_emit += 1

                        if (
                            len(completed_indices) >= self.checkpoint_interval
                            or next_to_emit == len(applicant_order)
                        ):
                            self._flush(
                                out_writer,
                                out_f,
                                fail_writer,
                                fail_f,
                                success_buffer,
                                failure_buffer,
                                completed_indices,
                            )
                            if self.on_progress is not None:
                                done_count = (
                                    len(already_done)
                                    + (next_to_emit)
                                )
                                self.on_progress(done_count, total)
                            success_buffer = []
                            failure_buffer = []
                            completed_indices = []

        elapsed = time.perf_counter() - start
        return BatchResult(
            total=total,
            succeeded=succeeded,
            failed=failed,
            output_csv=os.path.abspath(self.output_csv),
            failures_csv=os.path.abspath(self.failures_csv),
            elapsed_sec=elapsed,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _process_one(
        self, row_index: int, applicant: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Run the workflow on a single row and shape the output dict.

        Args:
            row_index: Original DataFrame index for the row.
            applicant: Row contents as a dict.

        Returns:
            A ``(success_row, failure_row)`` tuple where exactly one element is
            non-``None``.
        """
        try:
            result = self.workflow.process_application(applicant)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            failure_row = {
                "row_index": row_index,
                "error_class": type(exc).__name__,
                "error_message": _truncate(str(exc) or traceback.format_exc(limit=1)),
            }
            return None, failure_row

        success_row = {
            "row_index": row_index,
            "decision": getattr(result, "decision", ""),
            "probability": getattr(result, "probability", ""),
            "memo": getattr(result, "memo", ""),
            "adverse_action": getattr(result, "adverse_action", "") or "",
            "flags_json": json.dumps(list(getattr(result, "flags", []) or [])),
            "error": "",
        }
        return success_row, None

    def _flush(
        self,
        out_writer: csv.DictWriter,
        out_f: Any,
        fail_writer: csv.DictWriter,
        fail_f: Any,
        success_buffer: List[Dict[str, Any]],
        failure_buffer: List[Dict[str, Any]],
        completed_indices: List[int],
    ) -> None:
        """Persist buffered rows and update the checkpoint atomically-ish.

        Writes any accumulated success/failure rows, fsyncs the file handles,
        and rewrites the checkpoint file with the largest emitted ``row_index``.
        """
        if success_buffer:
            out_writer.writerows(success_buffer)
            out_f.flush()
        if failure_buffer:
            fail_writer.writerows(failure_buffer)
            fail_f.flush()
        if completed_indices:
            last_index = max(completed_indices)
            self._write_checkpoint(last_index)

    def _write_checkpoint(self, last_index: int) -> None:
        """Write the highest emitted row index to the checkpoint file.

        Args:
            last_index: The largest ``row_index`` that has been flushed.
        """
        existing = self._read_checkpoint()
        if existing is not None and existing > last_index:
            return  # don't move checkpoint backwards
        tmp = self.checkpoint_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(str(int(last_index)))
        os.replace(tmp, self.checkpoint_path)

    def _read_checkpoint(self) -> Optional[int]:
        """Return the integer row index stored in the checkpoint, or None."""
        if not Path(self.checkpoint_path).exists():
            return None
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                return int(f.read().strip())
        except (ValueError, OSError):
            return None

    def _load_checkpoint_skip_set(self) -> set[int]:
        """Return the set of row indices considered already processed.

        If the checkpoint file is absent, returns an empty set. Otherwise
        returns ``{0, 1, ..., last_index}`` so any DataFrame row at or below
        the checkpoint is skipped on resume.
        """
        last = self._read_checkpoint()
        if last is None:
            return set()
        if not Path(self.output_csv).exists():
            return set()
        return set(range(0, last + 1))


def _truncate(text: str, limit: int = 1000) -> str:
    """Truncate a string to ``limit`` characters with an ellipsis suffix."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
