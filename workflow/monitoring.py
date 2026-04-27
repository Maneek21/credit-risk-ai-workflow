"""Production monitoring for the credit-risk model.

Three signals every deployed credit model needs to track post-launch:

  1. **Population stability** — feature distributions shift when the borrower
     population changes (macro shock, new acquisition channel, marketing pivot).
     Detected here via PSI per feature.
  2. **Predictive performance** — the model's AUC will drift down as the
     world drifts away from the training distribution.
  3. **Approval rate** — sudden swings in approval rate are a leading
     indicator of either a data-pipeline bug or an upstream policy change
     before they show up in default rates.

This module ships a thin ``ModelMonitor`` that batches the three checks and
emits a list of ``Alert`` objects. The monitor is intentionally stateless —
the caller passes in baseline metrics (typically loaded from JSON written at
training time) and a window of current observations.

Example
-------
    from workflow.monitoring import ModelMonitor
    monitor = ModelMonitor(
        baseline_metrics={
            "AUC": 0.774,
            "approval_rate": 0.78,
            "feature_distributions": {"PAY_0": baseline_pay0_array},
        },
    )
    alerts = monitor.check_all(
        current_features={"PAY_0": recent_pay0_array},
        current_auc=0.71,
        current_approval_rate=0.90,
    )
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


__all__ = ["Alert", "compute_psi", "ModelMonitor"]


@dataclass(frozen=True)
class Alert:
    """A single monitoring alert.

    Attributes:
        severity: One of ``"INFO"``, ``"WARNING"``, ``"CRITICAL"``.
        metric: Name of the metric that fired (e.g. ``"PSI[PAY_0]"``,
            ``"AUC"``, ``"approval_rate"``).
        value: Observed value of the metric in the current window.
        baseline: Reference value the metric is being compared against.
        threshold: Threshold that was breached.
        message: Human-readable description of what happened.
        timestamp_utc: ISO-8601 UTC timestamp of when the alert was raised.
    """

    severity: str
    metric: str
    value: float
    baseline: float
    threshold: float
    message: str
    timestamp_utc: str


def _now_utc_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def compute_psi(
    baseline: Sequence[float] | np.ndarray | pd.Series,
    current: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 10,
) -> float:
    """Compute the Population Stability Index (PSI) between two distributions.

    The baseline distribution is sliced into ``n_bins`` quantile-based
    buckets, then both samples are binned identically and PSI is computed as

        PSI = sum_i (p_curr_i - p_base_i) * ln(p_curr_i / p_base_i)

    Bucket probabilities are floored at ``epsilon = 1e-4`` to keep ``log(0)``
    out of the result when a bucket is empty in one of the samples.

    Args:
        baseline: Reference (training-time) sample of a single feature.
        current: Current production sample of the same feature.
        n_bins: Number of quantile buckets. Default 10.

    Returns:
        PSI as a non-negative float. Conventional interpretation:
        < 0.10 stable; 0.10-0.25 moderate shift; > 0.25 major shift.
    """
    base_arr = np.asarray(baseline, dtype=float)
    curr_arr = np.asarray(current, dtype=float)
    base_arr = base_arr[~np.isnan(base_arr)]
    curr_arr = curr_arr[~np.isnan(curr_arr)]

    if base_arr.size == 0 or curr_arr.size == 0:
        return 0.0

    # Build bin edges from baseline quantiles. Make them unique to handle
    # heavily-discrete features (e.g. PAY_0 takes only ~10 integer values).
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(base_arr, quantiles))
    if edges.size < 2:
        return 0.0
    # Extend outer edges to capture values outside baseline range.
    edges[0] = -np.inf
    edges[-1] = np.inf

    base_counts, _ = np.histogram(base_arr, bins=edges)
    curr_counts, _ = np.histogram(curr_arr, bins=edges)

    epsilon = 1e-4
    p_base = np.maximum(base_counts / base_arr.size, epsilon)
    p_curr = np.maximum(curr_counts / curr_arr.size, epsilon)

    psi = float(np.sum((p_curr - p_base) * np.log(p_curr / p_base)))
    return psi


_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "psi_warning": 0.20,
    "psi_critical": 0.25,
    "auc_drop_warning": 0.03,
    "auc_drop_critical": 0.05,
    "approval_rate_change_warning_pp": 10.0,
}


class ModelMonitor:
    """Batched monitor over PSI, AUC drop, and approval-rate change.

    The monitor is stateless. Each ``check_*`` call recomputes from the
    arguments it was given; nothing is persisted between calls.

    Attributes:
        baseline_metrics: Reference snapshot from training time. Expected
            keys: ``"AUC"`` (float), ``"approval_rate"`` (float in [0, 1]),
            ``"feature_distributions"`` (dict of feature-name to array-like).
        thresholds: Trip points for each axis. Defaults are conservative;
            override via the ``thresholds`` constructor argument.
    """

    def __init__(
        self,
        baseline_metrics: Mapping[str, Any],
        thresholds: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Initialise.

        Args:
            baseline_metrics: Reference snapshot. See class docstring for
                expected keys.
            thresholds: Override any of the default trip points. Unknown
                keys are ignored.
        """
        self.baseline_metrics: Dict[str, Any] = dict(baseline_metrics)
        merged = dict(_DEFAULT_THRESHOLDS)
        if thresholds:
            merged.update({k: float(v) for k, v in thresholds.items() if k in merged})
        self.thresholds: Dict[str, float] = merged

    def check_psi(
        self, current_features: Mapping[str, Sequence[float] | np.ndarray | pd.Series]
    ) -> List[Alert]:
        """Compute PSI for every baseline feature and emit alerts.

        Args:
            current_features: Mapping from feature name to current sample.
                Features not present in the baseline are silently ignored;
                features in the baseline but missing here are skipped.

        Returns:
            One ``Alert`` per feature whose PSI breaches the warning or
            critical threshold. Empty list if everything is stable.
        """
        baseline_dists: Mapping[str, Any] = self.baseline_metrics.get(
            "feature_distributions", {}
        )
        warn = self.thresholds["psi_warning"]
        crit = self.thresholds["psi_critical"]
        alerts: List[Alert] = []

        for feature, baseline_sample in baseline_dists.items():
            if feature not in current_features:
                continue
            psi = compute_psi(baseline_sample, current_features[feature])
            if psi >= crit:
                alerts.append(
                    Alert(
                        severity="CRITICAL",
                        metric=f"PSI[{feature}]",
                        value=psi,
                        baseline=0.0,
                        threshold=crit,
                        message=(
                            f"PSI for {feature}={psi:.3f} exceeds critical "
                            f"threshold {crit:.2f}; major distribution shift."
                        ),
                        timestamp_utc=_now_utc_iso(),
                    )
                )
            elif psi >= warn:
                alerts.append(
                    Alert(
                        severity="WARNING",
                        metric=f"PSI[{feature}]",
                        value=psi,
                        baseline=0.0,
                        threshold=warn,
                        message=(
                            f"PSI for {feature}={psi:.3f} exceeds warning "
                            f"threshold {warn:.2f}; moderate distribution shift."
                        ),
                        timestamp_utc=_now_utc_iso(),
                    )
                )
        return alerts

    def check_auc(self, current_auc: float) -> Optional[Alert]:
        """Emit an alert if AUC has dropped beyond the configured tolerance.

        Args:
            current_auc: Most recently measured AUC (e.g. on the rolling
                evaluation window).

        Returns:
            ``None`` if the drop is within tolerance, otherwise a ``WARNING``
            or ``CRITICAL`` alert.
        """
        baseline_auc = float(self.baseline_metrics.get("AUC", 0.0))
        drop = baseline_auc - float(current_auc)
        warn = self.thresholds["auc_drop_warning"]
        crit = self.thresholds["auc_drop_critical"]

        if drop > crit:
            return Alert(
                severity="CRITICAL",
                metric="AUC",
                value=float(current_auc),
                baseline=baseline_auc,
                threshold=crit,
                message=(
                    f"AUC dropped {drop:.3f} from baseline {baseline_auc:.3f} "
                    f"to {current_auc:.3f} (critical threshold {crit:.2f})."
                ),
                timestamp_utc=_now_utc_iso(),
            )
        if drop > warn:
            return Alert(
                severity="WARNING",
                metric="AUC",
                value=float(current_auc),
                baseline=baseline_auc,
                threshold=warn,
                message=(
                    f"AUC dropped {drop:.3f} from baseline {baseline_auc:.3f} "
                    f"to {current_auc:.3f} (warning threshold {warn:.2f})."
                ),
                timestamp_utc=_now_utc_iso(),
            )
        return None

    def check_approval_rate(
        self, current_rate: float, days: int = 7
    ) -> Optional[Alert]:
        """Emit an alert if approval rate has moved by more than N percentage points.

        Args:
            current_rate: Current rolling approval rate as a fraction in
                ``[0, 1]``.
            days: Window length in days, included in the alert message.
                Default 7.

        Returns:
            ``WARNING`` alert if absolute change in percentage points exceeds
            the configured trip point, else ``None``.
        """
        baseline_rate = float(self.baseline_metrics.get("approval_rate", 0.0))
        change_pp = abs(float(current_rate) - baseline_rate) * 100.0
        warn_pp = self.thresholds["approval_rate_change_warning_pp"]

        if change_pp > warn_pp:
            direction = "increased" if current_rate > baseline_rate else "decreased"
            return Alert(
                severity="WARNING",
                metric="approval_rate",
                value=float(current_rate),
                baseline=baseline_rate,
                threshold=warn_pp,
                message=(
                    f"Approval rate {direction} by {change_pp:.1f}pp over "
                    f"{days}d (baseline {baseline_rate:.1%}, "
                    f"current {current_rate:.1%}, threshold {warn_pp:.1f}pp)."
                ),
                timestamp_utc=_now_utc_iso(),
            )
        return None

    def check_all(
        self,
        current_features: Mapping[str, Sequence[float] | np.ndarray | pd.Series],
        current_auc: Optional[float] = None,
        current_approval_rate: Optional[float] = None,
    ) -> List[Alert]:
        """Run every configured check and return the union of alerts.

        Args:
            current_features: See :meth:`check_psi`.
            current_auc: If provided, runs :meth:`check_auc`.
            current_approval_rate: If provided, runs :meth:`check_approval_rate`.

        Returns:
            A flat list of all alerts raised across the three checks.
        """
        alerts: List[Alert] = []
        alerts.extend(self.check_psi(current_features))
        if current_auc is not None:
            auc_alert = self.check_auc(current_auc)
            if auc_alert is not None:
                alerts.append(auc_alert)
        if current_approval_rate is not None:
            rate_alert = self.check_approval_rate(current_approval_rate)
            if rate_alert is not None:
                alerts.append(rate_alert)
        return alerts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_baseline(path: str) -> Dict[str, Any]:
    """Load a baseline-metrics JSON file from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_window_csv(path: str) -> pd.DataFrame:
    """Load a recent-window CSV into a DataFrame."""
    return pd.read_csv(path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run model-monitoring checks against a recent window of "
            "production data and print any alerts as JSON."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the baseline metrics JSON file.",
    )
    parser.add_argument(
        "--window-csv",
        required=True,
        help=(
            "Path to a CSV of recent production observations. May include "
            "columns for any baseline feature plus optional 'auc' and "
            "'approval' columns."
        ),
    )
    return parser


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    baseline = _load_baseline(args.baseline)
    window = _load_window_csv(args.window_csv)

    feature_names = list(baseline.get("feature_distributions", {}).keys())
    current_features: Dict[str, np.ndarray] = {
        name: window[name].to_numpy()
        for name in feature_names
        if name in window.columns
    }

    current_auc: Optional[float] = None
    if "auc" in window.columns and len(window["auc"].dropna()) > 0:
        current_auc = float(window["auc"].dropna().iloc[-1])

    current_approval_rate: Optional[float] = None
    if "approval" in window.columns and len(window["approval"].dropna()) > 0:
        current_approval_rate = float(window["approval"].dropna().mean())
    elif "approved" in window.columns and len(window["approved"].dropna()) > 0:
        current_approval_rate = float(window["approved"].dropna().mean())

    monitor = ModelMonitor(baseline_metrics=baseline)
    alerts = monitor.check_all(
        current_features=current_features,
        current_auc=current_auc,
        current_approval_rate=current_approval_rate,
    )

    payload = [asdict(a) for a in alerts]
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
