"""Tests for workflow.monitoring."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from workflow.monitoring import Alert, ModelMonitor, compute_psi


# ---------------------------------------------------------------------------
# compute_psi
# ---------------------------------------------------------------------------


def test_psi_identical_distributions_is_near_zero() -> None:
    rng = np.random.default_rng(42)
    sample = rng.normal(0.0, 1.0, size=10_000)
    psi = compute_psi(sample, sample.copy())
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_resampled_same_distribution_is_small() -> None:
    """Two fresh samples from the same distribution should yield small PSI."""
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 1.0, size=10_000)
    curr = rng.normal(0.0, 1.0, size=10_000)
    psi = compute_psi(base, curr)
    assert psi < 0.05


def test_psi_shifted_distribution_exceeds_warning() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 1.0, size=10_000)
    # Mean shift of 1 sigma is a major shift in PSI terms (~0.5+).
    curr = rng.normal(1.0, 1.0, size=10_000)
    psi = compute_psi(base, curr)
    assert psi > 0.20


# ---------------------------------------------------------------------------
# ModelMonitor.check_auc
# ---------------------------------------------------------------------------


def _baseline() -> Dict[str, object]:
    rng = np.random.default_rng(0)
    return {
        "AUC": 0.77,
        "approval_rate": 0.78,
        "feature_distributions": {
            "PAY_0": rng.normal(0.0, 1.0, size=5_000),
        },
    }


def test_check_auc_warning_for_moderate_drop() -> None:
    monitor = ModelMonitor(baseline_metrics=_baseline())
    alert = monitor.check_auc(current_auc=0.73)  # 0.04 drop
    assert alert is not None
    assert alert.severity == "WARNING"
    assert alert.metric == "AUC"
    assert alert.value == pytest.approx(0.73)
    assert alert.baseline == pytest.approx(0.77)


def test_check_auc_critical_for_large_drop() -> None:
    monitor = ModelMonitor(baseline_metrics=_baseline())
    alert = monitor.check_auc(current_auc=0.71)  # 0.06 drop
    assert alert is not None
    assert alert.severity == "CRITICAL"
    assert alert.metric == "AUC"


def test_check_auc_no_alert_for_small_drop() -> None:
    monitor = ModelMonitor(baseline_metrics=_baseline())
    assert monitor.check_auc(current_auc=0.76) is None  # 0.01 drop


# ---------------------------------------------------------------------------
# ModelMonitor.check_approval_rate
# ---------------------------------------------------------------------------


def test_check_approval_rate_alerts_on_12pp_jump() -> None:
    monitor = ModelMonitor(baseline_metrics=_baseline())
    alert = monitor.check_approval_rate(current_rate=0.90)  # 78 -> 90 = +12pp
    assert alert is not None
    assert alert.severity == "WARNING"
    assert alert.metric == "approval_rate"
    assert alert.value == pytest.approx(0.90)


def test_check_approval_rate_silent_within_tolerance() -> None:
    monitor = ModelMonitor(baseline_metrics=_baseline())
    assert monitor.check_approval_rate(current_rate=0.80) is None  # +2pp


# ---------------------------------------------------------------------------
# ModelMonitor.check_all
# ---------------------------------------------------------------------------


def test_check_all_returns_multiple_alerts_when_overlapping() -> None:
    rng = np.random.default_rng(5)
    baseline_pay0 = rng.normal(0.0, 1.0, size=5_000)
    baseline = {
        "AUC": 0.77,
        "approval_rate": 0.78,
        "feature_distributions": {"PAY_0": baseline_pay0},
    }
    monitor = ModelMonitor(baseline_metrics=baseline)

    # Force PSI breach via mean shift.
    shifted = rng.normal(1.0, 1.0, size=5_000)

    alerts: List[Alert] = monitor.check_all(
        current_features={"PAY_0": shifted},
        current_auc=0.71,           # critical AUC drop
        current_approval_rate=0.90, # +12pp approval-rate jump
    )

    assert len(alerts) >= 3
    metrics = {a.metric for a in alerts}
    assert "PSI[PAY_0]" in metrics
    assert "AUC" in metrics
    assert "approval_rate" in metrics
    severities = {a.severity for a in alerts}
    assert "CRITICAL" in severities
    assert "WARNING" in severities


def test_check_all_no_alerts_for_stable_window() -> None:
    rng = np.random.default_rng(11)
    baseline_pay0 = rng.normal(0.0, 1.0, size=5_000)
    baseline = {
        "AUC": 0.77,
        "approval_rate": 0.78,
        "feature_distributions": {"PAY_0": baseline_pay0},
    }
    monitor = ModelMonitor(baseline_metrics=baseline)

    alerts = monitor.check_all(
        current_features={"PAY_0": baseline_pay0.copy()},
        current_auc=0.77,
        current_approval_rate=0.79,
    )
    assert alerts == []
