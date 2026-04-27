"""Evaluation metrics for the training SDK.

Centralises the AUC / KS / Brier / ECE / fairness / SHAP plumbing that
was previously duplicated between ``benchmarks/src/common.py`` and
ad-hoc scripts. Stdlib + sklearn only — fairlearn is optional and
falls back to a hand-rolled disparate-impact computation when not
installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


# ---------------------------------------------------------------------------
# Core probabilistic metrics
# ---------------------------------------------------------------------------


def compute_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_prob))


def compute_ks(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max |TPR - FPR| across thresholds."""
    order = np.argsort(-np.asarray(y_prob))
    y_sorted = np.asarray(y_true)[order]
    tpr = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    fpr = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(tpr - fpr)))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — bin predictions, weight by bin size."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    n = len(y_prob)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def core_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Bundle the four canonical scoring metrics into one dict."""
    return {
        "auc":   compute_auc(y_true, y_prob),
        "ks":    compute_ks(y_true, y_prob),
        "brier": compute_brier(y_true, y_prob),
        "ece":   compute_ece(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Fairness — disparate impact + demographic parity + equalized odds
# ---------------------------------------------------------------------------


@dataclass
class FairnessReport:
    """Per-group fairness statistics for a single protected attribute.

    ``disparate_impact`` follows the four-fifths convention: the ratio
    of the lowest-positive-rate group to the highest. A value below the
    jurisdiction's threshold (US: 0.80) is a presumptive violation.
    """

    attribute: str
    groups: List[Dict[str, Any]] = field(default_factory=list)
    disparate_impact: float = 1.0
    demographic_parity_diff: float = 0.0
    equalized_odds_diff: Optional[float] = None
    passes: bool = True


def compute_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: pd.DataFrame,
    threshold: float = 0.80,
) -> Dict[str, FairnessReport]:
    """Compute per-attribute fairness metrics.

    ``y_pred`` is the binary decision (1 = denied / positive class). For
    a credit application the protected group's positive rate is the
    *deny rate*; lower is better, so the four-fifths ratio uses
    min/max approval rate. We report the ratio of selection rates
    (approval rates) directly.

    ``protected`` must align positionally with ``y_true`` and ``y_pred``
    (i.e. ``protected.iloc[i]`` corresponds to ``y_true[i]``). Any
    arbitrary index on ``protected`` is ignored — we group by position.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) != len(y_pred) or len(y_true) != len(protected):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)} y_pred={len(y_pred)} "
            f"protected={len(protected)}"
        )
    # Reset to a positional index so groupby returns 0-based positions
    # we can use to slice the numpy arrays directly.
    protected = protected.reset_index(drop=True)

    out: Dict[str, FairnessReport] = {}
    for col in protected.columns:
        groups: List[Dict[str, Any]] = []
        approval_rates: List[float] = []
        tpr_by_group: List[float] = []
        fpr_by_group: List[float] = []
        for value, mask in protected.groupby(col, observed=True).groups.items():
            positions = np.asarray(mask, dtype=int)
            if len(positions) == 0:
                continue
            sub_pred = y_pred[positions]
            sub_true = y_true[positions]
            approval_rate = float((sub_pred == 0).mean())
            approval_rates.append(approval_rate)
            # TPR / FPR for equalized odds (positive = denial in our framing)
            pos = sub_true == 1
            neg = sub_true == 0
            tpr = float((sub_pred[pos] == 1).mean()) if pos.any() else float("nan")
            fpr = float((sub_pred[neg] == 1).mean()) if neg.any() else float("nan")
            if not np.isnan(tpr):
                tpr_by_group.append(tpr)
            if not np.isnan(fpr):
                fpr_by_group.append(fpr)
            groups.append({
                "group": _stringify(value),
                "n": int(len(positions)),
                "approval_rate": approval_rate,
                "deny_rate": float((sub_pred == 1).mean()),
                "tpr": tpr,
                "fpr": fpr,
            })
        if not approval_rates:
            continue
        di = (min(approval_rates) / max(approval_rates)) if max(approval_rates) > 0 else 1.0
        dpd = max(approval_rates) - min(approval_rates)
        eod = None
        if tpr_by_group and fpr_by_group:
            eod = max(
                max(tpr_by_group) - min(tpr_by_group),
                max(fpr_by_group) - min(fpr_by_group),
            )
        out[col] = FairnessReport(
            attribute=col,
            groups=groups,
            disparate_impact=float(di),
            demographic_parity_diff=float(dpd),
            equalized_odds_diff=float(eod) if eod is not None else None,
            passes=di >= threshold,
        )
    return out


def _stringify(value: Any) -> str:
    if isinstance(value, (np.integer, np.floating)):
        return str(value.item())
    return str(value)


# ---------------------------------------------------------------------------
# SHAP — top-k features driving model decisions
# ---------------------------------------------------------------------------


def compute_shap_summary(
    model: Any,
    X_test: pd.DataFrame,
    top_n: int = 10,
    sample_size: int = 1000,
) -> List[Dict[str, Any]]:
    """Return the top-``n`` features ranked by mean |SHAP| on a sample.

    Falls back gracefully when SHAP isn't installed or the model isn't
    tree-based without a background dataset — returns an empty list
    rather than crashing the training run.
    """
    try:
        import shap  # type: ignore
    except ImportError:
        return []

    sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    estimator = model[-1] if hasattr(model, "named_steps") else model
    try:
        explainer = shap.TreeExplainer(estimator)
        # Pipeline preprocessing -> transformed features
        if hasattr(model, "named_steps"):
            X_transformed = model[:-1].transform(sample)
        else:
            X_transformed = sample
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        names = (
            list(model[0].get_feature_names_out())
            if hasattr(model, "named_steps") and hasattr(model[0], "get_feature_names_out")
            else list(sample.columns)
        )
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n]
        return [
            {"feature": names[i] if i < len(names) else f"f_{i}",
             "mean_abs_shap": float(mean_abs[i])}
            for i in order
        ]
    except Exception:  # noqa: BLE001 — SHAP is optional, never fail training over it
        return []
