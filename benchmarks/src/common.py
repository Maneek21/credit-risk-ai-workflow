"""Shared utilities — paths, RNG, metrics — reused across all phases.

CLAUDE.md working rules:
- random_state=42 everywhere
- save intermediate state — every phase writes a results/ artifact
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
REPORT = ROOT / "report"

for d in (DATA_RAW, DATA_PROC, RESULTS, REPORT):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# UCI columns
UCI_TARGET = "default payment next month"
UCI_FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# ---------- Metrics ----------

def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max |TPR - FPR| across thresholds."""
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    tpr = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    fpr = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(tpr - fpr)))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE — bin predictions, weight by bin size."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    n = len(y_prob)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def core_metrics(y_true, y_prob) -> dict:
    """AUC + KS + Brier + ECE."""
    from sklearn.metrics import roc_auc_score, brier_score_loss
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ks": ks_statistic(np.asarray(y_true), np.asarray(y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(np.asarray(y_true), np.asarray(y_prob)),
    }


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))


def banner(msg: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{msg}\n{bar}")
