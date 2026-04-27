#!/usr/bin/env python3
"""Compute accuracy metrics (AUC, KS, Brier, ECE) for trained models.

Usage:
  python scripts/compute_accuracy.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def ks_statistic(y_true, y_score):
    """Kolmogorov-Smirnov: max |TPR - FPR| across thresholds."""
    order = np.argsort(-np.asarray(y_score))
    y_sorted = np.asarray(y_true)[order]
    tpr = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    fpr = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(tpr - fpr)))


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE — binned calibration error."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / len(y_prob)) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def main():
    # Load models
    bundle_path = DATA_PROC / "models.joblib"
    if not bundle_path.exists():
        print(f"ERROR: No trained models found at {bundle_path}", file=sys.stderr)
        print("Run: python scripts/train_classical.py first", file=sys.stderr)
        sys.exit(1)

    bundle = joblib.load(bundle_path)
    models = bundle["models"]
    features = bundle["features"]

    # Recreate the same test split
    from prepare_data import load_uci
    df = load_uci()
    target = "default payment next month"
    X = df[features].copy()
    y = df[target].copy()

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED,
    )

    # Compute metrics
    rows = []
    for name, pipe in models.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        ks = ks_statistic(y_test.values, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        ece = expected_calibration_error(y_test.values, y_prob)

        rows.append({"model": name, "auc": auc, "ks": ks, "brier": brier, "ece": ece})
        print(f"{name:25s}  AUC={auc:.4f}  KS={ks:.4f}  Brier={brier:.4f}  ECE={ece:.4f}")

    # Save CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "accuracy.csv", index=False)
    print(f"\nSaved: {RESULTS / 'accuracy.csv'}")

    # ROC curves plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pipe in models.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        label = f"{name.replace('_', ' ').title()} (AUC={auc:.3f})"
        ax.plot(fpr, tpr, label=label, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Classical Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "roc_curves.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {RESULTS / 'roc_curves.png'}")


if __name__ == "__main__":
    main()
