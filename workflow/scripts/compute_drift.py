#!/usr/bin/env python3
"""Compute temporal drift metrics across multiple years of data.

Usage:
  python scripts/compute_drift.py --train-years 2018,2019 --test-years 2020,2021,2022
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def psi(expected, actual, bins=10):
    """Population Stability Index between two distributions."""
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    exp_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, breakpoints)[0] / len(actual)

    # Avoid log(0)
    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def main():
    parser = argparse.ArgumentParser(description="Compute temporal drift")
    parser.add_argument("--train-years", required=True, help="Comma-separated training years")
    parser.add_argument("--test-years", required=True, help="Comma-separated test years")
    parser.add_argument("--state", default="NY", help="HMDA state filter")
    args = parser.parse_args()

    train_years = [int(y) for y in args.train_years.split(",")]
    test_years = [int(y) for y in args.test_years.split(",")]

    # Load HMDA data for each year
    from prepare_data import load_hmda

    print("Loading training data...")
    train_dfs = []
    for year in train_years:
        df = load_hmda(year, args.state)
        df["year"] = year
        train_dfs.append(df)
    train_data = pd.concat(train_dfs, ignore_index=True)

    print("Loading test data...")
    test_dfs = {}
    for year in test_years:
        df = load_hmda(year, args.state)
        df["year"] = year
        test_dfs[year] = df

    # Select common numeric features available in HMDA
    numeric_features = ["loan_amount", "income"]
    # Filter to rows with valid features
    for col in numeric_features:
        train_data[col] = pd.to_numeric(train_data[col], errors="coerce")
        for year in test_years:
            test_dfs[year][col] = pd.to_numeric(test_dfs[year][col], errors="coerce")

    train_data = train_data.dropna(subset=numeric_features + ["approved"])

    # Train models on training period
    X_train = train_data[numeric_features]
    y_train = train_data["approved"]

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=SEED, max_iter=1000)),
        ]),
        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), random_state=SEED, max_iter=300)),
        ]),
    }

    try:
        from xgboost import XGBClassifier
        models["xgboost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(n_estimators=100, max_depth=4, random_state=SEED,
                                  use_label_encoder=False, eval_metric="logloss")),
        ])
    except ImportError:
        print("WARNING: xgboost not installed, skipping.", file=sys.stderr)

    for name, pipe in models.items():
        print(f"Training {name} on {train_years}...")
        pipe.fit(X_train, y_train)

    # Test on each year
    drift_rows = []
    for year in test_years:
        df_test = test_dfs[year].dropna(subset=numeric_features + ["approved"])
        X_test = df_test[numeric_features]
        y_test = df_test["approved"]
        approval_rate = y_test.mean()

        for name, pipe in models.items():
            try:
                y_prob = pipe.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            except Exception as e:
                auc = float("nan")
                print(f"  WARNING: {name} failed on {year}: {e}", file=sys.stderr)

            drift_rows.append({
                "year": year,
                "model": name,
                "auc": auc,
                "approval_rate": approval_rate,
                "n_samples": len(y_test),
            })
            print(f"  {year} / {name}: AUC={auc:.4f}, approval_rate={approval_rate:.4f}, n={len(y_test):,}")

    df_drift = pd.DataFrame(drift_rows)
    df_drift.to_csv(RESULTS / "drift_metrics.csv", index=False)
    print(f"\nSaved: {RESULTS / 'drift_metrics.csv'}")

    # Plot AUC over time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name in models:
        sub = df_drift[df_drift["model"] == name]
        ax1.plot(sub["year"], sub["auc"], marker="o", linewidth=2,
                 label=name.replace("_", " ").title())

    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("AUC", fontsize=12)
    ax1.set_title("Model Performance Over Time", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Approval rate over time
    approval_by_year = df_drift.groupby("year")["approval_rate"].first()
    ax2.bar(approval_by_year.index.astype(str), approval_by_year.values, color="#3b82f6", alpha=0.8)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("Approval Rate", fontsize=12)
    ax2.set_title("Approval Rate Drift", fontsize=14, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "drift.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {RESULTS / 'drift.png'}")


if __name__ == "__main__":
    main()
