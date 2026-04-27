#!/usr/bin/env python3
"""Compute fairness metrics (DI ratio, DP diff, EO diff) for trained models.

Usage:
  python scripts/compute_fairness.py --dataset uci
  python scripts/compute_fairness.py --dataset hmda --year 2022
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def disparate_impact_ratio(y_pred, sensitive_feature):
    """Compute disparate impact ratio: min(group rate) / max(group rate)."""
    groups = pd.Series(sensitive_feature)
    preds = pd.Series(y_pred)
    rates = preds.groupby(groups).mean()
    if rates.max() == 0:
        return float("nan")
    return float(rates.min() / rates.max())


def demographic_parity_diff(y_pred, sensitive_feature):
    """Max group approval rate - min group approval rate."""
    groups = pd.Series(sensitive_feature)
    preds = pd.Series(y_pred)
    rates = preds.groupby(groups).mean()
    return float(rates.max() - rates.min())


def selection_rates(y_pred, sensitive_feature):
    """Approval rate per group."""
    groups = pd.Series(sensitive_feature)
    preds = pd.Series(y_pred)
    return preds.groupby(groups).mean().to_dict()


def compute_fairness_uci(models, X_test, y_test):
    """Compute fairness metrics on UCI test set."""
    # Protected attributes
    sex_map = {1: "Male", 2: "Female"}
    edu_map = {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}
    age_bins = pd.cut(X_test["AGE"], bins=[0, 30, 50, 100], labels=["Under 30", "30-50", "Over 50"])

    sex = X_test["SEX"].map(sex_map)
    education = X_test["EDUCATION"].map(edu_map)

    rows = []
    di_rows = []

    for name, pipe in models.items():
        y_pred = pipe.predict(X_test)
        y_approve = (1 - y_pred)  # 0=approve in our convention (predict default → deny)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        # Convert to approval: approve if PD < 0.5
        y_approve = (y_prob < 0.5).astype(int)

        for attr_name, attr_vals in [("sex", sex), ("education", education), ("age", age_bins)]:
            di = disparate_impact_ratio(y_approve, attr_vals)
            dp = demographic_parity_diff(y_approve, attr_vals)
            rates = selection_rates(y_approve, attr_vals)

            rows.append({
                "model": name,
                "attribute": attr_name,
                "di_ratio": di,
                "dp_diff": dp,
                "four_fifths_pass": di >= 0.80 if not np.isnan(di) else None,
                **{f"rate_{k}": v for k, v in rates.items()},
            })

            di_rows.append({
                "model": name,
                "attribute": attr_name,
                "di_ratio": di,
                "threshold": 0.80,
                "pass": di >= 0.80 if not np.isnan(di) else None,
            })

        print(f"{name}: sex DI={disparate_impact_ratio(y_approve, sex):.3f}, "
              f"edu DI={disparate_impact_ratio(y_approve, education):.3f}")

    pd.DataFrame(rows).to_csv(RESULTS / "fairness.csv", index=False)
    pd.DataFrame(di_rows).to_csv(RESULTS / "disparate_impact.csv", index=False)
    print(f"\nSaved: {RESULTS / 'fairness.csv'}")
    print(f"Saved: {RESULTS / 'disparate_impact.csv'}")

    # Plot
    plot_disparate_impact(di_rows)


def plot_disparate_impact(di_rows):
    """Bar chart of DI ratios with four-fifths threshold."""
    df = pd.DataFrame(di_rows)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, attr in enumerate(["sex", "education", "age"]):
        ax = axes[i]
        sub = df[df["attribute"] == attr]
        ax.bar(sub["model"].str.replace("_", "\n"), sub["di_ratio"], color="#3b82f6", alpha=0.8)
        ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1.5, label="Four-fifths threshold")
        ax.set_title(f"DI Ratio: {attr.title()}", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Disparate Impact Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS / "disparate_impact.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {RESULTS / 'disparate_impact.png'}")


def main():
    parser = argparse.ArgumentParser(description="Compute fairness metrics")
    parser.add_argument("--dataset", default="uci", choices=["uci", "hmda"])
    parser.add_argument("--year", type=int, help="Year for HMDA")
    args = parser.parse_args()

    if args.dataset == "uci":
        bundle = joblib.load(DATA_PROC / "models.joblib")
        models = bundle["models"]
        features = bundle["features"]

        from prepare_data import load_uci
        df = load_uci()
        X = df[features].copy()
        y = df["default payment next month"].copy()
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=SEED,
        )
        compute_fairness_uci(models, X_test, y_test)
    else:
        print("HMDA fairness not yet implemented.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
