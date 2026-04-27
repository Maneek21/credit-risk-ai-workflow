#!/usr/bin/env python3
"""Compute SHAP values and summary plots for trained models.

Usage:
  python scripts/compute_shap.py
  python scripts/compute_shap.py --bootstrap 50   # stability analysis
"""
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

EXPLAIN_N = 200
BACKGROUND_N = 100

# Human-readable feature labels
LABEL_MAP = {
    "num__LIMIT_BAL": "Credit Limit",
    "num__AGE": "Age",
    "num__PAY_0": "Repayment Status (Sep)",
    "num__PAY_2": "Repayment Status (Aug)",
    "num__PAY_3": "Repayment Status (Jul)",
    "num__PAY_4": "Repayment Status (Jun)",
    "num__PAY_5": "Repayment Status (May)",
    "num__PAY_6": "Repayment Status (Apr)",
    "num__BILL_AMT1": "Bill Amount (Sep)",
    "num__BILL_AMT2": "Bill Amount (Aug)",
    "num__BILL_AMT3": "Bill Amount (Jul)",
    "num__BILL_AMT4": "Bill Amount (Jun)",
    "num__BILL_AMT5": "Bill Amount (May)",
    "num__BILL_AMT6": "Bill Amount (Apr)",
    "num__PAY_AMT1": "Payment Amount (Sep)",
    "num__PAY_AMT2": "Payment Amount (Aug)",
    "num__PAY_AMT3": "Payment Amount (Jul)",
    "num__PAY_AMT4": "Payment Amount (Jun)",
    "num__PAY_AMT5": "Payment Amount (May)",
    "num__PAY_AMT6": "Payment Amount (Apr)",
    "cat__SEX_2": "Sex: Female",
    "cat__EDUCATION_1": "Education: Graduate",
    "cat__EDUCATION_2": "Education: University",
    "cat__EDUCATION_3": "Education: High School",
    "cat__EDUCATION_4": "Education: Other",
    "cat__MARRIAGE_1": "Married",
    "cat__MARRIAGE_2": "Single",
    "cat__MARRIAGE_3": "Marital: Other",
}


def _transform(pipe, X):
    Xp = pipe.named_steps["pre"].transform(X)
    return Xp.toarray() if hasattr(Xp, "toarray") else Xp


def _feature_names(pipe):
    return pipe.named_steps["pre"].get_feature_names_out().tolist()


def _clean_names(raw_names):
    return [LABEL_MAP.get(n, n.replace("num__", "").replace("cat__", "")) for n in raw_names]


def _shap_values(name, pipe, X_explain, X_background):
    """Compute SHAP values using the appropriate explainer."""
    Xexp_t = _transform(pipe, X_explain)
    clf = pipe.named_steps["clf"]

    if name == "logistic_regression":
        explainer = shap.LinearExplainer(clf, _transform(pipe, X_background))
        return np.asarray(explainer.shap_values(Xexp_t))

    # XGBoost and MLP: use KernelExplainer for compatibility
    bg_t = _transform(pipe, X_background)
    f = lambda x: clf.predict_proba(x)[:, 1]
    explainer = shap.KernelExplainer(
        f, shap.sample(bg_t, min(50, len(bg_t)), random_state=SEED)
    )
    return np.asarray(explainer.shap_values(Xexp_t, nsamples=100, silent=True))


def compute_and_plot(models, X_test, X_train):
    """Compute SHAP values and generate summary plots."""
    rng = np.random.default_rng(SEED)
    explain_idx = rng.choice(len(X_test), size=min(EXPLAIN_N, len(X_test)), replace=False)
    X_explain = X_test.iloc[explain_idx].copy()
    X_background = X_train.sample(BACKGROUND_N, random_state=SEED)

    model_titles = {
        "logistic_regression": "Logistic Regression",
        "xgboost": "XGBoost",
        "mlp": "Multi-Layer Perceptron",
    }

    for name, pipe in models.items():
        print(f"  Computing SHAP: {name} ...")
        sv = _shap_values(name, pipe, X_explain, X_background)
        raw_feat = _feature_names(pipe)
        clean_feat = _clean_names(raw_feat)
        Xexp_t = _transform(pipe, X_explain)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            sv, Xexp_t, feature_names=clean_feat,
            show=False, max_display=15, plot_size=None,
        )
        ax = plt.gca()
        title = model_titles.get(name, name.replace("_", " ").title())
        ax.set_title(f"SHAP Feature Importance: {title}", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("SHAP Value (impact on default probability)", fontsize=11)
        plt.tight_layout()

        out = RESULTS / f"shap_summary_{name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close("all")
        print(f"    -> {out}")


def bootstrap_stability(models, X_train, y_train, n_bootstrap=50):
    """Bootstrap resampling to measure SHAP rank stability."""
    print(f"\nRunning bootstrap stability analysis (n={n_bootstrap})...")
    rng = np.random.default_rng(SEED)
    results = []

    for name, pipe in models.items():
        print(f"  {name}...")
        importance_ranks = []

        for i in range(n_bootstrap):
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train.iloc[idx]
            y_boot = y_train.iloc[idx]

            # Clone and refit
            from sklearn.base import clone
            pipe_clone = clone(pipe)
            pipe_clone.fit(X_boot, y_boot)

            # Get feature importances (absolute SHAP mean or coefficients)
            clf = pipe_clone.named_steps["clf"]
            if name == "logistic_regression":
                imp = np.abs(clf.coef_[0])
            elif name == "xgboost":
                imp = clf.feature_importances_
            else:
                # MLP: use first-layer weights as proxy
                imp = np.abs(clf.coefs_[0]).sum(axis=1)

            importance_ranks.append(np.argsort(-imp))

        # Pairwise Spearman correlations
        rhos = []
        for i in range(len(importance_ranks)):
            for j in range(i + 1, len(importance_ranks)):
                rho, _ = spearmanr(importance_ranks[i], importance_ranks[j])
                rhos.append(rho)

        mean_rho = np.mean(rhos)
        results.append({"model": name, "mean_spearman_rho": mean_rho, "n_bootstrap": n_bootstrap})
        print(f"    Mean Spearman ρ = {mean_rho:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS / "shap_stability.csv", index=False)
    print(f"\nSaved: {RESULTS / 'shap_stability.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP values")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap samples for stability")
    args = parser.parse_args()

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
    val_frac = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=SEED,
    )

    compute_and_plot(models, X_test, X_train)

    if args.bootstrap > 0:
        bootstrap_stability(models, X_train, y_train, args.bootstrap)


if __name__ == "__main__":
    main()
