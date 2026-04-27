#!/usr/bin/env python3
"""Train classical ML models (LR, XGBoost, MLP) on prepared data.

Usage:
  python scripts/train_classical.py --dataset uci
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]


def _build_preprocessor():
    """StandardScaler for numeric, OneHotEncoder for categorical."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(drop="if_binary", sparse_output=False), CATEGORICAL),
        ]
    )


def _build_models():
    """Return dict of name → sklearn Pipeline."""
    pre = _build_preprocessor

    models = {
        "logistic_regression": Pipeline([
            ("pre", pre()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ]),
        "xgboost": None,  # built below if available
        "mlp": Pipeline([
            ("pre", pre()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                random_state=SEED,
                early_stopping=True,
                validation_fraction=0.1,
            )),
        ]),
    }

    # XGBoost is optional — skip gracefully if not installed
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = Pipeline([
            ("pre", pre()),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                random_state=SEED,
                use_label_encoder=False,
                eval_metric="logloss",
            )),
        ])
    except ImportError:
        print("WARNING: xgboost not installed. Skipping XGBoost model.", file=sys.stderr)
        del models["xgboost"]

    return models


def train_uci():
    """Train all models on UCI data."""
    from prepare_data import load_uci
    from sklearn.model_selection import train_test_split

    df = load_uci()
    target = "default payment next month"
    X = df[FEATURES].copy()
    y = df[target].copy()

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED,
    )
    val_frac = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=SEED,
    )

    models = _build_models()
    fitted = {}

    for name, pipe in models.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)

        # Quick validation score
        from sklearn.metrics import roc_auc_score
        val_proba = pipe.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        print(f"  Validation AUC: {val_auc:.4f}")

        fitted[name] = pipe

    # Save all models
    out = DATA_PROC / "models.joblib"
    joblib.dump({"models": fitted, "features": FEATURES, "seed": SEED}, out)
    print(f"\nAll models saved to {out}")

    return fitted


def main():
    parser = argparse.ArgumentParser(description="Train classical ML models")
    parser.add_argument("--dataset", default="uci", choices=["uci", "hmda"])
    args = parser.parse_args()

    if args.dataset == "uci":
        train_uci()
    else:
        print("HMDA training not yet implemented.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
