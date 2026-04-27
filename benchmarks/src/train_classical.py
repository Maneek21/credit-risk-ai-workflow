"""Phase 2 — Classical ML baseline on UCI.

Trains LR, XGBoost, MLP on the same 70/15/15 split. Evaluates AUC, KS, Brier, ECE
on the held-out test set. Writes:
  results/02_accuracy.csv
  results/02_calibration.png
  results/02_roc_curves.png
  data/processed/uci_split_indices.json   (so Phase 3 reuses identical splits)
  data/processed/uci_models.joblib        (so Phase 3 reuses fit pipelines)
"""
from __future__ import annotations

import warnings

import joblib
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA_PROC, RESULTS, SEED, banner, core_metrics, write_json
from data_uci import split_uci, UCI_FEATURES

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC = [c for c in UCI_FEATURES if c not in CATEGORICAL]


def make_preprocessor(scale: bool) -> ColumnTransformer:
    if scale:
        num = StandardScaler()
    else:
        num = "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num, NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("pre", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED, n_jobs=-1)),
        ]),
        "xgboost": Pipeline([
            ("pre", make_preprocessor(scale=False)),
            ("clf", XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", tree_method="hist",
                random_state=SEED, n_jobs=-1,
            )),
        ]),
        "mlp": Pipeline([
            ("pre", make_preprocessor(scale=True)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu", solver="adam",
                max_iter=200, early_stopping=True, validation_fraction=0.15,
                random_state=SEED,
            )),
        ]),
    }


def main() -> None:
    banner("PHASE 2 — Classical ML baseline (UCI)")
    X_train, X_val, X_test, y_train, y_val, y_test = split_uci()
    print(f"train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # Save split indices for Phase 3 reuse (defensive — Phase 3 will re-split deterministically anyway)
    write_json(
        {"train_idx": X_train.index.tolist(), "val_idx": X_val.index.tolist(),
         "test_idx": X_test.index.tolist()},
        DATA_PROC / "uci_split_indices.json",
    )

    models = build_models()
    rows = []
    fitted = {}
    test_probs = {}

    for name, pipe in models.items():
        print(f"  fitting {name} ...")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        m = core_metrics(y_test, proba)
        m["model"] = name
        rows.append(m)
        fitted[name] = pipe
        test_probs[name] = proba
        print(f"    AUC={m['auc']:.4f}  KS={m['ks']:.4f}  Brier={m['brier']:.4f}  ECE={m['ece']:.4f}")

    df = pd.DataFrame(rows)[["model", "auc", "ks", "brier", "ece"]]
    df.to_csv(RESULTS / "02_accuracy.csv", index=False)
    print(f"\nwrote {RESULTS/'02_accuracy.csv'}")
    print(df.to_string(index=False))

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, p in test_probs.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={df.set_index('model').loc[name,'auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Phase 2 — ROC curves (UCI test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "02_roc_curves.png", dpi=130)
    plt.close(fig)

    # Calibration (reliability diagram)
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, p in test_probs.items():
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical default rate")
    ax.set_title("Phase 2 — Calibration (reliability diagram)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "02_calibration.png", dpi=130)
    plt.close(fig)

    # Persist fitted pipelines for Phase 3
    joblib.dump({"models": fitted, "test_probs": test_probs}, DATA_PROC / "uci_models.joblib")
    print(f"persisted fitted pipelines -> {DATA_PROC/'uci_models.joblib'}")


if __name__ == "__main__":
    main()
