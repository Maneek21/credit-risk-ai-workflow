"""Configurable training pipeline.

A :class:`TrainingPipeline` orchestrates download → load → preprocess →
split → train → evaluate → save → register. Configuration is supplied
via :class:`TrainingConfig`, which is loadable from YAML.

The SDK is opt-in. The existing :class:`workflow.CreditWorkflow` API
continues to accept any joblib-serialized sklearn / xgboost model
without going through this pipeline.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datasets import DatasetAdapter, resolve_adapter
from .evaluation import (
    FairnessReport,
    compute_fairness,
    compute_shap_summary,
    core_metrics,
)


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """All parameters for a single training run.

    Loadable from YAML via :meth:`from_yaml`. Every field has a default
    so a minimal YAML need only specify ``dataset``, ``data_path``, and
    ``jurisdiction``.
    """

    dataset: str                            # "uci" / "hmda" / "bondora" / dotted-path
    data_path: str                          # Where the raw file lives
    jurisdiction: str                       # ISO code: "US", "EU", "IN", ...

    model_type: str = "xgboost"             # "xgboost" / "logistic_regression" / "mlp"
    hyperparams: Dict[str, Any] = field(default_factory=dict)

    test_size: float = 0.20
    val_size: float = 0.10
    random_state: int = 42

    compute_fairness: bool = True
    compute_shap: bool = True
    compute_drift: bool = False
    temporal_column: Optional[str] = None

    output_dir: str = "./output"
    model_version: str = "1.0.0"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        """Load a config from a YAML file. Requires PyYAML."""
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML required for YAML configs: pip install pyyaml") from exc
        text = Path(path).read_text(encoding="utf-8")
        raw = yaml.safe_load(text) or {}
        # Drop unknown keys with a clear error rather than silently ignoring.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(raw.keys()) - known
        if unknown:
            raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    def to_yaml(self, path: str | Path) -> None:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML required for YAML configs: pip install pyyaml") from exc
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")


@dataclass
class TrainingResult:
    """Output of a training run — serialisable to JSON."""

    model_path: str
    metrics: Dict[str, float]
    fairness: Optional[Dict[str, Dict[str, Any]]]
    shap_top_features: List[Dict[str, Any]]
    config_used: TrainingConfig
    dataset_name: str
    dataset_region: str
    rows_total: int
    rows_train: int
    rows_test: int
    feature_count: int
    training_date_utc: str
    data_sha256: str
    jurisdiction_code: str
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["config_used"] = asdict(self.config_used)
        return d

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TrainingPipeline:
    """Train a credit-risk model from a :class:`TrainingConfig`."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.adapter: DatasetAdapter = resolve_adapter(config.dataset)
        # Lazy-resolve the jurisdiction so adapter resolution failures
        # surface before we try to load the regulatory module.
        self.jurisdiction = self._resolve_jurisdiction(config.jurisdiction)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingPipeline":
        return cls(TrainingConfig.from_yaml(path))

    @staticmethod
    def _resolve_jurisdiction(code: str):
        from workflow.jurisdictions import ALL_JURISDICTIONS
        if code not in ALL_JURISDICTIONS:
            raise ValueError(
                f"Unknown jurisdiction code {code!r}; "
                f"available: {sorted(ALL_JURISDICTIONS)}"
            )
        return ALL_JURISDICTIONS[code]()

    # -- Stage helpers ------------------------------------------------------

    def _build_estimator(self, hyperparams: Dict[str, Any]):
        """Construct an sklearn-compatible classifier per ``model_type``."""
        seed = self.config.random_state
        params = dict(hyperparams)
        params.pop("scale_pos_weight", None) if self.config.model_type != "xgboost" else None
        if self.config.model_type == "xgboost":
            from xgboost import XGBClassifier
            params.setdefault("n_estimators", 300)
            params.setdefault("max_depth", 5)
            params.setdefault("learning_rate", 0.1)
            params.setdefault("eval_metric", "logloss")
            params.setdefault("tree_method", "hist")
            params["random_state"] = seed
            params.setdefault("n_jobs", -1)
            return XGBClassifier(**params)
        if self.config.model_type == "logistic_regression":
            params.setdefault("max_iter", 2000)
            params.setdefault("n_jobs", -1)
            params["random_state"] = seed
            return LogisticRegression(**params)
        if self.config.model_type == "mlp":
            params.setdefault("hidden_layer_sizes", (64, 32))
            params.setdefault("activation", "relu")
            params.setdefault("solver", "adam")
            params.setdefault("max_iter", 200)
            params.setdefault("early_stopping", True)
            params["random_state"] = seed
            return MLPClassifier(**params)
        raise ValueError(f"Unsupported model_type: {self.config.model_type!r}")

    def _build_preprocessor(
        self,
        feature_cols: List[str],
        categorical_cols: List[str],
        scale: bool,
    ) -> ColumnTransformer:
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler() if scale else "passthrough", numeric_cols),
                ("cat",
                 OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                 categorical_cols),
            ],
            remainder="drop",
        )

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        """Stable SHA-256 of a DataFrame content (for reproducibility audit)."""
        h = hashlib.sha256()
        h.update(",".join(df.columns).encode("utf-8"))
        h.update(b"|")
        h.update(df.to_csv(index=False).encode("utf-8"))
        return h.hexdigest()

    def download(self, dest_dir: Optional[str] = None) -> str:
        """Run only the adapter download step. Returns local file path."""
        dest = dest_dir or str(Path(self.config.data_path).parent)
        return self.adapter.download(dest)

    # -- Main entry ---------------------------------------------------------

    def run(self) -> TrainingResult:
        """Execute the full pipeline. Writes model + metrics to ``output_dir``."""
        cfg = self.config
        meta = self.adapter.metadata()
        meta.validate()

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        X, y, protected = self.adapter.load_with_protected(cfg.data_path)
        rows_total = len(X)
        if rows_total == 0:
            raise RuntimeError(f"Adapter loaded 0 rows from {cfg.data_path!r}")

        data_hash = self._hash_dataframe(X.assign(__y=y))

        # Stratified train/test split. We reserve a val_size slice from
        # the train portion for early-stopping models in future expansion;
        # the current XGBoost / LR / MLP path doesn't need a separate val
        # split, but we honour the config so reports stay accurate.
        X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
            X, y, protected,
            test_size=cfg.test_size,
            stratify=y,
            random_state=cfg.random_state,
        )

        scale = cfg.model_type in ("logistic_regression", "mlp")
        preprocessor = self._build_preprocessor(
            feature_cols=meta.feature_columns,
            categorical_cols=meta.categorical_columns,
            scale=scale,
        )
        # Auto-compute scale_pos_weight when requested as "auto".
        hp = dict(cfg.hyperparams)
        if cfg.model_type == "xgboost" and hp.get("scale_pos_weight") == "auto":
            pos = float(y_train.sum())
            neg = float((y_train == 0).sum())
            hp["scale_pos_weight"] = neg / max(pos, 1.0)

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", self._build_estimator(hp)),
        ])
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = core_metrics(y_test.values, y_prob)

        fairness_serialised: Optional[Dict[str, Dict[str, Any]]] = None
        if cfg.compute_fairness and not prot_test.empty:
            fr = compute_fairness(
                y_test.values, y_pred, prot_test,
                threshold=self.jurisdiction.fairness_threshold,
            )
            fairness_serialised = {
                attr: asdict(report) for attr, report in fr.items()
            }

        shap_summary: List[Dict[str, Any]] = []
        if cfg.compute_shap:
            shap_summary = compute_shap_summary(model, X_test)

        # Save model + metrics
        model_filename = f"{meta.name}_{cfg.model_type}_v{cfg.model_version.replace('.', '_')}.joblib"
        model_path = output_dir / model_filename
        joblib.dump(model, model_path)

        result = TrainingResult(
            model_path=str(model_path),
            metrics=metrics,
            fairness=fairness_serialised,
            shap_top_features=shap_summary,
            config_used=cfg,
            dataset_name=meta.name,
            dataset_region=meta.region,
            rows_total=rows_total,
            rows_train=len(X_train),
            rows_test=len(X_test),
            feature_count=len(meta.feature_columns),
            training_date_utc=datetime.now(timezone.utc).isoformat(),
            data_sha256=data_hash,
            jurisdiction_code=self.jurisdiction.code,
            model_version=cfg.model_version,
        )
        result.write_json(model_path.with_suffix(".json"))
        return result
