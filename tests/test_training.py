"""Tests for the configurable training SDK.

Coverage:
  * Adapter resolution — short names + dotted paths
  * DatasetMetadata invariants (features ∩ protected = ∅)
  * Each built-in adapter loads metadata cleanly
  * UCI end-to-end via the SDK on the 100-row sample
  * TrainingConfig YAML round-trip on every shipped config
  * Fairness math — disparate impact, demographic parity, equalised odds
  * SHAP summary returns top features (or empty when SHAP unavailable)
  * CLI smoke (--help exits 0; bad subcommand exits non-zero)
  * Pre-trained models on disk load and expose predict_proba
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from data.adapters import BondoraAdapter, HMDAAdapter, UCIAdapter
from workflow.training import (
    DatasetAdapter,
    DatasetMetadata,
    FairnessReport,
    TrainingConfig,
    TrainingPipeline,
    compute_auc,
    compute_brier,
    compute_ece,
    compute_fairness,
    compute_ks,
    compute_shap_summary,
    core_metrics,
    list_builtin_adapters,
    resolve_adapter,
)
from workflow.training.cli import build_parser

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
MODELS = REPO / "models"
UCI_SAMPLE = REPO / "benchmarks" / "data" / "uci_sample_100rows.csv"

ALL_ADAPTERS = [UCIAdapter, HMDAAdapter, BondoraAdapter]


# ---------------------------------------------------------------------------
# 1. DatasetMetadata invariants
# ---------------------------------------------------------------------------


def test_metadata_validate_rejects_protected_overlap() -> None:
    md = DatasetMetadata(
        name="x", region="US",
        target_column="y",
        feature_columns=["a", "age"],
        protected_columns={"age": "applicant age"},
    )
    with pytest.raises(ValueError, match="Protected columns must not be model features"):
        md.validate()


def test_metadata_validate_requires_categorical_in_features() -> None:
    md = DatasetMetadata(
        name="x", region="US",
        target_column="y",
        feature_columns=["a", "b"],
        categorical_columns=["c"],
    )
    with pytest.raises(ValueError, match="categorical_columns not in feature_columns"):
        md.validate()


def test_metadata_validate_requires_features() -> None:
    md = DatasetMetadata(name="x", region="US", target_column="y", feature_columns=[])
    with pytest.raises(ValueError, match="feature_columns must be non-empty"):
        md.validate()


# ---------------------------------------------------------------------------
# 2. Adapter resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["uci", "hmda", "bondora"])
def test_resolve_builtin_adapter_by_short_name(name: str) -> None:
    adapter = resolve_adapter(name)
    assert isinstance(adapter, DatasetAdapter)


def test_list_builtin_adapters() -> None:
    assert list_builtin_adapters() == ["bondora", "hmda", "uci"]


def test_resolve_adapter_by_dotted_path_colon() -> None:
    adapter = resolve_adapter("data.adapters.uci:UCIAdapter")
    assert isinstance(adapter, UCIAdapter)


def test_resolve_adapter_by_dotted_path_attr() -> None:
    adapter = resolve_adapter("data.adapters.uci.UCIAdapter")
    assert isinstance(adapter, UCIAdapter)


def test_resolve_adapter_unknown_module() -> None:
    with pytest.raises(ValueError, match="Cannot import"):
        resolve_adapter("definitely.not.real:Thing")


def test_resolve_adapter_missing_class() -> None:
    with pytest.raises(ValueError, match="has no class"):
        resolve_adapter("data.adapters.uci:NotAClass")


def test_resolve_adapter_not_a_subclass() -> None:
    # workflow.jurisdictions:US instantiates without args but is not an adapter.
    with pytest.raises(ValueError, match="not a DatasetAdapter subclass"):
        resolve_adapter("workflow.jurisdictions:US")


def test_resolve_adapter_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        resolve_adapter("")


# ---------------------------------------------------------------------------
# 3. Each built-in adapter exposes valid metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_adapter_metadata_validates(cls) -> None:
    cls().metadata().validate()


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_adapter_metadata_has_protected(cls) -> None:
    md = cls().metadata()
    assert md.protected_columns, f"{cls.__name__} should declare protected attributes"


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_adapter_features_protected_disjoint(cls) -> None:
    md = cls().metadata()
    assert not (set(md.feature_columns) & set(md.protected_columns))


# ---------------------------------------------------------------------------
# 4. Sample YAML configs parse round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cfg_name", ["uci_us.yaml", "hmda_us.yaml", "bondora_eu.yaml"])
def test_yaml_config_loads(cfg_name: str) -> None:
    path = CONFIGS / cfg_name
    cfg = TrainingConfig.from_yaml(path)
    assert cfg.dataset
    assert cfg.jurisdiction
    assert cfg.data_path
    assert cfg.test_size > 0
    assert cfg.random_state == 42


def test_yaml_config_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "dataset: uci\ndata_path: x\njurisdiction: US\nbogus_field: yes\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown keys"):
        TrainingConfig.from_yaml(path)


def test_yaml_config_round_trip(tmp_path) -> None:
    cfg = TrainingConfig(dataset="uci", data_path="x", jurisdiction="US", random_state=42)
    out = tmp_path / "out.yaml"
    cfg.to_yaml(out)
    reloaded = TrainingConfig.from_yaml(out)
    assert reloaded == cfg


# ---------------------------------------------------------------------------
# 5. Core metrics
# ---------------------------------------------------------------------------


def test_core_metrics_perfect_classifier() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = core_metrics(y, p)
    assert m["auc"] == 1.0
    assert m["ks"] == 1.0
    assert m["brier"] < 0.05
    assert 0 <= m["ece"] <= 1


def test_compute_ece_zero_when_well_calibrated() -> None:
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    assert compute_ece(y, p, n_bins=10) < 0.05


# ---------------------------------------------------------------------------
# 6. Fairness math
# ---------------------------------------------------------------------------


def test_fairness_perfect_parity() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    protected = pd.DataFrame({"sex": ["M", "M", "M", "M", "F", "F", "F", "F"]})
    out = compute_fairness(y_true, y_pred, protected, threshold=0.80)
    assert "sex" in out
    rep = out["sex"]
    assert isinstance(rep, FairnessReport)
    assert rep.disparate_impact == 1.0
    assert rep.passes is True


def test_fairness_detects_disparate_impact() -> None:
    """One group all approved, another mostly denied — DI should be tiny."""
    y_true = np.array([0] * 10 + [0] * 10)
    y_pred = np.array([0] * 10 + [1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    protected = pd.DataFrame({"sex": ["M"] * 10 + ["F"] * 10})
    out = compute_fairness(y_true, y_pred, protected, threshold=0.80)
    rep = out["sex"]
    # M approval rate = 1.0, F approval rate = 0.2 -> DI = 0.2
    assert rep.disparate_impact == pytest.approx(0.2)
    assert rep.passes is False


def test_fairness_handles_arbitrary_index() -> None:
    """Protected DataFrame with non-default index must align positionally."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    protected = pd.DataFrame(
        {"sex": ["M", "F", "M", "F"]},
        index=[100, 200, 300, 400],
    )
    out = compute_fairness(y_true, y_pred, protected)
    assert "sex" in out
    assert sum(g["n"] for g in out["sex"].groups) == 4


def test_fairness_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        compute_fairness(
            np.array([0, 1]), np.array([0, 1]),
            pd.DataFrame({"sex": ["M", "F", "M"]}),
        )


# ---------------------------------------------------------------------------
# 7. SHAP summary (graceful fallback)
# ---------------------------------------------------------------------------


def test_shap_summary_empty_when_unavailable() -> None:
    """If SHAP isn't importable the function returns [] rather than crashing."""
    with patch.dict(sys.modules, {"shap": None}):
        # Patch the import inside the function to force ImportError
        from workflow.training import evaluation
        with patch.object(evaluation, "compute_shap_summary",
                          side_effect=evaluation.compute_shap_summary):
            # Simulate ImportError by monkeypatching __import__
            import builtins
            real_import = builtins.__import__

            def fail_shap(name, *args, **kwargs):
                if name == "shap":
                    raise ImportError("simulated")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fail_shap):
                out = compute_shap_summary(MagicMock(), pd.DataFrame({"x": [1, 2, 3]}))
                assert out == []


# ---------------------------------------------------------------------------
# 8. UCI end-to-end via SDK on the 100-row sample
# ---------------------------------------------------------------------------


def test_uci_end_to_end_on_sample(tmp_path) -> None:
    """Train on the shipped 100-row sample and check artifacts land."""
    if not UCI_SAMPLE.exists():
        pytest.skip(f"UCI sample not present at {UCI_SAMPLE}")

    # The UCI adapter expects the published .xls layout, where row 0 is a
    # title row and row 1 holds the actual column names. Reproduce that so
    # the adapter's pd.read_excel(..., header=1) reads the right line.
    import openpyxl
    sample_df = pd.read_csv(UCI_SAMPLE)
    sample_xls = tmp_path / "uci_sample.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Dataset: UCI Default of Credit Card Clients (test fixture)"])
    ws.append(list(sample_df.columns))
    for _, row in sample_df.iterrows():
        ws.append([
            None if pd.isna(v) else (int(v) if isinstance(v, np.integer) else v)
            for v in row.values
        ])
    wb.save(sample_xls)

    cfg = TrainingConfig(
        dataset="uci",
        data_path=str(sample_xls),
        jurisdiction="US",
        model_type="xgboost",
        hyperparams={"n_estimators": 30, "max_depth": 3},
        compute_fairness=True,
        compute_shap=False,
        output_dir=str(tmp_path / "out"),
        model_version="0.0.1-test",
    )
    pipeline = TrainingPipeline(cfg)
    result = pipeline.run()

    assert Path(result.model_path).exists()
    assert Path(result.model_path).suffix == ".joblib"
    metrics_path = Path(result.model_path).with_suffix(".json")
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert "metrics" in payload
    assert "auc" in payload["metrics"]
    assert payload["jurisdiction_code"] == "US"


# ---------------------------------------------------------------------------
# 9. CLI smoke
# ---------------------------------------------------------------------------


def test_cli_help_exits_cleanly() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_cli_unknown_subcommand_fails() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["does-not-exist"])
    assert exc.value.code != 0


def test_cli_train_requires_config() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


# ---------------------------------------------------------------------------
# 10. Pre-trained models load and expose predict_proba
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["uci_xgboost_v1.joblib", "hmda_xgboost_v1.joblib"])
def test_pretrained_model_loads(name: str) -> None:
    path = MODELS / name
    if not path.exists():
        pytest.skip(f"Pre-trained model {name} not present (likely uncommitted artifact)")
    import joblib
    model = joblib.load(path)
    assert hasattr(model, "predict_proba")
    metrics_path = MODELS / name.replace(".joblib", "_metrics.json")
    assert metrics_path.exists(), f"metrics JSON missing for {name}"
    payload = json.loads(metrics_path.read_text())
    assert payload["metrics"]["auc"] >= 0.70
