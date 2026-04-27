"""Tests for workflow.registry."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from workflow.registry import (
    ModelEntry,
    ModelRegistry,
    hash_training_data,
    main as cli_main,
)


def _entry(**overrides: Any) -> ModelEntry:
    base: Dict[str, Any] = dict(
        version="1.0.0",
        artifact_path="data/processed/uci_models.joblib",
        training_date="2026-04-01T00:00:00+00:00",
        training_data_sha256="a" * 64,
        hyperparameters={"n_estimators": 400, "max_depth": 6},
        validation_metrics={"AUC": 0.774, "KS": 0.45},
        feature_list=["LIMIT_BAL", "AGE", "PAY_0"],
    )
    base.update(overrides)
    return ModelEntry(**base)


def test_semver_required() -> None:
    with pytest.raises(ValueError):
        _entry(version="v1")


def test_status_must_be_valid() -> None:
    with pytest.raises(ValueError):
        _entry(status="alpha")


def test_register_then_list(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg.json")
    reg.register(_entry(version="1.0.0"))
    reg.register(_entry(version="1.1.0", training_date="2026-04-15T00:00:00+00:00"))
    versions = [e.version for e in reg.list()]
    assert versions == ["1.0.0", "1.1.0"]


def test_register_duplicate_rejected(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg.json")
    reg.register(_entry())
    with pytest.raises(ValueError):
        reg.register(_entry())


def test_promote_demotes_prior_champion(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg.json")
    reg.register(_entry(version="1.0.0", status="champion"))
    reg.register(_entry(version="1.1.0", training_date="2026-04-15T00:00:00+00:00"))
    reg.promote("1.1.0")
    a = reg.get("1.0.0")
    b = reg.get("1.1.0")
    assert a is not None and a.status == "retired"
    assert b is not None and b.status == "champion"


def test_champion_returns_only_champion(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg.json")
    reg.register(_entry(version="1.0.0"))
    assert reg.champion() is None
    reg.promote("1.0.0")
    c = reg.champion()
    assert c is not None and c.version == "1.0.0"


def test_unknown_version_promote_raises(tmp_path: Path) -> None:
    reg = ModelRegistry(tmp_path / "reg.json")
    with pytest.raises(ValueError):
        reg.promote("9.9.9")


def test_hash_training_data_deterministic(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    a.write_text("col1,col2\n1,2\n3,4\n", encoding="utf-8")
    h1 = hash_training_data(a)
    h2 = hash_training_data(a)
    assert h1 == h2 and len(h1) == 64


def test_cli_register_and_list(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    train = tmp_path / "train.csv"
    train.write_text("a,b\n1,2\n", encoding="utf-8")
    reg_path = tmp_path / "reg.json"
    rc = cli_main([
        "--path", str(reg_path), "register",
        "--model", "/tmp/m.joblib",
        "--version", "1.0.0",
        "--training-data", str(train),
        "--metrics", "AUC=0.774,KS=0.45",
        "--features", "LIMIT_BAL,AGE,PAY_0",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "registered 1.0.0" in captured.out
    rc = cli_main(["--path", str(reg_path), "list"])
    captured = capsys.readouterr()
    assert "1.0.0" in captured.out
    raw = json.loads(reg_path.read_text(encoding="utf-8"))
    assert raw["entries"][0]["version"] == "1.0.0"
