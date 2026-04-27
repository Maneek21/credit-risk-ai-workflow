"""Model registry — every audited decision must link to a known model version.

SR 11-7 §IV requires traceability between decisions and the specific model
version that produced them. This module provides:

  * A ``ModelEntry`` dataclass describing each registered model.
  * A ``ModelRegistry`` that persists entries to ``model_registry.json``
    (alongside the model artefacts).
  * A ``__main__`` CLI: register, list, promote, retire, get-current.

Status lifecycle::

    development → challenger → champion → retired

Only one model may be ``champion`` at a time. ``CreditWorkflow`` resolves
"the current production model" by looking up the champion at startup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_REGISTRY_PATH = Path("data/model_registry.json")

VALID_STATUSES = ("development", "challenger", "champion", "retired")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


@dataclass
class ModelEntry:
    """One row in the registry.

    Attributes:
        version: Semantic version, e.g. "1.0.0".
        artifact_path: Filesystem path to the joblib bundle.
        training_date: ISO-8601 date the model was trained.
        training_data_sha256: SHA-256 of the training CSV (hex).
        hyperparameters: Dict of hyperparameters used at training time.
        validation_metrics: AUC/KS/Brier on the held-out validation set.
        feature_list: Ordered feature names (must match model expectations).
        status: One of VALID_STATUSES.
        notes: Free-text description (e.g. "fixed missing-MARRIAGE bug").
    """

    version: str
    artifact_path: str
    training_date: str
    training_data_sha256: str
    hyperparameters: Dict[str, Any]
    validation_metrics: Dict[str, float]
    feature_list: List[str]
    status: str = "development"
    notes: str = ""

    def __post_init__(self) -> None:
        if not SEMVER_RE.match(self.version):
            raise ValueError(f"version must be semver (X.Y.Z), got: {self.version}")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got: {self.status}")


def hash_training_data(path: str | Path) -> str:
    """Return the SHA-256 of a file as a hex string. Streams in 1 MiB chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class _RegistryFile:
    """JSON-serialisable container for the on-disk registry."""

    entries: List[Dict[str, Any]] = field(default_factory=list)
    updated_utc: str = ""


class ModelRegistry:
    """Persistent model registry.

    Args:
        path: JSON file path. Created on first write.
    """

    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH) -> None:
        self.path = Path(path)

    # ---------- I/O ----------
    def _load(self) -> _RegistryFile:
        if not self.path.exists():
            return _RegistryFile()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return _RegistryFile(
            entries=list(raw.get("entries", [])),
            updated_utc=str(raw.get("updated_utc", "")),
        )

    def _save(self, file: _RegistryFile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file.updated_utc = datetime.now(timezone.utc).isoformat()
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(file), indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # ---------- CRUD-ish ops ----------
    def register(self, entry: ModelEntry) -> ModelEntry:
        """Add a new model. Raises if the version already exists."""
        file = self._load()
        if any(e["version"] == entry.version for e in file.entries):
            raise ValueError(f"version {entry.version} already registered")
        file.entries.append(asdict(entry))
        self._save(file)
        return entry

    def list(self) -> List[ModelEntry]:
        """Return all registered entries, ordered by training_date asc."""
        file = self._load()
        ordered = sorted(file.entries, key=lambda e: e.get("training_date", ""))
        return [ModelEntry(**e) for e in ordered]

    def get(self, version: str) -> Optional[ModelEntry]:
        for e in self._load().entries:
            if e["version"] == version:
                return ModelEntry(**e)
        return None

    def champion(self) -> Optional[ModelEntry]:
        """Return the single model in champion status, or None."""
        for e in self._load().entries:
            if e["status"] == "champion":
                return ModelEntry(**e)
        return None

    def promote(self, version: str) -> ModelEntry:
        """Set ``version`` to ``champion``; demote any prior champion to retired."""
        file = self._load()
        target = next((e for e in file.entries if e["version"] == version), None)
        if target is None:
            raise ValueError(f"unknown version: {version}")
        for e in file.entries:
            if e["status"] == "champion" and e["version"] != version:
                e["status"] = "retired"
        target["status"] = "champion"
        self._save(file)
        return ModelEntry(**target)

    def retire(self, version: str) -> ModelEntry:
        file = self._load()
        target = next((e for e in file.entries if e["version"] == version), None)
        if target is None:
            raise ValueError(f"unknown version: {version}")
        target["status"] = "retired"
        self._save(file)
        return ModelEntry(**target)


# --------------- CLI ---------------

def _parse_metrics(raw: str) -> Dict[str, float]:
    """Parse 'AUC=0.774,KS=0.45,Brier=0.16' into a dict."""
    out: Dict[str, float] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"metric chunk missing '=': {chunk!r}")
        k, v = chunk.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def _parse_features(raw: str) -> List[str]:
    return [f.strip() for f in raw.split(",") if f.strip()]


def _print_table(entries: List[ModelEntry]) -> None:
    if not entries:
        print("(no models registered)")
        return
    print(f"{'VERSION':<10} {'STATUS':<12} {'TRAINED':<25} AUC")
    print("-" * 60)
    for e in entries:
        auc = e.validation_metrics.get("AUC", float("nan"))
        print(f"{e.version:<10} {e.status:<12} {e.training_date:<25} {auc:.4f}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="workflow.registry", description="Model registry CLI")
    parser.add_argument("--path", default=str(DEFAULT_REGISTRY_PATH),
                        help="Registry JSON path (default: data/model_registry.json)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("register", help="register a new model version")
    p_reg.add_argument("--model", required=True, help="path to model joblib")
    p_reg.add_argument("--version", required=True)
    p_reg.add_argument("--training-data", required=True,
                       help="path to training CSV (hashed for fingerprint)")
    p_reg.add_argument("--metrics", required=True,
                       help="comma-separated, e.g. 'AUC=0.774,KS=0.45'")
    p_reg.add_argument("--features", required=True,
                       help="comma-separated feature list")
    p_reg.add_argument("--hyperparameters", default="{}",
                       help="JSON object")
    p_reg.add_argument("--notes", default="")

    sub.add_parser("list", help="list all registered models")

    p_get = sub.add_parser("get", help="show one entry")
    p_get.add_argument("version")

    p_pro = sub.add_parser("promote", help="promote a version to champion")
    p_pro.add_argument("--version", required=True)

    p_ret = sub.add_parser("retire", help="retire a version")
    p_ret.add_argument("--version", required=True)

    sub.add_parser("champion", help="show current champion")

    args = parser.parse_args(argv)
    reg = ModelRegistry(args.path)

    if args.cmd == "register":
        entry = ModelEntry(
            version=args.version,
            artifact_path=args.model,
            training_date=datetime.now(timezone.utc).isoformat(),
            training_data_sha256=hash_training_data(args.training_data),
            hyperparameters=json.loads(args.hyperparameters),
            validation_metrics=_parse_metrics(args.metrics),
            feature_list=_parse_features(args.features),
            notes=args.notes,
        )
        reg.register(entry)
        print(f"registered {entry.version}")
    elif args.cmd == "list":
        _print_table(reg.list())
    elif args.cmd == "get":
        e = reg.get(args.version)
        if e is None:
            print(f"no entry for {args.version}", file=sys.stderr)
            return 1
        print(json.dumps(asdict(e), indent=2))
    elif args.cmd == "promote":
        e = reg.promote(args.version)
        print(f"promoted {e.version} to champion")
    elif args.cmd == "retire":
        e = reg.retire(args.version)
        print(f"retired {e.version}")
    elif args.cmd == "champion":
        c = reg.champion()
        if c is None:
            print("(no champion set)")
            return 1
        print(json.dumps(asdict(c), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
