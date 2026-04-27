"""Dataset adapter abstraction for the training SDK.

A :class:`DatasetAdapter` knows how to download a dataset, load it from
disk, and expose a :class:`DatasetMetadata` schema describing which
columns are features, which are categorical, and which are protected
(used only for fairness evaluation, never as model input).

Concrete adapters live in :mod:`data.adapters` and are registered by a
short name ("uci", "hmda", "bondora"). Custom adapters from a
deploying institution are referenced by dotted Python path in YAML —
see ``examples/bring_your_own_data/`` for the pattern.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class DatasetMetadata:
    """Schema description for a dataset.

    Invariants enforced by :meth:`DatasetAdapter.validate`:
        * ``feature_columns`` and ``protected_columns`` MUST be disjoint.
          Protected attributes are for fairness evaluation only — they
          must not be model input features (ECOA / GDPR Art 22 / etc.).
        * Every ``categorical_column`` must appear in ``feature_columns``.
    """

    name: str
    region: str                            # ISO code or "TW" for Taiwan
    target_column: str                     # Binary target column name
    feature_columns: List[str]             # Model input features
    categorical_columns: List[str] = field(default_factory=list)
    protected_columns: Dict[str, str] = field(default_factory=dict)
    positive_label: int = 1                # Which value of target = "default"
    description: str = ""

    def validate(self) -> None:
        """Enforce structural invariants. Raises ``ValueError`` on conflict."""
        feat_set = set(self.feature_columns)
        prot_set = set(self.protected_columns)
        overlap = feat_set & prot_set
        if overlap:
            raise ValueError(
                f"Protected columns must not be model features (overlap: {sorted(overlap)})"
            )
        cat_not_in_features = set(self.categorical_columns) - feat_set
        if cat_not_in_features:
            raise ValueError(
                f"categorical_columns not in feature_columns: {sorted(cat_not_in_features)}"
            )
        if not self.feature_columns:
            raise ValueError("feature_columns must be non-empty")
        if not self.target_column:
            raise ValueError("target_column must be set")


class DatasetAdapter(ABC):
    """Abstract base class for dataset loading + preprocessing.

    Subclasses implement :meth:`metadata`, :meth:`load`, and
    :meth:`download`. The base class adds a :meth:`load_with_protected`
    helper that splits the loaded DataFrame into (features, target,
    protected) — the canonical shape the training pipeline consumes.
    """

    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Return the schema description for this dataset."""

    @abstractmethod
    def load(self, path: str) -> pd.DataFrame:
        """Load and preprocess the raw data from ``path``.

        Must return a DataFrame containing every column listed in the
        metadata's ``feature_columns``, ``protected_columns``, and
        ``target_column``. Rows with missing values in critical fields
        should already be dropped.
        """

    @abstractmethod
    def download(self, dest_dir: str) -> str:
        """Download the dataset to ``dest_dir``; return the local file path."""

    def load_with_protected(self, path: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Load + split into (features, target, protected_df).

        The training pipeline calls this. Protected columns are returned
        as a separate DataFrame so they can be used for fairness
        evaluation but never reach the model.
        """
        meta = self.metadata()
        meta.validate()
        df = self.load(path)
        missing = (set(meta.feature_columns)
                   | set(meta.protected_columns)
                   | {meta.target_column}) - set(df.columns)
        if missing:
            raise ValueError(
                f"{type(self).__name__}.load() returned DataFrame missing columns: "
                f"{sorted(missing)}"
            )
        X = df[meta.feature_columns].copy()
        y = df[meta.target_column].astype(int).copy()
        protected = (
            df[list(meta.protected_columns.keys())].copy()
            if meta.protected_columns else pd.DataFrame(index=df.index)
        )
        return X, y, protected


# ---------------------------------------------------------------------------
# Adapter resolution — short name OR dotted path
# ---------------------------------------------------------------------------

#: Registry of built-in adapters. Populated lazily by :func:`resolve_adapter`
#: to avoid eagerly importing every adapter module at package load time.
_BUILTIN_NAMES: Dict[str, str] = {
    "uci":     "data.adapters.uci:UCIAdapter",
    "hmda":    "data.adapters.hmda:HMDAAdapter",
    "bondora": "data.adapters.bondora:BondoraAdapter",
}


def resolve_adapter(spec: str) -> DatasetAdapter:
    """Resolve a string specifier to an instantiated :class:`DatasetAdapter`.

    Accepted forms:
      * Short name ("uci", "hmda", "bondora") — looked up in
        :data:`_BUILTIN_NAMES`.
      * Dotted path with class ("pkg.mod:ClassName" or
        "pkg.mod.ClassName") — imported and instantiated.

    Raises:
        ValueError: if the spec cannot be resolved or the resolved object
            is not a :class:`DatasetAdapter`.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("adapter spec must be non-empty")
    target = _BUILTIN_NAMES.get(spec.lower(), spec)
    module_path, _, class_name = target.partition(":")
    if not class_name:
        # "pkg.mod.ClassName" form
        module_path, _, class_name = target.rpartition(".")
    if not module_path or not class_name:
        raise ValueError(
            f"Cannot resolve adapter {spec!r}: expected 'name', 'pkg.mod:Class', or 'pkg.mod.Class'"
        )
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(f"Cannot import adapter module {module_path!r}: {exc}") from exc
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ValueError(f"Module {module_path!r} has no class {class_name!r}")
    instance = cls()
    if not isinstance(instance, DatasetAdapter):
        raise ValueError(
            f"{module_path}:{class_name} is not a DatasetAdapter subclass"
        )
    return instance


def list_builtin_adapters() -> List[str]:
    """Return the list of short names accepted by :func:`resolve_adapter`."""
    return sorted(_BUILTIN_NAMES.keys())
