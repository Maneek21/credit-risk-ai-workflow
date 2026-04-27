"""Configurable training SDK for credit-risk-ai-workflow.

The SDK is **opt-in**. Users who already have a trained model just pass
``model_path`` to :class:`workflow.CreditWorkflow`. The SDK exists to
accelerate onboarding for teams who want a reproducible build pipeline:
pull a YAML, run one command, get a registered model + metrics +
fairness report.

Public API:
    >>> from workflow.training import TrainingPipeline, TrainingConfig
    >>> pipeline = TrainingPipeline.from_yaml("configs/uci_us.yaml")
    >>> result = pipeline.run()  # writes joblib + metrics JSON
    >>> print(result.metrics["auc"])

Custom datasets — implement :class:`DatasetAdapter` and reference your
class by dotted path in the YAML; see ``examples/bring_your_own_data/``.
"""
from .datasets import DatasetAdapter, DatasetMetadata, list_builtin_adapters, resolve_adapter
from .evaluation import (
    FairnessReport,
    compute_auc,
    compute_brier,
    compute_ece,
    compute_fairness,
    compute_ks,
    compute_shap_summary,
    core_metrics,
)
from .pipeline import TrainingConfig, TrainingPipeline, TrainingResult

__all__ = [
    "DatasetAdapter",
    "DatasetMetadata",
    "FairnessReport",
    "TrainingConfig",
    "TrainingPipeline",
    "TrainingResult",
    "compute_auc",
    "compute_brier",
    "compute_ece",
    "compute_fairness",
    "compute_ks",
    "compute_shap_summary",
    "core_metrics",
    "list_builtin_adapters",
    "resolve_adapter",
]
