"""Credit Risk AI Workflow — Production-ready credit underwriting pipeline.

Architecture:
  Layer 1: Classical ML model DECIDES (XGBoost, AUC 0.774)
  Layer 2: LLM COMMUNICATES (credit memos, adverse action notices)
  Layer 3: Deterministic code ENFORCES (FCRA, bias filters, uncertainty flags)

Production hardening (opt-in via constructor kwargs):
  Phase 1 — AuditLogger        : immutable audit trail (SR 11-7 §IV)
  Phase 2 — PIIScrubber        : strip PII before LLM calls (GLBA §501)
  Phase 4 — ModelRegistry      : model version traceability
  Phase 5 — EscalationRouter   : human-in-the-loop routing
  Phase 6 — RateLimiter        : rpm/tpm/circuit-breaker/cost cap
  Phase 7 — ModelMonitor       : PSI / AUC drift detection
  Phase 8 — BatchProcessor     : portfolio-scale concurrent processing
  Phase 9 — WorkflowConfig     : layered config (env > .env > yaml > defaults)

Usage (basic):
    from workflow import CreditWorkflow

    wf = CreditWorkflow(
        model_path="path/to/xgboost_model.joblib",
        llm_provider="openai",
        llm_model="gpt-4o",
    )
    result = wf.process_application(applicant_data)

Usage (production):
    from workflow import (
        CreditWorkflow,
        AuditLogger, JSONLBackend,
        PIIScrubber,
        EscalationRouter, QueueBackend,
        RateLimiter,
    )

    wf = CreditWorkflow(
        model_path="path/to/model.joblib",
        model_version="1.2.0",
        audit_logger=AuditLogger(JSONLBackend("./audit_logs")),
        pii_scrubber=PIIScrubber(),
        rate_limiter=RateLimiter(rpm=60, hard_cap_usd=100),
        escalation_router=EscalationRouter(QueueBackend("./escalations")),
    )
"""

from .audit import AuditLogger, AuditRecord, JSONLBackend, MultiBackend, StdoutBackend
from .batch import BatchProcessor, BatchResult
from .config import (
    AuditConfig,
    EscalationConfig,
    LLMConfig,
    ModelConfig,
    MonitoringConfig,
    RateLimitConfig,
    SafetyConfig,
    WorkflowConfig,
)
from .escalation import EscalationRecord, EscalationRouter, QueueBackend, WebhookBackend
from .jurisdictions import (
    ALL_JURISDICTIONS,
    EU,
    UAE,
    UK,
    US,
    Australia,
    Brazil,
    Canada,
    ExplainabilityLevel,
    India,
    Japan,
    JurisdictionBase,
    Singapore,
)
from .monitoring import Alert, ModelMonitor, compute_psi
from .pii import MODE_FEATURE_ONLY, MODE_REDACTION, PIIScrubber
from .pipeline import CreditWorkflow, WorkflowResult
from .ratelimit import (
    CircuitBreaker,
    CostTracker,
    LLMUnavailableError,
    RateLimiter,
    TokenBucket,
)
from .registry import ModelEntry, ModelRegistry, hash_training_data
from .training import (
    DatasetAdapter,
    DatasetMetadata,
    TrainingConfig,
    TrainingPipeline,
    TrainingResult,
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "CreditWorkflow",
    "WorkflowResult",
    # Phase 1
    "AuditLogger",
    "AuditRecord",
    "JSONLBackend",
    "StdoutBackend",
    "MultiBackend",
    # Phase 2
    "PIIScrubber",
    "MODE_FEATURE_ONLY",
    "MODE_REDACTION",
    # Phase 4
    "ModelRegistry",
    "ModelEntry",
    "hash_training_data",
    # Phase 5
    "EscalationRouter",
    "EscalationRecord",
    "QueueBackend",
    "WebhookBackend",
    # Phase 6
    "RateLimiter",
    "TokenBucket",
    "CircuitBreaker",
    "CostTracker",
    "LLMUnavailableError",
    # Phase 7
    "ModelMonitor",
    "Alert",
    "compute_psi",
    # Phase 8
    "BatchProcessor",
    "BatchResult",
    # Phase 9
    "WorkflowConfig",
    "ModelConfig",
    "LLMConfig",
    "SafetyConfig",
    "AuditConfig",
    "RateLimitConfig",
    "EscalationConfig",
    "MonitoringConfig",
    # Jurisdictions (multi-market support — 10 markets)
    "JurisdictionBase",
    "ExplainabilityLevel",
    "ALL_JURISDICTIONS",
    "US",
    "UK",
    "EU",
    "India",
    "Canada",
    "Australia",
    "Singapore",
    "Japan",
    "UAE",
    "Brazil",
    # Training SDK (configurable training pipeline + dataset adapters)
    "TrainingPipeline",
    "TrainingConfig",
    "TrainingResult",
    "DatasetAdapter",
    "DatasetMetadata",
]
