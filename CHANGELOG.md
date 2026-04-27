# Changelog

All notable changes to this project follow [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] — 2026-04-27

### Production engineering for bank adoption

Closes the gap between the research-backed pipeline and what a bank's
Model Risk Management, compliance, and infrastructure teams need before
approving a pilot. Ten new components, all opt-in via constructor kwargs
on `CreditWorkflow` so the existing API is unchanged.

### Added

- **`workflow/audit.py`** — `AuditLogger` with `JSONLBackend`, `StdoutBackend`
  (SIEM-ready), and `MultiBackend` (fan-out). Immutable, append-only audit
  records covering every field SR 11-7 §IV requires for outcomes analysis.
- **`workflow/pii.py`** — `PIIScrubber` with feature-only and redaction
  modes. Strips SSN, names, addresses, phones, emails, and DOBs before any
  LLM call. Reversible token mapping for cases that need restore. GLBA §501
  alignment.
- **`workflow/registry.py`** — `ModelRegistry` (semver, training-data
  SHA-256 fingerprint, validation metrics, lifecycle status). CLI:
  `python -m workflow.registry register|list|get|promote|retire|champion`.
- **`workflow/escalation.py`** — `EscalationRouter` with `QueueBackend`
  (JSONL) and `WebhookBackend` (urllib stdlib, retries on 5xx). Default
  triggers: `BORDERLINE`, `PROTECTED_ATTR_DETECTED`, `LOW_CONFIDENCE`,
  `HIGH_VALUE_LOAN`. Per-priority SLA hours.
- **`workflow/ratelimit.py`** — `TokenBucket`, `CircuitBreaker`,
  `CostTracker`, and a composed `RateLimiter` with exponential backoff and
  hard spend cap (`LLMUnavailableError` raised when exceeded).
- **`workflow/monitoring.py`** — `ModelMonitor` with `compute_psi`,
  AUC-drop alerts, and approval-rate-shift detection. Default thresholds
  match the SR 11-7 monitoring conventions.
- **`workflow/batch.py`** — `BatchProcessor` for portfolios of 10K-100K
  applications. Concurrent (`ThreadPoolExecutor`), partial-failure
  tolerant, checkpoint-resumable.
- **`workflow/config.py`** — `WorkflowConfig` dataclass with layered
  loading (env > .env > yaml/json > defaults) and `validate()` schema
  check. Env var convention: `WORKFLOW_<SECTION>_<FIELD>`.
- **`docs/model_validation_report.md`** — ~8,200-word MRM validation
  document following SR 11-7 / OCC 2011-12. All numbers pulled from
  `benchmarks/results/` CSVs. Marked TEMPLATE; banks customize for their
  implementation.
- **`tests/`** directory — 104 pytest tests across 9 test files. Includes
  `test_safety_layers.py` integration tests verifying FCRA injection,
  protected-attribute filter, SHAP grounding, BORDERLINE flag behaviour,
  and adverse-action absence on APPROVE — all without API keys.

### Changed

- `CreditWorkflow.__init__` accepts new optional kwargs: `model_version`,
  `audit_logger`, `pii_scrubber`, `rate_limiter`, `escalation_router`. All
  default to `None` — existing callers see no behaviour change.
- `process_application` now returns metadata containing `decision_id`,
  `model_version`, `processing_time_ms`, `escalated`, `escalation_reason`,
  and `scrubbed_fields` in addition to the existing fields.
- `process_application` accepts an optional `loan_amount` argument used
  only for the `HIGH_VALUE_LOAN` escalation trigger.

### Fixed

- **Protected-attribute filter false positives.** The FCRA disclosure text
  contains "AGENCY", which contains the substring "age", which previously
  caused every denial to be flagged as `PROTECTED_ATTR_DETECTED`. The
  filter now uses word-bounded regex (`re.compile(rf"\b{kw}\b", IGNORECASE)`)
  so substring overlaps are no longer mis-classified. The filter is also
  applied to the LLM-generated body only, not the deterministic FCRA
  disclosure block.

### Notes

- All new components are stdlib-only where possible; no new third-party
  dependencies. `pyyaml` is supported by `WorkflowConfig.from_yaml` if
  installed, otherwise that method falls back to JSON.
- Type hints (mypy/pyright clean) and Google-style docstrings on all
  public methods.
- Append-only and fail-loudly throughout: audit and registry mutations
  raise on backend failure rather than silently dropping data.
