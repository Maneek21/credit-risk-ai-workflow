"""Tests for workflow.config (Phase 9)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import pytest

from workflow.config import (
    DEFAULT_PROTECTED_KEYWORDS,
    DEFAULT_SLA_HOURS,
    AuditConfig,
    EscalationConfig,
    LLMConfig,
    ModelConfig,
    MonitoringConfig,
    RateLimitConfig,
    SafetyConfig,
    WorkflowConfig,
)

try:  # match the runtime fallback in workflow.config
    import yaml  # type: ignore[import-not-found]

    _HAS_YAML = True
except ModuleNotFoundError:
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip any leaking ``WORKFLOW_*`` env vars before each test."""
    for key in list(os.environ):
        if key.startswith("WORKFLOW_"):
            monkeypatch.delenv(key, raising=False)
    yield


def _write_config_file(tmp_path: Path, payload: dict) -> Path:
    """Write a YAML config when PyYAML is available, JSON otherwise."""
    if _HAS_YAML:
        path = tmp_path / "workflow_config.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    else:
        path = tmp_path / "workflow_config.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults(clean_env: None) -> None:
    cfg = WorkflowConfig()

    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.llm, LLMConfig)
    assert isinstance(cfg.safety, SafetyConfig)
    assert isinstance(cfg.audit, AuditConfig)
    assert isinstance(cfg.ratelimit, RateLimitConfig)
    assert isinstance(cfg.escalation, EscalationConfig)
    assert isinstance(cfg.monitoring, MonitoringConfig)

    assert cfg.model.path == ""
    assert cfg.model.version == "1.0.0"
    assert cfg.model.threshold == 0.5

    assert cfg.llm.provider == "openai"
    assert cfg.llm.model == "gpt-4o"
    assert cfg.llm.temperature == 0.3
    assert cfg.llm.max_tokens == 800
    assert cfg.llm.endpoint_url is None

    assert cfg.safety.uncertainty_threshold == 0.35
    assert cfg.safety.di_threshold == 0.8
    assert cfg.safety.protected_keywords == DEFAULT_PROTECTED_KEYWORDS
    # ensure the default factory returns an independent list
    assert cfg.safety.protected_keywords is not DEFAULT_PROTECTED_KEYWORDS

    assert cfg.audit.backend == "jsonl"
    assert cfg.audit.path == "./audit_logs"
    assert cfg.audit.retention_days == 365

    assert cfg.ratelimit.rpm == 60
    assert cfg.ratelimit.tpm == 90_000
    assert cfg.ratelimit.max_spend_usd == 30.0
    assert cfg.ratelimit.circuit_breaker_threshold == 5

    assert cfg.escalation.triggers == ["BORDERLINE", "PROTECTED_ATTR_DETECTED"]
    assert cfg.escalation.webhook_url is None
    assert cfg.escalation.sla_hours == DEFAULT_SLA_HOURS

    assert cfg.monitoring.psi_threshold == 0.20
    assert cfg.monitoring.auc_drop_threshold == 0.03
    assert cfg.monitoring.window_days == 30


# ---------------------------------------------------------------------------
# YAML / JSON load
# ---------------------------------------------------------------------------


def test_yaml_or_json_round_trip(clean_env: None, tmp_path: Path) -> None:
    payload = {
        "model": {
            "path": "/tmp/model.joblib",
            "version": "2.3.4",
            "threshold": 0.55,
        },
        "llm": {
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "temperature": 0.1,
            "max_tokens": 1200,
            "endpoint_url": "https://gateway.example.com",
        },
        "ratelimit": {
            "rpm": 120,
            "tpm": 200_000,
            "max_spend_usd": 50.0,
            "circuit_breaker_threshold": 10,
        },
        "escalation": {
            "triggers": ["BORDERLINE"],
            "webhook_url": "https://hooks.example.com/x",
            "sla_hours": {"HIGH": 2, "MEDIUM": 12, "LOW": 48},
        },
    }
    path = _write_config_file(tmp_path, payload)

    cfg = WorkflowConfig.from_yaml(str(path))

    assert cfg.model.path == "/tmp/model.joblib"
    assert cfg.model.version == "2.3.4"
    assert cfg.model.threshold == 0.55

    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.model == "claude-opus-4-7"
    assert cfg.llm.temperature == 0.1
    assert cfg.llm.max_tokens == 1200
    assert cfg.llm.endpoint_url == "https://gateway.example.com"

    assert cfg.ratelimit.rpm == 120
    assert cfg.ratelimit.tpm == 200_000
    assert cfg.ratelimit.max_spend_usd == 50.0
    assert cfg.ratelimit.circuit_breaker_threshold == 10

    assert cfg.escalation.triggers == ["BORDERLINE"]
    assert cfg.escalation.webhook_url == "https://hooks.example.com/x"
    assert cfg.escalation.sla_hours == {"HIGH": 2, "MEDIUM": 12, "LOW": 48}

    # Untouched sections retain defaults.
    assert cfg.safety.di_threshold == 0.8


def test_yaml_missing_keys_use_defaults(clean_env: None, tmp_path: Path) -> None:
    path = _write_config_file(tmp_path, {"llm": {"model": "gpt-4o-mini"}})

    cfg = WorkflowConfig.from_yaml(str(path))

    assert cfg.llm.model == "gpt-4o-mini"
    assert cfg.llm.provider == "openai"  # default preserved
    assert cfg.audit.backend == "jsonl"  # default preserved


# ---------------------------------------------------------------------------
# Env var overrides
# ---------------------------------------------------------------------------


def test_env_var_override_simple(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKFLOW_LLM_MODEL", "gpt-4o-mini")

    cfg = WorkflowConfig.from_env()

    assert cfg.llm.model == "gpt-4o-mini"


def test_env_var_override_typed(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKFLOW_RATELIMIT_RPM", "240")
    monkeypatch.setenv("WORKFLOW_RATELIMIT_MAX_SPEND_USD", "12.5")
    monkeypatch.setenv("WORKFLOW_AUDIT_RETENTION_DAYS", "30")
    monkeypatch.setenv("WORKFLOW_LLM_ENDPOINT_URL", "https://custom.example.com")
    monkeypatch.setenv(
        "WORKFLOW_ESCALATION_TRIGGERS",
        "BORDERLINE,PROTECTED_ATTR_DETECTED,LOW_CONFIDENCE",
    )
    monkeypatch.setenv(
        "WORKFLOW_ESCALATION_SLA_HOURS",
        '{"HIGH": 1, "MEDIUM": 6, "LOW": 24}',
    )

    cfg = WorkflowConfig.from_env()

    assert cfg.ratelimit.rpm == 240
    assert isinstance(cfg.ratelimit.rpm, int)
    assert cfg.ratelimit.max_spend_usd == 12.5
    assert cfg.audit.retention_days == 30
    assert cfg.llm.endpoint_url == "https://custom.example.com"
    assert cfg.escalation.triggers == [
        "BORDERLINE",
        "PROTECTED_ATTR_DETECTED",
        "LOW_CONFIDENCE",
    ]
    assert cfg.escalation.sla_hours == {"HIGH": 1, "MEDIUM": 6, "LOW": 24}


def test_env_var_optional_none(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKFLOW_LLM_ENDPOINT_URL", "none")

    cfg = WorkflowConfig.from_env()

    assert cfg.llm.endpoint_url is None


# ---------------------------------------------------------------------------
# .env file parsing
# ---------------------------------------------------------------------------


def test_dotenv_file_parsing(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment line",
                "WORKFLOW_LLM_MODEL=gpt-4o-mini",
                'WORKFLOW_AUDIT_PATH="./logs/audit"',
                "export WORKFLOW_RATELIMIT_RPM=42",
                "WORKFLOW_SAFETY_DI_THRESHOLD=0.85",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cfg = WorkflowConfig.load(env_path=str(env_file))

    assert cfg.llm.model == "gpt-4o-mini"
    assert cfg.audit.path == "./logs/audit"
    assert cfg.ratelimit.rpm == 42
    assert cfg.safety.di_threshold == 0.85


# ---------------------------------------------------------------------------
# Layered priority: defaults < yaml < .env < os.environ
# ---------------------------------------------------------------------------


def test_layered_priority(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yaml_payload = {
        "llm": {"model": "yaml-model", "temperature": 0.7},
        "ratelimit": {"rpm": 100, "max_spend_usd": 10.0},
        "audit": {"path": "./from-yaml"},
    }
    yaml_path = _write_config_file(tmp_path, yaml_payload)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "WORKFLOW_LLM_MODEL=dotenv-model",
                "WORKFLOW_RATELIMIT_RPM=200",
                "WORKFLOW_AUDIT_PATH=./from-dotenv",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("WORKFLOW_LLM_MODEL", "process-env-model")
    # ratelimit.rpm intentionally NOT set in process env => .env value should win.
    # audit.path NOT set in process env or .env override comes from .env.

    cfg = WorkflowConfig.load(yaml_path=str(yaml_path), env_path=str(env_file))

    # process env wins over both .env and yaml
    assert cfg.llm.model == "process-env-model"
    # .env wins over yaml when process env is silent
    assert cfg.ratelimit.rpm == 200
    # .env still wins over yaml for other fields
    assert cfg.audit.path == "./from-dotenv"
    # yaml-only fields survive (no override above)
    assert cfg.llm.temperature == 0.7
    assert cfg.ratelimit.max_spend_usd == 10.0
    # untouched fields keep dataclass defaults
    assert cfg.llm.provider == "openai"
    assert cfg.monitoring.psi_threshold == 0.20


def test_layered_yaml_beats_defaults(
    clean_env: None, tmp_path: Path
) -> None:
    yaml_path = _write_config_file(tmp_path, {"llm": {"model": "yaml-only"}})

    cfg = WorkflowConfig.load(yaml_path=str(yaml_path))

    assert cfg.llm.model == "yaml-only"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_passes_on_defaults(clean_env: None) -> None:
    WorkflowConfig().validate()  # should not raise


def test_validate_rejects_bad_provider(clean_env: None) -> None:
    cfg = WorkflowConfig()
    cfg.llm.provider = "google"

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        cfg.validate()


def test_validate_rejects_negative_thresholds(clean_env: None) -> None:
    cfg = WorkflowConfig()
    cfg.safety.uncertainty_threshold = -0.1

    with pytest.raises(ValueError, match="uncertainty_threshold"):
        cfg.validate()

    cfg = WorkflowConfig()
    cfg.monitoring.auc_drop_threshold = -1.0
    with pytest.raises(ValueError, match="auc_drop_threshold"):
        cfg.validate()


def test_validate_rejects_zero_spend(clean_env: None) -> None:
    cfg = WorkflowConfig()
    cfg.ratelimit.max_spend_usd = 0.0

    with pytest.raises(ValueError, match="max_spend_usd"):
        cfg.validate()


def test_validate_rejects_di_out_of_range(clean_env: None) -> None:
    cfg = WorkflowConfig()
    cfg.safety.di_threshold = 1.5

    with pytest.raises(ValueError, match="di_threshold"):
        cfg.validate()
