"""Layered configuration for the credit-risk production workflow.

This module defines the :class:`WorkflowConfig` aggregate and its sub-dataclass
sections (model, llm, safety, audit, ratelimit, escalation, monitoring).

Configuration sources, in increasing order of priority:

    1. Hardcoded dataclass defaults.
    2. ``workflow_config.yaml`` (or JSON, when PyYAML is unavailable).
    3. ``.env`` file (parsed with stdlib).
    4. Operating-system environment variables (``WORKFLOW_<SECTION>_<FIELD>``).

Example:
    >>> cfg = WorkflowConfig.load(yaml_path="workflow_config.yaml")
    >>> cfg.validate()
    >>> cfg.llm.model
    'gpt-4o'
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

try:  # PyYAML is optional; fall back to JSON if missing.
    import yaml as _yaml  # type: ignore[import-not-found]

    _HAS_YAML = True
except ModuleNotFoundError:  # pragma: no cover - covered by env without yaml
    _yaml = None  # type: ignore[assignment]
    _HAS_YAML = False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Default protected keyword set, mirrored from ``workflow.pipeline``.
DEFAULT_PROTECTED_KEYWORDS: List[str] = [
    "race", "racial", "ethnicity", "ethnic", "skin color",
    "gender", "sex", "male", "female", "woman", "man",
    "age", "old", "young", "elderly", "senior",
    "religion", "religious", "muslim", "christian", "jewish", "hindu",
    "national origin", "immigrant", "foreign",
    "disability", "disabled", "handicap",
    "marital status", "married", "single", "divorced",
    "pregnant", "pregnancy",
]

#: Default escalation triggers.
DEFAULT_ESCALATION_TRIGGERS: List[str] = [
    "BORDERLINE",
    "PROTECTED_ATTR_DETECTED",
]

#: Default SLA hours by severity.
DEFAULT_SLA_HOURS: Dict[str, int] = {"HIGH": 4, "MEDIUM": 24, "LOW": 72}


# ---------------------------------------------------------------------------
# Sub-section dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Settings for the underlying scoring model.

    Attributes:
        path: Filesystem path to a serialized scikit-learn / XGBoost model.
        version: Semantic version string surfaced into audit records.
        threshold: Decision threshold applied to ``predict_proba``.
    """

    path: str = ""
    version: str = "1.0.0"
    threshold: float = 0.5


@dataclass
class LLMConfig:
    """Settings for the LLM memo / explanation provider.

    Attributes:
        provider: ``"openai"`` or ``"anthropic"``.
        model: Provider-specific model identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to request per generation.
        endpoint_url: Optional custom inference endpoint (e.g. Azure, gateway).
    """

    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 800
    endpoint_url: Optional[str] = None


@dataclass
class SafetyConfig:
    """Safety guardrails applied around model + LLM outputs.

    Attributes:
        uncertainty_threshold: Probability margin around 0.5 that triggers
            BORDERLINE escalation.
        di_threshold: Disparate impact ratio floor (regulatory 0.8 by default).
        protected_keywords: Tokens redacted from LLM output and used to detect
            protected-attribute leakage.
    """

    uncertainty_threshold: float = 0.35
    di_threshold: float = 0.8
    protected_keywords: List[str] = field(
        default_factory=lambda: list(DEFAULT_PROTECTED_KEYWORDS)
    )


@dataclass
class AuditConfig:
    """Audit log destination configuration.

    Attributes:
        backend: One of ``"jsonl"``, ``"stdout"``, or ``"both"``.
        path: Directory for JSONL logs.
        retention_days: Days to retain rotated audit logs.
    """

    backend: str = "jsonl"
    path: str = "./audit_logs"
    retention_days: int = 365


@dataclass
class RateLimitConfig:
    """Provider rate limit and spend governor.

    Attributes:
        rpm: Requests per minute ceiling.
        tpm: Tokens per minute ceiling.
        max_spend_usd: Hard spend cap (USD) before requests are blocked.
        circuit_breaker_threshold: Consecutive failures before tripping the
            circuit breaker.
    """

    rpm: int = 60
    tpm: int = 90_000
    max_spend_usd: float = 30.0
    circuit_breaker_threshold: int = 5


@dataclass
class EscalationConfig:
    """Escalation routing for human review.

    Attributes:
        triggers: Reason codes that route to the human queue.
        webhook_url: Optional outbound webhook for notifications.
        sla_hours: SLA in hours by severity (``HIGH``/``MEDIUM``/``LOW``).
    """

    triggers: List[str] = field(
        default_factory=lambda: list(DEFAULT_ESCALATION_TRIGGERS)
    )
    webhook_url: Optional[str] = None
    sla_hours: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_SLA_HOURS)
    )


@dataclass
class MonitoringConfig:
    """Drift / performance monitoring thresholds.

    Attributes:
        psi_threshold: Population Stability Index alert threshold.
        auc_drop_threshold: Absolute AUC drop that triggers an alert.
        window_days: Rolling window length in days.
    """

    psi_threshold: float = 0.20
    auc_drop_threshold: float = 0.03
    window_days: int = 30


# ---------------------------------------------------------------------------
# Top-level aggregate
# ---------------------------------------------------------------------------


@dataclass
class WorkflowConfig:
    """Aggregate configuration container for the production workflow.

    Attributes:
        model: Scoring model settings.
        llm: LLM provider settings.
        safety: Guardrail thresholds and protected keyword set.
        audit: Audit log backend / retention.
        ratelimit: Rate limits and spend cap.
        escalation: Human-review routing and SLAs.
        monitoring: Drift and performance monitoring thresholds.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    ratelimit: RateLimitConfig = field(default_factory=RateLimitConfig)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkflowConfig":
        """Build a config from a nested mapping, falling back to defaults.

        Args:
            data: Mapping keyed by section name (``"llm"``, ``"safety"``...).

        Returns:
            A populated :class:`WorkflowConfig`.
        """
        cfg = cls()
        _apply_mapping(cfg, data or {})
        return cfg

    @classmethod
    def from_env(cls) -> "WorkflowConfig":
        """Build a config from defaults overlaid with ``os.environ``.

        Returns:
            A populated :class:`WorkflowConfig`.
        """
        cfg = cls()
        _apply_env_dict(cfg, os.environ)
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "WorkflowConfig":
        """Load a config from YAML (or JSON if PyYAML is unavailable).

        When PyYAML is not installed, the file is parsed with
        :func:`json.loads`. Missing keys fall back to dataclass defaults.

        Args:
            path: Path to ``workflow_config.yaml`` (or ``.json``).

        Returns:
            A populated :class:`WorkflowConfig`.
        """
        text = Path(path).read_text(encoding="utf-8")
        data = _parse_structured(text, path)
        return cls.from_dict(data)

    @classmethod
    def load(
        cls,
        yaml_path: Optional[str] = None,
        env_path: Optional[str] = None,
    ) -> "WorkflowConfig":
        """Layered load: defaults < yaml/json < .env < ``os.environ``.

        Args:
            yaml_path: Optional path to a YAML/JSON config file.
            env_path: Optional path to a ``.env`` file.

        Returns:
            A fully resolved :class:`WorkflowConfig`.
        """
        cfg = cls()
        # 1. yaml/json overrides defaults.
        if yaml_path and Path(yaml_path).exists():
            text = Path(yaml_path).read_text(encoding="utf-8")
            data = _parse_structured(text, yaml_path)
            _apply_mapping(cfg, data or {})
        # 2. .env overrides yaml.
        if env_path and Path(env_path).exists():
            env_dict = _parse_dotenv(Path(env_path).read_text(encoding="utf-8"))
            _apply_env_dict(cfg, env_dict)
        # 3. process env overrides .env.
        _apply_env_dict(cfg, os.environ)
        return cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate the configuration, raising on logical errors.

        Raises:
            ValueError: On unsupported provider, negative thresholds,
                non-positive spend cap, or out-of-range disparate impact.
        """
        if self.llm.provider not in {"openai", "anthropic"}:
            raise ValueError(
                f"Unsupported LLM provider: {self.llm.provider!r}. "
                "Expected one of: openai, anthropic."
            )

        negative_checks: Dict[str, float] = {
            "model.threshold": self.model.threshold,
            "llm.temperature": self.llm.temperature,
            "llm.max_tokens": float(self.llm.max_tokens),
            "safety.uncertainty_threshold": self.safety.uncertainty_threshold,
            "safety.di_threshold": self.safety.di_threshold,
            "audit.retention_days": float(self.audit.retention_days),
            "ratelimit.rpm": float(self.ratelimit.rpm),
            "ratelimit.tpm": float(self.ratelimit.tpm),
            "ratelimit.circuit_breaker_threshold": float(
                self.ratelimit.circuit_breaker_threshold
            ),
            "monitoring.psi_threshold": self.monitoring.psi_threshold,
            "monitoring.auc_drop_threshold": self.monitoring.auc_drop_threshold,
            "monitoring.window_days": float(self.monitoring.window_days),
        }
        for name, value in negative_checks.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}.")

        if self.ratelimit.max_spend_usd <= 0:
            raise ValueError(
                f"ratelimit.max_spend_usd must be > 0, got "
                f"{self.ratelimit.max_spend_usd!r}."
            )

        if not 0 <= self.safety.di_threshold <= 1:
            raise ValueError(
                f"safety.di_threshold must be in [0, 1], got "
                f"{self.safety.di_threshold!r}."
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_SECTION_FIELDS = {
    "model": ModelConfig,
    "llm": LLMConfig,
    "safety": SafetyConfig,
    "audit": AuditConfig,
    "ratelimit": RateLimitConfig,
    "escalation": EscalationConfig,
    "monitoring": MonitoringConfig,
}


def _parse_structured(text: str, path: str) -> Dict[str, Any]:
    """Parse YAML if available, otherwise JSON.

    Args:
        text: File contents.
        path: Source path (used only for error context).

    Returns:
        A dict (possibly empty) representing the document.
    """
    if _HAS_YAML:
        loaded = _yaml.safe_load(text)  # type: ignore[union-attr]
        return loaded or {}
    # JSON fallback. Empty file => {}.
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError as exc:  # pragma: no cover - error path
        raise ValueError(
            f"PyYAML is not installed and {path!r} is not valid JSON: {exc}"
        ) from exc
    return loaded or {}


def _parse_dotenv(text: str) -> Dict[str, str]:
    """Minimal ``.env`` parser (stdlib only).

    Supports ``KEY=VALUE`` lines, optional ``export`` prefix, ``#`` comments,
    and surrounding single or double quotes on the value.

    Args:
        text: Raw ``.env`` file contents.

    Returns:
        Mapping from key to (string) value.
    """
    result: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip optional inline comment when value is unquoted.
        if value and value[0] not in {'"', "'"} and " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            result[key] = value
    return result


def _coerce(value: str, target_type: Any, current: Any) -> Any:
    """Coerce a string into the type implied by an existing dataclass value.

    Args:
        value: Raw string from env/.env.
        target_type: Annotation target type (may be ``Optional[...]``).
        current: Current dataclass value (used to infer concrete container
            element types when annotations are generic).

    Returns:
        The coerced value, or the original string if no coercion applies.
    """
    type_str = str(target_type)

    # Optional[T] handling - "None"/"null"/"" map to None.
    is_optional = "Optional" in type_str or "None" in type_str
    if is_optional and value.strip().lower() in {"", "none", "null"}:
        return None

    # bool first because bool is also int.
    if "bool" in type_str and "List" not in type_str and "Dict" not in type_str:
        return _coerce_bool(value)

    if "int" in type_str and "List" not in type_str and "Dict" not in type_str:
        return int(float(value))  # tolerate "60.0" -> 60

    if "float" in type_str and "List" not in type_str and "Dict" not in type_str:
        return float(value)

    if "List" in type_str or isinstance(current, list):
        # Try JSON first so users can pass JSON arrays; fall back to CSV.
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return [item.strip() for item in value.split(",") if item.strip()]

    if "Dict" in type_str or isinstance(current, dict):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                # Preserve int values where the original dict used ints.
                if current and all(isinstance(v, int) for v in current.values()):
                    return {str(k): int(v) for k, v in parsed.items()}
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
        # CSV of "k:v,k:v"
        out: Dict[str, Any] = {}
        for chunk in value.split(","):
            if ":" not in chunk:
                continue
            k, _, v = chunk.partition(":")
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
        return out

    return value


def _coerce_bool(value: str) -> bool:
    """Coerce common truthy/falsy strings to bool."""
    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}
    lowered = value.strip().lower()
    if lowered in truthy:
        return True
    if lowered in falsy:
        return False
    raise ValueError(f"Cannot coerce {value!r} to bool.")


def _apply_mapping(cfg: WorkflowConfig, data: Mapping[str, Any]) -> None:
    """Overlay a nested mapping onto an existing config in place.

    Args:
        cfg: Target config (mutated).
        data: Nested ``{section: {field: value}}`` mapping.
    """
    for section_name, section_cls in _SECTION_FIELDS.items():
        section_data = data.get(section_name)
        if not isinstance(section_data, Mapping):
            continue
        section_obj = getattr(cfg, section_name)
        for f in fields(section_cls):
            if f.name in section_data:
                value = section_data[f.name]
                if isinstance(value, str) and not _is_string_field(f.type):
                    value = _coerce(value, f.type, getattr(section_obj, f.name))
                setattr(section_obj, f.name, value)


def _is_string_field(annotation: Any) -> bool:
    """Return True if the annotation refers (only) to ``str``/``Optional[str]``."""
    type_str = str(annotation)
    if "List" in type_str or "Dict" in type_str:
        return False
    if "int" in type_str or "float" in type_str or "bool" in type_str:
        return False
    return "str" in type_str


def _apply_env_dict(cfg: WorkflowConfig, env: Mapping[str, str]) -> None:
    """Overlay env-style flat keys onto a config in place.

    Recognised keys follow ``WORKFLOW_<SECTION>_<FIELD>``. ``<SECTION>`` is the
    uppercase section name (``LLM``, ``RATELIMIT``...). ``<FIELD>`` is the
    uppercase dataclass field, including underscores (e.g.
    ``WORKFLOW_RATELIMIT_MAX_SPEND_USD``).

    Args:
        cfg: Target config (mutated).
        env: Flat mapping (e.g. ``os.environ`` or parsed ``.env``).
    """
    prefix = "WORKFLOW_"
    # Build a lookup of (section_upper -> section_name, {field_upper -> Field}).
    section_lookup: Dict[str, tuple[str, Dict[str, Any]]] = {}
    for section_name, section_cls in _SECTION_FIELDS.items():
        section_lookup[section_name.upper()] = (
            section_name,
            {f.name.upper(): f for f in fields(section_cls)},
        )

    for raw_key, raw_value in env.items():
        if not raw_key.startswith(prefix):
            continue
        remainder = raw_key[len(prefix):]
        # Match section (longest match wins, since RATELIMIT and RATE could clash).
        matched_section: Optional[str] = None
        matched_field_key: Optional[str] = None
        for section_upper, (_section_name, field_map) in section_lookup.items():
            head = section_upper + "_"
            if remainder.startswith(head):
                tail = remainder[len(head):]
                if tail in field_map:
                    matched_section = section_upper
                    matched_field_key = tail
                    break
        if matched_section is None or matched_field_key is None:
            continue
        section_name, field_map = section_lookup[matched_section]
        field_def = field_map[matched_field_key]
        section_obj = getattr(cfg, section_name)
        current_value = getattr(section_obj, field_def.name)
        try:
            coerced = _coerce(str(raw_value), field_def.type, current_value)
        except (ValueError, TypeError):
            # Leave silent coercion failures alone rather than crash imports.
            continue
        setattr(section_obj, field_def.name, coerced)


# Make sure the dataclass machinery is happy with our type hints.
assert is_dataclass(WorkflowConfig)


__all__ = [
    "AuditConfig",
    "DEFAULT_ESCALATION_TRIGGERS",
    "DEFAULT_PROTECTED_KEYWORDS",
    "DEFAULT_SLA_HOURS",
    "EscalationConfig",
    "LLMConfig",
    "ModelConfig",
    "MonitoringConfig",
    "RateLimitConfig",
    "SafetyConfig",
    "WorkflowConfig",
]
