"""Base interface for jurisdiction-specific regulatory modules.

Each supported market subclasses :class:`JurisdictionBase` and supplies
the values that vary by region — protected attributes, mandatory
disclosures, fairness thresholds, data residency, and the path to a
disclosure-text template under ``workflow/jurisdictions/templates/``.

The pipeline reads only this interface; nothing in :mod:`workflow.pipeline`
hardcodes US-specific text once a jurisdiction is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class ExplainabilityLevel(str, Enum):
    """How granular a per-decision explanation must be.

    BASIC        — provide reasons (most markets)
    DETAILED     — specific factor-level explanation (US, UK, EU member states)
    ALGORITHMIC  — full model-logic transparency (EU AI Act, high-risk)
    """

    BASIC = "basic"
    DETAILED = "detailed"
    ALGORITHMIC = "algorithmic"


_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def load_template(name: str) -> str:
    """Read a disclosure template file from ``workflow/jurisdictions/templates``.

    Templates are plain text and are intended to be reviewed/edited by
    qualified counsel without touching Python.
    """
    path = _TEMPLATE_DIR / name
    return path.read_text(encoding="utf-8")


@dataclass
class JurisdictionBase:
    """Region-specific regulatory module.

    Subclasses populate every field via ``super().__init__(...)``; the
    pipeline calls :meth:`get_disclosure_block`, :meth:`get_protected_keywords`,
    and :meth:`validate_notice` at runtime.
    """

    code: str                              # ISO 3166-1 alpha-2 (or "EU")
    name: str                              # Human-readable
    primary_regulator: str                 # CFPB / FCA / EBA / RBI / ...
    applicable_laws: List[str]             # FCRA / ECOA / GDPR / CCD / ...

    protected_attributes: List[str]        # Keyword denylist for LLM scan
    protected_categories: Dict[str, str]   # Category -> legal basis

    adverse_action_template: str           # Filename under templates/
    mandatory_disclosures: List[str]       # Substrings that MUST appear in notice
    disclosure_deadline_days: int          # Statutory deadline (0 = unspecified)
    max_denial_reasons: int                # How many reasons to cite

    explainability_level: ExplainabilityLevel
    requires_human_review: bool
    requires_model_registration: bool

    data_residency_required: bool
    cross_border_restrictions: List[str]   # Allowed regions if not in-country
    pii_law: str                           # Primary data-protection statute

    fairness_threshold: float              # DI floor (0.80 = US four-fifths)
    fairness_metric: str                   # "four_fifths" / "statistical_parity" / ...
    fairness_enforcement: str              # "hard" (legal mandate) / "soft" (guidance)

    _disclosure_cache: Optional[str] = field(
        default=None, init=False, repr=False, compare=False,
    )

    def get_disclosure_block(self) -> str:
        """Return the mandatory legal disclosure text. Lazy-loaded once."""
        if self._disclosure_cache is None:
            self._disclosure_cache = load_template(self.adverse_action_template)
        return self._disclosure_cache

    def get_protected_keywords(self) -> List[str]:
        """Return the protected-attribute keyword denylist."""
        return list(self.protected_attributes)

    def validate_notice(self, notice_text: str) -> List[str]:
        """Check every mandatory_disclosures string is present in ``notice_text``.

        Returns a list of human-readable issues — empty list = compliant.
        Subclasses can override to add jurisdiction-specific checks.
        """
        issues: List[str] = []
        for needle in self.mandatory_disclosures:
            if needle not in notice_text:
                issues.append(f"missing required disclosure: {needle!r}")
        return issues

    def data_residency_warning(self, llm_endpoint_region: Optional[str]) -> Optional[str]:
        """Return a human-readable warning if the LLM endpoint violates residency.

        Returns None when residency is not required, the endpoint is in-country,
        or the endpoint is on the cross_border allowlist. The warning is
        advisory — infrastructure decisions belong to the deploying institution.
        """
        if not self.data_residency_required:
            return None
        if llm_endpoint_region is None:
            return (
                f"{self.code}: jurisdiction requires data residency under "
                f"{self.pii_law}; LLM endpoint region unknown — verify before "
                f"production use"
            )
        if (
            llm_endpoint_region != self.code
            and llm_endpoint_region not in (self.cross_border_restrictions or [])
        ):
            return (
                f"{self.code}: jurisdiction prefers in-country processing "
                f"under {self.pii_law}; LLM endpoint is in "
                f"{llm_endpoint_region}"
            )
        return None
