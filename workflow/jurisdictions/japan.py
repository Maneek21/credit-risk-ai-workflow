"""Japan jurisdiction module.

Regulators: FSA (Financial Services Agency), BOJ (Bank of Japan).
Key laws:   FSA AI Discussion Paper v1.0 (March 2025),
            Banking Act (Act No. 59 of 1981),
            Instalment Sales Act (Act No. 159 of 1961, amended 2021
            for AI-based credit-limit determinations),
            Act on the Protection of Personal Information ("APPI",
            amended 2022 for cross-border transfers).

Distinguishing feature: APPI Article 2(3) defines a unique
**"special care-required personal information"** category — including
``creed``, ``social status``, ``family origin``, ``medical history``,
and ``criminal record`` — which differs in framing and scope from
GDPR special categories or the US protected-attribute list.
``Burakumin`` (descendants of historical outcaste communities) is
explicitly noted because employment- and credit-related discrimination
on this basis remains a documented concern in Japan.
"""
from __future__ import annotations

from .base import ExplainabilityLevel, JurisdictionBase
from .us import US_PROTECTED_KEYWORDS

#: Japan protected-attribute keyword denylist.
JAPAN_PROTECTED_KEYWORDS = sorted(set(US_PROTECTED_KEYWORDS) | {
    "creed",
    "social status",
    "family origin",
    "medical history",
    "criminal record", "criminal history",
    "burakumin",
})


class Japan(JurisdictionBase):
    """Japan — Banking Act / Instalment Sales Act / APPI / FSA AI Paper."""

    def __init__(self) -> None:
        super().__init__(
            code="JP",
            name="Japan",
            primary_regulator="FSA",
            applicable_laws=[
                "FSA AI Discussion Paper v1.0 (March 2025)",
                "Banking Act (Act No. 59 of 1981)",
                "Instalment Sales Act (Act No. 159 of 1961, amended 2021)",
                "Act on the Protection of Personal Information (APPI, amended 2022)",
            ],
            protected_attributes=list(JAPAN_PROTECTED_KEYWORDS),
            protected_categories={
                "race": "APPI Art 2(3) — special care-required",
                "creed": "APPI Art 2(3) — special care-required",
                "social status": "APPI Art 2(3) — special care-required",
                "family origin": "APPI Art 2(3) — special care-required",
                "physical or mental disability": "APPI Art 2(3) — special care-required",
                "medical history": "APPI Art 2(3) — special care-required",
                "criminal record": "APPI Art 2(3) — special care-required",
                "sex": "Equal Employment Opportunity Act",
            },
            adverse_action_template="japan_rejection_notice.txt",
            mandatory_disclosures=[
                "Banking Act",
                "Instalment Sales Act",
                "APPI",
                "Personal Information Protection Commission",
                "FSA",
            ],
            disclosure_deadline_days=0,
            max_denial_reasons=4,
            explainability_level=ExplainabilityLevel.DETAILED,
            requires_human_review=False,
            requires_model_registration=False,
            data_residency_required=False,
            cross_border_restrictions=[],
            pii_law="APPI",
            fairness_threshold=0.80,
            fairness_metric="fsa_fairness_principles",
            fairness_enforcement="soft",
        )
