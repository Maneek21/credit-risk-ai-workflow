"""Singapore jurisdiction module.

Regulator: MAS (Monetary Authority of Singapore).
Key laws:  MAS AI Risk Management Guidelines (November 2025),
           MAS FEAT Principles (Fairness, Ethics, Accountability,
           Transparency — 2018),
           Personal Data Protection Act 2012 ("PDPA"),
           Banking Act (Cap 19),
           MAS Technology Risk Management ("TRM") Guidelines.

Distinguishing feature: Singapore's framework is **principles-based**.
The Constitution (Article 12) protects against discrimination on the
narrow grounds of religion, race, descent, or place of birth, plus
language as a community right; the keyword set is therefore tighter
than the West's "anti-discrimination law" lists but adds ``descent``,
``place of birth``, ``mother tongue``, and ``dialect``.
"""
from __future__ import annotations

from .base import ExplainabilityLevel, JurisdictionBase
from .us import US_PROTECTED_KEYWORDS

#: Singapore protected-attribute keyword denylist.
SINGAPORE_PROTECTED_KEYWORDS = sorted(set(US_PROTECTED_KEYWORDS) | {
    "descent",
    "place of birth", "birthplace",
    "language", "mother tongue", "dialect",
})


class Singapore(JurisdictionBase):
    """Singapore — MAS FEAT, MAS AI Guidelines (2025), PDPA."""

    def __init__(self) -> None:
        super().__init__(
            code="SG",
            name="Singapore",
            primary_regulator="MAS",
            applicable_laws=[
                "MAS AI Risk Management Guidelines (November 2025)",
                "MAS FEAT Principles (2018)",
                "Personal Data Protection Act 2012 (PDPA)",
                "Banking Act (Cap 19)",
                "MAS Technology Risk Management Guidelines",
            ],
            protected_attributes=list(SINGAPORE_PROTECTED_KEYWORDS),
            protected_categories={
                "religion": "Constitution Art 12",
                "race": "Constitution Art 12",
                "descent": "Constitution Art 12",
                "place of birth": "Constitution Art 12",
                "language": "Constitution Art 152 (community rights)",
            },
            adverse_action_template="singapore_rejection_notice.txt",
            mandatory_disclosures=[
                "MAS",
                "FEAT",
                "PDPA",
                "Personal Data Protection Commission",
                "Financial Industry Disputes Resolution Centre",
            ],
            disclosure_deadline_days=0,
            max_denial_reasons=4,
            explainability_level=ExplainabilityLevel.DETAILED,
            requires_human_review=False,
            requires_model_registration=False,
            data_residency_required=False,
            cross_border_restrictions=[],
            pii_law="PDPA 2012",
            fairness_threshold=0.80,
            fairness_metric="feat_fairness_assessment",
            fairness_enforcement="soft",
        )
