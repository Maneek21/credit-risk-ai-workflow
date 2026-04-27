"""India jurisdiction module.

Regulators: RBI (Reserve Bank of India), with sectoral roles for SEBI
            (securities) and IRDAI (insurance).
Key laws:   RBI Fair Practices Code, RBI Digital Lending Directions 2025,
            Credit Information Companies (Regulation) Act 2005 ("CICRA"),
            Digital Personal Data Protection Act 2023 ("DPDPA"),
            Constitution of India Article 15.

Distinguishing feature: ``caste`` and related terms are first-class
protected attributes — the Constitution prohibits discrimination on the
basis of religion, race, caste, sex, or place of birth, and the keyword
list reflects that. The RBI's 2018 data-localisation circular requires
certain financial data to be stored in India, so ``data_residency_required``
is True.
"""
from __future__ import annotations

from .base import ExplainabilityLevel, JurisdictionBase
from .us import US_PROTECTED_KEYWORDS

#: India protected-attribute keyword denylist.
#: Constitution Art 15 + Rights of Persons with Disabilities Act 2016 +
#: domain-specific terms an LLM might surface.
INDIA_PROTECTED_KEYWORDS = sorted(set(US_PROTECTED_KEYWORDS) | {
    "caste", "dalit", "scheduled caste", "scheduled tribe", "tribal",
    "backward class", "obc",
    "place of birth", "birthplace",
    "mother tongue",
})


class India(JurisdictionBase):
    """India — RBI Fair Practices Code, DPDPA 2023, Constitution Art 15."""

    def __init__(self) -> None:
        super().__init__(
            code="IN",
            name="India",
            primary_regulator="RBI",
            applicable_laws=[
                "RBI Fair Practices Code",
                "RBI Digital Lending Directions 2025",
                "Credit Information Companies (Regulation) Act 2005 (CICRA)",
                "Digital Personal Data Protection Act 2023 (DPDPA)",
                "Constitution of India Article 15",
            ],
            protected_attributes=list(INDIA_PROTECTED_KEYWORDS),
            protected_categories={
                "religion": "Constitution Art 15",
                "race": "Constitution Art 15",
                "caste": "Constitution Art 15",
                "sex": "Constitution Art 15",
                "place of birth": "Constitution Art 15",
                "disability": "Rights of Persons with Disabilities Act 2016",
                "tribe": "Constitution Schedule V / VI",
            },
            adverse_action_template="india_rejection_notice.txt",
            mandatory_disclosures=[
                "RBI",
                "Fair Practices Code",
                "credit information",
                "DPDPA",
                "Integrated Ombudsman",
            ],
            disclosure_deadline_days=0,
            max_denial_reasons=4,
            explainability_level=ExplainabilityLevel.BASIC,
            requires_human_review=False,
            requires_model_registration=False,
            data_residency_required=True,    # RBI Storage of Payment System Data circular (2018)
            cross_border_restrictions=[],     # strict — onshore processing expected
            pii_law="DPDPA 2023",
            fairness_threshold=0.80,
            fairness_metric="fair_practices_code",
            fairness_enforcement="soft",
        )
