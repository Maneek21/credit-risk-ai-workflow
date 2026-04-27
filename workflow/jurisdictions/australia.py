"""Australia jurisdiction module.

Regulators: APRA (prudential), ASIC (conduct), OAIC (privacy).
Key laws:   National Consumer Credit Protection Act 2009 ("NCCP Act"),
            APRA CPS 230 (Operational Risk Management),
            APRA CPS 234 (Information Security),
            Privacy Act 1988 (with automated-decision disclosure
            obligations from December 2026),
            Australian Consumer Law,
            federal and state Anti-Discrimination Acts.

Distinguishing feature: ``carer responsibilities`` and ``family/carer``
status are explicitly protected, ``national extraction`` and
``social origin`` are recognised under the Fair Work Act / federal
discrimination framework, and ``intersex status`` is enumerated in
the Sex Discrimination Act 1984.
"""
from __future__ import annotations

from .base import ExplainabilityLevel, JurisdictionBase
from .us import US_PROTECTED_KEYWORDS

#: Australia protected-attribute keyword denylist.
AUSTRALIA_PROTECTED_KEYWORDS = sorted(set(US_PROTECTED_KEYWORDS) | {
    "carer responsibilities",
    "family responsibilities",
    "social origin",
    "national extraction",
    "intersex",
    "gender identity",
})


class Australia(JurisdictionBase):
    """Australia — NCCP Act / Privacy Act / APRA CPS 230 / responsible lending."""

    def __init__(self) -> None:
        super().__init__(
            code="AU",
            name="Australia",
            primary_regulator="APRA",
            applicable_laws=[
                "National Consumer Credit Protection Act 2009 (NCCP Act)",
                "APRA CPS 230 (Operational Risk Management)",
                "APRA CPS 234 (Information Security)",
                "Privacy Act 1988 (Cth)",
                "Australian Consumer Law (Schedule 2, CCA 2010)",
                "Anti-Discrimination Acts (federal and state)",
            ],
            protected_attributes=list(AUSTRALIA_PROTECTED_KEYWORDS),
            protected_categories={
                "race": "Racial Discrimination Act 1975",
                "colour": "Racial Discrimination Act 1975",
                "sex": "Sex Discrimination Act 1984",
                "sexual orientation": "Sex Discrimination Act 1984",
                "gender identity": "Sex Discrimination Act 1984",
                "intersex status": "Sex Discrimination Act 1984",
                "marital or relationship status": "Sex Discrimination Act 1984",
                "pregnancy": "Sex Discrimination Act 1984",
                "family or carer responsibilities": "Sex Discrimination Act 1984",
                "age": "Age Discrimination Act 2004",
                "disability": "Disability Discrimination Act 1992",
                "religion": "Fair Work Act 2009 / state laws",
                "political opinion": "Fair Work Act 2009",
                "national extraction": "Fair Work Act 2009",
                "social origin": "Fair Work Act 2009",
            },
            adverse_action_template="australia_rejection_notice.txt",
            mandatory_disclosures=[
                "National Consumer Credit Protection Act",
                "responsible lending",
                "credit reporting body",
                "Privacy Act",
                "Australian Financial Complaints Authority",
            ],
            disclosure_deadline_days=0,
            max_denial_reasons=4,
            explainability_level=ExplainabilityLevel.DETAILED,
            requires_human_review=False,
            requires_model_registration=False,
            data_residency_required=False,
            cross_border_restrictions=[],
            pii_law="Privacy Act 1988",
            fairness_threshold=0.80,
            fairness_metric="responsible_lending",
            fairness_enforcement="soft",
        )
