"""United States jurisdiction module.

Regulators: CFPB, OCC, Federal Reserve, FTC.
Key laws:   ECOA / Reg B, FCRA, SR 11-7, UDAAP, Dodd-Frank.

This module is the reference implementation — every other jurisdiction
follows the same shape and the existing test suite pins the US output
byte-for-byte against the legacy hardcoded constants in
:mod:`workflow.pipeline`.
"""
from __future__ import annotations

from .base import ExplainabilityLevel, JurisdictionBase

#: Protected-attribute keyword denylist used to scan LLM output.
#: Mirrors :data:`workflow.config.DEFAULT_PROTECTED_KEYWORDS` and the
#: legacy :attr:`workflow.pipeline.CreditWorkflow.PROTECTED_KEYWORDS`.
US_PROTECTED_KEYWORDS = [
    "race", "racial", "ethnicity", "ethnic", "skin color",
    "gender", "sex", "male", "female", "woman", "man",
    "age", "old", "young", "elderly", "senior",
    "religion", "religious", "muslim", "christian", "jewish", "hindu",
    "national origin", "immigrant", "foreign",
    "disability", "disabled", "handicap",
    "marital status", "married", "single", "divorced",
    "pregnant", "pregnancy",
]


class US(JurisdictionBase):
    """United States — ECOA / FCRA / four-fifths rule."""

    def __init__(self) -> None:
        super().__init__(
            code="US",
            name="United States",
            primary_regulator="CFPB",
            applicable_laws=[
                "ECOA / Regulation B",
                "FCRA",
                "SR 11-7 (Model Risk Management)",
                "UDAAP",
                "Dodd-Frank Act",
            ],
            protected_attributes=list(US_PROTECTED_KEYWORDS),
            protected_categories={
                "race": "ECOA / Regulation B",
                "color": "ECOA / Regulation B",
                "religion": "ECOA / Regulation B",
                "national origin": "ECOA / Regulation B",
                "sex": "ECOA / Regulation B",
                "marital status": "ECOA / Regulation B",
                "age": "ECOA / Regulation B",
                "receipt of public assistance": "ECOA / Regulation B",
                "exercise of rights under CCPA": "FCRA",
            },
            adverse_action_template="us_adverse_action.txt",
            mandatory_disclosures=[
                "FAIR CREDIT REPORTING ACT",
                "consumer reporting agency",
                "right to dispute",
                "free copy of your credit report",
            ],
            disclosure_deadline_days=30,
            max_denial_reasons=4,
            explainability_level=ExplainabilityLevel.DETAILED,
            requires_human_review=False,
            requires_model_registration=False,
            data_residency_required=False,
            cross_border_restrictions=[],
            pii_law="GLBA",
            fairness_threshold=0.80,
            fairness_metric="four_fifths",
            fairness_enforcement="hard",
        )
