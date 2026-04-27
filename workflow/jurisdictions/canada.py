"""Canada jurisdiction module.

Regulators: OSFI (federal prudential), FCAC (federal consumer),
            provincial regulators (Quebec AMF, Ontario FSRA, BC BCFSA, ...).
Key laws:   OSFI Guideline E-23 (Model Risk Management, effective May 2027),
            Bank Act (R.S.C. 1991, c. 46),
            Personal Information Protection and Electronic Documents Act
            ("PIPEDA"), Canadian Human Rights Act (R.S.C. 1985, c. H-6).

Distinguishing feature: the Canadian Human Rights Act enumerates
**13 prohibited grounds of discrimination** — one of the broadest lists
in the Americas, adding ``gender identity or expression``, ``family
status``, ``genetic characteristics``, and ``pardoned conviction`` on
top of the US ECOA list.
"""
from __future__ import annotations

from .base import ExplainabilityLevel, JurisdictionBase
from .us import US_PROTECTED_KEYWORDS

#: Canada protected-attribute keyword denylist — Canadian Human Rights Act.
CANADA_PROTECTED_KEYWORDS = sorted(set(US_PROTECTED_KEYWORDS) | {
    "gender identity", "gender expression",
    "family status",
    "genetic", "genetic characteristics",
    "pardoned conviction", "record suspended",
})


class Canada(JurisdictionBase):
    """Canada — Bank Act / CHRA / PIPEDA / OSFI E-23."""

    def __init__(self) -> None:
        super().__init__(
            code="CA",
            name="Canada",
            primary_regulator="OSFI",
            applicable_laws=[
                "OSFI Guideline E-23 (effective May 2027)",
                "Bank Act (R.S.C. 1991, c. 46)",
                "Personal Information Protection and Electronic Documents Act (PIPEDA)",
                "Canadian Human Rights Act (R.S.C. 1985, c. H-6)",
            ],
            protected_attributes=list(CANADA_PROTECTED_KEYWORDS),
            protected_categories={
                "race": "Canadian Human Rights Act s.3",
                "national or ethnic origin": "Canadian Human Rights Act s.3",
                "colour": "Canadian Human Rights Act s.3",
                "religion": "Canadian Human Rights Act s.3",
                "age": "Canadian Human Rights Act s.3",
                "sex": "Canadian Human Rights Act s.3",
                "sexual orientation": "Canadian Human Rights Act s.3",
                "gender identity or expression": "Canadian Human Rights Act s.3",
                "marital status": "Canadian Human Rights Act s.3",
                "family status": "Canadian Human Rights Act s.3",
                "genetic characteristics": "Canadian Human Rights Act s.3",
                "disability": "Canadian Human Rights Act s.3",
                "pardoned conviction": "Canadian Human Rights Act s.3",
            },
            adverse_action_template="canada_adverse_action.txt",
            mandatory_disclosures=[
                "Bank Act",
                "Canadian Human Rights Act",
                "PIPEDA",
                "credit reporting agency",
                "Financial Consumer Agency of Canada",
            ],
            disclosure_deadline_days=0,        # Varies by province
            max_denial_reasons=4,
            explainability_level=ExplainabilityLevel.DETAILED,
            requires_human_review=False,
            requires_model_registration=False,  # E-23 = internal model inventory, not public registration
            data_residency_required=False,
            cross_border_restrictions=[],
            pii_law="PIPEDA",
            fairness_threshold=0.80,
            fairness_metric="fairness_assessment_e23",
            fairness_enforcement="soft",
        )
