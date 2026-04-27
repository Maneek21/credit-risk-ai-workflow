"""United Kingdom jurisdiction module.

Regulators: FCA (conduct), PRA (prudential), ICO (data).
Key laws:   Consumer Credit Act 1974, Equality Act 2010, UK GDPR /
            Data Protection Act 2018, FCA CONC sourcebook, Consumer
            Duty (PRIN 2A, 2023).

The UK protected-characteristic list (Equality Act 2010) is broader
than the US ECOA list — it adds ``gender reassignment``, ``civil
partnership``, ``maternity``, ``sexual orientation``, and explicit
``belief``.
"""
from __future__ import annotations

from .base import ExplainabilityLevel, JurisdictionBase
from .us import US_PROTECTED_KEYWORDS

#: UK protected-characteristic keyword denylist.
#: Equality Act 2010 enumerates nine protected characteristics; we
#: include lexical variants the LLM is most likely to produce.
UK_PROTECTED_KEYWORDS = sorted(set(US_PROTECTED_KEYWORDS) | {
    "gender reassignment", "transgender", "trans",
    "civil partnership", "civil partner",
    "maternity", "paternity",
    "sexual orientation", "gay", "lesbian", "bisexual", "queer", "lgbt",
    "belief", "philosophical belief",
})


class UK(JurisdictionBase):
    """United Kingdom — Consumer Credit Act / Equality Act / Consumer Duty."""

    def __init__(self) -> None:
        super().__init__(
            code="UK",
            name="United Kingdom",
            primary_regulator="FCA",
            applicable_laws=[
                "Consumer Credit Act 1974",
                "Equality Act 2010",
                "UK GDPR / Data Protection Act 2018",
                "FCA CONC sourcebook",
                "Consumer Duty (PRIN 2A, 2023)",
            ],
            protected_attributes=list(UK_PROTECTED_KEYWORDS),
            protected_categories={
                "age": "Equality Act 2010",
                "disability": "Equality Act 2010",
                "gender reassignment": "Equality Act 2010",
                "marriage and civil partnership": "Equality Act 2010",
                "pregnancy and maternity": "Equality Act 2010",
                "race": "Equality Act 2010",
                "religion or belief": "Equality Act 2010",
                "sex": "Equality Act 2010",
                "sexual orientation": "Equality Act 2010",
            },
            adverse_action_template="uk_rejection_notice.txt",
            mandatory_disclosures=[
                "Consumer Credit Act 1974",
                "Consumer Duty",
                "credit reference agency",
                "Financial Ombudsman Service",
                "UK GDPR Article 22",
            ],
            disclosure_deadline_days=0,  # "without unreasonable delay" — no fixed day count
            max_denial_reasons=4,
            explainability_level=ExplainabilityLevel.DETAILED,
            requires_human_review=False,
            requires_model_registration=False,
            data_residency_required=False,
            cross_border_restrictions=[],
            pii_law="UK GDPR / Data Protection Act 2018",
            # The UK has no statutory four-fifths equivalent. We carry the US
            # 0.80 figure as a working benchmark; FCA Consumer Duty is
            # outcome-based ("good outcomes") rather than threshold-based.
            fairness_threshold=0.80,
            fairness_metric="outcome_based_consumer_duty",
            fairness_enforcement="soft",
        )
