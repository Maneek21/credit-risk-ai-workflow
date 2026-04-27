"""Brazil jurisdiction module.

Regulators: BCB (Banco Central do Brasil), CVM (securities),
            SENACON (consumer).
Key laws:   LGPD (Lei nº 13.709/2018, Brazil's GDPR equivalent),
            Consumer Defence Code (Lei nº 8.078/1990, "CDC"),
            Credit Information Act (Lei nº 12.414/2011, "Cadastro
            Positivo"),
            BCB Resolution 4.893/2021 (cybersecurity & technology),
            AI Bill PL 2338/2023 (passed Senate late 2024, pending
            in the Chamber of Deputies).

Distinguishing feature: **LGPD Article 20** explicitly grants the
data subject the right to request review of decisions made solely on
the basis of automated processing — including credit-scoring
decisions — and the controller must provide clear and adequate
information about the criteria and procedures used.
``Quilombola`` (descendants of historical maroon communities) and
``indigenous`` are included as protected-attribute keywords because
both groups are explicitly recognised under the Brazilian
Constitution and may surface in narrative LLM output.
"""
from __future__ import annotations

from .base import ExplainabilityLevel, JurisdictionBase
from .us import US_PROTECTED_KEYWORDS

#: Brazil protected-attribute keyword denylist — LGPD Art 5(II) sensitive
#: data + Constitutional protected groups.
BRAZIL_PROTECTED_KEYWORDS = sorted(set(US_PROTECTED_KEYWORDS) | {
    "origin",
    "trade union",
    "philosophical view", "philosophical conviction",
    "political view",
    "genetic data",
    "biometric data",
    "quilombola",
    "indigenous",
})


class Brazil(JurisdictionBase):
    """Brazil — LGPD / CDC / Cadastro Positivo / BCB."""

    def __init__(self) -> None:
        super().__init__(
            code="BR",
            name="Brazil",
            primary_regulator="BCB",
            applicable_laws=[
                "LGPD (Lei nº 13.709/2018)",
                "Consumer Defence Code (Lei nº 8.078/1990, CDC)",
                "Credit Information Act (Lei nº 12.414/2011, Cadastro Positivo)",
                "BCB Resolution 4.893/2021 (cybersecurity & technology)",
                "AI Bill PL 2338/2023 (pending)",
            ],
            protected_attributes=list(BRAZIL_PROTECTED_KEYWORDS),
            protected_categories={
                "racial or ethnic origin": "LGPD Art 5(II)",
                "religious conviction": "LGPD Art 5(II)",
                "political opinion": "LGPD Art 5(II)",
                "trade union membership": "LGPD Art 5(II)",
                "philosophical or religious belief": "LGPD Art 5(II)",
                "data concerning health": "LGPD Art 5(II)",
                "sex life": "LGPD Art 5(II)",
                "genetic data": "LGPD Art 5(II)",
                "biometric data": "LGPD Art 5(II)",
            },
            adverse_action_template="brazil_adverse_action.txt",
            mandatory_disclosures=[
                "LGPD",
                "Article 20",
                "Consumer Defence Code",
                "Cadastro Positivo",
                "ANPD",
            ],
            disclosure_deadline_days=0,    # CDC: "immediate"
            max_denial_reasons=4,
            explainability_level=ExplainabilityLevel.DETAILED,
            requires_human_review=False,    # LGPD Art 20 — right to *request* review
            requires_model_registration=False,
            data_residency_required=False,
            cross_border_restrictions=[],
            pii_law="LGPD",
            fairness_threshold=0.80,
            fairness_metric="lgpd_non_discrimination",
            fairness_enforcement="soft",
        )
