"""European Union jurisdiction module.

Regulators: EBA, ECB (SSM), national NCAs, national DPAs.
Key laws:   EU AI Act (Regulation (EU) 2024/1689), GDPR (Regulation
            (EU) 2016/679), Consumer Credit Directive 2023/2225,
            EBA Guidelines on loan origination and monitoring (EBA/GL/2020/06).

Credit scoring is a HIGH-RISK system under Annex III of the EU AI Act
(full enforcement Aug 2 2026), which means the deploying institution
must run a conformity assessment, register the system in the EU
database, document risk management, and provide meaningful information
about the logic to affected persons under GDPR Art 22(3) and AI Act
Art 13.

The protected-attribute list is the broadest among supported markets
because GDPR Art 9 "special categories" (health, political opinions,
trade union membership, biometric data, sexual orientation, etc.) sit
on top of general non-discrimination rules.
"""
from __future__ import annotations

from .base import ExplainabilityLevel, JurisdictionBase
from .uk import UK_PROTECTED_KEYWORDS

#: EU protected-attribute keyword denylist.
#: Layered on top of the UK Equality Act terms with GDPR Art 9
#: special-category data and EU non-discrimination terms.
EU_PROTECTED_KEYWORDS = sorted(set(UK_PROTECTED_KEYWORDS) | {
    "racial origin", "ethnic origin",
    "political opinion", "political opinions", "political view",
    "trade union", "trade-union membership",
    "religious belief", "religious beliefs", "philosophical beliefs",
    "genetic data", "biometric data",
    "health data", "health condition", "medical condition",
    "sex life",
})


class EU(JurisdictionBase):
    """European Union — GDPR, EU AI Act, Consumer Credit Directive 2023.

    Treated as a single jurisdiction covering the harmonised framework.
    Member-state divergences (e.g. national CCD transposition deadlines,
    national supervisory authorities) belong in deployment notes, not in
    this code.
    """

    def __init__(self) -> None:
        super().__init__(
            code="EU",
            name="European Union",
            primary_regulator="EBA / national NCAs",
            applicable_laws=[
                "EU AI Act (Regulation (EU) 2024/1689)",
                "GDPR (Regulation (EU) 2016/679)",
                "Consumer Credit Directive 2023/2225",
                "EBA Guidelines on loan origination (EBA/GL/2020/06)",
            ],
            protected_attributes=list(EU_PROTECTED_KEYWORDS),
            protected_categories={
                "racial or ethnic origin": "GDPR Art 9(1)",
                "political opinions": "GDPR Art 9(1)",
                "religious or philosophical beliefs": "GDPR Art 9(1)",
                "trade union membership": "GDPR Art 9(1)",
                "genetic data": "GDPR Art 9(1)",
                "biometric data": "GDPR Art 9(1)",
                "data concerning health": "GDPR Art 9(1)",
                "data concerning sex life or sexual orientation": "GDPR Art 9(1)",
                "non-discrimination (general)": "Charter of Fundamental Rights Art 21",
            },
            adverse_action_template="eu_adverse_action.txt",
            mandatory_disclosures=[
                "EU AI Act",
                "GDPR",
                "Article 22",
                "Consumer Credit Directive",
                "human intervention",
                "meaningful information about the logic",
            ],
            disclosure_deadline_days=0,  # "without undue delay" — member states may specify
            max_denial_reasons=4,
            explainability_level=ExplainabilityLevel.ALGORITHMIC,
            requires_human_review=True,           # AI Act Art 14 — human oversight
            requires_model_registration=True,     # AI Act Art 49 — EU database
            data_residency_required=True,
            cross_border_restrictions=["EEA"],    # Norway, Iceland, Liechtenstein OK
            pii_law="GDPR",
            # No single statutory threshold; AI Act mandates bias monitoring
            # and mitigation but not a numeric DI floor. Carry 0.80 as a
            # working benchmark; banks should align with their NCA guidance.
            fairness_threshold=0.80,
            fairness_metric="bias_monitoring_ai_act",
            fairness_enforcement="hard",
        )
