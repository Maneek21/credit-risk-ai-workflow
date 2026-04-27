"""United Arab Emirates jurisdiction module.

Regulators: CBUAE (Central Bank of the UAE),
            ADGM-FSRA (Abu Dhabi Global Market — Financial Services
            Regulatory Authority),
            DFSA (Dubai Financial Services Authority — DIFC).
Key laws:   Federal Decree-Law No. 6 of 2025 (consolidated financial
            regulation),
            CBUAE Guidelines for Financial Institutions Adopting
            Enabling Technologies,
            Federal Decree-Law No. 45 of 2021 on the Protection of
            Personal Data ("PDPL"),
            ADGM Data Protection Regulations 2021 / DIFC Data
            Protection Law 2020 (free-zone tracks),
            UAE National AI Strategy 2031.

Distinguishing feature: dual-track regulation — onshore CBUAE for
mainland banks vs ADGM/DIFC free-zone regulators with their own
GDPR-aligned data protection regimes. Islamic-finance structures
(murabaha, ijara, tawarruq) may shape the contractual form of the
credit but do not change the underlying creditworthiness assessment.
``Sectarian`` (e.g. Sunni / Shia identity) is included as a
protected-attribute keyword because politically-charged sectarian
framing is sensitive in the region and must not appear in a
credit memo.
"""
from __future__ import annotations

from .base import ExplainabilityLevel, JurisdictionBase
from .us import US_PROTECTED_KEYWORDS

#: UAE protected-attribute keyword denylist — PDPL Art 7 sensitive data
#: + region-specific sensitivities.
UAE_PROTECTED_KEYWORDS = sorted(set(US_PROTECTED_KEYWORDS) | {
    "tribal",
    "sectarian",
    "shia", "sunni",
    "health data",
    "biometric",
    "criminal record",
})


class UAE(JurisdictionBase):
    """UAE — CBUAE / ADGM-FSRA / DFSA / PDPL."""

    def __init__(self) -> None:
        super().__init__(
            code="AE",
            name="United Arab Emirates",
            primary_regulator="CBUAE",
            applicable_laws=[
                "Federal Decree-Law No. 6 of 2025 (consolidated financial regulation)",
                "CBUAE Guidelines for Financial Institutions Adopting Enabling Technologies",
                "Federal Decree-Law No. 45 of 2021 (PDPL)",
                "ADGM Data Protection Regulations 2021 (free-zone)",
                "DIFC Data Protection Law 2020 (free-zone)",
                "UAE National AI Strategy 2031",
            ],
            protected_attributes=list(UAE_PROTECTED_KEYWORDS),
            protected_categories={
                "race": "PDPL Art 7 — sensitive personal data",
                "ethnicity": "PDPL Art 7 — sensitive personal data",
                "religion": "PDPL Art 7 — sensitive personal data",
                "political opinion": "PDPL Art 7 — sensitive personal data",
                "criminal record": "PDPL Art 7 — sensitive personal data",
                "health data": "PDPL Art 7 — sensitive personal data",
                "biometric data": "PDPL Art 7 — sensitive personal data",
            },
            adverse_action_template="uae_rejection_notice.txt",
            mandatory_disclosures=[
                "CBUAE",
                "PDPL",
                "Federal Decree-Law No. 6 of 2025",
                "UAE Data Office",
            ],
            disclosure_deadline_days=0,
            max_denial_reasons=4,
            explainability_level=ExplainabilityLevel.BASIC,
            requires_human_review=False,
            requires_model_registration=False,
            data_residency_required=False,
            cross_border_restrictions=[],
            pii_law="PDPL (Federal Decree-Law No. 45/2021)",
            fairness_threshold=0.80,
            fairness_metric="cbuae_fair_treatment",
            fairness_enforcement="soft",
        )
