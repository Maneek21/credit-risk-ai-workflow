"""Tests for the jurisdiction abstraction (Phase 1 + Phase 2 of the plan).

Coverage:
  * Base interface — every supported jurisdiction loads, exposes a
    non-empty disclosure block, has a non-empty protected-attribute
    keyword list, and ships a non-empty ``mandatory_disclosures`` set.
  * US regression — US() must produce a disclosure block byte-for-byte
    equal to the legacy CreditWorkflow.FCRA_DISCLOSURE constant; its
    keyword list must equal the legacy class attribute.
  * UK / EU specifics — each must include region-specific legal text
    in the disclosure block and add region-specific keywords beyond US.
  * validate_notice — flags missing disclosures correctly.
  * Pipeline integration — passing jurisdiction=UK() changes the
    disclosure block appended to a denial.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# Set dummy API key BEFORE importing the workflow module.
os.environ.setdefault("OPENAI_API_KEY", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from workflow.jurisdictions import (  # noqa: E402
    ALL_JURISDICTIONS as JURISDICTION_REGISTRY,
    AUSTRALIA_PROTECTED_KEYWORDS,
    Australia,
    BRAZIL_PROTECTED_KEYWORDS,
    Brazil,
    CANADA_PROTECTED_KEYWORDS,
    Canada,
    EU,
    EU_PROTECTED_KEYWORDS,
    ExplainabilityLevel,
    INDIA_PROTECTED_KEYWORDS,
    India,
    JAPAN_PROTECTED_KEYWORDS,
    Japan,
    JurisdictionBase,
    SINGAPORE_PROTECTED_KEYWORDS,
    Singapore,
    UAE,
    UAE_PROTECTED_KEYWORDS,
    UK,
    UK_PROTECTED_KEYWORDS,
    US,
    US_PROTECTED_KEYWORDS,
)
from workflow.pipeline import CreditWorkflow  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (mirrors test_safety_layers.py)
# ---------------------------------------------------------------------------

ALL_JURISDICTIONS = [
    US, UK, EU,
    India, Canada, Australia, Singapore, Japan, UAE, Brazil,
]


def _sample_applicant() -> Dict[str, Any]:
    return {
        "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
        "PAY_0": 2, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 12000, "BILL_AMT2": 11500, "BILL_AMT3": 11000,
        "BILL_AMT4": 10500, "BILL_AMT5": 10000, "BILL_AMT6": 9500,
        "PAY_AMT1": 500, "PAY_AMT2": 500, "PAY_AMT3": 500,
        "PAY_AMT4": 500, "PAY_AMT5": 500, "PAY_AMT6": 500,
    }


def _make_mock_model(default_prob: float) -> MagicMock:
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array(
        [[1.0 - default_prob, default_prob]]
    )
    final = MagicMock()
    final.__class__.__name__ = "MockClassifier"
    mock_model.__getitem__.return_value = final
    del mock_model.named_steps
    return mock_model


def _shap_factors(features: List[str]) -> List[Dict[str, Any]]:
    return [
        {"feature": f, "shap_value": 0.5 - 0.1 * i,
         "direction": "increases risk" if i % 2 == 0 else "decreases risk"}
        for i, f in enumerate(features)
    ]


def _build_wf(default_prob: float, jurisdiction=None) -> CreditWorkflow:
    mock_model = _make_mock_model(default_prob)
    with patch("workflow.pipeline.joblib.load", return_value=mock_model), patch(
        "workflow.pipeline.OpenAI"
    ) as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()
        return CreditWorkflow(
            model_path="unused.joblib",
            llm_provider="openai",
            llm_model="gpt-4o-mock",
            jurisdiction=jurisdiction,
        )


# ---------------------------------------------------------------------------
# 1. Base interface — every jurisdiction loads cleanly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_JURISDICTIONS)
def test_jurisdiction_loads(cls) -> None:
    j = cls()
    assert isinstance(j, JurisdictionBase)
    assert isinstance(j.code, str) and len(j.code) >= 2
    assert j.name
    assert j.primary_regulator
    assert j.applicable_laws


@pytest.mark.parametrize("cls", ALL_JURISDICTIONS)
def test_disclosure_block_non_empty(cls) -> None:
    block = cls().get_disclosure_block()
    assert isinstance(block, str)
    assert len(block) > 100  # All real disclosures are at least a few sentences


@pytest.mark.parametrize("cls", ALL_JURISDICTIONS)
def test_protected_keywords_non_empty(cls) -> None:
    kws = cls().get_protected_keywords()
    assert isinstance(kws, list)
    assert len(kws) >= 10
    # All keywords must be lowercased non-empty strings.
    for kw in kws:
        assert isinstance(kw, str) and kw == kw.lower() and kw.strip()


@pytest.mark.parametrize("cls", ALL_JURISDICTIONS)
def test_mandatory_disclosures_non_empty(cls) -> None:
    md = cls().mandatory_disclosures
    assert isinstance(md, list)
    assert len(md) >= 3
    for d in md:
        assert isinstance(d, str) and d.strip()


# ---------------------------------------------------------------------------
# 2. US regression — byte-for-byte equality with legacy constants
# ---------------------------------------------------------------------------


def test_us_disclosure_byte_identical_to_legacy_constant() -> None:
    """The US template MUST match CreditWorkflow.FCRA_DISCLOSURE exactly.

    If this breaks, the migration has drifted and existing audit records
    will no longer endswith() the constant.
    """
    assert US().get_disclosure_block() == CreditWorkflow.FCRA_DISCLOSURE


def test_us_protected_keywords_match_legacy_constant() -> None:
    """US keyword list must equal the legacy class constant set."""
    assert sorted(US_PROTECTED_KEYWORDS) == sorted(CreditWorkflow.PROTECTED_KEYWORDS)


def test_us_validate_notice_passes_legacy_disclosure() -> None:
    issues = US().validate_notice(CreditWorkflow.FCRA_DISCLOSURE)
    assert issues == []


def test_us_validate_notice_fails_garbage() -> None:
    issues = US().validate_notice("hello world")
    assert len(issues) == len(US().mandatory_disclosures)


# ---------------------------------------------------------------------------
# 3. UK / EU — region-specific content present
# ---------------------------------------------------------------------------


def test_uk_disclosure_mentions_consumer_credit_act() -> None:
    block = UK().get_disclosure_block()
    assert "Consumer Credit Act 1974" in block
    assert "Consumer Duty" in block
    assert "Financial Ombudsman" in block


def test_uk_keywords_extend_us_with_equality_act_terms() -> None:
    uk_kws = set(UK_PROTECTED_KEYWORDS)
    us_kws = set(US_PROTECTED_KEYWORDS)
    assert us_kws.issubset(uk_kws)
    # Equality Act 2010 protected characteristics not in the US list:
    for added in {"gender reassignment", "civil partnership",
                  "sexual orientation", "maternity"}:
        assert added in uk_kws


def test_eu_disclosure_mentions_ai_act_and_gdpr() -> None:
    block = EU().get_disclosure_block()
    assert "EU AI Act" in block
    assert "GDPR" in block
    assert "Article 22" in block
    assert "Consumer Credit Directive" in block


def test_eu_marks_high_risk_oversight_requirements() -> None:
    eu = EU()
    assert eu.requires_human_review is True
    assert eu.requires_model_registration is True
    assert eu.data_residency_required is True
    assert eu.explainability_level == ExplainabilityLevel.ALGORITHMIC


def test_eu_keywords_include_gdpr_special_categories() -> None:
    eu_kws = set(EU_PROTECTED_KEYWORDS)
    for special in {"genetic data", "biometric data", "health data",
                    "trade union", "political opinion"}:
        assert special in eu_kws


# ---------------------------------------------------------------------------
# 4. Pipeline integration — jurisdiction wiring works end-to-end
# ---------------------------------------------------------------------------


def test_default_jurisdiction_is_us() -> None:
    wf = _build_wf(default_prob=0.20)
    assert isinstance(wf.jurisdiction, US)


def test_uk_jurisdiction_appends_uk_disclosure_on_denial() -> None:
    wf = _build_wf(default_prob=0.85, jurisdiction=UK())
    body = "Your application was not approved due to repayment delays."
    with patch.object(wf, "_call_llm", return_value=body), patch.object(
        wf, "_compute_shap_factors",
        return_value=_shap_factors(["PAY_0", "BILL_AMT1", "LIMIT_BAL"]),
    ):
        result = wf.process_application(_sample_applicant())

    assert result.decision == "DENY"
    assert result.adverse_action is not None
    assert result.adverse_action.endswith(UK().get_disclosure_block())
    # Crucially: it must NOT end with the US FCRA text.
    assert not result.adverse_action.endswith(CreditWorkflow.FCRA_DISCLOSURE)
    # The original LLM body is still present.
    assert body in result.adverse_action
    # Audit metadata records the active jurisdiction.
    assert result.metadata["jurisdiction"] == "UK"


def test_eu_jurisdiction_uses_broader_keyword_filter() -> None:
    """A memo mentioning trade-union membership must trip the EU filter."""
    wf = _build_wf(default_prob=0.20, jurisdiction=EU())
    bad_memo = "the borrower has stable trade union employment"
    with patch.object(wf, "_call_llm", return_value=bad_memo), patch.object(
        wf, "_compute_shap_factors",
        return_value=_shap_factors(["PAY_0", "BILL_AMT1", "LIMIT_BAL"]),
    ):
        result = wf.process_application(_sample_applicant())
    assert "PROTECTED_ATTR_DETECTED" in result.flags
    assert "trade union" in result.metadata["protected_attribute_hits"]


def test_explicit_protected_attributes_override_jurisdiction() -> None:
    """Per-call override still wins over jurisdiction defaults."""
    wf_factory_model = _make_mock_model(0.20)
    with patch("workflow.pipeline.joblib.load", return_value=wf_factory_model), patch(
        "workflow.pipeline.OpenAI"
    ) as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()
        wf = CreditWorkflow(
            model_path="unused.joblib",
            llm_provider="openai",
            llm_model="gpt-4o-mock",
            jurisdiction=EU(),
            protected_attributes=["custom_word_only"],
        )
    # The override takes precedence over EU's broad list.
    assert wf.protected_keywords == ["custom_word_only"]


# ---------------------------------------------------------------------------
# 5. Data residency warning helper
# ---------------------------------------------------------------------------


def test_us_no_residency_warning() -> None:
    assert US().data_residency_warning(None) is None
    assert US().data_residency_warning("us-east-1") is None


def test_eu_warns_when_endpoint_outside_eea() -> None:
    msg = EU().data_residency_warning("US")
    assert msg is not None and "EU" in msg


def test_eu_no_warn_when_endpoint_in_eea() -> None:
    assert EU().data_residency_warning("EEA") is None
    assert EU().data_residency_warning("EU") is None


# ---------------------------------------------------------------------------
# 6. New-jurisdiction specifics (Phase 3 + Phase 4)
# ---------------------------------------------------------------------------


def test_india_keywords_include_caste() -> None:
    """Constitution Art 15 protects against caste discrimination — must be
    represented in the keyword list."""
    kws = set(INDIA_PROTECTED_KEYWORDS)
    for term in {"caste", "dalit", "scheduled caste", "scheduled tribe"}:
        assert term in kws, f"India keyword list missing {term!r}"


def test_india_data_residency_required() -> None:
    """RBI Storage of Payment System Data circular (2018) requires
    in-country storage for certain financial data."""
    india = India()
    assert india.data_residency_required is True
    msg = india.data_residency_warning("US")
    assert msg is not None and "IN" in msg


def test_india_disclosure_mentions_rbi_fair_practices() -> None:
    block = India().get_disclosure_block()
    assert "RBI" in block
    assert "Fair Practices Code" in block
    assert "DPDPA" in block


def test_canada_keywords_include_13_chra_grounds() -> None:
    """Canadian Human Rights Act enumerates 13 grounds — verify the
    distinctive ones not present in the US list."""
    kws = set(CANADA_PROTECTED_KEYWORDS)
    for term in {"gender identity", "gender expression",
                 "family status", "genetic", "pardoned conviction"}:
        assert term in kws, f"Canada keyword list missing {term!r}"


def test_australia_keywords_include_carer_responsibilities() -> None:
    """Sex Discrimination Act 1984 protects family/carer responsibilities."""
    kws = set(AUSTRALIA_PROTECTED_KEYWORDS)
    for term in {"carer responsibilities", "family responsibilities",
                 "intersex", "social origin", "national extraction"}:
        assert term in kws, f"Australia keyword list missing {term!r}"


def test_singapore_keywords_include_descent_and_language() -> None:
    """Constitution Art 12 protects against discrimination on grounds of
    religion, race, descent, or place of birth; community language rights
    sit alongside."""
    kws = set(SINGAPORE_PROTECTED_KEYWORDS)
    for term in {"descent", "place of birth", "language",
                 "mother tongue", "dialect"}:
        assert term in kws, f"Singapore keyword list missing {term!r}"


def test_japan_keywords_include_appi_special_care() -> None:
    """APPI Art 2(3) special care-required information is the unique JP set."""
    kws = set(JAPAN_PROTECTED_KEYWORDS)
    for term in {"creed", "social status", "family origin",
                 "medical history", "criminal record", "burakumin"}:
        assert term in kws, f"Japan keyword list missing {term!r}"


def test_uae_keywords_include_pdpl_sensitive_data() -> None:
    """PDPL Art 7 sensitive data + region-specific sensitivities."""
    kws = set(UAE_PROTECTED_KEYWORDS)
    for term in {"tribal", "sectarian", "shia", "sunni",
                 "health data", "biometric", "criminal record"}:
        assert term in kws, f"UAE keyword list missing {term!r}"


def test_brazil_keywords_include_lgpd_sensitive_data() -> None:
    """LGPD Art 5(II) sensitive personal data categories."""
    kws = set(BRAZIL_PROTECTED_KEYWORDS)
    for term in {"trade union", "philosophical view",
                 "genetic data", "biometric data",
                 "quilombola", "indigenous"}:
        assert term in kws, f"Brazil keyword list missing {term!r}"


def test_brazil_disclosure_mentions_lgpd_art_20() -> None:
    """LGPD Art 20 right-to-review is the most distinctive Brazilian rule."""
    block = Brazil().get_disclosure_block()
    assert "LGPD" in block
    assert "Article 20" in block
    assert "Cadastro Positivo" in block


def test_canada_disclosure_mentions_pipeda_and_chra() -> None:
    block = Canada().get_disclosure_block()
    assert "PIPEDA" in block
    assert "Canadian Human Rights Act" in block
    assert "Bank Act" in block


def test_australia_disclosure_mentions_responsible_lending() -> None:
    block = Australia().get_disclosure_block()
    assert "National Consumer Credit Protection Act" in block
    assert "responsible lending" in block
    assert "Australian Financial Complaints Authority" in block


def test_singapore_disclosure_mentions_feat_and_pdpa() -> None:
    block = Singapore().get_disclosure_block()
    assert "MAS" in block
    assert "FEAT" in block
    assert "PDPA" in block


def test_japan_disclosure_mentions_appi_and_fsa() -> None:
    block = Japan().get_disclosure_block()
    assert "APPI" in block
    assert "Banking Act" in block
    assert "Instalment Sales Act" in block


def test_uae_disclosure_mentions_cbuae_and_pdpl() -> None:
    block = UAE().get_disclosure_block()
    assert "CBUAE" in block
    assert "PDPL" in block
    assert "Federal Decree-Law" in block


# ---------------------------------------------------------------------------
# 7. Cross-jurisdiction invariants
# ---------------------------------------------------------------------------


def test_all_jurisdictions_have_unique_codes() -> None:
    codes = [cls().code for cls in ALL_JURISDICTIONS]
    assert len(codes) == len(set(codes)), f"duplicate codes: {codes}"


def test_registry_matches_class_list() -> None:
    """ALL_JURISDICTIONS dict in the package must contain every class in
    ALL_JURISDICTIONS list and nothing else (so config-driven instantiation
    by ISO code stays in sync with the importable classes)."""
    by_code = {cls().code: cls for cls in ALL_JURISDICTIONS}
    assert set(by_code.keys()) == set(JURISDICTION_REGISTRY.keys())
    for code, cls in by_code.items():
        assert JURISDICTION_REGISTRY[code] is cls


@pytest.mark.parametrize("cls", ALL_JURISDICTIONS)
def test_every_jurisdiction_template_satisfies_its_own_validate_notice(cls) -> None:
    """A jurisdiction's own disclosure block must clear its own
    validate_notice() check. If this fails, the template has drifted from
    the mandatory_disclosures list."""
    j = cls()
    issues = j.validate_notice(j.get_disclosure_block())
    assert issues == [], (
        f"{j.code}: template missing mandatory disclosure substrings: {issues}"
    )


@pytest.mark.parametrize("cls", ALL_JURISDICTIONS)
def test_every_jurisdiction_supersets_us_keywords_or_documents_why(cls) -> None:
    """All non-US jurisdictions in this set are supersets of the US keyword
    list (we always start from US_PROTECTED_KEYWORDS and add region-specific
    terms — see the per-module docstrings)."""
    if cls is US:
        return
    us = set(US_PROTECTED_KEYWORDS)
    j = set(cls().get_protected_keywords())
    missing = us - j
    assert not missing, f"{cls.__name__} should superset US keywords; missing: {missing}"


@pytest.mark.parametrize("cls", ALL_JURISDICTIONS)
def test_every_jurisdiction_appends_its_own_block_on_denial(cls) -> None:
    """End-to-end: configure the pipeline with each jurisdiction and
    confirm the resulting adverse-action notice ends with that
    jurisdiction's disclosure block."""
    j = cls()
    wf = _build_wf(default_prob=0.85, jurisdiction=j)
    body = "Application not approved due to repayment history."
    with patch.object(wf, "_call_llm", return_value=body), patch.object(
        wf, "_compute_shap_factors",
        return_value=_shap_factors(["PAY_0", "BILL_AMT1", "LIMIT_BAL"]),
    ):
        result = wf.process_application(_sample_applicant())
    assert result.decision == "DENY"
    assert result.adverse_action is not None
    assert result.adverse_action.endswith(j.get_disclosure_block())
    assert result.metadata["jurisdiction"] == j.code
