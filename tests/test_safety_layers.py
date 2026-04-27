"""Integration tests for CreditWorkflow safety layers.

Verifies the four safety mechanisms in workflow.pipeline.CreditWorkflow:
  1. FCRA disclosure injection (deterministic, word-for-word)
  2. Protected-attribute keyword filter (and absence of false positives)
  3. SHAP grounding — prompt only contains computed top-k factors
  4. BORDERLINE uncertainty flag near the decision threshold

All tests mock the LLM client and joblib.load so they run with no API key
and without a serialized model on disk.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# Set dummy API key BEFORE importing the workflow module — the OpenAI client
# is constructed in CreditWorkflow.__init__ and reads OPENAI_API_KEY from env.
os.environ.setdefault("OPENAI_API_KEY", "test")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from workflow.pipeline import CreditWorkflow, WorkflowResult  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_applicant() -> Dict[str, Any]:
    """A minimal UCI-shaped borrower profile."""
    return {
        "LIMIT_BAL": 50000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 2,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 12000,
        "BILL_AMT2": 11500,
        "BILL_AMT3": 11000,
        "BILL_AMT4": 10500,
        "BILL_AMT5": 10000,
        "BILL_AMT6": 9500,
        "PAY_AMT1": 500,
        "PAY_AMT2": 500,
        "PAY_AMT3": 500,
        "PAY_AMT4": 500,
        "PAY_AMT5": 500,
        "PAY_AMT6": 500,
    }


def _make_mock_model(default_prob: float) -> MagicMock:
    """Build a mock sklearn pipeline whose predict_proba returns [1-p, p]."""
    mock_model = MagicMock()
    # predict_proba returns shape (n, 2): [P(class=0), P(class=1)]
    mock_model.predict_proba.return_value = np.array(
        [[1.0 - default_prob, default_prob]]
    )
    # The pipeline indexes self.model[-1] (final estimator) and self.model[0]
    # (preprocessor); make those work without exploding.
    final_estimator = MagicMock()
    final_estimator.__class__.__name__ = "MockClassifier"
    mock_model.__getitem__.return_value = final_estimator
    # named_steps presence drives the SHAP preprocessing branch — leave absent
    # so _compute_shap_factors falls into the simpler X_transformed = df path.
    del mock_model.named_steps
    return mock_model


def _shap_factors(features: List[str]) -> List[Dict[str, Any]]:
    """Build a deterministic top-k SHAP factor list."""
    return [
        {
            "feature": feat,
            "shap_value": 0.5 - 0.1 * i,
            "direction": "increases risk" if i % 2 == 0 else "decreases risk",
        }
        for i, feat in enumerate(features)
    ]


@pytest.fixture
def workflow_factory():
    """Factory that builds a CreditWorkflow with a mocked model and LLM client.

    Returns a callable: build(default_prob: float) -> CreditWorkflow.
    """

    def build(default_prob: float = 0.5) -> CreditWorkflow:
        mock_model = _make_mock_model(default_prob)
        with patch("workflow.pipeline.joblib.load", return_value=mock_model), patch(
            "workflow.pipeline.OpenAI"
        ) as mock_openai_cls:
            mock_openai_cls.return_value = MagicMock()
            wf = CreditWorkflow(
                model_path="unused.joblib",
                llm_provider="openai",
                llm_model="gpt-4o-mock",
            )
        return wf

    return build


# ---------------------------------------------------------------------------
# 1. FCRA injection — word-for-word
# ---------------------------------------------------------------------------


def test_fcra_disclosure_injected_verbatim_on_denial(workflow_factory) -> None:
    wf = workflow_factory(default_prob=0.85)  # forces DENY

    denial_body = "Your application was denied because of low PAY_0 history."
    with patch.object(wf, "_call_llm", return_value=denial_body), patch.object(
        wf,
        "_compute_shap_factors",
        return_value=_shap_factors(["PAY_0", "BILL_AMT1", "LIMIT_BAL"]),
    ):
        result = wf.process_application(_sample_applicant())

    assert result.decision == "DENY"
    assert result.adverse_action is not None
    # Word-for-word checks against CreditWorkflow.FCRA_DISCLOSURE
    assert "consumer reporting agency" in result.adverse_action
    assert "right to dispute" in result.adverse_action.lower()
    # FCRA heading appears in upper-case in the disclosure constant
    assert "FAIR CREDIT REPORTING ACT" in result.adverse_action
    # And the LLM body itself must still be present (FCRA is appended).
    assert denial_body in result.adverse_action
    # The injected text must equal the class constant verbatim (deterministic).
    assert result.adverse_action.endswith(CreditWorkflow.FCRA_DISCLOSURE)


# ---------------------------------------------------------------------------
# 2. Protected-attribute filter fires
# ---------------------------------------------------------------------------


def test_protected_attribute_filter_flags_offending_memo(workflow_factory) -> None:
    wf = workflow_factory(default_prob=0.20)  # APPROVE — only the memo path runs

    bad_memo = "the borrower is a young woman with stable employment"
    with patch.object(wf, "_call_llm", return_value=bad_memo), patch.object(
        wf,
        "_compute_shap_factors",
        return_value=_shap_factors(["PAY_0", "BILL_AMT1", "LIMIT_BAL"]),
    ):
        result = wf.process_application(_sample_applicant())

    assert "PROTECTED_ATTR_DETECTED" in result.flags
    hits = result.metadata["protected_attribute_hits"]
    assert "woman" in hits
    assert "young" in hits


# ---------------------------------------------------------------------------
# 3. Protected filter — no false positives on a clean memo
# ---------------------------------------------------------------------------


def test_protected_attribute_filter_clean_memo(workflow_factory) -> None:
    wf = workflow_factory(default_prob=0.20)  # APPROVE

    clean_memo = (
        "Borrower has consistent payment history and conservative credit utilization."
    )
    with patch.object(wf, "_call_llm", return_value=clean_memo), patch.object(
        wf,
        "_compute_shap_factors",
        return_value=_shap_factors(["PAY_0", "BILL_AMT1", "LIMIT_BAL"]),
    ):
        result = wf.process_application(_sample_applicant())

    assert "PROTECTED_ATTR_DETECTED" not in result.flags
    assert result.metadata["protected_attribute_hits"] == []


# ---------------------------------------------------------------------------
# 4. SHAP grounding — prompt contains only computed factors
# ---------------------------------------------------------------------------


def test_prompt_contains_only_shap_top_factors(workflow_factory) -> None:
    wf = workflow_factory(default_prob=0.20)  # APPROVE — single LLM call (memo only)

    top_factors = ["PAY_0", "BILL_AMT1", "LIMIT_BAL"]
    captured: Dict[str, str] = {}

    def fake_call_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return "Borrower has consistent payment history and low utilization."

    with patch.object(wf, "_call_llm", side_effect=fake_call_llm), patch.object(
        wf, "_compute_shap_factors", return_value=_shap_factors(top_factors)
    ):
        wf.process_application(_sample_applicant())

    prompt = captured["prompt"]
    # The TOP RISK FACTORS section must list each computed factor.
    assert "TOP RISK FACTORS" in prompt
    for feat in top_factors:
        assert feat in prompt, f"expected SHAP factor {feat!r} in prompt"

    # Slice out just the factors block (between the heading and the RULES section)
    factors_section = prompt.split("TOP RISK FACTORS", 1)[1].split("RULES", 1)[0]
    # Features NOT in the SHAP top-5 must not appear in the factors section.
    for missing in ("MARRIAGE", "EDUCATION", "PAY_4", "BILL_AMT6"):
        assert missing not in factors_section, (
            f"non-top-5 feature {missing!r} leaked into the SHAP factors block"
        )


# ---------------------------------------------------------------------------
# 5. BORDERLINE fires near threshold
# ---------------------------------------------------------------------------


def test_borderline_flag_fires_near_threshold(workflow_factory) -> None:
    wf = workflow_factory(default_prob=0.48)  # |0.48 - 0.5| = 0.02 < 0.35

    with patch.object(wf, "_call_llm", return_value="Memo body."), patch.object(
        wf,
        "_compute_shap_factors",
        return_value=_shap_factors(["PAY_0", "BILL_AMT1", "LIMIT_BAL"]),
    ):
        result = wf.process_application(_sample_applicant())

    assert "BORDERLINE" in result.flags


# ---------------------------------------------------------------------------
# 6. BORDERLINE does NOT fire when far from threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prob", [0.10, 0.85])
def test_borderline_flag_absent_when_confident(workflow_factory, prob: float) -> None:
    # uncertainty_threshold defaults to 0.35; |0.10-0.5|=0.40 and |0.85-0.5|=0.35
    # both fall outside the strict-less-than band, so no BORDERLINE.
    wf = workflow_factory(default_prob=prob)

    with patch.object(wf, "_call_llm", return_value="Memo body."), patch.object(
        wf,
        "_compute_shap_factors",
        return_value=_shap_factors(["PAY_0", "BILL_AMT1", "LIMIT_BAL"]),
    ):
        result = wf.process_application(_sample_applicant())

    assert "BORDERLINE" not in result.flags


# ---------------------------------------------------------------------------
# 7. Adverse-action notice is absent on APPROVE
# ---------------------------------------------------------------------------


def test_no_adverse_action_on_approve(workflow_factory) -> None:
    wf = workflow_factory(default_prob=0.20)  # APPROVE

    with patch.object(wf, "_call_llm", return_value="Memo body."), patch.object(
        wf,
        "_compute_shap_factors",
        return_value=_shap_factors(["PAY_0", "BILL_AMT1", "LIMIT_BAL"]),
    ):
        result = wf.process_application(_sample_applicant())

    assert isinstance(result, WorkflowResult)
    assert result.decision == "APPROVE"
    assert result.adverse_action is None
