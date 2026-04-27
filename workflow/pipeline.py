"""Core workflow pipeline — orchestrates ML prediction, LLM communication, and safety checks."""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import shap

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

if TYPE_CHECKING:
    from .audit import AuditLogger
    from .escalation import EscalationRouter
    from .jurisdictions.base import JurisdictionBase
    from .pii import PIIScrubber
    from .ratelimit import RateLimiter


@dataclass
class WorkflowResult:
    """Result of processing a credit application through the workflow."""

    decision: str  # "APPROVE" or "DENY"
    probability: float  # Default probability from classical model
    memo: str  # Credit memo (SHAP-grounded)
    adverse_action: Optional[str]  # Denial letter (if denied), FCRA-compliant
    shap_factors: List[Dict[str, Any]]  # Top SHAP contributors
    flags: List[str] = field(default_factory=list)  # e.g., ["BORDERLINE", "MANUAL_REVIEW"]
    metadata: Dict[str, Any] = field(default_factory=dict)


class CreditWorkflow:
    """Six-phase credit underwriting workflow.

    Separates concerns:
      - Classical ML handles prediction (high accuracy, auditable)
      - LLM handles communication (natural language, professional)
      - Deterministic code handles compliance (FCRA, bias, uncertainty)

    Parameters
    ----------
    model_path : str or Path
        Path to a joblib-serialized sklearn/xgboost pipeline.
    llm_provider : str
        "openai" or "anthropic".
    llm_model : str
        Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514").
    uncertainty_threshold : float
        Probability range [0.5 - threshold, 0.5 + threshold] triggers
        BORDERLINE flag and human review routing. Default: 0.35.
    protected_attributes : list
        Attribute names to filter from LLM output. Default: standard set.
    """

    # Legacy US-only defaults retained as class attributes for backwards
    # compatibility with code that referenced them directly. Runtime behaviour
    # now flows through self.jurisdiction (defaults to US()), which sources
    # the same values from workflow/jurisdictions/us.py and
    # workflow/jurisdictions/templates/us_adverse_action.txt. These constants
    # MUST stay byte-identical to those sources — see tests/test_jurisdictions.py.
    PROTECTED_KEYWORDS = [
        "race", "racial", "ethnicity", "ethnic", "skin color",
        "gender", "sex", "male", "female", "woman", "man",
        "age", "old", "young", "elderly", "senior",
        "religion", "religious", "muslim", "christian", "jewish", "hindu",
        "national origin", "immigrant", "foreign",
        "disability", "disabled", "handicap",
        "marital status", "married", "single", "divorced",
        "pregnant", "pregnancy",
    ]

    FCRA_DISCLOSURE = (
        "\n\n---\n"
        "IMPORTANT NOTICES UNDER THE FAIR CREDIT REPORTING ACT (FCRA):\n\n"
        "1. You have the right to obtain a free copy of your credit report from the "
        "consumer reporting agency that supplied the report used in this decision.\n"
        "2. You have the right to dispute the accuracy or completeness of any "
        "information in your credit report.\n"
        "3. The consumer reporting agency did not make the adverse decision and is "
        "unable to explain why the decision was made.\n"
        "4. You may contact the credit reporting agency at:\n"
        "   [AGENCY NAME, ADDRESS, PHONE — to be filled by institution]\n"
    )

    def __init__(
        self,
        model_path: str,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o",
        uncertainty_threshold: float = 0.35,
        protected_attributes: Optional[List[str]] = None,
        *,
        model_version: str = "unversioned",
        audit_logger: Optional["AuditLogger"] = None,
        pii_scrubber: Optional["PIIScrubber"] = None,
        rate_limiter: Optional["RateLimiter"] = None,
        escalation_router: Optional["EscalationRouter"] = None,
        jurisdiction: Optional["JurisdictionBase"] = None,
    ):
        # Lazy import to avoid a circular dependency through workflow.__init__.
        if jurisdiction is None:
            from .jurisdictions.us import US
            jurisdiction = US()
        self.jurisdiction = jurisdiction

        self.model = joblib.load(model_path)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.uncertainty_threshold = uncertainty_threshold
        # Precedence: explicit override > jurisdiction default > legacy class const
        self.protected_keywords = (
            protected_attributes
            or self.jurisdiction.get_protected_keywords()
        )
        # Pre-compile word-bounded keyword patterns.
        self._protected_patterns = [
            (kw, re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE))
            for kw in self.protected_keywords
        ]
        self.model_version = model_version
        self.audit_logger = audit_logger
        self.pii_scrubber = pii_scrubber
        self.rate_limiter = rate_limiter
        self.escalation_router = escalation_router

        # Initialize LLM client
        if llm_provider == "openai":
            if OpenAI is None:
                raise ImportError("pip install openai")
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif llm_provider == "anthropic":
            if Anthropic is None:
                raise ImportError("pip install anthropic")
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {llm_provider}")

        # SHAP explainer (lazy-loaded)
        self._explainer = None

    def _get_explainer(self, X_background: Optional[pd.DataFrame] = None):
        """Get or create SHAP explainer."""
        if self._explainer is None:
            try:
                self._explainer = shap.TreeExplainer(self.model[-1])
            except Exception:
                if X_background is None:
                    raise ValueError(
                        "Model is not tree-based. Provide X_background for KernelExplainer."
                    )
                self._explainer = shap.KernelExplainer(
                    self.model.predict_proba, X_background.iloc[:100]
                )
        return self._explainer

    def process_application(
        self,
        applicant_data: Dict[str, Any],
        X_background: Optional[pd.DataFrame] = None,
        approval_threshold: float = 0.5,
        loan_amount: Optional[float] = None,
    ) -> WorkflowResult:
        t_start = time.perf_counter()
        flags: List[str] = []
        decision_id = str(uuid.uuid4())

        # PII scrub
        if self.pii_scrubber is not None:
            llm_features, scrubbed_fields = self.pii_scrubber.scrub(applicant_data)
        else:
            llm_features, scrubbed_fields = dict(applicant_data), []

        # Default prediction
        df = pd.DataFrame([applicant_data])
        prob = float(self.model.predict_proba(df)[:, 1][0])
        decision = "DENY" if prob >= approval_threshold else "APPROVE"

        if abs(prob - 0.5) < self.uncertainty_threshold:
            flags.append("BORDERLINE")

        shap_factors = self._compute_shap_factors(df, X_background)

        memo_prompt = self._build_memo_prompt(llm_features, decision, prob, shap_factors)
        memo = self._call_llm(memo_prompt)

        adverse_action: Optional[str] = None
        adverse_prompt: Optional[str] = None
        disclosure_block = self.jurisdiction.get_disclosure_block()
        if decision == "DENY":
            adverse_prompt = self._build_adverse_action_prompt(llm_features, prob, shap_factors)
            adverse_body = self._call_llm(adverse_prompt)
            adverse_action = adverse_body + disclosure_block

        # Protected-attribute filter
        protected_hits = self._check_protected_attributes(memo)
        if decision == "DENY":
            adverse_body_for_check = (
                adverse_action[: len(adverse_action) - len(disclosure_block)]
                if adverse_action else ""
            )
            protected_hits += self._check_protected_attributes(adverse_body_for_check)
        if protected_hits:
            flags.append("PROTECTED_ATTR_DETECTED")

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Escalation routing
        escalated = False
        escalation_reason: Optional[str] = None
        if self.escalation_router is not None:
            trigger = self.escalation_router.should_escalate(
                flags=flags,
                probability=prob,
                loan_amount=loan_amount,
            )
            if trigger:
                escalated = True
                escalation_reason = trigger
                self.escalation_router.route(
                    decision_id=decision_id,
                    reason=trigger,
                    flags=flags,
                    context={
                        "probability": prob,
                        "decision": decision,
                        "shap_factors": shap_factors,
                        "loan_amount": loan_amount,
                    },
                )

        # Audit log
        if self.audit_logger is not None:
            from .audit import AuditRecord
            self.audit_logger.log(AuditRecord.new(
                decision_id=decision_id,
                model_version=self.model_version,
                applicant_features=llm_features,
                probability=prob,
                decision=decision,
                shap_factors=shap_factors,
                llm_prompt=memo_prompt + ("\n---ADVERSE---\n" + adverse_prompt if adverse_prompt else ""),
                llm_response=memo + ("\n---ADVERSE---\n" + (adverse_action or "") if decision == "DENY" else ""),
                safety_results={
                    "flags": flags,
                    "protected_attribute_hits": protected_hits,
                    "fcra_injected": decision == "DENY",
                    "escalated": escalated,
                    "escalation_reason": escalation_reason,
                },
                final_output={"memo": memo, "adverse_action": adverse_action},
                processing_time_ms=elapsed_ms,
                scrubbed_fields=scrubbed_fields,
            ))

        return WorkflowResult(
            decision=decision,
            probability=prob,
            memo=memo,
            adverse_action=adverse_action,
            shap_factors=shap_factors,
            flags=flags,
            metadata={
                "decision_id": decision_id,
                "model": str(type(self.model[-1]).__name__),
                "model_version": self.model_version,
                "llm_model": self.llm_model,
                "jurisdiction": self.jurisdiction.code,
                "protected_attribute_hits": protected_hits,
                "scrubbed_fields": scrubbed_fields,
                "processing_time_ms": elapsed_ms,
                "escalated": escalated,
                "escalation_reason": escalation_reason,
            },
        )

    def _compute_shap_factors(
        self, df: pd.DataFrame, X_background: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """Compute top SHAP factors for the prediction."""
        try:
            explainer = self._get_explainer(X_background)
            if hasattr(self.model, "named_steps"):
                preprocessor = self.model[:-1]
                X_transformed = preprocessor.transform(df)
            else:
                X_transformed = df

            shap_values = explainer.shap_values(X_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            if hasattr(self.model[0], "get_feature_names_out"):
                feature_names = list(self.model[0].get_feature_names_out())
            else:
                feature_names = list(df.columns)

            abs_vals = np.abs(shap_values[0])
            top_idx = np.argsort(abs_vals)[-5:][::-1]

            return [
                {
                    "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                    "shap_value": float(shap_values[0][i]),
                    "direction": "increases risk" if shap_values[0][i] > 0 else "decreases risk",
                }
                for i in top_idx
            ]
        except Exception as e:
            return [{"feature": "SHAP_UNAVAILABLE", "error": str(e)}]

    def _generate_memo(self, applicant, decision, prob, shap_factors) -> str:
        prompt = self._build_memo_prompt(applicant, decision, prob, shap_factors)
        return self._call_llm(prompt)

    def _generate_adverse_action(self, applicant, prob, shap_factors) -> str:
        prompt = self._build_adverse_action_prompt(applicant, prob, shap_factors)
        notice = self._call_llm(prompt)
        notice += self.jurisdiction.get_disclosure_block()
        return notice

    def _build_memo_prompt(self, applicant, decision, prob, shap_factors) -> str:
        factors_text = "\n".join(
            f"  - {f['feature']}: SHAP={f.get('shap_value', 'N/A'):.4f} ({f.get('direction', '')})"
            for f in shap_factors
            if "error" not in f
        )
        return f"""You are a credit analyst drafting a credit risk assessment memo.

DECISION: {decision}
DEFAULT PROBABILITY: {prob:.3f}

BORROWER PROFILE:
{json.dumps(applicant, indent=2)}

TOP RISK FACTORS (from SHAP analysis — you MUST cite only these):
{factors_text}

RULES:
1. You may ONLY cite factors listed above. Do not invent additional reasons.
2. Do NOT mention race, gender, age, religion, national origin, or any protected attribute.
3. Structure: Executive Summary → Borrower Profile → Risk Assessment → Recommendation.
4. Be specific — cite actual values from the profile, not generic statements.
5. Keep to 200-300 words.

Write the credit memo now:"""

    def _build_adverse_action_prompt(self, applicant, prob, shap_factors) -> str:
        factors_text = "\n".join(
            f"  {i+1}. {f['feature']} ({f.get('direction', '')})"
            for i, f in enumerate(shap_factors[:4])
            if "error" not in f
        )
        return f"""You are drafting an adverse action notice (denial letter) for a credit application.

DEFAULT PROBABILITY: {prob:.3f}
PRIMARY REASONS FOR DENIAL (from model analysis):
{factors_text}

RULES:
1. List the top 4 reasons for denial using ONLY the factors above.
2. Use plain language a consumer can understand — no jargon.
3. Do NOT mention race, gender, age, religion, national origin, or any protected class.
4. Do NOT include legal disclosures — those are added separately by the system.
5. Be respectful and professional.
6. Keep to 150 words maximum.

Write the adverse action explanation now:"""

    def _check_protected_attributes(self, text: str) -> List[str]:
        """Return protected-attribute keywords appearing as whole words in text."""
        hits: List[str] = []
        for kw, pat in self._protected_patterns:
            if pat.search(text):
                hits.append(kw)
        return hits

    def _do_llm_call(self, prompt: str) -> tuple[str, int, int]:
        """Make the actual provider call and return (text, in_tokens, out_tokens)."""
        if self.llm_provider == "openai":
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
            )
            text = response.choices[0].message.content or ""
            usage = response.usage
            in_t = getattr(usage, "prompt_tokens", 0)
            out_t = getattr(usage, "completion_tokens", 0)
            return text, in_t, out_t
        if self.llm_provider == "anthropic":
            response = self.client.messages.create(
                model=self.llm_model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            usage = response.usage
            in_t = getattr(usage, "input_tokens", 0)
            out_t = getattr(usage, "output_tokens", 0)
            return text, in_t, out_t
        raise ValueError(f"Unsupported provider: {self.llm_provider}")

    def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM. Optionally routed through the rate limiter."""
        if self.rate_limiter is None:
            text, _in_t, _out_t = self._do_llm_call(prompt)
            return text

        est_in = max(1, len(prompt) // 4)
        self.rate_limiter.acquire(self.llm_model, est_in)
        try:
            text, in_t, out_t = self.rate_limiter.with_retry(
                self._do_llm_call, prompt
            )
            self.rate_limiter.record_call(self.llm_model, in_t, out_t, success=True)
            return text
        except Exception:
            self.rate_limiter.record_call(self.llm_model, 0, 0, success=False)
            raise
