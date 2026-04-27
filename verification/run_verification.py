"""End-to-end verification runner.

Runs 500 stratified UCI profiles through CreditWorkflow with every
production component active (audit, PII, rate limiter, escalation).

Usage:
    cd credit-risk-ai-workflow
    PHASE6_WORKERS=8 python verification/run_verification.py

Outputs:
    verification/pipeline_results_500.csv
    verification/audit_trail.jsonl  (one record per profile)
    verification/escalation_queue.jsonl
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# Load .env from repo root
ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent
for env_file in (ROOT / ".env", PARENT / ".env"):
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
        break

sys.path.insert(0, str(ROOT))

from workflow import (  # noqa: E402
    AuditLogger,
    CreditWorkflow,
    EscalationRouter,
    JSONLBackend,
    PIIScrubber,
    QueueBackend,
    RateLimiter,
)

VERIFICATION_DIR = ROOT / "verification"
PROFILES_CSV = VERIFICATION_DIR / "test_profiles_500.csv"
RESULTS_CSV = VERIFICATION_DIR / "pipeline_results_500.csv"
AUDIT_DIR = VERIFICATION_DIR / "audit_logs"
ESCALATION_DIR = VERIFICATION_DIR / "escalations"
MODEL_PATH = ROOT / "data" / "processed" / "xgboost_model.joblib"

LLM_MODEL = os.environ.get("VERIFICATION_LLM_MODEL", "gpt-4o-mini")
MAX_WORKERS = int(os.environ.get("VERIFICATION_WORKERS", "8"))
MOCK_LLM = os.environ.get("VERIFICATION_MOCK_LLM", "0") == "1"

# Pricing (USD per 1M tokens)
PRICES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.5, "output": 10.0},
}


def _build_workflow() -> CreditWorkflow:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ESCALATION_DIR.mkdir(parents=True, exist_ok=True)
    audit = AuditLogger(JSONLBackend(AUDIT_DIR))
    pii = PIIScrubber()
    limiter = RateLimiter(
        rpm=400,                   # gpt-4o-mini tier limits are generous
        tpm=200_000,
        prices=PRICES,
        hard_cap_usd=15.0,
        failure_threshold=10,
    )
    router = EscalationRouter(
        QueueBackend(ESCALATION_DIR),
        triggers={
            "BORDERLINE": "MEDIUM",
            "PROTECTED_ATTR_DETECTED": "HIGH",
            "HIGH_VALUE_LOAN": "HIGH",
        },
        slas={"HIGH": 4, "MEDIUM": 24, "LOW": 72},
    )
    wf = CreditWorkflow(
        model_path=str(MODEL_PATH),
        llm_provider="openai",
        llm_model=LLM_MODEL,
        uncertainty_threshold=0.15,
        model_version="1.0.0",
        audit_logger=audit,
        pii_scrubber=pii,
        rate_limiter=limiter,
        escalation_router=router,
    )
    if MOCK_LLM:
        # Deterministic mock; still exercises every safety/audit/escalation
        # layer because they don't care where the LLM text comes from.
        def _mock(self, prompt: str) -> str:  # noqa: ARG001
            if "adverse action" in prompt.lower() or "denial" in prompt.lower():
                return ("Your application was not approved due to elevated "
                        "repayment delay history and high credit utilisation "
                        "relative to your assigned limit. You may request "
                        "reconsideration by submitting updated documentation.")
            return ("CREDIT MEMO\n\nExecutive Summary: Based on repayment history "
                    "and exposure metrics, the model recommends the decision shown.\n\n"
                    "Risk Assessment: Repayment status (PAY_0) and credit limit "
                    "(LIMIT_BAL) are the dominant factors per SHAP attribution.\n\n"
                    "Recommendation: see decision header.")
        CreditWorkflow._call_llm = _mock  # type: ignore[method-assign]
    return wf


def main() -> int:
    if not PROFILES_CSV.exists():
        print(f"missing {PROFILES_CSV}", file=sys.stderr)
        return 1
    profiles = pd.read_csv(PROFILES_CSV)
    print(f"loaded {len(profiles)} profiles ({int(profiles['actual_default'].sum())} defaults)")
    print(f"workers={MAX_WORKERS} mock_llm={MOCK_LLM} model={LLM_MODEL}")

    feature_cols = [c for c in profiles.columns if c != "actual_default"]
    wf = _build_workflow()
    results: list[dict] = []
    res_lock = threading.Lock()

    def _process(idx: int, row: pd.Series) -> dict:
        applicant = {c: row[c] for c in feature_cols}
        # Cast numpy types to native to avoid JSON-serialisation friction
        applicant = {k: (int(v) if hasattr(v, "item") and "int" in str(type(v)) else
                          float(v) if hasattr(v, "item") else v)
                     for k, v in applicant.items()}
        try:
            r = wf.process_application(applicant)
            return {
                "profile_idx": idx,
                "actual_default": int(row["actual_default"]),
                "decision": r.decision,
                "probability": r.probability,
                "flags": "|".join(r.flags),
                "escalated": bool(r.metadata.get("escalated")),
                "shap_top_feature": r.shap_factors[0]["feature"] if r.shap_factors else "N/A",
                "memo_length": len(r.memo) if r.memo else 0,
                "adverse_action_length": len(r.adverse_action) if r.adverse_action else 0,
                "protected_attr_hits": len(r.metadata.get("protected_attribute_hits", [])),
                "decision_id": r.metadata.get("decision_id", ""),
                "processing_time_ms": r.metadata.get("processing_time_ms", 0.0),
                "scrubbed_fields_count": len(r.metadata.get("scrubbed_fields", [])),
                "error": "",
            }
        except Exception as e:  # noqa: BLE001
            return {
                "profile_idx": idx,
                "actual_default": int(row["actual_default"]),
                "decision": "ERROR",
                "probability": -1.0,
                "flags": "",
                "escalated": False,
                "shap_top_feature": "N/A",
                "memo_length": 0,
                "adverse_action_length": 0,
                "protected_attr_hits": 0,
                "decision_id": "",
                "processing_time_ms": 0.0,
                "scrubbed_fields_count": 0,
                "error": f"{type(e).__name__}: {e}",
            }

    t0 = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_process, i, row): i for i, row in profiles.iterrows()}
        for f in as_completed(futures):
            res = f.result()
            with res_lock:
                results.append(res)
                completed += 1
                if completed % 50 == 0:
                    print(f"  {completed}/{len(profiles)} done "
                          f"({time.time()-t0:.1f}s, errors={sum(1 for r in results if r['decision']=='ERROR')})",
                          flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["profile_idx"])
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"\nwrote {RESULTS_CSV} in {elapsed:.1f}s")
    n_err = sum(1 for r in results if r["decision"] == "ERROR")
    print(f"errors: {n_err}/{len(results)}")
    if MOCK_LLM:
        print("(mock LLM mode — no real API calls)")
    elif wf.rate_limiter is not None:
        spend = wf.rate_limiter.cost.total_spend if wf.rate_limiter.cost else 0
        print(f"LLM spend: ${spend:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
