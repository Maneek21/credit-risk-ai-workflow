"""Compute end-to-end verification metrics from the 500-profile run.

Reads:
    verification/pipeline_results_500.csv
    verification/audit_logs/*.jsonl
    verification/escalations/*.jsonl

Writes:
    verification/verification_summary.json

Prints a human-readable report to stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
VERIFICATION_DIR = ROOT / "verification"
RESULTS_CSV = VERIFICATION_DIR / "pipeline_results_500.csv"
AUDIT_DIR = VERIFICATION_DIR / "audit_logs"
ESCALATION_DIR = VERIFICATION_DIR / "escalations"
SUMMARY_JSON = VERIFICATION_DIR / "verification_summary.json"


def _read_jsonl_dir(d: Path) -> list[dict]:
    out: list[dict] = []
    if not d.exists():
        return out
    for p in sorted(d.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def main() -> int:
    if not RESULTS_CSV.exists():
        print(f"missing {RESULTS_CSV}")
        return 1

    results = pd.read_csv(RESULTS_CSV)

    print("=" * 70)
    print("CREDIT-RISK-AI-WORKFLOW — END-TO-END VERIFICATION REPORT")
    print("=" * 70)

    # 1. Pipeline health
    errors = results[results["decision"] == "ERROR"]
    n_total = len(results)
    n_ok = n_total - len(errors)
    success_rate = n_ok / n_total if n_total else 0.0
    print("\n1. PIPELINE HEALTH")
    print(f"   Profiles processed:    {n_total}")
    print(f"   Successful:            {n_ok}")
    print(f"   Errors:                {len(errors)}")
    print(f"   Success rate:          {success_rate * 100:.1f}%")
    if len(errors):
        print("   Sample error messages:")
        for msg in errors["error"].dropna().unique()[:3]:
            print(f"     - {msg[:140]}")

    # 2. Decision accuracy
    valid = results[results["decision"] != "ERROR"].copy()
    valid["predicted_default"] = (valid["decision"] == "DENY").astype(int)
    auc = roc_auc_score(valid["actual_default"], valid["probability"])
    acc = accuracy_score(valid["actual_default"], valid["predicted_default"])
    tn, fp, fn, tp = confusion_matrix(
        valid["actual_default"], valid["predicted_default"]
    ).ravel()
    print("\n2. DECISION ACCURACY (vs actual defaults)")
    print(f"   AUC (probability):     {auc:.4f}")
    print(f"   Accuracy (decision):   {acc:.4f}")
    print(f"   True Positives:        {tp} (correctly denied defaulters)")
    print(f"   True Negatives:        {tn} (correctly approved non-defaulters)")
    print(f"   False Positives:       {fp} (denied non-defaulters)")
    print(f"   False Negatives:       {fn} (approved defaulters)")
    print(f"   Deny rate:             {valid['predicted_default'].mean() * 100:.1f}%")
    print(f"   Actual default rate:   {valid['actual_default'].mean() * 100:.1f}%")

    # 3. Safety layer activations
    print("\n3. SAFETY LAYER ACTIVATIONS")
    borderline = int(valid["flags"].str.contains("BORDERLINE", na=False).sum())
    protected_flagged = int(
        valid["flags"].str.contains("PROTECTED_ATTR_DETECTED", na=False).sum()
    )
    protected_keyword_hits = int(valid["protected_attr_hits"].sum())
    escalated = int(valid["escalated"].sum())
    scrubbed_total = int(valid.get("scrubbed_fields_count", pd.Series([0])).sum())
    print(f"   Borderline flags:        {borderline}")
    print(f"   Protected-attr profiles: {protected_flagged}")
    print(f"   Protected-attr keyword hits (sum across profiles): {protected_keyword_hits}")
    print(f"   Escalations triggered:   {escalated} ({escalated/len(valid)*100:.1f}%)")
    print(f"   PII fields scrubbed:     {scrubbed_total} (across all profiles)")

    # 4. FCRA compliance
    denials = valid[valid["decision"] == "DENY"]
    has_adverse = int((denials["adverse_action_length"] > 0).sum())
    fcra_coverage = has_adverse / len(denials) if len(denials) else 1.0
    print("\n4. FCRA COMPLIANCE")
    print(f"   Denials issued:        {len(denials)}")
    print(f"   With adverse notice:   {has_adverse}")
    if len(denials):
        print(f"   FCRA coverage:         {fcra_coverage * 100:.1f}%")
    else:
        print("   FCRA coverage:         N/A (no denials)")

    # 5. Memo generation
    print("\n5. MEMO GENERATION")
    has_memo = int((valid["memo_length"] > 0).sum())
    print(f"   Memos generated:       {has_memo}")
    print(f"   Avg memo length:       {valid['memo_length'].mean():.0f} chars")
    print(f"   Min/Max:               {valid['memo_length'].min()} / {valid['memo_length'].max()} chars")

    # 6. Audit trail
    audit_records = _read_jsonl_dir(AUDIT_DIR)
    print("\n6. AUDIT TRAIL")
    print(f"   Records written:       {len(audit_records)}")
    print(f"   Records expected:      {len(valid)}")
    audit_coverage = len(audit_records) / len(valid) if len(valid) else 0.0
    print(f"   Coverage:              {audit_coverage * 100:.1f}%")
    audit_fields_ok = 0
    audit_fields_required = 0
    if audit_records:
        sample = audit_records[0]
        required = [
            "timestamp_utc",
            "decision_id",
            "decision",
            "probability",
            "shap_factors",
            "model_version",
            "applicant_features",
            "llm_prompt",
            "llm_response",
            "safety_results",
            "final_output",
            "processing_time_ms",
        ]
        present = [f for f in required if f in sample]
        audit_fields_ok = len(present)
        audit_fields_required = len(required)
        print(f"   Fields per record:     {len(sample.keys())}")
        print(f"   Required fields OK:    {audit_fields_ok}/{audit_fields_required}")
        # Verify all unique decision_ids
        unique_ids = len({r.get("decision_id") for r in audit_records})
        print(f"   Unique decision_ids:   {unique_ids}")

    # 7. Escalation queue
    esc_records = _read_jsonl_dir(ESCALATION_DIR)
    print("\n7. ESCALATION QUEUE")
    print(f"   Cases escalated:       {len(esc_records)}")
    print(f"   Expected (from flags): {escalated}")
    if esc_records:
        priorities: dict[str, int] = {}
        for r in esc_records:
            p = r.get("priority", "UNKNOWN")
            priorities[p] = priorities.get(p, 0) + 1
        for p, n in sorted(priorities.items()):
            print(f"     {p}: {n}")

    # 8. Performance
    print("\n8. PERFORMANCE")
    avg_time = float(valid["processing_time_ms"].mean())
    p95_time = float(valid["processing_time_ms"].quantile(0.95))
    p99_time = float(valid["processing_time_ms"].quantile(0.99))
    print(f"   Avg processing time:   {avg_time:.0f} ms")
    print(f"   P95 processing time:   {p95_time:.0f} ms")
    print(f"   P99 processing time:   {p99_time:.0f} ms")

    # 9. SHAP grounding
    print("\n9. SHAP GROUNDING")
    top_features = valid["shap_top_feature"].value_counts().head(5)
    print("   Top features driving decisions:")
    for feat, count in top_features.items():
        print(f"     {feat}: {count} ({count/len(valid)*100:.1f}%)")

    # Verdict
    checks = [
        ("Success rate >= 99%", success_rate >= 0.99),
        ("AUC >= 0.70", auc >= 0.70),
        ("FCRA coverage = 100%", (has_adverse == len(denials)) if len(denials) else True),
        ("Audit coverage = 100%", audit_coverage >= 1.0),
        ("No unhandled errors", len(errors) == 0),
    ]
    all_pass = all(p for _, p in checks)
    print("\n" + "=" * 70)
    print("VERDICT:")
    for name, passed in checks:
        print(f"   [{'PASS' if passed else 'FAIL'}] {name}")
    print(
        f"\n   OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED — investigate'}"
    )
    print("=" * 70)

    summary = {
        "profiles_processed": int(n_total),
        "successful": int(n_ok),
        "errors": int(len(errors)),
        "success_rate": float(success_rate),
        "auc": float(auc),
        "accuracy": float(acc),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "deny_rate": float(valid["predicted_default"].mean()),
        "actual_default_rate": float(valid["actual_default"].mean()),
        "borderline_flags": borderline,
        "protected_attr_profiles_flagged": protected_flagged,
        "protected_attr_keyword_hits": protected_keyword_hits,
        "escalations": escalated,
        "pii_fields_scrubbed_total": scrubbed_total,
        "denials_issued": int(len(denials)),
        "denials_with_adverse_action": has_adverse,
        "fcra_coverage": float(fcra_coverage),
        "memos_generated": has_memo,
        "avg_memo_length": float(valid["memo_length"].mean()),
        "audit_records": len(audit_records),
        "audit_coverage": float(audit_coverage),
        "audit_fields_ok": int(audit_fields_ok),
        "audit_fields_required": int(audit_fields_required),
        "escalation_queue_records": len(esc_records),
        "avg_processing_ms": avg_time,
        "p95_processing_ms": p95_time,
        "p99_processing_ms": p99_time,
        "top_shap_features": {str(k): int(v) for k, v in top_features.items()},
        "checks": [{"name": n, "passed": p} for n, p in checks],
        "all_checks_passed": all_pass,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {SUMMARY_JSON}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
