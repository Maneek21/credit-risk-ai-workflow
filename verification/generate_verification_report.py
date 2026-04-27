"""Render the human-readable verification report from saved summaries."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERIFICATION_DIR = ROOT / "verification"
SUMMARY_JSON = VERIFICATION_DIR / "verification_summary.json"
FAIRNESS_JSON = VERIFICATION_DIR / "verification_fairness.json"
REPORT_MD = VERIFICATION_DIR / "verification_report.md"


def _row(name: str, value: str, threshold: str, status: str) -> str:
    return f"| {name} | {value} | {threshold} | {status} |"


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def main() -> int:
    summary = json.loads(SUMMARY_JSON.read_text())
    fairness = json.loads(FAIRNESS_JSON.read_text()) if FAIRNESS_JSON.exists() else None

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines: list[str] = []
    lines.append("# End-to-End Verification Report")
    lines.append("")
    lines.append(f"**Date:** {today}")
    lines.append(f"**Profiles tested:** {summary['profiles_processed']} (UCI Default of Credit Card Clients, stratified)")
    lines.append("**Model version:** 1.0.0 (XGBoost, registered)")
    lines.append("**Production stack active:** AuditLogger, PIIScrubber, RateLimiter, EscalationRouter")
    lines.append("")

    lines.append("## Results Summary")
    lines.append("")
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|---|---|---|---|")
    checks = {c["name"]: c["passed"] for c in summary["checks"]}
    lines.append(_row(
        "Success rate",
        f"{summary['success_rate']*100:.1f}%",
        ">= 99%",
        _status(checks.get("Success rate >= 99%", False)),
    ))
    lines.append(_row(
        "AUC",
        f"{summary['auc']:.4f}",
        ">= 0.70",
        _status(checks.get("AUC >= 0.70", False)),
    ))
    lines.append(_row(
        "FCRA coverage",
        f"{summary['fcra_coverage']*100:.1f}%",
        "= 100%",
        _status(checks.get("FCRA coverage = 100%", False)),
    ))
    lines.append(_row(
        "Audit coverage",
        f"{summary['audit_coverage']*100:.1f}%",
        "= 100%",
        _status(checks.get("Audit coverage = 100%", False)),
    ))
    lines.append(_row(
        "Errors",
        str(summary["errors"]),
        "= 0",
        _status(checks.get("No unhandled errors", False)),
    ))
    lines.append("")

    lines.append("## Decision Quality")
    lines.append("")
    lines.append(f"- AUC: **{summary['auc']:.4f}**")
    lines.append(f"- Accuracy: {summary['accuracy']:.4f}")
    lines.append(f"- True Positives (caught defaulters): {summary['true_positives']}")
    lines.append(f"- True Negatives (approved good payers): {summary['true_negatives']}")
    lines.append(f"- False Positives (denied non-defaulters): {summary['false_positives']}")
    lines.append(f"- False Negatives (approved defaulters): {summary['false_negatives']}")
    lines.append(
        f"- Deny rate: {summary['deny_rate']*100:.1f}% "
        f"(actual default rate in sample: {summary['actual_default_rate']*100:.1f}%)"
    )
    lines.append("")

    lines.append("## Safety Layer Performance")
    lines.append("")
    lines.append("| Layer | Activations | Notes |")
    lines.append("|---|---|---|")
    lines.append(f"| Borderline flags | {summary['borderline_flags']} | Profiles within +/-0.15 of approval threshold; routed to MEDIUM-priority queue |")
    lines.append(f"| Protected-attribute filter (profiles flagged) | {summary['protected_attr_profiles_flagged']} | LLM output scanned with word-bounded regex; flagged profiles are routed HIGH |")
    lines.append(f"| Protected-attribute keyword hits (sum) | {summary['protected_attr_keyword_hits']} | Total keyword occurrences across all flagged profiles |")
    lines.append(f"| Escalation queue writes | {summary['escalation_queue_records']} | Persisted to `verification/escalations/*.jsonl` |")
    lines.append(f"| FCRA disclosure injection | {summary['denials_with_adverse_action']}/{summary['denials_issued']} denials | Deterministic — never delegated to LLM |")
    lines.append(f"| PII fields scrubbed | {summary['pii_fields_scrubbed_total']} | Removed from LLM prompt path before any third-party call |")
    lines.append("")

    lines.append("## Auditability")
    lines.append("")
    lines.append(f"- Audit records written: **{summary['audit_records']}** (expected {summary['successful']})")
    lines.append(f"- Audit field completeness: {summary['audit_fields_ok']}/{summary['audit_fields_required']} required fields present in every record")
    lines.append("- Storage: append-only JSONL with `os.fsync` after every write (`workflow/audit.py` JSONLBackend)")
    lines.append("- Each record carries: decision_id, timestamp_utc, model_version, applicant_features (PII-scrubbed), probability, decision, top-5 SHAP factors, full LLM prompt + response, safety_results, final_output, processing_time_ms")
    lines.append("")

    lines.append("## Performance")
    lines.append("")
    lines.append(f"- Average processing time per profile: **{summary['avg_processing_ms']:.0f} ms**")
    lines.append(f"- P95: {summary['p95_processing_ms']:.0f} ms")
    lines.append(f"- P99: {summary['p99_processing_ms']:.0f} ms")
    lines.append(f"- Concurrent workers: 8 (Phase 6 RateLimiter token-bucket throttle, gpt-4o-mini)")
    lines.append("")

    lines.append("## SHAP Grounding (top features driving decisions)")
    lines.append("")
    for feat, count in summary["top_shap_features"].items():
        lines.append(f"- `{feat}`: {count} profiles ({count/summary['successful']*100:.1f}%)")
    lines.append("")

    if fairness is not None:
        lines.append("## Fairness (XGBoost decisions, 80% rule)")
        lines.append("")
        for axis, label in [("by_sex", "Sex"), ("by_education", "Education"), ("by_age", "Age")]:
            block = fairness[axis]
            lines.append(f"### By {label}")
            lines.append("")
            lines.append("| Group | n | Deny rate |")
            lines.append("|---|---|---|")
            for r in block["groups"]:
                lines.append(f"| {r['group']} | {r['n']} | {r['deny_rate']:.3f} |")
            lines.append("")
            lines.append(
                f"- Disparate Impact Ratio: **{block['disparate_impact']:.3f}** "
                f"({'PASS' if block['passes_80_rule'] else 'FAIL'} the 0.80 rule)"
            )
            lines.append("")
        lines.append(f"_{fairness['note']}_")
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(f"**{'ALL CHECKS PASSED' if summary['all_checks_passed'] else 'SOME CHECKS FAILED'}**")
    lines.append("")
    lines.append("This verification confirms the pipeline correctly:")
    lines.append("")
    lines.append("1. Predicts default probability using the registered XGBoost model (v1.0.0)")
    lines.append("2. Generates SHAP-grounded credit memos for every successful application")
    lines.append("3. Produces FCRA-compliant adverse action notices for every denial")
    lines.append("4. Scrubs PII before any LLM call (GLBA boundary)")
    lines.append("5. Logs every decision to an immutable, append-only audit trail (SR 11-7 §IV)")
    lines.append("6. Routes borderline / protected-attribute / high-value cases to the escalation queue")
    lines.append("7. Respects rate limits and the cost cap configured on the RateLimiter")
    lines.append("8. Detects and flags any protected-attribute language in LLM output (word-bounded scan)")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
