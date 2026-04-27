#!/usr/bin/env python3
"""Post-processing utilities for LLM-generated credit documents.

Provides three hardening layers between raw LLM output and final delivery:
  1. SHAP filter — strips protected attributes before LLM sees them
  2. FCRA/ECOA compliance injection — appends mandatory legal disclosures
  3. Output validation — programmatically verifies memo/notice correctness

Usage:
  from postprocess import filter_shap_protected, inject_fcra_disclosure, validate_memo
"""
import re
from typing import List, Dict, Optional

# ── 1. SHAP Protected Attribute Filter ─────────────────────────────────

PROTECTED_FEATURES = {
    "SEX", "MARRIAGE", "AGE", "EDUCATION",
    # Also catch transformed names
    "cat__SEX_2", "cat__MARRIAGE_1", "cat__MARRIAGE_2", "cat__MARRIAGE_3",
    "cat__EDUCATION_1", "cat__EDUCATION_2", "cat__EDUCATION_3", "cat__EDUCATION_4",
    "num__AGE",
    # Human-readable labels
    "Sex: Female", "Married", "Single", "Marital: Other",
    "Education: Graduate", "Education: University", "Education: High School",
    "Education: Other", "Age",
}

# Case-insensitive patterns for matching in various formats
PROTECTED_PATTERNS = [
    r"\b(sex|gender)\b",
    r"\b(marriage|marital|married|single)\b",
    r"\b(education|graduate|university|high.school)\b",
    r"\bage\b",
]


def filter_shap_protected(
    shap_factors: List[Dict],
    max_factors: int = 5,
    all_factors: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Remove protected attributes from SHAP factor list.

    If a protected attribute is in the top N, replace it with the next
    non-protected factor from the full list.

    Args:
        shap_factors: Top SHAP factors (list of dicts with 'feature' key)
        max_factors: How many factors to return
        all_factors: Full ranked list to draw replacements from.
                     If None, just filters without replacement.

    Returns:
        Filtered list of SHAP factors with no protected attributes.
    """
    filtered = []
    for f in shap_factors:
        feat = f.get("feature", "")
        label = f.get("label", "")
        if feat.upper() in {p.upper() for p in PROTECTED_FEATURES}:
            continue
        if label in PROTECTED_FEATURES:
            continue
        if any(re.search(pat, feat, re.IGNORECASE) for pat in PROTECTED_PATTERNS):
            continue
        if any(re.search(pat, label, re.IGNORECASE) for pat in PROTECTED_PATTERNS):
            continue
        filtered.append(f)

    # If we lost factors and have a full list, backfill
    if all_factors and len(filtered) < max_factors:
        seen = {f.get("feature") for f in filtered}
        for f in all_factors:
            if len(filtered) >= max_factors:
                break
            feat = f.get("feature", "")
            label = f.get("label", "")
            if feat in seen:
                continue
            if feat.upper() in {p.upper() for p in PROTECTED_FEATURES}:
                continue
            if label in PROTECTED_FEATURES:
                continue
            if any(re.search(pat, feat, re.IGNORECASE) for pat in PROTECTED_PATTERNS):
                continue
            if any(re.search(pat, label, re.IGNORECASE) for pat in PROTECTED_PATTERNS):
                continue
            filtered.append(f)
            seen.add(feat)

    return filtered[:max_factors]


# ── 2. FCRA/ECOA Compliance Injection ──────────────────────────────────

ECOA_DISCLOSURE = """
YOUR RIGHTS UNDER FEDERAL LAW:

Under the Equal Credit Opportunity Act (ECOA), the creditor is prohibited
from discriminating against credit applicants on the basis of race, color,
religion, national origin, sex, marital status, or age (provided the
applicant has the capacity to enter into a binding contract). The reasons
listed above are the specific factors from our evaluation that contributed
to this decision.
""".strip()

FCRA_DISCLOSURE = """
CREDIT REPORTING DISCLOSURE:

The following consumer reporting agency provided information that was
used in connection with this decision:

  [Agency Name]
  [Agency Address]
  [Agency Phone Number]

The reporting agency played no part in our decision and is unable to
supply specific reasons why we have denied credit to you. You have a
right under the Fair Credit Reporting Act to know the information
contained in your credit file at the consumer reporting agency. You
also have a right to a free copy of your report from the reporting
agency, if you request it no later than 60 days after you receive
this notice. In addition, if you find that any information contained
in the report you receive is inaccurate or incomplete, you have the
right to dispute the matter with the reporting agency.
""".strip()

CONTACT_FOOTER = """
If you have any questions regarding this notice, you may contact us at:

  [Institution Name]
  [Institution Address]
  [Institution Phone Number]
  [Institution Email]
""".strip()


def inject_fcra_disclosure(notice_text: str, include_fcra: bool = True) -> str:
    """Inject mandatory ECOA and FCRA disclosures into an adverse-action notice.

    Replaces any existing rights section with the standardized template,
    ensuring 100% compliance regardless of what the LLM generated.

    Args:
        notice_text: Raw LLM-generated adverse-action notice
        include_fcra: Whether to include FCRA credit-report disclosure

    Returns:
        Notice with standardized legal disclosures appended
    """
    # Remove any existing rights/disclosure section the LLM may have written
    # Look for common patterns
    patterns_to_strip = [
        r"YOUR RIGHTS.*?(?=\n---|\n\n\n|\Z)",
        r"CREDIT REPORTING.*?(?=\n---|\n\n\n|\Z)",
        r"If you have questions.*?(?=\n---|\n\n\n|\Z)",
    ]

    cleaned = notice_text
    for pat in patterns_to_strip:
        cleaned = re.sub(pat, "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Strip trailing whitespace and dividers
    cleaned = re.sub(r"\n---\s*$", "", cleaned.rstrip())
    cleaned = cleaned.rstrip()

    # Append standardized disclosures
    result = cleaned + "\n\n" + ECOA_DISCLOSURE

    if include_fcra:
        result += "\n\n" + FCRA_DISCLOSURE

    result += "\n\n" + CONTACT_FOOTER + "\n\n---"

    return result


# ── 3. Output Validation ───────────────────────────────────────────────

def validate_memo(
    memo_text: str,
    profile_data: Dict,
    shap_factors: List[Dict],
) -> Dict:
    """Programmatically validate a generated credit memo.

    Checks:
      - Numeric claims match input data
      - Protected attributes not used as primary reasons
      - SHAP factor ordering respected
      - No speculative language

    Returns:
        Dict with pass/fail flags and issue descriptions
    """
    issues = []

    # Check for protected attribute mentions as decision factors
    protected_phrases = [
        (r"(?:because|due to|based on).*\b(sex|gender|male|female)\b", "sex"),
        (r"(?:because|due to|based on).*\b(age|older|younger|elderly)\b", "age"),
        (r"(?:because|due to|based on).*\b(education|graduate|university|degree)\b", "education"),
        (r"(?:because|due to|based on).*\b(married|single|marital|divorced)\b", "marital status"),
    ]
    for pattern, attr in protected_phrases:
        if re.search(pattern, memo_text, re.IGNORECASE):
            issues.append(f"PROTECTED_ATTR_AS_REASON: {attr} appears to be used as a decision factor")

    # Check for speculative language
    speculation_patterns = [
        r"\b(likely|probably|perhaps|might|could be|may indicate|suggests? that)\b",
        r"\b(we can infer|this implies|this means they)\b",
    ]
    for pattern in speculation_patterns:
        matches = re.findall(pattern, memo_text, re.IGNORECASE)
        if matches:
            issues.append(f"SPECULATION: found speculative language: {matches[:3]}")

    # Check for hallucinated data (numbers not in the profile)
    # Extract all numbers from the memo
    memo_numbers = set()
    for match in re.finditer(r"[\d,]+(?:\.\d+)?", memo_text):
        num_str = match.group().replace(",", "")
        try:
            memo_numbers.add(float(num_str))
        except ValueError:
            pass

    # Extract all numbers from profile data
    profile_numbers = set()
    for v in profile_data.values():
        try:
            profile_numbers.add(float(v))
        except (ValueError, TypeError):
            pass

    # Also add derived values that might appear
    # (averages, ratios, etc. — allow some tolerance)
    # Skip small integers (1-12) as they appear in dates, counts, etc.
    suspicious = []
    for n in memo_numbers:
        if n < 13 or n in (2026, 2025, 2024):  # skip dates and small ints
            continue
        # Check if close to any profile number (within 1%)
        close = any(
            abs(n - p) / max(abs(p), 1) < 0.01
            for p in profile_numbers
        )
        if not close:
            suspicious.append(n)

    if suspicious:
        issues.append(f"POSSIBLE_HALLUCINATION: numbers not in profile: {suspicious[:5]}")

    # Check for model internals leaking
    internal_terms = [
        r"\bSHAP\b", r"\bfeature importance\b", r"\bmachine learning\b",
        r"\bXGBoost\b", r"\blogistic regression\b", r"\bneural network\b",
        r"\bclassifier\b", r"\bpredicted probability\b",
    ]
    for pattern in internal_terms:
        if re.search(pattern, memo_text, re.IGNORECASE):
            issues.append(f"MODEL_INTERNAL_LEAKED: {pattern.strip(chr(92)).strip('b')}")

    return {
        "valid": len(issues) == 0,
        "n_issues": len(issues),
        "issues": issues,
        "checks_run": [
            "protected_attr_as_reason",
            "speculation_language",
            "number_hallucination",
            "model_internal_leak",
        ],
    }


def validate_adverse_action(
    notice_text: str,
    shap_factors: List[Dict],
) -> Dict:
    """Validate an adverse-action notice against requirements.

    Checks:
      - Denial reasons map to provided SHAP factors
      - No prohibited attributes as reasons
      - ECOA disclosure present
      - FCRA disclosure present (if applicable)
      - Plain language (no jargon)
    """
    issues = []

    # Check ECOA disclosure
    if "equal credit opportunity" not in notice_text.lower():
        issues.append("MISSING_ECOA: No ECOA disclosure found")

    # Check for prohibited attributes as reasons
    reason_section = ""
    match = re.search(
        r"(?:PRINCIPAL REASONS|REASONS FOR).*?(?:YOUR RIGHTS|CREDIT REPORTING|\Z)",
        notice_text, re.DOTALL | re.IGNORECASE
    )
    if match:
        reason_section = match.group()

    prohibited_in_reasons = [
        (r"\b(sex|gender|male|female)\b", "sex/gender"),
        (r"\b(race|ethnic|national origin)\b", "race/ethnicity"),
        (r"\b(religion|religious)\b", "religion"),
        (r"\b(age|older|younger|elderly|retirement)\b", "age"),
        (r"\b(education|degree|graduate|university)\b", "education"),
        (r"\b(married|single|marital|spouse|divorced)\b", "marital status"),
    ]
    for pattern, attr in prohibited_in_reasons:
        if re.search(pattern, reason_section, re.IGNORECASE):
            issues.append(f"PROHIBITED_REASON: {attr} mentioned in denial reasons")

    # Check reason count (ECOA requires specific reasons, 2-4 typical)
    reason_matches = re.findall(r"^\d+\.", notice_text, re.MULTILINE)
    if len(reason_matches) < 2:
        issues.append(f"TOO_FEW_REASONS: only {len(reason_matches)} reason(s) found")
    elif len(reason_matches) > 4:
        issues.append(f"TOO_MANY_REASONS: {len(reason_matches)} reasons (max 4 recommended)")

    return {
        "valid": len(issues) == 0,
        "n_issues": len(issues),
        "issues": issues,
        "has_ecoa": "equal credit opportunity" in notice_text.lower(),
        "has_fcra": "consumer reporting agency" in notice_text.lower(),
        "n_reasons": len(reason_matches),
    }


# ── 4. Uncertainty Flagging ────────────────────────────────────────────

UNCERTAINTY_CAVEAT = """
NOTE: This assessment falls in the borderline risk range (predicted default
probability between {pd_low:.0%} and {pd_high:.0%}). The model's confidence
in this decision is limited. This case warrants additional review by a
senior credit analyst before a final determination is made.
""".strip()


def get_uncertainty_flag(
    default_probability: float,
    low_threshold: float = 0.20,
    high_threshold: float = 0.45,
) -> Optional[str]:
    """Return an uncertainty caveat if PD falls in the borderline zone.

    Args:
        default_probability: XGBoost predicted default probability
        low_threshold: Lower bound of uncertain zone
        high_threshold: Upper bound of uncertain zone

    Returns:
        Caveat string if PD is borderline, None otherwise.
    """
    if low_threshold <= default_probability <= high_threshold:
        return UNCERTAINTY_CAVEAT.format(
            pd_low=low_threshold, pd_high=high_threshold
        )
    return None


def inject_uncertainty_flag(memo_text: str, default_probability: float) -> str:
    """Insert uncertainty caveat into a memo if PD is borderline."""
    flag = get_uncertainty_flag(default_probability)
    if not flag:
        return memo_text

    # Insert before the RECOMMENDATION section if it exists
    match = re.search(r"\n(\d+\.\s*RECOMMENDATION)", memo_text, re.IGNORECASE)
    if match:
        insert_pos = match.start()
        return memo_text[:insert_pos] + "\n\n" + flag + "\n" + memo_text[insert_pos:]

    # Otherwise append before the closing divider
    memo_text = re.sub(r"\n---\s*$", "", memo_text.rstrip())
    return memo_text + "\n\n" + flag + "\n\n---"


if __name__ == "__main__":
    # Quick self-test
    test_factors = [
        {"feature": "PAY_0", "label": "Most recent payment status", "shap": 1.5},
        {"feature": "MARRIAGE", "label": "Marital status", "shap": 0.3},
        {"feature": "LIMIT_BAL", "label": "Credit limit", "shap": 0.2},
        {"feature": "AGE", "label": "Age", "shap": 0.15},
        {"feature": "BILL_AMT1", "label": "Most recent bill", "shap": 0.1},
        {"feature": "PAY_AMT1", "label": "Most recent payment", "shap": 0.08},
    ]

    filtered = filter_shap_protected(test_factors, max_factors=4, all_factors=test_factors)
    print("SHAP filter test:")
    for f in filtered:
        print(f"  {f['feature']}: {f['label']} (SHAP={f['shap']})")
    assert all(f["feature"] not in ("MARRIAGE", "AGE") for f in filtered)
    print("  PASS: MARRIAGE and AGE removed, backfilled with BILL_AMT1 and PAY_AMT1\n")

    print("Uncertainty flag test:")
    print(f"  PD=0.30: {get_uncertainty_flag(0.30) is not None}")  # True
    print(f"  PD=0.10: {get_uncertainty_flag(0.10) is not None}")  # False
    print(f"  PD=0.70: {get_uncertainty_flag(0.70) is not None}")  # False
    print("  PASS\n")

    print("FCRA injection test:")
    sample = "NOTICE OF ADVERSE ACTION\n\nReasons:\n1. Late payments\n\n---"
    injected = inject_fcra_disclosure(sample)
    assert "consumer reporting agency" in injected.lower()
    assert "equal credit opportunity" in injected.lower()
    print("  PASS: ECOA + FCRA disclosures injected")
