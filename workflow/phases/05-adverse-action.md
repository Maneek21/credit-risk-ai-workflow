# Phase 05 — Adverse Action Notices

**Task type:** LLM-driven, grounded in SHAP explanations
**When to use:** A borrower has been denied credit and needs a legally
compliant explanation of the reasons for denial.

## What this phase does

Generates adverse-action notices that comply with the Equal Credit Opportunity
Act (ECOA) and Fair Credit Reporting Act (FCRA) requirements. Uses an LLM to
translate model outputs and SHAP explanations into plain-language denial
reasons that a consumer can understand.

This is another high-value LLM task — the challenge is converting technical
model output (SHAP values, feature importances) into specific, accurate,
non-discriminatory denial reasons. An LLM excels at this translation task
when properly grounded.

## Regulatory requirements

See `references/regulatory-context.md` for full details. Key requirements:

### ECOA (Regulation B, 12 CFR 1002.9)

- Creditors must notify applicants of adverse action within 30 days
- Notice must include **specific reasons** for the denial (up to 4)
- Reasons must be the actual factors that caused the denial, not generic
- Must not reference protected characteristics as reasons

### FCRA (15 U.S.C. § 1681m)

- If a credit report was used, must identify the credit reporting agency
- Must inform the applicant of their right to obtain a free credit report
- Must inform of right to dispute inaccurate information

### Prohibited reasons (never include in adverse-action notice)

- Race, color, national origin
- Sex, marital status
- Age (unless used in a demonstrably valid scoring system)
- Religion
- Receipt of public assistance
- Exercise of rights under the Consumer Credit Protection Act

## Generating the notice

### Step 1 — Get SHAP-grounded denial reasons

For the denied borrower, get the top SHAP contributors to the denial:

```python
# Get SHAP values for the specific borrower
shap_values = compute_individual_shap(model, borrower_features)

# Get top 4 features pushing toward denial (positive SHAP = higher PD)
top_factors = sorted(
    zip(feature_names, shap_values),
    key=lambda x: x[1],
    reverse=True
)[:4]
```

### Step 2 — Map features to human-readable reasons

| Model feature | Adverse-action reason template |
|---|---|
| PAY_0 (recent repayment status) | "Recent payment delinquency on revolving accounts" |
| PAY_2–PAY_6 (historical repayment) | "History of late payments over the past 6 months" |
| LIMIT_BAL (credit limit) | "Insufficient existing credit history/limits" |
| BILL_AMT (bill amounts) | "High outstanding revolving balance relative to credit limit" |
| PAY_AMT (payment amounts) | "Minimum payment behavior on existing accounts" |
| AGE | DO NOT USE — protected characteristic |
| SEX | DO NOT USE — protected characteristic |
| EDUCATION | DO NOT USE — near-zero predictive value, potential proxy |
| MARRIAGE | DO NOT USE — protected characteristic |

### Step 3 — Generate the notice with LLM

Use the prompt template at `assets/prompt_templates/adverse_action.txt`.
The prompt provides:
- The borrower's profile data
- The model's decision and default probability
- The top 4 SHAP-grounded denial reasons (already mapped to human-readable)
- The feature values that triggered each reason

The LLM's job is to:
1. Write a clear, professional notice following the template
2. Explain each denial reason in plain language
3. Contextualize each reason (e.g., "Your most recent payment was 2 months
   late, which is a strong indicator of default risk")
4. Include required FCRA disclosures
5. NOT add any reasons beyond those grounded in SHAP

## Notice template

```
NOTICE OF ADVERSE ACTION

Date: {date}
Applicant: {borrower_id}
Application: {application_type}

Dear Applicant,

We have reviewed your application for credit and regret to inform you
that we are unable to approve your request at this time.

PRINCIPAL REASONS FOR THIS DECISION:

1. {reason_1} — {plain_language_explanation}
2. {reason_2} — {plain_language_explanation}
3. {reason_3} — {plain_language_explanation}
4. {reason_4} — {plain_language_explanation}

YOUR RIGHTS:

Under the Equal Credit Opportunity Act, you have the right to know why
your application was denied. The reasons listed above are the specific
factors from our evaluation that contributed to this decision.

{FCRA_disclosure_if_applicable}

You may contact us at {contact_info} if you have questions about this
notice or wish to discuss your application further.
```

## Validation checklist

After generating each notice, verify:

- [ ] Exactly 2–4 specific denial reasons listed
- [ ] Each reason maps to a SHAP-identified factor (not hallucinated)
- [ ] No protected characteristics appear as reasons
- [ ] No speculative reasoning (no "likely," "probably," "may indicate")
- [ ] FCRA disclosure included if credit report was referenced
- [ ] Plain language — readable at an 8th-grade level
- [ ] No internal model details exposed (no "SHAP value," "feature importance")

## Post-generation hardening

### Deterministic FCRA/ECOA injection

Do NOT rely on the LLM to include legal disclosures. In Phase 6C testing,
36% of notices (5/14) omitted the FCRA credit-report disclosure.

Instead, the LLM generates only the denial reasons section. Then
`scripts/postprocess.py` deterministically appends the required legal
boilerplate:

```python
from postprocess import inject_fcra_disclosure
final_notice = inject_fcra_disclosure(llm_notice, include_fcra=True)
```

This strips any LLM-generated rights section and replaces it with the
standardized ECOA disclosure, FCRA credit-report disclosure, and contact
fo