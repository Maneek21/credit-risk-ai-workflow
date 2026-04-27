# Phase 03 — Credit Memo Generation

**Task type:** LLM-driven (structured data → professional narrative)
**When to use:** The user needs a credit memo, underwriting summary, or
risk assessment narrative for a borrower profile.

## What this phase does

Uses an LLM to draft a credit memo from structured borrower data and model
outputs. This is the highest-value LLM application in production credit
workflows — Moody's Agentic Solutions reduced credit memo prep from ~40 hours
to ~2–3 minutes, and HSBC uses generative AI to draft corporate credit memos.

Unlike default prediction (Phase 02), memo generation plays to LLM strengths:
synthesizing structured data into coherent narrative, following templates,
and producing professional prose. This is where LLMs belong in the pipeline.

## Prerequisites

Before generating a memo, you need:

1. **Borrower profile** — structured fields from Phase 01 or directly from
   the dataset (credit limit, repayment history, demographics).
2. **Model prediction** — default probability and decision from Phase 02
   (the classical model, not the LLM).
3. **SHAP explanation** — top contributing features from `scripts/compute_shap.py`.
   This grounds the memo in the model's actual reasoning, preventing hallucination.
4. **Fairness check** — results from Phase 04, confirming no disparate-impact
   flags for this borrower's demographic group.

## Memo structure

The generated memo should follow this template (adapted from industry standard
credit committee memo format):

```
CREDIT RISK ASSESSMENT MEMO
============================

Borrower:       {name or ID}
Date:           {date}
Analyst:        AI-Assisted (Model: {model_name}, Version: {version})
Recommendation: {APPROVE / DENY}

1. EXECUTIVE SUMMARY
   One-paragraph recommendation with the predicted default probability
   and the key factors driving the decision.

2. BORROWER PROFILE
   Tabular summary of the borrower's financial position, with each
   value contextualized against the population distribution.

3. RISK ASSESSMENT
   - Primary risk factors (Tier 1): repayment history, payment patterns
   - Secondary factors (Tier 2): credit utilization, limit level
   - Mitigating factors: any positive signals offsetting risk
   Each factor must cite the specific data point and its SHAP contribution.

4. MODEL DETAILS
   - Model used: {model_name}
   - Predicted default probability: {PD}
   - Confidence interval: {if available from bootstrap}
   - Key SHAP drivers (top 5 features with values and contributions)

5. FAIRNESS & COMPLIANCE CHECK
   - Disparate impact status for borrower's demographic groups
   - Any flags from Phase 04 fairness audit
   - Statement that demographic factors were not used as primary
     decision drivers

6. RECOMMENDATION
   Final recommendation with conditions (if conditional approval)
   or specific denial reasons (if denied).
```

## Prompt template

Use the template at `assets/prompt_templates/memo_draft.txt`. The prompt
provides the borrower data, model output, and SHAP factors as structured
input, then instructs the LLM to produce the memo following the template above.

### Grounding rules (critical)

The prompt enforces these rules to prevent hallucination:

1. **Every numeric claim must trace to the input data.** The LLM must not
   invent income figures, employment details, or financial metrics not
   present in the borrower profile.
2. **Risk factors must align with SHAP output.** If SHAP says PAY_0 is the
   top driver, the memo must lead with repayment status — not education or age.
3. **Population context must use the provided statistics.** When describing
   whether a value is "high" or "low," reference the population mean/median
   from the statistics table, not the LLM's general knowledge.
4. **No speculative reasoning.** Do not infer employment status from age,
   income from credit limit, or stability from marital status.

## Quality rubric

Score each generated memo on these dimensions (useful for Phase 6.5 evaluation):

| Dimension | Score 1 (Poor) | Score 3 (Adequate) | Score 5 (Excellent) |
|---|---|---|---|
| **Factual grounding** | Contains invented data points | All claims traceable but some lack context | Every claim cites specific input data with population context |
| **Risk identification** | Misses primary risk factors | Identifies main risks but ordering doesn't match SHAP | Risk factors ordered by SHAP importance with correct directionality |
| **Compliance** | No fairness mention | Mentions compliance generically | Specific DI ratios cited, demographic factors correctly relegated |
| **Hallucination** | 3+ unsupported claims | 1–2 minor unsupported claims | Zero hallucinated facts |
| **Professional quality** | Disorganized, missing sections | Follows template, some rough prose | Publication-ready, clear executive summary, logical flow |

## Output

Save generated memos to `results/memos/` with filename pattern
`memo_{borrower_id}_{model}_{date}.md`.

Save quality scores (if evaluated) to `results/memo_quality.csv` with columns:
`borrower_id, model, factual_grounding, risk_identification, compliance,
hallucination, professional_quality, total_score, evaluator`

## Post-generation hardening

After the LLM generates a memo, apply these deterministic checks from
`scripts/postprocess.py`:

### SHAP protected-attribute filter (pre-generation)

Before passing SHAP factors to the LLM, call `filter_shap_protected()`
to strip SEX, MARRIAGE, AGE, and EDUCATION from the fact