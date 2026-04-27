# Phase 04 — Fairness Audit

**Task type:** Deterministic (Python scripts only — never LLM)
**When to use:** The user wants to audit a model's predictions for bias,
disparate impact, or regulatory compliance across protected attributes.

## What this phase does

Computes fairness metrics for any binary classifier's predictions across
protected demographic groups. Uses Fairlearn for computation and checks
results against regulatory thresholds (four-fifths rule, ECOA requirements).

**This phase MUST be run with deterministic scripts.** Do not ask an LLM
to compute fairness metrics — the results will be unreliable and non-reproducible.

## Why this matters

From our empirical evaluation:

- Every LLM tested violated the four-fifths rule on education level
  (DI ratios 0.49–0.64 for university vs. graduate)
- In the test population, graduates and university-educated borrowers
  default at virtually identical rates (45.5% vs 46.3%) — meaning
  education-based denial is both unfair AND not predictive
- Classical models also show bias: XGBoost violates for "2+ minority
  races" (DI 0.7524) on HMDA data
- The McKinsey/IACPM 2025 survey found 75% of financial institutions
  cite risk and governance (including fairness) as the top adoption barrier

## Running the audit

```bash
python scripts/compute_fairness.py --dataset uci --model all
python scripts/compute_fairness.py --dataset hmda --model all --year 2022
```

## Metrics computed

For each protected attribute and each model, the script computes:

### Group-level metrics

| Metric | Definition | Regulatory threshold | Reference |
|---|---|---|---|
| **Selection rate** | P(APPROVE \| group) | — | Base metric |
| **Disparate impact ratio** | min(group rate) / max(group rate) | ≥ 0.80 (four-fifths rule) | `references/fairness-metrics.md` |
| **Demographic parity difference** | max(group rate) - min(group rate) | ≤ 0.10 (guideline) | `references/fairness-metrics.md` |
| **Equalized odds difference** | max difference in TPR or FPR across groups | ≤ 0.10 (guideline) | `references/fairness-metrics.md` |

### Protected attributes

| Attribute | Groups | Available in |
|---|---|---|
| Sex | Male, Female | UCI, HMDA |
| Education | Graduate, University, High School, Other | UCI |
| Age | Under 30, 30–50, Over 50 | UCI, HMDA |
| Race | White, Black, Asian, Hispanic, Other | HMDA only |
| Ethnicity | Hispanic, Not Hispanic | HMDA only |

### Intersectional analysis

When the dataset supports it, compute metrics for intersections:
- Race × Sex (e.g., Black Female, White Male)
- Race × Age (e.g., Black Under-30, White Over-50)

Intersectional groups with fewer than 30 observations should be flagged
as underpowered and excluded from regulatory conclusions.

## Interpreting results

### Four-fifths rule (disparate impact)

The four-fifths rule (80% rule) from the EEOC Uniform Guidelines states
that a selection rate for any protected group less than 80% of the rate
for the highest-rate group constitutes evidence of adverse impact.

```
DI ratio = (lowest group approval rate) / (highest group approval rate)

If DI ratio < 0.80 → FAILS four-fifths rule → evidence of disparate impact
If DI ratio >= 0.80 → PASSES four-fifths rule
```

### What to do when a model fails

1. **Identify the source.** Use SHAP (from `scripts/compute_shap.py`) to
   determine whether the protected attribute or its proxies are driving
   the disparity.
2. **Check base rates.** If the actual default rate differs substantially
   across groups, some disparity may be statistically justified. Compare
   the model's approval-rate gap to the actual default-rate gap.
3. **Consider fairness-constrained thresholding.** Japinye & Adedugbe (2025)
   showed this reduces demographic-parity gaps by 59–67% at 3.2–5.8%
   accuracy cost.
4. **Document findings.** Record the disparity, its source, and any
   mitigation in the credit memo (Phase 03) and compliance report.

## Output files

| File | Contents |
|---|---|
| `results/fairness.csv` | All fairness metrics by attribute, group, and model |
| `results/disparate_impact.csv` | DI ratios with pass/fail flags per model and attribute |
| `results/disparate_impact.png` | Bar chart of DI ratios with 0.80 threshold line |
| `results/fairness_intersectional.csv` | Intersectional metrics (if applicable) |

## LLM-specific fairness evaluation

When evaluating LLM predictions from Phase 02, join the LLM decisions
with the original dataset to recover protected attributes:

```python
decisions = pd.read_csv("results/llm_decisions.csv")
data = load_dataset()  # has SEX, EDUCATION, AGE columns
merged = decisions.merge(data, on="row_index")
# Now compute DI ratios on merged["decision"] grouped by merged["SEX"], etc.
```

This is how we discovered that all four LLMs violated the four-fifths rule
on education while passing on sex — the LLMs are "selectively fair" on
alignment-trained dimensions but not on others.
