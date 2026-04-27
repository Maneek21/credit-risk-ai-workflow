# Fairness Metrics Reference

## Overview

Credit fairness metrics measure whether a model's decisions
disproportionately affect protected demographic groups. US lending
regulation requires lenders to monitor and mitigate disparate impact.

## Metrics

### Demographic Parity Difference (DP Diff)

**Definition:** The maximum difference in approval rates across demographic
groups for a protected attribute.

**Formula:** DP_diff = max_g(P(APPROVE|group=g)) - min_g(P(APPROVE|group=g))

**Threshold:** DP_diff ≤ 0.10 is a common guideline (no regulatory mandate
for the exact number, but larger gaps attract scrutiny).

**Example:** If males are approved at 75% and females at 70%, DP_diff = 0.05.

### Disparate Impact Ratio (DI Ratio)

**Definition:** The ratio of the lowest group's approval rate to the highest
group's approval rate.

**Formula:** DI = min_g(P(APPROVE|group=g)) / max_g(P(APPROVE|group=g))

**Regulatory threshold:** DI ≥ 0.80 (the "four-fifths rule" or "80% rule")

**Source:** EEOC Uniform Guidelines on Employee Selection Procedures (1978),
29 CFR 1607.4(D). Originally for employment; widely applied to lending by
analogy and regulatory practice.

**Example:** If graduate school applicants are approved at 60% and university
applicants at 30%, DI = 30/60 = 0.50. This FAILS the four-fifths rule.

**Important nuance:** The four-fifths rule is a screening tool, not a
definitive legal standard. A DI ratio below 0.80 creates a prima facie case
of disparate impact; the lender can defend with business necessity and
demonstrable predictive validity.

### Equalized Odds Difference (EO Diff)

**Definition:** The maximum difference across groups in either true positive
rate or false positive rate.

**Formula:** EO_diff = max(max_g(TPR_g) - min_g(TPR_g), max_g(FPR_g) - min_g(FPR_g))

**Threshold:** EO_diff ≤ 0.10 is a common guideline.

**Why it matters:** Equalized odds checks whether the model is not just
equally approving across groups, but equally accurate — a model could have
equal approval rates but systematically miss defaults in one group.

### Selection Rate

**Definition:** P(APPROVE|group=g) — the raw approval rate for each group.

**Not a metric itself** but the building block for DI ratio and DP diff.
Always report selection rates alongside the derived metrics so readers can
see the actual numbers.

## Protected Attributes in Credit

### Legally protected (ECOA, Regulation B)

- Race, color, national origin
- Sex (including gender identity and sexual orientation, per CFPB)
- Marital status
- Age (with narrow exceptions for empirically derived credit scoring systems)
- Religion
- Receipt of public assistance income

### Commonly monitored (not explicitly listed in ECOA but tracked)

- Education level — not legally protected, but our empirical testing shows
  LLMs treat it as a primary factor despite near-zero predictive value.
  DI ratios of 0.49–0.64 for university vs. graduate across all tested LLMs.
- Income source
- Geography (potential proxy for race)

## Empirical Findings from This Project

### Classical models (HMDA 2022, NY)

| Model | Race DP Diff | Race DI | Sex DP Diff | Sex DI |
|---|---|---|---|---|
| Logistic Regression | 0.466 | varies by group | 0.025 | ~0.97 |
| XGBoost | 0.648 | 0.75 (2+ minority) | 0.019 | ~0.98 |
| MLP | 0.598 | varies | 0.022 | ~0.97 |

### LLMs (UCI, zero-shot, 200 profiles)

| Model | Education DI | Sex DI | Age DP Diff |
|---|---|---|---|
| Claude Opus 4.7 | 0.529 | 1.005 | 0.181 |
| GPT-4o | 0.503 | 0.895 | 0.134 |
| GPT-4o + skill | 0.711 | 0.974 | 0.183 |

Key insight: LLMs show "selective fairness" — they pass on sex (an
alignment-trained dimension) but fail on education (not alignment-trained).
The skill improved education DI from 0.50 to 0.71 but still fails four-fifths.

## Fairness-Constrained Thresholding

Japinye & Adedugbe (2025) demonstrated that post-hoc threshold adjustment
per demographic group can reduce DP gaps by 59–67% at 3.2–5.8% accuracy cost.

Approach: instead of a single global threshold, set group-specific thresholds
that equalize approval rates while minimizing total accuracy loss.

This is implemented in `scripts/compute_fairness.py` as an optional flag:
```bash
python scripts/compute_fairness.py --constrained-threshold --max-accuracy-loss 0.05
```

## Limitations

- Fairness metrics measure outcomes, not intent. A passing DI ratio does
  not prove the model is fair — it may use proxies for protected attributes.
- Different fairness definitions can conflict. It is mathematically
  impossible to satisfy demographic parity, equalized odds, and calibration
  simultaneously (except in degenerate cases). Choose the metric most
  relevant to the regulatory context.
- Small subgroups produce unstable metrics. Require n ≥ 30 per group
  for reliable DI ratio computation.
