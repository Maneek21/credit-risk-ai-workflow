# Methodology Reference — Accuracy & Calibration Metrics

## AUC (Area Under the ROC Curve)

**What it measures:** Discrimination — the model's ability to rank defaulters
above non-defaulters.

**Range:** 0.5 (random) to 1.0 (perfect).

**Interpretation:**
- 0.50–0.60: Poor — barely better than random
- 0.60–0.70: Fair — some discriminatory power
- 0.70–0.80: Good — standard for production credit models
- 0.80–0.90: Excellent — well-tuned model on rich data
- 0.90+: Exceptional — verify for data leakage

**Formula:** AUC = P(score(defaulter) > score(non-defaulter)) across all pairs.

**Why it matters for credit:** AUC measures whether the model can distinguish
good borrowers from bad. A model with AUC 0.70 correctly ranks a random
defaulter above a random non-defaulter 70% of the time.

**Limitations:** AUC is threshold-independent — it doesn't tell you about
calibration (whether a predicted 30% default rate is actually 30%).

## KS Statistic (Kolmogorov-Smirnov)

**What it measures:** Maximum separation between the cumulative distribution
of defaulters and non-defaulters.

**Range:** 0.0 (no separation) to 1.0 (perfect separation).

**Interpretation:**
- < 0.20: Poor separation
- 0.20–0.40: Acceptable
- 0.40–0.60: Good
- > 0.60: Excellent

**Formula:** KS = max|F_default(t) - F_non_default(t)| across all thresholds t.

**Why it matters for credit:** KS is widely used in credit scoring because
it identifies the threshold at which the model best separates defaults from
non-defaults — directly useful for setting approval cutoffs.

**Relationship to AUC:** KS ≈ 2 × (AUC - 0.5) for well-behaved models.

## Brier Score

**What it measures:** Calibration — how close predicted probabilities are to
actual outcomes.

**Range:** 0.0 (perfect) to 1.0 (worst).

**Formula:** Brier = (1/N) × Σ(predicted_probability - actual_outcome)²

**Interpretation:** Lower is better. A Brier score of 0.15 means the average
squared error between predicted probability and actual 0/1 outcome is 0.15.

**Why it matters for credit:** Calibration is critical for pricing — if the
model predicts 20% default probability, approximately 20% of such borrowers
should actually default. Miscalibration leads to mispriced loans.

**Benchmark:** For a 22% default rate, predicting the base rate for everyone
gives Brier = 0.22 × 0.78 = 0.172. Any useful model should beat this.

## ECE (Expected Calibration Error)

**What it measures:** Calibration, binned — the average absolute difference
between predicted confidence and actual accuracy within probability bins.

**Range:** 0.0 (perfect) to 1.0 (worst).

**Formula:**
1. Bin predictions into B equal-width bins (default B=10)
2. For each bin: gap = |avg_predicted_probability - actual_frequency|
3. ECE = Σ(bin_size/total) × gap

**Interpretation:** ECE of 0.05 means predictions are off by 5 percentage
points on average within each bin.

**Why it matters for credit:** ECE gives a more granular view of calibration
than Brier — it reveals where the model is well-calibrated and where it
breaks down (e.g., overconfident at high probabilities).

## Gini Coefficient

**What it measures:** Another discrimination metric, linearly related to AUC.

**Formula:** Gini = 2 × AUC - 1

**Range:** 0.0 (random) to 1.0 (perfect).

**Usage:** Common in European banking. If someone asks for Gini, compute
from AUC.

## Confusion Matrix Derived Metrics

For binary credit decisions (APPROVE/DENY), at a given threshold:

| Metric | Formula | Credit context |
|---|---|---|
| Accuracy | (TP + TN) / N | Overall correct rate |
| Precision (DENY) | TP / (TP + FP) | Of those denied, how many would actually default? |
| Recall (DENY) | TP / (TP + FN) | Of actual defaulters, how many did we catch? |
| Deny rate | (TP + FP) / N | Proportion of applicants denied — compare to base rate |
| False positive rate | FP / (FP + TN) | Good borrowers incorrectly denied |

Where: TP = correctly denied (would default), FP = incorrectly denied
(would not default), TN = correctly approved, FN = incorrectly approved
(will default).
