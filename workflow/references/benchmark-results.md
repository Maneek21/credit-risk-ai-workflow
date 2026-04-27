# Benchmark Results Reference

## Published Academic Benchmarks

### Japinye & Adedugbe 2025 — XGBoost + SHAP Multi-Market Evaluation

**Source:** SSRJAI, 2025
**URL:** https://ssrpublisher.com/wp-content/uploads/2025/09/Explainable-AI-for-Credit-Scoring-with-SHAP-Calibrated-Ensembles-A-Multi-Market-Evaluation-on-Public-Lending-Data.pdf

| Dataset | N | XGBoost AUC | SHAP Kendall τ |
|---|---|---|---|
| UCI Default of Credit Card Clients | 30,000 | 0.892 | 0.94 ± 0.03 |
| Home Credit Default Risk | 307,511 | 0.901 | — |
| Lending Club | 887,379 | 0.923 | — |

Fairness-constrained thresholding: DP gap reduced 59–67% at 3.2–5.8% cost.

### CALM — LLMs as Credit Scorers (arXiv 2310.00566)

**Source:** "Empowering Many, Biasing a Few" — first benchmark of LLMs on credit.

Models tested: Bloomz, Vicuna, Llama 1/2, Llama 2-chat, ChatGLM2, ChatGPT, GPT-4.

**Key finding:** Zero-shot LLMs underperform expert ML systems on structured
credit data. This confirms our own findings — LLMs hit a hard ceiling on
tabular credit prediction.

### Zero is Not Hero Yet (arXiv 2305.16633)

Zero-shot ChatGPT vs fine-tuned RoBERTa on financial NLP tasks. Fine-tuned
models still win on structured tasks. Foundational citation for the "where
LLMs belong" argument.

---

## Our Empirical Results

### Classical Models — UCI (Phase 02)

| Model | AUC | KS | Brier | ECE |
|---|---|---|---|---|
| Logistic Regression | 0.713 | — | — | — |
| XGBoost | 0.774 | — | — | — |
| MLP | 0.766 | — | — | — |

Our XGBoost AUC (0.774) is lower than Japinye's (0.892) because we used
default hyperparameters for reproducibility rather than extensive tuning.
The relative ordering (XGBoost > MLP > LR) is consistent.

### SHAP Stability — UCI (Phase 03)

Bootstrap (n=50) pairwise Spearman ρ on feature importance ranks:

| Model | Mean ρ | Interpretation |
|---|---|---|
| XGBoost | 0.877 | Good stability — feature rankings are consistent |
| MLP | 0.799 | Moderate stability |
| Logistic Regression | 0.679 | Lower stability — coefficients shift with resampling |

### LLM Zero-Shot — UCI (Phase 06)

200 stratified profiles, 2 runs each, 4 models:

| Model | AUC | Deny Rate | Confidence (mean) | Consistency |
|---|---|---|---|---|
| Claude Opus 4.7 | 0.573 | 55% | 0.835 | 99.5% |
| GPT-4o | 0.575 | 70% | 0.862 | 98.0% |
| GPT-5.4 | 0.562 | 62.5% | 0.885 | 99.0% |
| GPT-OSS-120B | 0.548 | 68% | 0.789 | 90.0% |

**Total API spend:** $8.57 (Claude $7.47, GPT-4o $0.47, GPT-5.4 $0.63, GPT-OSS $0)

### LLM Bias — UCI (Phase 06 analysis)

Education DI ratio (university / graduate approval rate):

| Model | Education DI | Four-Fifths | Sex DI | Four-Fifths |
|---|---|---|---|---|
| Claude Opus 4.7 | 0.529 | FAIL | 1.005 | PASS |
| GPT-4o | 0.503 | FAIL | 0.895 | PASS |
| GPT-5.4 | — | FAIL | — | PASS |
| GPT-OSS-120B | — | FAIL | — | PASS |

Pattern: all LLMs pass on sex (alignment-trained), all fail on education
(not alignment-trained). "Selective fairness."

### Skill-Augmented — UCI (Phase 6B)

Single system prompt with base-rate anchoring, feature hierarchy, bias guards:

| Metric | GPT-4o Baseline | GPT-4o + Skill | Delta |
|---|---|---|---|
| AUC | 0.620 | 0.631 | +0.011 |
| Deny rate | 70% | 47% | -23pp |
| Education DI | 0.503 | 0.711 | +0.208 |
| Mean confidence | 0.862 | 0.762 | -0.100 |
| Sex DI | 0.895 | 0.974 | +0.079 |

| Metric | Claude Baseline | Claude + Skill | Delta |
|---|---|---|---|
| AUC | 0.648 | 0.692 | +0.044 |
| Deny rate | 55% | 37.9% | -17.1pp |
| Education DI | 0.529 | 0.370 | -0.159 |

**Critical finding:** Same skill produced opposite effects — GPT-4o improved
on education DI (0.50→0.71), Claude worsened (0.53→0.37). Prompt-based
interventions are model-specific and unpredictable. This is the strongest
argument for the workflow architecture: don't try to make LLMs better
underwriters; use them where they excel (Phases 01, 03, 05) and let
deterministic models handle prediction (Phase 02).

---

## Summary: Where Each Tool Belongs

| Task | Best tool | Evidence |
|---|---|---|
| Default prediction | XGBoost (AUC 0.774) | 2× better than best LLM; deterministic; auditable |
| Document extraction | LLM | Rocket: 70% automated; no classical alternative |
| Credit memo drafting | LLM | Moody's: 40h → 3min; requires narrative synthesis |
| Fairness auditing | Deterministic script | Must be exact, reproducible, threshold-checkable |
| Adverse action notices | LLM + SHAP grounding | Translates model output to plain language |
| Drift monitoring | Deterministic script | Must compare distributions precisely over time |
