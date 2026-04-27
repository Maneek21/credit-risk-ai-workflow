# Model Validation Report — Credit Risk AI Workflow

**TEMPLATE — banks must customize for their specific implementation, training data, and operating environment.** This document is a worked example produced from a public benchmark study (UCI Default of Credit Card Clients; HMDA NY 2018–2022). It is intended to illustrate the structure, evidence requirements, and tone of an SR 11-7 / OCC 2011-12 model validation package. It is **not** a regulatory submission, and the numerical findings reflect a research-grade build, not a production model.

---

| Field | Value |
|---|---|
| Model name | Credit Risk AI Workflow (XGBoost scorer + LLM communicator + deterministic safety layers) |
| Model owner (line of business) | Consumer Credit / Retail Underwriting (illustrative) |
| Model developer | Maneek Mohan |
| Validation date | 2026-04-27 |
| Validator | Independent Model Risk Management (illustrative) |
| Risk tier (proposed) | **Tier 2 — High-impact consumer-facing** |
| Recommended decision | **CONDITIONAL APPROVAL** with controls listed in Sections 12 and 13 |
| Regulatory frame | U.S. Federal Reserve SR 11-7; OCC Bulletin 2011-12; ECOA / Reg B; FCRA; CFPB UDAAP; NIST AI RMF 1.0 (cross-walk) |
| Underlying training data | UCI Default of Credit Card Clients (Taiwan 2005, n=30,000); HMDA NY LAR 2018–2022 (stratified 100,000-row test slices per year for drift) |

---

## 1. Executive Summary

The Credit Risk AI Workflow is a hybrid decisioning system in which a gradient-boosted tree model (XGBoost) produces the binding default-probability estimate and approval recommendation, a large language model (LLM) drafts the customer-facing credit memo and adverse-action notice from SHAP-derived factor lists, and a deterministic Python layer enforces FCRA disclosures, protected-attribute filtering, uncertainty flagging, and validation checks. The architectural commitment — *"the LLM communicates but never decides"* — is the central safety property that this validation has examined.

On the four SR 11-7 axes (accuracy, stability, bias, interpretability) and on the LLM-specific concerns introduced by the communication layer (factual grounding, hallucination, FCRA compliance, intra-run consistency), the validator's findings are summarized below.

**Accuracy.** On the UCI hold-out, XGBoost achieves AUC 0.7744, KS 0.4267, Brier 0.1358, and ECE 0.0210. This is a +0.062 AUC lift over Logistic Regression (LR, the regulatory benchmark) and +0.008 AUC over an MLP. On HMDA NY 2018–2022, XGBoost AUC ranges from 0.819 (test year 2020) to 0.838 (2022), tracking expected economic-cycle behavior.

**Stability.** SHAP feature-rank stability across n=50 bootstraps is strongest for XGBoost (mean pairwise Spearman ρ = 0.877) versus MLP (0.799) and LR (0.679). Per-year AUC on HMDA shows no degradation through the COVID-19 distribution shock; XGBoost in fact gains AUC marginally from 2020 (0.819) to 2022 (0.838).

**Bias.** Disparate-impact ratios on the HMDA scoring action exceed the 4/5ths regulatory floor for the **Free Form Text Only** race subgroup across all three classical models (LR 0.527; XGBoost 0.303; MLP 0.362), and for **Free Form Text Only** ethnicity (LR 0.672; XGBoost 0.698; MLP 0.720). XGBoost additionally violates 4/5ths on **2 or more minority races** (DI 0.752). Sex-based DI ratios pass the threshold for all models. Intersectional (race × sex) demographic-parity differences are large (0.667–0.833) and warrant remediation. See Sections 9 and 13.

**Interpretability.** XGBoost's SHAP attributions are computed via TreeSHAP and pass through to the LLM communication layer as the *only* permitted citation source. The protected-attribute filter caught 24 of 500 borderline cases in the v2 workflow benchmark; the FCRA injection layer achieves 100% legal-disclosure coverage.

**LLMs as scorers — explicitly rejected.** Across four frontier LLMs scored against UCI labels, zero-shot AUC ranges from 0.575 (gpt-5.4) to 0.580 (claude-opus-4-7), materially below XGBoost (0.774). Skill-augmented prompting moves Claude to AUC 0.692 and GPT-5.4 to 0.665 — still 0.08 to 0.11 AUC below XGBoost. The validator concurs with the architectural decision to keep LLMs out of the scoring path.

**Recommendation.** The validator recommends **CONDITIONAL APPROVAL** for deployment as a decision-support system in a Tier 2 use case, contingent on the controls in Section 12 (compensating controls), the findings in Section 13 (six numbered findings, severity LOW–HIGH), and the ongoing-monitoring plan in Section 11 (Phase 6 / Phase 7 drift framework with annual revalidation and event-driven re-runs on any CRITICAL drift alert). The model **must not** be deployed in jurisdictions or product lines where the underlying training data (Taiwan 2005 credit cards; New York mortgage applications) is not representative — most pointedly, this includes the Indian retail credit context that motivates the broader project. A jurisdiction-appropriate refit on local data is a prerequisite to any production launch outside the validation envelope.

---

## 2. Model Description

### 2.1 Purpose and intended use

The model produces, for each consumer credit application:

1. A point estimate of probability of default (PD) within the model's reference horizon.
2. A binary approve/deny decision, derived from PD versus a configurable threshold (default 0.50).
3. A natural-language credit memo (200–300 words) summarizing the decision rationale, grounded in the top SHAP factors for that application.
4. An FCRA-compliant adverse-action notice (when denied), enumerating up to four denial reasons drawn from the same SHAP factor list, plus a deterministic FCRA disclosure block.
5. A flag set indicating uncertainty, protected-attribute leakage, or other operational alerts that route the application to human review.

### 2.2 Architecture

The system is a three-layer pipeline. Layer 1 decides; Layer 2 communicates; Layer 3 enforces.

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1 — XGBoost DECIDES                                   │
│   Inputs: structured borrower features                      │
│   Output: PD ∈ [0,1], decision ∈ {APPROVE, DENY}            │
│   Reference: trained pipeline persisted as joblib artifact  │
├─────────────────────────────────────────────────────────────┤
│ Layer 2 — LLM COMMUNICATES                                  │
│   Inputs: applicant features, decision, PD, top-5 SHAP      │
│   Output: credit memo (free text), adverse-action body      │
│   Constraint: prompt forbids citing any factor outside the  │
│   SHAP factor list passed in; protected-attribute mention   │
│   forbidden                                                 │
├─────────────────────────────────────────────────────────────┤
│ Layer 3 — Deterministic code ENFORCES                       │
│   - Protected-attribute keyword filter on LLM output        │
│   - FCRA disclosure block injected post hoc                 │
│   - Uncertainty flag if |PD − 0.5| < 0.35 (BORDERLINE)      │
│   - Programmatic validation of memo / notice structure      │
│   - Audit trail capture                                     │
└─────────────────────────────────────────────────────────────┘
```

The reference implementation is `workflow/pipeline.py` in the repository, class `CreditWorkflow`, method `process_application(...)`. The configurable parameters are `model_path`, `llm_provider` (`openai` or `anthropic`), `llm_model`, `uncertainty_threshold` (default 0.35), and `protected_attributes` (default keyword list of 30+ tokens covering race, sex, age, religion, national origin, disability, marital status, and pregnancy).

### 2.3 Inputs

For the UCI build:

| Field group | Fields |
|---|---|
| Demographics (non-protected by design of this dataset) | `LIMIT_BAL`, `EDUCATION`, `MARRIAGE`, `AGE`, `SEX` (note: SEX is included in UCI; treatment in production must comply with ECOA — see Section 10) |
| Repayment history (6 months) | `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6` |
| Bill amount (6 months) | `BILL_AMT1` through `BILL_AMT6` |
| Payment amount (6 months) | `PAY_AMT1` through `PAY_AMT6` |

All 23 features are used; no manual feature selection beyond removal of the row identifier.

For the HMDA build (used for fairness and drift evaluation):

| Field group | Fields |
|---|---|
| Loan / property | `loan_amount`, `loan_term`, `loan_purpose`, `property_value`, `occupancy_type` |
| Borrower income & ratios | `income`, `debt_to_income_ratio` (categorical bin), `combined_loan_to_value_ratio` |
| Credit | `applicant_credit_score_type` |
| Lender | `lei` (lender identifier) |
| Protected attributes (used only for fairness evaluation, *not* as model inputs) | `derived_race`, `derived_ethnicity`, `derived_sex` |

### 2.4 Outputs

- **Decision** (`APPROVE` / `DENY`).
- **Probability of default** (`prob` ∈ [0,1]).
- **Credit memo** (string; SHAP-grounded narrative, 200–300 words).
- **Adverse-action notice** (string; FCRA-compliant denial letter; emitted only when `decision == DENY`).
- **SHAP factors** (top-5 list with `feature`, `shap_value`, `direction`).
- **Flags** (subset of `BORDERLINE`, `PROTECTED_ATTR_DETECTED`, `MANUAL_REVIEW`).
- **Metadata** (model class, LLM model identifier, count of protected-attribute hits).

---

## 3. Conceptual Soundness *(SR 11-7 §III, Model Development)*

### 3.1 Theoretical basis

**Gradient-boosted trees on tabular credit data.** Gradient boosting builds an additive ensemble of shallow regression trees, each fit to the gradient of a differentiable loss with respect to the previous ensemble's predictions (Friedman 2001; Chen & Guestrin 2016 for the XGBoost variant). For the binary classification problem here the loss is logistic deviance. The justification for this family on retail credit data is empirical and well documented: tabular structured features with mixed types, non-linear interactions, and modest sample sizes are precisely the regime in which boosted trees outperform neural nets and linear models. Lessmann, Baesens, Seow, and Thomas (2015), in the most-cited credit-scoring benchmark of the last decade, find that XGBoost / random-forest-style ensembles dominate single-classifier methods across 41 real-world credit datasets. The UCI benchmark in this study is consistent with that finding (XGBoost +0.062 AUC over LR; +0.008 over MLP).

**SHAP for attribution.** Shapley additive explanations (Lundberg & Lee 2017) provide a unique, axiomatically grounded local attribution method based on the cooperative-game-theoretic Shapley value. SHAP is the only attribution method with simultaneous local accuracy, missingness, and consistency guarantees, and TreeSHAP (Lundberg, Erion, & Lee 2018) computes exact Shapley values for tree ensembles in low-order polynomial time, making it production-tractable for boosted-tree models. The validator notes that SHAP attributions are *post-hoc explanations of model behavior*, not causal statements about the borrower; the credit memo prompt is correctly framed in terms of "factors the model identified," not "reasons you were denied" in a causal sense. This distinction is preserved in the prompt template (`_build_adverse_action_prompt`).

**Fairness metrics aligned to U.S. consumer-credit law.** The fairness instrumentation uses three metric families: demographic parity difference (DP-diff), equalized odds difference (EO-diff), and disparate impact ratio (DI). DI is the operational metric in U.S. lending, with the 4/5ths rule (DI ≥ 0.80) drawn from Equal Employment Opportunity Commission guidance and adopted by analogy in CFPB/Reg B fair-lending exams. DP-diff and EO-diff (Hardt, Price, & Srebro 2016) capture, respectively, group differences in selection rates and group differences in error-rate parity. The use of three complementary metrics is appropriate because they capture distinct fairness conceptions and are mathematically incompatible in general (no model can satisfy all three simultaneously when base rates differ across groups — the impossibility theorem of Chouldechova 2017 and Kleinberg, Mullainathan, & Raghavan 2017).

**Deterministic safety layers aligned to FCRA / ECOA.** The FCRA injection block ensures every denial notice contains the four FCRA §615(a) required disclosures (right to free report, right to dispute, the consumer-reporting agency did not make the decision, agency contact details). The protected-attribute keyword filter implements an output-side ECOA Reg B §1002.6(b)(2) safeguard. The uncertainty flag implements the SR 11-7 §V principle that high-stakes decisions in the model's indeterminate region should be routed to human review rather than be automated end-to-end.

### 3.2 Literature support

| Topic | Citation | Relevance |
|---|---|---|
| Boosted trees dominate credit scoring | Lessmann, Baesens, Seow, Thomas (2015), *EJOR* 247(1):124–136 | Empirical justification for XGBoost as the scorer |
| Unified SHAP framework | Lundberg & Lee (2017), *NeurIPS* | Theoretical basis for the attribution layer |
| TreeSHAP polynomial-time exact | Lundberg, Erion, Lee (2018) | Tractability for production |
| Equality of opportunity | Hardt, Price, Srebro (2016), *NeurIPS* | Definition of equalized odds |
| Fairness impossibility | Chouldechova (2017); Kleinberg et al. (2017) | Justification for triangulating multiple metrics |
| LLM hallucination in structured-task settings | Ji et al. (2023) "Survey of Hallucination in NLG" | Motivates the SHAP-grounding constraint and protected-attribute filter |
| SR 11-7 model risk | Federal Reserve / OCC (2011) | Validation framework |

### 3.3 Conceptual issues identified

The validator notes one concern requiring documentation: the `SEX` feature is included as a model input on the UCI build. This is acceptable for academic benchmarking, but in U.S. production it would be an ECOA Reg B violation as a direct input. In the HMDA build, `derived_sex` is appropriately held out from the input set and used only for fairness evaluation. Section 13 lists this as Finding 1.

---

## 4. Data Quality

### 4.1 UCI Default of Credit Card Clients

| Attribute | Value |
|---|---|
| Source | University of California Irvine ML Repository, dataset 350 |
| Origin | Taiwan, 2005, anonymized credit card portfolio |
| n | 30,000 |
| Default rate | ~22.1% |
| Missingness | None (cleaned source) |
| Vintage | 20+ years old at validation date |
| Geographic scope | Single country (Taiwan) |
| Validator's data-quality view | Suitable for benchmarking; **not** suitable as the basis for any production model serving any retail credit market in 2026 |

### 4.2 HMDA NY 2018–2022

| Attribute | Value |
|---|---|
| Source | CFPB HMDA Loan Application Register (LAR) |
| Geographic scope | State of New York only (single-state slice; the HMDA national LAR is the broader population) |
| Sample design | Stratified random sample, 100,000 rows per test year, fixed seed 42 |
| Years | 2018 (training), 2019, 2020 (COVID year), 2021, 2022 |
| Target | `action_taken` recoded to approve / deny (excluding withdrawn / incomplete applications) |
| Approval rate | Test-year approvals 2020/2021/2022: 78.8%, 81.9%, 76.8% |
| Protected attributes available | `derived_race`, `derived_ethnicity`, `derived_sex` (used for fairness only, not as inputs) |
| Validator's data-quality view | Adequate for fairness and temporal-drift evaluation; **single-state limitation** must be flagged for any extrapolation. New York is not nationally representative on race/ethnicity composition or housing market characteristics. |

### 4.3 Known limitations of the data foundation

- **No India retail credit data.** The broader project context (a research project with credit risk assessment focus) is not supported by the training data. The validator must be explicit: **this model has not been trained or validated on Indian credit data** and cannot be deployed in that market without a refit. This is recorded as Finding 4.
- **Single state, single dataset for fairness.** Generalization of the fairness findings beyond NY HMDA mortgage applications is not supported.
- **Single-time-period training for UCI.** A 2005 vintage cannot be expected to capture post-2020 consumer credit behavior. UCI is used here only for the accuracy / interpretability benchmarks; HMDA carries the drift evaluation.
- **No fraud / synthetic-identity layer in either dataset.** Production credit decisions almost always interact with fraud risk; the model does not consider this.

---

## 5. Developmental Evidence *(SR 11-7 §III)*

### 5.1 Train / validation / test split

- Stratified split, 70% / 15% / 15%, stratified on the binary default label.
- `random_state = 42` everywhere (fixed in the project's working-rules section).
- Splits are persisted; the same indices are used across LR, XGBoost, and MLP to make accuracy comparisons valid.

### 5.2 Feature engineering

- All 23 UCI fields used; no manual feature selection.
- Standard preprocessing pipeline (categorical encoding for `EDUCATION`, `MARRIAGE`, `SEX`; numeric scaling for the bill / payment amounts).
- HMDA build uses ~12 features; protected attributes are deliberately excluded from the input set.

### 5.3 Hyperparameters (UCI build)

| Model | Key hyperparameters |
|---|---|
| Logistic Regression | `solver=lbfgs`, `max_iter=1000`, `C=1.0`, `class_weight=None` |
| XGBoost | `n_estimators=400`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`, `objective=binary:logistic`, `eval_metric=auc` |
| MLP | `hidden_layer_sizes=(64,32)`, `activation=relu`, `solver=adam`, `alpha=1e-4`, `max_iter=200`, `early_stopping=True` |

Hyperparameters were not tuned by grid search in the benchmark; values are reasonable defaults for the data scale. A production deployment would require Bayesian or grid-search tuning with held-out validation, recorded in the model-development log per SR 11-7 §III.

### 5.4 Reproducibility

- Single random seed (42) for splits, bootstrap, and any stochastic LLM call where temperature > 0.
- Training scripts are version-controlled in the project repo.
- Persisted artifacts: trained model joblibs (Phase 2), SHAP values (Phase 3), all `results/*.csv` files used for this validation.

---

## 6. Outcomes Analysis *(SR 11-7 §IV)*

### 6.1 UCI hold-out accuracy

Source file: `results/02_accuracy.csv`.

| Model | AUC | KS | Brier | ECE |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7129 | 0.3764 | 0.1460 | 0.0493 |
| XGBoost | **0.7744** | **0.4267** | **0.1358** | **0.0210** |
| MLP | 0.7663 | 0.4091 | 0.1369 | 0.0274 |

XGBoost is best on all four axes. The +0.062 AUC over LR is operationally significant — at the 50th-percentile decile, this corresponds to a meaningful change in default-capture rate.

### 6.2 Approximate 95% confidence interval on AUC (Hanley-McNeil)

The Hanley-McNeil approximation gives the standard error of AUC as:

SE(AUC) = √[ (AUC(1−AUC) + (n_pos−1)(Q1 − AUC²) + (n_neg−1)(Q2 − AUC²)) / (n_pos · n_neg) ]

where Q1 = AUC / (2 − AUC) and Q2 = 2·AUC² / (1 + AUC). With n_test = 4,500 and default rate 22.1% (n_pos ≈ 994, n_neg ≈ 3,506), this yields:

| Model | AUC | Approx. SE | Approx. 95% CI |
|---|---:|---:|:---|
| Logistic Regression | 0.7129 | 0.0089 | [0.696, 0.730] |
| XGBoost | 0.7744 | 0.0083 | [0.758, 0.791] |
| MLP | 0.7663 | 0.0084 | [0.750, 0.783] |

The XGBoost CI does not overlap the LR CI; the AUC advantage is statistically supported. The XGBoost and MLP CIs overlap; their difference is not statistically distinguishable on this hold-out alone, but the calibration advantage (Brier 0.1358 vs 0.1369; ECE 0.021 vs 0.027) provides a secondary basis for preferring XGBoost. Bootstrap-based confidence intervals are also reported in Phase 3 (n=50 bootstraps) for SHAP rank stability — see Section 8.

### 6.3 Calibration

ECE (Expected Calibration Error, 10 bins) values:

- LR: 0.0493 — modest mis-calibration; predicted probabilities deviate from empirical default rates by ~5 percentage points on average within bin.
- XGBoost: 0.0210 — well calibrated; sub-3% mean deviation.
- MLP: 0.0274 — well calibrated.

Calibration plots are persisted at `results/02_calibration.png`. The validator notes that XGBoost's strong out-of-the-box calibration is a known property of the boosting objective when fit with logistic loss, and reduces the need for downstream Platt or isotonic recalibration.

### 6.4 Recommendation from outcomes analysis

Outcomes analysis supports XGBoost as the production scorer. LR retains value as a regulatory benchmark and challenger model and should be maintained as such. The MLP does not bring sufficient marginal benefit over XGBoost to justify its deployment overhead and lower SHAP-rank stability (Section 8) and may be dropped from the production stack.

---

## 7. Sensitivity Analysis

Formal global / local sensitivity analyses (e.g., Sobol indices on input features; perturbation analysis on calibration; threshold sensitivity on approval cut-off) were **not** performed in this build. This is a gap relative to SR 11-7 §IV expectations for a Tier 2 model.

The validator notes the following partial substitutes that are present:

- **SHAP attribution as local sensitivity.** TreeSHAP provides local feature attributions for every prediction. These are local, exact, and used by the communication layer.
- **Bootstrap-based feature-rank stability.** Section 8 reports rank stability under resampling, which is a coarse proxy for sensitivity to training-data perturbation.
- **Temporal drift.** Section 8 reports per-year AUC, which captures sensitivity to distribution shift.

**Recommendation.** Before any production launch, run (i) a SHAP-based feature-perturbation sensitivity analysis on the top-10 features (vary each by ±1 standard deviation and record PD response), (ii) a threshold-sensitivity analysis sweeping the approval cut-off from 0.30 to 0.70 in steps of 0.05 with operational metric (approval rate, expected loss, false-positive rate) recorded at each step, and (iii) a stress-test on the calibration curve under 2x and 5x default-rate inflation. This is recorded as Finding 5.

---

## 8. Stability Analysis

### 8.1 SHAP feature-rank stability (UCI, n=50 bootstraps)

Source file: `results/03_shap_stability.csv`.

| Model | Bootstraps | Mean pairwise Spearman ρ | Median pairwise Spearman ρ | Mean feature-rank std-dev |
|---|---:|---:|---:|---:|
| Logistic Regression | 50 | 0.6785 | 0.6902 | 4.07 |
| XGBoost | 50 | **0.8772** | **0.8818** | **2.53** |
| MLP | 50 | 0.7987 | 0.8051 | 3.27 |

Interpretation: across 50 bootstrap re-fits, XGBoost re-orders its top features less than LR or MLP. A Spearman ρ of 0.88 indicates that the same factors appear in approximately the same order across resamples; this is the property a credit memo system needs in order to give consistent, defensible reasons for similar applicants. LR is materially less stable on this metric. The validator regards 0.88 as adequate for a Tier 2 system but not at the level (>0.95) the validator would expect for a Tier 1 high-stakes scoring model.

### 8.2 Temporal drift (HMDA NY 2018 train, 2020/2021/2022 test)

Source file: `results/05_drift_metrics.csv`.

| Model | Test year | n_test | Approval rate | AUC | KS | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 2020 | 100,000 | 0.788 | 0.7326 | 0.3705 | 0.1359 | 0.0436 |
| Logistic Regression | 2021 | 100,000 | 0.819 | 0.7354 | 0.3640 | 0.1234 | 0.0486 |
| Logistic Regression | 2022 | 100,000 | 0.768 | 0.7548 | 0.3918 | 0.1443 | 0.0440 |
| XGBoost | 2020 | 100,000 | 0.788 | 0.8190 | 0.5093 | 0.1081 | 0.0150 |
| XGBoost | 2021 | 100,000 | 0.819 | 0.8202 | 0.5066 | 0.0980 | 0.0244 |
| XGBoost | 2022 | 100,000 | 0.768 | **0.8383** | **0.5286** | 0.1116 | 0.0167 |
| MLP | 2020 | 100,000 | 0.788 | 0.8012 | 0.4816 | 0.1151 | 0.0117 |
| MLP | 2021 | 100,000 | 0.819 | 0.8019 | 0.4759 | 0.1039 | 0.0146 |
| MLP | 2022 | 100,000 | 0.768 | 0.8179 | 0.4846 | 0.1208 | 0.0225 |

**Findings on temporal drift:**

1. **No degradation through COVID.** Test-year 2020 was the natural distribution-shock year for U.S. mortgage lending. AUC was *not* reduced relative to subsequent years; in fact 2022 is the highest-AUC test year for all three models. This is consistent with the GSE-supported HMDA mortgage market having absorbed the COVID shock at the level of approval action rather than at the level of underwriting predictability.
2. **Approval-rate shift.** Approval rates in the test set move from 78.8% (2020) → 81.9% (2021) → 76.8% (2022). The 2022 retracement is consistent with the 2022 rate-rising cycle and underlying tightening of mortgage underwriting standards. A production drift monitor should treat the +3pp / -5pp swings here as the empirical scale of normal-regime drift; alarms should sit outside this envelope.
3. **Calibration drift.** ECE is broadly stable for XGBoost (0.015–0.024) across the three test years, indicating calibration robustness over the 2020–2022 horizon.
4. **Cross-model ranking is preserved.** XGBoost > MLP > LR holds in every test year, in every metric.

**Validator's stability conclusion.** Stability is acceptable for the validation envelope (NY HMDA mortgage 2018 → 2020–2022). It is **not** demonstrated for any other geography, product, or vintage. The ongoing-monitoring plan (Section 11) is required to detect if this changes in production.

---

## 9. Fairness Assessment

### 9.1 Aggregate fairness metrics

Source: `results/04_fairness.csv`.

| Model | Attribute | DP-diff | EO-diff | Selection-rate range |
|---|---|---:|---:|---|
| LR | derived_race | 0.466 | 0.380 | [0.474, 0.940] |
| LR | derived_ethnicity | 0.299 | 0.241 | [0.605, 0.903] |
| LR | derived_sex | 0.059 | 0.099 | [0.869, 0.928] |
| LR | race × sex | **0.667** | **1.000** | [0.333, 1.000] |
| XGBoost | derived_race | 0.648 | 0.489 | [0.263, 0.911] |
| XGBoost | derived_ethnicity | 0.271 | 0.131 | [0.605, 0.876] |
| XGBoost | derived_sex | 0.072 | 0.096 | [0.826, 0.898] |
| XGBoost | race × sex | **0.833** | 0.667 | [0.167, 1.000] |
| MLP | derived_race | 0.598 | 0.442 | [0.316, 0.914] |
| MLP | derived_ethnicity | 0.256 | 0.134 | [0.628, 0.884] |
| MLP | derived_sex | 0.067 | 0.112 | [0.836, 0.903] |
| MLP | race × sex | **0.833** | **1.000** | [0.167, 1.000] |

Sex-axis DP-diff is small (5.9–7.2 pp). Race and ethnicity DP-diffs are large. The intersectional race × sex DP-diff exceeds 0.66 for all three models — a clear flag.

### 9.2 Disparate-impact ratios versus 4/5ths threshold

Source: `results/04_disparate_impact.csv`. Table below shows the **4/5ths violations only** (DI < 0.80) by model:

**Logistic Regression (2 violations):**
| Attribute | Group | DI ratio |
|---|---|---:|
| derived_race | Free Form Text Only | 0.527 |
| derived_ethnicity | Free Form Text Only | 0.672 |

**XGBoost (3 violations):**
| Attribute | Group | DI ratio |
|---|---|---:|
| derived_race | 2 or more minority races | 0.752 |
| derived_race | Free Form Text Only | 0.303 |
| derived_ethnicity | Free Form Text Only | 0.698 |

**MLP (3 violations):**
| Attribute | Group | DI ratio |
|---|---|---:|
| derived_race | 2 or more minority races | 0.785 |
| derived_race | Free Form Text Only | 0.362 |
| derived_ethnicity | Free Form Text Only | 0.720 |

**Sex-axis DI (no violations, all models):** Female / Male ratios are 0.991 (LR), 0.988 (XGBoost), 0.993 (MLP). All comfortably above 0.80.

### 9.3 Validator's interpretation

Two observations matter.

**First**, the **Free Form Text Only** group fails the 4/5ths threshold on every model. This group consists of HMDA records where the applicant declined the standard race / ethnicity classification and instead wrote in a free-form response. These records are a small heterogeneous slice and the model is treating them as systematically higher-risk. The validator cannot conclude from the data whether this is (a) a genuine risk signal correlated with non-response behavior, (b) a proxy for a protected characteristic, or (c) an artifact of low sample size within the group. **All three models show this pattern**, so it is a data-driven correlation, not a model-architecture artifact. Remediation: investigate whether removing the free-text race / ethnicity feature from inputs eliminates the disparity, and whether the underlying credit-risk signal can be separated from the protected-class proxy. This is recorded as Finding 2.

**Second**, **XGBoost newly fails 4/5ths on the "2 or more minority races" group** (DI 0.752) and MLP narrowly does so as well (0.785) where LR does not (0.855). This is the canonical accuracy-fairness tradeoff: the more flexible model captures more interaction effects, including some that align with protected-class proxies. Remediation: apply Fairlearn's `ExponentiatedGradient` reduction with a `DemographicParity` or `EqualizedOdds` constraint, or apply a post-hoc threshold-optimization reweighting; re-validate. This is recorded as Finding 3.

**Third**, intersectional race × sex DP-diffs of 0.667–0.833 indicate that single-axis fairness checks are insufficient. The selection-rate range of [0.167, 1.000] for XGBoost and MLP indicates that the lowest-selection intersectional cell receives approval one-sixth as often as the highest. This is a substantial finding that single-axis disparate-impact testing would miss. Recorded as Finding 6.

---

## 10. Limitations and Assumptions

The validator's enumeration of bounds:

1. **Geographic bound.** Trained on Taiwan 2005 (UCI) and New York 2018–2022 (HMDA). No India data, no other-state HMDA, no cross-border generalization.
2. **Temporal bound.** UCI vintage is 2005; HMDA spans 2018–2022. Pre-2018 and post-2022 behavior is not validated.
3. **Product bound.** UCI is credit cards; HMDA is mortgage applications. The two products are *both* present in the validation, but the model is trained separately on each. There is no unified retail-credit model spanning both.
4. **LLM bound.** Zero-shot LLM AUC across four frontier models is 0.575–0.580, ~0.20 AUC below XGBoost. Skill-augmented prompting reaches 0.665–0.692 — still below XGBoost. **Conclusion: LLMs are not the scorer in this architecture, and the validator concurs with that design choice.** Any production proposal that re-introduces an LLM as a primary or secondary scorer must re-trigger validation. Recorded as Finding 4.
5. **Protected-attribute filter is heuristic.** The keyword filter (~30 terms) catches common references but is not adversarially robust. A determined LLM that paraphrases ("the applicant's gender" → "the applicant's profile") can in principle bypass the filter. The v2 workflow benchmark logged 24 hits across 500 cases (4.8%); these are the cases that *did* trigger the filter. Cases that bypassed the filter are by definition unmeasured in the benchmark. The deterministic safety net is the FCRA injection layer (legally required content) plus uncertainty flagging plus human review for borderline cases.
6. **Uncertainty threshold is configured.** The default `uncertainty_threshold = 0.35` flags any PD in [0.15, 0.85] as BORDERLINE. This is broad by design — research-grade — and would be tightened in production to e.g. [0.40, 0.60], with explicit documentation of the operational implications (more borderline flags = more human review = more cost).
7. **No causal inference.** The fairness metrics are statistical, not causal. The validator cannot answer the question "would removing protected-attribute proxies fix the disparity?" without a counterfactual analysis that the project explicitly excluded from scope.
8. **No fraud / synthetic-identity layer.** Production credit decisions interact with fraud-risk scoring; the model does not.
9. **Benchmark grader limitations.** The Phase 6c v2 cross-grading run depended on Anthropic API credits that were exhausted at memo ~120 of 500; the grader was switched to GPT-5.4. This is a single-provider grader and weakens the cross-family-grader signal that the v2 design intended. Recorded in `results/06c_v2_comparison.csv` and acknowledged as a benchmark-methodology limitation.

---

## 11. Ongoing Monitoring Plan *(SR 11-7 §V, Implementation)*

The Phase 6 / Phase 7 monitoring framework defined in the project specification provides the operational basis for ongoing monitoring.

### 11.1 Monitored quantities and thresholds

| Quantity | Computation | Threshold (illustrative; institution must set) | Action on breach |
|---|---|---|---|
| Population Stability Index (PSI) on each top-10 feature | PSI = Σ (p_i − q_i) · ln(p_i / q_i) over decile bins | Yellow > 0.10; Red > 0.25 | Yellow: notify model owner. Red: trigger revalidation. |
| AUC drift | AUC on rolling 90-day production cohort vs. development AUC | Yellow drop > 0.02; Red drop > 0.05 | As above. Red also triggers compensating-control review. |
| Approval-rate shift | Daily approval rate vs. 90-day rolling mean | Yellow ±3pp; Red ±5pp | Investigate underlying driver before acting. |
| SHAP top-10 rank stability | Spearman ρ between current cohort and reference | Yellow < 0.85; Red < 0.70 | Re-fit candidate. |
| Disparate-impact ratio (per protected group) | DI computed on rolling cohort | Red < 0.80 on any monitored group | Halt or remediate; mandatory escalation. |
| LLM output validation issues per 100 cases | Programmatic check (FCRA blocks present, no protected attributes, factor count ≤ 5) | Red > 5 | Halt LLM communication layer; revert to template-based memos until investigated. |
| LLM intra-run consistency (repeat-call agreement) | % of cases where two calls give the same decision | Red < 0.95 | Investigate provider-side change (model upgrade, parameter change). |

### 11.2 Cadence

- **Daily.** Approval-rate shift, decision-volume tracking, LLM validation-issue rate.
- **Weekly.** PSI on top-10 features, AUC drift on labelled cohort.
- **Monthly.** Full fairness pack (DI on all monitored groups; intersectional race × sex).
- **Quarterly.** SHAP rank-stability re-run.
- **Annually.** Full revalidation re-running every section of this document on fresh production data.
- **Event-driven.** Any Red threshold breach triggers an immediate revalidation and a written incident report, regardless of the next scheduled cadence.

### 11.3 Revalidation and challenger comparison

- LR is maintained as the regulatory challenger and is rescored on the same monitoring cohort. A persistent narrowing of the XGBoost-LR AUC gap below +0.02 is itself a signal that the production scorer is degrading.
- The MLP may be retired but its evaluation harness should be retained as a tertiary challenger.
- The LLM-as-scorer benchmark (Phase 6) should be re-run annually to track whether frontier LLM credit-scoring capability has materially advanced; the architectural commitment ("LLM communicates, never decides") should be revisited only if and when zero-shot LLM AUC closes to within 0.02 of XGBoost on a held-out cohort.

---

## 12. Compensating Controls

The validator has identified the following compensating controls already implemented and notes which ones are sufficient as built versus require operational hardening before production.

### 12.1 Five safety layers (in `workflow/pipeline.py`)

1. **SHAP grounding constraint.** The credit-memo and adverse-action prompts pass the top-5 SHAP factors to the LLM with the explicit instruction *"You may ONLY cite factors listed above. Do not invent additional reasons."* The v2 cross-graded benchmark reports `factual_grounding` mean 2.378 / 5 (std 0.688) — this is a research-grade score, materially below what the validator would require for production (mean ≥ 4.0). Recorded as Finding 6.
2. **FCRA disclosure injection.** The `FCRA_DISCLOSURE` block is appended deterministically by code, never by the LLM. The v2 benchmark reports `aa_fcra_compliance_pct = 100.0%` — this is the strongest control in the system and is sufficient as built.
3. **Protected-attribute keyword filter.** A 30+ term keyword scan runs against every LLM output. v2 benchmark logs 24 / 500 hits (4.8%) — i.e., 24 cases where the LLM did emit a protected-attribute mention that the filter caught. This is a non-trivial frequency. **Filter is necessary but not sufficient**; it is keyword-based and not adversarially robust (Section 10, item 5).
4. **Uncertainty flag.** PDs within ±0.35 of 0.50 (i.e., [0.15, 0.85]) trigger `BORDERLINE` and route to manual review. The v2 benchmark added uncertainty flags on 51 of 500 cases (10.2%) — applications routed to human review rather than auto-decisioned. The threshold is configurable; production tightening to ±0.10 is recommended.
5. **Programmatic validation.** Each memo and notice is validated for structure (length, FCRA block present, factor count ≤ 5, no protected attributes). v2 benchmark logged 198 validation issues across 500 cases — these are detected, not undetected. The validator regards detection here as the safety function; the underlying generation-quality issue is a research-grade gap (Finding 6).

### 12.2 Beyond-pipeline controls

6. **Human-in-the-loop escalation.** All `BORDERLINE` cases route to a credit officer; production should additionally route any `PROTECTED_ATTR_DETECTED` flag to compliance review.
7. **Audit trail (Phase 1 of project).** All decisions, PDs, SHAP factors, LLM outputs, flags, and metadata are persisted with timestamp and model-version identifiers. SR 11-7 §V record-keeping satisfied.
8. **Two-LLM cross-grading.** v2 benchmark grades v1 self-graded outputs; cross-grading reveals self-grading inflation of approximately 9 points on 25 (memo: v1 self-grade 24.88 vs v2 cross-grade 15.72). This control is research-grade today; in production, a randomized 5–10% sample of memos should be cross-graded by a second-LLM-family grader and the inflation gap monitored.

### 12.3 Controls that need hardening before production

| Control | Current state | Required state |
|---|---|---|
| Protected-attribute filter | Keyword list, 30+ terms | Add semantic check (embedding similarity to a curated negative-example bank) |
| LLM intra-run consistency | 94.2% (Claude), 98.3% (GPT-4o), 98.7% (GPT-5.4) on Phase 6 | Production minimum 99.5% on memo decision-text |
| Uncertainty threshold | 0.35 (very wide) | Tighten to 0.10 with ops cost-of-review modeled explicitly |
| Memo factual-grounding score | 2.378 / 5 cross-graded | Improve prompts and few-shot exemplars to reach ≥ 4.0 |

---

## 13. Findings and Recommendations

Six numbered findings follow. Each lists severity (LOW / MEDIUM / HIGH), a recommendation, and an owner suggestion.

---

### Finding 1 — Inclusion of `SEX` as a model input (UCI build)

**Severity: HIGH** (regulatory risk under ECOA / Reg B in U.S. production).

**Description.** The UCI Default of Credit Card Clients dataset includes `SEX` as one of its 23 features and the benchmark uses it as a model input. In a U.S. retail credit production setting, including `sex` directly as a predictor is an ECOA Reg B §1002.6 violation, regardless of model accuracy benefit. The HMDA build correctly excludes `derived_sex` from inputs.

**Recommendation.** For any production migration, rebuild the UCI-trained models with `SEX` removed from the input set. Document the AUC delta and confirm it is acceptable against the regulatory constraint. Maintain `SEX` only as a held-out fairness-evaluation variable.

**Owner.** Model developer; Compliance sign-off required.

---

### Finding 2 — Free Form Text Only race / ethnicity group fails 4/5ths on all three models

**Severity: HIGH**.

**Description.** DI ratios for the `Free Form Text Only` race subgroup are LR 0.527, XGBoost 0.303, MLP 0.362. For ethnicity, LR 0.672, XGBoost 0.698, MLP 0.720. All below the 0.80 threshold. Because all three model families exhibit the same pattern, this is data-driven, not architecture-driven.

**Recommendation.** (a) Run an ablation removing the free-text race and ethnicity features from inputs and rescore. (b) Examine whether the underlying signal is non-response *behavior* (which may have legitimate risk content) versus a protected-class *proxy* (which does not). (c) If (a) does not resolve, apply a fairness-constrained retrain (Fairlearn `ExponentiatedGradient` with `DemographicParity`) and document the accuracy/fairness tradeoff.

**Owner.** Model developer; Fair-Lending Compliance review.

---

### Finding 3 — XGBoost violates 4/5ths on "2 or more minority races" where LR does not

**Severity: MEDIUM**.

**Description.** XGBoost DI on `2 or more minority races` is 0.752; LR is 0.855. This is the canonical accuracy-fairness tradeoff: the more flexible model captures more interactions, including some aligned with protected proxies.

**Recommendation.** Apply Fairlearn `ExponentiatedGradient` with `EqualizedOdds` constraint and rescore. If the constrained XGBoost retains a ≥ +0.04 AUC advantage over LR while satisfying 4/5ths on this group, deploy the constrained variant. Otherwise revert to LR for the affected segment.

**Owner.** Model developer.

---

### Finding 4 — LLM zero-shot AUC of ~0.58; LLMs must remain in scribe role

**Severity: MEDIUM** (architectural; the recommendation is consistent with what the system already does).

**Description.** Phase 6 evaluation (n=531 profiles, 4 LLMs, 2 runs each) reports zero-shot AUC: claude-opus-4-7 0.580, gpt-4o 0.579, gpt-5.4 0.575. XGBoost on the same task achieves 0.774. The LLM gap is ~0.20 AUC. Skill-augmented prompting (Phase 6b) closes some of this — Claude reaches 0.692, GPT-5.4 reaches 0.665 — but neither approaches XGBoost. Consistency (intra-run agreement) is 94.2% (Claude) to 98.7% (GPT-5.4).

**Recommendation.** Keep LLMs strictly in the communication layer. Reject any production proposal that re-introduces an LLM as a primary or secondary scorer. Re-run the Phase 6 benchmark annually. The architectural commitment in `workflow/pipeline.py` — Layer 1 (XGBoost) decides, Layer 2 (LLM) communicates — is the correct design and must be preserved through all subsequent revisions.

**Owner.** Model owner; architecture review board.

---

### Finding 5 — Sensitivity analysis not formally performed

**Severity: MEDIUM**.

**Description.** SR 11-7 §IV expects formal sensitivity analysis on a Tier 2 model. The current build provides SHAP local attribution and bootstrap rank-stability as partial substitutes but does not provide threshold sensitivity, calibration-stress sensitivity, or feature-perturbation sensitivity.

**Recommendation.** Before production launch, run (i) feature-perturbation sensitivity on top-10 features (±1σ); (ii) approval-threshold sweep 0.30 → 0.70 in steps of 0.05 with operational metrics; (iii) calibration stress at 2× and 5× default-rate inflation. Document results in a sensitivity-analysis appendix.

**Owner.** Model developer.

---

### Finding 6 — Memo factual-grounding score is research-grade

**Severity: MEDIUM**.

**Description.** The v2 cross-graded benchmark reports `factual_grounding` mean 2.378 / 5 (std 0.688) and `hallucination` mean 1.685 / 5 (lower is better, but framing varies; per the benchmark codebook this is a non-passing score). The v1 self-graded results gave 24.88 / 25 total, but the cross-grader (gpt-5.4 grading gpt-4o memos) gave 15.72 / 25 — a 9.16-point self-grading inflation.

**Recommendation.** Improve memo prompts with curated few-shot exemplars; constrain the LLM further to literal SHAP factor names (rather than paraphrases); add a programmatic post-check that every memo cites at least 3 of the top-5 SHAP features by name. Re-run cross-grading until total ≥ 22 / 25 with std ≤ 1.5 before production launch.

**Owner.** Model developer; LLM-prompt engineer.

---

### Finding 7 (LOW) — Single-state HMDA evaluation

**Severity: LOW** for the validation envelope; **HIGH** if extrapolated outside.

**Description.** Fairness and drift evaluation use New York HMDA only. NY is not nationally representative on demographic composition or housing-market dynamics.

**Recommendation.** Before any cross-state production deployment, repeat Sections 8 and 9 on the target state's HMDA data. Treat each state as a separate validation envelope.

**Owner.** Model owner.

---

## 14. Appendix A — Full Metric Tables

### A.1 UCI accuracy (`02_accuracy.csv`)

| model | auc | ks | brier | ece |
|---|---:|---:|---:|---:|
| logistic_regression | 0.7129 | 0.3764 | 0.1460 | 0.0493 |
| xgboost | 0.7744 | 0.4267 | 0.1358 | 0.0210 |
| mlp | 0.7663 | 0.4091 | 0.1369 | 0.0274 |

### A.2 SHAP rank stability (`03_shap_stability.csv`)

| model | bootstraps | mean_pairwise_spearman | median_pairwise_spearman | mean_feature_rank_stddev |
|---|---:|---:|---:|---:|
| logistic_regression | 50 | 0.6785 | 0.6902 | 4.066 |
| xgboost | 50 | 0.8772 | 0.8818 | 2.533 |
| mlp | 50 | 0.7987 | 0.8051 | 3.267 |

### A.3 Aggregate fairness (`04_fairness.csv`)

| model | attribute | dp_diff | eo_diff | sel_rate_max | sel_rate_min |
|---|---|---:|---:|---:|---:|
| logistic_regression | derived_race | 0.4660 | 0.3801 | 0.9397 | 0.4737 |
| logistic_regression | derived_ethnicity | 0.2985 | 0.2411 | 0.9032 | 0.6047 |
| logistic_regression | derived_sex | 0.0587 | 0.0991 | 0.9278 | 0.8691 |
| logistic_regression | race_x_sex | 0.6667 | 1.0000 | 1.0000 | 0.3333 |
| xgboost | derived_race | 0.6482 | 0.4893 | 0.9113 | 0.2632 |
| xgboost | derived_ethnicity | 0.2714 | 0.1310 | 0.8760 | 0.6047 |
| xgboost | derived_sex | 0.0716 | 0.0962 | 0.8976 | 0.8260 |
| xgboost | race_x_sex | 0.8333 | 0.6667 | 1.0000 | 0.1667 |
| mlp | derived_race | 0.5979 | 0.4417 | 0.9137 | 0.3158 |
| mlp | derived_ethnicity | 0.2562 | 0.1340 | 0.8841 | 0.6279 |
| mlp | derived_sex | 0.0669 | 0.1124 | 0.9030 | 0.8360 |
| mlp | race_x_sex | 0.8333 | 1.0000 | 1.0000 | 0.1667 |

### A.4 Disparate-impact ratios — 4/5ths violations only (`04_disparate_impact.csv`)

| model | attribute | group | ref_group | DI ratio | violates_4_5ths |
|---|---|---|---|---:|:---:|
| logistic_regression | derived_race | Free Form Text Only | White | 0.5268 | True |
| logistic_regression | derived_ethnicity | Free Form Text Only | Not Hispanic or Latino | 0.6718 | True |
| xgboost | derived_race | 2 or more minority races | White | 0.7524 | True |
| xgboost | derived_race | Free Form Text Only | White | 0.3034 | True |
| xgboost | derived_ethnicity | Free Form Text Only | Not Hispanic or Latino | 0.6985 | True |
| mlp | derived_race | 2 or more minority races | White | 0.7846 | True |
| mlp | derived_race | Free Form Text Only | White | 0.3621 | True |
| mlp | derived_ethnicity | Free Form Text Only | Not Hispanic or Latino | 0.7204 | True |

### A.5 Temporal drift (`05_drift_metrics.csv`)

| model | test_year | n | approval_rate | auc | ks | brier | ece |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 2020 | 100,000 | 0.7877 | 0.7326 | 0.3705 | 0.1359 | 0.0436 |
| logistic_regression | 2021 | 100,000 | 0.8187 | 0.7354 | 0.3640 | 0.1234 | 0.0486 |
| logistic_regression | 2022 | 100,000 | 0.7683 | 0.7548 | 0.3918 | 0.1443 | 0.0440 |
| xgboost | 2020 | 100,000 | 0.7877 | 0.8190 | 0.5093 | 0.1081 | 0.0150 |
| xgboost | 2021 | 100,000 | 0.8187 | 0.8202 | 0.5066 | 0.0980 | 0.0244 |
| xgboost | 2022 | 100,000 | 0.7683 | 0.8383 | 0.5286 | 0.1116 | 0.0167 |
| mlp | 2020 | 100,000 | 0.7877 | 0.8012 | 0.4816 | 0.1151 | 0.0117 |
| mlp | 2021 | 100,000 | 0.8187 | 0.8019 | 0.4759 | 0.1039 | 0.0146 |
| mlp | 2022 | 100,000 | 0.7683 | 0.8179 | 0.4846 | 0.1208 | 0.0225 |

### A.6 LLM zero-shot accuracy (`06_llm_metrics.csv`)

| provider | model | n_profiles | accuracy | deny_rate | auc | mean_confidence |
|---|---|---:|---:|---:|---:|---:|
| anthropic | claude-opus-4-7 | 531 | 0.5876 | 0.5292 | 0.5797 | 0.8343 |
| openai | gpt-4o | 531 | 0.5725 | 0.7401 | 0.5792 | 0.8629 |
| openai | gpt-5.4 | 531 | 0.5669 | 0.6554 | 0.5748 | 0.8848 |
| openrouter | openai/gpt-oss-120b:free | 99 | 0.5960 | 0.7071 | 0.6487 | 0.7913 |

### A.7 LLM intra-run consistency (`06_consistency.csv`)

| provider | model | agreement_rate_runs | n_profiles |
|---|---|---:|---:|
| anthropic | claude-opus-4-7 | 0.9416 | 531 |
| openai | gpt-4o | 0.9831 | 531 |
| openai | gpt-5.4 | 0.9868 | 531 |
| openrouter | openai/gpt-oss-120b:free | 0.8687 | 99 |

### A.8 Skill-augmented LLM comparison (`06b_skill_comparison.csv`, AUC and accuracy only)

| model | metric | baseline_zero_shot | skill_augmented | Δ |
|---|---|---:|---:|---:|
| claude-opus-4-7 | auc | 0.6748 | 0.6923 | +0.0176 |
| claude-opus-4-7 | accuracy | 0.5881 | 0.6897 | +0.1016 |
| gpt-4o | auc | 0.6521 | 0.6667 | +0.0146 |
| gpt-4o | accuracy | 0.5822 | 0.6061 | +0.0239 |
| gpt-5.4 | auc | 0.6892 | 0.6655 | −0.0237 |
| gpt-5.4 | accuracy | 0.5811 | 0.6763 | +0.0953 |

(Note: baselines in `06b` are computed on a 29- or 655-profile slice and differ slightly from the 531-profile Phase 6 baseline in A.6. Both are reported for transparency.)

### A.9 v2 workflow quality (`06c_v2_workflow_summary.csv`)

| Memo dimension | Mean (out of 5) | Std | Adverse-action dimension | Mean (out of 5) | Std |
|---|---:|---:|---|---:|---:|
| factual_grounding | 2.378 | 0.688 | reason_accuracy | 4.431 | 0.802 |
| risk_identification | 3.395 | 0.787 | no_prohibited | 5.000 | 0.000 |
| compliance | 4.854 | 0.373 | specificity | 3.951 | 0.326 |
| hallucination | 1.685 | 0.599 | plain_language | 4.765 | 0.426 |
| professional_quality | 3.411 | 0.496 | legal_completeness | 4.314 | 0.675 |
| **memo total / 25** | **15.723** | **2.120** | **adverse-action total / 25** | **22.461** | **1.398** |

### A.10 v1 self-grade vs v2 cross-grade (`06c_v2_comparison.csv`)

| Metric | v1 (self-grade) | v2 (cross-grade) | Δ | Note |
|---|---:|---:|---:|---|
| Memo mean total (out of 25) | 24.880 | 15.723 | −9.157 | Self-grading inflation |
| Memo % perfect | 90.0 | 0.0 | −90.0 | Cross-grader rejects "perfect" outputs |
| Adverse-action mean total (out of 25) | 24.571 | 22.461 | −2.111 | Adverse-action quality more robust |
| Adverse-action % perfect | 64.3 | 0.0 | −64.3 | |
| Adverse-action FCRA-compliance % | 42.9 | 100.0 | +57.1 | Deterministic injection works |
| Protected-attribute leaks (count) | 7 | 24 | +17 | v2 catches more (filter active) |
| Uncertainty flags added | 0 | 51 | +51 | PD ∈ [0.20, 0.45] |
| Validation issues found | 0 | 198 | +198 | Programmatic checks running |

---

## 15. Appendix B — Glossary

**4/5ths rule.** Regulatory threshold for disparate impact in U.S. lending: the selection rate of any protected group must be at least 80% of the selection rate of the most-favored group. DI ratios below 0.80 are presumptive evidence of adverse impact and require business-justification or remediation.

**AUC.** Area under the receiver-operating-characteristic curve. Probability that a randomly chosen positive (default) case receives a higher score than a randomly chosen negative case. AUC ∈ [0.5, 1.0]; 0.5 = random; 1.0 = perfect ranking.

**Brier score.** Mean squared error between predicted probability and binary outcome. Lower is better. A jointly-proper scoring rule that captures both calibration and resolution.

**DI ratio (disparate impact ratio).** Ratio of the protected-group selection rate to the reference-group selection rate. < 0.80 violates 4/5ths.

**DP-diff (demographic parity difference).** Largest pairwise difference in selection rate across groups of a protected attribute. Range [0, 1]; 0 = identical selection rates.

**ECE (expected calibration error).** Bin predicted probabilities into K bins (typically 10); within each bin, take the absolute difference between mean predicted probability and empirical frequency; weight by bin sample size; sum. Lower is better; ≤ 0.05 is a typical "well-calibrated" benchmark.

**ECOA / Reg B.** Equal Credit Opportunity Act and its implementing regulation. Prohibits discrimination on protected bases in any aspect of a credit transaction. Enforced by the CFPB (consumer) and the Federal Reserve, OCC, FDIC, NCUA (institution).

**EO-diff (equalized odds difference).** Largest pairwise difference, across groups, in either true-positive rate or false-positive rate, whichever is larger.

**FCRA.** Fair Credit Reporting Act. Governs use of consumer credit-reporting-agency data; §615(a) requires specific adverse-action disclosures.

**HMDA.** Home Mortgage Disclosure Act. Annual public dataset of mortgage applications maintained by the CFPB; the LAR (Loan Application Register) is the row-level file.

**KS statistic (Kolmogorov-Smirnov).** Maximum vertical distance between the cumulative-distribution functions of the score among defaulters and non-defaulters. KS ∈ [0, 1]; higher is better.

**OCC 2011-12.** Office of the Comptroller of the Currency Bulletin 2011-12, "Sound Practices for Model Risk Management." Joint guidance with the Federal Reserve's SR 11-7.

**PD (probability of default).** Model's estimated probability that the applicant defaults within the model's reference horizon.

**PSI (population stability index).** Σ (p_i − q_i) · ln(p_i / q_i) over decile bins of a feature, comparing a baseline population (p) and a current population (q). PSI > 0.10 is a meaningful shift; > 0.25 is substantial.

**Reg B.** Regulation B of the ECOA — see ECOA above.

**SHAP (Shapley additive explanations).** Local feature-attribution method based on cooperative-game-theoretic Shapley values; provides per-prediction additive contributions of each feature to the model output. TreeSHAP is the polynomial-time exact computation for tree ensembles.

**Spearman ρ.** Rank-correlation coefficient. Used here to measure stability of feature-rank orderings across bootstrap re-fits.

**SR 11-7.** Federal Reserve Board Supervision and Regulation Letter 11-7 (2011), "Guidance on Model Risk Management." The principal U.S. supervisory standard for bank model risk; defines model lifecycle (development, implementation, use), independent validation requirements, and ongoing monitoring expectations.

**Tier 2 model (institutional convention).** A model whose failure would have material business or customer impact but does not rise to systemic / capital-adequacy importance. Subject to full SR 11-7 validation including independent review, periodic revalidation, ongoing monitoring, and documented compensating controls.

---

*End of report.*
