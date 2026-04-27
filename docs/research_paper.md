# AI in Credit Risk Assessment — Report Summary

**Author:** Maneek Mohan  
**Date:** April 2026

---

## Research Question

Should AI chatbots (like ChatGPT and Claude) make credit decisions, or should they be used for something else in the lending pipeline?

---

## Key Findings

### 1. Traditional models still win on prediction

XGBoost achieved an AUC of 0.774 on the UCI credit dataset — roughly 20 points above the best chatbot (0.580). On structured tabular data, statistical learning trained on actual defaults beats general-purpose AI knowledge. This aligns with published research (CALM benchmark, 2023).

### 2. Chatbots have hidden biases ("Selective Fairness")

All four chatbots (Claude Opus 4.7, GPT-4o, GPT-5.4, GPT-OSS-120B) passed gender fairness tests but failed on education and age. Graduate-degree holders were approved at roughly double the rate of others (DI 0.49–0.64, all failing the four-fifths rule). We call this "selective fairness" — fair on the dimensions AI safety training covers, biased on everything else.

### 3. Better instructions don't reliably fix bias

Giving chatbots a detailed "skill" (with base rates, factor hierarchy, and bias warnings) improved GPT-5.4's education fairness to passing (DI 0.81) but worsened Claude's (DI 0.51 → 0.37). The same instructions produced opposite effects on different models. Prompt engineering is not a compliance strategy.

### 4. Architecture beats prompts

A hybrid workflow — XGBoost predicts, GPT-4o communicates, deterministic code enforces compliance — achieved:

- **100% FCRA compliance** (vs. 43% from chatbots alone)
- **0 protected attribute leaks** in final output (24 caught and removed by filters)
- **15.7/25 memo quality** (solid first draft, needs senior review)
- **22.5/25 adverse action quality** (near-perfect denial letters)

### 5. AI self-grading is unreliable

When GPT-4o graded its own memos, it scored 24.9/25. When GPT-5.4 graded the same output: 15.7/25. A 9.2-point gap on a 25-point scale. Any study using AI self-evaluation should be treated with scepticism.

---

## Methodology at a Glance

| Component | Detail |
|---|---|
| Datasets | UCI Credit (30K rows, Taiwan) + HMDA Mortgages (500K+ rows, NY 2018–2022) |
| Classical models | Logistic Regression, XGBoost, MLP |
| Chatbots tested | Claude Opus 4.7, GPT-4o, GPT-5.4, GPT-OSS-120B |
| LLM sample | 500 profiles × 2 runs × 4 models |
| Workflow test | 500 profiles through full pipeline (XGBoost → GPT-4o → safety layers) |
| Total API spend | $11.87 |
| Fairness standard | Four-fifths rule (DI < 0.80 = regulatory red flag) |

---

## The Workflow Architecture

```
Layer 1: XGBoost makes the decision
    → Probability of default, trained on actual data

Layer 2: GPT-4o communicates the decision  
    → Credit memos and denial letters grounded in SHAP values

Layer 3: Deterministic code enforces compliance
    → FCRA injection, protected attribute filter, uncertainty flags, cross-model grading
```

**Five safety layers:**
1. SHAP filter (chatbot can only cite real factors)
2. FCRA injection (legal disclosures added by code, not AI)
3. Uncertainty flagging (borderline cases → human review)
4. Protected attribute filter (blocks mentions of race, gender, etc.)
5. Cross-model grading (independent AI quality check)

---

## Recommendations

1. **Do not** use chatbots for credit decisions (0.55–0.58 AUC is unacceptable)
2. **Do** use chatbots for communication — memos, denial letters, document extraction
3. **Never** deploy without deterministic safety layers
4. **Use** cross-model grading, not self-grading
5. **Monitor** continuously (SR 11-7 compliance)

---

## Limitations

- UCI data is from Taiwan (2005); HMDA is US mortgages — may not generalise
- Default hyperparameters used for reproducibility (production would tune higher)
- Claude skill-augmented results based on only 29 profiles (API credits exhausted)
- Cross-grader was GPT-5.4 (same provider as memo author) — Claude grader would be more independent
- Results reflect April 2026 model versions

---

## Files

| Item | Location |
|---|---|
| Full report | `report/final_report_v8.docx` |
| Supporting data (20 CSVs) | `report/supporting_data/` |
| Analysis code | `src/` |
| Raw data | `data/raw/` |
| All result artifacts | `results/` |

---

*This study confirms what Moody's, Rocket Mortgage, and HSBC already practice: the chatbot communicates but doesn't decide.*
