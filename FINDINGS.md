# Key Findings

**4 LLMs × 500 profiles × 2 runs = 4,000+ API calls. Here's what we learned.**

---

## 1. Chatbots can't predict credit risk

| Model | AUC | Interpretation |
|---|---|---|
| XGBoost | 0.774 | Strong discriminatory power |
| GPT-4o (zero-shot) | 0.579 | Barely above coin flip (0.50) |
| Claude 3.5 Sonnet | 0.564 | Worse than GPT-4o |
| GPT-4o (skill prompt) | 0.667 | Better, still 11 AUC points behind |
| **This workflow** | **0.774** | XGBoost decides, LLM communicates |

LLMs aren't bad at language — they're bad at structured tabular prediction. A 20-point AUC gap means real money: thousands of defaults approved, thousands of good borrowers denied.

---

## 2. "Selective fairness" — fair where trained, biased where not

Every LLM we tested passes gender fairness (disparate impact > 0.80). Every single one fails on education and age.

| Protected attribute | GPT-4o DI | Claude DI | Threshold |
|---|---|---|---|
| Gender | 0.89 ✅ | 0.92 ✅ | 0.80 |
| Education | 0.49 ❌ | 0.51 ❌ | 0.80 |
| Age | 0.63 ❌ | 0.58 ❌ | 0.80 |

Why? Safety training covers gender extensively. Education bias ("if they didn't go to college, they're risky") isn't in the RLHF training data. The model is fair where OpenAI/Anthropic trained it to be fair, and biased everywhere else.

---

## 3. Same prompt, opposite effects

We gave GPT-4o and Claude the same "be fair, don't use protected attributes" skill prompt.

- GPT-4o education DI: 0.39 → 0.70 (improved)
- Claude education DI: 0.51 → 0.37 (got *worse*)

Same instruction, opposite outcome. Prompt engineering is not a reliable fairness mechanism.

---

## 4. Real US mortgage data shows racial disparate impact

We trained XGBoost on 100K HMDA mortgage records (NY 2022) and ran our fairness pipeline:

| Attribute | Disparate Impact | Passes? |
|---|---|---|
| Race | 0.34 | ❌ (threshold: 0.80) |
| Ethnicity | 0.78 | ❌ |
| Sex | 0.91 | ✅ |

This isn't a bug in our pipeline — it's the pipeline catching real bias in the training data. The system works exactly as designed: flag it, don't hide it.

---

## 5. Self-grading is unreliable

We asked GPT-4o to grade its own credit memos on a 25-point scale, then had Claude grade the same memos independently.

- GPT-4o self-grade: 21.3 / 25
- Claude cross-grade: 12.1 / 25
- **Gap: 9.2 points** (37% inflation)

Never let a model evaluate its own output.

---

## 6. The workflow catches what chatbots miss

With all safety layers active on 500 profiles:

| Check | Result |
|---|---|
| FCRA compliance (denial letters) | 102/102 (100%) |
| Protected attribute leaks caught | 24/500 profiles |
| Borderline cases flagged for human review | 51/500 (10.2%) |
| SHAP adherence (only citing real factors) | 97% |
| Total LLM cost | $0.17 |

---

## The takeaway

Don't make chatbots smarter at credit. Make them do what they're actually good at — communicating decisions that a proper model already made, inside a cage of deterministic safety checks.

The credit industry doesn't need to choose between AI and traditional models. It needs to assign each tool to what it does best.

---

*Full methodology: [docs/research_paper.md](docs/research_paper.md)*  
*Reproduce the benchmark: [benchmarks/](benchmarks/)*  
*Author: Maneek Mohan — [github.com/maneek21](https://github.com/maneek21)*
