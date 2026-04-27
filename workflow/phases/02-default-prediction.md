# Phase 02 — Default Prediction

**Task type:** Deterministic (classical ML) + optional LLM evaluation
**When to use:** The user wants to train, evaluate, or compare default
prediction models on structured borrower data.

## What this phase does

Trains and evaluates classical ML models (Logistic Regression, XGBoost, MLP)
on structured credit data, and optionally benchmarks frontier LLMs on the
same task. This phase produces the core accuracy metrics that every other
phase builds on.

## Classical model pipeline

### Step 1 — Prepare data

```bash
python scripts/prepare_data.py --dataset uci
```

Loads the dataset, cleans undocumented codes, applies stratified 70/15/15
train/val/test split with `random_state=42`. See `references/dataset-schemas.md`
for column mappings and cleaning rules.

Supported datasets:
- `uci` — UCI Default of Credit Card Clients (30K rows, Taiwan credit cards)
- `hmda` — HMDA LAR data (requires year parameter: `--year 2022`)
- `lendingclub` — Lending Club (optional, requires Kaggle download)

### Step 2 — Train classical models

```bash
python scripts/train_classical.py
```

Trains three models with identical preprocessing (StandardScaler for numeric,
OneHotEncoder for categorical):

| Model | Library | Key hyperparameters |
|---|---|---|
| Logistic Regression | `sklearn.linear_model` | `C=1.0`, `max_iter=1000`, `random_state=42` |
| XGBoost | `xgboost` | `n_estimators=300`, `max_depth=5`, `learning_rate=0.1`, `random_state=42` |
| MLP | `sklearn.neural_network` | `hidden_layer_sizes=(128, 64)`, `max_iter=500`, `random_state=42` |

Saves fitted pipelines to `data/processed/models.joblib`.

### Step 3 — Compute accuracy metrics

```bash
python scripts/compute_accuracy.py
```

Computes on the held-out test set:

| Metric | What it measures | See |
|---|---|---|
| AUC | Discrimination — can the model rank defaulters above non-defaulters? | `references/methodology.md` |
| KS statistic | Maximum separation between cumulative default/non-default distributions | `references/methodology.md` |
| Brier score | Calibration — how close are predicted probabilities to actual outcomes? | `references/methodology.md` |
| ECE | Expected calibration error — binned version of Brier | `references/methodology.md` |

Outputs: `results/accuracy.csv`, `results/calibration.png`, `results/roc_curves.png`

### Expected baseline results (UCI dataset)

From our empirical evaluation and published benchmarks:

| Model | AUC (ours) | AUC (Japinye 2025) | Notes |
|---|---|---|---|
| Logistic Regression | 0.713 | — | Regulatory baseline; fully interpretable |
| XGBoost | 0.774 | 0.892–0.923 | Japinye uses 3 datasets with tuning |
| MLP | 0.766 | — | Simple neural net for spectrum |

## LLM evaluation (optional)

To benchmark frontier LLMs on the same task:

```bash
python scripts/llm_credit_eval.py --provider anthropic --model claude-opus-4-7
python scripts/llm_credit_eval.py --provider openai --model gpt-4o
```

### How it works

1. Samples 200 profiles from the test set (stratified by default label).
2. Formats each profile as a structured text description.
3. Sends to the LLM with the prompt template from
   `assets/prompt_templates/zero_shot_underwriting.txt`.
4. Parses the JSON response for decision (APPROVE/DENY), confidence, and reasoning.
5. Logs every call to `results/api_log.csv` (model, tokens, cost).
6. Saves decisions to `results/llm_decisions.csv`.

### Prompt template selection

| Template | Use when |
|---|---|
| `zero_shot_underwriting.txt` | Baseline evaluation — no examples, minimal instruction |
| `few_shot_underwriting.txt` | Enhanced evaluation — includes 3 worked examples with reasoning |

The zero-shot template is intentionally minimal to establish a fair baseline.
The few-shot template adds population statistics, feature hierarchy, and
calibration guidance — derived from failure-mode analysis of the zero-shot results.

### Known LLM limitations on this task

Empirical findings from our evaluation of 4 LLMs on 200 UCI profiles:

| Failure mode | Observed impact | Evidence |
|---|---|---|
| Over-denial | 55–70% deny rate vs 22% true default rate | All 4 LLMs deny at 2.5–3× the base rate |
| Education bias | DI ratio 0.49–0.64 (violates four-fifths rule) | LLMs treat education as a primary factor despite near-zero predictive value |
| Overconfidence | Wrong answers get 0.81–0.87 confidence | Only 1–4 point gap between correct and incorrect confidence |
| No distributional context | LLMs cannot judge "high" vs "low" values | No population reference → conservative default |
| AUC ceiling | 0.55–0.63 across all models and prompt strategies | Fundamental limitation of text-based tabular reasoning |

These findings motivate the workflow architecture: use classical models for
prediction (Phase 02), LLMs for explanation and communication (Phases 03, 05).

## Output files

| File | Contents |
|---|---|
| `results/accuracy.csv` | AUC, KS, Brier, ECE per model |
| `results/roc_curves.png` | ROC curves for all models |
| `results/calibration.png` | Calibration plots for all models |
| `results/llm_decisions.csv` | LLM decisions with confidence and reasoning |
| `results/llm_metrics.csv` | LLM accuracy metrics for comparison |
| `results/api_log.csv` | API call log with tokens and cost |
