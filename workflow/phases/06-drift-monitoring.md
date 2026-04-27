# Phase 06 — Drift Monitoring

**Task type:** Deterministic (Python scripts only — never LLM)
**When to use:** The user wants to check whether a deployed credit model's
performance has degraded over time, or whether the input data distribution
has shifted.

## What this phase does

Measures temporal drift in model performance and data distributions by
training on historical data and testing on subsequent periods. Detects
both concept drift (the relationship between features and default changes)
and data drift (the feature distributions shift).

**This phase MUST be run with deterministic scripts.** Drift metrics require
exact computation over full datasets.

## Why drift matters in credit

Credit risk models degrade over time due to:

1. **Economic shocks** — COVID-19 caused a massive distribution shift in
   2020 (our HMDA analysis shows approval rates swung from 78.8% to 81.9%
   to 76.8% across 2020–2022)
2. **Population shifts** — applicant demographics change over time
3. **Policy changes** — lending standards tighten or loosen
4. **Concept drift** — the relationship between features and default
   evolves (e.g., post-pandemic, prior delinquency may be less predictive)

SR 11-7 (Federal Reserve guidance on model risk management) requires
ongoing monitoring of model performance — this phase implements that.

## Running drift analysis

```bash
python scripts/compute_drift.py --train-years 2018,2019 --test-years 2020,2021,2022
```

### What the script does

1. Trains LR + XGBoost + MLP on the training period (e.g., HMDA 2018–2019)
2. Tests on each subsequent year independently
3. Computes AUC, KS, Brier, ECE for each model × year combination
4. Computes approval-rate drift per year
5. Optionally: computes Population Stability Index (PSI) for each feature

## Metrics

### Performance drift

| Metric | What it detects | Alert threshold |
|---|---|---|
| AUC degradation | Loss of discriminatory power | Δ AUC > 0.05 from training |
| KS degradation | Loss of separation | Δ KS > 0.05 from training |
| Brier increase | Calibration degradation | Δ Brier > 0.03 from training |
| Approval rate shift | Decision boundary sensitivity | Δ approval > 5pp |

### Data drift (feature-level)

| Metric | What it detects | Alert threshold |
|---|---|---|
| PSI (Population Stability Index) | Overall distribution shift | PSI > 0.25 = major shift |
| Mean shift | Location parameter change | > 1 std dev from training mean |
| Variance ratio | Spread change | > 2× or < 0.5× training variance |

### PSI interpretation

```
PSI < 0.10  → No significant shift
PSI 0.10–0.25 → Moderate shift — investigate
PSI > 0.25  → Major shift — model retraining recommended
```

## Expected findings

From our HMDA 2018–2022 drift analysis:

| Year | XGBoost AUC | LR AUC | MLP AUC | Approval Rate |
|---|---|---|---|---|
| 2018–2019 (train) | baseline | baseline | baseline | baseline |
| 2020 | stable/improving | stable | stable | 78.8% |
| 2021 | stable/improving | stable | stable | 81.9% (+3.1pp) |
| 2022 | stable | stable | stable | 76.8% (-5.1pp) |

Key finding: AUC was surprisingly stable across the COVID period — the
models' ranking ability held up even as approval rates shifted. This
suggests concept drift was less severe than data drift for this population.

## Output files

| File | Contents |
|---|---|
| `results/drift_metrics.csv` | AUC, KS, Brier, approval rate per model × year |
| `results/drift.png` | Line chart of AUC over time per model |
| `results/drift_approval_rate.png` | Approval rate over time |
| `results/drift_psi.csv` | PSI per feature per year (if computed) |

## Monitoring schedule

For production models, SR 11-7 suggests:

- **Monthly:** Approval rate monitoring, PSI on top 10 features
- **Quarterly:** Full performance metrics on recent data
- **Annually:** Full model revalidation with updated training data
- **Event-driven:** After any economic shock, policy change, or
  regulatory update, run a full drift analysis immediately

## Integration

- **Phase 02** provides the trained models to evaluate
- **Phase 04** should be re-run if drift is detected — fairness metrics
  may shift with the population
- Drift findings inform **Phase 03** memo generation — a model with
  detected drift should include a caveat in the credit memo
