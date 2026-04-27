# Bring Your Own Data

The training SDK is designed so that a bank engineer can plug in an
internal loan portfolio without forking the repo. The contract is
small: a `DatasetAdapter` subclass that knows how to load the data,
plus a YAML config pointing the trainer at it.

This guide takes you from "I have a CSV" to "I have a registered model
and a fairness report" in five steps.

---

## 1. Copy the template

```bash
cp examples/bring_your_own_data/custom_adapter_template.py \
   my_adapter.py
cp examples/bring_your_own_data/sample_config.yaml \
   my_bank.yaml
```

The template's class is called `MyBankAdapter`. Rename it to whatever
you like — just remember to update the dotted-path reference in your
YAML.

---

## 2. Fill in the schema

Open `my_adapter.py` and edit the three sections marked `TODO` in
`metadata()`:

```python
return DatasetMetadata(
    name="acme_personal_loans",
    region="IN",
    target_column="default_flag",
    feature_columns=[
        "loan_amount", "term_months", "interest_rate",
        "annual_income", "credit_score", "dti_ratio",
        "loan_purpose", "employment_status",
    ],
    categorical_columns=["loan_purpose", "employment_status"],
    protected_columns={
        "applicant_gender": "Self-reported gender",
        "applicant_age_band": "18-25 / 26-45 / 46-60 / 60+",
        "applicant_caste": "Constitution Art 15 protected (India only)",
    },
    description="ACME Bank personal loans, FY 2024-25, 240K applications",
)
```

> **Critical rule.** `feature_columns` and `protected_columns` MUST
> be disjoint sets. `DatasetMetadata.validate()` raises if they
> overlap. Protected attributes are evaluated for fairness only —
> they never reach the model.

---

## 3. Implement `load()`

`load()` returns a DataFrame containing every column you listed in
`metadata()` — features, target, and protected attributes. The
training pipeline validates this; missing columns are a hard error.

The template includes commented examples for the typical preprocessing
moves: dropping rows with missing critical fields, capping right-tail
outliers, and engineering protected-attribute bins (age band) without
leaking the raw value into the model.

```python
def load(self, path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["origination_date"])
    df = df.dropna(subset=["loan_amount", "annual_income", "credit_score"])
    for col in ("loan_amount", "annual_income"):
        df[col] = df[col].clip(upper=df[col].quantile(0.99))
    df["applicant_age_band"] = pd.cut(
        df["age"], bins=[0, 25, 45, 60, 200],
        labels=["18-25", "26-45", "46-60", "60+"],
    ).astype(str)
    return df
```

---

## 4. Point the YAML at your adapter

In `my_bank.yaml`:

```yaml
dataset: examples.bring_your_own_data.custom_adapter_template:MyBankAdapter
data_path: /opt/data/acme/personal_loans_2024.csv
jurisdiction: IN
```

The dotted path can reference any importable Python module. Use either
`pkg.mod:Class` or `pkg.mod.Class` — the resolver accepts both.

---

## 5. Train

```bash
python -m workflow.training train --config my_bank.yaml
```

The trainer downloads (if needed), loads, splits 80/20 stratified,
fits an XGBoost pipeline, and writes:

```
output/my_bank/
├── acme_personal_loans_xgboost_v1_0_0.joblib   # the model
└── acme_personal_loans_xgboost_v1_0_0.json     # metrics + fairness + SHAP
```

---

## 6. Plug into the workflow

Use the trained model with the full credit pipeline, configured for
your jurisdiction:

```python
from workflow import CreditWorkflow, India

wf = CreditWorkflow(
    model_path="output/my_bank/acme_personal_loans_xgboost_v1_0_0.joblib",
    model_version="1.0.0",
    jurisdiction=India(),
    llm_provider="openai",
    llm_model="gpt-4o-mini",
)

result = wf.process_application(applicant_dict)
print(result.decision, result.probability)
print(result.adverse_action)   # ends with the India() disclosure block
```

---

## What this is, and what it isn't

This SDK is an **onboarding accelerator**, not a model-development
framework. It standardises the boilerplate (download → split → train →
evaluate → save → metrics JSON) so a bank's first model run goes from
"days of plumbing" to "one command". It does not replace your MLOps
stack — for production, you'll integrate the resulting `.joblib` into
your existing model registry, CI/CD, and monitoring.

**No PII is sent to any external service** by the training SDK itself.
The pipeline runs entirely on your infrastructure. PII protection
matters at *inference* time when the LLM communication layer is
active — see `workflow/pii.py`.
