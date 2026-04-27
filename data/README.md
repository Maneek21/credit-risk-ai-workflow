# Data — Built-in Adapters & Download Utilities

This directory holds two things:

1. **Adapters** (`adapters/`) — `DatasetAdapter` implementations that
   know how to download, load, and preprocess each public dataset.
2. **Local raw / processed files** (`raw/`, `processed/`) — created on
   first run when you call `python data/download.py`. Not checked in.

The training SDK (`workflow.training`) consumes these adapters to turn
a YAML config into a trained model, fairness report, and SHAP summary.

---

## Datasets

| Adapter | Source | Region | Rows (approx.) | Target | Protected attributes |
|---|---|---|---|---|---|
| `UCIAdapter`     | [UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | Taiwan (TW) | 30,000 | `default` (next month) | `SEX`, `AGE_BUCKET` (engineered) |
| `HMDAAdapter`    | [CFPB HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/) (default slice: NY 2022) | United States | ~100,000 sampled | `denied` (action_taken=3 vs 1) | `derived_race`, `derived_ethnicity`, `derived_sex`, `applicant_age` |
| `BondoraAdapter` | [Bondora Public Reports](https://www.bondora.com/en/public-reports) / [Kaggle mirror](https://www.kaggle.com/datasets/sid321axn/bondora-peer-to-peer-lending-loan-data) | Estonia / EU | ~100,000 closed loans | `default` (Status=Defaulted) | `Gender`, `Country`, `Age` |

Protected columns are **never** model features. The adapter contract
(`DatasetMetadata.validate()`) raises if `feature_columns` and
`protected_columns` overlap.

---

## Quick start

Download a dataset:

```bash
# UCI (small, no API key needed)
python data/download.py --dataset uci --dest data/raw/

# HMDA (NY 2022 only, ~30 MB)
python data/download.py --dataset hmda --dest data/raw/

# Bondora (requires Kaggle CLI authenticated, OR manual download)
pip install kaggle  # one-time
kaggle datasets download sid321axn/bondora-peer-to-peer-lending-loan-data \
    -p data/raw/ --unzip
```

Then train:

```bash
python -m workflow.training train --config configs/uci_us.yaml
python -m workflow.training train --config configs/hmda_us.yaml
python -m workflow.training train --config configs/bondora_eu.yaml
```

Each run writes a `.joblib` model and a `.json` metrics file to the
config's `output_dir`.

---

## Bringing your own dataset

Subclass `DatasetAdapter`, list your features and protected columns
in `metadata()`, implement `load()`. See
`examples/bring_your_own_data/` for a complete template.

```yaml
# my_bank.yaml
dataset: examples.bring_your_own_data.custom_adapter_template:MyBankAdapter
data_path: /path/to/portfolio.csv
jurisdiction: IN
```

---

## Caveats

- **HMDA download** uses the CFPB streaming-CSV endpoint. The full
  national LAR is hundreds of millions of rows; the adapter pulls
  one state-year (default: NY 2022) and stratified-samples to 100K.
- **Bondora download** depends on Kaggle credentials; if `kaggle`
  isn't on `PATH` the adapter raises a `RuntimeError` with manual-
  download instructions. We don't bundle the data because of licence
  terms.
- **UCI shipped with the repo.** A 100-row sample lives at
  `benchmarks/data/uci_sample_100rows.csv` for offline tests. The
  full file is downloaded on first use.
