# Dataset Schemas Reference

## UCI Default of Credit Card Clients

**Source:** https://archive.ics.uci.edu/dataset/350
**Size:** 30,000 rows × 24 features + 1 target
**Population:** Taiwan credit card holders, 2005
**Default rate:** 22.12%

### Columns

| Column | Type | Values | Notes |
|---|---|---|---|
| LIMIT_BAL | Numeric | NT$ | Credit limit. Mean 167,484; Median 140,000 |
| SEX | Categorical | 1=Male, 2=Female | Protected attribute |
| EDUCATION | Categorical | 1=Graduate, 2=University, 3=High School, 4=Other | Raw data includes 0, 5, 6 — collapse to 4 (Other) |
| MARRIAGE | Categorical | 1=Married, 2=Single, 3=Other | Raw data includes 0 — collapse to 3 (Other) |
| AGE | Numeric | Years | Range 21–79. Mean 35.5, Median 34 |
| PAY_0 | Numeric | -2 to 8 | Most recent repayment status (Sep). -1=paid in full, 0=revolving, 1+=months late |
| PAY_2–PAY_6 | Numeric | -2 to 8 | Repayment status for Aug through Apr |
| BILL_AMT1–6 | Numeric | NT$ | Bill statement amount, Sep through Apr |
| PAY_AMT1–6 | Numeric | NT$ | Payment amount, Sep through Apr |
| default payment next month | Binary | 0=No, 1=Yes | **Target variable** |

### Data cleaning rules

```python
# Collapse undocumented education codes to "Other" (4)
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

# Collapse undocumented marriage code to "Other" (3)
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
```

### Feature engineering for LLM evaluation

When presenting profiles to LLMs, compute summary features:

```python
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
bill_cols = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
pay_amt_cols = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

profile["pay_avg"] = df[pay_cols].mean(axis=1)
profile["bill_avg"] = df[bill_cols].mean(axis=1)
profile["pay_amt_avg"] = df[pay_amt_cols].mean(axis=1)
```

### Train/Val/Test split

Stratified 70/15/15 with `random_state=42`:

```python
from sklearn.model_selection import train_test_split

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
val_frac = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=42
)
```

---

## HMDA (Home Mortgage Disclosure Act) LAR Data

**Source:** https://ffiec.cfpb.gov/data-browser/data/
**API:** CFPB v2 Data Browser API
**Years available:** 2018–2023 (v2 API does not serve 2017)
**Size:** Varies by state/year. NY ~400K rows/year after filtering.

### Key columns for this workflow

| Column | Type | Values | Notes |
|---|---|---|---|
| action_taken | Categorical | 1=Originated, 3=Denied | **Target**: 1→APPROVE, 3→DENY. Filter to only 1 and 3. |
| loan_amount | Numeric | Dollars | |
| income | Numeric | $1000s | Applicant gross annual income |
| debt_to_income_ratio | Categorical | Ranges | Bucketed: "<20%", "20%-<30%", etc. |
| derived_race | Categorical | White, Black, Asian, etc. | Derived by CFPB from race fields |
| derived_ethnicity | Categorical | Hispanic/Latino, Not | |
| derived_sex | Categorical | Male, Female, Joint | |
| applicant_age | Categorical | Ranges | "<25", "25-34", "35-44", etc. |
| loan_purpose | Categorical | 1=Purchase, 31=Refinance, 32=Cash-out | |
| property_type | Categorical | 1=Conventional, 2=FHA, etc. | |
| county_code | Categorical | FIPS code | |

### Filtering rules

```python
# Keep only approve/deny (drop withdrawn, file-closed, preapproval, etc.)
df = df[df["action_taken"].isin([1, 3])]

# Create binary target
df["approved"] = (df["action_taken"] == 1).astype(int)

# Remove "joint" sex for clean male/female comparison
df_sex = df[df["derived_sex"].isin(["Male", "Female"])]
```

### HMDA API query

```
GET https://ffiec.cfpb.gov/v2/data-browser-api/view/csv
?states=NY&years={year}&actions_taken=1,3
```

---

## Lending Club (Optional)

**Source:** https://www.kaggle.com/datasets/wordsforthewise/lending-club
**Size:** ~2.2M rows (2007–2018)
**Note:** Requires Kaggle account. Skip if friction — HMDA covers drift.

### Key columns

| Column | Type | Target mapping |
|---|---|---|
| loan_status | Categorical | "Fully Paid"→0, "Charged Off"→1, "Default"→1 |
| loan_amnt | Numeric | Loan amount |
| int_rate | Numeric | Interest rate (%) |
| annual_inc | Numeric | Self-reported income |
| dti | Numeric | Debt-to-income ratio |
| fico_range_low/high | Numeric | FICO score range |
| emp_length | Categorical | Employment length |
| home_ownership | Categorical | RENT, OWN, MORTGAGE |
| purpose | Categorical | Loan purpose |
| addr_state | Categorical | State |

### Cleaning rules

```python
# Keep only terminal statuses
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])]

# Binary target
df["default"] = df["loan_status"].isin(["Charged Off", "Default"]).astype(int)

# Remove "%" from int_rate if string
df["int_rate"] = df["int_rate"].str.replace("%", "").astype(float)
```
