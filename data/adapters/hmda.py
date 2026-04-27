"""HMDA Mortgage Data adapter.

Source: CFPB HMDA Loan Application Register (LAR), public bulk export.
        https://ffiec.cfpb.gov/data-browser/

This adapter targets a single state-year slice (default: New York 2022)
that the CFPB exposes as a streaming CSV via the public data-browser
API. The full national LAR is hundreds of millions of rows; one state
in one year is small enough to download in a few minutes and large
enough to expose meaningful protected-attribute distributions.

Why HMDA matters for this repo: it is the only public US lending
dataset with **real protected attributes** — derived race, ethnicity,
sex, and applicant age — and is therefore the canonical fairness
benchmark for credit risk models in the United States.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from workflow.training.datasets import DatasetAdapter, DatasetMetadata


HMDA_BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
DEFAULT_STATE = "NY"
DEFAULT_YEAR = 2022
SAMPLE_SIZE = 100_000

#: Action-taken codes we keep — 1 = originated (approved), 3 = denied.
#: Other codes (withdrawn, file closed for incompleteness, preapproval-
#: denied, ...) are excluded so the binary target is well-defined.
ACTION_KEEP = {1, 3}

#: Mapping of debt-to-income range strings to midpoint floats. Public
#: HMDA data publishes DTI as binned ranges to limit re-identification
#: risk; we map each bucket to its midpoint as the model input.
DTI_MIDPOINTS = {
    "<20%": 10.0,
    "20%-<30%": 25.0,
    "30%-<36%": 33.0,
    "36": 36.0,
    "37": 37.0,
    "38": 38.0,
    "39": 39.0,
    "40": 40.0,
    "41": 41.0,
    "42": 42.0,
    "43": 43.0,
    "44": 44.0,
    "45": 45.0,
    "46": 46.0,
    "47": 47.0,
    "48": 48.0,
    "49": 49.0,
    "50%-60%": 55.0,
    ">60%": 65.0,
}

_NUMERIC_DROP_TOKENS = {"Exempt", "NA", ""}


class HMDAAdapter(DatasetAdapter):
    """HMDA LAR slice (default: NY 2022) sampled to 100K rows."""

    def __init__(
        self,
        state: str = DEFAULT_STATE,
        year: int = DEFAULT_YEAR,
        sample_size: int = SAMPLE_SIZE,
    ) -> None:
        self.state = state
        self.year = year
        self.sample_size = sample_size

    def metadata(self) -> DatasetMetadata:
        # Column names match the public HMDA LAR schema (CFPB 2022).
        # Notable substitutions vs. the spec draft:
        #   * combined_loan_to_value_ratio -> loan_to_value_ratio
        #     (the "combined" variant is not exposed in the public CSV)
        #   * property_type                -> derived_dwelling_category
        #     (the public CSV exposes the derived category, not the raw code)
        feature_columns = [
            "loan_amount",
            "loan_type",
            "loan_purpose",
            "occupancy_type",
            "derived_dwelling_category",
            "income",
            "debt_to_income_ratio",
            "loan_to_value_ratio",
            "loan_term",
            "balloon_payment",
            "interest_only_payment",
            "negative_amortization",
        ]
        return DatasetMetadata(
            name="hmda",
            region="US",
            target_column="denied",
            feature_columns=feature_columns,
            categorical_columns=[
                "loan_type", "loan_purpose", "occupancy_type",
                "derived_dwelling_category",
                "balloon_payment", "interest_only_payment", "negative_amortization",
            ],
            protected_columns={
                "derived_race": "CFPB-derived applicant race",
                "derived_ethnicity": "CFPB-derived applicant ethnicity",
                "derived_sex": "CFPB-derived applicant sex",
                "applicant_age": "Applicant age bucket (HMDA codebook)",
            },
            positive_label=1,
            description=(
                f"HMDA LAR — {self.state} {self.year} — sampled to "
                f"{self.sample_size:,} rows; target: 1=denied, 0=originated"
            ),
        )

    def download(self, dest_dir: str) -> str:
        dest = Path(dest_dir) / f"hmda_{self.state.lower()}_{self.year}.csv"
        if dest.exists() and dest.stat().st_size > 0:
            return str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HMDA_BASE}?states={self.state}&years={self.year}"
        urllib.request.urlretrieve(url, dest)
        return str(dest)

    # -- preprocessing ------------------------------------------------------

    @staticmethod
    def _parse_dti(value: object) -> Optional[float]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        s = str(value).strip()
        if s in _NUMERIC_DROP_TOKENS:
            return None
        if s in DTI_MIDPOINTS:
            return DTI_MIDPOINTS[s]
        # Fallback: numeric string like "47.5"
        try:
            return float(s.rstrip("%"))
        except ValueError:
            return None

    @staticmethod
    def _to_float(value: object) -> Optional[float]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        s = str(value).strip()
        if s in _NUMERIC_DROP_TOKENS:
            return None
        try:
            return float(s)
        except ValueError:
            return None

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)

        # Filter to clean approve/deny outcomes and binarise.
        if "action_taken" not in df.columns:
            raise ValueError("HMDA file missing 'action_taken' column")
        df = df[df["action_taken"].isin(ACTION_KEEP)].copy()
        df["denied"] = (df["action_taken"] == 3).astype(int)

        # Numeric coercions / range parses.
        df["debt_to_income_ratio"] = df["debt_to_income_ratio"].map(self._parse_dti)
        for col in ("loan_to_value_ratio", "interest_rate", "loan_term",
                    "loan_amount", "income"):
            if col in df.columns:
                df[col] = df[col].map(self._to_float)

        # Drop rows with missing critical fields. We don't impute LAR
        # because HMDA reporters are required to fill these — missing
        # signals "Exempt" filing or data-quality issues we don't want
        # to model around.
        critical = [
            "loan_amount", "income", "debt_to_income_ratio",
            "loan_to_value_ratio", "loan_term",
        ]
        df = df.dropna(subset=[c for c in critical if c in df.columns])

        # Cap loan_amount and income at the 99th percentile to limit
        # the influence of outliers (HMDA has a long right tail —
        # multi-family loans, jumbo mortgages).
        for col in ("loan_amount", "income"):
            if col in df.columns:
                cap = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=cap)

        # Stratified sample to keep memory reasonable. We iterate explicitly
        # rather than using ``groupby().apply()`` because the latter has a
        # known footgun where the grouping column can be reduced into the
        # index, silently dropping the target column from the result.
        if len(df) > self.sample_size:
            parts = []
            total = len(df)
            for _label, group in df.groupby("denied"):
                n = int(round(self.sample_size * len(group) / total))
                if n > 0:
                    parts.append(group.sample(n=min(n, len(group)), random_state=42))
            df = pd.concat(parts).reset_index(drop=True)

        return df
