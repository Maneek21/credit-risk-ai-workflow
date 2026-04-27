"""Bondora P2P Lending adapter.

Source: Bondora (Estonia) public reports — https://www.bondora.com/en/public-reports
        Mirrored on Kaggle: ``sid321axn/bondora-peer-to-peer-lending-loan-data``.

Bondora exposes loan-level performance data for a European P2P
platform: ~200K issued loans, mostly originated in Estonia, Finland,
and Spain. Default labels are settled (each row reflects a closed
loan), making it a clean binary-target dataset for EU-jurisdiction
training.

Download pattern: Bondora's bulk CSV requires either the Kaggle CLI
(``kaggle datasets download sid321axn/bondora-peer-to-peer-lending-loan-data``)
or a direct fetch from the Bondora public-reports page. We try Kaggle
first and fall back to a documented manual-download path.
"""
from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from workflow.training.datasets import DatasetAdapter, DatasetMetadata


KAGGLE_DATASET = "sid321axn/bondora-peer-to-peer-lending-loan-data"
SAMPLE_SIZE = 100_000
BONDORA_FILENAME = "bondora_loan_data.csv"

#: Status values in the bulk feed. We keep closed loans — Defaulted
#: rows are positives, Repaid rows are negatives, Late rows can be
#: configured either way (``treat_late_as_default``), Current rows are
#: dropped because the label hasn't yet been observed.
STATUS_DEFAULT = {"Defaulted"}
STATUS_LATE = {"Late"}


class BondoraAdapter(DatasetAdapter):
    """Bondora P2P (Estonia / EU) — sampled to 100K closed loans."""

    def __init__(
        self,
        sample_size: int = SAMPLE_SIZE,
        treat_late_as_default: bool = False,
    ) -> None:
        self.sample_size = sample_size
        self.treat_late_as_default = treat_late_as_default

    def metadata(self) -> DatasetMetadata:
        feature_columns = [
            "AppliedAmount", "Amount", "Interest", "LoanDuration", "MonthlyPayment",
            "IncomeTotal", "ExistingLiabilities", "LiabilitiesTotal",
            "DebtToIncome", "FreeCash",
            "EmploymentStatus", "EmploymentDurationCurrentEmployer",
            "HomeOwnershipType", "Education", "MaritalStatus", "NrOfDependants",
            "PreviousRepaymentsBeforeLoan", "AmountOfPreviousLoansBeforeLoan",
        ]
        return DatasetMetadata(
            name="bondora",
            region="EU",
            target_column="default",
            feature_columns=feature_columns,
            categorical_columns=[
                "EmploymentStatus", "EmploymentDurationCurrentEmployer",
                "HomeOwnershipType", "Education", "MaritalStatus", "NrOfDependants",
            ],
            protected_columns={
                "Gender": "Applicant gender (M/F, often missing)",
                "Country": "Borrower country (proxy for national origin)",
                "Age": "Applicant age (numeric — also subject to age-discrimination check)",
            },
            positive_label=1,
            description=(
                f"Bondora P2P closed loans, sampled to {self.sample_size:,}; "
                f"target: 1=Defaulted, 0=Repaid"
                + (" or Late" if not self.treat_late_as_default else "")
            ),
        )

    def download(self, dest_dir: str) -> str:
        dest = Path(dest_dir) / BONDORA_FILENAME
        if dest.exists() and dest.stat().st_size > 0:
            return str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        kaggle = shutil.which("kaggle")
        if kaggle:
            tmp = dest.parent / ".bondora_tmp"
            tmp.mkdir(exist_ok=True)
            try:
                subprocess.run(
                    [kaggle, "datasets", "download", "-d", KAGGLE_DATASET,
                     "-p", str(tmp), "--unzip"],
                    check=True,
                )
                # Kaggle archive contains LoanData.csv at top level.
                for candidate in tmp.iterdir():
                    if candidate.suffix == ".csv":
                        candidate.replace(dest)
                        break
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)
            if dest.exists():
                return str(dest)

        raise RuntimeError(
            "Bondora download requires either the Kaggle CLI "
            "(`pip install kaggle && kaggle datasets download "
            f"{KAGGLE_DATASET} -p {dest_dir} --unzip`) or a manual export "
            "from https://www.bondora.com/en/public-reports. Place the "
            f"resulting CSV at {dest} and re-run."
        )

    # -- preprocessing ------------------------------------------------------

    def _binarize_status(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["Status"] != "Current"].copy()
        positive = STATUS_DEFAULT | (STATUS_LATE if self.treat_late_as_default else set())
        df["default"] = df["Status"].isin(positive).astype(int)
        return df

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        if "Status" not in df.columns:
            raise ValueError("Bondora file missing 'Status' column")

        df = self._binarize_status(df)

        # Median imputation for numeric features that frequently have NaN.
        numeric_features = [
            "IncomeTotal", "DebtToIncome", "FreeCash", "MonthlyPayment",
            "ExistingLiabilities", "LiabilitiesTotal",
            "PreviousRepaymentsBeforeLoan", "AmountOfPreviousLoansBeforeLoan",
        ]
        for col in numeric_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())

        # Cap right-tail outliers at the 99th percentile.
        for col in ("AppliedAmount", "Amount", "IncomeTotal", "LiabilitiesTotal"):
            if col in df.columns:
                cap = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=cap)

        # Drop rows that are >30% missing across the model features.
        feature_cols = [c for c in self.metadata().feature_columns if c in df.columns]
        if feature_cols:
            keep_mask = df[feature_cols].isna().mean(axis=1) <= 0.30
            df = df[keep_mask].copy()

        # Re-impute remaining NaN in features (categorical -> "unknown",
        # numeric -> median) so the downstream OneHotEncoder doesn't choke.
        for col in feature_cols:
            if df[col].dtype == object:
                df[col] = df[col].fillna("unknown")
            else:
                df[col] = df[col].fillna(df[col].median())

        # Stratified sample to cap the dataset size. Explicit per-class
        # iteration avoids a pandas footgun where ``groupby().apply()`` can
        # silently drop the grouping column from the result.
        if len(df) > self.sample_size:
            parts = []
            total = len(df)
            for _label, group in df.groupby("default"):
                n = int(round(self.sample_size * len(group) / total))
                if n > 0:
                    parts.append(group.sample(n=min(n, len(group)), random_state=42))
            df = pd.concat(parts).reset_index(drop=True)

        return df
