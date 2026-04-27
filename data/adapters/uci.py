"""UCI Default of Credit Card Clients adapter.

Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Taiwan, 2005 — 30,000 credit-card clients, 23 explanatory variables,
binary default-next-month label. Small enough that the raw .xls is
reasonable to ship for offline reproducibility.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from workflow.training.datasets import DatasetAdapter, DatasetMetadata


UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)
UCI_FILENAME = "uci_default_credit.xls"


class UCIAdapter(DatasetAdapter):
    """UCI Default of Credit Card Clients (Taiwan, 30K rows)."""

    def metadata(self) -> DatasetMetadata:
        feature_columns = [
            "LIMIT_BAL", "EDUCATION", "MARRIAGE", "AGE",
            "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        ]
        return DatasetMetadata(
            name="uci",
            region="TW",
            target_column="default",
            feature_columns=feature_columns,
            categorical_columns=["EDUCATION", "MARRIAGE"],
            # SEX is a Constitutional protected attribute under most
            # jurisdictions — exclude from features, evaluate fairness only.
            protected_columns={
                "SEX": "1=male, 2=female (UCI codebook)",
                "AGE_BUCKET": "Age decile (engineered for fairness eval)",
            },
            positive_label=1,
            description="Taiwan credit-card clients, default within 30 days",
        )

    def download(self, dest_dir: str) -> str:
        dest = Path(dest_dir) / UCI_FILENAME
        if dest.exists():
            return str(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(UCI_URL, dest)
        return str(dest)

    def load(self, path: str) -> pd.DataFrame:
        # Header row is the second row (row index 1) in the published .xls.
        df = pd.read_excel(path, header=1)
        df.columns = df.columns.str.strip()

        # Standardise the target name.
        target_aliases = ["default payment next month", "default.payment.next.month"]
        for alias in target_aliases:
            if alias in df.columns:
                df = df.rename(columns={alias: "default"})
                break
        if "default" not in df.columns:
            raise ValueError(
                f"UCI loader could not find target column; tried {target_aliases}"
            )

        # Drop the row identifier — never useful as a feature.
        df = df.drop(columns=[c for c in ("ID",) if c in df.columns])

        # Clip EDUCATION / MARRIAGE to documented categories. The UCI
        # codebook says EDUCATION ∈ {1,2,3,4} and MARRIAGE ∈ {1,2,3} but
        # the raw file contains stray 0/5/6 values that mean "other".
        if "EDUCATION" in df.columns:
            df["EDUCATION"] = df["EDUCATION"].clip(lower=1, upper=4)
        if "MARRIAGE" in df.columns:
            df["MARRIAGE"] = df["MARRIAGE"].clip(lower=1, upper=3)

        # Engineer AGE_BUCKET for fairness analysis (raw AGE leaks into
        # the model otherwise — keep AGE as a feature, derive the bucket
        # for protected_columns).
        if "AGE" in df.columns:
            df["AGE_BUCKET"] = pd.cut(
                df["AGE"],
                bins=[0, 25, 35, 45, 60, 200],
                labels=["<=25", "26-35", "36-45", "46-60", "60+"],
            ).astype(str)

        return df
