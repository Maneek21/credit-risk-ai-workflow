#!/usr/bin/env python3
"""Load, clean, and split credit datasets for the underwriting workflow.

Supports:
  - UCI Default of Credit Card Clients (--dataset uci)
  - HMDA LAR data (--dataset hmda --year YYYY)
  - Lending Club (--dataset lendingclub)

Usage:
  python scripts/prepare_data.py --dataset uci
  python scripts/prepare_data.py --dataset hmda --year 2022
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

for d in (DATA_RAW, DATA_PROC):
    d.mkdir(parents=True, exist_ok=True)


# ── UCI ─────────────────────────────────────────────────────────────────

def load_uci() -> pd.DataFrame:
    """Load and clean UCI Default of Credit Card Clients."""
    path = DATA_RAW / "uci_default_credit_card.xls"
    if not path.exists():
        print(f"ERROR: UCI dataset not found at {path}", file=sys.stderr)
        print("Download from: https://archive.ics.uci.edu/dataset/350", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(path, header=1, index_col=0)

    # Collapse undocumented codes (see references/dataset-schemas.md)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    print(f"UCI loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Default rate: {df['default payment next month'].mean():.4f}")
    return df


def split_uci() -> dict:
    """Stratified 70/15/15 train/val/test split."""
    df = load_uci()

    target = "default payment next month"
    features = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]

    X = df[features].copy()
    y = df[target].copy()

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED,
    )
    val_frac = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=SEED,
    )

    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }

    out = DATA_PROC / "uci_splits.npz"
    np.savez(
        out,
        X_train=X_train.values, X_val=X_val.values, X_test=X_test.values,
        y_train=y_train.values, y_val=y_val.values, y_test=y_test.values,
        feature_names=features,
        train_index=X_train.index.values,
        val_index=X_val.index.values,
        test_index=X_test.index.values,
    )

    print(f"Splits saved to {out}")
    print(f"  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    print(f"  default rates: train={y_train.mean():.4f}, val={y_val.mean():.4f}, test={y_test.mean():.4f}")
    return splits


# ── HMDA ────────────────────────────────────────────────────────────────

def load_hmda(year: int, state: str = "NY") -> pd.DataFrame:
    """Load HMDA LAR data from CFPB API (approve/deny only)."""
    import urllib.request
    import io

    url = (
        f"https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"
        f"?states={state}&years={year}&actions_taken=1,3"
    )
    print(f"Fetching HMDA {year} ({state}) from CFPB API...")

    cache = DATA_RAW / f"hmda_{state}_{year}.csv"
    if cache.exists():
        print(f"  Using cached file: {cache}")
        df = pd.read_csv(cache, low_memory=False)
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "credit-underwriting-workflow/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(data), low_memory=False)
        df.to_csv(cache, index=False)
        print(f"  Cached to {cache}")

    df["approved"] = (df["action_taken"] == 1).astype(int)
    print(f"HMDA {year} loaded: {df.shape[0]:,} rows, approval rate: {df['approved'].mean():.4f}")
    return df


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare credit datasets")
    parser.add_argument("--dataset", required=True, choices=["uci", "hmda", "lendingclub"])
    parser.add_argument("--year", type=int, help="Year for HMDA data")
    parser.add_argument("--state", default="NY", help="State for HMDA data (default: NY)")
    args = parser.parse_args()

    if args.dataset == "uci":
        split_uci()
    elif args.dataset == "hmda":
        if not args.year:
            print("ERROR: --year required for HMDA dataset", file=sys.stderr)
            sys.exit(1)
        load_hmda(args.year, args.state)
    elif args.dataset == "lendingclub":
        print("Lending Club loader not yet implemented. Download from:")
        print("https://www.kaggle.com/datasets/wordsforthewise/lending-club")
        sys.exit(1)


if __name__ == "__main__":
    main()
