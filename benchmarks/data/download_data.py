#!/usr/bin/env python3
"""Download datasets for benchmarking.

Downloads:
  - UCI Default of Credit Card Clients (30K rows)
  - HMDA LAR data (optional, requires year argument)

Usage:
  python download_data.py              # UCI only
  python download_data.py --hmda 2022  # UCI + HMDA 2022
"""
import argparse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00350/default%20of%20credit%20card%20clients.xls"
)
HMDA_BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"


def download_uci():
    dest = DATA_DIR / "uci_default_credit.xls"
    if dest.exists():
        print(f"UCI data already exists: {dest}")
        return
    print("Downloading UCI Default of Credit Card Clients...")
    urllib.request.urlretrieve(UCI_URL, dest)
    print(f"Saved to {dest} ({dest.stat().st_size / 1024:.0f} KB)")


def download_hmda(year: int):
    dest = DATA_DIR / f"hmda_{year}_ny.csv"
    if dest.exists():
        print(f"HMDA {year} already exists: {dest}")
        return
    url = f"{HMDA_BASE}?states=NY&years={year}"
    print(f"Downloading HMDA {year} (NY only, may take a few minutes)...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmda", type=int, help="HMDA year to download (e.g., 2022)")
    args = parser.parse_args()

    download_uci()
    if args.hmda:
        download_hmda(args.hmda)
    print("\nDone. Run benchmarks with: python src/train_classical.py")
