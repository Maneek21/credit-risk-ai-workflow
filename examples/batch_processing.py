#!/usr/bin/env python3
"""Batch processing example — process multiple applications with safety metrics.

Demonstrates:
  - Processing a batch of applicants
  - Tracking safety layer activations
  - Generating summary statistics
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from workflow import CreditWorkflow


def main():
    model_path = Path(__file__).parent.parent / "data" / "processed" / "xgboost_model.joblib"
    data_path = Path(__file__).parent.parent / "benchmarks" / "data" / "uci_sample_100rows.csv"

    if not model_path.exists():
        print("Train the model first: cd benchmarks && python src/train_classical.py")
        sys.exit(1)

    wf = CreditWorkflow(
        model_path=str(model_path),
        llm_provider="openai",
        llm_model="gpt-4o",
    )

    # Load sample data
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in ["ID", "default.payment.next.month"]]

    results = []
    for idx, row in df.head(10).iterrows():  # Process first 10 for demo
        applicant = row[feature_cols].to_dict()
        result = wf.process_application(applicant)
        results.append({
            "id": row.get("ID", idx),
            "decision": result.decision,
            "probability": result.probability,
            "flags": "|".join(result.flags),
            "protected_attr_hits": len(result.metadata.get("protected_attribute_hits", [])),
        })
        print(f"  Processed applicant {row.get('ID', idx)}: {result.decision} (p={result.probability:.3f})")

    # Summary
    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY ({len(results_df)} applications)")
    print(f"  Approvals: {(results_df['decision'] == 'APPROVE').sum()}")
    print(f"  Denials: {(results_df['decision'] == 'DENY').sum()}")
    print(f"  Borderline flags: {results_df['flags'].str.contains('BORDERLINE').sum()}")
    print(f"  Protected attr hits: {results_df['protected_attr_hits'].sum()}")


if __name__ == "__main__":
    main()
