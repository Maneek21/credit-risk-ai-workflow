#!/usr/bin/env python3
"""Basic usage example — process a single credit application.

Prerequisites:
  1. Train a model: cd benchmarks && python src/train_classical.py
  2. Set API key: export OPENAI_API_KEY=sk-...
  3. Run this script: python examples/basic_usage.py
"""
import sys
from pathlib import Path

# Add parent to path so we can import workflow
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflow import CreditWorkflow

# Sample borrower profile (UCI dataset format)
sample_applicant = {
    "LIMIT_BAL": 200000,
    "SEX": 1,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 50000,
    "BILL_AMT2": 48000,
    "BILL_AMT3": 45000,
    "BILL_AMT4": 43000,
    "BILL_AMT5": 40000,
    "BILL_AMT6": 38000,
    "PAY_AMT1": 5000,
    "PAY_AMT2": 5000,
    "PAY_AMT3": 4500,
    "PAY_AMT4": 4500,
    "PAY_AMT5": 4000,
    "PAY_AMT6": 4000,
}


def main():
    # Initialize workflow with pre-trained model
    model_path = Path(__file__).parent.parent / "data" / "processed" / "xgboost_model.joblib"

    if not model_path.exists():
        print("Model not found. Train first:")
        print("  cd benchmarks && python src/train_classical.py")
        sys.exit(1)

    wf = CreditWorkflow(
        model_path=str(model_path),
        llm_provider="openai",
        llm_model="gpt-4o",
    )

    # Process the application
    result = wf.process_application(sample_applicant)

    # Display results
    print(f"{'='*60}")
    print(f"DECISION: {result.decision}")
    print(f"Default Probability: {result.probability:.3f}")
    print(f"Flags: {result.flags or 'None'}")
    print(f"{'='*60}")
    print(f"\nCREDIT MEMO:\n{result.memo}")

    if result.adverse_action:
        print(f"\n{'='*60}")
        print(f"ADVERSE ACTION NOTICE:\n{result.adverse_action}")

    print(f"\nSHAP Factors:")
    for f in result.shap_factors:
        if "error" not in f:
            print(f"  {f['feature']}: {f['shap_value']:.4f} ({f['direction']})")


if __name__ == "__main__":
    main()
