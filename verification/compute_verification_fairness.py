"""Fairness verification on the 500-profile pipeline run.

Computes disparate-impact ratios for SEX, EDUCATION, and AGE.
Writes verification/verification_fairness.json.

Caveat: bias measured here is the XGBoost model's, not the LLM's. The
LLM only writes the memo / adverse-action notice — the deterministic
classical model owns the approve/deny decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VERIFICATION_DIR = ROOT / "verification"
RESULTS_CSV = VERIFICATION_DIR / "pipeline_results_500.csv"
PROFILES_CSV = VERIFICATION_DIR / "test_profiles_500.csv"
OUT_JSON = VERIFICATION_DIR / "verification_fairness.json"

EDU_LABELS = {1: "Grad School", 2: "University", 3: "High School", 4: "Other",
              5: "Other-5", 6: "Other-6", 0: "Unknown"}
SEX_LABELS = {1: "Male", 2: "Female"}


def _di(rates: pd.Series) -> float:
    if rates.empty or rates.max() == 0:
        return 1.0
    return float(rates.min() / rates.max())


def _group_table(merged: pd.DataFrame, col: str, labels: dict | None = None) -> dict:
    grp = merged.groupby(col)["denied"]
    rates = grp.mean()
    counts = grp.size()
    rows = []
    for k in rates.index:
        label = labels.get(k, f"Code {k}") if labels else str(k)
        rows.append({
            "group": label,
            "code": (int(k) if hasattr(k, "item") else k),
            "n": int(counts[k]),
            "deny_rate": float(rates[k]),
        })
    di = _di(rates)
    return {"groups": rows, "disparate_impact": di, "passes_80_rule": di >= 0.80}


def main() -> int:
    if not RESULTS_CSV.exists() or not PROFILES_CSV.exists():
        print("missing inputs")
        return 1

    results = pd.read_csv(RESULTS_CSV)
    profiles = pd.read_csv(PROFILES_CSV)

    # results carries profile_idx (the row index in profiles that was processed).
    # Merge on that to bring SEX/EDUCATION/AGE alongside the decisions.
    merged = results.merge(
        profiles[["SEX", "EDUCATION", "AGE"]],
        left_on="profile_idx", right_index=True,
    )
    merged = merged[merged["decision"] != "ERROR"].copy()
    merged["denied"] = (merged["decision"] == "DENY").astype(int)

    print("FAIRNESS VERIFICATION (500 profiles)")
    print("=" * 50)

    sex = _group_table(merged, "SEX", SEX_LABELS)
    print("\nBy Sex:")
    for r in sex["groups"]:
        print(f"  {r['group']} (n={r['n']}): deny rate = {r['deny_rate']:.3f}")
    print(
        f"  Disparate Impact Ratio: {sex['disparate_impact']:.3f} "
        f"{'PASS' if sex['passes_80_rule'] else 'FAIL'}"
    )

    edu = _group_table(merged, "EDUCATION", EDU_LABELS)
    print("\nBy Education:")
    for r in edu["groups"]:
        print(f"  {r['group']} (n={r['n']}): deny rate = {r['deny_rate']:.3f}")
    print(
        f"  Disparate Impact Ratio: {edu['disparate_impact']:.3f} "
        f"{'PASS' if edu['passes_80_rule'] else 'FAIL'}"
    )

    merged["age_group"] = pd.cut(
        merged["AGE"], bins=[0, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "60+"],
    )
    age_grp = merged.groupby("age_group", observed=True)["denied"]
    age_rates = age_grp.mean()
    age_counts = age_grp.size()
    print("\nBy Age Group:")
    age_rows = []
    for k in age_rates.index:
        print(f"  {k} (n={int(age_counts[k])}): deny rate = {age_rates[k]:.3f}")
        age_rows.append({
            "group": str(k), "code": str(k), "n": int(age_counts[k]),
            "deny_rate": float(age_rates[k]),
        })
    di_age = _di(age_rates)
    print(f"  Disparate Impact Ratio: {di_age:.3f} {'PASS' if di_age >= 0.80 else 'FAIL'}")

    print("\nNOTE: These are the MODEL's decisions (XGBoost), not the LLM's.")
    print("The LLM only communicates — it cannot alter approve/deny outcomes.")

    out = {
        "by_sex": sex,
        "by_education": edu,
        "by_age": {
            "groups": age_rows,
            "disparate_impact": di_age,
            "passes_80_rule": di_age >= 0.80,
        },
        "note": (
            "Decisions are made by the XGBoost classical model; the LLM only "
            "drafts memos/adverse-action notices. Bias here reflects the model "
            "trained on UCI Default of Credit Card Clients (Taiwan, 2005)."
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nFairness summary saved to {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
