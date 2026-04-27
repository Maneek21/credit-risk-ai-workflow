"""Phase 6B — Skill-augmented LLM evaluation (Claude Opus 4.7 only).

Re-runs the same 200 stratified UCI profiles from Phase 6, but with the structured
credit-risk skill loaded as a system prompt. Tests whether the skill improves
accuracy / calibration / fairness over the zero-shot baseline.

  - Identical sample (random_state=42, same row_index set as Phase 6).
  - Single run per profile (consistency was 99.5% in baseline; one run suffices).
  - Anthropic prompt caching enabled on the system prompt → ~5x cost reduction.
  - Per-call checkpoint flush; skip rows already in the output CSV.

Outputs:
  results/06b_skill_decisions.csv
  results/06b_skill_comparison.csv
  results/06b_api_log.csv
  results/06b_accuracy_comparison.png
  results/06b_bias_comparison.png
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import RESULTS, ROOT, SEED, banner
from data_uci import split_uci

# ---- Config ----
RUNS_PER_PROFILE = 1            # baseline already established run-to-run consistency
SAMPLE_PER_CLASS = 250          # 500 total — must mirror Phase 6 sampling. Strict superset
                                # of the earlier seed-42 100/100 sample, so existing rows
                                # are reused via skip-existing.
SPEND_CAP_USD = float(os.environ.get("PHASE6B_SPEND_CAP_USD", "12.0"))

# Per-model pricing (USD per 1M tokens) and cache multipliers.
# Anthropic: cache write 1.25x input, cache read 0.10x input.
# OpenAI: prompt caching is automatic; we treat cached_tokens at 0.50x input.
MODEL_PRICING = {
    "claude-opus-4-7": {
        "input": 15.0, "output": 75.0,
        "cache_write_mult": 1.25, "cache_read_mult": 0.10,
    },
    "gpt-4o": {
        "input": 2.5, "output": 10.0,
        "cache_write_mult": 1.0, "cache_read_mult": 0.50,
    },
    "gpt-5.4": {
        "input": 2.5, "output": 10.0,
        "cache_write_mult": 1.0, "cache_read_mult": 0.50,
    },
}

DECISIONS_CSV = RESULTS / "06b_skill_decisions.csv"
API_LOG_CSV = RESULTS / "06b_api_log.csv"
COMPARISON_CSV = RESULTS / "06b_skill_comparison.csv"
ACCURACY_PNG = RESULTS / "06b_accuracy_comparison.png"
BIAS_PNG = RESULTS / "06b_bias_comparison.png"


# ---- Env loader ----
def _load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ---- Skill system prompt extraction ----
def _load_system_prompt() -> str:
    """Extract the fenced system-prompt block from the skill markdown."""
    skill_path = ROOT / "skills" / "credit_risk_assessment_SKILL.md"
    text = skill_path.read_text(encoding="utf-8")
    # Find the ``` block under "## System Prompt"
    after_header = text.split("## System Prompt", 1)[1]
    m = re.search(r"```\s*\n(.*?)\n```", after_header, re.DOTALL)
    if not m:
        raise RuntimeError("Could not extract system prompt fenced block from skill MD")
    return m.group(1).strip()


# ---- Borrower profile in the skill's exact format ----
def _profile_text(row: pd.Series) -> str:
    edu_map = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
    sex_map = {1: "male", 2: "female"}
    mar_map = {1: "married", 2: "single", 3: "other"}
    pay_avg = row[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].mean()
    bill_avg = row[[f"BILL_AMT{i}" for i in range(1, 7)]].mean()
    pay_amt_avg = row[[f"PAY_AMT{i}" for i in range(1, 7)]].mean()
    return (
        "Assess this borrower:\n"
        f"- Credit limit (NTD): {int(row['LIMIT_BAL']):,}\n"
        f"- Sex: {sex_map.get(int(row['SEX']), 'unknown')}\n"
        f"- Education: {edu_map.get(int(row['EDUCATION']), 'other')}\n"
        f"- Marital status: {mar_map.get(int(row['MARRIAGE']), 'other')}\n"
        f"- Age: {int(row['AGE'])}\n"
        f"- Repayment status (most recent month): {int(row['PAY_0'])}\n"
        f"- Average repayment status over 6 months: {pay_avg:.2f}\n"
        f"- Average monthly bill (NTD): {bill_avg:,.0f}\n"
        f"- Average monthly payment (NTD): {pay_amt_avg:,.0f}\n"
    )


# ---- JSON parser ----
def _parse_json(text: str) -> dict:
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


# ---- Sample identical to Phase 6 ----
def _build_sample() -> pd.DataFrame:
    _, _, X_test, _, _, y_test = split_uci()
    test = X_test.copy()
    test["y"] = y_test.values
    pos = test[test["y"] == 1].sample(SAMPLE_PER_CLASS, random_state=SEED)
    neg = test[test["y"] == 0].sample(SAMPLE_PER_CLASS, random_state=SEED)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)
    sample = sample.reset_index().rename(columns={sample.index.name or "index": "row_index"})
    return sample


def _verify_sample_matches_phase6(sample: pd.DataFrame) -> None:
    """Cross-check: the row_index set should match Phase 6's existing decisions."""
    phase6 = RESULTS / "06_llm_decisions.csv"
    if not phase6.exists():
        print("  warning: no Phase 6 baseline on disk; skipping row_index cross-check")
        return
    df6 = pd.read_csv(phase6)
    if "row_index" not in df6.columns:
        print("  warning: Phase 6 decisions has no row_index column; skipping cross-check")
        return
    baseline_set = set(df6[df6["provider"] == "claude"]["row_index"].astype(int).unique())
    new_set = set(sample["row_index"].astype(int).unique())
    if baseline_set and baseline_set != new_set:
        missing = baseline_set - new_set
        extra = new_set - baseline_set
        print(f"  WARNING: row_index mismatch. missing={len(missing)} extra={len(extra)}")
    else:
        print(f"  cross-check OK: {len(new_set)} row_indexes match Phase 6 baseline")


# ---- API callers (per provider) ----
def _call_claude(system_prompt: str, profile: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=600,
        # Anthropic deprecated `temperature` for claude-opus-4-7 in March 2026.
        system=[{"type": "text", "text": system_prompt,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": profile}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    usage = resp.usage
    return {
        "text": text,
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }


def _make_openai_skill_caller(model: str):
    """Builder for an OpenAI Chat Completions runner with system+user messages."""
    use_completion_tokens = model.startswith("gpt-5")

    def _call(system_prompt: str, profile: str) -> dict:
        import openai
        client = openai.OpenAI()
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": profile},
            ],
        )
        if use_completion_tokens:
            kwargs["max_completion_tokens"] = 600
        else:
            kwargs["max_tokens"] = 600
            kwargs["temperature"] = 0.0
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content if resp.choices else ""
        usage = resp.usage
        # OpenAI auto-caches; cached prompt tokens reported in usage.prompt_tokens_details
        cached = 0
        try:
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
        except Exception:
            pass
        prompt_tok = getattr(usage, "prompt_tokens", 0)
        # We charge "input_tokens" as non-cached portion, cache_read_tokens for the cached portion.
        return {
            "text": text or "",
            "input_tokens": max(prompt_tok - cached, 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "cache_creation_tokens": 0,
            "cache_read_tokens": cached,
        }
    return _call


# ---- Runner registry ----
def register_runners(only: list[str] | None = None) -> list[tuple[str, callable, str]]:
    """Return list of (model_string, caller, api_key_env_name) for available runners."""
    out: list[tuple[str, callable, str]] = []
    if (only is None or "claude-opus-4-7" in only) and os.environ.get("ANTHROPIC_API_KEY"):
        out.append(("claude-opus-4-7", _call_claude, "ANTHROPIC_API_KEY"))
    if (only is None or "gpt-4o" in only) and os.environ.get("OPENAI_API_KEY"):
        out.append(("gpt-4o", _make_openai_skill_caller("gpt-4o"), "OPENAI_API_KEY"))
    if (only is None or "gpt-5.4" in only) and os.environ.get("OPENAI_API_KEY"):
        out.append(("gpt-5.4", _make_openai_skill_caller("gpt-5.4"), "OPENAI_API_KEY"))
    return out


def _cost_usd(model: str, in_tok: int, out_tok: int, cache_create: int, cache_read: int) -> float:
    p = MODEL_PRICING.get(model, {"input": 0, "output": 0,
                                  "cache_write_mult": 0, "cache_read_mult": 0})
    return (
        (in_tok / 1e6) * p["input"]
        + (out_tok / 1e6) * p["output"]
        + (cache_create / 1e6) * p["input"] * p["cache_write_mult"]
        + (cache_read / 1e6) * p["input"] * p["cache_read_mult"]
    )


# ---- Checkpointed write helpers ----
DECISIONS_COLS = [
    "row_index", "y_true", "model",
    "decision", "confidence", "default_probability",
    "primary_factors", "secondary_factors", "reasoning", "error",
]
API_LOG_COLS = [
    "ts", "model", "input_tokens", "output_tokens",
    "cache_creation_tokens", "cache_read_tokens",
    "cost_usd", "cumulative_spend", "elapsed_s", "error",
]


def _append_csv(path: Path, row: dict, cols: list[str]) -> None:
    write_header = not path.exists()
    pd.DataFrame([{k: row.get(k, "") for k in cols}]).to_csv(
        path, mode="a", header=write_header, index=False,
    )


def _existing_done_pairs() -> set[tuple[int, str]]:
    """Set of (row_index, model) pairs that have a SUCCESSFUL decision on disk."""
    if not DECISIONS_CSV.exists():
        return set()
    try:
        df = pd.read_csv(DECISIONS_CSV)
        ok = df[df["error"].fillna("") == ""]
        return {(int(r["row_index"]), str(r["model"])) for _, r in ok.iterrows()}
    except Exception:
        return set()


# ---- Bias / accuracy metrics ----
def _accuracy_block(df: pd.DataFrame) -> dict:
    """For a frame with columns row_index, y_true, decision, confidence, default_probability."""
    df = df.copy()
    df["pred_deny"] = (df["decision"].fillna("").str.upper() == "DENY").astype(int)
    n = len(df)
    if n == 0:
        return {"n": 0}
    y = df["y_true"].astype(int).values
    pred = df["pred_deny"].values
    accuracy = float((pred == y).mean())
    deny_rate = float(pred.mean())
    # AUC: prefer default_probability if present, else use confidence-as-deny score
    if df["default_probability"].notna().any():
        score = df["default_probability"].astype(float).fillna(0.5).values
    else:
        score = np.where(pred == 1, df["confidence"].astype(float).fillna(0.5),
                                     1 - df["confidence"].astype(float).fillna(0.5)).astype(float)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y, score)) if len(set(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "n": n,
        "auc": auc,
        "accuracy": accuracy,
        "deny_rate": deny_rate,
        "mean_confidence": float(df["confidence"].astype(float).mean(skipna=True)),
        "mean_default_probability": float(df["default_probability"].astype(float).mean(skipna=True)),
    }


def _bias_block(df: pd.DataFrame, sample: pd.DataFrame) -> dict:
    """Join decisions with UCI sample to get SEX/EDUCATION/AGE, return DI + DP figures."""
    feats = sample[["row_index", "SEX", "EDUCATION", "AGE"]].copy()
    df = df.merge(feats, on="row_index", how="left").copy()
    df["pred_approve"] = (df["decision"].fillna("").str.upper() == "APPROVE").astype(int)

    # ---- Sex (1=male reference, 2=female) ----
    sex_rates = df.groupby("SEX")["pred_approve"].mean()
    male = sex_rates.get(1, np.nan)
    female = sex_rates.get(2, np.nan)
    sex_dp = float(abs(female - male)) if pd.notna(male) and pd.notna(female) else float("nan")
    sex_di = float(female / male) if pd.notna(male) and male > 0 else float("nan")

    # ---- Education (1=graduate ref, 2=university, 3=high school) ----
    edu_rates = df.groupby("EDUCATION")["pred_approve"].mean()
    grad = edu_rates.get(1, np.nan)
    univ = edu_rates.get(2, np.nan)
    hs = edu_rates.get(3, np.nan)
    edu_min_lower = np.nanmin([univ, hs]) if pd.notna(univ) or pd.notna(hs) else np.nan
    edu_di = float(edu_min_lower / grad) if pd.notna(grad) and grad > 0 else float("nan")
    rates_for_dp = [r for r in [grad, univ, hs] if pd.notna(r)]
    edu_dp = float(max(rates_for_dp) - min(rates_for_dp)) if rates_for_dp else float("nan")

    # ---- Age (4 buckets: <30, 30-39, 40-49, 50+) ----
    age = df["AGE"].astype(float)
    bins = [0, 30, 40, 50, 200]
    labels = ["<30", "30-39", "40-49", "50+"]
    df["age_bucket"] = pd.cut(age, bins=bins, labels=labels, right=False)
    age_rates = df.groupby("age_bucket", observed=True)["pred_approve"].mean()
    age_dp = float(age_rates.max() - age_rates.min()) if not age_rates.empty else float("nan")

    return {
        "sex_di_ratio": sex_di,
        "sex_dp_diff": sex_dp,
        "education_di_ratio": edu_di,
        "education_dp_diff": edu_dp,
        "age_dp_diff": age_dp,
        "_edu_grad": float(grad) if pd.notna(grad) else np.nan,
        "_edu_univ": float(univ) if pd.notna(univ) else np.nan,
        "_edu_hs": float(hs) if pd.notna(hs) else np.nan,
    }


_MODEL_TO_BASELINE_KEY = {
    "claude-opus-4-7": ("claude", "claude-opus-4-7"),
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-5.4": ("openai", "gpt-5.4"),
}


def _baseline_block(sample: pd.DataFrame, model: str) -> tuple[dict, dict]:
    """Pull the Phase-6 zero-shot baseline for a specific model."""
    p = RESULTS / "06_llm_decisions.csv"
    if not p.exists():
        return ({}, {})
    df = pd.read_csv(p)
    if "row_index" not in df.columns:
        return ({}, {})
    key = _MODEL_TO_BASELINE_KEY.get(model)
    if not key:
        return ({}, {})
    provider, base_model = key
    sub = df[(df["provider"] == provider) & (df["model"] == base_model)].copy()
    if sub.empty:
        return ({}, {})

    sub["pred_deny"] = (sub["decision"].fillna("").str.upper() == "DENY").astype(int)
    agg = sub.groupby("row_index").agg(
        y_true=("y_true", "first"),
        pred_mean=("pred_deny", "mean"),
        confidence=("confidence", "mean"),
    ).reset_index()
    agg["decision"] = np.where(agg["pred_mean"] >= 0.5, "DENY", "APPROVE")
    agg["default_probability"] = np.nan
    acc = _accuracy_block(agg.rename(columns={"pred_mean": "_p"}))
    bias = _bias_block(agg, sample)
    return acc, bias


# ---- Plots ----
def _plot_accuracy(baseline: dict, skill: dict, out_path: Path, model: str) -> None:
    metrics = ["auc", "accuracy", "deny_rate"]
    labels = ["AUC", "Accuracy", "Deny rate"]
    bvals = [baseline.get(m, np.nan) for m in metrics]
    svals = [skill.get(m, np.nan) for m in metrics]
    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, bvals, w, label="Baseline (zero-shot)", color="#888")
    ax.bar(x + w/2, svals, w, label="Skill-augmented", color="#0B2F6B")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Phase 6B — {model}: zero-shot vs skill-augmented")
    for xi, b, s in zip(x, bvals, svals):
        if pd.notna(b): ax.text(xi - w/2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        if pd.notna(s): ax.text(xi + w/2, s + 0.01, f"{s:.3f}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_bias(baseline_bias: dict, skill_bias: dict, out_path: Path, model: str) -> None:
    cats = ["Graduate", "University", "High school"]
    bvals = [baseline_bias.get("_edu_grad"), baseline_bias.get("_edu_univ"), baseline_bias.get("_edu_hs")]
    svals = [skill_bias.get("_edu_grad"), skill_bias.get("_edu_univ"), skill_bias.get("_edu_hs")]
    bvals = [v if pd.notna(v) else 0 for v in bvals]
    svals = [v if pd.notna(v) else 0 for v in svals]
    x = np.arange(len(cats))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w/2, bvals, w, label="Baseline (zero-shot)", color="#888")
    ax.bar(x + w/2, svals, w, label="Skill-augmented", color="#0B2F6B")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Approval rate")
    if bvals[0] > 0:
        ax.axhline(0.8 * bvals[0], color="#c0392b", linestyle="--", linewidth=1,
                   label=f"4/5 of graduate (baseline) = {0.8*bvals[0]:.3f}")
    if svals[0] > 0:
        ax.axhline(0.8 * svals[0], color="#27ae60", linestyle="--", linewidth=1,
                   label=f"4/5 of graduate (skill) = {0.8*svals[0]:.3f}")
    ax.set_title(f"Phase 6B — {model} education approval rates: zero-shot vs skill")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---- Main ----
def main() -> None:
    banner("PHASE 6B — Skill-augmented LLM evaluation (multi-model)")
    _load_env()

    only = None
    for arg in sys.argv[1:]:
        if arg.startswith("--models="):
            only = [s.strip() for s in arg.split("=", 1)[1].split(",") if s.strip()]
    runners = register_runners(only)
    if not runners:
        print("No runners available — set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        return
    print(f"  runners: {[m for m, _, _ in runners]}")

    system_prompt = _load_system_prompt()
    print(f"  loaded skill system prompt: {len(system_prompt):,} chars")

    sample = _build_sample()
    _verify_sample_matches_phase6(sample)
    print(f"  sample: {len(sample)} profiles "
          f"({int(sample['y'].sum())} default, {int((1-sample['y']).sum())} non-default)")

    done = _existing_done_pairs()
    if done:
        print(f"  resuming — {len(done)} (row_index, model) pairs already done")

    spend = 0.0
    if API_LOG_CSV.exists():
        try:
            spend = float(pd.read_csv(API_LOG_CSV)["cost_usd"].sum())
        except Exception:
            pass
    print(f"  starting cumulative spend: ${spend:.4f}")

    # Build (row, model) work list
    work = []
    for _, row in sample.iterrows():
        for model, fn, _env in runners:
            if (int(row["row_index"]), model) in done:
                continue
            work.append((row, model, fn))
    print(f"  to do: {len(work)} (profile, model) calls "
          f"across {len(runners)} runner(s)")

    blocked: set[str] = set()
    consec_fails: dict[str, int] = {m: 0 for m, _, _ in runners}
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    csv_lock = threading.Lock()
    spend_lock = threading.Lock()
    fail_lock = threading.Lock()
    MAX_WORKERS = int(os.environ.get("PHASE6B_WORKERS", "8"))

    def _do_one(row, model, fn):
        nonlocal spend
        if model in blocked:
            return None
        profile = _profile_text(row)
        t0 = time.time()
        err = ""
        text = ""
        in_tok = out_tok = cc = cr = 0
        try:
            r = fn(system_prompt, profile)
            text = r["text"]
            in_tok = r["input_tokens"]; out_tok = r["output_tokens"]
            cc = r["cache_creation_tokens"]; cr = r["cache_read_tokens"]
            with fail_lock:
                consec_fails[model] = 0
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            with fail_lock:
                consec_fails[model] += 1
                if consec_fails[model] >= 5 and model not in blocked:
                    blocked.add(model)
                    print(f"    [BLOCKING {model} — 5 consecutive failures]", flush=True)

        cost = _cost_usd(model, in_tok, out_tok, cc, cr)
        with spend_lock:
            spend += cost
            cur_spend = spend
        parsed = _parse_json(text) or {}
        decision = str(parsed.get("decision", "")).upper()
        confidence = parsed.get("confidence")
        default_prob = parsed.get("default_probability")
        primary = parsed.get("primary_factors", [])
        secondary = parsed.get("secondary_factors", [])
        reasoning = parsed.get("reasoning", "")

        with csv_lock:
            _append_csv(DECISIONS_CSV, {
                "row_index": int(row["row_index"]), "y_true": int(row["y"]),
                "model": model, "decision": decision,
                "confidence": float(confidence) if confidence is not None else "",
                "default_probability": float(default_prob) if default_prob is not None else "",
                "primary_factors": json.dumps(primary)[:500],
                "secondary_factors": json.dumps(secondary)[:500],
                "reasoning": (reasoning or "")[:500], "error": err,
            }, DECISIONS_COLS)
            _append_csv(API_LOG_CSV, {
                "ts": time.time(), "model": model,
                "input_tokens": in_tok, "output_tokens": out_tok,
                "cache_creation_tokens": cc, "cache_read_tokens": cr,
                "cost_usd": cost, "cumulative_spend": cur_spend,
                "elapsed_s": time.time() - t0, "error": err,
            }, API_LOG_COLS)
        return (row, model, decision, default_prob, confidence, in_tok, out_tok,
                cc, cr, err, cur_spend)

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_do_one, row, model, fn) for row, model, fn in work]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"    futures error: {e}", flush=True); continue
            if res is None:
                continue
            row, model, decision, default_prob, confidence, in_tok, out_tok, cc, cr, err, cur_spend = res
            completed += 1
            pd_disp = f"{float(default_prob):.2f}" if default_prob not in (None, "") else "----"
            conf_disp = f"{float(confidence):.2f}" if confidence not in (None, "") else "----"
            short = model.split("/")[-1]
            if completed % 10 == 0 or err:
                print(f"  [{completed:4d}/{len(work)}] {short:18s} row={int(row['row_index']):5d} "
                      f"decision={decision or '???':7s} PD={pd_disp} conf={conf_disp} "
                      f"toks={in_tok}/{out_tok} c/r={cc}/{cr} "
                      f"spend=${cur_spend:.4f}{(' ERR='+err[:60]) if err else ''}",
                      flush=True)
            if cur_spend >= SPEND_CAP_USD:
                print(f"  spend cap ${SPEND_CAP_USD} hit — cancelling remaining.")
                for f2 in futures: f2.cancel()
                break

    print(f"\n  total spend: ${spend:.4f}")

    # ---- Comparison + plots (per-model; errored rows excluded) ----
    skill_df_all = pd.read_csv(DECISIONS_CSV)
    skill_df_all = skill_df_all[skill_df_all["error"].fillna("") == ""].copy()
    print(f"\n  successful skill decisions on disk: {len(skill_df_all)}")
    if skill_df_all.empty:
        print("  no successful rows — cannot compute comparison.")
        return

    rows = []
    for model in sorted(skill_df_all["model"].unique()):
        subset = skill_df_all[skill_df_all["model"] == model].copy()
        skill_acc = _accuracy_block(subset)
        skill_bias = _bias_block(subset, sample)
        base_acc, base_bias = _baseline_block(sample, model)
        for k in ["auc", "accuracy", "deny_rate", "mean_confidence", "mean_default_probability"]:
            b = base_acc.get(k, float("nan"))
            s = skill_acc.get(k, float("nan"))
            rows.append({"model": model, "metric": k,
                         "baseline_zero_shot": b, "skill_augmented": s,
                         "delta": (s - b) if pd.notna(b) and pd.notna(s) else float("nan"),
                         "n_skill": skill_acc.get("n", 0)})
        for k in ["education_di_ratio", "education_dp_diff",
                  "sex_di_ratio", "sex_dp_diff", "age_dp_diff"]:
            b = base_bias.get(k, float("nan"))
            s = skill_bias.get(k, float("nan"))
            rows.append({"model": model, "metric": k,
                         "baseline_zero_shot": b, "skill_augmented": s,
                         "delta": (s - b) if pd.notna(b) and pd.notna(s) else float("nan"),
                         "n_skill": skill_acc.get("n", 0)})
        # Per-model PNGs
        slug = model.replace("/", "_").replace(":", "_")
        acc_png = RESULTS / f"06b_accuracy_comparison_{slug}.png"
        bias_png = RESULTS / f"06b_bias_comparison_{slug}.png"
        _plot_accuracy(base_acc, skill_acc, acc_png, model)
        _plot_bias(base_bias, skill_bias, bias_png, model)
        print(f"  {model}: wrote {acc_png.name} + {bias_png.name}")

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(COMPARISON_CSV, index=False)
    print(f"\n  wrote {COMPARISON_CSV}")
    print(cmp_df.round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
