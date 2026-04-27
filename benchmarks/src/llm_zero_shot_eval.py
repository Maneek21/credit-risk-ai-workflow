"""Phase 6 — LLM credit assessment (multi-provider, idempotent).

Sample 200 borrower profiles from UCI test split (stratified by default label),
ask each registered LLM for an approval decision twice. Skip combinations that
already have rows in results/06_llm_decisions.csv so the script is safe to
re-run after adding new models.

Spend cap: $LLM_SPEND_CAP_USD (default 30). Free models count as $0.

Outputs (recomputed from union of existing + new decisions):
  results/06_llm_decisions.csv
  results/06_llm_metrics.csv
  results/06_consistency.csv
  results/06_api_log.csv      (appended)
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

from common import RESULTS, ROOT, SEED, banner
from data_uci import split_uci

# ---- Config ----
SAMPLE_PER_CLASS = 250              # 500 total UCI test profiles (Claude/OpenAI)
                                    # Existing 200 from earlier seed-42 100/100 run are a strict
                                    # subset and skip-existing logic reuses them.
OR_SAMPLE_LIMIT = 50                # First 50 of those for OpenRouter free models
                                    # (free-tier rate-limited; 50 × 4 models × 2 runs = 400 calls)
OR_SLEEP_SEC = 1.0                  # Account has credits; modest pacing only
OR_BLOCK_AFTER_PROFILES = 1         # Block a model after one fully-failed profile
RUNS_PER_PROFILE = 2
SPEND_CAP_USD = float(os.environ.get("LLM_SPEND_CAP_USD", "30"))

CLAUDE_MODEL = "claude-opus-4-7"
OPENAI_MODEL = "gpt-4o"
OPENAI_GPT5_MODEL = "gpt-5.4"        # Latest flagship as of 2026-03-05
OPENROUTER_MODELS = [
    # Dropped llama-3.3-70b and gemma-3-27b — both rate-limited persistently on
    # OR free tier even after the account had credits. See NOTES.md.
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "openai/gpt-oss-120b:free",
]

# Approximate USD/1M tokens. Free models = 0/0. GPT-5.4 estimated at GPT-5
# tier pricing — actual billing is logged per-call to api_log.csv.
PRICES: dict[str, dict[str, float]] = {
    CLAUDE_MODEL: {"input": 15.0, "output": 75.0},
    OPENAI_MODEL: {"input": 2.5, "output": 10.0},
    OPENAI_GPT5_MODEL: {"input": 2.5, "output": 10.0},
    **{m: {"input": 0.0, "output": 0.0} for m in OPENROUTER_MODELS},
}

PROMPT_TEMPLATE = dedent("""
    You are a credit risk analyst. Decide whether to APPROVE or DENY this credit
    application based ONLY on the borrower profile below. Output strict JSON.

    Borrower profile:
    {profile}

    Respond with JSON only:
    {{
      "decision": "APPROVE" | "DENY",
      "confidence": <float 0..1>,
      "top_reasons": ["...", "...", "..."]
    }}
""").strip()


# ---- Env ----
def _load_env() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ---- Profile + parsing ----
def _profile_text(row: pd.Series) -> str:
    edu_map = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
    sex_map = {1: "male", 2: "female"}
    mar_map = {1: "married", 2: "single", 3: "other"}
    pay_avg = row[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]].mean()
    bill_avg = row[[f"BILL_AMT{i}" for i in range(1, 7)]].mean()
    pay_amt_avg = row[[f"PAY_AMT{i}" for i in range(1, 7)]].mean()
    return (
        f"- Credit limit (NTD): {int(row['LIMIT_BAL']):,}\n"
        f"- Sex: {sex_map.get(int(row['SEX']), 'unknown')}\n"
        f"- Education: {edu_map.get(int(row['EDUCATION']), 'other')}\n"
        f"- Marital status: {mar_map.get(int(row['MARRIAGE']), 'other')}\n"
        f"- Age: {int(row['AGE'])}\n"
        f"- Repayment status (most recent month, -1=on time, +N=N months late): {int(row['PAY_0'])}\n"
        f"- Average repayment status over 6 months: {pay_avg:.2f}\n"
        f"- Average monthly bill (NTD): {bill_avg:,.0f}\n"
        f"- Average monthly payment (NTD): {pay_amt_avg:,.0f}\n"
    )


def _parse_json(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICES.get(model, {"input": 0, "output": 0})
    return (in_tok / 1e6) * p["input"] + (out_tok / 1e6) * p["output"]


# ---- Provider callers ----
def _call_claude(prompt: str) -> tuple[str, int, int]:
    import anthropic
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens


def _call_openai(prompt: str) -> tuple[str, int, int]:
    import openai
    c = openai.OpenAI()
    r = c.chat.completions.create(
        model=OPENAI_MODEL, temperature=0.0, max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens


def _call_gpt5(prompt: str) -> tuple[str, int, int]:
    """GPT-5.x family uses `max_completion_tokens` and rejects `temperature`."""
    import openai
    c = openai.OpenAI()
    r = c.chat.completions.create(
        model=OPENAI_GPT5_MODEL, max_completion_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    content = r.choices[0].message.content if r.choices else ""
    return content or "", r.usage.prompt_tokens, r.usage.completion_tokens


def _make_openrouter_caller(model: str):
    """Build a caller for an OpenRouter model with patient retry on 429/503.
    Up to 10 attempts; backoff schedule: 5/10/20/30/45/60/60/60/60/60 sec.
    """
    def _call(prompt: str) -> tuple[str, int, int]:
        import openai
        c = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={
                "HTTP-Referer": "https://iitd.ac.in/",
                "X-Title": "AI Credit Risk Assessment",
            },
        )
        backoffs = [3, 8]   # max ~11s of waiting before giving up
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                r = c.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                in_t = r.usage.prompt_tokens if r.usage else 0
                out_t = r.usage.completion_tokens if r.usage else 0
                content = r.choices[0].message.content if r.choices else ""
                return content or "", in_t, out_t
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                retryable = any(s in msg for s in ("429", "rate", "503", "timeout", "504"))
                if retryable and attempt < len(backoffs):
                    time.sleep(backoffs[attempt])
                    continue
                raise
        if last_err:
            raise last_err
        return "", 0, 0
    return _call


# ---- Runner registry ----
def register_runners() -> list[tuple[str, str, callable]]:
    runners: list[tuple[str, str, callable]] = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        runners.append(("claude", CLAUDE_MODEL, _call_claude))
    if os.environ.get("OPENAI_API_KEY"):
        runners.append(("openai", OPENAI_MODEL, _call_openai))
        runners.append(("openai", OPENAI_GPT5_MODEL, _call_gpt5))
    # OpenRouter free-tier free models are kept off the scaled run because the
    # rate-limited free tier can't sustain concurrent requests. Their existing
    # 50-profile pilot data stays in 06_llm_decisions.csv. Set
    # PHASE6_INCLUDE_OPENROUTER=1 to opt back in.
    if (os.environ.get("OPENROUTER_API_KEY")
            and os.environ.get("PHASE6_INCLUDE_OPENROUTER") == "1"):
        for m in OPENROUTER_MODELS:
            runners.append(("openrouter", m, _make_openrouter_caller(m)))
    return runners


# ---- Skip-existing ----
def _existing_done(existing: pd.DataFrame | None, row_index: int,
                   provider: str, model: str, runs: int) -> bool:
    if existing is None or existing.empty:
        return False
    sub = existing[(existing["row_index"] == row_index)
                   & (existing["provider"] == provider)
                   & (existing["model"] == model)]
    if sub.empty:
        return False
    return sub["run"].nunique() >= runs


def _load_existing_decisions() -> pd.DataFrame | None:
    p = RESULTS / "06_llm_decisions.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        # Reject placeholder-format file (single "note: ..." column)
        if "row_index" not in df.columns:
            return None
        return df
    except Exception:
        return None


# ---- Flush ----
def _flush(existing: pd.DataFrame | None, new_decisions: list[dict],
           api_log: list[dict] | None = None) -> None:
    """Rewrite decisions.csv with existing+new (idempotent). If api_log is passed,
    APPEND those rows to api_log.csv as a one-shot at end-of-run; do not call
    api_log path at every checkpoint or rows duplicate."""
    if new_decisions:
        new_df = pd.DataFrame(new_decisions)
        combined = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df
        combined = combined.drop_duplicates(
            subset=["row_index", "provider", "model", "run"], keep="last"
        ).reset_index(drop=True)
        combined.to_csv(RESULTS / "06_llm_decisions.csv", index=False)
    if api_log is not None:
        log_path = RESULTS / "06_api_log.csv"
        log_df = pd.DataFrame(api_log)
        if log_path.exists():
            try:
                old = pd.read_csv(log_path)
                log_df = pd.concat([old, log_df], ignore_index=True)
            except Exception:
                pass
        log_df.to_csv(log_path, index=False)


# ---- Aggregate ----
def _recompute_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    consistency_rows = []
    df = df.copy()
    df["pred"] = (df["decision"].fillna("").str.upper() == "DENY").astype(int)
    for (provider, model), sub in df.groupby(["provider", "model"]):
        agg = sub.groupby("row_index").agg(
            y_true=("y_true", "first"),
            pred=("pred", "mean"),
            confidence=("confidence", "mean"),
            n=("run", "size"),
        ).reset_index()
        y = agg["y_true"].values
        p_call = (agg["pred"] >= 0.5).astype(int).values
        accuracy = float((p_call == y).mean()) if len(y) else float("nan")
        deny_rate = float(p_call.mean()) if len(p_call) else float("nan")
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y, agg["pred"])) if len(y) > 1 and len(set(y)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        metric_rows.append({
            "provider": provider, "model": model,
            "n_profiles_scored": len(agg),
            "accuracy": accuracy, "deny_rate": deny_rate, "auc": auc,
            "mean_confidence": float(agg["confidence"].mean(skipna=True)),
        })
        wide = sub.pivot_table(index="row_index", columns="run", values="pred", aggfunc="first")
        if 0 in wide.columns and 1 in wide.columns:
            agree = float((wide[0] == wide[1]).mean())
        else:
            agree = float("nan")
        consistency_rows.append({
            "provider": provider, "model": model,
            "agreement_rate_runs": agree, "n_profiles": int(len(wide)),
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(consistency_rows)


# ---- Main ----
def main() -> None:
    banner("PHASE 6 — LLM credit assessment")
    _load_env()
    runners = register_runners()
    print(f"  registered runners: {len(runners)}")
    for prov, mdl, _ in runners:
        print(f"    - {prov:10s}  {mdl}")
    if not runners:
        print("  no providers — set at least one of "
              "ANTHROPIC_API_KEY/OPENAI_API_KEY/OPENROUTER_API_KEY.")
        return

    _, _, X_test, _, _, y_test = split_uci()
    test = X_test.copy()
    test["y"] = y_test.values
    pos = test[test["y"] == 1].sample(SAMPLE_PER_CLASS, random_state=SEED)
    neg = test[test["y"] == 0].sample(SAMPLE_PER_CLASS, random_state=SEED)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)
    sample = sample.reset_index().rename(columns={sample.index.name or "index": "row_index"})
    print(f"  prompt sample: {len(sample)} profiles "
          f"({int(sample['y'].sum())} default, {int((1-sample['y']).sum())} non-default)")

    existing = _load_existing_decisions()
    if existing is not None:
        print(f"  existing decisions on disk: {len(existing):,}")

    new_decisions = []
    api_log = []
    spend = 0.0
    aborted = False
    skipped = 0
    or_runners = [(p, m, fn) for p, m, fn in runners if p == "openrouter"]
    seq_runners = [(p, m, fn) for p, m, fn in runners if p != "openrouter"]
    print(f"  sequential commercial runners: {len(seq_runners)}  "
          f"sequential OR runners: {len(or_runners)}  (sample limit OR={OR_SAMPLE_LIMIT})")
    or_blocked: set[str] = set()
    or_consec_fails: dict[str, int] = {m: 0 for _, m, _ in or_runners}

    def _record(row, provider, model, run, text, in_t, out_t, err, t0):
        nonlocal spend
        cost = _cost_usd(model, in_t, out_t)
        spend += cost
        parsed = _parse_json(text) or {}
        decision = str(parsed.get("decision", "")).upper()
        new_decisions.append({
            "row_index": int(row["row_index"]),
            "y_true": int(row["y"]),
            "provider": provider, "model": model, "run": run,
            "raw_response": (text or "")[:2000],
            "decision": decision,
            "confidence": float(parsed.get("confidence", np.nan) or np.nan)
                           if parsed.get("confidence") not in (None, "") else np.nan,
            "top_reasons": json.dumps(parsed.get("top_reasons", []))[:500],
            "error": err,
        })
        api_log.append({
            "ts": time.time(), "provider": provider, "model": model,
            "input_tokens": in_t, "output_tokens": out_t, "cost_usd": cost,
            "cumulative_spend": spend, "elapsed_s": time.time() - t0,
            "error": err,
        })
        return decision

    # ---- Build commercial work list (skip already-done) ----
    seq_work = []
    for i, row in sample.iterrows():
        profile = _profile_text(row)
        prompt = PROMPT_TEMPLATE.format(profile=profile)
        for provider, model, fn in seq_runners:
            if _existing_done(existing, int(row["row_index"]), provider, model, RUNS_PER_PROFILE):
                skipped += 1
                continue
            for run in range(RUNS_PER_PROFILE):
                seq_work.append((row, prompt, provider, model, fn, run))
    print(f"  commercial work to do: {len(seq_work)} (skipped existing: {skipped})")

    # ---- OR runners (sequential, rate-limited; only if explicitly enabled) ----
    if or_runners:
        for i, row in sample.iterrows():
            if i >= OR_SAMPLE_LIMIT:
                break
            profile = _profile_text(row)
            prompt = PROMPT_TEMPLATE.format(profile=profile)
            for provider, model, fn in or_runners:
                if model in or_blocked:
                    continue
                if _existing_done(existing, int(row["row_index"]), provider, model, RUNS_PER_PROFILE):
                    skipped += RUNS_PER_PROFILE
                    continue
                profile_errors = 0
                for run in range(RUNS_PER_PROFILE):
                    t0 = time.time()
                    try:
                        text, in_t, out_t = fn(prompt)
                        err = ""
                    except Exception as e:  # noqa: BLE001
                        text, in_t, out_t = "", 0, 0
                        err = f"{type(e).__name__}: {e}"
                    _record(row, provider, model, run, text, in_t, out_t, err, t0)
                    if err: profile_errors += 1
                    time.sleep(OR_SLEEP_SEC)
                if profile_errors == RUNS_PER_PROFILE:
                    or_consec_fails[model] += 1
                else:
                    or_consec_fails[model] = 0
                if or_consec_fails[model] >= OR_BLOCK_AFTER_PROFILES:
                    or_blocked.add(model)

    # ---- Run commercial concurrently ----
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    spend_lock = threading.Lock()
    record_lock = threading.Lock()
    MAX_WORKERS = int(os.environ.get("PHASE6_WORKERS", "10"))

    def _do_one(row, prompt, provider, model, fn, run):
        nonlocal spend
        if spend >= SPEND_CAP_USD:
            return None
        t0 = time.time()
        try:
            text, in_t, out_t = fn(prompt)
            err = ""
        except Exception as e:  # noqa: BLE001
            text, in_t, out_t = "", 0, 0
            err = f"{type(e).__name__}: {e}"
        with record_lock:
            _record(row, provider, model, run, text, in_t, out_t, err, t0)
        return (row, provider, model, run)

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_do_one, *args) for args in seq_work]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"    futures error: {e}", flush=True); continue
            if res is None: continue
            completed += 1
            if completed % 50 == 0:
                with record_lock:
                    n = len(new_decisions)
                print(f"  --- progress: {completed}/{len(seq_work)} done, "
                      f"new_rows={n} spend=${spend:.4f} ---", flush=True)
                # Periodic flush so partial state is on disk
                with record_lock:
                    _flush(existing, new_decisions)
            if spend >= SPEND_CAP_USD:
                print(f"  spend cap ${SPEND_CAP_USD} hit — cancelling remaining.")
                for f2 in futures: f2.cancel()
                aborted = True
                break

    print(f"  total new decisions: {len(new_decisions)}, skipped: {skipped}, spend: ${spend:.4f}")

    # Combine and dedupe by (row_index, provider, model, run)
    combined = pd.DataFrame(new_decisions)
    if existing is not None:
        combined = pd.concat([existing, combined], ignore_index=True)
    if not combined.empty:
        combined = (combined
                    .drop_duplicates(subset=["row_index", "provider", "model", "run"], keep="last")
                    .reset_index(drop=True))
        combined.to_csv(RESULTS / "06_llm_decisions.csv", index=False)
        print(f"  wrote {RESULTS/'06_llm_decisions.csv'} (rows={len(combined):,})")

        m, c = _recompute_metrics(combined)
        m.to_csv(RESULTS / "06_llm_metrics.csv", index=False)
        c.to_csv(RESULTS / "06_consistency.csv", index=False)
        print(f"  wrote {RESULTS/'06_llm_metrics.csv'}, {RESULTS/'06_consistency.csv'}")
        print("\nMetrics:")
        print(m.to_string(index=False))
        print("\nConsistency:")
        print(c.to_string(index=False))

    # Append api_log
    log_path = RESULTS / "06_api_log.csv"
    new_log_df = pd.DataFrame(api_log)
    if log_path.exists():
        try:
            old = pd.read_csv(log_path)
            new_log_df = pd.concat([old, new_log_df], ignore_index=True)
        except Exception:
            pass
    new_log_df.to_csv(log_path, index=False)
    print(f"  wrote {log_path} (rows={len(new_log_df):,})")


if __name__ == "__main__":
    main()
