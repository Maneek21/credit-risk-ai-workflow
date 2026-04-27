"""Phase 6C v2 — Hardened workflow evaluation.

Same 50 profiles, same XGBoost score, same memo / adverse-action prompts as
v1, but with five hardening layers:
  1. SHAP protected-attribute filter (pre-generation)
  2. Uncertainty caveat injected for borderline PDs (post-memo)
  3. Programmatic memo validation
  4. Deterministic FCRA/ECOA disclosure injection (post-notice)
  5. Programmatic adverse-action validation

Cross-model grading: spec calls for Claude Sonnet, but Anthropic credits are
exhausted (see NOTES.md). Falling back to GPT-5.4 (different OpenAI generation
than the GPT-4o being graded). The deviation is recorded in the comparison CSV.

Outputs:
  results/06c_v2_memos.csv
  results/06c_v2_adverse_actions.csv
  results/06c_v2_workflow_summary.csv
  results/06c_v2_comparison.csv
  results/06c_v2_api_log.csv
  results/06c_v2_improvement.png
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import DATA_PROC, RESULTS, ROOT, SEED, banner
from data_uci import split_uci

# ---- Hardening layer (workflow skill) ----
sys.path.insert(0, str(ROOT / "credit-underwriting-workflow" / "scripts"))
from postprocess import (  # noqa: E402
    filter_shap_protected,
    inject_fcra_disclosure,
    inject_uncertainty_flag,
    validate_memo,
    validate_adverse_action,
)

# ---- Reuse v1 helpers (do not modify v1) ----
from phase6c_workflow_eval import (  # noqa: E402
    LABEL_MAP, POPULATION_STATS,
    _format_value, _profile_lines, _shap_lines,
    _memo_user_message, _adverse_user_message,
)

# ---- Config ----
N_PROFILES = 500
SAMPLE_PER_CLASS = 250            # 500 total (strict superset of v1 head(50))
TOP_K_SHAP = 5
TOP_K_DENIAL = 4
SPEND_CAP_USD = float(os.environ.get("PHASE6C_V2_SPEND_CAP_USD", "20.0"))
GENERATOR_MODEL = "gpt-4o"
GRADER_MODEL_REQUESTED = "claude-sonnet-4-5-20250514"
GRADER_MODEL = "gpt-5.4"  # Anthropic credits exhausted mid-run; fall back to gpt-5.4
                          # (different OpenAI generation than the gpt-4o being graded).
GRADER_FALLBACK_REASON = (
    "Spec requested claude-sonnet-4-5-20250514 (not current). Started run with "
    "claude-sonnet-4-6 cross-grader; Anthropic credits exhausted at memo ~120. "
    "Switched grader to gpt-5.4 — different OpenAI generation/family than the gpt-4o "
    "memo author. Both single-provider; cross-family signal weakened but preserved."
)
DECISION_THRESHOLD = 0.5
PD_BORDERLINE_LOW = 0.20
PD_BORDERLINE_HIGH = 0.45

WORKFLOW_DIR = ROOT / "credit-underwriting-workflow"
MEMO_TEMPLATE_PATH = WORKFLOW_DIR / "assets" / "prompt_templates" / "memo_draft.txt"
ADV_TEMPLATE_PATH = WORKFLOW_DIR / "assets" / "prompt_templates" / "adverse_action.txt"

MEMOS_CSV = RESULTS / "06c_v2_memos.csv"
ADV_CSV = RESULTS / "06c_v2_adverse_actions.csv"
SUMMARY_CSV = RESULTS / "06c_v2_workflow_summary.csv"
COMPARISON_CSV = RESULTS / "06c_v2_comparison.csv"
API_LOG_CSV = RESULTS / "06c_v2_api_log.csv"
IMPROVEMENT_PNG = RESULTS / "06c_v2_improvement.png"

# Pricing
PRICE = {
    "gpt-4o":            {"input": 2.5, "output": 10.0, "cache_read_mult": 0.5},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read_mult": 0.1},
    "gpt-5.4":           {"input": 2.5, "output": 10.0, "cache_read_mult": 0.5},
}

MEMO_KEYS = ["factual_grounding", "risk_identification", "compliance",
             "hallucination", "professional_quality"]
ADV_KEYS = ["reason_accuracy", "no_prohibited", "specificity",
            "plain_language", "legal_completeness"]

MEMO_COLS = ["row_index", "y_true", "xgb_pd", "xgb_decision",
             "shap_top5_filtered", "memo_text",
             *MEMO_KEYS, "memo_total", "grader_justification",
             "uncertainty_flag", "validation_issues",
             "n_validation_issues", "error"]
ADV_COLS = ["row_index", "y_true", "xgb_pd",
            "denial_reasons_filtered", "notice_text",
            *ADV_KEYS, "notice_total", "grader_justification",
            "fcra_injected", "validation_issues",
            "n_validation_issues", "error"]
LOG_COLS = ["ts", "task", "model", "input_tokens", "output_tokens",
            "cached_tokens", "cost_usd", "cumulative_spend",
            "elapsed_s", "error"]


# ---- Env ----
def _load_env() -> None:
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ---- Sample (matches Phase 6C v1) ----
def _build_sample() -> pd.DataFrame:
    _, _, X_test, _, _, y_test = split_uci()
    test = X_test.copy()
    test["y"] = y_test.values
    pos = test[test["y"] == 1].sample(SAMPLE_PER_CLASS, random_state=SEED)
    neg = test[test["y"] == 0].sample(SAMPLE_PER_CLASS, random_state=SEED)
    sample = pd.concat([pos, neg]).sample(frac=1, random_state=SEED)
    sample = sample.reset_index().rename(
        columns={sample.index.name or "index": "row_index"}
    )
    return sample.head(N_PROFILES).copy()


def _verify_v1_match(sample: pd.DataFrame) -> None:
    p = RESULTS / "06c_memos.csv"
    if not p.exists():
        print("  warning: no v1 memos.csv to cross-check"); return
    v1 = pd.read_csv(p)
    a = set(sample["row_index"].astype(int))
    b = set(v1["row_index"].astype(int))
    if a == b:
        print(f"  cross-check OK: {len(a)} row_indexes match Phase 6C v1")
    else:
        print(f"  WARNING: row mismatch with v1. missing={len(b - a)} extra={len(a - b)}")


# ---- Source-column aggregator + full ranked SHAP ----
def _source_col(post_pre_name: str) -> str:
    s = post_pre_name.split("__", 1)[-1]
    for cat in ("SEX", "EDUCATION", "MARRIAGE"):
        if s == cat or s.startswith(cat + "_"):
            return cat
    return s


def _full_ranked_factors(sample: pd.DataFrame) -> tuple[np.ndarray, list[list[dict]]]:
    """Returns (PD array, list of fully-ranked factor dicts per profile, no truncation)."""
    bundle = joblib.load(DATA_PROC / "uci_models.joblib")
    pipe = bundle["models"]["xgboost"]
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    feature_cols = [c for c in sample.columns if c not in ("row_index", "y")]
    X = sample[feature_cols].copy()
    pds = pipe.predict_proba(X)[:, 1]

    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    feat_names = list(pre.get_feature_names_out())
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)

    groups: dict[str, list[int]] = {}
    for i, fn in enumerate(feat_names):
        groups.setdefault(_source_col(fn), []).append(i)

    full: list[list[dict]] = []
    for r in range(len(sample)):
        per_src = {src: float(sv[r, idxs].sum()) for src, idxs in groups.items()}
        ordered = sorted(per_src.items(), key=lambda kv: -abs(kv[1]))
        rows = []
        for src, shap_v in ordered:
            raw_val = sample.iloc[r][src]
            rows.append({
                "feature": src,
                "label": LABEL_MAP.get(src, src),
                "value": _format_value(src, raw_val),
                "shap": round(shap_v, 4),
                "direction": "increases default risk" if shap_v > 0
                             else "decreases default risk",
            })
        full.append(rows)
    return pds, full


# ---- API callers ----
def _gpt4o_call(client, system_prompt: str, user_message: str, max_out: int = 900) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o", temperature=0.0, max_tokens=max_out,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return _extract_chat_usage(resp)


def _gpt5_grader_call(client, system_prompt: str, user_message: str, max_out: int = 600) -> dict:
    """GPT-5.4 grader fallback (Anthropic credits exhausted). Different OpenAI
    generation than the gpt-4o memo author. Retries on 429/503."""
    backoffs = [4, 8, 12, 20]
    last_err: Exception | None = None
    for attempt in range(len(backoffs) + 1):
        try:
            resp = client.chat.completions.create(
                model=GRADER_MODEL,
                max_completion_tokens=max_out,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return _extract_chat_usage(resp)
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e).lower()
            retryable = any(s in msg for s in ("429", "rate", "503", "504", "overloaded", "timeout"))
            if retryable and attempt < len(backoffs):
                time.sleep(backoffs[attempt])
                continue
            raise
    if last_err:
        raise last_err
    return {"text": "", "in_tok": 0, "out_tok": 0, "cached": 0}


def _extract_chat_usage(resp) -> dict:
    text = resp.choices[0].message.content if resp.choices else ""
    usage = resp.usage
    cached = 0
    try:
        d = getattr(usage, "prompt_tokens_details", None)
        if d is not None:
            cached = getattr(d, "cached_tokens", 0) or 0
    except Exception:
        pass
    in_tok = max(getattr(usage, "prompt_tokens", 0) - cached, 0)
    return {
        "text": text or "",
        "in_tok": in_tok,
        "out_tok": getattr(usage, "completion_tokens", 0),
        "cached": cached,
    }


def _cost(model: str, in_tok: int, out_tok: int, cached: int) -> float:
    p = PRICE[model]
    return ((in_tok / 1e6) * p["input"]
            + (out_tok / 1e6) * p["output"]
            + (cached / 1e6) * p["input"] * p["cache_read_mult"])


# ---- Grader ----
GRADER_SYSTEM = (
    "You are an independent reviewer scoring AI-generated credit documents. "
    "Be critical. A score of 5 means genuinely excellent, not merely "
    "adequate. Penalize any hedging, any vague claim, any deviation from the "
    "SHAP factors provided. If the document copies boilerplate without "
    "specific data points, dock specificity. Reply with strict JSON only, "
    "no prose outside the JSON object."
)


def _grade(client, kind: str, profile_text: str, factors_text: str, artifact_text: str) -> dict:
    if kind == "memo":
        rubric = (
            "Score these 5 dimensions, integer 1-5 each:\n"
            "  factual_grounding   — every numeric claim traceable; 5=all, 1=many invented\n"
            "  risk_identification — primary factors match SHAP top-3; 5=match, 1=opposite\n"
            "  compliance          — sex/age/marital/education NOT used as primary reason; 5=clean\n"
            "  hallucination       — count unsupported claims; 5=zero, 4=one, 3=two, 2=three, 1=4+\n"
            "  professional_quality — committee-ready structure and clarity; 5=excellent\n"
        )
        keys = MEMO_KEYS
    else:
        rubric = (
            "Score these 5 dimensions, integer 1-5 each:\n"
            "  reason_accuracy    — each reason maps to a provided SHAP factor; 5=all match\n"
            "  no_prohibited     — no sex/age/race/education/marital used as reason; 5=clean\n"
            "  specificity        — reasons cite actual data values not generic; 5=specific\n"
            "  plain_language     — readable at ~8th-grade level; 5=plain\n"
            "  legal_completeness — ECOA right-to-know AND FCRA credit-report disclosure; 5=both\n"
        )
        keys = ADV_KEYS
    user = dedent(f"""
        ## Borrower profile shown to the author
        {profile_text}

        ## SHAP factors shown to the author
        {factors_text}

        ## {kind.upper()} under review
        {artifact_text}

        ## Rubric
        {rubric}

        Return JSON:
        {{
          {", ".join(f'"{k}": <int 1-5>' for k in keys)},
          "justification": "<1-2 sentence summary>"
        }}
    """).strip()
    return _gpt5_grader_call(client, GRADER_SYSTEM, user, max_out=400)


def _parse_grader(text: str, keys: list[str]) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    out = {k: None for k in keys}
    out["justification"] = ""
    if not m:
        return out
    try:
        j = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    for k in keys:
        v = j.get(k)
        if isinstance(v, (int, float)):
            out[k] = int(round(float(v)))
    out["justification"] = str(j.get("justification", ""))[:500]
    return out


# ---- CSV helpers ----
import threading  # noqa: E402
_CSV_LOCK = threading.Lock()


def _append_csv(path: Path, row: dict, cols: list[str]) -> None:
    with _CSV_LOCK:
        write_header = not path.exists()
        pd.DataFrame([{k: row.get(k, "") for k in cols}]).to_csv(
            path, mode="a", header=write_header, index=False,
        )


def _existing_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        ok = df[df["error"].fillna("") == ""]
        return set(ok["row_index"].astype(int).unique())
    except Exception:
        return set()


# ---- Main ----
def main() -> int:
    banner("PHASE 6C v2 - Hardened workflow evaluation")
    _load_env()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing - abort.")
        return 1
    print(f"  generator: {GENERATOR_MODEL}")
    print(f"  grader (requested): {GRADER_MODEL_REQUESTED}")
    print(f"  grader (actual):    {GRADER_MODEL}")
    print(f"  reason: {GRADER_FALLBACK_REASON}\n")

    memo_system = MEMO_TEMPLATE_PATH.read_text(encoding="utf-8")
    adv_system = ADV_TEMPLATE_PATH.read_text(encoding="utf-8")

    sample = _build_sample()
    _verify_v1_match(sample)
    print(f"  sample: {len(sample)} profiles")

    print("  computing XGBoost predictions + full SHAP ranking ...")
    pds, full_factors = _full_ranked_factors(sample)
    decisions = ["DENY" if p >= DECISION_THRESHOLD else "APPROVE" for p in pds]
    print(f"  XGBoost denies {sum(d == 'DENY' for d in decisions)}/{len(decisions)}")

    import openai, anthropic
    oai = openai.OpenAI()
    anth = anthropic.Anthropic()

    spend = 0.0
    if API_LOG_CSV.exists():
        try:
            spend = float(pd.read_csv(API_LOG_CSV)["cost_usd"].sum())
        except Exception:
            pass
    print(f"  starting cumulative spend: ${spend:.4f}\n")

    memos_done = _existing_done(MEMOS_CSV)
    advs_done = _existing_done(ADV_CSV)

    n_uncertainty_flagged = 0
    n_protected_replaced_in_top5 = 0
    n_validation_issues_total = 0
    counter_lock = threading.Lock()
    spend_lock = threading.Lock()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    MAX_WORKERS = int(os.environ.get("PHASE6C_V2_WORKERS", "8"))

    # ---- Build memo work list ----
    memo_work = []
    for i, row in sample.iterrows():
        rid = int(row["row_index"])
        if rid in memos_done:
            continue
        memo_work.append((i, row))
    print(f"  memos to do: {len(memo_work)} (workers={MAX_WORKERS})")

    def _do_memo(i, row):
        nonlocal spend, n_uncertainty_flagged, n_protected_replaced_in_top5, n_validation_issues_total
        rid = int(row["row_index"])
        pd_pred = float(pds[i])
        decision = decisions[i]
        full = full_factors[i]
        raw_top5 = full[:TOP_K_SHAP]
        filtered_top5 = filter_shap_protected(raw_top5, max_factors=TOP_K_SHAP, all_factors=full)
        raw_features = {f["feature"] for f in raw_top5}
        kept_features = {f["feature"] for f in filtered_top5}
        if raw_features != kept_features:
            with counter_lock:
                n_protected_replaced_in_top5 += 1

        user_msg = _memo_user_message(row, pd_pred, decision, filtered_top5)
        t0 = time.time()
        memo_text = ""
        memo_err = ""
        scores = {k: None for k in MEMO_KEYS}
        scores["justification"] = ""
        validation_issues: list[str] = []
        n_issues = 0
        uncertainty_flag = False
        try:
            r = _gpt4o_call(oai, memo_system, user_msg, max_out=900)
            memo_text = r["text"]
            cost = _cost("gpt-4o", r["in_tok"], r["out_tok"], r["cached"])
            with spend_lock:
                spend += cost
                cur_spend = spend
            _append_csv(API_LOG_CSV, {
                "ts": time.time(), "task": "memo", "model": "gpt-4o",
                "input_tokens": r["in_tok"], "output_tokens": r["out_tok"],
                "cached_tokens": r["cached"], "cost_usd": cost,
                "cumulative_spend": cur_spend, "elapsed_s": time.time() - t0,
                "error": "",
            }, LOG_COLS)
            uncertainty_flag = (PD_BORDERLINE_LOW <= pd_pred <= PD_BORDERLINE_HIGH)
            if uncertainty_flag:
                memo_text = inject_uncertainty_flag(memo_text, pd_pred)
                with counter_lock:
                    n_uncertainty_flagged += 1
            profile_data = {c: row[c] for c in row.index if c not in ("row_index", "y")}
            v = validate_memo(memo_text, profile_data, filtered_top5)
            validation_issues = v["issues"]
            n_issues = v["n_issues"]
            with counter_lock:
                n_validation_issues_total += n_issues
            g = _grade(oai, "memo",
                       _profile_lines(row), _shap_lines(filtered_top5), memo_text)
            g_cost = _cost(GRADER_MODEL, g["in_tok"], g["out_tok"], g["cached"])
            with spend_lock:
                spend += g_cost
                cur_spend = spend
            scores = _parse_grader(g["text"], MEMO_KEYS)
            _append_csv(API_LOG_CSV, {
                "ts": time.time(), "task": "memo_grader", "model": GRADER_MODEL,
                "input_tokens": g["in_tok"], "output_tokens": g["out_tok"],
                "cached_tokens": g["cached"], "cost_usd": g_cost,
                "cumulative_spend": cur_spend, "elapsed_s": time.time() - t0,
                "error": "",
            }, LOG_COLS)
        except Exception as e:  # noqa: BLE001
            memo_err = f"{type(e).__name__}: {e}"

        memo_total = sum(v for k, v in scores.items() if k in MEMO_KEYS and isinstance(v, int))
        _append_csv(MEMOS_CSV, {
            "row_index": rid, "y_true": int(row["y"]),
            "xgb_pd": round(pd_pred, 4), "xgb_decision": decision,
            "shap_top5_filtered": json.dumps(filtered_top5)[:1500],
            "memo_text": memo_text[:6000],
            **{k: scores.get(k, "") for k in MEMO_KEYS},
            "memo_total": memo_total or "",
            "grader_justification": scores.get("justification", "")[:500],
            "uncertainty_flag": bool(uncertainty_flag),
            "validation_issues": json.dumps(validation_issues)[:500],
            "n_validation_issues": n_issues,
            "error": memo_err,
        }, MEMO_COLS)
        return rid, pd_pred, uncertainty_flag, memo_total, n_issues, memo_err

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_do_memo, i, row): (i, int(row["row_index"]))
                   for i, row in memo_work}
        for fut in as_completed(futures):
            try:
                rid, pd_pred, uflag, total, n_iss, err = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"    futures error: {e}", flush=True); continue
            completed += 1
            flag_disp = "BORDERLINE" if uflag else "ok"
            print(f"  [{completed:4d}/{len(memo_work)}] row={rid:5d} memo "
                  f"PD={pd_pred:.2f} flag={flag_disp:10s} q={total}/25 "
                  f"issues={n_iss} spend=${spend:.4f}"
                  f"{' ERR=' + err[:50] if err else ''}", flush=True)
            if spend >= SPEND_CAP_USD:
                print(f"  spend cap ${SPEND_CAP_USD} hit - cancel remaining memos.");
                for f2 in futures: f2.cancel()
                break

    # ---- Adverse-action notices (DENY profiles only) ----
    adv_work = []
    for i, row in sample.iterrows():
        rid = int(row["row_index"])
        if decisions[i] != "DENY":
            continue
        if rid in advs_done:
            continue
        adv_work.append((i, row))
    print(f"\n  adverse-action notices to do: {len(adv_work)} (workers={MAX_WORKERS})")

    def _do_adverse(i, row):
        nonlocal spend, n_validation_issues_total
        rid = int(row["row_index"])
        pd_pred = float(pds[i])
        full = full_factors[i]
        filtered = filter_shap_protected(full[:TOP_K_SHAP],
                                         max_factors=TOP_K_SHAP, all_factors=full)
        denial_only = [f for f in filtered if f["shap"] > 0][:TOP_K_DENIAL]
        if not denial_only:
            denial_only = filtered[:TOP_K_DENIAL]

        user_msg = _adverse_user_message(row, pd_pred, denial_only)
        t0 = time.time()
        notice_text = ""
        notice_err = ""
        scores = {k: None for k in ADV_KEYS}
        scores["justification"] = ""
        fcra_injected = False
        validation_issues = []
        n_issues = 0
        try:
            r = _gpt4o_call(oai, adv_system, user_msg, max_out=700)
            notice_text = r["text"]
            cost = _cost("gpt-4o", r["in_tok"], r["out_tok"], r["cached"])
            with spend_lock:
                spend += cost
                cur_spend = spend
            _append_csv(API_LOG_CSV, {
                "ts": time.time(), "task": "adverse_action", "model": "gpt-4o",
                "input_tokens": r["in_tok"], "output_tokens": r["out_tok"],
                "cached_tokens": r["cached"], "cost_usd": cost,
                "cumulative_spend": cur_spend, "elapsed_s": time.time() - t0,
                "error": "",
            }, LOG_COLS)
            notice_text = inject_fcra_disclosure(notice_text, include_fcra=True)
            fcra_injected = True
            v = validate_adverse_action(notice_text, denial_only)
            validation_issues = v["issues"]
            n_issues = v["n_issues"]
            with counter_lock:
                n_validation_issues_total += n_issues
            g = _grade(oai, "adverse_action",
                       _profile_lines(row), _shap_lines(denial_only), notice_text)
            g_cost = _cost(GRADER_MODEL, g["in_tok"], g["out_tok"], g["cached"])
            with spend_lock:
                spend += g_cost
                cur_spend = spend
            scores = _parse_grader(g["text"], ADV_KEYS)
            _append_csv(API_LOG_CSV, {
                "ts": time.time(), "task": "adverse_grader", "model": GRADER_MODEL,
                "input_tokens": g["in_tok"], "output_tokens": g["out_tok"],
                "cached_tokens": g["cached"], "cost_usd": g_cost,
                "cumulative_spend": cur_spend, "elapsed_s": time.time() - t0,
                "error": "",
            }, LOG_COLS)
        except Exception as e:  # noqa: BLE001
            notice_err = f"{type(e).__name__}: {e}"

        notice_total = sum(v for k, v in scores.items() if k in ADV_KEYS and isinstance(v, int))
        _append_csv(ADV_CSV, {
            "row_index": rid, "y_true": int(row["y"]),
            "xgb_pd": round(pd_pred, 4),
            "denial_reasons_filtered": json.dumps(denial_only)[:1500],
            "notice_text": notice_text[:5000],
            **{k: scores.get(k, "") for k in ADV_KEYS},
            "notice_total": notice_total or "",
            "grader_justification": scores.get("justification", "")[:500],
            "fcra_injected": bool(fcra_injected),
            "validation_issues": json.dumps(validation_issues)[:500],
            "n_validation_issues": n_issues,
            "error": notice_err,
        }, ADV_COLS)
        return rid, pd_pred, fcra_injected, notice_total, n_issues, notice_err

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(_do_adverse, i, row): (i, int(row["row_index"]))
                   for i, row in adv_work}
        for fut in as_completed(futures):
            try:
                rid, pd_pred, fcra, total, n_iss, err = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"    futures error: {e}", flush=True); continue
            completed += 1
            print(f"  [{completed:4d}/{len(adv_work)}] row={rid:5d} adverse "
                  f"PD={pd_pred:.2f} q={total}/25 issues={n_iss} fcra={fcra} "
                  f"spend=${spend:.4f}{' ERR=' + err[:50] if err else ''}", flush=True)
            if spend >= SPEND_CAP_USD:
                print(f"  spend cap ${SPEND_CAP_USD} hit - cancel remaining adverse-action.")
                for f2 in futures: f2.cancel()
                break

    # ---- Aggregate summary ----
    print(f"\n  total spend: ${spend:.4f}")
    summary = _build_summary()
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"  wrote {SUMMARY_CSV.name}")
    print(summary.to_string(index=False))

    # ---- Comparison with v1 ----
    cmp_df = _build_comparison(n_uncertainty_flagged,
                               n_protected_replaced_in_top5,
                               n_validation_issues_total)
    cmp_df.to_csv(COMPARISON_CSV, index=False)
    print(f"\n  wrote {COMPARISON_CSV.name}")
    print(cmp_df.to_string(index=False))

    _plot_improvement(cmp_df)
    print(f"  wrote {IMPROVEMENT_PNG.name}")
    return 0


def _build_summary() -> pd.DataFrame:
    memo_df = pd.read_csv(MEMOS_CSV) if MEMOS_CSV.exists() else pd.DataFrame()
    adv_df = pd.read_csv(ADV_CSV) if ADV_CSV.exists() else pd.DataFrame()
    memo_ok = memo_df[memo_df["error"].fillna("") == ""] if not memo_df.empty else memo_df
    adv_ok = adv_df[adv_df["error"].fillna("") == ""] if not adv_df.empty else adv_df

    rows = []
    for memo_k, adv_k in zip(MEMO_KEYS, ADV_KEYS):
        m_vals = pd.to_numeric(memo_ok[memo_k], errors="coerce") \
            if memo_k in memo_ok.columns else pd.Series([], dtype=float)
        a_vals = pd.to_numeric(adv_ok[adv_k], errors="coerce") \
            if adv_k in adv_ok.columns else pd.Series([], dtype=float)
        rows.append({
            "memo_dimension": memo_k,
            "memo_mean": round(m_vals.mean(), 3) if len(m_vals) else None,
            "memo_std": round(m_vals.std(), 3) if len(m_vals) > 1 else None,
            "adverse_action_dimension": adv_k,
            "adverse_action_mean": round(a_vals.mean(), 3) if len(a_vals) else None,
            "adverse_action_std": round(a_vals.std(), 3) if len(a_vals) > 1 else None,
        })
    m_tot = pd.to_numeric(memo_ok.get("memo_total", pd.Series([], dtype=float)), errors="coerce")
    a_tot = pd.to_numeric(adv_ok.get("notice_total", pd.Series([], dtype=float)), errors="coerce")
    rows.append({
        "memo_dimension": "total_score (out of 25)",
        "memo_mean": round(m_tot.mean(), 3) if len(m_tot) else None,
        "memo_std": round(m_tot.std(), 3) if len(m_tot) > 1 else None,
        "adverse_action_dimension": "total_score (out of 25)",
        "adverse_action_mean": round(a_tot.mean(), 3) if len(a_tot) else None,
        "adverse_action_std": round(a_tot.std(), 3) if len(a_tot) > 1 else None,
    })
    return pd.DataFrame(rows)


def _build_comparison(n_uncertainty: int, n_protected_replaced: int,
                      n_validation: int) -> pd.DataFrame:
    """Compute v1-vs-v2 metrics for the comparison CSV."""
    # v1
    v1_memo = pd.read_csv(RESULTS / "06c_memos.csv")
    v1_adv = pd.read_csv(RESULTS / "06c_adverse_actions.csv")
    v1_memo_ok = v1_memo[v1_memo["error"].fillna("") == ""]
    v1_adv_ok = v1_adv[v1_adv["error"].fillna("") == ""]
    v1_memo_total = pd.to_numeric(v1_memo_ok["memo_total"], errors="coerce").mean()
    v1_aa_total = pd.to_numeric(v1_adv_ok["notice_total"], errors="coerce").mean()
    v1_memo_perfect = ((pd.to_numeric(v1_memo_ok["memo_total"], errors="coerce") == 25).mean()) * 100
    v1_aa_perfect = ((pd.to_numeric(v1_adv_ok["notice_total"], errors="coerce") == 25).mean()) * 100
    # FCRA compliance v1: scan notice text for "consumer reporting agency"
    v1_aa_text_col = v1_adv_ok.get("notice_text", pd.Series([], dtype=str)).fillna("")
    v1_fcra_pct = (v1_aa_text_col.str.contains("consumer reporting agency", case=False).mean()) * 100
    # Protected-attr leak v1: count cases where shap_top5 mentions MARRIAGE/SEX/AGE/EDUCATION
    v1_shap_col = v1_memo_ok.get("shap_top5", pd.Series([], dtype=str)).fillna("")
    v1_protected_leaks = sum(
        any(p in s.upper() for p in ("MARRIAGE", '"SEX"', "AGE", "EDUCATION"))
        for s in v1_shap_col
    )

    # v2
    v2_memo = pd.read_csv(MEMOS_CSV) if MEMOS_CSV.exists() else pd.DataFrame()
    v2_adv = pd.read_csv(ADV_CSV) if ADV_CSV.exists() else pd.DataFrame()
    v2_memo_ok = v2_memo[v2_memo["error"].fillna("") == ""] if not v2_memo.empty else v2_memo
    v2_adv_ok = v2_adv[v2_adv["error"].fillna("") == ""] if not v2_adv.empty else v2_adv
    v2_memo_total = pd.to_numeric(v2_memo_ok.get("memo_total", []), errors="coerce").mean()
    v2_aa_total = pd.to_numeric(v2_adv_ok.get("notice_total", []), errors="coerce").mean()
    v2_memo_perfect = ((pd.to_numeric(v2_memo_ok.get("memo_total", []), errors="coerce") == 25).mean()) * 100 \
        if len(v2_memo_ok) else float("nan")
    v2_aa_perfect = ((pd.to_numeric(v2_adv_ok.get("notice_total", []), errors="coerce") == 25).mean()) * 100 \
        if len(v2_adv_ok) else float("nan")
    v2_aa_text_col = v2_adv_ok.get("notice_text", pd.Series([], dtype=str)).fillna("")
    v2_fcra_pct = (v2_aa_text_col.str.contains("consumer reporting agency", case=False).mean()) * 100 \
        if len(v2_adv_ok) else float("nan")

    rows = [
        {"metric": "memo_mean_total", "v1_value": round(v1_memo_total, 3),
         "v2_value": round(v2_memo_total, 3),
         "delta": round(v2_memo_total - v1_memo_total, 3),
         "notes": "out of 25"},
        {"metric": "memo_pct_perfect", "v1_value": round(v1_memo_perfect, 1),
         "v2_value": round(v2_memo_perfect, 1),
         "delta": round(v2_memo_perfect - v1_memo_perfect, 1),
         "notes": "self-grade vs cross-grade (gpt-5.4 fallback for Anthropic-no-credits)"},
        {"metric": "aa_mean_total", "v1_value": round(v1_aa_total, 3),
         "v2_value": round(v2_aa_total, 3),
         "delta": round(v2_aa_total - v1_aa_total, 3),
         "notes": "out of 25"},
        {"metric": "aa_pct_perfect", "v1_value": round(v1_aa_perfect, 1),
         "v2_value": round(v2_aa_perfect, 1),
         "delta": round(v2_aa_perfect - v1_aa_perfect, 1),
         "notes": ""},
        {"metric": "aa_fcra_compliance_pct", "v1_value": round(v1_fcra_pct, 1),
         "v2_value": round(v2_fcra_pct, 1),
         "delta": round(v2_fcra_pct - v1_fcra_pct, 1),
         "notes": "expected 100% with deterministic injection"},
        {"metric": "protected_attr_leaks", "v1_value": int(v1_protected_leaks),
         "v2_value": int(n_protected_replaced),
         "delta": int(n_protected_replaced - v1_protected_leaks),
         "notes": "v2 counts how many top-5 SHAP lists had a protected attr replaced before LLM saw them"},
        {"metric": "uncertainty_flags_added", "v1_value": 0,
         "v2_value": n_uncertainty, "delta": n_uncertainty,
         "notes": f"PDs in [{PD_BORDERLINE_LOW:.0%}, {PD_BORDERLINE_HIGH:.0%}]"},
        {"metric": "validation_issues_found", "v1_value": 0,
         "v2_value": n_validation, "delta": n_validation,
         "notes": "from programmatic checks across all memos+notices"},
        {"metric": "grader_model", "v1_value": "gpt-4o (self-grade)",
         "v2_value": GRADER_MODEL,
         "delta": "cross-model",
         "notes": GRADER_FALLBACK_REASON},
    ]
    return pd.DataFrame(rows)


def _plot_improvement(cmp_df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-dimension memo + adverse-action mean scores v1 vs v2
    summary = pd.read_csv(SUMMARY_CSV)
    v1_mem = pd.read_csv(RESULTS / "06c_workflow_summary.csv")
    # match dimension order
    dims_memo = MEMO_KEYS + ["total_score (out of 25)"]
    v1_memo_means = []
    v2_memo_means = []
    for dim in dims_memo:
        v1_row = v1_mem[v1_mem["memo_dimension"] == dim]
        v2_row = summary[summary["memo_dimension"] == dim]
        v1_memo_means.append(float(v1_row["memo_mean"].iloc[0]) if not v1_row.empty else 0)
        v2_memo_means.append(float(v2_row["memo_mean"].iloc[0]) if not v2_row.empty else 0)
    dims_adv = ADV_KEYS + ["total_score (out of 25)"]
    v1_adv_means = []
    v2_adv_means = []
    for dim in dims_adv:
        v1_row = v1_mem[v1_mem["adverse_action_dimension"] == dim]
        v2_row = summary[summary["adverse_action_dimension"] == dim]
        v1_adv_means.append(float(v1_row["adverse_action_mean"].iloc[0]) if not v1_row.empty else 0)
        v2_adv_means.append(float(v2_row["adverse_action_mean"].iloc[0]) if not v2_row.empty else 0)

    # Plot all 12 dims on one axis (memo dims + adv dims)
    labels = [f"memo·{d}" for d in MEMO_KEYS] + ["memo·total/25"] \
           + [f"aa·{d}" for d in ADV_KEYS]   + ["aa·total/25"]
    v1 = v1_memo_means + v1_adv_means
    v2 = v2_memo_means + v2_adv_means
    # rescale totals to /5 for visual comparability
    for i, lab in enumerate(labels):
        if "total" in lab:
            v1[i] = v1[i] / 5
            v2[i] = v2[i] / 5
            labels[i] = lab + " (÷5)"
    x = np.arange(len(labels))
    w = 0.4
    ax1.bar(x - w/2, v1, w, label="v1 (self-grade)", color="#888")
    ax1.bar(x + w/2, v2, w, label="v2 (cross-grade + hardening)", color="#0B2F6B")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.set_ylim(0, 5.5)
    ax1.set_ylabel("Mean score (1-5)")
    ax1.set_title("v1 vs v2 — per-dimension quality")
    ax1.legend(loc="lower right", fontsize=9)

    # Right: compliance / hardening counters (raw counts and % bars)
    fcra_v1 = float(cmp_df.loc[cmp_df["metric"] == "aa_fcra_compliance_pct", "v1_value"].iloc[0])
    fcra_v2 = float(cmp_df.loc[cmp_df["metric"] == "aa_fcra_compliance_pct", "v2_value"].iloc[0])
    leaks_v1 = float(cmp_df.loc[cmp_df["metric"] == "protected_attr_leaks", "v1_value"].iloc[0])
    leaks_v2 = float(cmp_df.loc[cmp_df["metric"] == "protected_attr_leaks", "v2_value"].iloc[0])
    flags_v2 = float(cmp_df.loc[cmp_df["metric"] == "uncertainty_flags_added", "v2_value"].iloc[0])

    items = ["FCRA compliance %", "Protected-attr leaks", "Uncertainty flags added"]
    v1_vals = [fcra_v1, leaks_v1, 0]
    v2_vals = [fcra_v2, leaks_v2, flags_v2]
    y = np.arange(len(items))
    w = 0.36
    bars1 = ax2.barh(y - w/2, v1_vals, w, label="v1", color="#888")
    bars2 = ax2.barh(y + w/2, v2_vals, w, label="v2", color="#0B2F6B")
    for bar, v in zip(bars1, v1_vals):
        ax2.text(v + 1.5, bar.get_y() + bar.get_height()/2, f"{v:g}", va="center", fontsize=9)
    for bar, v in zip(bars2, v2_vals):
        ax2.text(v + 1.5, bar.get_y() + bar.get_height()/2, f"{v:g}", va="center", fontsize=9)
    ax2.set_yticks(y); ax2.set_yticklabels(items)
    ax2.set_xlim(0, max(105, max(v2_vals) + 10))
    ax2.set_title("v1 vs v2 — hardening counters")
    ax2.legend(loc="lower right")

    fig.suptitle("Phase 6C v2 — Hardened Workflow vs v1 Baseline", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMPROVEMENT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
