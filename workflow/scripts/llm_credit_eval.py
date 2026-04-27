#!/usr/bin/env python3
"""Provider-agnostic LLM credit evaluation.

Dispatches to Anthropic, OpenAI, or Google APIs based on --provider flag.
Evaluates LLM zero-shot credit decisioning on sampled borrower profiles.

Usage:
  python scripts/llm_credit_eval.py --provider anthropic --model claude-opus-4-7
  python scripts/llm_credit_eval.py --provider openai --model gpt-4o
  python scripts/llm_credit_eval.py --provider google --model gemini-2.0-flash
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")

# ── Profile formatting ────��─────────────────────────────────────────────

SEX_MAP = {1: "male", 2: "female"}
EDU_MAP = {1: "graduate school", 2: "university", 3: "high school", 4: "other"}
MARRIAGE_MAP = {1: "married", 2: "single", 3: "other"}


def format_profile(row) -> str:
    """Format a UCI row as a structured borrower profile."""
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

    pay_avg = np.mean([row[c] for c in pay_cols])
    bill_avg = np.mean([row[c] for c in bill_cols])
    pay_amt_avg = np.mean([row[c] for c in pay_amt_cols])

    return (
        f"Assess this borrower:\n"
        f"- Credit limit (NTD): {row['LIMIT_BAL']:,.0f}\n"
        f"- Sex: {SEX_MAP.get(row['SEX'], 'unknown')}\n"
        f"- Education: {EDU_MAP.get(row['EDUCATION'], 'other')}\n"
        f"- Marital status: {MARRIAGE_MAP.get(row['MARRIAGE'], 'other')}\n"
        f"- Age: {row['AGE']:.0f}\n"
        f"- Repayment status (most recent month): {row['PAY_0']:.0f}\n"
        f"- Average repayment status over 6 months: {pay_avg:.2f}\n"
        f"- Average monthly bill (NTD): {bill_avg:,.0f}\n"
        f"- Average monthly payment (NTD): {pay_amt_avg:,.0f}"
    )


def load_prompt_template(template_name: str) -> str:
    """Load a prompt template from assets/prompt_templates/."""
    path = ROOT / "assets" / "prompt_templates" / template_name
    if not path.exists():
        print(f"WARNING: Template {path} not found, using minimal prompt.", file=sys.stderr)
        return (
            "You are a credit risk analyst. Given the borrower profile below, "
            "decide whether to APPROVE or DENY the application. "
            "Respond with JSON: {\"decision\": \"APPROVE\" or \"DENY\", "
            "\"confidence\": float 0-1, \"top_reasons\": [str, str, str]}"
        )
    return path.read_text().strip()


# ── Provider dispatch ─────────────���─────────────────────────────────────

def call_anthropic(system_prompt, user_message, model, temperature=0):
    """Call Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()
    t0 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    elapsed = time.time() - t0
    text = response.content[0].text
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "elapsed_s": elapsed,
    }
    return text, usage


def call_openai(system_prompt, user_message, model, temperature=0):
    """Call OpenAI API."""
    import openai
    client = openai.OpenAI()
    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
    )
    elapsed = time.time() - t0
    text = response.choices[0].message.content
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "elapsed_s": elapsed,
    }
    return text, usage


def call_google(system_prompt, user_message, model, temperature=0):
    """Call Google Generative AI API."""
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gen_model = genai.GenerativeModel(model, system_instruction=system_prompt)
    t0 = time.time()
    response = gen_model.generate_content(
        user_message,
        generation_config=genai.types.GenerationConfig(temperature=temperature, max_output_tokens=1024),
    )
    elapsed = time.time() - t0
    text = response.text
    # Google doesn't expose token counts the same way
    usage = {
        "input_tokens": getattr(response.usage_metadata, "prompt_token_count", 0),
        "output_tokens": getattr(response.usage_metadata, "candidates_token_count", 0),
        "elapsed_s": elapsed,
    }
    return text, usage


PROVIDERS = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "google": call_google,
}


# ── Response parsing ──────────────��─────────────────────────────────────

def parse_response(raw: str) -> dict:
    """Extract decision, confidence, and reasoning from LLM response."""
    # Try JSON parse
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```json\s*", "", raw)
        clean = re.sub(r"```\s*$", "", clean).strip()
        data = json.loads(clean)
        return {
            "decision": data.get("decision", "").upper(),
            "confidence": float(data.get("confidence", 0)),
            "default_probability": float(data.get("default_probability", -1)),
            "primary_factors": json.dumps(data.get("primary_factors", data.get("top_reasons", []))),
            "secondary_factors": json.dumps(data.get("secondary_factors", [])),
            "reasoning": data.get("reasoning", ""),
            "error": "",
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: regex extraction
    decision = ""
    if "APPROVE" in raw.upper():
        decision = "APPROVE"
    elif "DENY" in raw.upper():
        decision = "DENY"

    conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
    confidence = float(conf_match.group(1)) if conf_match else 0.0

    return {
        "decision": decision,
        "confidence": confidence,
        "default_probability": -1,
        "primary_factors": "[]",
        "secondary_factors": "[]",
        "reasoning": "",
        "error": "JSON parse failed; used regex fallback",
    }


# ── Main evaluation loop ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLM credit evaluation")
    parser.add_argument("--provider", required=True, choices=list(PROVIDERS.keys()))
    parser.add_argument("--model", required=True, help="Model string (e.g., claude-opus-4-7)")
    parser.add_argument("--template", default="zero_shot_underwriting.txt", help="Prompt template file")
    parser.add_argument("--n-profiles", type=int, default=200, help="Number of profiles to evaluate")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per profile")
    parser.add_argument("--spend-cap", type=float, default=30.0, help="Max API spend in USD")
    args = parser.parse_args()

    # Validate API key
    key_map = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
    key_name = key_map[args.provider]
    if not os.getenv(key_name):
        print(f"ERROR: {key_name} not set in .env", file=sys.stderr)
        sys.exit(1)

    # Load data and sample profiles
    from prepare_data import load_uci
    df = load_uci()
    features = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]
    target = "default payment next month"
    X = df[features].copy()
    y = df[target].copy()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED,
    )

    # Stratified sample from test set
    rng = np.random.default_rng(SEED)
    n = min(args.n_profiles, len(X_test))
    pos_idx = y_test[y_test == 1].index
    neg_idx = y_test[y_test == 0].index
    n_pos = int(n * y_test.mean())
    n_neg = n - n_pos
    sample_idx = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False),
    ])

    system_prompt = load_prompt_template(args.template)
    call_fn = PROVIDERS[args.provider]

    # Output files
    decisions_path = RESULTS / "llm_decisions.csv"
    api_log_path = RESULTS / "api_log.csv"

    # Check for existing results (resume logic)
    done_keys = set()
    if decisions_path.exists():
        existing = pd.read_csv(decisions_path)
        for _, row in existing.iterrows():
            done_keys.add((int(row["row_index"]), int(row["run"])))

    total_cost = 0.0
    completed = len(done_keys)

    # Pricing (approximate, per 1M tokens)
    pricing = {
        "anthropic": {"input": 15.0, "output": 75.0},
        "openai": {"input": 5.0, "output": 15.0},
        "google": {"input": 1.25, "output": 5.0},
    }
    price = pricing.get(args.provider, {"input": 10.0, "output": 50.0})

    for i, idx in enumerate(sample_idx):
        for run in range(1, args.runs + 1):
            if (idx, run) in done_keys:
                continue

            if total_cost >= args.spend_cap:
                print(f"\nSpend cap reached (${total_cost:.2f}). Stopping.")
                return

            row = df.loc[idx]
            profile_text = format_profile(row)
            y_true = int(y.loc[idx])

            try:
                raw_text, usage = call_fn(system_prompt, profile_text, args.model)
                parsed = parse_response(raw_text)

                # Estimate cost
                cost = (usage["input_tokens"] * price["input"] + usage["output_tokens"] * price["output"]) / 1e6
                total_cost += cost

            except Exception as e:
                parsed = {
                    "decision": "", "confidence": 0, "default_probability": -1,
                    "primary_factors": "[]", "secondary_factors": "[]",
                    "reasoning": "", "error": str(e),
                }
                raw_text = ""
                usage = {"input_tokens": 0, "output_tokens": 0, "elapsed_s": 0}
                cost = 0

            # Write decision
            with open(decisions_path, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow([
                        "row_index", "y_true", "provider", "model", "run",
                        "decision", "confidence", "default_probability",
                        "primary_factors", "secondary_factors", "reasoning", "error",
                    ])
                writer.writerow([
                    idx, y_true, args.provider, args.model, run,
                    parsed["decision"], parsed["confidence"], parsed["default_probability"],
                    parsed["primary_factors"], parsed["secondary_factors"],
                    parsed["reasoning"], parsed["error"],
                ])

            # Write API log
            with open(api_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow([
                        "timestamp", "provider", "model", "row_index", "run",
                        "input_tokens", "output_tokens", "elapsed_s", "cost_usd",
                    ])
                writer.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%S"), args.provider, args.model,
                    idx, run, usage["input_tokens"], usage["output_tokens"],
                    f"{usage['elapsed_s']:.2f}", f"{cost:.6f}",
                ])

            completed += 1
            print(f"[{completed}/{n * args.runs}] row={idx} y={y_true} "
                  f"decision={parsed['decision']} conf={parsed['confidence']:.2f} "
                  f"cost=${total_cost:.2f}")

    print(f"\nDone. Total cost: ${total_cost:.2f}")
    print(f"Decisions: {decisions_path}")
    print(f"API log: {api_log_path}")


if __name__ == "__main__":
    main()
