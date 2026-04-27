"""Minimal CLI for the training SDK — argparse only.

Subcommands:
    train      — run a full training pipeline from a YAML config
    download   — fetch a dataset to disk
    evaluate   — score a saved model against new data
    quickstart — run training with sensible defaults for a built-in dataset

Invocation:
    python -m workflow.training train --config configs/uci_us.yaml
    python -m workflow.training download --dataset uci --dest ./data/
    python -m workflow.training evaluate --model models/uci_xgboost_v1.joblib \\
        --data ./data/uci_default_credit.xls --adapter uci
    python -m workflow.training quickstart --dataset uci --jurisdiction US
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import joblib

from .datasets import list_builtin_adapters, resolve_adapter
from .evaluation import core_metrics
from .pipeline import TrainingConfig, TrainingPipeline


def _cmd_train(args: argparse.Namespace) -> int:
    pipeline = TrainingPipeline.from_yaml(args.config)
    result = pipeline.run()
    print(f"Model written to {result.model_path}")
    print(f"  AUC   = {result.metrics['auc']:.4f}")
    print(f"  KS    = {result.metrics['ks']:.4f}")
    print(f"  Brier = {result.metrics['brier']:.4f}")
    print(f"  ECE   = {result.metrics['ece']:.4f}")
    if result.fairness:
        print("Fairness (disparate impact ratios):")
        for attr, report in result.fairness.items():
            verdict = "PASS" if report["passes"] else "FAIL"
            print(f"  {attr}: {report['disparate_impact']:.3f} [{verdict}]")
    if result.shap_top_features:
        print("Top features by mean |SHAP|:")
        for f in result.shap_top_features[:5]:
            print(f"  {f['feature']}: {f['mean_abs_shap']:.4f}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    adapter = resolve_adapter(args.dataset)
    Path(args.dest).mkdir(parents=True, exist_ok=True)
    path = adapter.download(args.dest)
    print(f"Downloaded {args.dataset} -> {path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    adapter = resolve_adapter(args.adapter)
    model = joblib.load(args.model)
    X, y, _protected = adapter.load_with_protected(args.data)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = core_metrics(y.values, y_prob)
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_quickstart(args: argparse.Namespace) -> int:
    """Train with built-in defaults — for users who just want a model now."""
    cfg = TrainingConfig(
        dataset=args.dataset,
        data_path=args.data_path or f"./data/{args.dataset}_data",
        jurisdiction=args.jurisdiction,
        output_dir=args.output_dir or f"./output/{args.dataset}_{args.jurisdiction.lower()}",
    )
    if args.download:
        adapter = resolve_adapter(args.dataset)
        Path(cfg.data_path).parent.mkdir(parents=True, exist_ok=True)
        cfg.data_path = adapter.download(str(Path(cfg.data_path).parent))
    pipeline = TrainingPipeline(cfg)
    result = pipeline.run()
    print(f"Model: {result.model_path}")
    print(f"AUC: {result.metrics['auc']:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workflow.training",
        description="Configurable training SDK for credit-risk-ai-workflow.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a model from a YAML config")
    p_train.add_argument("--config", required=True, help="Path to YAML config file")
    p_train.set_defaults(func=_cmd_train)

    p_dl = sub.add_parser("download", help="download a built-in dataset")
    p_dl.add_argument("--dataset", required=True,
                      help=f"One of: {', '.join(list_builtin_adapters())}")
    p_dl.add_argument("--dest", default="./data", help="Destination directory")
    p_dl.set_defaults(func=_cmd_download)

    p_eval = sub.add_parser("evaluate", help="score a saved model on a dataset")
    p_eval.add_argument("--model", required=True, help="Path to .joblib model")
    p_eval.add_argument("--data", required=True, help="Path to data file")
    p_eval.add_argument("--adapter", required=True,
                        help="Adapter short name or dotted path")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_qs = sub.add_parser("quickstart", help="train with sensible defaults")
    p_qs.add_argument("--dataset", required=True,
                      choices=list_builtin_adapters())
    p_qs.add_argument("--jurisdiction", required=True,
                      help="ISO code: US, UK, EU, IN, CA, AU, SG, JP, AE, BR")
    p_qs.add_argument("--data-path", dest="data_path", default=None)
    p_qs.add_argument("--output-dir", dest="output_dir", default=None)
    p_qs.add_argument("--download", action="store_true",
                      help="Download the dataset before training")
    p_qs.set_defaults(func=_cmd_quickstart)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
