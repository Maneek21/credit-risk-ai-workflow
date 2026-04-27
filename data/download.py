"""Unified dataset downloader.

Thin convenience wrapper around the adapter ``download()`` methods.
Equivalent to ``python -m workflow.training download --dataset <name> --dest <dir>``;
provided as a script-friendly entry point for batch onboarding.

Usage:
    python data/download.py --dataset uci --dest data/raw/
    python data/download.py --dataset hmda --dest data/raw/
    python data/download.py --dataset bondora --dest data/raw/
"""
from __future__ import annotations

import argparse
import sys

from workflow.training.datasets import list_builtin_adapters, resolve_adapter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        choices=list_builtin_adapters(),
        help="Built-in adapter short name",
    )
    parser.add_argument("--dest", default="./data/raw",
                        help="Destination directory (default: ./data/raw)")
    args = parser.parse_args(argv)

    adapter = resolve_adapter(args.dataset)
    path = adapter.download(args.dest)
    print(f"Downloaded {args.dataset} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
