#!/usr/bin/env python3
"""Run one geneformer cell-type extraction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PIPELINES = {
    "czi": Path(__file__).with_name("extract_geneformer_czi.py"),
    "testis": Path(__file__).with_name("extract_geneformer_testis.py"),
    "zheng68k": Path(__file__).with_name("extract_geneformer_zheng68k.py"),
    "pbmc3k": Path(__file__).with_name("extract_geneformer_pbmc3k.py")}


def main():
    parser = argparse.ArgumentParser(
        description="Run exactly one dataset-specific geneformer extraction pipeline."
    )
    parser.add_argument(
        "--dataset",
        choices=['czi', 'testis', 'zheng68k', 'pbmc3k'],
        required=True,
        help="Dataset pipeline to run.",
    )
    args = parser.parse_args()

    pipeline = PIPELINES[args.dataset]
    command = [sys.executable, str(pipeline)]
    print("Running:", " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
