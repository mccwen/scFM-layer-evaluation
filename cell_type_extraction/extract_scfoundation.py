#!/usr/bin/env python3
"""Run one scfoundation cell-type extraction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PIPELINES = {
    "czi": Path(__file__).with_name("extract_scfoundation_czi.py"),
    "testis": Path(__file__).with_name("extract_scfoundation_testis.py"),
    "zheng68k": Path(__file__).with_name("extract_scfoundation_zheng68k.py"),
    "pbmc3k": Path(__file__).with_name("extract_scfoundation_pbmc3k.py")}


def main():
    parser = argparse.ArgumentParser(
        description="Run exactly one dataset-specific scfoundation extraction pipeline."
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
