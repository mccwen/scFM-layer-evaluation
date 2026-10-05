#!/usr/bin/env python3
"""Run one scfoundation perturbation extraction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PIPELINES = {
    "schmidt": Path(__file__).with_name("extract_scfoundation_schmidt.py"),
    "wessels": Path(__file__).with_name("extract_scfoundation_wessels.py"),
    "sciplex": Path(__file__).with_name("extract_scfoundation_sciplex.py")}


def main():
    parser = argparse.ArgumentParser(
        description="Run exactly one dataset-specific scfoundation extraction pipeline."
    )
    parser.add_argument(
        "--dataset",
        choices=['schmidt', 'wessels', 'sciplex'],
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
