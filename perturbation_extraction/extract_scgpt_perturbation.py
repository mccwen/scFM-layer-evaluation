#!/usr/bin/env python3
"""Run one scgpt perturbation extraction pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PIPELINES = {
    "schmidt": Path(__file__).with_name("extract_scgpt_schmidt.py"),
    "wessels": Path(__file__).with_name("extract_scgpt_wessels.py"),
    "sciplex": Path(__file__).with_name("extract_scgpt_sciplex.py")}


def main():
    parser = argparse.ArgumentParser(
        description="Run exactly one dataset-specific scgpt extraction pipeline."
    )
    parser.add_argument(
        "--dataset",
        choices=['schmidt', 'wessels', 'sciplex'],
        required=True,
        help="Dataset pipeline to run.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local checkpoint directory; used by the scGPT pipeline.",
    )
    args = parser.parse_args()

    env = os.environ.copy()

    if args.model_dir:
        env["SCGPT_MODEL_DIR"] = str(args.model_dir)
    pipeline = PIPELINES[args.dataset]
    command = [sys.executable, str(pipeline)]
    print("Running:", " ".join(command))
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
