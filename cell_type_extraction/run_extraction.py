#!/usr/bin/env python3
"""Run one dataset-specific cell-type extraction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = {
    "geneformer": "extract_geneformer.py",
    "scbert": "extract_scbert.py",
    "scfoundation": "extract_scfoundation.py",
    "scgpt": "extract_scgpt.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dispatch one cell-type extraction pipeline for one scFM."
    )
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument(
        "--dataset",
        choices=["czi", "testis", "zheng68k", "pbmc3k"],
        required=True,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local scGPT checkpoint directory when --model=scgpt.",
    )
    args = parser.parse_args()

    dispatcher = Path(__file__).with_name(MODELS[args.model])
    command = [sys.executable, str(dispatcher), "--dataset", args.dataset]
    if args.model == "scgpt" and args.model_dir is not None:
        command.extend(["--model-dir", str(args.model_dir)])
    print("Running:", " ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
