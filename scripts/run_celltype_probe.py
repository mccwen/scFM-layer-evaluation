#!/usr/bin/env python3
from __future__ import annotations

import argparse

from scfm_eval.probing.cell_type import run_cell_type_cv_from_activations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-fold cell-type probing from saved layer activations.")
    parser.add_argument("--layer-dir", required=True)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--min-cells-per-class",
        type=int,
        default=15,
        help="Minimum cells required per class before probing (default: 15).",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cells", type=int, default=None)
    args = parser.parse_args()
    run_cell_type_cv_from_activations(**vars(args))


if __name__ == "__main__":
    main()
