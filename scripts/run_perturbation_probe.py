#!/usr/bin/env python3
from __future__ import annotations

import argparse

from scfm_eval.probing.perturbation import run_perturbation_cv_from_activations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run perturbation-level CV from saved layer activations.")
    parser.add_argument("--layer-dir", required=True)
    parser.add_argument("--h5ad", dest="h5ad_path", required=True)
    parser.add_argument("--perturbation-key", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--context-key", default=None)
    parser.add_argument("--control-label", default=None)
    parser.add_argument("--expression-layer", default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top-deg-k", type=int, default=100)
    args = parser.parse_args()
    run_perturbation_cv_from_activations(**vars(args))


if __name__ == "__main__":
    main()
