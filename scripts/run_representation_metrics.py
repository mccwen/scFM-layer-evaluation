#!/usr/bin/env python3
from __future__ import annotations

import argparse

from scfm_eval.representation.metrics import compute_layer_representation_metrics, compute_layerwise_cka


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute representation metrics from saved layer activations.")
    parser.add_argument("--layer-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--metric", choices=["quality", "cka"], default="quality")
    parser.add_argument("--max-cells", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.metric == "quality":
        compute_layer_representation_metrics(args.layer_dir, args.output_csv, args.max_cells, args.seed)
    else:
        compute_layerwise_cka(args.layer_dir, args.output_csv, args.max_cells, args.seed)


if __name__ == "__main__":
    main()
