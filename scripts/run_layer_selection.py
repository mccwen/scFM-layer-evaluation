#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scfm_eval.reviewer_analyses.layer_selection import (
    earliest_on_par_layer,
    lodo_layer_selection,
    summarize_layer_folds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-selection utilities for reviewer analyses.")
    sub = parser.add_subparsers(dest="command", required=True)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--input-csv", required=True)
    summarize.add_argument("--metric", required=True)
    summarize.add_argument("--output-csv", required=True)

    eop = sub.add_parser("earliest-on-par")
    eop.add_argument("--input-csv", required=True)
    eop.add_argument("--metric", required=True)
    eop.add_argument("--final-layer", type=int, required=True)
    eop.add_argument("--margin", type=float, default=0.01)
    eop.add_argument("--relative-margin", action="store_true")

    lodo = sub.add_parser("lodo")
    lodo.add_argument("--metrics-csv", required=True)
    lodo.add_argument("--metric", required=True)
    lodo.add_argument("--output-csv", required=True)
    lodo.add_argument("--margin", type=float, default=0.01)
    lodo.add_argument("--relative-margin", action="store_true")

    args = parser.parse_args()
    if args.command == "summarize":
        df = summarize_layer_folds(args.input_csv, args.metric)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
    elif args.command == "earliest-on-par":
        print(
            earliest_on_par_layer(
                args.input_csv,
                args.metric,
                args.final_layer,
                args.margin,
                args.relative_margin,
            )
        )
    elif args.command == "lodo":
        df = lodo_layer_selection(args.metrics_csv, args.metric, margin=args.margin, relative_margin=args.relative_margin)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
