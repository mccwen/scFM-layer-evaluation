from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _metric_direction(metric: str) -> bool:
    lower_is_better = {"mse", "mse_delta", "rmse", "mae"}
    return metric not in lower_is_better


def summarize_layer_folds(input_csv: str | Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if metric not in df.columns:
        raise KeyError(f"Metric {metric!r} not found in {input_csv}; columns={list(df.columns)}")
    return (
        df.groupby("layer", as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"), n_folds=(metric, "count"))
        .sort_values("layer")
    )


def best_layer(input_csv: str | Path, metric: str) -> int:
    summary = summarize_layer_folds(input_csv, metric)
    idx = summary["mean"].idxmax() if _metric_direction(metric) else summary["mean"].idxmin()
    return int(summary.loc[idx, "layer"])


def earliest_on_par_layer(
    input_csv: str | Path,
    metric: str,
    final_layer: int,
    margin: float,
    relative_margin: bool = False,
) -> int:
    """Select earliest layer practically on par with the final layer."""
    summary = summarize_layer_folds(input_csv, metric)
    final_row = summary.loc[summary["layer"] == final_layer]
    if final_row.empty:
        raise ValueError(f"Final layer {final_layer} not present in {input_csv}")
    final_value = float(final_row.iloc[0]["mean"])
    higher_is_better = _metric_direction(metric)
    if relative_margin:
        tolerance = abs(final_value) * margin
    else:
        tolerance = margin
    if higher_is_better:
        eligible = summary[summary["mean"] >= final_value - tolerance]
    else:
        eligible = summary[summary["mean"] <= final_value + tolerance]
    if eligible.empty:
        return int(final_layer)
    return int(eligible["layer"].min())


def lodo_layer_selection(
    metrics_csv: str | Path,
    metric: str,
    dataset_col: str = "dataset",
    model_col: str = "model",
    final_layer_col: str = "final_layer",
    margin: float = 0.01,
    relative_margin: bool = False,
    selection_mode: str = "best",
) -> pd.DataFrame:
    """Leave-one-dataset-out layer selection from per-fold layer metrics.

    ``best`` selects the highest-performing source-dataset mean (or lowest
    error), resolving ties toward earlier layers. ``earliest-on-par`` selects
    the earliest layer within the supplied practical margin of the source
    final-layer mean; this is the perturbation-response mode.
    """
    if selection_mode not in {"best", "earliest-on-par"}:
        raise ValueError(
            "selection_mode must be 'best' or 'earliest-on-par', "
            f"got {selection_mode!r}"
        )
    df = pd.read_csv(metrics_csv)
    required = {dataset_col, model_col, "layer", metric, final_layer_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {metrics_csv}: {sorted(missing)}")
    rows = []
    for model, model_df in df.groupby(model_col):
        for heldout in sorted(model_df[dataset_col].unique()):
            train = model_df[model_df[dataset_col] != heldout]
            test = model_df[model_df[dataset_col] == heldout]
            final_layer = int(test[final_layer_col].iloc[0])
            # Give each source dataset equal weight after averaging its folds.
            train_summary = (
                train.groupby([dataset_col, "layer"], as_index=False)[metric]
                .mean()
                .groupby("layer", as_index=False)[metric]
                .mean()
            )
            final_value = float(train_summary.loc[train_summary["layer"] == final_layer, metric].iloc[0])
            if selection_mode == "best":
                if _metric_direction(metric):
                    selected = int(train_summary.loc[train_summary[metric].idxmax(), "layer"])
                else:
                    selected = int(train_summary.loc[train_summary[metric].idxmin(), "layer"])
            else:
                tolerance = abs(final_value) * margin if relative_margin else margin
                if _metric_direction(metric):
                    eligible = train_summary[train_summary[metric] >= final_value - tolerance]
                else:
                    eligible = train_summary[train_summary[metric] <= final_value + tolerance]
                selected = int(eligible["layer"].min()) if not eligible.empty else final_layer
            heldout_summary = test.groupby("layer", as_index=True)[metric].mean()
            heldout_best = int(heldout_summary.idxmax() if _metric_direction(metric) else heldout_summary.idxmin())
            heldout_means = test.groupby("layer")[metric].mean()
            rows.append(
                {
                    "model": model,
                    "heldout_dataset": heldout,
                    "metric": metric,
                    "selection_mode": selection_mode,
                    "selected_layer": selected,
                    "final_layer": final_layer,
                    "heldout_best_layer": heldout_best,
                    "selected_value": float(heldout_means.loc[selected]),
                    "final_value": float(heldout_means.loc[final_layer]),
                    "best_value": float(heldout_means.loc[heldout_best]),
                }
            )
    return pd.DataFrame(rows)
