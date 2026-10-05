from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from scfm_eval.io import load_layer_activations


def effective_rank(X: np.ndarray, eps: float = 1e-12) -> float:
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    vals = np.square(s)
    vals = vals[vals > eps]
    if vals.size == 0:
        return float("nan")
    probs = vals / vals.sum()
    return float(np.exp(-(probs * np.log(probs)).sum()))


def intrinsic_dim_2nn(X: np.ndarray) -> float:
    """Two-nearest-neighbor intrinsic dimension estimator."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] < 3:
        return float("nan")
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    distances, _ = nbrs.kneighbors(X)
    r1 = np.maximum(distances[:, 1], 1e-12)
    r2 = np.maximum(distances[:, 2], 1e-12)
    mu = r2 / r1
    valid = mu > 1
    if valid.sum() < 2:
        return float("nan")
    return float(1.0 / np.mean(np.log(mu[valid])))


def matrix_entropy_normalized(X: np.ndarray, eps: float = 1e-12) -> float:
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False)
    vals = np.square(s)
    vals = vals[vals > eps]
    if vals.size == 0:
        return float("nan")
    probs = vals / vals.sum()
    entropy = -(probs * np.log(probs)).sum()
    return float(entropy / math.log(len(probs))) if len(probs) > 1 else 0.0


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64) - np.mean(X, axis=0, keepdims=True)
    Y = np.asarray(Y, dtype=np.float64) - np.mean(Y, axis=0, keepdims=True)
    numerator = np.square(X.T @ Y).sum()
    denominator = math.sqrt(np.square(X.T @ X).sum() * np.square(Y.T @ Y).sum())
    return float(numerator / denominator) if denominator > 0 else float("nan")


def compute_layer_representation_metrics(
    layer_dir: str | Path,
    output_csv: str | Path,
    max_cells: int | None = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    layers = load_layer_activations(layer_dir)
    rows = []
    rng = np.random.default_rng(seed)
    for layer_id, X in layers.items():
        X_eval = X
        if max_cells is not None and X.shape[0] > max_cells:
            idx = np.sort(rng.choice(X.shape[0], size=max_cells, replace=False))
            X_eval = X[idx]
        rows.append(
            {
                "layer": layer_id,
                "n_cells": X_eval.shape[0],
                "n_features": X_eval.shape[1],
                "matrix_entropy_normalized": matrix_entropy_normalized(X_eval),
                "effective_rank": effective_rank(X_eval),
                "intrinsic_dim_2nn": intrinsic_dim_2nn(X_eval),
                "median_centered_norm": float(np.median(np.linalg.norm(X_eval - X_eval.mean(axis=0), axis=1))),
                "median_feature_std": float(np.median(np.std(X_eval, axis=0))),
            }
        )
    result = pd.DataFrame(rows).sort_values("layer")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result


def compute_layerwise_cka(
    layer_dir: str | Path,
    output_csv: str | Path,
    max_cells: int | None = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    layers = load_layer_activations(layer_dir)
    layer_ids = list(layers)
    n_rows = next(iter(layers.values())).shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows)
    if max_cells is not None and n_rows > max_cells:
        idx = np.sort(rng.choice(n_rows, size=max_cells, replace=False))
    arrays = {layer: X[idx] for layer, X in layers.items()}
    rows = []
    for a in layer_ids:
        for b in layer_ids:
            rows.append({"layer_a": a, "layer_b": b, "linear_cka": linear_cka(arrays[a], arrays[b])})
    result = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
