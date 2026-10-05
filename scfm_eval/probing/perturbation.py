from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from scfm_eval.io import align_activation_rows, load_layer_activations
from scfm_eval.preprocessing.common import (
    compute_control_delta_targets,
    infer_control_label,
    perturbation_size_folds,
    select_expression_matrix,
)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b)[0])


def _top_delta_genes(y_true: np.ndarray, k: int) -> np.ndarray:
    delta = np.mean(y_true, axis=0)
    return np.argsort(np.abs(delta))[::-1][: min(k, delta.size)]


def _retrieval_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    perturbations: np.ndarray,
) -> dict[str, float | bool]:
    """Compute held-out perturbation retrieval and collapse diagnostics."""
    names = np.unique(perturbations)
    pred_means = np.stack([predicted[perturbations == name].mean(axis=0) for name in names])
    truth_means = np.stack([truth[perturbations == name].mean(axis=0) for name in names])
    similarity = np.full((len(names), len(names)), np.nan, dtype=float)
    for i in range(len(names)):
        for j in range(len(names)):
            similarity[i, j] = _safe_pearson(pred_means[i], truth_means[j])

    top1 = []
    reciprocal_rank = []
    for i in range(len(names)):
        scores = np.nan_to_num(similarity[i], nan=-np.inf)
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(order == i)[0]) + 1
        top1.append(rank == 1)
        reciprocal_rank.append(1.0 / rank)

    pred_var = float(np.var(pred_means, axis=0).mean())
    truth_var = float(np.var(truth_means, axis=0).mean())
    variance_ratio = pred_var / truth_var if truth_var > 0 else float("nan")
    if len(names) >= 2:
        pred_pairwise = np.linalg.norm(
            pred_means[:, None, :] - pred_means[None, :, :], axis=2
        )
        truth_pairwise = np.linalg.norm(
            truth_means[:, None, :] - truth_means[None, :, :], axis=2
        )
        upper = np.triu(np.ones(pred_pairwise.shape, dtype=bool), k=1)
        pred_distance = float(pred_pairwise[upper].mean())
        truth_distance = float(truth_pairwise[upper].mean())
        pairwise_distance_ratio = (
            pred_distance / truth_distance if truth_distance > 0 else float("nan")
        )
    else:
        pairwise_distance_ratio = float("nan")
    unique_top1_fraction = float(
        np.unique(np.argmax(np.nan_to_num(similarity, nan=-np.inf), axis=1)).size
        / len(names)
    )
    collapse_warning = bool(
        (np.isfinite(variance_ratio) and variance_ratio < 0.10)
        or (
            np.isfinite(pairwise_distance_ratio)
            and pairwise_distance_ratio < 0.10
        )
        or unique_top1_fraction < 0.25
    )
    return {
        "retrieval_top1": float(np.mean(top1)),
        "retrieval_mrr": float(np.mean(reciprocal_rank)),
        "predicted_observed_variance_ratio": variance_ratio,
        "predicted_observed_pairwise_distance_ratio": pairwise_distance_ratio,
        "unique_top1_fraction": unique_top1_fraction,
        "collapse_warning": collapse_warning,
    }


def _aggregate_perturbations(
    X: np.ndarray,
    y_delta: np.ndarray,
    perturbations: np.ndarray,
    control_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average activations and delta targets once per non-control perturbation."""
    names = np.asarray(
        [name for name in pd.unique(perturbations) if name != control_label],
        dtype=str,
    )
    X_mean = np.stack([X[perturbations == name].mean(axis=0) for name in names])
    y_mean = np.stack([y_delta[perturbations == name].mean(axis=0) for name in names])
    return X_mean, y_mean, names


def run_perturbation_cv_from_activations(
    layer_dir: str | Path,
    h5ad_path: str | Path,
    perturbation_key: str,
    output_csv: str | Path,
    context_key: str | None = None,
    control_label: str | None = None,
    expression_layer: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
    alpha: float = 1e-4,
    top_deg_k: int | None = None,
    standardize_features: bool = True,
) -> pd.DataFrame:
    """Run perturbation-level CV using saved activations and delta targets.

    Perturbation conditions, not cells, are assigned to folds. Controls are kept
    in every training split to define matched expression deltas. For each fold,
    cell activations and cell-level deltas are averaged within each non-control
    perturbation before fitting multi-output ridge regression. Feature scaling
    is fit on the resulting training-perturbation means only. If ``top_deg_k``
    is supplied, genes are selected from training perturbation means only;
    held-out targets are never used for feature selection.
    """
    adata = sc.read_h5ad(h5ad_path)
    if perturbation_key not in adata.obs:
        raise KeyError(f"Missing perturbation column {perturbation_key!r}")
    perturb = adata.obs[perturbation_key].astype(str).to_numpy()
    context = adata.obs[context_key].astype(str).to_numpy() if context_key else None
    control_label = infer_control_label(perturb, preferred=control_label)

    expression = select_expression_matrix(adata, layer=expression_layer)
    y_delta = compute_control_delta_targets(expression, perturb, control_label, context)
    keep_idx = np.arange(adata.n_obs)
    layers = load_layer_activations(layer_dir)
    folds = perturbation_size_folds(perturb, control_label, n_splits=n_splits, seed=seed)

    rows = []
    for layer_id, X_full in layers.items():
        X = align_activation_rows(X_full, keep_idx, adata.n_obs, adata.n_obs, layer_id)
        for fold, heldout_perts in enumerate(folds, start=1):
            test_mask = np.isin(perturb, heldout_perts)
            train_mask = (~test_mask) & (perturb != control_label)
            eval_mask = test_mask & (perturb != control_label)
            if train_mask.sum() == 0 or eval_mask.sum() == 0:
                continue
            X_train, y_train, train_names = _aggregate_perturbations(
                X[train_mask], y_delta[train_mask], perturb[train_mask], control_label
            )
            X_eval, truth, eval_names = _aggregate_perturbations(
                X[eval_mask], y_delta[eval_mask], perturb[eval_mask], control_label
            )
            if standardize_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_eval = scaler.transform(X_eval)
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            pred = model.predict(X_eval)
            if top_deg_k is not None:
                cols = _top_delta_genes(y_train, top_deg_k)
                pred_eval = pred[:, cols]
                truth_eval = truth[:, cols]
            else:
                pred_eval = pred
                truth_eval = truth
            retrieval = _retrieval_metrics(pred_eval, truth_eval, eval_names)
            rows.append(
                {
                    "layer": layer_id,
                    "fold": fold,
                    "heldout_perturbations": ";".join(map(str, heldout_perts)),
                    "mse_delta": mean_squared_error(truth_eval, pred_eval),
                    "pcc_delta": _safe_pearson(truth_eval, pred_eval),
                    "n_train": int(len(train_names)),
                    "n_test": int(len(eval_names)),
                    "n_train_cells": int(train_mask.sum()),
                    "n_test_cells": int(eval_mask.sum()),
                    "control_label": control_label,
                    "alpha": alpha,
                    "standardize_features": standardize_features,
                    **retrieval,
                }
            )
    result = pd.DataFrame(rows).sort_values(["layer", "fold"])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
