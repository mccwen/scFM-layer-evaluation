from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

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
    alpha: float = 1.0,
    top_deg_k: int | None = 100,
) -> pd.DataFrame:
    """Run perturbation-level CV using saved activations and delta targets.

    Perturbation conditions, not cells, are assigned to folds. Controls are kept
    in every training split to define matched expression deltas. The output is a
    compact per-layer/per-fold metrics table suitable for downstream layer
    selection analyses.
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
            train_mask = ~test_mask
            train_mask = train_mask | (perturb == control_label)
            eval_mask = test_mask & (perturb != control_label)
            if eval_mask.sum() == 0:
                continue
            model = Ridge(alpha=alpha)
            model.fit(X[train_mask], y_delta[train_mask])
            pred = model.predict(X[eval_mask])
            truth = y_delta[eval_mask]
            if top_deg_k is not None:
                cols = _top_delta_genes(truth, top_deg_k)
                pred_eval = pred[:, cols]
                truth_eval = truth[:, cols]
            else:
                pred_eval = pred
                truth_eval = truth
            rows.append(
                {
                    "layer": layer_id,
                    "fold": fold,
                    "heldout_perturbations": ";".join(map(str, heldout_perts)),
                    "mse_delta": mean_squared_error(truth_eval, pred_eval),
                    "pcc_delta": _safe_pearson(truth_eval, pred_eval),
                    "n_train": int(train_mask.sum()),
                    "n_test": int(eval_mask.sum()),
                    "control_label": control_label,
                }
            )
    result = pd.DataFrame(rows).sort_values(["layer", "fold"])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
