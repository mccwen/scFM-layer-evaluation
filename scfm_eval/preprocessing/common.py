from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import scanpy as sc


@dataclass(frozen=True)
class CellTypePreprocessResult:
    adata: object
    kept_labels: list[str]


CONTROL_LABEL_CANDIDATES = [
    "control",
    "ctrl",
    "vehicle",
    "dmso",
    "untreated",
    "wildtype",
    "wt",
    "none",
]


def require_obs_column(adata, key: str, role: str) -> None:
    if key not in adata.obs.columns:
        raise KeyError(
            f"Missing {role} column '{key}'. Available obs columns: {list(adata.obs.columns)}"
        )


def preprocess_cell_type_adata(
    adata,
    label_key: str,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 3,
    min_cells_per_class: int = 15,
    copy: bool = True,
) -> CellTypePreprocessResult:
    """Apply the manuscript cell-type preprocessing.

    Steps encoded here match the evaluation protocol:
    1. Verify the cell-type label column.
    2. Remove rare labels before stratified five-fold probing.
    3. Apply simple count-level QC filters used before model-specific tokenization.

    Model-specific scripts may additionally map genes to the model vocabulary or
    use model-required normalization/binning after this shared filtering step.
    """
    require_obs_column(adata, label_key, "cell-type label")
    adata = adata.copy() if copy else adata

    labels = adata.obs[label_key].astype(str)
    counts = labels.value_counts()
    kept_labels = counts[counts >= min_cells_per_class].index.astype(str).tolist()
    adata = adata[labels.isin(kept_labels).to_numpy()].copy()

    if min_genes_per_cell is not None and min_genes_per_cell > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
    if min_cells_per_gene is not None and min_cells_per_gene > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    return CellTypePreprocessResult(adata=adata, kept_labels=kept_labels)


def infer_control_label(values: Iterable[str], preferred: str | None = None) -> str:
    values = pd.Series(list(values)).astype(str)
    unique = values.unique().tolist()
    if preferred is not None and preferred in unique:
        return preferred
    lower_map = {v.lower(): v for v in unique}
    for candidate in CONTROL_LABEL_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    raise ValueError(
        "Could not infer control label. Set obs.control_label in the dataset catalog. "
        f"Available labels include: {unique[:20]}"
    )


def select_expression_matrix(adata, layer: str | None = None) -> np.ndarray:
    """Return dense float32 expression matrix for perturbation targets.

    The perturbation analyses use a log-normalized expression layer when provided
    by the dataset (for example ``logNor`` in Schmidt/Wessels workflows) and
    otherwise fall back to ``adata.X``. The returned matrix is dense because the
    ridge probes operate on NumPy arrays.
    """
    X = adata.layers[layer] if layer and layer in adata.layers else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def perturbation_size_folds(
    perturbations: Iterable[str],
    control_label: str,
    n_splits: int = 5,
    seed: int = 42,
) -> list[list[str]]:
    """Create the manuscript single-seed perturbation-level folds.

    Perturbations, not cells, are assigned to folds. Conditions are sorted by
    size before round-robin assignment so folds are approximately balanced by the
    number of cells per perturbation. The control condition is excluded from test
    folds and remains available for delta construction.
    """
    values = pd.Series(list(perturbations)).astype(str)
    counts = values.value_counts()
    pert_names = [p for p in counts.index.tolist() if p != control_label]
    if len(pert_names) < 2:
        raise ValueError("Need at least two non-control perturbations for perturbation CV.")
    rng = np.random.default_rng(seed)
    # Stable tie-breaking inside equal-sized perturbations.
    jitter = {p: rng.random() for p in pert_names}
    pert_names = sorted(pert_names, key=lambda p: (-counts[p], jitter[p]))
    folds = [[] for _ in range(min(n_splits, len(pert_names)))]
    for i, pert in enumerate(pert_names):
        folds[i % len(folds)].append(pert)
    return folds


def compute_control_delta_targets(
    expression: np.ndarray,
    perturbations: Iterable[str],
    control_label: str,
    contexts: Iterable[str] | None = None,
) -> np.ndarray:
    """Compute cell-level expression deltas relative to matched controls.

    If contexts are provided, controls are matched within context. If no context
    column exists, all cells share a single context named ``all``.
    """
    perturb = np.asarray(list(perturbations)).astype(str)
    if contexts is None:
        context = np.repeat("all", len(perturb))
    else:
        context = np.asarray(list(contexts)).astype(str)
    if expression.shape[0] != len(perturb):
        raise ValueError("Expression rows and perturbation labels are not aligned.")

    deltas = np.zeros_like(expression, dtype=np.float32)
    for ctx in np.unique(context):
        ctx_mask = context == ctx
        ctrl_mask = ctx_mask & (perturb == control_label)
        if ctrl_mask.sum() == 0:
            raise ValueError(f"No control cells found for context '{ctx}'.")
        deltas[ctx_mask] = expression[ctx_mask] - expression[ctrl_mask].mean(axis=0)
    return deltas
