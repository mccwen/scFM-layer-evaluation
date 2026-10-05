from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from scfm_eval.io import align_activation_rows, load_layer_activations


def load_cell_type_labels(
    h5ad_path: str | Path,
    label_key: str,
    min_cells_per_class: int = 5,
    max_cells: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load labels and row indices after rare-class filtering."""
    adata = sc.read_h5ad(h5ad_path)
    if label_key not in adata.obs:
        raise KeyError(f"Missing label column {label_key!r}; columns={list(adata.obs.columns)}")
    labels_full = adata.obs[label_key].astype(str).to_numpy()
    counts = pd.Series(labels_full).value_counts()
    keep_labels = counts[counts >= min_cells_per_class].index
    keep_idx = np.where(np.isin(labels_full, keep_labels))[0]
    if max_cells is not None and len(keep_idx) > max_cells:
        rng = np.random.default_rng(seed)
        keep_idx = np.sort(rng.choice(keep_idx, size=max_cells, replace=False))
    return labels_full[keep_idx], keep_idx, len(labels_full)


def run_cell_type_cv_from_activations(
    layer_dir: str | Path,
    h5ad_path: str | Path,
    label_key: str,
    output_csv: str | Path,
    min_cells_per_class: int = 5,
    n_splits: int = 5,
    seed: int = 42,
    max_cells: int | None = None,
    logreg_max_iter: int = 1000,
    logreg_n_jobs: int = 4,
) -> pd.DataFrame:
    """Run single-seed stratified CV for each saved activation layer.

    The saved CSV is intentionally per-fold/layer only: ``layer``, ``fold``,
    ``accuracy``, and ``macro_f1``. Aggregation and plotting are separate so
    the repository does not prescribe manuscript figures as outputs.
    """
    labels, keep_idx, n_original = load_cell_type_labels(
        h5ad_path=h5ad_path,
        label_key=label_key,
        min_cells_per_class=min_cells_per_class,
        max_cells=max_cells,
        seed=seed,
    )
    y = LabelEncoder().fit_transform(labels)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    layers = load_layer_activations(layer_dir)

    rows = []
    for layer_id, X_full in layers.items():
        X = align_activation_rows(X_full, keep_idx, len(y), n_original, layer_id)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y), start=1):
            clf = LogisticRegression(
                max_iter=logreg_max_iter,
                n_jobs=logreg_n_jobs,
                solver="lbfgs",
                multi_class="auto",
                random_state=seed,
            )
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            rows.append(
                {
                    "layer": layer_id,
                    "fold": fold,
                    "accuracy": accuracy_score(y[test_idx], pred),
                    "macro_f1": f1_score(y[test_idx], pred, average="macro"),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                }
            )
    result = pd.DataFrame(rows).sort_values(["layer", "fold"])
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    return result
