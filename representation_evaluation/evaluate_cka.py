#!/usr/bin/env python3
"""Layer-wise linear CKA for saved scFM activations."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_layer_idx(path: Path) -> int:
    patterns = [
        r"layer_(\d+)_activations\.pt$",
        r"layer_(\d+)\.pt$",
        r"cell_emb_layer_(\d+)\.pt$",
    ]
    for pattern in patterns:
        match = re.search(pattern, path.name)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not parse layer index from {path.name}")


def find_layer_files(embedding_dir: Path) -> list[Path]:
    patterns = [
        "layer_*_activations.pt",
        "layer_*.pt",
        "cell_emb_layer_*.pt",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(embedding_dir.glob(pattern))
    files = sorted(set(files), key=parse_layer_idx)
    if not files:
        raise FileNotFoundError(f"No layer embeddings found in {embedding_dir}")
    return files


def infer_dataset_name(embedding_dir: Path) -> str:
    return embedding_dir.name.replace(" ", "_")


def load_tensor_2d(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        tensor_items = [(k, v) for k, v in obj.items() if isinstance(v, torch.Tensor)]
        if len(tensor_items) != 1:
            raise ValueError(
                f"Expected tensor or single-tensor dict in {path.name}; got keys: {list(obj.keys())}"
            )
        arr = tensor_items[0][1].detach().cpu().numpy()
    else:
        arr = np.asarray(obj)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D activations in {path.name}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"NaN/Inf detected in {path.name}")
    return arr.astype(np.float64, copy=False)


def subsample_rows(arrays: list[np.ndarray], max_cells: int, seed: int) -> list[np.ndarray]:
    n_rows = arrays[0].shape[0]
    for arr in arrays[1:]:
        if arr.shape[0] != n_rows:
            raise ValueError(f"Layer row-count mismatch: {n_rows} vs {arr.shape[0]}")
    if n_rows <= max_cells:
        return arrays
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n_rows, size=max_cells, replace=False))
    return [arr[idx] for arr in arrays]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y

    numerator = np.square(xty).sum()
    denominator = math.sqrt(np.square(xtx).sum() * np.square(yty).sum())
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def compute_cka_matrix(layer_arrays: list[np.ndarray]) -> np.ndarray:
    n_layers = len(layer_arrays)
    mat = np.eye(n_layers, dtype=np.float64)
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            score = linear_cka(layer_arrays[i], layer_arrays[j])
            mat[i, j] = score
            mat[j, i] = score
    return mat


def save_matrix_csv(path: Path, layer_ids: list[int], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + [f"layer_{i}" for i in layer_ids])
        for layer_id, row in zip(layer_ids, matrix):
            writer.writerow([f"layer_{layer_id}"] + [f"{x:.8f}" for x in row])


def plot_heatmap(path: Path, matrix: np.ndarray, layer_ids: list[int], title: str, cmap: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(layer_ids)))
    ax.set_yticks(range(len(layer_ids)))
    ax.set_xticklabels(layer_ids)
    ax.set_yticklabels(layer_ids)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Linear CKA")
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summarize_matrix(dataset: str, layer_ids: list[int], matrix: np.ndarray) -> dict[str, float | int | str]:
    off_diag = matrix[~np.eye(len(layer_ids), dtype=bool)]
    adjacent = [matrix[i, i + 1] for i in range(len(layer_ids) - 1)]
    return {
        "dataset": dataset,
        "n_layers": len(layer_ids),
        "mean_offdiag_cka": float(np.nanmean(off_diag)),
        "mean_adjacent_cka": float(np.nanmean(adjacent)) if adjacent else float("nan"),
        "min_adjacent_cka": float(np.nanmin(adjacent)) if adjacent else float("nan"),
        "max_adjacent_cka": float(np.nanmax(adjacent)) if adjacent else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute layer-wise linear CKA from .pt activations.")
    parser.add_argument("--embedding-dir", type=Path, required=True, help="Directory containing layer .pt files.")
    parser.add_argument("--output-root", type=Path, default=Path("cka"), help="Output directory root.")
    parser.add_argument("--dataset-name", type=str, default=None, help="Optional dataset name for output files.")
    parser.add_argument("--max-cells", type=int, default=3000, help="Max rows used per layer for CKA.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for row subsampling.")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap.")
    args = parser.parse_args()

    figure_dir = args.output_root / "figures"
    table_dir = args.output_root / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or infer_dataset_name(args.embedding_dir)
    layer_files = find_layer_files(args.embedding_dir)
    layer_ids = [parse_layer_idx(p) for p in layer_files]

    layer_arrays = [load_tensor_2d(p) for p in layer_files]
    layer_arrays = subsample_rows(layer_arrays, args.max_cells, args.seed)
    cka_matrix = compute_cka_matrix(layer_arrays)

    save_matrix_csv(table_dir / f"{dataset_name}_layerwise_cka.csv", layer_ids, cka_matrix)
    plot_heatmap(
        figure_dir / f"{dataset_name}_layerwise_cka.svg",
        cka_matrix,
        layer_ids,
        title=f"{dataset_name} | layer-wise CKA",
        cmap=args.cmap,
    )

    summary = summarize_matrix(dataset_name, layer_ids, cka_matrix)
    summary_path = table_dir / f"{dataset_name}_cka_summary.csv"
    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "dataset",
            "n_layers",
            "mean_offdiag_cka",
            "mean_adjacent_cka",
            "min_adjacent_cka",
            "max_adjacent_cka",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)

    print(f"Saved CKA matrix: {table_dir / f'{dataset_name}_layerwise_cka.csv'}")
    print(f"Saved CKA heatmap: {figure_dir / f'{dataset_name}_layerwise_cka.svg'}")
    print(f"Saved CKA summary: {summary_path}")


if __name__ == "__main__":
    main()
