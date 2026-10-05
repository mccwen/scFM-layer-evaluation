#!/usr/bin/env python3
"""Representation quality metrics from layer activations and cell labels."""

# Credit: This code is adapted from Dr. Skean's GitHub repository listed below with permission
# https://github.com/OFSkean/information_flow

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import repitl.matrix_itl as itl
from dadapy.data import Data as IDData


def layer_index_from_name(name: str) -> int:
    match = re.search(r"layer_(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse layer index from {name}")
    return int(match.group(1))


def load_layers(path: Path, prefix: str) -> tuple[list[torch.Tensor], list[int]]:
    files = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No layer files found in {path} with prefix '{prefix}'")
    files = sorted(files, key=layer_index_from_name)

    layers: list[torch.Tensor] = []
    layer_ids: list[int] = []
    for f in files:
        tensor = torch.load(path / f, map_location="cpu")
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D tensor in {f}, got {tuple(tensor.shape)}")
        layers.append(tensor)
        layer_ids.append(layer_index_from_name(f))
    return layers, layer_ids


def entropy_denominator(mode: str, n_rows: int, n_cols: int) -> float:
    log_n = math.log(n_rows)
    log_d = math.log(n_cols)
    lookup = {
        "logN": log_n,
        "logD": log_d,
        "logNlogD": log_n * log_d,
        "maxEntropy": min(log_n, log_d),
        "length": float(n_rows),
    }
    if mode == "raw":
        return 1.0
    if mode not in lookup:
        raise ValueError(f"Unknown normalization: {mode}")
    return lookup[mode]


def trace_normalized_operator(x: np.ndarray) -> tuple[torch.Tensor, int, int]:
    arr = np.asarray(x, dtype=np.float64)
    n_rows, n_cols = arr.shape
    operator = arr.T @ arr if n_rows > n_cols else arr @ arr.T
    operator = np.clip(operator, a_min=0.0, a_max=None)
    trace_val = float(np.trace(operator))
    if not np.isfinite(trace_val) or trace_val <= 0:
        raise ValueError("Trace-normalized operator is undefined for zero-variance input.")
    normalized = operator / trace_val
    return torch.from_numpy(normalized).double(), n_rows, n_cols


def compute_matrix_entropy(x: np.ndarray, alpha: float = 1.0, normalization: str = "logD") -> float:
    operator, n_rows, n_cols = trace_normalized_operator(x)
    raw_entropy = float(itl.matrixAlphaEntropy(operator, alpha=alpha).item())
    denom = entropy_denominator(normalization, n_rows, n_cols)
    return raw_entropy if normalization == "raw" else raw_entropy / denom


def centered_covariance(x: np.ndarray) -> torch.Tensor:
    x_t = torch.as_tensor(x, dtype=torch.float64)
    centered = x_t - x_t.mean(dim=0, keepdim=True)
    n_rows = centered.shape[0]
    if n_rows < 2:
        raise ValueError("Effective rank requires at least two observations.")
    return (centered.T @ centered) / (n_rows - 1)


def compute_effective_rank(x: np.ndarray, eps: float = 1e-12) -> float:
    cov = centered_covariance(x)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=eps)
    spectrum = eigvals / eigvals.sum()
    spectral_entropy = -(spectrum * torch.log(spectrum)).sum()
    return float(torch.exp(spectral_entropy).item())


def estimate_intrinsic_dimension_2nn(x: np.ndarray) -> float:
    data = IDData(np.asarray(x, dtype=np.float64))
    intrinsic_dim, _, _ = data.compute_id_2NN()
    return float(intrinsic_dim)


def build_class_means(layer_emb: np.ndarray, labels: np.ndarray, label_names: list[str]) -> np.ndarray:
    means = []
    for label in label_names:
        idx = np.where(labels == label)[0]
        means.append(layer_emb[idx].mean(axis=0))
    return np.vstack(means)


def compute_metrics_for_dataset(
    adata,
    layers: list[torch.Tensor],
    layer_ids: list[int],
    label_key: str,
    min_label_cells: int,
    alpha: float,
    entropy_norm: str,
) -> pd.DataFrame:
    if label_key not in adata.obs:
        raise ValueError(f"Missing '{label_key}' in adata.obs")

    labels = adata.obs[label_key].astype(str).values
    label_names = [lb for lb in np.unique(labels) if (labels == lb).sum() >= min_label_cells]
    if not label_names:
        raise ValueError("No labels passed min-label-cells filter.")

    results = []
    for layer_id, layer_emb in zip(layer_ids, layers):
        layer_emb_np = layer_emb.detach().cpu().numpy()
        class_means = build_class_means(layer_emb_np, labels, label_names)

        ent_norm = compute_matrix_entropy(class_means, alpha=alpha, normalization=entropy_norm)
        ent_raw = compute_matrix_entropy(class_means, alpha=alpha, normalization="raw")
        eff_rank = compute_effective_rank(class_means)
        id_2nn = estimate_intrinsic_dimension_2nn(class_means)

        results.append(
            {
                "layer": layer_id,
                f"entropy_{entropy_norm}": ent_norm,
                "entropy_raw": ent_raw,
                "effective_rank": eff_rank,
                "intrinsic_dim_2nn": id_2nn,
                "n_labels": len(label_names),
            }
        )
        print(
            f"Layer {layer_id}: Ent({entropy_norm})={ent_norm:.4f}, "
            f"EffRank={eff_rank:.2f}, ID={id_2nn:.2f}"
        )

    return pd.DataFrame(results).sort_values("layer")


def plot_metrics(df: pd.DataFrame, dataset_name: str, entropy_norm: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    panels = [
        (f"entropy_{entropy_norm}", f"Matrix Entropy ({entropy_norm})", "tab:blue"),
        ("effective_rank", "Effective Rank", "tab:green"),
        ("intrinsic_dim_2nn", "Intrinsic Dimension (2NN)", "tab:orange"),
    ]
    for ax, (column, title, color) in zip(axes, panels):
        ax.plot(df["layer"], df[column], "-o", color=color, linewidth=2, markersize=5)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(f"Cell Type Representation Metrics ({dataset_name})")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_dataset_arg(items: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Dataset arg must be NAME=PATH, got: {item}")
        name, path = item.split("=", 1)
        parsed.append((name.strip(), Path(path).expanduser().resolve()))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute representation-quality metrics per layer.")
    parser.add_argument("--layer-dir", type=Path, required=True, help="Directory containing layer .pt files.")
    parser.add_argument("--layer-prefix", type=str, default="layer_", help="Layer file prefix.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset in NAME=PATH format. Repeat flag for multiple datasets.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("representation_quality_metrics"))
    parser.add_argument("--label-key", type=str, default="louvain", help="Label column in adata.obs.")
    parser.add_argument("--min-label-cells", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--entropy-norm",
        type=str,
        default="logD",
        choices=["logN", "logD", "logNlogD", "maxEntropy", "raw", "length"],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = args.layer_dir.expanduser().resolve()
    layers, layer_ids = load_layers(layer_dir, args.layer_prefix)
    print(f"Loaded {len(layer_ids)} layers: {layer_ids}")

    datasets = parse_dataset_arg(args.dataset)
    for dataset_name, dataset_path in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        adata = sc.read_h5ad(dataset_path)
        df = compute_metrics_for_dataset(
            adata=adata,
            layers=layers,
            layer_ids=layer_ids,
            label_key=args.label_key,
            min_label_cells=args.min_label_cells,
            alpha=args.alpha,
            entropy_norm=args.entropy_norm,
        )

        csv_path = args.output_dir / f"{dataset_name}_celltype_rep_quality_metrics_v2.csv"
        fig_path = args.output_dir / f"{dataset_name}_celltype_rep_quality_metrics_v2.svg"
        df.to_csv(csv_path, index=False)
        plot_metrics(df, dataset_name, args.entropy_norm, fig_path)
        print(f"Saved: {csv_path}")
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
