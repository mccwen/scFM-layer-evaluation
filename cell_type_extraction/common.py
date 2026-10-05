from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_numpy_2d(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}.")
    return arr


def save_layer_pt_files(
    layer_embeddings: Mapping[int, torch.Tensor | np.ndarray],
    out_dir: Path,
    prefix: str = "layer_",
) -> Dict[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[int, Path] = {}
    for layer_idx in sorted(layer_embeddings.keys()):
        arr = to_numpy_2d(layer_embeddings[layer_idx])
        path = out_dir / f"{prefix}{layer_idx}.pt"
        torch.save(torch.from_numpy(arr.astype(np.float32, copy=False)), path)
        saved[layer_idx] = path
    return saved


def save_all_layers_csv(
    layer_embeddings: Mapping[int, torch.Tensor | np.ndarray],
    out_csv: Path,
    obs_names: Iterable[str] | None = None,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    layer_ids = sorted(layer_embeddings.keys())
    first = to_numpy_2d(layer_embeddings[layer_ids[0]])
    n_cells, emb_dim = first.shape

    if obs_names is None:
        obs_names = [str(i) for i in range(n_cells)]
    else:
        obs_names = [str(x) for x in obs_names]
    if len(obs_names) != n_cells:
        raise ValueError(
            f"obs_names length {len(obs_names)} does not match n_cells {n_cells}."
        )

    dim_cols = [f"dim_{i}" for i in range(emb_dim)]
    header = ["layer", "cell_index", "obs_name"] + dim_cols
    first_write = True

    for layer_idx in layer_ids:
        arr = to_numpy_2d(layer_embeddings[layer_idx])
        if arr.shape != (n_cells, emb_dim):
            raise ValueError(
                f"Layer {layer_idx} has shape {arr.shape}, expected {(n_cells, emb_dim)}."
            )
        df = pd.DataFrame(arr, columns=dim_cols)
        df.insert(0, "obs_name", obs_names)
        df.insert(0, "cell_index", np.arange(n_cells, dtype=np.int64))
        df.insert(0, "layer", int(layer_idx))
        df.to_csv(
            out_csv,
            mode="w" if first_write else "a",
            index=False,
            header=first_write,
        )
        first_write = False


def save_run_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def add_common_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data_path", type=str, required=True, help="Input .h5ad file.")
    parser.add_argument(
        "--cell_type_col",
        type=str,
        default="cell_type",
        help="Cell-type column in adata.obs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: model-specific name beside input data.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--min_genes_per_cell",
        type=int,
        default=200,
        help="QC filter: minimum genes per cell.",
    )
    parser.add_argument(
        "--min_cells_per_gene",
        type=int,
        default=3,
        help="QC filter: minimum cells per gene.",
    )
    parser.add_argument(
        "--min_cells_per_class",
        type=int,
        default=15,
        help="Class-size filter: minimum cells per cell type.",
    )
    return parser


def default_output_dir(data_path: str, model_name: str) -> Path:
    stem = Path(data_path).stem
    return Path.cwd() / f"{model_name}_{stem}_layer_activations"
