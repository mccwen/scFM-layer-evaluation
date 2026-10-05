from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


_LAYER_PATTERNS = [
    r"layer_(\d+)(?:_|\.pt$)",
    r"cell_emb_layer_(\d+)\.pt$",
    r"cls_emb_layer_(\d+)\.pt$",
    r"mean_emb_layer_(\d+)\.pt$",
]


def parse_layer_idx(path: str | Path) -> int:
    """Parse a layer index from common activation filenames."""
    name = Path(path).name
    for pattern in _LAYER_PATTERNS:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not parse layer index from {name}")


def find_layer_files(
    layer_dir: str | Path,
    patterns: Iterable[str] = ("layer_*.pt", "cell_emb_layer_*.pt", "cls_emb_layer_*.pt", "mean_emb_layer_*.pt"),
) -> list[Path]:
    """Return sorted activation tensor paths from a layer directory."""
    layer_dir = Path(layer_dir)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(layer_dir.glob(pattern))
    parsed = []
    for path in sorted(set(files)):
        try:
            parsed.append((parse_layer_idx(path), path))
        except ValueError:
            continue
    if not parsed:
        raise FileNotFoundError(f"No layer activation files found in {layer_dir}")
    return [path for _, path in sorted(parsed, key=lambda x: (x[0], x[1].name))]


def tensor_to_2d_array(obj, path: str | Path) -> np.ndarray:
    """Normalize saved activation payloads to a 2D float32 NumPy matrix."""
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        for key in ("embeddings", "activations", "X", "tensor"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                arr = value.detach().cpu().numpy()
                break
            if value is not None:
                arr = np.asarray(value)
                break
        else:
            tensor_items = [(k, v) for k, v in obj.items() if isinstance(v, torch.Tensor)]
            if len(tensor_items) != 1:
                raise ValueError(f"Expected tensor-like payload in {path}; keys={list(obj.keys())}")
            arr = tensor_items[0][1].detach().cpu().numpy()
    else:
        arr = np.asarray(obj)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D activations in {path}, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"NaN/Inf values detected in {path}")
    return arr.astype(np.float32, copy=False)


def load_activation_matrix(path: str | Path) -> np.ndarray:
    return tensor_to_2d_array(torch.load(path, map_location="cpu"), path)


def load_layer_activations(layer_dir: str | Path) -> dict[int, np.ndarray]:
    """Load all layer activation files as ``{layer_id: matrix}``."""
    layers: dict[int, np.ndarray] = {}
    for path in find_layer_files(layer_dir):
        layer = parse_layer_idx(path)
        if layer in layers:
            raise ValueError(f"Duplicate activation file for layer {layer}: {path}")
        layers[layer] = load_activation_matrix(path)
    return dict(sorted(layers.items()))


def align_activation_rows(
    X_full: np.ndarray,
    keep_idx: np.ndarray,
    n_filtered: int,
    n_original: int,
    layer_id: int | str,
) -> np.ndarray:
    """Align activations with filtered labels/targets.

    Saved activations are accepted when they are already filtered, match the
    original AnnData rows, or contain enough rows to apply ``keep_idx``.
    """
    n_rows = X_full.shape[0]
    if n_rows == n_filtered:
        return X_full
    if n_rows == n_original or (len(keep_idx) and n_rows > int(np.max(keep_idx))):
        return X_full[keep_idx]
    raise ValueError(
        f"Layer {layer_id} has {n_rows} rows; expected {n_filtered} filtered rows "
        f"or {n_original} original rows. Activations and labels may be misaligned."
    )
