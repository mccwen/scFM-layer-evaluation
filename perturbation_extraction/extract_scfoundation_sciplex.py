"""Layer-wise scfoundation pipeline for SciPlex.

This file contains one dataset pipeline only. Raw data, checkpoints, and
generated activations are supplied through local configuration.
"""

import os

import sys

import math

import random

import numpy as np

import pandas as pd

import torch

import scanpy as sc

import scipy.sparse

import matplotlib.pyplot as plt

from tqdm import tqdm

from pathlib import Path

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error


# ==========================================
# Config
# ==========================================
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "models")
CKPT_PATH = os.path.join(MODEL_DIR, "models.ckpt")
GENE_INDEX_PATH = os.path.join(BASE_DIR, "OS_scRNA_gene_index.19264.tsv")

DATA_PATH = "combo_sciplex_prep_hvg_filtered.h5ad"
OUTPUT_DIR = os.path.join(BASE_DIR, "scFoundation_sciplex_prep_hvg_unseen_gene_probe_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 4
K_FOLDS = 5
MIN_PERT_CELLS = 25
RIDGE_L2 = 1e-4

PERT_COL_OVERRIDE = "condition"
CONTROL_LABEL_OVERRIDE = "control"

# ==========================================
# Reproducibility
# ==========================================
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ==========================================
# Setup
# ==========================================
sys.path.append(BASE_DIR)
from load import load_model_frommmf, gatherData  # noqa: E402

# ==========================================
# Gene Alignment
# ==========================================
def main_gene_selection(X_df, gene_list):
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(
        np.zeros((X_df.shape[0], len(to_fill_columns))),
        columns=to_fill_columns,
        index=X_df.index
    )
    X_df = pd.concat([X_df, padding_df], axis=1)
    return X_df[gene_list]

# ==========================================
# Load SciPlex
# ==========================================
def load_sciplex(gene_index_path):
    adata = sc.read_h5ad(DATA_PATH)
    adata.obs_names_make_unique()

    X = adata.X
    print("Using adata.X (already log-transformed)")

    if "gene_name" in adata.var.columns:
        gene_symbols = adata.var["gene_name"].astype(str)
    elif "feature_name" in adata.var.columns:
        gene_symbols = adata.var["feature_name"].astype(str)
    else:
        gene_symbols = adata.var_names.astype(str)

    gene_symbols = gene_symbols.str.upper()

    df = pd.DataFrame(
        X.toarray() if scipy.sparse.issparse(X) else X,
        index=adata.obs_names,
        columns=gene_symbols
    )

    gene_list = pd.read_csv(gene_index_path, sep="\t")["gene_name"].astype(str).str.upper().tolist()
    df = main_gene_selection(df, gene_list)
    df = df.loc[:, ~df.columns.duplicated()]

    matched = np.isin(gene_list, df.columns).sum()
    print(f"Matched genes to index: {matched} / {len(gene_list)}")
    if matched == 0:
        raise ValueError("No genes matched the index list. Check gene naming.")

    norm_data = df.values.astype(np.float32)
    row_sums = norm_data.sum(axis=1, keepdims=True).astype(np.float32)
    print("Expr stats:", np.min(norm_data), np.max(norm_data), np.mean(norm_data))
    return adata, norm_data, row_sums

# ==========================================
# Pooling (faithful to scFoundation)
# ==========================================
def pool_hidden(h, pad_mask, pool_style="all"):
    valid = ~pad_mask
    valid_f = valid.float().unsqueeze(-1)
    B, T, D = h.shape

    counts = valid.sum(dim=1).clamp(min=1).long()
    idx_last = counts - 1
    idx_second = torch.clamp(counts - 2, min=0)

    last = h[torch.arange(B, device=h.device), idx_last]
    second = h[torch.arange(B, device=h.device), idx_second]
    mean = (h * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1)
    h_masked = h.masked_fill(~valid.unsqueeze(-1), -1e9)
    maxv = h_masked.max(dim=1).values

    if pool_style == "cls":
        return h[:, 0]
    if pool_style == "mean":
        return mean
    if pool_style == "max":
        return maxv
    if pool_style == "all":
        return torch.cat([last, second, maxv, mean], dim=-1)
    return mean

# ==========================================
# Hook utilities
# ==========================================
def get_activation_capturer(model):
    activations = {}
    hooks = []

    def save_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            activations[name] = output
        return hook

    def register_hooks():
        for h in hooks:
            h.remove()
        hooks.clear()
        activations.clear()
        encoder = model.encoder
        layers = list(encoder.transformer_encoder)
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(save_activation(f"layer_{i+1}")))
        return len(layers)

    def remove_hooks():
        for h in hooks:
            h.remove()

    return activations, register_hooks, remove_hooks

# ==========================================
# Perturbation helpers
# ==========================================
def pick_perturbation_column(adata):
    if PERT_COL_OVERRIDE is not None:
        return PERT_COL_OVERRIDE
    candidates = [
        "perturbation", "pert", "condition", "treatment",
        "perturbation_group", "perturbation_label", "target_gene"
    ]
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise ValueError("No suitable perturbation column found in adata.obs.")

def pick_control_label(values):
    if CONTROL_LABEL_OVERRIDE is not None:
        return CONTROL_LABEL_OVERRIDE
    control_candidates = ["control", "ctrl", "vehicle", "dmso", "untreated", "wildtype", "wt", "none"]
    lower_map = {v.lower(): v for v in values}
    for c in control_candidates:
        if c in lower_map:
            return lower_map[c]
    raise ValueError("No control label found. Set CONTROL_LABEL_OVERRIDE manually.")

def make_size_folds(pert_names, pert_labels, k=5, seed=SEED):
    rng = np.random.default_rng(seed)
    counts = {p: int((pert_labels == p).sum()) for p in pert_names}
    pert_sorted = sorted(pert_names, key=lambda p: counts[p], reverse=True)

    folds = [[] for _ in range(k)]
    fold_sizes = [0] * k
    for p in pert_sorted:
        idx = int(np.argmin(fold_sizes))
        folds[idx].append(p)
        fold_sizes[idx] += counts[p]
    return folds

def pearson_corr_safe(x, y, eps=1e-8):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.std() < eps or y.std() < eps:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)

def compute_delta_metrics_with_train_control(
    y_test, y_pred, test_labels,
    y_train, y_pred_train, train_labels, control_label
):
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    ctrl_mask_train = train_labels == control_label
    if ctrl_mask_train.sum() == 0:
        raise ValueError("No control cells found in TRAIN split.")

    pred_ctrl_mean = y_pred_train[ctrl_mask_train].mean(axis=0)
    true_ctrl_mean = y_train[ctrl_mask_train].mean(axis=0)

    perts = [p for p in np.unique(test_labels) if p != control_label]
    mse_list, pcc_list = [], []

    for p in perts:
        mask = test_labels == p
        if mask.sum() == 0:
            continue
        true_mean = y_test[mask].mean(axis=0)
        pred_mean = y_pred[mask].mean(axis=0)
        delta_true = true_mean - true_ctrl_mean
        delta_pred = pred_mean - pred_ctrl_mean
        mse_list.append(mean_squared_error(delta_true, delta_pred))
        pcc_list.append(pearson_corr_safe(delta_true, delta_pred))

    if len(mse_list) == 0:
        return float("nan"), 0.0, 0
    return float(np.nanmean(mse_list)), float(np.nanmean(pcc_list)), len(perts)

# ==========================================
# MAIN
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {CKPT_PATH}")
    model, config = load_model_frommmf(CKPT_PATH, key="cell")
    model = model.to(device).eval()

    adata, norm_data, row_sums = load_sciplex(GENE_INDEX_PATH)

    batches = []
    for i in range(0, len(norm_data), BATCH_SIZE):
        expr = norm_data[i:i+BATCH_SIZE]
        sums = row_sums[i:i+BATCH_SIZE]
        meta = np.hstack([
            np.full((len(expr), 1), 4.0),
            np.log10(sums + 1e-8)
        ])
        full = np.hstack([expr, meta])
        batches.append(torch.tensor(full).float())

    activations, register_hooks, remove_hooks = get_activation_capturer(model)
    n_layers = register_hooks()

    pad_token_id = config["pad_token_id"]
    pool_style = getattr(model, "pool_style", "all")
    layer_features = {f"layer_{i}": [] for i in range(n_layers + 1)}

    print("Extracting activations...")
    with torch.no_grad():
        for raw_x in tqdm(batches):
            raw_x = raw_x.to(device)
            mask = raw_x > 0
            x, x_pad = gatherData(raw_x, mask, pad_token_id)
            x_emb = model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)

            gene_ids = torch.arange(raw_x.shape[1], device=device).unsqueeze(0)
            gene_ids = gene_ids.repeat(raw_x.shape[0], 1)
            pos_ids, _ = gatherData(gene_ids, mask, pad_token_id)
            p_emb = model.pos_emb(pos_ids)
            x_input = x_emb + p_emb
            activations["layer_0"] = x_input

            _ = model.encoder(x_input, padding_mask=x_pad)

            for k, v in activations.items():
                if k == f"layer_{n_layers}" and getattr(model.encoder, "norm", None) is not None:
                    v = model.encoder.norm(v)
                pooled = pool_hidden(v, x_pad, pool_style=pool_style)
                layer_features[k].append(pooled.detach().cpu().numpy())

            activations.clear()

    remove_hooks()

    print("Saving per-layer tensors...")
    for k in layer_features:
        layer_features[k] = np.concatenate(layer_features[k], axis=0)
        save_path = os.path.join(OUTPUT_DIR, f"scfoundation_{k}.pt")
        torch.save(torch.tensor(layer_features[k]), save_path)
        print(f"Saved {k} → {save_path}")

    expr = norm_data.astype(np.float32)
    pert_col = pick_perturbation_column(adata)
    pert_labels = adata.obs[pert_col].astype(str).values
    control_label = pick_control_label(np.unique(pert_labels))

    valid_mask = ~pd.isna(pert_labels)
    pert_labels = pert_labels[valid_mask]
    expr = expr[valid_mask]

    pert_names_all = sorted(set(pert_labels) - {control_label})
    pert_names_set = {p for p in pert_names_all if int((pert_labels == p).sum()) >= MIN_PERT_CELLS}
    pert_names = sorted(pert_names_set)

    keep_mask = np.isin(pert_labels, pert_names) | (pert_labels == control_label)
    pert_labels = pert_labels[keep_mask]
    expr = expr[keep_mask]

    folds = make_size_folds(pert_names, pert_labels, k=K_FOLDS, seed=SEED)
    layer_files = sorted(Path(OUTPUT_DIR).glob("scfoundation_layer_*.pt"),
                         key=lambda p: int(p.stem.split("_")[-1]))

    results = []
    for layer_path in layer_files:
        layer_idx = int(layer_path.stem.split("_")[-1])
        emb = torch.load(layer_path).numpy().astype(np.float32)
        emb = emb[valid_mask][keep_mask]

        mse_fold_list, pcc_fold_list = [], []
        for group in folds:
            group = [p for p in group if p in pert_names_set]
            if len(group) == 0:
                continue

            test_mask = np.isin(pert_labels, group)
            train_mask = ~test_mask

            idx_train = np.where(train_mask)[0]
            idx_test = np.where(test_mask)[0]
            if idx_train.size == 0 or idx_test.size == 0:
                continue

            X_train, X_test = emb[idx_train], emb[idx_test]
            y_train, y_test = expr[idx_train], expr[idx_test]
            train_labels = pert_labels[idx_train]
            test_labels = pert_labels[idx_test]

            reg = Ridge(alpha=RIDGE_L2)
            reg.fit(X_train, y_train)
            y_pred_train = reg.predict(X_train)
            y_pred_test = reg.predict(X_test)

            mse_delta, pcc_delta, _ = compute_delta_metrics_with_train_control(
                y_test, y_pred_test, test_labels,
                y_train, y_pred_train, train_labels, control_label
            )

            mse_fold_list.append(mse_delta)
            pcc_fold_list.append(pcc_delta)

        mse_delta = float(np.nanmean(mse_fold_list)) if mse_fold_list else float("nan")
        pcc_delta = float(np.nanmean(pcc_fold_list)) if pcc_fold_list else 0.0
        results.append({"layer": layer_idx, "mse_delta": mse_delta, "pcc_delta": pcc_delta})
        print(f"Layer {layer_idx}: MSE={mse_delta:.6f}, PCC={pcc_delta:.4f}")

    df = pd.DataFrame(results).sort_values("layer")
    csv_path = os.path.join(
        OUTPUT_DIR,
        "scfoundation_sciplex_5fold_sizebased_mse_pcc_per_layer.csv"
    )
    df.to_csv(csv_path, index=False)
    print(f"Saved results CSV: {csv_path}")


if __name__ == "__main__":
    main()
