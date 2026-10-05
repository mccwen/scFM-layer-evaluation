"""Layer-wise scbert pipeline for Schmidt.

This file contains one dataset pipeline only. Raw data, checkpoints, and
generated activations are supplied through local configuration.
"""

import os

import re

import random

import time

from pathlib import Path

import numpy as np

import pandas as pd

import scanpy as sc

from scipy import sparse

import torch

import matplotlib.pyplot as plt

import torch.nn as nn

from sklearn.metrics import mean_squared_error

from performer_pytorch.performer_pytorch import PerformerLM

# ============================================================
# 1. Reproducibility (single seed)
# ============================================================
SEED = 42

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ============================================================
# 2. Configuration
# ============================================================
model_path    = "ckpts_folder/panglao_pretrain.pth"
gene2vec_path = "data/gene2vec_16906.npy"
output_dir    = "scbert_sciplex_probing"

DATA_PATH     = "Schmidt.h5ad"
PANG_LAO_PATH = "./data/panglao_10000.h5ad"
LAYER_KEY = "logNor"

BIN_NUM   = 5
CLASS     = BIN_NUM + 2
EMB_BATCH_SIZE = 8

# CV config (fixed 5-fold, single seed)
K_FOLDS = 5
PROBE_EPOCHS = 50
PROBE_BATCH_SIZE = 256
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 0.0

# Perturbation metadata
PERTURBATION_KEY = "condition"
CONTROL_LABEL = "control"
CONTEXT_KEY = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 3. Load data using adata.layers["logNor"]
# ============================================================
adata = sc.read_h5ad(DATA_PATH)

# Fix gene identifiers
if "symbol-0" in adata.var.columns:
    adata.var_names = adata.var["symbol-0"].astype(str)
adata.var_names_make_unique()

# Use log-normalized expression layer instead of adata.X
if LAYER_KEY not in adata.layers:
    raise ValueError(
        f"Expected adata.layers['{LAYER_KEY}'], but it was not found. "
        f"Available layers: {list(adata.layers.keys())}"
    )

X_log = adata.layers[LAYER_KEY]

if sparse.issparse(X_log):
    X_log = X_log.tocsr()
else:
    X_log = sparse.csr_matrix(X_log)

adata_log = sc.AnnData(
    X=X_log,
    obs=adata.obs.copy(),
    var=adata.var.copy()
)

print(f"Using adata.layers['{LAYER_KEY}']")
print("Input shape:", adata_log.shape)

# ============================================================
# 4. Align genes to Panglao (zero-fill missing)
# ============================================================
panglao = sc.read_h5ad(PANG_LAO_PATH)

adata_log.var_names_make_unique()
panglao.var_names_make_unique()

ref_genes = panglao.var_names.astype(str).str.upper()
ref_index = pd.Index(ref_genes)
adata_index = pd.Index(adata_log.var_names.astype(str).str.upper())

common = ref_index.intersection(adata_index)
print("Common genes:", len(common))

src_idx = adata_index.get_indexer(common)
tgt_idx = ref_index.get_indexer(common)

X_src = adata_log.X.tocsr()
X_common = X_src[:, src_idx].tocoo()

n_cells = adata_log.n_obs
n_ref   = len(ref_genes)

X_aligned = sparse.csr_matrix(
    (X_common.data, (X_common.row, tgt_idx[X_common.col])),
    shape=(n_cells, n_ref)
)

adata_aligned = sc.AnnData(
    X=X_aligned,
    var=panglao.var.copy(),
    obs=adata_log.obs.copy()
)

print("Aligned shape:", adata_aligned.shape)

GENE_NUM = n_ref
SEQ_LEN  = GENE_NUM + 1

# ============================================================
# 5. Load gene2vec (positional embedding)
# ============================================================
gene2vec = np.load(gene2vec_path)
if gene2vec.shape[0] == GENE_NUM:
    gene2vec = np.vstack([gene2vec, np.zeros((1, gene2vec.shape[1]))])
elif gene2vec.shape[0] != GENE_NUM + 1:
    raise ValueError(f"gene2vec rows ({gene2vec.shape[0]}) != aligned genes ({GENE_NUM+1})")

gene2vec_t = torch.tensor(gene2vec, dtype=torch.float32, device=device)

# ============================================================
# 6. Load scBERT
# ============================================================
model = PerformerLM(
    num_tokens  = CLASS,
    dim         = 200,
    depth       = 6,
    heads       = 10,
    max_seq_len = SEQ_LEN
).to(device)

ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval().float()

# overwrite pos_emb with gene2vec
if hasattr(model, "pos_emb") and hasattr(model.pos_emb, "emb"):
    with torch.no_grad():
        model.pos_emb.emb.weight.copy_(gene2vec_t)

layers = model.performer.net.layers
n_layers = len(layers)

# ============================================================
# 7. Extraction (MASKED MEAN POOLING)
# ============================================================
@torch.inference_mode()
def extract_embeddings(model, X_sparse, batch_size):
    n_cells = X_sparse.shape[0]
    dim = 200

    layer_storage = {
        i: torch.empty((n_cells, dim), dtype=torch.float32)
        for i in range(n_layers + 2)
    }

    for i in range(0, n_cells, batch_size):
        batch = X_sparse[i:i+batch_size]
        if sparse.issparse(batch):
            batch = batch.toarray()

        # scBERT discretization: clip -> int
        batch = np.clip(batch, 0, CLASS - 2)
        x = torch.tensor(batch, dtype=torch.int64, device=device)

        # append trailing 0 token
        zeros = torch.zeros((x.shape[0], 1), dtype=torch.int64, device=device)
        x = torch.cat([x, zeros], dim=1)

        # mask for nonzero genes
        mask = (x > 0).float()

        tok = model.token_emb(x)
        h = tok + model.pos_emb(x)
        if hasattr(model, "dropout"):
            h = model.dropout(h)

        def masked_mean(h, mask):
            masked_h = h * mask.unsqueeze(-1)
            sum_h = masked_h.sum(dim=1)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return sum_h / denom

        # layer 0
        layer_storage[0][i:i+x.shape[0]] = masked_mean(h, mask).cpu()

        # transformer layers (residual)
        for li, layer in enumerate(layers):
            attn, ff = layer
            h = h + attn(h)
            h = h + ff(h)
            layer_storage[li+1][i:i+x.shape[0]] = masked_mean(h, mask).cpu()

        # post norm
        h = model.norm(h)
        layer_storage[n_layers+1][i:i+x.shape[0]] = masked_mean(h, mask).cpu()

    return layer_storage

print("Extracting...")
layer_embeddings = extract_embeddings(
    model,
    adata_aligned.X,
    EMB_BATCH_SIZE
)

# ============================================================
# 8. Save outputs (embeddings)
# ============================================================
out = Path(output_dir)
out.mkdir(exist_ok=True)

for k, v in layer_embeddings.items():
    label = "input" if k == 0 else (f"block_{k}" if k <= n_layers else "postnorm")
    path = out / f"layer_{k}_{label}.pt"
    torch.save(v, path)
    print("Saved:", path)

# ============================================================
# 9. 5-fold CV probe (single seed, size-based split)
# ============================================================
CONTROL_LABEL_CANDIDATES = [
    "control", "ctrl", "vehicle", "dmso", "untreated", "wildtype", "wt", "none"
]

PERTURBATION_KEY_CANDIDATES = [
    "perturbation", "perturbation_name", "pert", "condition", "treatment"
]

def _infer_key(obs, key, candidates):
    if key is not None and key in obs.columns:
        return key
    for c in candidates:
        if c in obs.columns:
            return c
    raise ValueError(f"Could not infer key. Set explicitly from: {candidates}")

def _guess_control_label(values, preferred=None):
    if preferred and preferred in values:
        return preferred
    lower_map = {v.lower(): v for v in values}
    for c in CONTROL_LABEL_CANDIDATES:
        if c in lower_map:
            return lower_map[c]
    raise ValueError("No control label found. Set CONTROL_LABEL explicitly.")

def _compute_delta(X, adata, context_key, perturb_key, control_label):
    context = adata.obs[context_key].astype(str).values
    perturb = adata.obs[perturb_key].astype(str).values
    delta = np.zeros_like(X, dtype=np.float32)
    for ctx in np.unique(context):
        ctx_mask = context == ctx
        ctrl_mask = (perturb == control_label) & ctx_mask
        if ctrl_mask.sum() == 0:
            continue
        ctrl_mean = X[ctrl_mask].mean(axis=0)
        delta[ctx_mask] = X[ctx_mask] - ctrl_mean
    return delta

def _build_kfold_splits_size_based(perturb, control_label, k_folds, seed):
    perts = sorted(set(perturb) - {control_label})
    if len(perts) < k_folds:
        raise ValueError(f"Need at least {k_folds} perturbations for {k_folds}-fold CV.")

    counts = {p: int((perturb == p).sum()) for p in perts}
    perts_sorted = sorted(perts, key=lambda p: counts[p], reverse=True)

    folds = [[] for _ in range(k_folds)]
    fold_sizes = [0] * k_folds

    for p in perts_sorted:
        idx = int(np.argmin(fold_sizes))
        folds[idx].append(p)
        fold_sizes[idx] += counts[p]

    splits = []
    for group in folds:
        test_mask = np.isin(perturb, group)
        train_mask = ~test_mask
        splits.append((group, train_mask, test_mask))
    return splits

def _pcc_delta(delta_true, delta_pred):
    a = delta_true.reshape(-1)
    b = delta_pred.reshape(-1)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return 0.0 if not np.isfinite(corr) else corr

def _mse_delta(delta_true, delta_pred):
    return float(np.mean((delta_pred - delta_true) ** 2))

def train_linear_probe(x_train, y_train, epochs, batch_size, lr, weight_decay, device):
    x_train = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)

    in_dim = x_train.shape[1]
    out_dim = y_train.shape[1]
    model = torch.nn.Linear(in_dim, out_dim, bias=True).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    n = x_train.shape[0]
    idx = torch.arange(n, device=device)

    for _ in range(epochs):
        perm = idx[torch.randperm(n)]
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start+batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

@torch.inference_mode()
def predict(model, x, device):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    return model(x).cpu().numpy()

print("Running 5-fold CV (single seed, size-based split)...")

pert_key = _infer_key(adata_aligned.obs, PERTURBATION_KEY, PERTURBATION_KEY_CANDIDATES)
if CONTEXT_KEY is None:
    adata_aligned.obs["_context"] = "all"
    context_key = "_context"
else:
    context_key = CONTEXT_KEY

perturb = adata_aligned.obs[pert_key].astype(str).values
control_label = _guess_control_label(perturb, preferred=CONTROL_LABEL)

# expression matrix for targets
if sparse.issparse(adata_aligned.X):
    X_expr = adata_aligned.X.toarray().astype(np.float32)
else:
    X_expr = np.asarray(adata_aligned.X, dtype=np.float32)

y_all = _compute_delta(
    X=X_expr,
    adata=adata_aligned,
    context_key=context_key,
    perturb_key=pert_key,
    control_label=control_label,
)

splits = _build_kfold_splits_size_based(
    perturb=perturb,
    control_label=control_label,
    k_folds=K_FOLDS,
    seed=SEED,
)

results = []

for k in range(n_layers + 2):
    emb = layer_embeddings[k].numpy().astype(np.float32)

    mse_list = []
    pcc_list = []
    used_perts = 0

    for group, train_mask, test_mask in splits:
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        x_train = emb[train_mask]
        y_train = y_all[train_mask]

        model = train_linear_probe(
            x_train=x_train,
            y_train=y_train,
            epochs=PROBE_EPOCHS,
            batch_size=PROBE_BATCH_SIZE,
            lr=PROBE_LR,
            weight_decay=PROBE_WEIGHT_DECAY,
            device=device,
        )

        for pert_name in group:
            pert_mask = perturb == pert_name
            if pert_mask.sum() == 0:
                continue
            x_test = emb[pert_mask]
            y_test = y_all[pert_mask]

            y_pred = predict(model, x_test, device=device)
            delta_true = y_test.mean(axis=0)
            delta_pred = y_pred.mean(axis=0)

            mse_list.append(_mse_delta(delta_true, delta_pred))
            pcc_list.append(_pcc_delta(delta_true, delta_pred))
            used_perts += 1

    pcc_list = [p for p in pcc_list if np.isfinite(p)]
    mse_delta = float(np.mean(mse_list)) if mse_list else float("nan")
    pcc_delta = float(np.mean(pcc_list)) if pcc_list else 0.0

    results.append(
        {
            "layer": k,
            "mse_delta": mse_delta,
            "pcc_delta": pcc_delta,
            "n_perts": used_perts,
        }
    )
    print(f"Layer {k} | MSE-delta={mse_delta:.6f} | PCC-delta={pcc_delta:.4f} | perts={used_perts}")

df = pd.DataFrame(results).sort_values("layer")
cv_csv = out / "layerwise_mse_pcc_delta_5fold_size_based.csv"
df.to_csv(cv_csv, index=False)
print("Saved CV metrics:", cv_csv)

print("Done.")
