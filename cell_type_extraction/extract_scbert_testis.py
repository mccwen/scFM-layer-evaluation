"""Layer-wise scbert pipeline for Tabula Sapiens testis.

This file contains one dataset pipeline only. Raw data, checkpoints, and
generated activations are supplied through local configuration.
"""

import torch

import torch.nn as nn

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

import numpy as np

import os

import scanpy as sc

from scipy import sparse

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import StratifiedKFold

from pathlib import Path

import random

from performer_pytorch.performer_pytorch import PerformerLM, Performer, FastAttention, SelfAttention

# ============================================================
# 1. Reproducibility
# ============================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# 2. Config
# ============================================================
MODEL_PATH = "ckpts_folder/panglao_pretrain.pth"
GENE2VEC_PATH = "data/gene2vec_16906.npy"
PANGLAO_PATH = "./data/panglao_10000.h5ad"
DATA_PATH = "testis_annotated.h5ad"

OUTPUT_DIR = "scbert_ts_testis_layer_probing"
CELL_TYPE_COL = "cell_type"

MIN_GENES_PER_CELL = 200
MIN_CELLS_PER_GENE = 3
MIN_CELLTYPE_SIZE = 15

BIN_NUM = 5
CLASS = BIN_NUM + 2
BATCH_SIZE = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

out = Path(OUTPUT_DIR)
out.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Helpers
# ============================================================
def get_raw_counts(adata):
    if adata.raw is not None:
        return adata.raw.X.copy(), adata.raw.var.copy()
    if "counts" in adata.layers:
        return adata.layers["counts"].copy(), adata.var.copy()
    return adata.X.copy(), adata.var.copy()


def masked_mean(h, x):
    mask = (x > 0).float().unsqueeze(-1)
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1)


def extract_embeddings(model, layers, x_disc, batch_size):
    # Faithful scBERT export:
    # layer 0=input embedding, layers 1-6=transformer blocks, layer 7=post-norm.
    layer_storage = {i: [] for i in range(len(layers) + 2)}

    with torch.no_grad():
        for i in range(0, x_disc.shape[0], batch_size):
            x = torch.as_tensor(x_disc[i : i + batch_size], dtype=torch.long, device=device)

            h = model.token_emb(x).float() + model.pos_emb(x).float()
            h = model.dropout(h)
            layer_storage[0].append(masked_mean(h, x).cpu())

            for li, layer in enumerate(layers, start=1):
                attn, ff = layer
                h = h + attn(h, pos_emb=model.layer_pos_emb(h))
                h = h + ff(h)
                layer_storage[li].append(masked_mean(h, x).cpu())

            h = model.norm(h)
            layer_storage[len(layers) + 1].append(masked_mean(h, x).cpu())

    return {k: torch.cat(v, 0) for k, v in layer_storage.items()}


# ============================================================
# 4. Load dataset and save original indexing
# ============================================================
adata = sc.read_h5ad(DATA_PATH)
adata.obs_names_make_unique()
original_obs_names = adata.obs_names.copy()

raw_x, raw_var = get_raw_counts(adata)
adata_counts = sc.AnnData(
    X=raw_x,
    var=raw_var.copy(),
    obs=adata.obs.copy(),
)
adata_counts.obs_names_make_unique()

if CELL_TYPE_COL not in adata_counts.obs.columns:
    raise ValueError(
        f"CELL_TYPE_COL='{CELL_TYPE_COL}' not found. "
        f"Available columns: {list(adata_counts.obs.columns)}"
    )

sc.pp.filter_cells(adata_counts, min_genes=MIN_GENES_PER_CELL)
sc.pp.filter_genes(adata_counts, min_cells=MIN_CELLS_PER_GENE)


# ============================================================
# 5. Filter cell types (<15) and save kept indices
# ============================================================
ct_counts = adata_counts.obs[CELL_TYPE_COL].astype(str).value_counts()
keep_ct = ct_counts[ct_counts >= MIN_CELLTYPE_SIZE].index

adata_counts = adata_counts[
    adata_counts.obs[CELL_TYPE_COL].astype(str).isin(keep_ct)
].copy()

filtered_obs_names = adata_counts.obs_names.copy()
keep_idx = original_obs_names.get_indexer(filtered_obs_names)
if np.any(keep_idx < 0):
    raise ValueError("Could not map filtered obs_names back to original h5ad rows.")

np.save(out / "keep_idx.npy", keep_idx.astype(np.int64))
np.save(out / "keep_obs_names.npy", filtered_obs_names.to_numpy())

(
    adata_counts.obs[CELL_TYPE_COL]
    .astype(str)
    .value_counts()
    .rename_axis(CELL_TYPE_COL)
    .reset_index(name="count")
    .to_csv(out / "cell_type_counts_after_filter.csv", index=False)
)

print("After filtering:", adata_counts.shape)
print(f"Saved keep_idx.npy with {len(keep_idx)} retained cells.")


# ============================================================
# 6. Normalize
# ============================================================
sc.pp.normalize_total(adata_counts, target_sum=1e4)
sc.pp.log1p(adata_counts, base=2)


# ============================================================
# 7. Gene alignment
# ============================================================
panglao = sc.read_h5ad(PANGLAO_PATH)
ref_genes = panglao.var_names.astype(str).str.upper().tolist()

adata_counts.var_names = adata_counts.var_names.astype(str).str.upper()

gene_num = len(ref_genes)
seq_len = gene_num + 1

counts = sparse.lil_matrix((adata_counts.shape[0], gene_num), dtype=np.float32)
lookup = {g: i for i, g in enumerate(adata_counts.var_names)}

for i, gene in enumerate(ref_genes):
    if gene in lookup:
        counts[:, i] = adata_counts.X[:, lookup[gene]]

aligned = sc.AnnData(X=counts.tocsr(), var=panglao.var.copy(), obs=adata_counts.obs.copy())


# ============================================================
# 8. Discretization
# ============================================================
x_disc = np.zeros((aligned.shape[0], gene_num), dtype=np.int64)

for i in range(0, aligned.shape[0], 512):
    batch = aligned.X[i : i + 512].toarray()
    batch = np.clip(batch, 0, CLASS - 2)
    x_disc[i : i + 512] = batch.astype(np.int64)

x_disc = np.concatenate(
    [x_disc, np.zeros((x_disc.shape[0], 1), dtype=np.int64)],
    axis=1,
)


# ============================================================
# 9. Load gene2vec + model
# ============================================================
gene2vec = np.load(GENE2VEC_PATH).astype(np.float32)
if gene2vec.shape[0] == gene_num:
    gene2vec = np.vstack([gene2vec, np.zeros((1, gene2vec.shape[1]), dtype=np.float32)])

gene2vec_t = torch.tensor(gene2vec, dtype=torch.float32, device=device)

model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    heads=10,
    max_seq_len=seq_len,
    local_attn_heads=0,
    g2v_position_emb=True,
).to(device)

ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model = model.float()
model.eval()
model.pos_emb.emb.weight.data = gene2vec_t.float()

layers = model.performer.net.layers
if len(layers) != 6:
    raise ValueError(f"Expected a 6-layer scBERT stack, found {len(layers)} layers.")


# ============================================================
# 10. Extract and save activations (layers 0-6)
# ============================================================
print("Extracting embeddings...")
layer_embeddings = extract_embeddings(model, layers, x_disc, BATCH_SIZE)

for layer_id, tensor in layer_embeddings.items():
    torch.save(tensor, out / f"layer_{layer_id}.pt")

print(f"Saved {len(layer_embeddings)} activation tensors (layers 0-{len(layer_embeddings)-1}).")


# ============================================================
# 11. Labels
# ============================================================
labels = aligned.obs[CELL_TYPE_COL].astype(str).tolist()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
np.save(out / "label_classes.npy", label_encoder.classes_)


# ============================================================
# 12. 5-fold CV probing
# ============================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results = []

for layer_id in sorted(layer_embeddings):
    x = layer_embeddings[layer_id].numpy()

    accs = []
    f1s = []

    for train_idx, test_idx in skf.split(x, y):
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=SEED)
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])

        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))

    results.append(
        {
            "layer": layer_id,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s, ddof=1)),
            "n_folds_used": 5,
            "n_cells": int(x.shape[0]),
            "n_classes": int(len(np.unique(y))),
            "min_celltype_size": MIN_CELLTYPE_SIZE,
        }
    )

    print(
        f"Layer {layer_id}: "
        f"acc={np.mean(accs):.4f}±{np.std(accs, ddof=1):.4f}, "
        f"f1={np.mean(f1s):.4f}±{np.std(f1s, ddof=1):.4f}"
    )

df = pd.DataFrame(results).sort_values("layer")
csv_path = out / "scBERT_ts_testis_layer_metrics_5fold_mean.csv"
df.to_csv(csv_path, index=False)
