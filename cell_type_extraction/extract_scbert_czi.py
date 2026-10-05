"""Layer-wise scbert pipeline for CZI immune.

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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 2. Configuration
# ============================================================
model_path    = "ckpts_folder/panglao_pretrain.pth"
gene2vec_path = "data/gene2vec_16906.npy"
output_dir    = "scbert_CZI_immune_layer_probing_FIXED_v2"

CELL_TYPE_COL = "cell_type"

bin_num    = 5                  # scBERT uses bin_num + 2 tokens
BATCH_SIZE = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 3. Load dataset (RAW COUNTS ONLY)
# ============================================================
adata = sc.read_h5ad(
    "CZI_human_embryonic_meninges_at5-13_weeks_post_conception_immune_cells.h5ad"
)
print("Original dataset shape:", adata.shape)

# Use raw counts if available, else fallback to X
if adata.raw is not None:
    print("Using adata.raw.X")
    X_source = adata.raw.X
    var_source = adata.raw.var
else:
    print("WARNING: adata.raw is None → assuming adata.X is raw counts")
    X_source = adata.X
    var_source = adata.var

adata_counts = sc.AnnData(
    X=X_source.copy(),
    var=var_source.copy(),
    obs=adata.obs.copy()
)

# ============================================================
# 4. Gene alignment (panglao reference)
# ============================================================
panglao = sc.read_h5ad("./data/panglao_10000.h5ad")

if "gene_name" in adata_counts.var.columns:
    gene_symbols = adata_counts.var["gene_name"]
elif "feature_name" in adata_counts.var.columns:
    gene_symbols = adata_counts.var["feature_name"]
else:
    gene_symbols = adata_counts.var_names

adata_counts.var["gene_symbol"] = gene_symbols.astype(str).str.upper()
adata_counts.var_names = adata_counts.var["gene_symbol"]
adata_counts.var_names_make_unique()

ref_genes = panglao.var_names.astype(str).str.upper().tolist()
gene_num = len(ref_genes)

counts = sparse.lil_matrix((adata_counts.shape[0], len(ref_genes)), dtype=np.float32)
lookup = {g: i for i, g in enumerate(adata_counts.var_names)}

matched = 0
for i, g in enumerate(ref_genes):
    if g in lookup:
        counts[:, i] = adata_counts.X[:, lookup[g]]
        matched += 1

print(f"Matched genes: {matched} / {len(ref_genes)}")

new = sc.AnnData(
    X   = counts.tocsr(),
    var = panglao.var,
    obs = adata_counts.obs
)
print("Aligned dataset:", new.shape)

# ============================================================
# 5. scBERT preprocess: filter + normalize + log1p(base=2)
# ============================================================
sc.pp.filter_cells(new, min_genes=200)
sc.pp.normalize_total(new, target_sum=1e4)
sc.pp.log1p(new, base=2)

# Filter rare cell types
cell_type_counts = new.obs[CELL_TYPE_COL].value_counts()
keep = cell_type_counts[cell_type_counts >= 5].index
new = new[new.obs[CELL_TYPE_COL].isin(keep)].copy()
print("After filtering rare cell types:", new.shape)

# ============================================================
# 6. Discretize to token ids (scBERT style)
# ============================================================
CLASS = bin_num + 2  # token vocab size in scBERT
CLIP_MAX = CLASS - 2  # values capped at CLASS-2

def discretize_scbert(X_sparse, clip_max, batch_size=512):
    X_bin = []
    for i in range(0, X_sparse.shape[0], batch_size):
        batch = X_sparse[i:i+batch_size].A
        batch = batch.astype(np.int64)  # floor by casting (as in scBERT)
        batch[batch > clip_max] = clip_max
        X_bin.append(batch)
    return np.vstack(X_bin)

X_binned = discretize_scbert(new.X, CLIP_MAX, batch_size=512)
print("Unique bins:", np.unique(X_binned))

# append trailing 0 token (SEQ_LEN = gene_num + 1)
X_binned = np.hstack([X_binned, np.zeros((X_binned.shape[0], 1), dtype=np.int64)])
SEQ_LEN = gene_num + 1

# ============================================================
# 7. Load model (scBERT PerformerLM)
# ============================================================
model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    heads=10,
    max_seq_len=SEQ_LEN,
    local_attn_heads=0,
    g2v_position_emb=True
).to(device)

ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval().float()

# load gene2vec weights explicitly for safety
gene2vec = np.load(gene2vec_path)
gene2vec = np.concatenate((gene2vec, np.zeros((1, gene2vec.shape[1]))), axis=0)
model.pos_emb.emb.weight.data = torch.tensor(gene2vec, dtype=torch.float32, device=device)

layers = model.performer.net.layers
n_layers = len(layers)

# ============================================================
# 8. Masked pooling (ignore token id 0)
# ============================================================
def masked_mean(h, x):
    mask = (x > 0).float().unsqueeze(-1)
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

# ============================================================
# 9. Extract embeddings (faithful PerformerLM forward)
# ============================================================
def extract_cell_embeddings(model, X_binned):
    layer_storage = {i: [] for i in range(n_layers + 2)}

    with torch.no_grad():
        for i in range(0, X_binned.shape[0], BATCH_SIZE):
            x = torch.tensor(X_binned[i:i+BATCH_SIZE]).long().to(device)

            h = model.token_emb(x)
            h = h + model.pos_emb(x)
            h = model.dropout(h)

            layer_storage[0].append(masked_mean(h, x).cpu())

            for li, layer in enumerate(layers):
                attn, ff = layer
                h = h + attn(h, pos_emb=model.layer_pos_emb(h))
                h = h + ff(h)
                layer_storage[li + 1].append(masked_mean(h, x).cpu())

            h = model.norm(h)
            layer_storage[n_layers + 1].append(masked_mean(h, x).cpu())

    return {k: torch.cat(v, dim=0) for k, v in layer_storage.items()}

layer_embeddings = extract_cell_embeddings(model, X_binned)

# ============================================================
# 10. Save embeddings
# ============================================================
out = Path(output_dir)
out.mkdir(exist_ok=True)

for k, v in layer_embeddings.items():
    torch.save(v, out / f"layer_{k}.pt")

labels = new.obs[CELL_TYPE_COL].tolist()
torch.save(labels, out / "cell_types.pt")

# ============================================================
# 11. Linear probing (5-fold CV)
# ============================================================
le = LabelEncoder()
y = le.fit_transform(labels)

results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for li in sorted(layer_embeddings.keys()):
    X = layer_embeddings[li].numpy()

    acc_list, f1_list = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        acc_list.append(accuracy_score(y[test_idx], y_pred))
        f1_list.append(f1_score(y[test_idx], y_pred, average="macro"))

    results.append({
        "layer": li,
        "accuracy": np.mean(acc_list),
        "macro_f1": np.mean(f1_list)
    })

df = pd.DataFrame(results)
df.to_csv(out / "scbert_CZI_layer_probe_results.csv", index=False)
