"""Layer-wise scbert pipeline for PBMC3K.

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
# 2. Config
# ============================================================
model_path    = "ckpts_folder/panglao_pretrain.pth"
gene2vec_path = "data/gene2vec_16906.npy"
panglao_path  = "./data/panglao_10000.h5ad"

output_dir    = "scbert_scanpy_dataset_3kpmbc_layer_probing"
CELL_TYPE_COL = "cell_type"

bin_num   = 5
CLASS     = bin_num + 2
BATCH_SIZE = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 3. Load PBMC3K (labels from processed, counts from raw)
# ============================================================
adata_lab = sc.datasets.pbmc3k_processed()
adata_lab.obs.rename(columns={"louvain": "cell_type"}, inplace=True)

adata_counts = sc.datasets.pbmc3k()

# align by intersection
common = adata_counts.obs_names.intersection(adata_lab.obs_names)
adata_counts = adata_counts[common].copy()
adata_lab = adata_lab[common].copy()

# filter rare cell types (>=5)
cell_type_counts = adata_lab.obs[CELL_TYPE_COL].value_counts()
keep = cell_type_counts[cell_type_counts >= 5].index
mask = adata_lab.obs[CELL_TYPE_COL].isin(keep).to_numpy()

counts = adata_counts.X[mask]
var = adata_counts.var.copy()
obs = adata_lab.obs[mask].copy()

adata_counts = sc.AnnData(X=counts, var=var, obs=obs)

# QC
sc.pp.filter_cells(adata_counts, min_genes=200)
sc.pp.filter_genes(adata_counts, min_cells=3)
print("After QC:", adata_counts.shape)

# normalize + log1p base=2
sc.pp.normalize_total(adata_counts, target_sum=1e4)
sc.pp.log1p(adata_counts, base=2)

# ============================================================
# 4. Align genes to Panglao reference
# ============================================================
panglao = sc.read_h5ad(panglao_path)

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
SEQ_LEN = gene_num + 1

counts = sparse.lil_matrix((adata_counts.shape[0], gene_num), dtype=np.float32)
lookup = {g: i for i, g in enumerate(adata_counts.var_names)}

matched = 0
for i, g in enumerate(ref_genes):
    if g in lookup:
        counts[:, i] = adata_counts.X[:, lookup[g]]
        matched += 1

print(f"Matched genes: {matched} / {gene_num}")

new = sc.AnnData(
    X=counts.tocsr(),
    var=panglao.var,
    obs=adata_counts.obs
)
print("Aligned dataset:", new.shape)

# ============================================================
# 5. Discretize (faithful scBERT)
# ============================================================
print("Discretizing expression values...")

X_disc = np.zeros((new.shape[0], gene_num), dtype=np.int64)
for i in range(0, new.shape[0], 512):
    batch = new.X[i:i+512].toarray()
    batch = np.clip(batch, 0, CLASS - 2)
    X_disc[i:i+512] = batch.astype(np.int64)

# append trailing 0 token
X_disc = np.concatenate(
    [X_disc, np.zeros((X_disc.shape[0], 1), dtype=np.int64)],
    axis=1
)
print("Discretized shape:", X_disc.shape)

# ============================================================
# 6. Load gene2vec (positional embedding)
# ============================================================
gene2vec = np.load(gene2vec_path)

if gene2vec.shape[1] != 200:
    raise ValueError("gene2vec dim mismatch. Expected 200.")

if gene2vec.shape[0] == gene_num:
    gene2vec = np.vstack([gene2vec, np.zeros((1, gene2vec.shape[1]))])
elif gene2vec.shape[0] != gene_num + 1:
    raise ValueError("gene2vec length mismatch with gene_num+1.")

gene2vec_t = torch.tensor(gene2vec, dtype=torch.float32).to(device)

# ============================================================
# 7. Load scBERT model
# ============================================================
model = PerformerLM(
    num_tokens  = CLASS,
    dim         = 200,
    depth       = 6,
    heads       = 10,
    max_seq_len = SEQ_LEN,
    local_attn_heads=0,
    g2v_position_emb=True
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
print("Transformer blocks:", n_layers)

# ============================================================
# 8. Masked mean pooling (ignore zeros)
# ============================================================
def masked_mean(h, x):
    mask = (x > 0).float().unsqueeze(-1)
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

# ============================================================
# 9. Extract per-layer embeddings
# ============================================================
def extract_cell_embeddings(model, X_disc, batch_size):
    model.eval()
    layer_storage = {i: [] for i in range(n_layers + 2)}

    with torch.no_grad():
        for i in range(0, X_disc.shape[0], batch_size):
            x = torch.tensor(X_disc[i:i+batch_size]).long().to(device)

            h = model.token_emb(x) + model.pos_emb(x)
            if hasattr(model, "dropout"):
                h = model.dropout(h)

            # layer 0 (input)
            layer_storage[0].append(masked_mean(h, x).cpu())

            for li, layer in enumerate(layers):
                attn, ff = layer
                h = h + attn(h, pos_emb=model.layer_pos_emb(h))
                h = h + ff(h)
                layer_storage[li + 1].append(masked_mean(h, x).cpu())

            h = model.norm(h)
            layer_storage[n_layers + 1].append(masked_mean(h, x).cpu())

    return {k: torch.cat(v, dim=0) for k, v in layer_storage.items()}

print("\nExtracting embeddings...")
layer_embeddings = extract_cell_embeddings(model, X_disc, BATCH_SIZE)

# ============================================================
# 10. Save embeddings
# ============================================================
out = Path(output_dir)
out.mkdir(exist_ok=True)

for k, v in layer_embeddings.items():
    label = "input" if k == 0 else (f"block_{k}" if k <= n_layers else "postnorm")
    torch.save(v, out / f"layer_{k}_{label}.pt")

labels = new.obs[CELL_TYPE_COL].tolist()
torch.save(labels, out / "cell_types.pt")

# ============================================================
# 11. 5-fold linear probing
# ============================================================
print("\n5-fold linear probing...")
le = LabelEncoder()
y = le.fit_transform(labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = []
for li in sorted(layer_embeddings.keys()):
    X = layer_embeddings[li].numpy()

    accs, f1s = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])

        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))

    results.append({
        "layer": li,
        "accuracy_mean": np.mean(accs),
        "macro_f1_mean": np.mean(f1s),
    })
    print(f"Layer {li}: acc={results[-1]['accuracy_mean']:.4f}, f1={results[-1]['macro_f1_mean']:.4f}")

df = pd.DataFrame(results)
df.to_csv(out / "scbert_pbmc3k_layer_probe_results.csv", index=False)
