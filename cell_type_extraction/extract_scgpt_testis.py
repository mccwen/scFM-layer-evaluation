"""Layer-wise scgpt pipeline for Tabula Sapiens testis.

This file contains one dataset pipeline only. Raw data, checkpoints, and
generated activations are supplied through local configuration.
"""

import anndata as ad

import scanpy as sc

import torch

import json

import numpy as np

import inspect

from torch.utils.data import DataLoader, SequentialSampler

from tqdm import tqdm

import os

import random

import pandas as pd

from pathlib import Path

import glob

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import StratifiedKFold

from scgpt.model import TransformerModel

from scgpt.utils import load_pretrained

from scgpt.tokenizer.gene_tokenizer import GeneVocab

from scgpt.data_collator import DataCollator

from scgpt.tokenizer import get_default_gene_vocab

import time

from sklearn.preprocessing import LabelEncoder, StandardScaler

SCGPT_MODEL_DIR = os.environ.get("SCGPT_MODEL_DIR")
if not SCGPT_MODEL_DIR:
    raise RuntimeError(
        "Set SCGPT_MODEL_DIR to the local scGPT checkpoint directory "
        "before running this extractor."
    )

import scanpy as sc
import anndata as ad
import torch
import json
import numpy as np
import inspect
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import os
import random
import pandas as pd
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.data_collator import DataCollator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
start_time = time.time()

# ============================================================
# 0. Seed
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
# 1. Load data
# ============================================================
model_dir = SCGPT_MODEL_DIR

with open(f"{model_dir}/args.json") as f:
    configs = json.load(f)

adata = sc.read_h5ad(
    "ts_testis_annotated.h5ad"
)
adata.obs_names_make_unique()
original_obs_names = adata.obs_names.copy()

print("Original cells:", adata.n_obs)

# ============================================================
# 2. FILTER CELL TYPES (< 15 threshold) immediately after read
# ============================================================
CELLTYPE_COL = "cell_type"

ct_counts = adata.obs[CELLTYPE_COL].astype(str).value_counts()
keep_ct = ct_counts[ct_counts >= 15].index
adata = adata[adata.obs[CELLTYPE_COL].astype(str).isin(keep_ct)].copy()
adata.obs_names_make_unique()

print("After cell-type filter (>=15):", adata.n_obs)

# ============================================================
# 3. RAW counts
# ============================================================
if adata.raw is not None:
    counts = adata.raw.X
    var = adata.raw.var.copy()
elif "counts" in adata.layers:
    counts = adata.layers["counts"]
    var = adata.var.copy()
else:
    counts = adata.X
    var = adata.var.copy()

adata_proc = ad.AnnData(
    X=counts,
    obs=adata.obs.copy(),
    var=var.copy()
)
adata_proc.obs_names_make_unique()

# ============================================================
# 4. QC
# ============================================================
sc.pp.filter_cells(adata_proc, min_genes=200)
sc.pp.filter_genes(adata_proc, min_cells=3)

print("After QC:", adata_proc.n_obs)

# ============================================================
# 5. Vocab
# ============================================================
vocab_path = configs.get("vocab_path", None)
if vocab_path is not None and os.path.exists(vocab_path):
    vocab = GeneVocab.from_file(vocab_path)
else:
    from scgpt.tokenizer import get_default_gene_vocab
    vocab = get_default_gene_vocab()

for tok in ["<pad>", "<cls>", "<eoc>"]:
    if tok not in vocab:
        vocab.append_token(tok)

pad_token = "<pad>"
cls_token = "<cls>"
pad_idx = vocab[pad_token]
cls_id = vocab[cls_token]
pad_value = configs["pad_value"]

vocab_stoi = vocab.get_stoi()
vocab_genes = set(str(g).upper() for g in vocab_stoi.keys())

# ============================================================
# 6. Gene matching
# ============================================================
def normalize_gene_series(values):
    return pd.Series(values).astype(str).str.upper()

def choose_best_gene_id_source(var, vocab_genes):
    candidates = []

    def add_candidate(name, values):
        if values is None:
            return
        norm = normalize_gene_series(values)
        overlap = norm.isin(vocab_genes).sum()
        candidates.append((name, norm.to_numpy(), overlap))

    for col in [
        "gene_name",
        "Gene",
        "gene_symbol",
        "gene_symbols",
        "feature_name",
        "symbol",
        "gene",
        "feature",
    ]:
        if col in var.columns:
            add_candidate(col, var[col])

    add_candidate("var_names", var.index)

    if not candidates:
        raise RuntimeError(f"No candidate gene-id columns found in var. Columns: {list(var.columns)}")

    print("Gene-vocab overlap candidates:")
    for name, _, overlap in candidates:
        print(f"  {name}: {overlap}")

    best_name, best_values, best_overlap = max(candidates, key=lambda x: x[2])

    if best_overlap == 0:
        raise RuntimeError(
            "No overlap found between dataset gene IDs and scGPT vocabulary. "
            f"Available var columns: {list(var.columns)}"
        )

    print(f"Best gene ID source: {best_name} (matched {best_overlap} genes)")
    return best_values

chosen_gene_ids = choose_best_gene_id_source(adata_proc.var, vocab_genes)
adata_proc.var_names = pd.Index(chosen_gene_ids)
adata_proc.var_names_make_unique()

gene_ids = []
for g in adata_proc.var_names:
    if g in vocab_stoi:
        gene_ids.append(vocab_stoi[g])
    else:
        gene_ids.append(-1)

adata_proc.var["id_in_vocab"] = gene_ids
adata_proc = adata_proc[:, adata_proc.var["id_in_vocab"] >= 0].copy()

gene_ids = np.array(adata_proc.var["id_in_vocab"])

print("After vocab filter:", adata_proc.n_obs, "cells,", adata_proc.n_vars, "genes")

# ============================================================
# 7. Remove empty cells
# ============================================================
nnz = np.asarray((adata_proc.X > 0).sum(axis=1)).reshape(-1)
adata_proc = adata_proc[nnz >= 1].copy()

print("After removing empty cells:", adata_proc.n_obs)

if adata_proc.n_obs == 0:
    raise RuntimeError("No cells left after filtering, vocab matching, and empty-cell removal.")

# ============================================================
# 8. SAVE keep_idx
# ============================================================
outdir = "scgpt_extract_cls_embedding_per_layer_Testis_v3"
os.makedirs(outdir, exist_ok=True)

keep_idx = original_obs_names.get_indexer(adata_proc.obs_names)
if np.any(keep_idx < 0):
    raise ValueError("Could not map filtered obs_names back to the original h5ad rows.")

np.save(os.path.join(outdir, "keep_idx.npy"), keep_idx.astype(np.int64))
np.save(os.path.join(outdir, "keep_obs_names.npy"), adata_proc.obs_names.to_numpy())

print("Saved keep_idx:", len(keep_idx))

# ============================================================
# 9. Model
# ============================================================
sig = inspect.signature(TransformerModel.__init__)
filtered_configs = {k: v for k, v in configs.items() if k in sig.parameters}
filtered_configs["vocab"] = vocab
filtered_configs["pad_token"] = pad_token
filtered_configs["ntoken"] = len(vocab)
filtered_configs.setdefault("d_model", configs["embsize"])
filtered_configs.setdefault("nhead", configs["nheads"])

model = TransformerModel(**filtered_configs)
state = torch.load(f"{model_dir}/best_model.pt", map_location="cpu")
load_pretrained(model, state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

if hasattr(model.transformer_encoder, "enable_nested_tensor"):
    model.transformer_encoder.enable_nested_tensor = False
if hasattr(model.transformer_encoder, "use_nested_tensor"):
    model.transformer_encoder.use_nested_tensor = False

# ============================================================
# 10. Dataset
# ============================================================
class ProbingDataset(torch.utils.data.Dataset):
    def __init__(self, X, gene_ids, cls_id):
        self.X = X
        self.gene_ids = gene_ids
        self.cls_id = cls_id

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        row = self.X[idx]
        if hasattr(row, "toarray"):
            row = row.toarray().ravel()

        nz = np.nonzero(row)[0]
        genes = self.gene_ids[nz]
        values = row[nz].astype(np.float32)

        genes = np.insert(genes, 0, self.cls_id)
        values = np.insert(values, 0, 0.0)

        return {
            "genes": torch.tensor(genes, dtype=torch.long),
            "expressions": torch.tensor(values, dtype=torch.float32),
        }

dataset = ProbingDataset(adata_proc.X, gene_ids, cls_id)

# ============================================================
# 11. Collator
# ============================================================
data_collator = DataCollator(
    do_padding=True,
    pad_token_id=pad_idx,
    pad_value=pad_value,
    do_mlm=False,
    do_binning=True,
    max_length=configs["max_seq_len"],
    sampling=True,
    keep_first_n_tokens=1,
)

loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=SequentialSampler(dataset),
    collate_fn=data_collator,
    num_workers=0
)

# ============================================================
# 12.Per-layer extraction without hooks
# ============================================================
def build_encoder_input(model, gene_ids, values):
    src = model.encoder(gene_ids)
    model.cur_gene_token_embs = src

    values_emb = model.value_encoder(values)
    input_emb_style = getattr(model, "input_emb_style", "continuous")

    if input_emb_style == "scaling":
        if values_emb.dim() == 2:
            total_embs = src * values_emb.unsqueeze(2)
        else:
            total_embs = src * values_emb
    else:
        if values_emb.dim() == 2:
            values_emb = values_emb.unsqueeze(2)
        total_embs = src + values_emb

    if getattr(model, "bn", None) is not None:
        total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

    return total_embs


def apply_transformer_layer(layer, hidden, src_key_padding_mask):
    sig = inspect.signature(layer.forward)
    kwargs = {}

    if "src_mask" in sig.parameters:
        kwargs["src_mask"] = None
    if "src_key_padding_mask" in sig.parameters:
        kwargs["src_key_padding_mask"] = src_key_padding_mask
    if "is_causal" in sig.parameters:
        kwargs["is_causal"] = False

    return layer(hidden, **kwargs)


def extract_cell_emb_layers(model, loader, pad_idx, device):
    model.eval()
    n_layers = len(model.transformer_encoder.layers)
    layer_outputs = {i: [] for i in range(n_layers + 1)}
    n_batches = 0

    for batch in tqdm(loader, desc="Extracting layers"):
        n_batches += 1
        g = batch["gene"].to(device)
        v = batch["expr"].to(device)
        padding_mask = g.eq(pad_idx)

        with torch.no_grad():
            h = build_encoder_input(model, g, v)

            layer_outputs[0].append(h[:, 0, :].detach().cpu())

            for i, layer in enumerate(model.transformer_encoder.layers):
                h = apply_transformer_layer(layer, h, padding_mask)

                if i == n_layers - 1 and getattr(model.transformer_encoder, "norm", None) is not None:
                    h = model.transformer_encoder.norm(h)

                layer_outputs[i + 1].append(h[:, 0, :].detach().cpu())

    if n_batches == 0:
        raise RuntimeError("DataLoader produced no batches. No activations were extracted.")

    extracted = []
    for i in range(n_layers + 1):
        if len(layer_outputs[i]) == 0:
            raise RuntimeError(f"No activations were captured for layer {i}.")
        extracted.append(torch.cat(layer_outputs[i], dim=0))

    return extracted


cell_layers = extract_cell_emb_layers(model, loader, pad_idx, device)

print("Extracted layers:", len(cell_layers))
for i, emb in enumerate(cell_layers):
    print(f"Layer {i} shape: {tuple(emb.shape)}")

# ============================================================
# 13. Save embeddings
# ============================================================
for i, emb in enumerate(cell_layers):
    torch.save(emb.cpu(), os.path.join(outdir, f"cell_emb_layer_{i}.pt"))

# ============================================================
# 14. 5-Fold CV (mean ± std)
# ============================================================
labels = adata_proc.obs["cell_type"].astype(str).values
le = LabelEncoder()
y = le.fit_transform(labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for i, X in enumerate(cell_layers):
    X = X.numpy()

    acc_folds = []
    f1_folds = []

    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])

        acc_folds.append(accuracy_score(y[te], pred))
        f1_folds.append(f1_score(y[te], pred, average="macro"))

    results.append({
        "layer": i,
        "accuracy_mean": np.mean(acc_folds),
        "accuracy_std": np.std(acc_folds, ddof=1),
        "macro_f1_mean": np.mean(f1_folds),
        "macro_f1_std": np.std(f1_folds, ddof=1),
    })

    print(
        f"Layer {i}: "
        f"acc={np.mean(acc_folds):.4f}±{np.std(acc_folds, ddof=1):.4f}, "
        f"f1={np.mean(f1_folds):.4f}±{np.std(f1_folds, ddof=1):.4f}"
    )

df = pd.DataFrame(results).sort_values("layer")
df.to_csv(
    os.path.join(outdir, "scgpt_ts_testis_layer_embedding.csv.csv"),
    index=False,
)
