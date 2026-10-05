"""Layer-wise scgpt pipeline for CZI immune.

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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ============================================================
# 0. Set seed
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
# 1. Load data (RAW counts only)
# ============================================================
model_dir = SCGPT_MODEL_DIR

with open(f"{model_dir}/args.json") as f:
    configs = json.load(f)

adata = sc.read_h5ad(
    "CZI_human_embryonic_meninges_at5-13_weeks_post_conception_immune_cells.h5ad"
)
print("Original cells:", adata.n_obs)

# Filter rare cell types (>=5)
cell_type_counts = adata.obs["cell_type"].value_counts()
keep_cells = cell_type_counts[cell_type_counts >= 5].index
mask = adata.obs["cell_type"].isin(keep_cells).to_numpy()

# ---- RAW counts only (no normalize/log1p) ----
counts = adata.X
var = adata.var
counts = counts[mask]

adata_proc = ad.AnnData(
    X=counts,
    obs=adata.obs[mask].copy(),
    var=var.copy()
)
print("After cell-type filter:", adata_proc.n_obs)

# QC
sc.pp.filter_cells(adata_proc, min_genes=200)
sc.pp.filter_genes(adata_proc, min_cells=3)

# ============================================================
# 2. Load vocab from checkpoint
# ============================================================
vocab_path = configs.get("vocab_path", None)
if vocab_path is not None and os.path.exists(vocab_path):
    vocab = GeneVocab.from_file(vocab_path)
else:
    # fallback (less faithful)
    from scgpt.tokenizer import get_default_gene_vocab
    vocab = get_default_gene_vocab()

# Ensure specials
for tok in ["<pad>", "<cls>", "<eoc>"]:
    if tok not in vocab:
        vocab.append_token(tok)

pad_token = "<pad>"
cls_token = "<cls>"
pad_idx = vocab[pad_token]
cls_id = vocab[cls_token]
pad_value = configs["pad_value"]

def get_vocab_tokens(vocab):
    if hasattr(vocab, "get_stoi"):
        return list(vocab.get_stoi().keys())
    if hasattr(vocab, "itos"):
        return list(vocab.itos)
    return list(vocab)

def get_vocab_stoi(vocab):
    if hasattr(vocab, "get_stoi"):
        return vocab.get_stoi()
    if hasattr(vocab, "token2idx"):
        return vocab.token2idx
    if hasattr(vocab, "_token2id"):
        return vocab._token2id
    toks = get_vocab_tokens(vocab)
    return {t: i for i, t in enumerate(toks)}

vocab_tokens = get_vocab_tokens(vocab)
vocab_stoi = get_vocab_stoi(vocab)

def is_ensembl_vocab(tokens):
    sample = tokens[:1000] if len(tokens) > 1000 else tokens
    return sum(t.startswith("ENSG") for t in sample) > 0.5 * len(sample)

vocab_is_ensembl = is_ensembl_vocab(vocab_tokens)

# ============================================================
# 3. Match genes to vocab (AUTO DETECT)
# ============================================================
def strip_ensembl_version(x):
    return x.split(".")[0] if isinstance(x, str) else x

def add_candidate(name, series, preprocess):
    if series is None:
        return None
    s = pd.Series(series).astype(str)
    return (name, s.apply(preprocess))

candidates = []
candidates.append(add_candidate("gene_name", adata_proc.var.get("gene_name", None), lambda x: x.upper()))
candidates.append(add_candidate("feature_name", adata_proc.var.get("feature_name", None), lambda x: x.upper()))
candidates.append(add_candidate("ensembl_id", adata_proc.var.get("ensembl_id", None), strip_ensembl_version))
candidates.append(add_candidate("ensembl_clean", adata_proc.var.get("ensembl_clean", None), lambda x: x))
candidates.append(add_candidate("Gene stable ID", adata_proc.var.get("Gene stable ID", None), lambda x: x))
candidates.append(add_candidate("var_names", adata_proc.var_names, lambda x: x.upper()))
candidates = [c for c in candidates if c is not None]

if vocab_is_ensembl:
    vocab_set = set(vocab_tokens)
    preprocess_vocab = lambda x: x
else:
    vocab_set = set([t.upper() for t in vocab_tokens])
    preprocess_vocab = lambda x: x.upper()

best = None
best_n = -1
for name, series in candidates:
    vals = series.values
    vals_proc = [preprocess_vocab(v) for v in vals]
    n_match = sum(v in vocab_set for v in vals_proc)
    if n_match > best_n:
        best_n = n_match
        best = (name, series, vals_proc)

if best is None or best_n == 0:
    raise RuntimeError(
        f"No gene IDs match scGPT vocab. Available var columns: {list(adata_proc.var.columns)}"
    )

best_name, best_series, best_proc = best
print(f"Best gene ID source: {best_name} (matched {best_n} genes)")

processed_to_token = {}
for tok in vocab_tokens:
    key = preprocess_vocab(tok)
    if key not in processed_to_token:
        processed_to_token[key] = tok

ids = []
for v in best_proc:
    if v in processed_to_token:
        ids.append(vocab_stoi[processed_to_token[v]])
    else:
        ids.append(-1)

adata_proc.var["id_in_vocab"] = ids
adata_proc = adata_proc[:, adata_proc.var["id_in_vocab"] >= 0].copy()
gene_ids = np.array(adata_proc.var["id_in_vocab"])
print("After vocab filter:", adata_proc.n_obs, "cells,", adata_proc.n_vars, "genes")

if adata_proc.n_vars == 0:
    raise RuntimeError("0 genes matched scGPT vocab. Check gene ID mapping.")

# keep cells with at least one gene (plus CLS)
if hasattr(adata_proc.X, "toarray"):
    nnz = np.asarray((adata_proc.X > 0).sum(axis=1)).ravel()
else:
    nnz = (adata_proc.X > 0).sum(axis=1)
adata_proc = adata_proc[nnz >= 1].copy()

# ============================================================
# 4. Build model
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

# Disable nested tensors (fixes NotImplementedError in CLS slicing)
if hasattr(model.transformer_encoder, "enable_nested_tensor"):
    model.transformer_encoder.enable_nested_tensor = False
if hasattr(model.transformer_encoder, "use_nested_tensor"):
    model.transformer_encoder.use_nested_tensor = False
for layer in model.transformer_encoder.layers:
    if hasattr(layer, "enable_nested_tensor"):
        layer.enable_nested_tensor = False

# ============================================================
# 5. Dataset (insert <cls> token explicitly)
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

        # insert CLS at position 0 (value = 0) to match scGPT
        genes = np.insert(genes, 0, self.cls_id)
        values = np.insert(values, 0, 0.0)

        return {
            "genes": torch.tensor(genes, dtype=torch.long),
            "expressions": torch.tensor(values, dtype=torch.float32),
        }

dataset = ProbingDataset(adata_proc.X, gene_ids, cls_id)

# ============================================================
# 6. DataCollator (bins values)
# ============================================================
data_collator = DataCollator(
    do_padding=True,
    pad_token_id=pad_idx,
    pad_value=pad_value,
    do_mlm=False,
    do_binning=True,
    max_length=configs["max_seq_len"],
    sampling=True,
    keep_first_n_tokens=1,  # keep CLS unchanged
)

loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=SequentialSampler(dataset),
    collate_fn=data_collator,
    drop_last=False,
    num_workers=0
)

# ============================================================
# 7. Per-layer cell embedding extraction
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

    with torch.no_grad():
        for batch in loader:
            n_batches += 1
            g = batch["gene"].to(device)
            v = batch["expr"].to(device)
            padding_mask = g.eq(pad_idx)

            h = build_encoder_input(model, g, v)
            layer_outputs[0].append(h[:, 0, :].detach().cpu())

            for i, layer in enumerate(model.transformer_encoder.layers):
                h = apply_transformer_layer(layer, h, padding_mask)

                if i == n_layers - 1 and getattr(model.transformer_encoder, "norm", None) is not None:
                    h = model.transformer_encoder.norm(h)

                layer_outputs[i + 1].append(h[:, 0, :].detach().cpu())

    if n_batches == 0:
        raise RuntimeError("DataLoader produced no batches. No activations were extracted.")

    cell_layers = []
    for i in range(n_layers + 1):
        if len(layer_outputs[i]) == 0:
            raise RuntimeError(f"No activations were captured for layer {i}.")
        cell_layers.append(torch.cat(layer_outputs[i], dim=0))

    return cell_layers

# ============================================================
# 8. Run extraction
# ============================================================
cell_layers = extract_cell_emb_layers(model, loader, pad_idx, device)
print(f"Extracted {len(cell_layers)} layer embeddings")

# ============================================================
# 9. Save per-layer files
# ============================================================
outdir = "scgpt_extract_cell_embedding_per_layer_CZI_immune_v2"
os.makedirs(outdir, exist_ok=True)

for i, emb in enumerate(cell_layers):
    save_path = os.path.join(outdir, f"cell_emb_layer_{i}.pt")
    torch.save(emb.contiguous().cpu(), save_path)
    print(f"Saved layer {i} -> {save_path} | shape = {tuple(emb.shape)}")

# ============================================================
# 10. Linear probe (5-fold CV)
# ============================================================
labels = adata_proc.obs["cell_type"].astype(str).values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for layer_idx, X in enumerate(cell_layers):
    X = X.numpy()
    acc_list, f1_list = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            solver="lbfgs",
            multi_class="auto",
            random_state=42
        )
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        acc_list.append(accuracy_score(y[test_idx], y_pred))
        f1_list.append(f1_score(y[test_idx], y_pred, average="macro"))

    results.append((layer_idx, np.mean(acc_list), np.mean(f1_list)))
    print(f"Layer {layer_idx}: acc={results[-1][1]:.4f}, f1={results[-1][2]:.4f}")

df = pd.DataFrame(results, columns=["layer", "accuracy", "macro_f1"])
df.to_csv(os.path.join(outdir, "scgpt_CZI_immune_layer_embedding.csv"), index=False)
