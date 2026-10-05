"""Layer-wise scgpt pipeline for PBMC3K.

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
from pathlib import Path
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.data_collator import DataCollator

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
# 1. Config
# ============================================================
model_dir = SCGPT_MODEL_DIR
outdir = "scgpt_extract_embedding_per_layer_pbmc3k_raw_logNor"
os.makedirs(outdir, exist_ok=True)

with open(f"{model_dir}/args.json") as f:
    configs = json.load(f)

# ============================================================
# 2. Load PBMC3K
#    - labels from processed
#    - raw counts from pbmc3k()
# ============================================================
adata_lab = sc.datasets.pbmc3k_processed()
adata_lab.obs.rename(columns={"louvain": "cell_type"}, inplace=True)

adata_counts = sc.datasets.pbmc3k()

# align by intersection
common = adata_counts.obs_names.intersection(adata_lab.obs_names)
adata_counts = adata_counts[common].copy()
adata_lab = adata_lab[common].copy()

# filter rare cell types (>=5)
cell_type_counts = adata_lab.obs["cell_type"].value_counts()
keep = cell_type_counts[cell_type_counts >= 5].index
mask = adata_lab.obs["cell_type"].isin(keep).to_numpy()

counts = adata_counts.X[mask]
var = adata_counts.var.copy()
obs = adata_lab.obs[mask].copy()

adata_proc = ad.AnnData(X=counts, obs=obs, var=var)

# ============================================================
# 3. Load vocab (checkpoint)
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
# 4. Gene matching (auto-detect)
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
candidates.append(add_candidate("gene_ids", adata_proc.var.get("gene_ids", None), strip_ensembl_version))
candidates.append(add_candidate("ensembl_id", adata_proc.var.get("ensembl_id", None), strip_ensembl_version))
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

# remove cells with zero genes
if hasattr(adata_proc.X, "toarray"):
    nnz = np.asarray((adata_proc.X > 0).sum(axis=1)).ravel()
else:
    nnz = (adata_proc.X > 0).sum(axis=1)
adata_proc = adata_proc[nnz >= 1].copy()

# ============================================================
# 5. Build model
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

# Disable nested tensors (avoid NotImplementedError)
if hasattr(model.transformer_encoder, "enable_nested_tensor"):
    model.transformer_encoder.enable_nested_tensor = False
if hasattr(model.transformer_encoder, "use_nested_tensor"):
    model.transformer_encoder.use_nested_tensor = False
for layer in model.transformer_encoder.layers:
    if hasattr(layer, "enable_nested_tensor"):
        layer.enable_nested_tensor = False

# ============================================================
# 6. Dataset (insert CLS explicitly)
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
# 7. DataCollator (binning)
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
    drop_last=False,
    num_workers=0
)

# ============================================================
# 8. Per-layer cell embedding extraction
# ============================================================
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

cell_layers = extract_cell_emb_layers(model, loader, pad_idx, device)
print(f"Extracted {len(cell_layers)} layer embeddings")

# ============================================================
# 9. Save embeddings
# ============================================================
for i, emb in enumerate(cell_layers):
    save_path = os.path.join(outdir, f"cell_emb_layer_{i}.pt")
    torch.save(emb.contiguous().cpu(), save_path)
    print(f"Saved layer {i} -> {save_path}")

# ============================================================
# 10. 5-fold CV probe
# ============================================================
labels = adata_proc.obs["cell_type"].astype(str).values
y = LabelEncoder().fit_transform(labels)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = []
for layer_idx, X in enumerate(cell_layers):
    X = X.numpy()
    acc_list, f1_list = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc_list.append(accuracy_score(y_test, y_pred))
        f1_list.append(f1_score(y_test, y_pred, average="macro"))

    results.append({
        "layer": layer_idx,
        "accuracy": float(np.mean(acc_list)),
        "macro_f1": float(np.mean(f1_list))
    })
    print(f"Layer {layer_idx}: acc={results[-1]['accuracy']:.4f}, f1={results[-1]['macro_f1']:.4f}")

df = pd.DataFrame(results)
df.to_csv(os.path.join(outdir, "scgpt_pbmc3k_layer_results.csv"), index=False)
