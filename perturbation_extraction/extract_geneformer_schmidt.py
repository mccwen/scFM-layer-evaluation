"""Layer-wise geneformer pipeline for Schmidt.

This file contains one dataset pipeline only. Raw data, checkpoints, and
generated activations are supplied through local configuration.
"""

import warnings

from tqdm import TqdmWarning

import pandas as pd

import numpy as np

import scanpy as sc

import mygene

from pathlib import Path

from geneformer import TranscriptomeTokenizer

import os

import random

import torch

import torch.nn as nn

from tqdm.auto import tqdm

from datasets import load_from_disk

import matplotlib.pyplot as plt

from transformers import BertForMaskedLM

# Convert gene names to emsembl IDs for schmidt data (source: http://www.perturbase.cn/repository)

adata=sc.read_h5ad("Schmidt.h5ad")

# query my genes for emsemble iDs
genes= adata.var_names.tolist()

mg = mygene.MyGeneInfo()

res = mg.querymany(
    genes,
    scopes="symbol",      # input type
    fields="ensembl.gene",
    species="human",      # or "mouse"
    as_dataframe=True
)
print(res["ensembl.gene"].nunique())
res["ensembl.gene"].unique()

res= pd.DataFrame(res)
res2= res["ensembl.gene"].dropna().reset_index()
res2_dict= dict(zip(res2["query"], res2["ensembl.gene"]))


adata.var["ensembl_id"]= adata.var_names.map(res2_dict)
mask = adata.var["ensembl_id"].notna()
adata= adata[:, mask].copy()

adata = adata[:, adata.var["ensembl_id"].notna()].copy()
adata.var["gene_name"]= adata.var.index
adata.var.index = adata.var["ensembl_id"]
adata.obs["n_counts"] = adata.obs["total_counts"]
adata.write_h5ad("geneformer_ensemblID_Schmidt.h5ad")
adata

# Tokenize Schmidy data to meet Geneformer requirements

input_path = Path("./geneformer_ensemblID_Schmidt.h5ad")
data_dir = input_path.parent

tk = TranscriptomeTokenizer(
    nproc=4,
    model_version="V2",
    custom_attr_name_dict={"perturbation": "perturbation"}
)

tk.tokenize_data(
    data_directory=str(data_dir),
    output_directory=str(data_dir / "tokenized"),
    output_prefix="Schmidt",
    file_format="h5ad",
    input_identifier=input_path.stem  # <-- must match filename stem
)

# Extract layer-wise activations and 5-fold cross validation to predict perturbation-inducced gene expression shift

# ============================================================
# Geneformer: Unseen-Gene Expression Probe (Schmidt)
# Mean-pool per layer
# + Single-seed 5-fold perturbation-size CV (control always in train)
# ============================================================
import random

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from datasets import load_from_disk
import matplotlib.pyplot as plt
from transformers import BertForMaskedLM

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
# 2. Configuration
# ============================================================
MODEL_DIR = "Geneformer-V2-104M"
DATASET_PATH = "./tokenized/Schmidt.dataset"

# IMPORTANT: use the Ensembl‑converted Schmidt file you used for tokenization
SCHMIDT_PATH = "./geneformer_ensemblID_Schmidt.h5ad"

OUTPUT_DIR = "./geneformer_schmidt_unseen_gene_results"
LAYER_KEY = "logNor"
BATCH_SIZE = 16

# Perturbation CV settings
PERT_KEY = "perturbation"
CONTROL_LABEL = "control"
K_FOLDS = 5
MIN_PERT_CELLS = 25
RIDGE_L2 = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 3. Utilities
# ============================================================
def pad_tensor_list(tensor_list, max_len, pad_token_id, model_input_size):
    padded_list = []
    for t in tensor_list:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.long)
        if len(t) > model_input_size:
            t = t[:model_input_size]
        pad_len = max_len - len(t)
        if pad_len > 0:
            t = torch.cat([t, torch.full((pad_len,), pad_token_id, dtype=t.dtype)], dim=0)
        padded_list.append(t)
    return torch.stack(padded_list)

def gen_attention_mask(input_ids, pad_token_id=0):
    return (input_ids != pad_token_id).long()

def get_model_input_size(model):
    return model.config.max_position_embeddings if hasattr(model, "config") else 2048

class GeneProbe(nn.Module):
    def __init__(self, cell_dim, gene_dim):
        super().__init__()
        self.linear = nn.Linear(cell_dim + gene_dim, 1)

    def forward(self, cell_emb, gene_emb):
        x = torch.cat([cell_emb, gene_emb], dim=-1)
        return self.linear(x).squeeze(-1)

def mean_pool_gene_tokens(hidden_state, input_ids, pad_id, cls_id, eos_id):
    # mask out PAD/CLS/EOS, keep only gene tokens
    mask = (input_ids != pad_id)
    if cls_id is not None:
        mask = mask & (input_ids != cls_id)
    if eos_id is not None:
        mask = mask & (input_ids != eos_id)

    mask = mask.unsqueeze(-1)  # [B, L, 1]
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts

def fit_ridge_closed_form(X, Y, l2=1e-4):
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    reg = np.eye(d + 1) * l2
    reg[-1, -1] = 0
    W = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ Y)
    return W

def predict_ridge(X, W):
    Xb = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return Xb @ W

def pcc_delta(y_true, y_pred, eps=1e-8):
    pccs = []
    for i in range(y_true.shape[0]):
        yt = y_true[i]
        yp = y_pred[i]
        if yt.std() < eps or yp.std() < eps:
            continue
        pcc = np.corrcoef(yt, yp)[0, 1]
        if np.isfinite(pcc):
            pccs.append(pcc)
    return float(np.mean(pccs)) if pccs else 0.0

def build_stratified_kfold_groups(pert_names, pert_sizes, k, seed):
    rng = np.random.default_rng(seed)
    perts = list(pert_names)
    rng.shuffle(perts)
    perts.sort(key=lambda p: pert_sizes[p], reverse=True)

    groups = [[] for _ in range(k)]
    sizes = [0] * k
    for p in perts:
        idx = np.argmin(sizes)
        groups[idx].append(p)
        sizes[idx] += pert_sizes[p]
    return groups

def resolve_special_token_ids(model, tokenizer):
    # Try model config first
    pad_id = getattr(model.config, "pad_token_id", None)
    cls_id = getattr(model.config, "cls_token_id", None)
    eos_id = getattr(model.config, "sep_token_id", None)

    # Try tokenizer attributes
    if pad_id is None and hasattr(tokenizer, "pad_token_id"):
        pad_id = tokenizer.pad_token_id
    if cls_id is None and hasattr(tokenizer, "cls_token_id"):
        cls_id = tokenizer.cls_token_id
    if eos_id is None and hasattr(tokenizer, "sep_token_id"):
        eos_id = tokenizer.sep_token_id

    # Try tokenizer dicts
    token_dict = None
    for attr in ["gene_token_dict", "token_dict", "token_dictionary", "gene_vocab"]:
        if hasattr(tokenizer, attr):
            obj = getattr(tokenizer, attr)
            if isinstance(obj, dict):
                token_dict = obj
                break

    if token_dict is not None:
        if cls_id is None:
            for k in ["[CLS]", "<cls>", "CLS"]:
                if k in token_dict:
                    cls_id = token_dict[k]
                    break
        if eos_id is None:
            for k in ["[SEP]", "<sep>", "<eos>", "SEP", "EOS"]:
                if k in token_dict:
                    eos_id = token_dict[k]
                    break
        if pad_id is None:
            for k in ["[PAD]", "<pad>", "PAD"]:
                if k in token_dict:
                    pad_id = token_dict[k]
                    break

    if pad_id is None:
        pad_id = 0

    return pad_id, cls_id, eos_id

# ============================================================
# 4. Load Schmidt expression
# ============================================================
print("Loading Schmidt...")
adata = sc.read_h5ad(SCHMIDT_PATH)

if LAYER_KEY in adata.layers:
    X = adata.layers[LAYER_KEY]
else:
    print(f"Warning: '{LAYER_KEY}' not found, using adata.X.")
    X = adata.X

if hasattr(X, "toarray"):
    X = X.toarray()

expr = X.astype(np.float32)
genes = np.array(adata.var_names)

# ============================================================
# 5. Load tokenized Geneformer dataset
# ============================================================
print("Loading tokenized Geneformer dataset...")
dataset = load_from_disk(DATASET_PATH)

if "cell_id" in dataset.column_names:
    cell_ids = dataset["cell_id"]
    adata = adata[cell_ids].copy()
    expr = expr[adata.obs_names.get_indexer(cell_ids)]
else:
    print("Warning: no cell_id in dataset; assuming same order as Schmidt.h5ad")

# ============================================================
# 6. Load Geneformer model + vocab (no .pkl)
# ============================================================
print(f"Loading Geneformer model from {MODEL_DIR}...")
model = BertForMaskedLM.from_pretrained(MODEL_DIR, output_hidden_states=True)
model = model.to(device).eval()

print("Loading gene vocab from TranscriptomeTokenizer (no .pkl)...")
tk = TranscriptomeTokenizer(model_version="V2")

pad_token_id, cls_token_id, eos_token_id = resolve_special_token_ids(model, tk)
model_input_size = get_model_input_size(model)

# robustly grab the gene vocab
gene_to_token = None
for attr in ["gene_token_dict", "token_dict", "token_dictionary", "gene_vocab"]:
    if hasattr(tk, attr):
        obj = getattr(tk, attr)
        if isinstance(obj, dict):
            gene_to_token = obj
            break

if gene_to_token is None:
    raise ValueError("Could not access gene token dict from TranscriptomeTokenizer.")

# keep genes that exist in vocab
valid_mask = np.array([g in gene_to_token for g in genes])
genes = genes[valid_mask]
expr = expr[:, valid_mask]

token_ids = np.array([gene_to_token[g] for g in genes], dtype=np.int64)

# gene embeddings from model vocab
with torch.no_grad():
    gene_emb = model.get_input_embeddings().weight[token_ids].detach().cpu().numpy()

print(f"Using {len(genes)} genes that exist in Geneformer vocab.")
print(f"Special token ids: pad={pad_token_id}, cls={cls_token_id}, eos/sep={eos_token_id}")

# ============================================================
# 7. Extract layer embeddings (mean pooled, faithful)
# ============================================================
def get_all_layer_embs(model, filtered_input_data, pad_token_id, forward_batch_size, output_dir, device):
    model.eval()
    model.to(device)

    total_batch_length = len(filtered_input_data)
    all_layer_embs_list = {}

    print(f"Extracting embeddings from {total_batch_length} cells...")

    for i in tqdm(range(0, total_batch_length, forward_batch_size)):
        max_range = min(i + forward_batch_size, total_batch_length)
        minibatch = filtered_input_data.select(range(i, max_range))

        max_len = max(minibatch["length"])
        input_data_minibatch = minibatch["input_ids"]

        input_data_minibatch = pad_tensor_list(
            input_data_minibatch, max_len, pad_token_id, model_input_size
        )

        attn_mask = gen_attention_mask(input_data_minibatch, pad_token_id).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_data_minibatch.to(device),
                attention_mask=attn_mask,
                output_hidden_states=True
            )

        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            mean_embs = mean_pool_gene_tokens(
                hidden_state,
                input_data_minibatch.to(device),
                pad_token_id,
                cls_token_id,
                eos_token_id
            )
            all_layer_embs_list.setdefault(layer_idx, []).append(mean_embs.cpu())

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    final_results = {}
    for layer_idx, embs_list in all_layer_embs_list.items():
        layer_tensor = torch.cat(embs_list, dim=0)
        final_results[layer_idx] = layer_tensor.numpy()
        torch.save(layer_tensor, out / f"layer_{layer_idx}_activations.pt")

    return final_results

print("Extracting layer embeddings...")
layer_embeddings = get_all_layer_embs(
    model=model,
    filtered_input_data=dataset,
    pad_token_id=pad_token_id,
    forward_batch_size=BATCH_SIZE,
    output_dir=OUTPUT_DIR,
    device=device
)

# ============================================================
# 8. Single-seed perturbation-size 5-fold CV (control always in train)
# ============================================================
perts = adata.obs[PERT_KEY].astype(str).values
if CONTROL_LABEL not in np.unique(perts):
    raise ValueError(f"Control label '{CONTROL_LABEL}' not found in {PERT_KEY}.")

ctrl_mean = expr[perts == CONTROL_LABEL].mean(axis=0)

pert_names_all = sorted(set(perts) - {CONTROL_LABEL})
pert_sizes_all = {p: int((perts == p).sum()) for p in pert_names_all}
pert_names = [p for p in pert_names_all if pert_sizes_all[p] >= MIN_PERT_CELLS]
pert_sizes = {p: pert_sizes_all[p] for p in pert_names}

folds = build_stratified_kfold_groups(pert_names, pert_sizes, K_FOLDS, SEED)

results = []

for layer_id in sorted(layer_embeddings.keys()):
    emb = layer_embeddings[layer_id]

    Xp, Yp = [], []
    for p in pert_names:
        m = perts == p
        Xp.append(emb[m].mean(0))
        Yp.append(expr[m].mean(0) - ctrl_mean)

    Xp, Yp = np.vstack(Xp), np.vstack(Yp)

    mse_list, pcc_list = [], []
    for g in folds:
        g = [p for p in g if p in pert_names]
        if len(g) == 0:
            continue

        test = np.isin(pert_names, g)
        train = ~test

        W = fit_ridge_closed_form(Xp[train], Yp[train], RIDGE_L2)
        pred = predict_ridge(Xp[test], W)

        mse_list.append(np.mean((pred - Yp[test])**2))
        pcc_list.append(pcc_delta(Yp[test], pred))

    results.append((layer_id, float(np.mean(mse_list)), float(np.mean(pcc_list))))

df_out = pd.DataFrame(results, columns=["layer", "mse_delta", "pcc_delta"]).sort_values("layer")

metrics_csv_path = os.path.join(
    OUTPUT_DIR,
    "geneformer_5fold_perturb_size_mse_pcc_per_layer.csv"
)
df_out.to_csv(metrics_csv_path, index=False)
print("Saved metrics CSV:", metrics_csv_path)
print(df_out.head())
