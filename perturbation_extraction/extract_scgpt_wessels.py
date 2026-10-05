"""Layer-wise scgpt pipeline for Wessels.

This file contains one dataset pipeline only. Raw data, checkpoints, and
generated activations are supplied through local configuration.
"""

import os

import json

import random

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import scanpy as sc

import scipy.sparse

import inspect

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, SequentialSampler

from scgpt.model import TransformerModel

from scgpt.utils import load_pretrained

from scgpt.tokenizer import get_default_gene_vocab

from scgpt.data_collator import DataCollator

from sklearn.metrics import mean_squared_error

from scipy import sparse

from pathlib import Path

from sklearn.linear_model import Ridge

from scgpt.tokenizer.gene_tokenizer import GeneVocab

SCGPT_MODEL_DIR = os.environ.get("SCGPT_MODEL_DIR")
if not SCGPT_MODEL_DIR:
    raise RuntimeError(
        "Set SCGPT_MODEL_DIR to the local scGPT checkpoint directory "
        "before running this extractor."
    )

from scipy import sparse
from pathlib import Path
from sklearn.linear_model import Ridge

from scgpt.tokenizer.gene_tokenizer import GeneVocab

# ========================
# 0. Reproducibility
# ========================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========================
# 1. Paths / Config
# ========================
MODEL_DIR = SCGPT_MODEL_DIR
DATA_PATH = "Wessels.h5ad"

ACT_DIR = "scgpt_cls_layers_Wessels_logNor"
OUTDIR = "scgpt_cls_layers_Wessels_logNor/scgpt_cls_layers_Wessels_logNor_perturb_effect_predict_delta"
os.makedirs(ACT_DIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

MIN_PERT_CELLS = 25
K_FOLDS = 5
RIDGE_ALPHA = 1.0

PERT_COL_OVERRIDE = None
CONTROL_LABEL_OVERRIDE = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# 2. Load model configs + vocab
# ========================
with open(f"{MODEL_DIR}/args.json") as f:
    configs = json.load(f)

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

# ========================
# 3. Load Wessels (logNor) as input
# ========================
adata = sc.read_h5ad(DATA_PATH)
if "logNor" not in adata.layers:
    raise ValueError("Expected layer['logNor'] in Wessels.h5ad but did not find it.")

adata_proc = sc.AnnData(
    X=adata.layers["logNor"],
    obs=adata.obs.copy(),
    var=adata.var.copy()
)

# ========================
# 4. Map genes to vocab (auto detect)
# ========================
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
candidates.append(add_candidate("var_names", adata_proc.var_names, lambda x: x.upper()))
candidates = [c for c in candidates if c is not None]

vocab_is_ensembl = is_ensembl_vocab(vocab_tokens)
if vocab_is_ensembl:
    vocab_set = set(vocab_tokens)
    preprocess_vocab = lambda x: x
else:
    vocab_set = set([t.upper() for t in vocab_tokens])
    preprocess_vocab = lambda x: x.upper()

best = None
best_n = -1
for name, series in candidates:
    vals_proc = [preprocess_vocab(v) for v in series.values]
    n_match = sum(v in vocab_set for v in vals_proc)
    if n_match > best_n:
        best_n = n_match
        best = (name, series, vals_proc)

if best is None or best_n == 0:
    raise RuntimeError("No gene IDs match scGPT vocab.")

_, _, best_proc = best

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

# remove empty cells
if hasattr(adata_proc.X, "toarray"):
    nnz = np.asarray((adata_proc.X > 0).sum(axis=1)).ravel()
else:
    nnz = (adata_proc.X > 0).sum(axis=1)
adata_proc = adata_proc[nnz >= 1].copy()

# ========================
# 5. Build scGPT model
# ========================
sig = inspect.signature(TransformerModel.__init__)
filtered_configs = {k: v for k, v in configs.items() if k in sig.parameters}
filtered_configs["vocab"] = vocab
filtered_configs["pad_token"] = pad_token
filtered_configs["ntoken"] = len(vocab)
filtered_configs.setdefault("d_model", configs["embsize"])
filtered_configs.setdefault("nhead", configs["nheads"])

model = TransformerModel(**filtered_configs)
state = torch.load(f"{MODEL_DIR}/best_model.pt", map_location="cpu")
load_pretrained(model, state)

model.to(device).eval()

# disable nested tensor to avoid slice error
if hasattr(model.transformer_encoder, "enable_nested_tensor"):
    model.transformer_encoder.enable_nested_tensor = False
if hasattr(model.transformer_encoder, "use_nested_tensor"):
    model.transformer_encoder.use_nested_tensor = False
for layer in model.transformer_encoder.layers:
    if hasattr(layer, "enable_nested_tensor"):
        layer.enable_nested_tensor = False

# ========================
# 6. Dataset + Collator
# ========================
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

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    sampler=torch.utils.data.SequentialSampler(dataset),
    collate_fn=data_collator,
    drop_last=False,
    num_workers=0
)

# ========================
# 7. Extract layer-wise cell embeddings
# ========================
def extract_cell_emb_layers(model, loader, pad_idx, device):
    model.eval()
    n_layers = len(model.transformer_encoder.layers)
    layer_outputs = {i: [] for i in range(n_layers + 1)}

    def layer0_hook(module, args):
        x = args[0]
        layer_outputs[0].append(x[:, 0, :].detach().cpu())

    h0 = model.transformer_encoder.layers[0].register_forward_pre_hook(layer0_hook)

    hooks = []
    for li, layer in enumerate(model.transformer_encoder.layers):
        def make_hook(idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if idx == n_layers - 1 and hasattr(model, "final_layernorm"):
                    out = model.final_layernorm(out)
                layer_outputs[idx + 1].append(out[:, 0, :].detach().cpu())
            return hook
        hooks.append(layer.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        for batch in loader:
            g = batch["gene"].to(device)
            v = batch["expr"].to(device)
            model(g, values=v, src_key_padding_mask=g.eq(pad_idx))

    h0.remove()
    for h in hooks:
        h.remove()

    cell_layers = [torch.cat(layer_outputs[i], dim=0) for i in range(n_layers + 1)]
    return cell_layers

print("\nExtracting scGPT layer embeddings...")
cell_layers = extract_cell_emb_layers(model, loader, pad_idx, device)

for i, emb in enumerate(cell_layers):
    save_path = os.path.join(ACT_DIR, f"cell_emb_layer_{i}.pt")
    torch.save(emb, save_path)
    print(f"Saved layer {i} -> {save_path}")

# ========================
# 8. 5-fold CV (size-based)
# ========================
def pick_perturbation_column(adata):
    if PERT_COL_OVERRIDE is not None:
        return PERT_COL_OVERRIDE
    candidates = ["perturbation", "pert", "condition", "treatment",
                  "perturbation_group", "perturbation_label", "target_gene"]
    for c in candidates:
        if c in adata.obs.columns:
            return c
    raise ValueError("No suitable perturbation column found.")

def pick_control_label(values):
    if CONTROL_LABEL_OVERRIDE is not None:
        return CONTROL_LABEL_OVERRIDE
    control_candidates = ["control", "ctrl", "vehicle", "dmso", "untreated", "wildtype", "wt", "none"]
    lower_map = {v.lower(): v for v in values}
    for c in control_candidates:
        if c in lower_map:
            return lower_map[c]
    raise ValueError("No control label found.")

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

def pearson_corr(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.std() < 1e-8 or y.std() < 1e-8:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

pert_col = pick_perturbation_column(adata_proc)
pert_labels = adata_proc.obs[pert_col].astype(str).values
control_label = pick_control_label(np.unique(pert_labels))

pert_names_all = sorted(set(pert_labels) - {control_label})
pert_names_set = {p for p in pert_names_all if int((pert_labels == p).sum()) >= MIN_PERT_CELLS}
pert_names = sorted(pert_names_set)

keep_mask = np.isin(pert_labels, pert_names) | (pert_labels == control_label)
pert_labels = pert_labels[keep_mask]

Y = adata_proc.X
if sparse.issparse(Y):
    Y = Y.toarray()
Y = np.asarray(Y, dtype=np.float32)
Y = Y[keep_mask]

fold_groups = make_size_folds(pert_names, pert_labels, k=K_FOLDS, seed=SEED)

results = []

for layer_idx, X in enumerate(cell_layers):
    X = X.numpy()
    X = X[keep_mask]

    mse_list = []
    pcc_list = []

    for group in fold_groups:
        group = [p for p in group if p in pert_names_set]
        if len(group) == 0:
            continue

        test_mask = np.isin(pert_labels, group)
        train_mask = ~test_mask  # control always in train

        idx_train = np.where(train_mask)[0]
        idx_test = np.where(test_mask)[0]

        X_train = X[idx_train]
        X_test = X[idx_test]
        y_train = Y[idx_train]
        y_test = Y[idx_test]
        test_labels = pert_labels[idx_test]
        train_labels = pert_labels[idx_train]

        reg = Ridge(alpha=RIDGE_ALPHA)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        ctrl_mask_train = train_labels == control_label
        true_ctrl_mean = y_train[ctrl_mask_train].mean(axis=0)

        for p in group:
            mask = test_labels == p
            if mask.sum() == 0:
                continue
            true_mean = y_test[mask].mean(axis=0)
            pred_mean = y_pred[mask].mean(axis=0)

            delta_true = true_mean - true_ctrl_mean
            delta_pred = pred_mean - true_ctrl_mean

            mse_list.append(mean_squared_error(delta_true, delta_pred))
            pcc_list.append(pearson_corr(delta_true, delta_pred))

    mse_delta = float(np.nanmean(mse_list)) if mse_list else float("nan")
    pcc_delta = float(np.nanmean(pcc_list)) if pcc_list else float("nan")

    results.append((layer_idx, mse_delta, pcc_delta))
    print(f"Layer {layer_idx}: MSE={mse_delta:.6f}, PCC={pcc_delta:.4f}")

df = pd.DataFrame(results, columns=["layer", "mse_delta", "pcc_delta"])
csv_path = os.path.join(OUTDIR, "scgpt_wessels_mse_pcc_per_layer.csv")
df.to_csv(csv_path, index=False)
print(f"Saved results CSV to {csv_path}")

# ========================
# 9. Plot
# ========================
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

axes[0].plot(df["layer"], df["mse_delta"], marker='o', color='blue')
axes[0].set_title("MSE-Delta per Layer", fontsize=13)
axes[0].set_xlabel("Layer Index", fontsize=11)
axes[0].set_ylabel("MSE-Delta", fontsize=11)
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].plot(df["layer"], df["pcc_delta"], marker='o', color='orange')
axes[1].set_title("PCC-Delta per Layer", fontsize=13)
axes[1].set_xlabel("Layer Index", fontsize=11)
axes[1].set_ylabel("PCC-Delta", fontsize=11)
axes[1].grid(True, linestyle="--", alpha=0.5)

fig.suptitle("Perturbation Shift Prediction per scGPT Layer (Wessels)", fontsize=14)

plot_path = os.path.join(OUTDIR, "scgpt_wessels_mse_pcc_per_layer.svg")
fig.tight_layout()
fig.savefig(plot_path, bbox_inches="tight")

print(f"Plot saved to {plot_path}")
