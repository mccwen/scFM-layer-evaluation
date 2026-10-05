"""Layer-wise scgpt pipeline for Schmidt.

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

# ============================================================
# scGPT: Unseen-Gene Linear Probe + MSE / MSE-delta / PCC-delta
# Extract CLS activations per layer, save .pt, probe unseen genes
# ============================================================
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

# =========================
# Config
# =========================
SEED = 42
MODEL_DIR = SCGPT_MODEL_DIR
DATASET_PATH = "Schmidt.h5ad"

ACT_DIR = "scgpt_layers_Schmidt_logNor"
OUT_DIR = "scgpt_layers_Schmidt_logNo"
os.makedirs(ACT_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

GENE_HOLDOUT_FRAC = 0.2
RANDOM_SEED = 7
PAIRS_PER_CELL_TRAIN = 256
PAIRS_PER_CELL_EVAL = 256
BATCH_SIZE = 1024
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4

PERT_COL_OVERRIDE = None       # e.g. "perturbation"
CONTROL_LABEL_OVERRIDE = None  # e.g. "control"

# =========================
# Reproducibility
# =========================
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# Helper: probe model
# =========================
class GeneProbe(nn.Module):
    def __init__(self, cell_dim, gene_dim):
        super().__init__()
        self.linear = nn.Linear(cell_dim + gene_dim, 1)

    def forward(self, cell_emb, gene_emb):
        x = torch.cat([cell_emb, gene_emb], dim=-1)
        return self.linear(x).squeeze(-1)

def get_steps(n_cells, pairs_per_cell, batch_size):
    total_pairs = n_cells * pairs_per_cell
    return max(1, int(np.ceil(total_pairs / batch_size)))

def pearson_corr(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

# =========================
# Helper: find gene embedding
# =========================
def find_gene_embedding_layer(model, vocab_size):
    import torch.nn as nn
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and module.num_embeddings == vocab_size:
            candidates.append((name, module))
    if not candidates:
        raise ValueError("Could not find gene embedding layer with num_embeddings == vocab_size.")
    if len(candidates) > 1:
        print("Warning: multiple embedding layers match vocab size. Using:", candidates[0][0])
    return candidates[0][1]

# =========================
# 1. Load model config + vocab
# =========================
with open(f"{MODEL_DIR}/args.json") as f:
    configs = json.load(f)

vocab = get_default_gene_vocab()
for tok in ["<pad>", "<cls>", "<eoc>"]:
    if tok not in vocab:
        vocab.append_token(tok)

pad_token = "<pad>"
cls_token = "<cls>"
pad_idx = vocab[pad_token]
cls_id = vocab[cls_token]
pad_value = configs["pad_value"]

# =========================
# 2. Load Schmidt (logNor) + filter genes to vocab
# =========================
adata = sc.read_h5ad(DATASET_PATH)
if "logNor" not in adata.layers:
    raise ValueError("Expected 'logNor' in adata.layers for Schmidt.")

X = adata.layers["logNor"]
adata_proc = sc.AnnData(
    X=X,
    obs=adata.obs.copy(),
    var=adata.var.copy()
)

adata_proc.var["id_in_vocab"] = [
    vocab[g] if g in vocab else -1
    for g in adata_proc.var_names
]
adata_proc = adata_proc[:, adata_proc.var["id_in_vocab"] >= 0]
gene_ids_vocab = np.array(adata_proc.var["id_in_vocab"])

Y = adata_proc.X
if scipy.sparse.issparse(Y):
    Y = Y.toarray()
Y = np.asarray(Y, dtype=np.float32)

n_cells, n_genes = Y.shape

# =========================
# 3. Build model (for gene embeddings + extraction)
# =========================
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

if hasattr(model.transformer_encoder, "enable_nested_tensor"):
    model.transformer_encoder.enable_nested_tensor = False
for layer in model.transformer_encoder.layers:
    if hasattr(layer, "enable_nested_tensor"):
        layer.enable_nested_tensor = False

# =========================
# 4. Get gene embeddings aligned to adata genes
# =========================
gene_emb_layer = find_gene_embedding_layer(model, len(vocab))
gene_emb_all = gene_emb_layer.weight.detach().cpu().numpy().astype(np.float32)
gene_emb = gene_emb_all[gene_ids_vocab]

# =========================
# 5. Extract CLS activations per layer + save
# =========================
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

dataset = ProbingDataset(adata_proc.X, gene_ids_vocab, cls_id)
loader = DataLoader(
    dataset,
    batch_size=32,
    sampler=SequentialSampler(dataset),  # FIX: use dataset, not sparse matrix
    collate_fn=data_collator,
    drop_last=False,
    num_workers=0
)

def extract_cls_layers_faithful(model, loader, pad_idx, device):
    model.eval()
    n_layers = len(model.transformer_encoder.layers)
    cls_outputs = {i: [] for i in range(n_layers + 1)}

    def layer0_hook(module, args):
        x = args[0]
        cls_outputs[0].append(x[:, 0, :].detach().cpu())

    h0 = model.transformer_encoder.layers[0].register_forward_pre_hook(layer0_hook)

    hooks = []
    for li, layer in enumerate(model.transformer_encoder.layers):
        def make_hook(idx):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if idx == n_layers - 1 and hasattr(model, "final_layernorm"):
                    out = model.final_layernorm(out)
                cls_outputs[idx + 1].append(out[:, 0, :].detach().cpu())
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

    cls_layers = [torch.cat(cls_outputs[i], dim=0) for i in range(n_layers + 1)]
    return cls_layers

cls_layers = extract_cls_layers_faithful(model, loader, pad_idx, device)
print(f"Extracted {len(cls_layers)} CLS layers")

for i, emb in enumerate(cls_layers):
    torch.save(emb, os.path.join(ACT_DIR, f"cls_layer_{i}.pt"))
    print(f"Saved layer {i} -> {ACT_DIR}/cls_layer_{i}.pt")

# =========================
# 6. Perturbation metadata
# =========================
def pick_perturbation_column(adata):
    if PERT_COL_OVERRIDE is not None:
        if PERT_COL_OVERRIDE not in adata.obs.columns:
            raise ValueError(f"PERT_COL_OVERRIDE='{PERT_COL_OVERRIDE}' not in adata.obs.columns.")
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

pert_col = pick_perturbation_column(adata_proc)
pert_labels = adata_proc.obs[pert_col].astype(str).values
control_label = pick_control_label(np.unique(pert_labels))
print(f"Using perturbation column: {pert_col}")
print(f"Using control label: {control_label}")

# =========================
# 7. Unseen-gene split
# =========================
rng = np.random.default_rng(RANDOM_SEED)
all_gene_idx = np.arange(n_genes)
rng.shuffle(all_gene_idx)
n_test = int(n_genes * GENE_HOLDOUT_FRAC)
test_genes = all_gene_idx[:n_test]
train_genes = all_gene_idx[n_test:]

# =========================
# 8. Per-layer probe + MSE, MSE-delta, PCC-delta
# =========================
results = []

for layer_idx, X in enumerate(cls_layers):
    X = X.numpy().astype(np.float32)

    cell_emb_t = torch.tensor(X, device=device)
    gene_emb_t = torch.tensor(gene_emb, device=device)

    cell_dim = cell_emb_t.shape[1]
    gene_dim = gene_emb_t.shape[1]

    probe = GeneProbe(cell_dim=cell_dim, gene_dim=gene_dim).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    steps_per_epoch = get_steps(n_cells, PAIRS_PER_CELL_TRAIN, BATCH_SIZE)
    eval_steps = get_steps(n_cells, PAIRS_PER_CELL_EVAL, BATCH_SIZE)

    # ---- Train on train genes ----
    probe.train()
    for _ in range(EPOCHS):
        for _ in range(steps_per_epoch):
            cell_ids = rng.integers(0, n_cells, size=BATCH_SIZE)
            gene_idx = rng.choice(train_genes, size=BATCH_SIZE, replace=True)

            yb = Y[cell_ids, gene_idx]
            yb = torch.tensor(yb, dtype=torch.float32, device=device)

            cell_ids_t = torch.tensor(cell_ids, device=device)
            gene_idx_t = torch.tensor(gene_idx, device=device)

            cell_batch = cell_emb_t[cell_ids_t]
            gene_batch = gene_emb_t[gene_idx_t]

            opt.zero_grad(set_to_none=True)
            preds = probe(cell_batch, gene_batch)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()

    # ---- Eval MSE on unseen genes ----
    probe.eval()
    mse_sum = 0.0
    count = 0

    with torch.no_grad():
        for _ in range(eval_steps):
            cell_ids = rng.integers(0, n_cells, size=BATCH_SIZE)
            gene_idx = rng.choice(test_genes, size=BATCH_SIZE, replace=True)

            yb = Y[cell_ids, gene_idx]
            yb = torch.tensor(yb, dtype=torch.float32, device=device)

            cell_ids_t = torch.tensor(cell_ids, device=device)
            gene_idx_t = torch.tensor(gene_idx, device=device)

            cell_batch = cell_emb_t[cell_ids_t]
            gene_batch = gene_emb_t[gene_idx_t]

            preds = probe(cell_batch, gene_batch)
            mse_sum += torch.sum((preds - yb) ** 2).item()
            count += yb.numel()

    mse = mse_sum / max(count, 1)

    # ---- Compute MSE-delta + PCC-delta on test genes ----
    batch_size_pred = 512
    pred_test = []
    with torch.no_grad():
        for i in range(0, n_cells, batch_size_pred):
            cell_batch = cell_emb_t[i:i+batch_size_pred]
            gene_batch = gene_emb_t[test_genes]

            cb = cell_batch.unsqueeze(1).repeat(1, len(test_genes), 1)
            gb = gene_batch.unsqueeze(0).repeat(cell_batch.shape[0], 1, 1)

            cb = cb.reshape(-1, cb.shape[-1])
            gb = gb.reshape(-1, gb.shape[-1])

            preds = probe(cb, gb).reshape(cell_batch.shape[0], len(test_genes))
            pred_test.append(preds.cpu().numpy())

    pred_test = np.concatenate(pred_test, axis=0)
    true_test = Y[:, test_genes]

    ctrl_mask = pert_labels == control_label
    if ctrl_mask.sum() == 0:
        raise ValueError("No control cells found. Set CONTROL_LABEL_OVERRIDE.")

    true_ctrl_mean = true_test[ctrl_mask].mean(axis=0)
    pred_ctrl_mean = pred_test[ctrl_mask].mean(axis=0)

    perts = [p for p in np.unique(pert_labels) if p != control_label]
    mse_delta_list = []
    pcc_delta_list = []

    for p in perts:
        mask = pert_labels == p
        if mask.sum() == 0:
            continue
        true_mean = true_test[mask].mean(axis=0)
        pred_mean = pred_test[mask].mean(axis=0)

        delta_true = true_mean - true_ctrl_mean
        delta_pred = pred_mean - pred_ctrl_mean

        mse_delta_list.append(mean_squared_error(delta_true, delta_pred))
        pcc_delta_list.append(pearson_corr(delta_true, delta_pred))

    mse_delta = float(np.nanmean(mse_delta_list))
    pcc_delta = float(np.nanmean(pcc_delta_list))

    results.append((layer_idx, mse, mse_delta, pcc_delta))
    print(f"Layer {layer_idx}: MSE={mse:.6f} | MSE-delta={mse_delta:.6f} | PCC-delta={pcc_delta:.4f}")

# =========================
# 9. Save CSV + plots
# =========================
df_out = pd.DataFrame(results, columns=["layer", "mse_unseen_genes", "mse_delta", "pcc_delta"])
csv_path = os.path.join(OUT_DIR, "scgpt_unseen_gene_mse_mse_delta_pcc_delta_per_layer.csv")
df_out.to_csv(csv_path, index=False)

# MSE plot
plt.figure(figsize=(7, 4))
plt.plot(df_out["layer"], df_out["mse_unseen_genes"], marker="o", color="blue")
plt.xticks(df_out["layer"])
plt.xlabel("Layer")
plt.ylabel("MSE (Unseen Genes)")
plt.title("scGPT Unseen-Gene Probe (Schmidt, logNor)")
plt.grid(True, linestyle="--", alpha=0.5)
mse_plot_path = os.path.join(OUT_DIR, "scgpt_unseen_gene_mse_per_layer.png")
plt.tight_layout()
plt.savefig(mse_plot_path, dpi=300)

# MSE-delta plot
plt.figure(figsize=(7, 4))
plt.plot(df_out["layer"], df_out["mse_delta"], marker="o", color="orange")
plt.xticks(df_out["layer"])
plt.xlabel("Layer")
plt.ylabel("MSE-delta")
plt.title("scGPT Unseen-Gene Perturbation Delta (Schmidt, logNor)")
plt.grid(True, linestyle="--", alpha=0.5)
mse_delta_plot_path = os.path.join(OUT_DIR, "scgpt_unseen_gene_mse_delta_per_layer.png")
plt.tight_layout()
plt.savefig(mse_delta_plot_path, dpi=300)

# PCC-delta plot
plt.figure(figsize=(7, 4))
plt.plot(df_out["layer"], df_out["pcc_delta"], marker="o", color="green")
plt.xticks(df_out["layer"])
plt.xlabel("Layer")
plt.ylabel("PCC-delta")
plt.title("scGPT Unseen-Gene Perturbation PCC-delta (Schmidt, logNor)")
plt.grid(True, linestyle="--", alpha=0.5)
pcc_delta_plot_path = os.path.join(OUT_DIR, "scgpt_unseen_gene_pcc_delta_per_layer.png")
plt.tight_layout()
plt.savefig(pcc_delta_plot_path, dpi=300)

print("Saved:", csv_path)
print("Saved:", mse_plot_path)
print("Saved:", mse_delta_plot_path)
print("Saved:", pcc_delta_plot_path)
