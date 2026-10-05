"""Layer-wise scfoundation pipeline for PBMC3K.

This file contains one dataset pipeline only. Raw data, checkpoints, and
generated activations are supplied through local configuration.
"""

import scanpy as sc

import torch

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

import sys

import os

import scipy.sparse


#

# ==========================================
# 0. Configuration
# ==========================================
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "models")
CKPT_PATH = os.path.join(MODEL_DIR, "models.ckpt")
GENE_INDEX_PATH = os.path.join(BASE_DIR, "OS_scRNA_gene_index.19264.tsv")

CELL_TYPE_COLUMN = "louvain"
BATCH_SIZE = 1
OUTPUT_DIR = os.path.join(BASE_DIR, "scFoundation_scanpyPBMC3K_layer_embeddings")
DATA_PATH = "pbmc3k_processed.h5ad"

N_SPLITS = 5
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. Setup & Imports
# ==========================================
sys.path.append(BASE_DIR)
from load import load_model_frommmf, gatherData  # noqa: E402

# ==========================================
# 2. Gene Alignment Helper
# ==========================================
def align_to_vocab(X_df, gene_list):
    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(
        np.zeros((X_df.shape[0], len(to_fill_columns))),
        columns=to_fill_columns,
        index=X_df.index,
    )
    X_df = pd.concat([X_df, padding_df], axis=1)
    return X_df[gene_list]

def _get_raw_counts(adata):
    if adata.raw is not None:
        return adata.raw.X, adata.raw.var
    if "counts" in adata.layers:
        return adata.layers["counts"], adata.var
    return adata.X, adata.var

# ==========================================
# 3. Data Loader
# ==========================================
def prepare_dataloader(gene_index_path, batch_size=BATCH_SIZE):
    print(f"Loading dataset from {DATA_PATH} ...")
    adata = sc.read_h5ad(DATA_PATH)

    counts, var = _get_raw_counts(adata)
    if scipy.sparse.issparse(counts):
        counts = counts.toarray()

    if "gene_name" in var.columns:
        gene_symbols = var["gene_name"].astype(str).str.upper().values
    else:
        gene_symbols = var.index.astype(str).str.upper().values

    adata_tmp = sc.AnnData(
        X=counts,
        var=pd.DataFrame(index=gene_symbols),
        obs=adata.obs.copy()
    )

    sc.pp.filter_cells(adata_tmp, min_genes=200)
    sc.pp.filter_genes(adata_tmp, min_cells=3)

    ct_counts = adata_tmp.obs[CELL_TYPE_COLUMN].astype(str).value_counts()
    keep_ct = ct_counts[ct_counts >= 5].index
    adata_tmp = adata_tmp[adata_tmp.obs[CELL_TYPE_COLUMN].astype(str).isin(keep_ct)].copy()

    labels_raw = adata_tmp.obs[CELL_TYPE_COLUMN].astype(str).values
    le = LabelEncoder()
    labels = le.fit_transform(labels_raw)

    counts_filtered = adata_tmp.X.copy()
    gene_symbols_filtered = adata_tmp.var_names

    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    lognor = adata_tmp.X

    counts_df = pd.DataFrame(counts_filtered, columns=gene_symbols_filtered)
    lognor_df = pd.DataFrame(lognor, columns=gene_symbols_filtered)

    counts_df = counts_df.groupby(axis=1, level=0).sum()
    lognor_df = lognor_df.groupby(axis=1, level=0).sum()

    gene_list = pd.read_csv(gene_index_path, sep="\t")["gene_name"].astype(str).str.upper().tolist()

    counts_df = align_to_vocab(counts_df, gene_list)
    lognor_df = align_to_vocab(lognor_df, gene_list)

    norm_data = lognor_df.values
    counts_data = counts_df.values
    row_sums = counts_data.sum(axis=1, keepdims=True)

    batches = []
    for i in range(0, len(norm_data), batch_size):
        batch_expr = norm_data[i:i+batch_size]
        batch_counts = counts_data[i:i+batch_size]
        batch_sums = row_sums[i:i+batch_size]

        mask_genes = batch_counts > 0
        mask_meta = np.ones((batch_expr.shape[0], 2), dtype=bool)
        value_mask = np.hstack([mask_genes, mask_meta])

        meta = np.hstack([
            np.full((len(batch_expr), 1), 4.0),
            np.log10(batch_sums + 1e-8)
        ])

        batch_full = np.hstack([batch_expr, meta])

        batches.append({
            "raw_x": torch.tensor(batch_full).float(),
            "mask": torch.tensor(value_mask),
            "labels": labels[i:i+batch_size]
        })

    return batches, labels

# ==========================================
# 4. scFoundation Pooling (official)
# ==========================================
def pool_cell_embedding_all(geneemb):
    geneemb1 = geneemb[:, -1, :]
    geneemb2 = geneemb[:, -2, :]
    geneemb3 = torch.max(geneemb[:, :-2, :], dim=1).values
    geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)
    return torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)

# ==========================================
# 5. Main
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_model_frommmf(CKPT_PATH, key="cell")
    model = model.to(device).eval()

    batches, all_labels = prepare_dataloader(GENE_INDEX_PATH)

    pad_token_id = config["pad_token_id"]
    features = {}

    print("Extracting features...")
    with torch.inference_mode():
        for batch in tqdm(batches):
            raw_x = batch["raw_x"].to(device)

            x, x_pad = gatherData(raw_x, batch["mask"].to(device), pad_token_id)
            x_pad = x_pad.bool()

            x_emb = model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)

            gene_ids = torch.arange(raw_x.shape[1], device=device).unsqueeze(0).repeat(raw_x.shape[0], 1)
            pos_ids, _ = gatherData(gene_ids, batch["mask"].to(device), pad_token_id)
            p_emb = model.pos_emb(pos_ids)

            x_input = x_emb + p_emb

            # layer_0
            features.setdefault("layer_0", []).append(pool_cell_embedding_all(x_input).cpu().numpy())

            # transformer encoder layers
            h = x_input
            for i, layer in enumerate(model.encoder.transformer_encoder):
                h = layer(h, src_key_padding_mask=x_pad)
                features.setdefault(f"layer_{i+1}", []).append(pool_cell_embedding_all(h).cpu().numpy())

            # final LayerNorm on last layer (official output)
            if hasattr(model.encoder, "norm") and model.encoder.norm is not None:
                h = model.encoder.norm(h)
                features[f"layer_{len(model.encoder.transformer_encoder)}"][-1] = pool_cell_embedding_all(h).cpu().numpy()

    for k in features:
        features[k] = np.concatenate(features[k], axis=0)

    # Save per-layer embeddings
    print("Saving per-layer embeddings...")
    for k, v in features.items():
        save_path = os.path.join(OUTPUT_DIR, f"{k}.pt")
        torch.save(torch.tensor(v), save_path)
        print(f"Saved {k} -> {save_path} | shape = {v.shape}")

    all_labels = np.array(all_labels)

    print("\n=== 5-Fold Linear Probe ===")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = []
    for layer_name in sorted(features.keys(), key=lambda x: int(x.split("_")[1])):
        X = features[layer_name]
        acc_list, f1_list = [], []

        for train_idx, test_idx in skf.split(X, all_labels):
            clf = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)
            clf.fit(X[train_idx], all_labels[train_idx])
            pred = clf.predict(X[test_idx])
            acc_list.append(accuracy_score(all_labels[test_idx], pred))
            f1_list.append(f1_score(all_labels[test_idx], pred, average="macro"))

        results.append({
            "layer": int(layer_name.split("_")[1]),
            "accuracy_mean": float(np.mean(acc_list)),
            "macro_f1_mean": float(np.mean(f1_list)),
        })

        print(f"{layer_name}: acc={np.mean(acc_list):.4f}, f1={np.mean(f1_list):.4f}")

    df = pd.DataFrame(results).sort_values("layer")
    df.to_csv(os.path.join(OUTPUT_DIR, "scFoundation_pbmc3k_layer_embedding.csv"), index=False)


if __name__ == "__main__":
    main()
