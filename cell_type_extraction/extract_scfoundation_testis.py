"""Layer-wise scfoundation pipeline for Tabula Sapiens testis.

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


# ==========================================
# 0. Configuration
# ==========================================
BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "models")
CKPT_PATH = os.path.join(MODEL_DIR, "models.ckpt")
GENE_INDEX_PATH = os.path.join(BASE_DIR, "OS_scRNA_gene_index.19264.tsv")

CELL_TYPE_COLUMN = "cell_type"
MIN_GENES_PER_CELL = 200
MIN_CELLS_PER_GENE = 3
MIN_CELLS_PER_CLASS = 15

BATCH_SIZE = 1
N_SPLITS = 5
SEED = 42

DATA_PATH = "ts_testis_annotated.h5ad"
OUTPUT_DIR = os.path.join(BASE_DIR, "scFoundation_ts_testis_layer_embeddings")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 1. Setup & Imports
# ==========================================
sys.path.append(BASE_DIR)
from load import load_model_frommmf, gatherData  # noqa: E402


# ==========================================
# 2. Gene Alignment Helpers
# ==========================================
def align_to_vocab(x_df, gene_list):
    to_fill_columns = list(set(gene_list) - set(x_df.columns))
    padding_df = pd.DataFrame(
        np.zeros((x_df.shape[0], len(to_fill_columns)), dtype=np.float32),
        columns=to_fill_columns,
        index=x_df.index,
    )
    x_df = pd.concat([x_df, padding_df], axis=1)
    return x_df[gene_list]


def get_raw_counts(adata):
    if adata.raw is not None:
        return adata.raw.X, adata.raw.var
    if "counts" in adata.layers:
        return adata.layers["counts"], adata.var
    return adata.X, adata.var


def is_ensembl(tokens):
    sample = tokens[:1000] if len(tokens) > 1000 else tokens
    return sum(t.startswith("ENSG") for t in sample) > 0.5 * len(sample)


def choose_best_gene_ids(var, gene_list):
    gene_list = [str(g) for g in gene_list]
    gene_list_upper = [g.upper() for g in gene_list]
    gene_list_strip = [g.split(".")[0] for g in gene_list_upper]

    gene_list_is_ensembl = is_ensembl(gene_list_upper)
    gene_set = set(gene_list_strip if gene_list_is_ensembl else gene_list_upper)

    candidates = []

    def add_candidate(name, series):
        if series is None:
            return
        values = pd.Series(series).astype(str)
        if gene_list_is_ensembl:
            proc = values.str.upper().str.split(".").str[0]
        else:
            proc = values.str.upper()
        overlap = proc.isin(gene_set).sum()
        candidates.append((name, proc.values, overlap))

    for col in ["gene_name", "Gene", "gene_symbol", "gene_id", "gene_ids", "feature_name"]:
        if col in var.columns:
            add_candidate(col, var[col])

    add_candidate("var_names", var.index)

    if not candidates:
        raise RuntimeError(f"No gene identifier columns found in var. Columns: {list(var.columns)}")

    best = max(candidates, key=lambda x: x[2])
    if best[2] == 0:
        raise RuntimeError(
            "No overlap with scFoundation vocab. "
            f"Candidates: {[(c[0], c[2]) for c in candidates]}"
        )

    print(f"Best gene ID source: {best[0]} (matched {best[2]} genes)")
    return best[1]


# ==========================================
# 3. Data Loader
# ==========================================
def prepare_dataloader(gene_index_path, batch_size=BATCH_SIZE):
    print(f"Loading dataset from {DATA_PATH} ...")
    adata = sc.read_h5ad(DATA_PATH)
    adata.obs_names_make_unique()
    original_obs_names = adata.obs_names.copy()

    counts, var = get_raw_counts(adata)
    if scipy.sparse.issparse(counts):
        counts = counts.toarray()

    gene_list = (
        pd.read_csv(gene_index_path, sep="\t")["gene_name"].astype(str).str.upper().tolist()
    )
    gene_symbols = choose_best_gene_ids(var, gene_list)

    adata_tmp = sc.AnnData(
        X=counts,
        var=pd.DataFrame(index=gene_symbols),
        obs=adata.obs.copy(),
    )
    adata_tmp.obs_names_make_unique()

    sc.pp.filter_cells(adata_tmp, min_genes=MIN_GENES_PER_CELL)
    sc.pp.filter_genes(adata_tmp, min_cells=MIN_CELLS_PER_GENE)

    ct_counts = adata_tmp.obs[CELL_TYPE_COLUMN].astype(str).value_counts()
    keep_ct = ct_counts[ct_counts >= MIN_CELLS_PER_CLASS].index
    adata_tmp = adata_tmp[adata_tmp.obs[CELL_TYPE_COLUMN].astype(str).isin(keep_ct)].copy()

    counts_filtered = np.asarray(adata_tmp.X.copy(), dtype=np.float32)
    gene_symbols_filtered = adata_tmp.var_names.astype(str)

    filtered_obs_names = adata_tmp.obs_names.copy()
    keep_idx_after_class_filter = original_obs_names.get_indexer(filtered_obs_names)
    if np.any(keep_idx_after_class_filter < 0):
        raise ValueError("Could not map filtered obs_names back to original h5ad rows.")

    labels_raw = adata_tmp.obs[CELL_TYPE_COLUMN].astype(str).values
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels_raw)

    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    lognor = np.asarray(adata_tmp.X, dtype=np.float32)

    counts_df = pd.DataFrame(counts_filtered, columns=gene_symbols_filtered, index=filtered_obs_names)
    lognor_df = pd.DataFrame(lognor, columns=gene_symbols_filtered, index=filtered_obs_names)

    counts_df = counts_df.T.groupby(level=0).sum().T
    lognor_df = lognor_df.T.groupby(level=0).sum().T

    counts_df = align_to_vocab(counts_df, gene_list)
    lognor_df = align_to_vocab(lognor_df, gene_list)

    norm_data = lognor_df.values.astype(np.float32)
    counts_data = counts_df.values.astype(np.float32)

    # Keep only cells with at least one matched nonzero gene after vocab alignment.
    valid_gene_mask = (counts_data > 0).sum(axis=1) > 0
    norm_data = norm_data[valid_gene_mask]
    counts_data = counts_data[valid_gene_mask]
    labels = labels[valid_gene_mask]
    labels_raw = labels_raw[valid_gene_mask]
    kept_obs_names = filtered_obs_names[valid_gene_mask]
    keep_idx = keep_idx_after_class_filter[valid_gene_mask]

    if norm_data.shape[0] == 0:
        raise RuntimeError("All cells filtered out after vocab alignment. Check gene mapping.")

    row_sums = counts_data.sum(axis=1, keepdims=True)

    batches = []
    for i in range(0, len(norm_data), batch_size):
        batch_expr = norm_data[i : i + batch_size]
        batch_counts = counts_data[i : i + batch_size]
        batch_sums = row_sums[i : i + batch_size]

        mask_genes = batch_counts > 0
        mask_meta = np.ones((batch_expr.shape[0], 2), dtype=bool)
        value_mask = np.hstack([mask_genes, mask_meta])

        meta = np.hstack(
            [
                np.full((len(batch_expr), 1), 4.0, dtype=np.float32),
                np.log10(batch_sums + 1e-8).astype(np.float32),
            ]
        )
        batch_full = np.hstack([batch_expr, meta]).astype(np.float32)

        batches.append(
            {
                "raw_x": torch.tensor(batch_full).float(),
                "mask": torch.tensor(value_mask),
                "labels": labels[i : i + batch_size],
            }
        )

    label_counts = pd.Series(labels_raw).value_counts().sort_values(ascending=False)

    return {
        "batches": batches,
        "labels": labels,
        "labels_raw": labels_raw,
        "label_encoder": encoder,
        "keep_idx": keep_idx.astype(np.int64),
        "kept_obs_names": kept_obs_names.to_numpy(),
        "label_counts": label_counts,
    }


# ==========================================
# 4. scFoundation Pooling
# ==========================================
def pool_cell_embedding_all(geneemb):
    geneemb1 = geneemb[:, -1, :]
    geneemb2 = geneemb[:, -2, :]
    gene_tokens = geneemb[:, :-2, :]
    geneemb3 = torch.max(gene_tokens, dim=1).values
    geneemb4 = torch.mean(gene_tokens, dim=1)
    return torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)


# ==========================================
# 5. Main
# ==========================================
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, config = load_model_frommmf(CKPT_PATH, key="cell")
    model = model.to(device).eval()

    data_info = prepare_dataloader(GENE_INDEX_PATH)
    batches = data_info["batches"]
    all_labels = np.asarray(data_info["labels"])

    np.save(os.path.join(OUTPUT_DIR, "keep_idx.npy"), data_info["keep_idx"])
    np.save(os.path.join(OUTPUT_DIR, "keep_obs_names.npy"), data_info["kept_obs_names"])
    np.save(os.path.join(OUTPUT_DIR, "label_classes.npy"), data_info["label_encoder"].classes_)
    data_info["label_counts"].rename_axis(CELL_TYPE_COLUMN).reset_index(name="count").to_csv(
        os.path.join(OUTPUT_DIR, "scFoundation_5fold_plot_filtered15.csv"),
        index=False,
    )

    pad_token_id = config["pad_token_id"]
    features = {}

    print("Extracting features...")
    with torch.inference_mode():
        for batch in tqdm(batches):
            raw_x = batch["raw_x"].to(device)
            batch_mask = batch["mask"].to(device)

            x, x_pad = gatherData(raw_x, batch_mask, pad_token_id)
            x_pad = x_pad.bool()

            x_emb = model.token_emb(torch.unsqueeze(x, 2).float(), output_weight=0)

            gene_ids = torch.arange(raw_x.shape[1], device=device).unsqueeze(0)
            gene_ids = gene_ids.repeat(raw_x.shape[0], 1)
            pos_ids, _ = gatherData(gene_ids, batch_mask, pad_token_id)
            p_emb = model.pos_emb(pos_ids)

            x_input = x_emb + p_emb

            features.setdefault("layer_0", []).append(pool_cell_embedding_all(x_input).cpu().numpy())

            h = x_input
            for i, layer in enumerate(model.encoder.transformer_encoder):
                h = layer(h, src_key_padding_mask=x_pad)
                pooled = pool_cell_embedding_all(h).cpu().numpy()
                features.setdefault(f"layer_{i + 1}", []).append(pooled)

            if hasattr(model.encoder, "norm") and model.encoder.norm is not None:
                h = model.encoder.norm(h)
                features[f"layer_{len(model.encoder.transformer_encoder)}"][-1] = (
                    pool_cell_embedding_all(h).cpu().numpy()
                )

    for key in features:
        features[key] = np.concatenate(features[key], axis=0)

    print("Saving per-layer embeddings...")
    for key, value in features.items():
        save_path = os.path.join(OUTPUT_DIR, f"{key}.pt")
        torch.save(torch.tensor(value), save_path)
        print(f"Saved {key} -> {save_path} | shape = {value.shape}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = []
    for layer_name in sorted(features.keys(), key=lambda x: int(x.split("_")[1])):
        x = features[layer_name]
        acc_list, f1_list = [], []

        for train_idx, test_idx in skf.split(x, all_labels):
            clf = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=SEED)
            clf.fit(x[train_idx], all_labels[train_idx])
            pred = clf.predict(x[test_idx])
            acc_list.append(accuracy_score(all_labels[test_idx], pred))
            f1_list.append(f1_score(all_labels[test_idx], pred, average="macro"))

        acc_mean = float(np.mean(acc_list))
        f1_mean = float(np.mean(f1_list))

        results.append(
            {
                "layer": int(layer_name.split("_")[1]),
                "accuracy_mean": acc_mean,
                "macro_f1_mean": f1_mean,
                "n_folds_used": N_SPLITS,
                "n_cells": int(x.shape[0]),
                "n_classes": int(len(np.unique(all_labels))),
                "min_cells_per_class": MIN_CELLS_PER_CLASS,
            }
        )
        print(f"{layer_name}: acc={acc_mean:.4f}±{acc_std:.4f}, f1={f1_mean:.4f}±{f1_std:.4f}")

    df = pd.DataFrame(results).sort_values("layer")
    csv_path = os.path.join(OUTPUT_DIR, "scFoundation_ts_testis_layer_embedding.csv")
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
