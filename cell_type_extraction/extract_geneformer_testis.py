"""Layer-wise geneformer pipeline for Tabula Sapiens testis.

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

import sys

import torch

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score

from sklearn.preprocessing import LabelEncoder

from transformers import BertForMaskedLM

from datasets import load_from_disk

import matplotlib.pyplot as plt

from IPython import get_ipython

import torch.multiprocessing as mp

from tqdm.auto import tqdm

# =========================
# Load data
# =========================

adata = sc.read_h5ad(
    "ts_testis_annotated.h5ad"
)
adata.obs.rename(columns={"total_counts": "n_counts"}, inplace=True)

count= adata.obs["cell_type"].value_counts()

valid_types= count[count >= 15].index
adata = adata[adata.obs["cell_type"].isin(valid_types)].copy()

# =========================
# 1. Select correct Ensembl column
# =========================
if "ensembl_id" in adata.var.columns:
    ensembl = adata.var["ensembl_id"]
    print("Using 'ensembl_id'")
elif "ensembl_clean" in adata.var.columns:
    ensembl = adata.var["ensembl_clean"]
    print("Using 'ensembl_clean'")
else:
    raise ValueError("No Ensembl ID column found.")

# =========================
# 2. Clean Ensembl IDs (CRITICAL)
# =========================
adata.var["ensembl_id"] = (
    ensembl.astype(str)
    .str.replace(r"\..*", "", regex=True)  # remove version numbers
    .str.upper()
)

# Remove missing
adata = adata[:, adata.var["ensembl_id"].notna()].copy()

# =========================
# 3. Set as var_names (Geneformer requirement)
# =========================
adata.var_names = adata.var["ensembl_id"]
adata.var_names_make_unique()

# =========================
# 4. Sanity checks
# =========================
print("Final shape:", adata.shape)
print("Unique genes:", len(set(adata.var_names)))
print("Duplicate genes:", adata.var_names.duplicated().sum())

# =========================
# 5. Save
# =========================
adata.write_h5ad("geneformer_ts_testis_ensemblID.h5ad")

print("Saved Geneformer-ready dataset.")


input_path = Path("geneformer_ts_testis_ensemblID.h5ad")  # ✅ FIXED
data_dir = input_path.parent

tk = TranscriptomeTokenizer(
    nproc=4,
    model_version="V2",
    custom_attr_name_dict={"cell_type": "cell_type"}
)

tk.tokenize_data(
    data_directory=str(data_dir),
    output_directory=str(data_dir / "tokenized"),
    output_prefix="ts_testis",
    file_format="h5ad",
    input_identifier=input_path.stem
)


# ============================================================
# Geneformer Layer Probing with Linear Classifier
# Mean-pool extraction per layer
# Testis cell type prediction
# Supports 2 GPUs + 4 CPU cores when run as a script
# ============================================================
import os
from pathlib import Path

# ---- CPU thread limits (4 cores) ----
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import BertForMaskedLM
from geneformer import TranscriptomeTokenizer
from datasets import load_from_disk
import scanpy as sc
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration
# ==========================================
MODEL_DIR = "Geneformer-V2-104M"

RAW_H5AD = "ts_testis_annotated.h5ad"
DATASET_PATH = "./tokenized/ts_testis.dataset"
OUTPUT_DIR = "geneformer_ts_testis_layer_probing_results"

CELL_TYPE_COL = "cell_type"
CELL_ID_COL = "cell_id"

MAX_NCELLS = None
BATCH_SIZE = 4
FORCE_TOKENIZE = False

MIN_CELLS_PER_CLASS = 5
N_SPLITS = 5
SEED = 42

# ==========================================
# 2. Notebook detection
# ==========================================
def in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False

IN_NOTEBOOK = in_notebook()

# ==========================================
# 3. Tokenization helper
# ==========================================
def maybe_tokenize():
    dataset_path = Path(DATASET_PATH)

    if (not FORCE_TOKENIZE) and dataset_path.exists():
        return

    print("Building Geneformer-ready h5ad + tokenizing testis data...")

    adata = sc.read_h5ad(RAW_H5AD)

    if CELL_TYPE_COL not in adata.obs:
        raise ValueError(
            f"'{CELL_TYPE_COL}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # Preserve row identity for future alignment checks.
    adata.obs[CELL_ID_COL] = adata.obs_names.astype(str)

    # Geneformer tokenizer expects n_counts in obs.
    if "n_counts" not in adata.obs:
        if "total_counts" in adata.obs:
            adata.obs["n_counts"] = adata.obs["total_counts"]
        else:
            x_sum = adata.X.sum(axis=1)
            if hasattr(x_sum, "A1"):
                x_sum = x_sum.A1
            else:
                x_sum = np.asarray(x_sum).ravel()
            adata.obs["n_counts"] = x_sum

    tokenize_input_dir = Path(OUTPUT_DIR) / "tokenize_input"
    tokenize_input_dir.mkdir(parents=True, exist_ok=True)

    tmp_h5ad = tokenize_input_dir / "testis_geneformer_input.h5ad"
    adata.write_h5ad(tmp_h5ad)

    tk = TranscriptomeTokenizer(
        nproc=4,
        model_version="V2",
        custom_attr_name_dict={
            CELL_TYPE_COL: CELL_TYPE_COL,
            CELL_ID_COL: CELL_ID_COL,
        },
    )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    tk.tokenize_data(
        data_directory=str(tokenize_input_dir),
        output_directory=str(dataset_path.parent),
        output_prefix=dataset_path.stem,
        file_format="h5ad",
        input_identifier=tmp_h5ad.name,
    )

# ==========================================
# 4. Geneformer extraction helpers
# ==========================================
def get_model_input_size(model):
    if hasattr(model, "config"):
        return model.config.max_position_embeddings
    return 2048

def pad_tensor_list(tensor_list, max_len, pad_token_id, model_input_size):
    padded_list = []
    target_len = min(max_len, model_input_size)

    for t in tensor_list:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.long)

        if len(t) > target_len:
            t = t[:target_len]

        pad_len = target_len - len(t)
        if pad_len > 0:
            t = torch.cat(
                [t, torch.full((pad_len,), pad_token_id, dtype=t.dtype)],
                dim=0,
            )

        padded_list.append(t)

    return torch.stack(padded_list)

def gen_attention_mask(input_ids, pad_token_id=0):
    return (input_ids != pad_token_id).long()

def resolve_special_token_ids(model, tokenizer):
    pad_id = getattr(model.config, "pad_token_id", None)
    cls_id = getattr(model.config, "cls_token_id", None)
    eos_id = getattr(model.config, "sep_token_id", None)

    if pad_id is None and hasattr(tokenizer, "pad_token_id"):
        pad_id = tokenizer.pad_token_id
    if cls_id is None and hasattr(tokenizer, "cls_token_id"):
        cls_id = tokenizer.cls_token_id
    if eos_id is None and hasattr(tokenizer, "sep_token_id"):
        eos_id = tokenizer.sep_token_id

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

def mean_pool_gene_tokens(hidden_state, input_ids, pad_id, cls_id, eos_id):
    # Faithful Geneformer cell embedding: pool gene tokens only.
    # Exclude PAD, CLS, and EOS/SEP tokens.
    mask = input_ids != pad_id

    if cls_id is not None:
        mask = mask & (input_ids != cls_id)

    if eos_id is not None:
        mask = mask & (input_ids != eos_id)

    mask = mask.unsqueeze(-1)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)

    return summed / counts

def load_model_custom(model_directory, device):
    model = BertForMaskedLM.from_pretrained(
        model_directory,
        output_hidden_states=True,
    ).to(device)
    model.eval()
    return model

def get_all_layer_embs(
    model,
    filtered_input_data,
    pad_token_id,
    cls_token_id,
    eos_token_id,
    forward_batch_size,
    output_dir,
    device="cpu",
):
    model.eval()
    model_input_size = get_model_input_size(model)
    total_batch_length = len(filtered_input_data)

    all_layer_embs_list = {}
    print(f"Extracting embeddings from {total_batch_length} cells...")

    for i in tqdm(range(0, total_batch_length, forward_batch_size)):
        max_range = min(i + forward_batch_size, total_batch_length)
        minibatch = filtered_input_data.select(range(i, max_range))

        max_len = max(minibatch["length"])
        input_ids = minibatch["input_ids"]

        input_ids = pad_tensor_list(
            input_ids,
            max_len,
            pad_token_id,
            model_input_size,
        )

        attention_mask = gen_attention_mask(input_ids, pad_token_id).to(device)
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            mean_embs = mean_pool_gene_tokens(
                hidden_state=hidden_state,
                input_ids=input_ids,
                pad_id=pad_token_id,
                cls_id=cls_token_id,
                eos_id=eos_token_id,
            )

            all_layer_embs_list.setdefault(layer_idx, []).append(mean_embs.cpu())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_results = {}
    for layer_idx, embs_list in all_layer_embs_list.items():
        layer_tensor = torch.cat(embs_list, dim=0)
        final_results[layer_idx] = layer_tensor.numpy()
        torch.save(layer_tensor, output_dir / f"layer_{layer_idx}_activations.pt")

    return final_results

# ==========================================
# 5. Dataset / label helpers
# ==========================================
def get_labels_from_dataset(dataset):
    if CELL_TYPE_COL not in dataset.column_names:
        raise ValueError(
            f"Column '{CELL_TYPE_COL}' not found in tokenized dataset. "
            f"Available columns: {dataset.column_names}. "
            f"Retokenize with custom_attr_name_dict={{'{CELL_TYPE_COL}': '{CELL_TYPE_COL}'}}."
        )

    raw_labels = np.array(dataset[CELL_TYPE_COL])
    if pd.isna(raw_labels).any():
        raise ValueError(f"Found NaN labels in dataset column '{CELL_TYPE_COL}'.")

    return raw_labels.astype(str)

def filter_dataset_for_cv(dataset, labels, min_cells_per_class, max_ncells=None, seed=42):
    label_counts = pd.Series(labels).value_counts()
    keep_labels = label_counts[label_counts >= min_cells_per_class].index

    keep_mask = np.isin(labels, keep_labels)
    keep_idx = np.where(keep_mask)[0]

    if max_ncells is not None and len(keep_idx) > max_ncells:
        rng = np.random.default_rng(seed)
        keep_idx = np.sort(rng.choice(keep_idx, size=max_ncells, replace=False))

    dataset = dataset.select(keep_idx.tolist())
    labels = labels[keep_idx]

    return dataset, labels

def load_filtered_dataset(dataset_path):
    dataset = load_from_disk(dataset_path)
    labels = get_labels_from_dataset(dataset)

    dataset, labels = filter_dataset_for_cv(
        dataset=dataset,
        labels=labels,
        min_cells_per_class=MIN_CELLS_PER_CLASS,
        max_ncells=MAX_NCELLS,
        seed=SEED,
    )

    return dataset, labels

# ==========================================
# 6. Multi-GPU helpers
# ==========================================
def merge_rank_outputs(output_dir, world_size):
    merged = {}

    for rank in range(world_size):
        rank_dir = Path(output_dir) / f"rank{rank}"

        for p in sorted(rank_dir.glob("layer_*_activations.pt")):
            merged.setdefault(p.name, []).append(torch.load(p, map_location="cpu"))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, parts in merged.items():
        layer_tensor = torch.cat(parts, dim=0)
        torch.save(layer_tensor, out_dir / name)

def worker_multi_gpu(rank, model_dir, dataset_path, output_dir, batch_size):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    dataset, _ = load_filtered_dataset(dataset_path)

    world_size = torch.cuda.device_count()
    shard = dataset.shard(num_shards=world_size, index=rank, contiguous=True)

    model = BertForMaskedLM.from_pretrained(
        model_dir,
        output_hidden_states=True,
    ).to(device)
    model.eval()

    tk = TranscriptomeTokenizer(model_version="V2")
    pad_token_id, cls_token_id, eos_token_id = resolve_special_token_ids(model, tk)

    print(
        f"[rank {rank}] Special token ids: "
        f"pad={pad_token_id}, cls={cls_token_id}, eos/sep={eos_token_id}"
    )

    get_all_layer_embs(
        model=model,
        filtered_input_data=shard,
        pad_token_id=pad_token_id,
        cls_token_id=cls_token_id,
        eos_token_id=eos_token_id,
        forward_batch_size=batch_size,
        output_dir=f"{output_dir}/rank{rank}",
        device=device,
    )

def run_multi_gpu_embedding_extraction(dataset_path, model_dir, output_dir, batch_size):
    import torch.multiprocessing as mp

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")

    mp.spawn(
        worker_multi_gpu,
        args=(model_dir, dataset_path, output_dir, batch_size),
        nprocs=world_size,
        join=True,
    )

def load_merged_layer_embeddings(output_dir):
    layer_embeddings = {}

    for p in Path(output_dir).glob("layer_*_activations.pt"):
        layer_idx = int(p.name.split("_")[1])
        layer_embeddings[layer_idx] = torch.load(p, map_location="cpu").numpy()

    return layer_embeddings

# ==========================================
# 7. Main
# ==========================================
def main():
    maybe_tokenize()

    print(f"Loading dataset from {DATASET_PATH}...")
    dataset, labels = load_filtered_dataset(DATASET_PATH)

    print(
        f"After filtering rare classes: "
        f"{len(labels)} cells, {len(np.unique(labels))} classes"
    )

    if len(np.unique(labels)) < 2:
        raise ValueError("Need at least two classes for cell type prediction.")

    # ==========================================
    # Extract faithful Geneformer layer embeddings
    # ==========================================
    if torch.cuda.device_count() > 1 and not IN_NOTEBOOK:
        run_multi_gpu_embedding_extraction(
            dataset_path=DATASET_PATH,
            model_dir=MODEL_DIR,
            output_dir=OUTPUT_DIR,
            batch_size=BATCH_SIZE,
        )

        merge_rank_outputs(OUTPUT_DIR, torch.cuda.device_count())
        print("Merged multi-GPU outputs.")

        layer_embeddings = load_merged_layer_embeddings(OUTPUT_DIR)

    else:
        if torch.cuda.device_count() > 1 and IN_NOTEBOOK:
            print("NOTE: Multi-GPU disabled in notebook. Run as a script for multi-GPU.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {MODEL_DIR} on {device}...")

        model = load_model_custom(MODEL_DIR, device)
        tk = TranscriptomeTokenizer(model_version="V2")
        pad_token_id, cls_token_id, eos_token_id = resolve_special_token_ids(model, tk)

        print(
            f"Special token ids: "
            f"pad={pad_token_id}, cls={cls_token_id}, eos/sep={eos_token_id}"
        )

        layer_embeddings = get_all_layer_embs(
            model=model,
            filtered_input_data=dataset,
            pad_token_id=pad_token_id,
            cls_token_id=cls_token_id,
            eos_token_id=eos_token_id,
            forward_batch_size=BATCH_SIZE,
            output_dir=OUTPUT_DIR,
            device=device,
        )

    # Sanity check
    for layer_idx, X in layer_embeddings.items():
        if X.shape[0] != len(labels):
            raise ValueError(
                f"Layer {layer_idx} has {X.shape[0]} embeddings, "
                f"but there are {len(labels)} labels."
            )

    # ==========================================
    # Linear probing with 5-fold CV
    # ==========================================
    print("\nStarting Linear Probing with 5-Fold Cross Validation...")

    le = LabelEncoder()
    y = le.fit_transform(labels)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    results = []

    for layer in sorted(layer_embeddings.keys()):
        X = layer_embeddings[layer]

        fold_accs = []
        fold_f1s = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = LogisticRegression(
                max_iter=1000,
                n_jobs=4,
                solver="lbfgs",
                multi_class="auto",
            )
            clf.fit(X_train, y_train)

            preds = clf.predict(X_test)
            fold_accs.append(accuracy_score(y_test, preds))
            fold_f1s.append(f1_score(y_test, preds, average="macro"))

        avg_acc = float(np.mean(fold_accs))
        avg_f1 = float(np.mean(fold_f1s))

        print(
            f"Layer {layer}: "
            f"Mean Accuracy = {avg_acc:.4f}, "
            f"Mean Macro-F1 = {avg_f1:.4f}"
        )

        results.append({
            "layer": layer,
            "accuracy_mean": avg_acc,
            "macro_f1_mean": avg_f1,
        })

    # ==========================================
    # Save results
    # ==========================================
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    res_df = pd.DataFrame(results).sort_values("layer")
    csv_path = out_dir / "Geneformer_V2_104M_layer_ts_testis_metrics.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
