"""Layer-wise geneformer pipeline for Zheng68k PBMC.

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

adata = sc.read_h5ad("Zheng68k_pmbc.merged.h5ad")
genes = adata.var_names.tolist()

mg = mygene.MyGeneInfo()
res = mg.querymany(
    genes,
    scopes="symbol",
    fields="ensembl.gene",
    species="human",
    as_dataframe=True
)

def extract_ensembl(x):
    if isinstance(x, list):
        x = x[0]
    if isinstance(x, dict):
        return x.get("gene", np.nan)
    if pd.isna(x):
        return np.nan
    return str(x)

res = pd.DataFrame(res)
res["ensembl_id"] = res["ensembl.gene"].apply(extract_ensembl)

# map back to var_names
res2 = res["ensembl_id"].dropna()
res2_dict = res2.to_dict()  # index is query gene symbol

adata.var["ensembl_id"] = adata.var_names.map(res2_dict)
adata = adata[:, adata.var["ensembl_id"].notna()].copy()

# make ensembl_id a clean string column (required by Geneformer)
adata.var["ensembl_id"] = adata.var["ensembl_id"].astype(str).str.upper()

# set index to ensembl but KEEP column
adata.var.index = adata.var["ensembl_id"]

adata.write_h5ad("geneformer_Zheng68k_ensemblID.h5ad")


input_path = Path("./geneformer_Zheng68k_ensemblID.h5ad")
data_dir = input_path.parent

tk = TranscriptomeTokenizer(
    nproc=4,
    model_version="V2",
    custom_attr_name_dict={"Cell type": "Cell type"}
)

tk.tokenize_data(
    data_directory=str(data_dir),
    output_directory=str(data_dir / "tokenized"),
    output_prefix="Zheng68k",
    file_format="h5ad",
    input_identifier=input_path.stem
)


# ==========================================================
# Geneformer Layer Probing with Linear Classifier
# Mean-pool extraction per layer
# Zheng68k cell type prediction with 5-fold CV
# ==========================================================
import os
from pathlib import Path

# ---- CPU thread limits ----
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
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

SOURCE_H5AD = "Zheng68k_pmbc.merged.h5ad"
RAW_H5AD = "./geneformer_Zheng68k_rawX.h5ad"
DATASET_PATH = "./tokenized/Zheng68k_rawX.dataset"
OUTPUT_DIR = "geneformer_Zheng68k_rawX_layer_probing_results"

CELL_TYPE_COL = "Cell type"
CELL_ID_COL = "cell_id"

MAX_NCELLS = None
BATCH_SIZE = 16
FORCE_TOKENIZE = False
SEED = 42
N_SPLITS = 5
MIN_CELLS_PER_CLASS = N_SPLITS

# ==========================================
# 2. Tokenization helper
# ==========================================
def maybe_tokenize():
    dataset_path = Path(DATASET_PATH)

    if (not FORCE_TOKENIZE) and dataset_path.exists():
        return

    print("Building raw.X h5ad + tokenizing Zheng68k...")

    adata = sc.read_h5ad(SOURCE_H5AD)

    if CELL_TYPE_COL not in adata.obs:
        raise ValueError(
            f"'{CELL_TYPE_COL}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    if adata.raw is None:
        raise ValueError("adata.raw is None — cannot build raw.X dataset")

    adata_raw = sc.AnnData(
        X=adata.raw.X.copy(),
        obs=adata.obs.copy(),
        var=adata.raw.var.copy(),
    )

    # Preserve row identity and labels in the tokenized dataset.
    adata_raw.obs[CELL_ID_COL] = adata_raw.obs_names.astype(str)
    adata_raw.obs[CELL_TYPE_COL] = adata_raw.obs[CELL_TYPE_COL].astype(str)

    # Geneformer tokenizer expects n_counts in obs.
    if "n_counts" not in adata_raw.obs:
        if "total_counts" in adata_raw.obs:
            adata_raw.obs["n_counts"] = adata_raw.obs["total_counts"]
        else:
            x_sum = adata_raw.X.sum(axis=1)
            if hasattr(x_sum, "A1"):
                x_sum = x_sum.A1
            else:
                x_sum = np.asarray(x_sum).ravel()
            adata_raw.obs["n_counts"] = x_sum

    adata_raw.write_h5ad(RAW_H5AD)

    tk = TranscriptomeTokenizer(
        nproc=4,
        model_version="V2",
        custom_attr_name_dict={
            CELL_TYPE_COL: CELL_TYPE_COL,
            CELL_ID_COL: CELL_ID_COL,
        },
    )

    Path(DATASET_PATH).parent.mkdir(parents=True, exist_ok=True)

    tk.tokenize_data(
        data_directory=str(Path(RAW_H5AD).parent),
        output_directory=str(Path(DATASET_PATH).parent),
        output_prefix=Path(DATASET_PATH).stem,
        file_format="h5ad",
        input_identifier=Path(RAW_H5AD).name,
    )

# ==========================================
# 3. Geneformer extraction helpers
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
    model.to(device)

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
# 4. Dataset / label helpers
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
# 5. Main computation
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {MODEL_DIR} on {device}...")

    model = load_model_custom(MODEL_DIR, device)
    tk = TranscriptomeTokenizer(model_version="V2")

    pad_token_id, cls_token_id, eos_token_id = resolve_special_token_ids(model, tk)

    print(
        f"Special token ids: "
        f"pad={pad_token_id}, cls={cls_token_id}, eos/sep={eos_token_id}"
    )

    if cls_token_id is None or eos_token_id is None:
        print(
            "Warning: cls_token_id or eos_token_id could not be resolved. "
            "Pooling may not exclude all special tokens."
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

        acc_list = []
        f1_list = []

        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(
                max_iter=1000,
                n_jobs=4,
                solver="lbfgs",
                multi_class="auto",
            )

            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])

            acc_list.append(accuracy_score(y[test_idx], preds))
            f1_list.append(f1_score(y[test_idx], preds, average="macro"))

        acc_mean = float(np.mean(acc_list))
        f1_mean = float(np.mean(f1_list))

        results.append({
            "layer": layer,
            "accuracy_mean": acc_mean,
            "macro_f1_mean": f1_mean,
        })

        print(
            f"Layer {layer}: "
            f"Accuracy = {acc_mean:.4f} ± {acc_std:.4f}, "
            f"Macro-F1 = {f1_mean:.4f} ± {f1_std:.4f}"
        )

    res_df = pd.DataFrame(results).sort_values("layer")

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "Geneformer_Zheng68k_layer_probing_results.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
