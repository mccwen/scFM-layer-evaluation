# Datasets and preprocessing

This artifact intentionally separates **dataset provenance**, **local file paths**, and **model-specific preprocessing**.

## Dataset catalog

Copy:

```bash
cp configs/dataset_catalog.example.json configs/dataset_catalog.local.json
```

Then edit `local_path` fields to point to your `.h5ad` files. The example catalog records the source used in the manuscript, source URL, citation, required `obs` columns, and preprocessing choices.

## Cell-type classification preprocessing

Extraction-time shared preprocessing is implemented in `scfm_eval/preprocessing/common.py`:

1. Read the `.h5ad` file.
2. Verify the cell-type label column from the dataset catalog.
3. Remove rare cell types before stratified CV.
4. Filter cells with fewer than 200 detected genes.
5. Filter genes observed in fewer than 3 cells.
6. Apply model-specific vocabulary mapping/tokenization in the extraction script.
7. Train/evaluate a fixed five-fold linear probe on saved layer activations.

The manuscript used a minimum class size sufficient for valid stratified five-fold CV. Some legacy scripts used `>=5`; the final standardized analyses use `>=15` unless noted in the run config.

## Perturbation-response preprocessing

Shared perturbation preprocessing is implemented in `scfm_eval/preprocessing/common.py`:

1. Read the `.h5ad` file.
2. Identify `perturbation_key`, `control_label`, and optional `context_key` from the dataset catalog.
3. Use a log-normalized expression layer when the dataset provides one, e.g. `logNor`; otherwise use the configured expression matrix.
4. Compute expression deltas relative to matched controls, within context if a context column is present.
5. Split perturbations, not cells, into a single-seed five-fold CV split balanced by perturbation size.
6. Fit standardized ridge probes from layer activations to expression deltas
   using a fixed ridge penalty of `1e-4`.
7. Report PCC-delta, MSE-delta, retrieval metrics, and collapse diagnostics.

The optional top-gene evaluation mode selects genes from training perturbations
within each fold. It never uses held-out target deltas for feature selection.

## Model-specific preprocessing notes

- scGPT: use the checkpoint vocabulary; detect whether the vocabulary uses Ensembl IDs or gene symbols; prepend CLS; use model binning/padding settings from `args.json`.
- scBERT: use scBERT-compatible gene ordering and tokenization; masked mean pooling over non-padding tokens.
- Geneformer: use rank/value representation expected by Geneformer; mean-pool valid gene-token states, excluding padding/special tokens.
- scFoundation: map inputs to the fixed scFoundation gene vocabulary and use the released `pool_type=all` cell representation.

## Files not included

Raw datasets, model checkpoints, and full layer activation tensors are not bundled in the GitHub repo. The repo contains scripts, configs, schemas, and lightweight summary outputs. Large artifacts should be stored externally and referenced from the catalog or release notes.
