# Layer-Wise Evaluation of Single-Cell Foundation Models

This repository provides a code-only evaluation pipeline for layer-wise analyses
of single-cell foundation models (scFMs). It is designed as a software artifact:
datasets, checkpoints, saved activations, result tables, and paper figures are
not redistributed.

Supported evaluation layers:

1. Activation extraction for Geneformer, scGPT, scFoundation, and scBERT.
2. Downstream probing from saved activations for cell-type classification and
   perturbation response prediction.
3. Representation metrics, including effective rank, intrinsic dimension, and
   linear CKA.
4. Reviewer-analysis utilities such as earliest-on-par and leave-one-dataset-out
   layer selection.
5. Dataset provenance and preprocessing configuration.

## Repository Layout

```text
configs/                         Example dataset and pipeline configs
docs/                            Dataset provenance and preprocessing notes
results_schema/                  Expected output-table schemas, no real results
scfm_eval/                       Reusable Python package
  preprocessing/                 Shared AnnData preprocessing helpers
  probing/                       Cell-type and perturbation probes
  representation/                Geometry and CKA metrics
  reviewer_analyses/             Layer-selection utilities
scripts/                         Thin CLI entry points
cell_type_extraction/            Model-faithful extraction scripts from notebooks
perturbation_extraction/         Model-faithful perturbation extraction scripts
representation_evaluation/       Legacy-compatible representation scripts
```

## What Is Not Included

The repo intentionally excludes:

- raw datasets (`.h5ad`, `.loom`, `.h5`, `.mtx`)
- saved activations (`.pt`, `.npy`, `.npz`)
- model checkpoints and weights
- generated result CSVs/tables
- manuscript figures

Users provide local paths through copied config files.

## Quick Start

Install the reusable package and its core dependencies:

```bash
pip install -e .
```

For representation-quality metrics, install the optional analysis dependencies:

```bash
pip install -e ".[analysis]"
```

Model-specific extraction dependencies are intentionally optional because the
four scFMs have different upstream installation requirements. See
`docs/datasets_and_preprocessing.md` before installing the relevant model package.

Copy the example configs if you want a record of local paths and run settings:

```bash
cp configs/dataset_catalog.example.json configs/dataset_catalog.local.json
cp configs/pipeline.example.json configs/pipeline.local.json
```

These JSON files are reference templates and provenance records; the reusable
CLI entry points currently receive their paths and settings explicitly through
command-line arguments.

Run a cell-type probe from saved layer activations:

```bash
python scripts/run_celltype_probe.py \
  --layer-dir /path/to/saved/layer_activations \
  --h5ad /path/to/dataset.h5ad \
  --label-key "Cell type" \
  --min-cells-per-class 15 \
  --output-csv outputs/celltype_per_fold.csv
```

Run a perturbation response probe from saved activations:

```bash
python scripts/run_perturbation_probe.py \
  --layer-dir /path/to/saved/layer_activations \
  --h5ad /path/to/perturbation_dataset.h5ad \
  --perturbation-key perturbation \
  --context-key cell_type \
  --control-label control \
  --expression-layer logNor \
  --output-csv outputs/perturbation_per_fold.csv
```

The perturbation probe first averages cells within each perturbation, then
standardizes the resulting activation features using training-fold
statistics, uses ridge penalty `1e-4`, and reports PCC-delta, MSE-delta,
perturbation retrieval, and collapse diagnostics. Optional gene selection is
performed from training perturbations only; it is disabled by default.

Compute representation quality metrics or CKA:

```bash
python scripts/run_representation_metrics.py \
  --layer-dir /path/to/saved/layer_activations \
  --metric quality \
  --output-csv outputs/representation_quality.csv

python scripts/run_representation_metrics.py \
  --layer-dir /path/to/saved/layer_activations \
  --metric cka \
  --output-csv outputs/layerwise_cka_long.csv
```

Run layer-selection utilities:

```bash
python scripts/run_layer_selection.py earliest-on-par \
  --input-csv outputs/celltype_per_fold.csv \
  --metric macro_f1 \
  --final-layer 12 \
  --margin 0.01
```

## Local Paths And Legacy Extractors

The files under `cell_type_extraction/` and `perturbation_extraction/` preserve
model-faithful, dataset-specific extraction workflows from the original analyses.
They are not intended to download data or checkpoints automatically. Provide
local paths through each script's arguments or documented environment variables;
the JSON configs are reference templates rather than an automatic config loader.
Never commit private filesystem paths, datasets, checkpoints, or generated outputs.

Each extraction directory now has one dispatcher per model and one pipeline per
dataset. For example:

    python cell_type_extraction/extract_scgpt.py --dataset pbmc3k --model-dir /path/to/scGPT_human
    python perturbation_extraction/extract_geneformer_perturbation.py --dataset schmidt

The dispatcher runs only the requested dataset pipeline; it does not execute
the other datasets in the same task.

For the legacy scGPT extractors, copy `.env.example` to `.env.local`, edit the
checkpoint path, and source it locally:

```bash
cp .env.example .env.local
# edit .env.local
source .env.local
```

The reusable probing and representation-analysis CLIs operate on saved activations
and accept all input and output paths explicitly.

## Dataset Provenance And Preprocessing

See `docs/datasets_and_preprocessing.md` and
`configs/dataset_catalog.example.json`. Extraction-time preprocessing details
are encoded both in documentation and in `scfm_eval/preprocessing/common.py`.
The reusable probing CLIs expect saved activations and labels to already reflect
the same extraction-time QC and alignment; they do not silently re-filter genes
or cells after activation extraction.

## Relationship To Historical Scripts

The model-specific extraction folders preserve notebook-faithful scripts from
the original analyses. New reusable code should live under `scfm_eval/`, with
thin CLIs under `scripts/`. Avoid adding generated outputs to the repository.
