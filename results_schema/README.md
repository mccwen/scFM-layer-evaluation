# Output Schemas

This repository does not ship result tables or figures. The schemas below define
what the pipeline writes when users run it locally.

## Cell-Type Probing Per-Fold CSV

Produced by `scripts/run_celltype_probe.py`.

Required columns:

- `layer`: integer layer index.
- `fold`: 1-based stratified CV fold.
- `accuracy`: fold accuracy.
- `macro_f1`: fold macro-F1.
- `n_train`: number of training cells.
- `n_test`: number of held-out cells.

## Perturbation Probing Per-Fold CSV

Produced by `scripts/run_perturbation_probe.py`.

Required columns:

- `layer`: integer layer index.
- `fold`: 1-based perturbation-level CV fold.
- `heldout_perturbations`: semicolon-separated perturbation names in the test fold.
- `mse_delta`: mean squared error on expression deltas.
- `pcc_delta`: Pearson correlation on expression deltas.
- `n_train`: number of training cells, including controls.
- `n_test`: number of held-out non-control cells.
- `control_label`: control condition label used to compute deltas.
- `alpha`: ridge penalty used for the fold.
- `standardize_features`: whether training-fold activation scaling was used.
- `retrieval_top1`: top-1 perturbation retrieval accuracy within the held-out fold.
- `retrieval_mrr`: mean reciprocal rank within the held-out fold.
- `predicted_observed_variance_ratio`: predicted/observed perturbation-level variance ratio.
- `predicted_observed_pairwise_distance_ratio`: predicted/observed mean Euclidean distance ratio between held-out perturbation delta means.
- `unique_top1_fraction`: fraction of unique observed perturbations assigned by top-1 retrieval.
- `collapse_warning`: conservative collapse flag from variance, pairwise-distance, and retrieval diagnostics.

## Representation Metrics CSV

Produced by `scripts/run_representation_metrics.py --metric quality`.

Required columns:

- `layer`
- `n_cells`
- `n_features`
- `matrix_entropy_normalized`
- `effective_rank`
- `intrinsic_dim_2nn`
- `median_centered_norm`
- `median_feature_std`

## CKA Long CSV

Produced by `scripts/run_representation_metrics.py --metric cka`.

Required columns:

- `layer_a`
- `layer_b`
- `linear_cka`

## Layer Selection CSV

Produced by `scripts/run_layer_selection.py lodo`.

Required columns:

- `model`
- `heldout_dataset`
- `metric`
- `selected_layer`
- `final_layer`
- `heldout_best_layer`
- `selected_value`
- `final_value`
- `best_value`
