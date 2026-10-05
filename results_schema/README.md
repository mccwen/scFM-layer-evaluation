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
