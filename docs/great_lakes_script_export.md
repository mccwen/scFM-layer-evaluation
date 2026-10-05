# Great Lakes Script Export

The public repository should contain code and documentation, not paper data or
manuscript result artifacts. If rebuilding the repo from cluster working folders,
export only scripts and configuration-like text files.

Example source folders:

- `$MODEL_ROOT/Geneformer`
- `$MODEL_ROOT/scFoundation/scFoundation/model`
- `$MODEL_ROOT/scBERT`
- `$MODEL_ROOT/scgpt`

Recommended exclusions for public import:

- raw datasets: `.h5ad`, `.loom`, `.mtx`, `.h5`, `.hdf5`
- activations and arrays: `.pt`, `.pth`, `.npy`, `.npz`, `.pkl`
- model checkpoints: `.ckpt`, `.bin`, `.safetensors`
- generated results: `.csv`, `.tsv`, `.parquet`, manuscript figures
- environment copies: `scgpt_env/`, `__pycache__/`, `.ipynb_checkpoints/`, `.git/`, `build/`

Historical notebooks can be kept outside the public repository as provenance.
When logic is needed, port it into `scfm_eval/` modules or thin scripts under
`scripts/` with configurable paths.

## Portable cluster configuration

Set `MODEL_ROOT` to a site-specific directory in a shell or batch script; the
public repository does not assume a particular cluster filesystem. For the
legacy scGPT extractors, set `SCGPT_MODEL_DIR` to the local checkpoint directory.
