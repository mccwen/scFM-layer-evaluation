# Cell-Type Extraction Pipelines

Each model has one dispatcher and one dataset-specific pipeline per dataset.
The dispatcher runs exactly one pipeline at a time; it never executes all
datasets sequentially.

Examples:

    python cell_type_extraction/extract_scgpt.py --dataset pbmc3k --model-dir /path/to/scGPT_human
    python cell_type_extraction/extract_geneformer.py --dataset czi
    python cell_type_extraction/extract_scbert.py --dataset zheng68k
    python cell_type_extraction/extract_scfoundation.py --dataset testis

The dataset-specific files preserve the original model-faithful extraction
logic. Supply local datasets, checkpoints, vocabularies, and model repositories
through the configuration expected by each pipeline. Do not commit those
artifacts.
