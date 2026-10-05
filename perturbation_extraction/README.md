# Perturbation Extraction Pipelines

Each model has one dispatcher and one dataset-specific pipeline per perturbation
dataset. The dispatcher runs exactly one pipeline at a time.

Examples:

    python perturbation_extraction/extract_scgpt_perturbation.py --dataset schmidt --model-dir /path/to/scGPT_human
    python perturbation_extraction/extract_geneformer_perturbation.py --dataset wessels
    python perturbation_extraction/extract_scbert_perturbation.py --dataset sciplex
    python perturbation_extraction/extract_scfoundation_perturbation.py --dataset schmidt

The dataset-specific files preserve the original model-faithful extraction
logic. Supply local datasets, checkpoints, vocabularies, and model repositories
through the configuration expected by each pipeline. Do not commit those
artifacts.
