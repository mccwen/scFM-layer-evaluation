from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    source_name: str
    source_url: str
    reference: str
    local_path: Path
    obs: dict[str, Any]
    preprocessing: dict[str, Any]


def load_dataset_catalog(path: str | Path) -> dict[str, DatasetSpec]:
    """Load dataset provenance and preprocessing metadata.

    The catalog is intentionally JSON so users can run the pipeline without a
    YAML dependency. Copy ``configs/dataset_catalog.example.json`` and edit the
    local paths to match your machine or cluster.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    specs = {}
    for name, raw in payload["datasets"].items():
        specs[name] = DatasetSpec(
            name=name,
            task=raw["task"],
            source_name=raw["source_name"],
            source_url=raw["source_url"],
            reference=raw["reference"],
            local_path=Path(raw["local_path"]),
            obs=dict(raw.get("obs", {})),
            preprocessing=dict(raw.get("preprocessing", {})),
        )
    return specs


def write_dataset_manifest(catalog_path: str | Path, output_path: str | Path) -> None:
    """Write a flat CSV-style manifest for methods supplements."""
    specs = load_dataset_catalog(catalog_path)
    rows = [
        "dataset,task,source_name,source_url,reference,local_path,label_key,perturbation_key,control_label"
    ]
    for spec in specs.values():
        obs = spec.obs
        rows.append(
            ",".join(
                [
                    spec.name,
                    spec.task,
                    spec.source_name.replace(",", ";"),
                    spec.source_url,
                    spec.reference.replace(",", ";"),
                    str(spec.local_path),
                    str(obs.get("label_key", "")),
                    str(obs.get("perturbation_key", "")),
                    str(obs.get("control_label", "")),
                ]
            )
        )
    Path(output_path).write_text("\n".join(rows) + "\n")
