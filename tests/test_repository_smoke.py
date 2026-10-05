from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_parse() -> None:
    files = sorted(ROOT.rglob("*.py"))
    assert files
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_repository_has_no_local_artifacts() -> None:
    forbidden_suffixes = {".h5ad", ".pt", ".ckpt", ".npy", ".npz", ".csv"}
    forbidden_fragments = (
        chr(47) + "Users" + chr(47),
        chr(47) + "home" + chr(47),
        chr(47) + "nfs" + chr(47),
        "@" + "umich.edu",
    )
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts or not path.is_file():
            continue
        assert path.suffix not in forbidden_suffixes, path
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert not any(fragment in text for fragment in forbidden_fragments), path
