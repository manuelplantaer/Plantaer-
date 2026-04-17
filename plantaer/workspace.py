"""Workspace = ExperimentStore + MaterialRegistry sharing a directory.

A workspace is a single directory containing:
- ``experiments.sqlite``
- ``materials.json``
- ``blobs/``

One workspace per company/project. The Streamlit app operates on a workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .registry import MaterialRegistry
from .store import ExperimentStore


@dataclass
class Workspace:
    root: Path
    store: ExperimentStore
    registry: MaterialRegistry

    @classmethod
    def open(cls, root: str | Path) -> "Workspace":
        root_p = Path(root)
        root_p.mkdir(parents=True, exist_ok=True)
        store = ExperimentStore(root_p / "experiments.sqlite", root_p / "blobs")
        registry = MaterialRegistry(root_p / "materials.json")
        return cls(root=root_p, store=store, registry=registry)


__all__ = ["Workspace"]
