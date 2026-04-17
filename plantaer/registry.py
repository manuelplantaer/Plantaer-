"""JSON-backed registry of user-declared materials.

One `MaterialDomain` per registered material. Adding, updating, or extending
a material edits `materials.json` in the workspace; no code change required.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .schema import MaterialDomain


class MaterialRegistry:
    """Persistent registry of material domains."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._materials: dict[str, MaterialDomain] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._materials = {}
            return
        raw = json.loads(self.path.read_text() or "{}")
        self._materials = {
            name: MaterialDomain.model_validate(body) for name, body in raw.items()
        }

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: json.loads(domain.model_dump_json())
            for name, domain in self._materials.items()
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def list_materials(self) -> list[str]:
        with self._lock:
            return sorted(self._materials.keys())

    def get(self, name: str) -> MaterialDomain:
        with self._lock:
            if name not in self._materials:
                raise KeyError(f"unknown material {name!r}")
            return self._materials[name]

    def add_material(self, domain: MaterialDomain, *, overwrite: bool = False) -> None:
        with self._lock:
            if domain.name in self._materials and not overwrite:
                raise ValueError(
                    f"material {domain.name!r} already exists; pass overwrite=True"
                )
            self._materials[domain.name] = domain
            self._persist()

    def delete(self, name: str) -> None:
        with self._lock:
            self._materials.pop(name, None)
            self._persist()


__all__ = ["MaterialRegistry"]
