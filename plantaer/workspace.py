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
from .schema import InputSpec, MaterialDomain
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

    # -----------------------------------------------------------------
    # Rename coordinators: update the registry AND any affected rows in
    # the experiment store in a single logical operation. The user is the
    # source of truth for what things are called; code names follow.
    # -----------------------------------------------------------------

    def rename_material(self, old_name: str, new_name: str) -> None:
        """Rename a material and all experiments pointing to it."""
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("new material name must be non-empty")
        if old_name == new_name:
            return
        if new_name in self.registry.list_materials():
            raise ValueError(f"material {new_name!r} already exists")
        domain = self.registry.get(old_name)
        updated = domain.model_copy(update={"name": new_name})
        self.registry.add_material(updated)  # inserts under new name
        self.registry.delete(old_name)
        self.store.rename_domain(old_name, new_name)

    def rename_column(
        self, material_name: str, old_col: str, new_col: str
    ) -> None:
        """Rename an input/target column in the schema + every stored row."""
        new_col = new_col.strip()
        if not new_col:
            raise ValueError("new column name must be non-empty")
        if old_col == new_col:
            return
        domain = self.registry.get(material_name)
        existing = {s.name for s in domain.all_specs()}
        if new_col in existing:
            raise ValueError(f"{new_col!r} already exists on {material_name!r}")
        if old_col not in existing:
            raise KeyError(f"no column {old_col!r} on {material_name!r}")

        def _rename_in(specs: list[InputSpec]) -> list[InputSpec]:
            out = []
            for s in specs:
                if s.name == old_col:
                    out.append(s.model_copy(update={"name": new_col}))
                else:
                    out.append(s)
            return out

        updated = MaterialDomain(
            name=domain.name,
            description=domain.description,
            inputs=_rename_in(domain.inputs),
            targets=_rename_in(domain.targets),
        )
        self.registry.add_material(updated, overwrite=True)

        # Migrate stored experiment rows: remap values[old_col] -> values[new_col].
        for exp in list(self.store.iter_experiments(material_name)):
            if old_col not in exp.values and old_col not in exp.artifacts:
                continue
            new_values = {
                (new_col if k == old_col else k): v for k, v in exp.values.items()
            }
            new_artifacts = {
                (new_col if k == old_col else k): v
                for k, v in exp.artifacts.items()
            }
            migrated = exp.model_copy(
                update={"values": new_values, "artifacts": new_artifacts}
            )
            self.store.upsert(migrated)

    def rename_experiment(
        self, material_name: str, old_id: str, new_id: str
    ) -> None:
        new_id = new_id.strip()
        if not new_id:
            raise ValueError("new experiment id must be non-empty")
        if old_id == new_id:
            return
        try:
            self.store.get(material_name, new_id)
            raise ValueError(f"experiment id {new_id!r} already exists")
        except KeyError:
            pass
        self.store.rename_id(material_name, old_id, new_id)


__all__ = ["Workspace"]
