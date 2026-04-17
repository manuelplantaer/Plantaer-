"""Seed a demo Plantaer workspace with two independent materials.

Run::

    python examples/seed_workspace.py

Then launch the UI against the same workspace::

    PLANTAER_WORKSPACE=./plantaer_workspace streamlit run plantaer/ui/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plantaer.schema import (
    Category,
    Experiment,
    InputSpec,
    InputType,
    InputValue,
    MaterialDomain,
)
from plantaer.workspace import Workspace


def _alloys_domain() -> MaterialDomain:
    return MaterialDomain(
        name="Binary alloys",
        description="Fe/Ni, Cu/Zn, Ti/Al binaries under different sintering conditions.",
        inputs=[
            InputSpec(name="formula", category=Category.COMPOSITION, type=InputType.COMPOSITION),
            InputSpec(
                name="sinter_temp_c",
                category=Category.PROCESSING,
                type=InputType.NUMERIC,
                bounds=(400.0, 1200.0),
                unit="C",
            ),
            InputSpec(
                name="hold_min",
                category=Category.PROCESSING,
                type=InputType.NUMERIC,
                bounds=(1.0, 600.0),
                unit="min",
            ),
            InputSpec(
                name="atmosphere",
                category=Category.ENVIRONMENT,
                type=InputType.CATEGORICAL,
                choices=["air", "argon", "vacuum"],
            ),
        ],
        targets=[
            InputSpec(
                name="yield_strength",
                category=Category.TARGET,
                type=InputType.NUMERIC,
                unit="MPa",
            )
        ],
    )


def _oxides_domain() -> MaterialDomain:
    return MaterialDomain(
        name="Perovskite-ish oxides",
        description="Simple ABO3-like mixtures with a bandgap target.",
        inputs=[
            InputSpec(name="formula", category=Category.COMPOSITION, type=InputType.COMPOSITION),
            InputSpec(
                name="anneal_temp_c",
                category=Category.PROCESSING,
                type=InputType.NUMERIC,
                bounds=(300.0, 1000.0),
                unit="C",
            ),
            InputSpec(
                name="precursor",
                category=Category.COMPONENT,
                type=InputType.CATEGORICAL,
                choices=["nitrate", "acetate", "sol-gel"],
            ),
        ],
        targets=[
            InputSpec(
                name="bandgap_ev",
                category=Category.TARGET,
                type=InputType.NUMERIC,
                unit="eV",
            )
        ],
    )


def _synth_alloys(n: int, seed: int = 0) -> list[Experiment]:
    rng = np.random.default_rng(seed)
    pairs = [("Fe", "Ni"), ("Cu", "Zn"), ("Ti", "Al")]
    out = []
    for i in range(n):
        a = float(rng.uniform(0.1, 0.9))
        A, B = pairs[i % len(pairs)]
        temp = float(rng.uniform(400, 1200))
        hold = float(rng.uniform(1, 600))
        atm = ["air", "argon", "vacuum"][i % 3]
        y = (
            0.7 * (temp / 1000.0)
            + 0.4 * a
            + 0.15 * (1 if atm == "argon" else 0)
            + 0.08 * rng.standard_normal()
        )
        out.append(
            Experiment(
                id=f"alloy_{i:05d}",
                domain="Binary alloys",
                values={
                    "formula": InputValue(value={A: round(a, 3), B: round(1 - a, 3)}),
                    "sinter_temp_c": InputValue(value=temp, unit="C"),
                    "hold_min": InputValue(value=hold, unit="min"),
                    "atmosphere": InputValue(value=atm),
                    "yield_strength": InputValue(value=float(y * 400 + 200), unit="MPa"),
                },
            )
        )
    return out


def _synth_oxides(n: int, seed: int = 1) -> list[Experiment]:
    rng = np.random.default_rng(seed)
    A_sites = ["Sr", "Ba", "Ca"]
    B_sites = ["Ti", "Zr", "Nb"]
    out = []
    for i in range(n):
        a = A_sites[i % len(A_sites)]
        b = B_sites[(i // 3) % len(B_sites)]
        formula = {a: 1.0, b: 1.0, "O": 3.0}
        temp = float(rng.uniform(300, 1000))
        prec = ["nitrate", "acetate", "sol-gel"][i % 3]
        y = (
            2.0
            + 0.5 * (temp / 1000.0)
            + (0.3 if b == "Ti" else 0.8 if b == "Zr" else 1.2)
            + 0.15 * rng.standard_normal()
        )
        out.append(
            Experiment(
                id=f"oxide_{i:05d}",
                domain="Perovskite-ish oxides",
                values={
                    "formula": InputValue(value=formula),
                    "anneal_temp_c": InputValue(value=temp, unit="C"),
                    "precursor": InputValue(value=prec),
                    "bandgap_ev": InputValue(value=float(y), unit="eV"),
                },
            )
        )
    return out


def main() -> None:
    root = Path("./plantaer_workspace").resolve()
    ws = Workspace.open(root)
    ws.registry.add_material(_alloys_domain(), overwrite=True)
    ws.registry.add_material(_oxides_domain(), overwrite=True)
    ws.store.bulk_upsert(_synth_alloys(120))
    ws.store.bulk_upsert(_synth_oxides(80))
    print(f"Seeded workspace at {root}")
    for name in ws.registry.list_materials():
        print(f"  • {name}: {ws.store.count(name)} experiments")


if __name__ == "__main__":
    main()
