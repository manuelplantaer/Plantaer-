"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from plantaer.schema import (
    Category,
    Experiment,
    InputSpec,
    InputType,
    InputValue,
    MaterialDomain,
)


@pytest.fixture
def tabular_domain() -> MaterialDomain:
    """Toy material with 1 composition, 2 numeric, 1 categorical input, 1 target."""
    return MaterialDomain(
        name="ToyAlloy",
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


def synth_experiments(domain: MaterialDomain, n: int, seed: int = 0) -> list[Experiment]:
    """Generate n synthetic experiments under ``tabular_domain``'s schema.

    Deterministic function of inputs so learners have signal to fit.
    """
    rng = np.random.default_rng(seed)
    out: list[Experiment] = []
    elements = [["Fe", "Ni"], ["Cu", "Zn"], ["Ti", "Al"]]
    for i in range(n):
        pair = elements[i % len(elements)]
        a = float(rng.uniform(0.2, 0.8))
        formula = {pair[0]: round(a, 2), pair[1]: round(1 - a, 2)}
        temp = float(rng.uniform(400, 1200))
        hold = float(rng.uniform(1, 600))
        atm = ["air", "argon", "vacuum"][i % 3]
        # Target = f(inputs) + noise so learners have signal.
        y = (
            0.5 * temp / 1000.0
            + 0.3 * (a - 0.5)
            + 0.1 * (1 if atm == "argon" else 0)
            + 0.05 * rng.standard_normal()
        )
        out.append(
            Experiment(
                id=f"e{i:05d}",
                domain=domain.name,
                values={
                    "formula": InputValue(value=formula),
                    "sinter_temp_c": InputValue(value=temp, unit="C"),
                    "hold_min": InputValue(value=hold, unit="min"),
                    "atmosphere": InputValue(value=atm),
                    "yield_strength": InputValue(value=float(y), unit="MPa"),
                },
            )
        )
    return out


@pytest.fixture
def small_dataset(tabular_domain):
    return synth_experiments(tabular_domain, n=12, seed=1)


@pytest.fixture
def medium_dataset(tabular_domain):
    return synth_experiments(tabular_domain, n=200, seed=2)
