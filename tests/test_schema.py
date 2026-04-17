"""Schema and registry tests."""

from __future__ import annotations

import pytest

from plantaer.registry import MaterialRegistry
from plantaer.schema import (
    Category,
    Experiment,
    InputSpec,
    InputType,
    InputValue,
    MaterialDomain,
)


def test_inputspec_bounds_only_on_numeric():
    with pytest.raises(ValueError):
        InputSpec(
            name="atm",
            category=Category.ENVIRONMENT,
            type=InputType.CATEGORICAL,
            bounds=(0.0, 1.0),
        )


def test_inputspec_target_must_be_numeric():
    with pytest.raises(ValueError):
        InputSpec(name="y", category=Category.TARGET, type=InputType.CATEGORICAL)


def test_material_domain_rejects_duplicate_columns():
    with pytest.raises(ValueError):
        MaterialDomain(
            name="dup",
            inputs=[
                InputSpec(name="x", category=Category.PROCESSING, type=InputType.NUMERIC),
                InputSpec(name="x", category=Category.ENVIRONMENT, type=InputType.NUMERIC),
            ],
            targets=[
                InputSpec(name="y", category=Category.TARGET, type=InputType.NUMERIC)
            ],
        )


def test_experiment_validate_rejects_unknown_column(tabular_domain):
    exp = Experiment(
        id="e0",
        domain="ToyAlloy",
        values={"nonsense": InputValue(value=1.0)},
    )
    with pytest.raises(ValueError):
        exp.validate_against(tabular_domain)


def test_experiment_validate_numeric_bounds(tabular_domain):
    exp = Experiment(
        id="e0",
        domain="ToyAlloy",
        values={"sinter_temp_c": InputValue(value=99999.0)},
    )
    with pytest.raises(ValueError):
        exp.validate_against(tabular_domain)


def test_registry_roundtrip(tmp_path, tabular_domain):
    path = tmp_path / "materials.json"
    reg = MaterialRegistry(path)
    reg.add_material(tabular_domain)
    reg2 = MaterialRegistry(path)
    assert reg2.list_materials() == ["ToyAlloy"]
    assert reg2.get("ToyAlloy") == tabular_domain


def test_all_eight_input_types_accepted():
    specs = [
        InputSpec(name=f"x{i}", category=Category.METADATA, type=t)
        for i, t in enumerate(InputType)
        if t != InputType.NUMERIC  # numeric is Target here, test it separately
    ]
    MaterialDomain(
        name="types",
        inputs=specs,
        targets=[
            InputSpec(name="y", category=Category.TARGET, type=InputType.NUMERIC)
        ],
    )
