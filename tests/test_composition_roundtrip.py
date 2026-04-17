"""Regression: compositions survive the DataFrame round-trip.

``st.data_editor`` stringifies dict cells via ``str(dict)``, producing
``"{'Fe': 0.6}"``-style garbage that ``parse_formula`` can't decode. We
render compositions as chemical formulas at the DataFrame boundary AND
self-heal Python-dict-repr strings in ``as_element_dict``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plantaer.featurize.composition import (
    as_element_dict,
    dict_to_formula,
    featurize_composition,
    parse_formula,
)
from plantaer.schema import (
    Category,
    Experiment,
    InputSpec,
    InputType,
    InputValue,
    MaterialDomain,
)
from plantaer.ui.tabular import experiments_to_frame, frame_to_experiments


@pytest.fixture
def comp_domain() -> MaterialDomain:
    return MaterialDomain(
        name="Alloy",
        inputs=[
            InputSpec(
                name="formula", category=Category.COMPOSITION, type=InputType.COMPOSITION
            )
        ],
        targets=[
            InputSpec(name="y", category=Category.TARGET, type=InputType.NUMERIC)
        ],
    )


# ---------------------------------------------------------------------------
# dict_to_formula canonicalization
# ---------------------------------------------------------------------------


def test_dict_to_formula_fractional():
    assert dict_to_formula({"Fe": 0.7, "Ni": 0.3}) == "Fe0.7Ni0.3"


def test_dict_to_formula_integer():
    assert dict_to_formula({"Fe": 2, "O": 3}) == "Fe2O3"
    assert dict_to_formula({"Si": 1, "O": 2}) == "SiO2"


def test_dict_to_formula_rejects_unknown_element():
    with pytest.raises(ValueError):
        dict_to_formula({"Xx": 1.0})


def test_dict_to_formula_parse_round_trip():
    original = {"Fe": 0.7, "Ni": 0.3}
    formula = dict_to_formula(original)
    parsed = parse_formula(formula)
    assert parsed == original


# ---------------------------------------------------------------------------
# as_element_dict self-heals Python dict-repr strings
# ---------------------------------------------------------------------------


def test_as_element_dict_accepts_python_repr_string():
    # What st.data_editor produces when it str()-ifies a dict cell.
    s = "{'Fe': 0.61, 'Ni': 0.39}"
    result = as_element_dict(s)
    assert result == {"Fe": 0.61, "Ni": 0.39}


def test_as_element_dict_rejects_garbled_string():
    with pytest.raises(ValueError):
        as_element_dict("{garbage}")


def test_featurize_composition_accepts_python_repr_string():
    """End-to-end: the featurizer no longer chokes on stringified dicts."""
    vec, mask = featurize_composition("{'Fe': 0.5, 'Ni': 0.5}")
    assert mask.sum() > 0
    assert vec.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DataFrame round-trip: dict input → formula string column → parseable value
# ---------------------------------------------------------------------------


def test_experiments_to_frame_renders_composition_as_formula_string(comp_domain):
    exp = Experiment(
        id="e0",
        domain="Alloy",
        values={
            "formula": InputValue(value={"Fe": 0.61, "Ni": 0.39}),
            "y": InputValue(value=1.0),
        },
    )
    frame = experiments_to_frame(comp_domain, [exp])
    cell = frame["formula"].iloc[0]
    assert isinstance(cell, str)
    assert cell == "Fe0.61Ni0.39"


def test_full_roundtrip_survives_stringification(comp_domain):
    """Even if a pipeline stringifies the dict, featurize still works."""
    exp = Experiment(
        id="e0",
        domain="Alloy",
        values={
            "formula": InputValue(value={"Fe": 0.6, "Ni": 0.4}),
            "y": InputValue(value=2.0),
        },
    )
    frame = experiments_to_frame(comp_domain, [exp])
    # Simulate Streamlit's str() corruption on a later hypothetical code path.
    frame.loc[0, "formula"] = str({"Fe": 0.6, "Ni": 0.4})
    exps = frame_to_experiments(comp_domain, frame)
    assert len(exps) == 1
    from plantaer.featurize import build_matrices
    X, M, _ = build_matrices(comp_domain, exps)
    # No crash, non-zero features, mask set.
    assert X[0].sum() == pytest.approx(1.0)
    assert M[0].sum() > 0


def test_cross_validate_on_composition_seeded_workspace(tmp_path):
    """The Streamlit-Cloud boot path would crash here before the fix."""
    from plantaer.workspace import Workspace
    from plantaer.models import cross_validate
    from examples.seed_workspace import _alloys_domain, _synth_alloys

    ws = Workspace.open(tmp_path / "wk")
    ws.registry.add_material(_alloys_domain())
    ws.store.bulk_upsert(_synth_alloys(30))
    exps = list(ws.store.iter_experiments("Binary alloys"))
    # Simulate the editor round-trip that poisoned rows previously.
    frame = experiments_to_frame(_alloys_domain(), exps)
    assert isinstance(frame["formula"].iloc[0], str)
    roundtripped = frame_to_experiments(_alloys_domain(), frame)
    for exp in roundtripped:
        ws.store.upsert(exp)
    # Now cross_validate — this used to raise ValueError on parse_formula.
    report = cross_validate(_alloys_domain(), list(ws.store.iter_experiments("Binary alloys")))
    assert report is not None
    assert np.isfinite(report.r2_model).all()
