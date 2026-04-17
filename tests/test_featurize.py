"""Featurizer + presence-mask tests."""

from __future__ import annotations

import numpy as np

from plantaer.featurize import build_matrices, build_target_matrix, feature_names
from plantaer.featurize.composition import (
    N_ELEMENTS,
    featurize_composition,
    parse_formula,
)
from plantaer.schema import Experiment, InputValue


def test_parse_formula_fractional():
    assert parse_formula("Fe0.7Ni0.3") == {"Fe": 0.7, "Ni": 0.3}


def test_featurize_composition_missing_is_zero():
    vec, mask = featurize_composition(None)
    assert vec.shape == (N_ELEMENTS,)
    assert mask.sum() == 0
    assert vec.sum() == 0


def test_featurize_composition_normalized():
    vec, mask = featurize_composition({"Fe": 2.0, "Ni": 2.0})
    assert mask.sum() == N_ELEMENTS
    assert abs(vec.sum() - 1.0) < 1e-9


def test_build_matrices_shape_is_stable(tabular_domain, small_dataset):
    X, M, layout = build_matrices(tabular_domain, small_dataset)
    assert X.shape == M.shape == (len(small_dataset), layout.dim)
    # Feature layout must have exactly one slice per declared input
    assert set(layout.slices.keys()) == {s.name for s in tabular_domain.inputs}


def test_missing_optional_inputs_zero_imputed(tabular_domain, small_dataset):
    exp = small_dataset[0].model_copy(update={"values": {
        k: v for k, v in small_dataset[0].values.items() if k != "atmosphere"
    }})
    X, M, layout = build_matrices(tabular_domain, [exp])
    sl = layout.slices["atmosphere"]
    assert np.all(X[0, sl] == 0.0)
    assert np.all(M[0, sl] == 0.0)
    # Other blocks still present
    sl_temp = layout.slices["sinter_temp_c"]
    assert M[0, sl_temp].sum() == 1.0


def test_target_matrix(tabular_domain, small_dataset):
    Y, Ym, names = build_target_matrix(tabular_domain, small_dataset)
    assert Y.shape == (len(small_dataset), 1)
    assert names == ["yield_strength"]
    assert Ym.sum() == len(small_dataset)


def test_feature_names_stable_under_duplicate_call(tabular_domain):
    l1 = feature_names(tabular_domain)
    l2 = feature_names(tabular_domain)
    assert l1.names == l2.names
