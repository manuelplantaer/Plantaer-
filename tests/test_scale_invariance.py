"""Cross-scale invariance: any N, any D, any missing-modality mix must fit and predict."""

from __future__ import annotations

import numpy as np
import pytest

from plantaer.design import Objective, suggest_experiments
from plantaer.featurize import build_matrices
from plantaer.models.selector import fit_for_domain
from plantaer.schema import (
    Category,
    Experiment,
    InputSpec,
    InputType,
    InputValue,
    MaterialDomain,
)
from tests.conftest import synth_experiments


@pytest.mark.parametrize("n", [1, 7, 40, 300])
def test_any_n_fits_and_predicts(tabular_domain, n):
    fitted = fit_for_domain(
        tabular_domain, synth_experiments(tabular_domain, n=n, seed=n)
    )
    X, _, _ = build_matrices(
        tabular_domain, synth_experiments(tabular_domain, n=3, seed=99)
    )
    mean, std = fitted.predict(X)
    assert mean.shape == (3, 1)
    assert np.isfinite(mean).all() and np.isfinite(std).all()


@pytest.mark.parametrize("extra_numeric_cols", [0, 20, 100])
def test_any_d_fits(extra_numeric_cols):
    """Arbitrarily many declared numeric inputs still fit without error."""
    inputs = [
        InputSpec(
            name=f"x{i}",
            category=Category.PROCESSING,
            type=InputType.NUMERIC,
            bounds=(0.0, 1.0),
        )
        for i in range(extra_numeric_cols)
    ]
    inputs.append(
        InputSpec(
            name="driver",
            category=Category.PROCESSING,
            type=InputType.NUMERIC,
            bounds=(0.0, 1.0),
        )
    )
    dom = MaterialDomain(
        name="wide",
        inputs=inputs,
        targets=[
            InputSpec(name="y", category=Category.TARGET, type=InputType.NUMERIC)
        ],
    )
    rng = np.random.default_rng(0)
    exps = []
    for i in range(40):
        row = {s.name: InputValue(value=float(rng.uniform(0, 1))) for s in inputs}
        row["y"] = InputValue(value=float(row["driver"].value + 0.01 * rng.standard_normal()))
        exps.append(Experiment(id=f"e{i}", domain="wide", values=row))
    fitted = fit_for_domain(dom, exps)
    probe_X, _, _ = build_matrices(dom, exps[:3])
    m, s = fitted.predict(probe_X)
    assert m.shape == (3, 1) and np.isfinite(m).all()


def test_missing_modalities_do_not_crash(tabular_domain, small_dataset):
    # Drop composition from half the rows, atmosphere from the other half.
    patched = []
    for i, e in enumerate(small_dataset):
        vals = dict(e.values)
        if i % 2 == 0:
            vals.pop("formula", None)
        else:
            vals.pop("atmosphere", None)
        patched.append(e.model_copy(update={"values": vals}))
    fitted = fit_for_domain(tabular_domain, patched)
    # BO still works too.
    suggestions = suggest_experiments(
        fitted, Objective(target="yield_strength"), n_suggestions=2
    )
    assert len(suggestions) == 2


def test_empty_dataset_still_predicts_and_suggests(tabular_domain):
    fitted = fit_for_domain(tabular_domain, [])
    X, _, _ = build_matrices(
        tabular_domain, synth_experiments(tabular_domain, n=1, seed=0)
    )
    mean, std = fitted.predict(X)
    assert mean.shape == (1, 1) and np.isfinite(mean).all()
    # Falls back to random in-bounds sampling
    suggestions = suggest_experiments(
        fitted, Objective(target="yield_strength"), n_suggestions=4
    )
    assert len(suggestions) == 4


def test_model_beats_mean_baseline_on_medium_data(tabular_domain, medium_dataset):
    """Sanity floor: learned model's LOO-style residuals beat predicting the mean."""
    train = medium_dataset[: int(0.8 * len(medium_dataset))]
    test = medium_dataset[int(0.8 * len(medium_dataset)) :]
    fitted = fit_for_domain(tabular_domain, train)
    X_test, _, _ = build_matrices(tabular_domain, test)
    pred, _ = fitted.predict(X_test)
    y_true = np.array([e.values["yield_strength"].value for e in test])
    mean_only = np.full_like(y_true, float(np.mean(
        [e.values["yield_strength"].value for e in train]
    )))
    mse_model = ((y_true - pred[:, 0]) ** 2).mean()
    mse_mean = ((y_true - mean_only) ** 2).mean()
    assert mse_model < mse_mean * 0.9  # ≥10% improvement over the mean predictor
