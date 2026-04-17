"""Math-correctness tests.

These verify the analytical pieces of the pipeline — EI closed form, GP
interpolation, composition normalization, presence-mask invariance — against
either ground truth or Monte-Carlo references.
"""

from __future__ import annotations

import numpy as np
import pytest

from plantaer.design.bo import _ei
from plantaer.design import Objective, suggest_experiments
from plantaer.featurize import build_matrices
from plantaer.featurize.composition import (
    N_ELEMENTS,
    featurize_composition,
    parse_formula,
)
from plantaer.models import (
    BootstrapGBM,
    MeanBaseline,
    SklearnGP,
    cross_validate,
    fit_for_domain,
)
from plantaer.schema import (
    Category,
    Experiment,
    InputSpec,
    InputType,
    InputValue,
    MaterialDomain,
)


# ---------------------------------------------------------------------------
# Expected Improvement — closed form vs Monte-Carlo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mean,std,best,sign",
    [
        (0.5, 0.3, 0.4, 1.0),     # maximize, good candidate
        (0.1, 0.3, 0.4, 1.0),     # maximize, bad candidate (still +EI from tail)
        (0.4, 0.1, 0.4, 1.0),     # at incumbent
        (-1.0, 2.0, 0.0, 1.0),    # high-variance exploration
        (0.9, 0.2, 0.8, -1.0),    # minimize: mean above incumbent = tail-only EI
        (0.5, 0.2, 0.8, -1.0),    # minimize: mean below incumbent = bulk EI
    ],
)
def test_ei_matches_monte_carlo(mean, std, best, sign):
    analytical = float(
        _ei(np.array([mean]), np.array([std]), best=best, sign=sign)[0]
    )
    rng = np.random.default_rng(42)
    samples = rng.normal(loc=mean, scale=std, size=200_000)
    improvement = sign * (samples - best)
    mc = float(np.maximum(improvement, 0.0).mean())
    assert analytical == pytest.approx(mc, abs=0.01, rel=0.02)


def test_ei_is_nonnegative():
    rng = np.random.default_rng(0)
    m = rng.standard_normal(64)
    s = np.abs(rng.standard_normal(64)) + 1e-6
    assert (_ei(m, s, best=0.0, sign=1.0) >= 0).all()
    assert (_ei(m, s, best=0.0, sign=-1.0) >= 0).all()


def test_ei_zero_variance_limit():
    """As sigma -> 0, EI -> max(0, sign*(mean-best))."""
    m = np.array([0.5, 0.3, 0.7])
    s = np.full_like(m, 1e-10)
    for sign in (1.0, -1.0):
        expected = np.maximum(sign * (m - 0.4), 0.0)
        got = _ei(m, s, best=0.4, sign=sign)
        assert np.allclose(got, expected, atol=1e-6)


def test_ei_prefers_higher_mean_when_maximizing():
    # Equal variance, higher mean -> higher EI when maximizing.
    m = np.array([0.1, 0.5, 0.9])
    s = np.full_like(m, 0.3)
    ei = _ei(m, s, best=0.4, sign=1.0)
    assert ei[2] > ei[1] > ei[0]


def test_ei_prefers_lower_mean_when_minimizing():
    m = np.array([0.1, 0.5, 0.9])
    s = np.full_like(m, 0.3)
    ei = _ei(m, s, best=0.4, sign=-1.0)
    assert ei[0] > ei[1] > ei[2]


# ---------------------------------------------------------------------------
# BO incumbent uses the true direction-aware observed extremum
# ---------------------------------------------------------------------------


def _simple_domain() -> MaterialDomain:
    return MaterialDomain(
        name="linear",
        inputs=[
            InputSpec(
                name="x",
                category=Category.PROCESSING,
                type=InputType.NUMERIC,
                bounds=(0.0, 1.0),
            )
        ],
        targets=[
            InputSpec(name="y", category=Category.TARGET, type=InputType.NUMERIC)
        ],
    )


def _linear_experiments(n: int, slope: float = 2.0, seed: int = 0) -> list[Experiment]:
    rng = np.random.default_rng(seed)
    exps = []
    for i in range(n):
        x = float(rng.uniform(0, 1))
        y = slope * x + 0.05 * rng.standard_normal()
        exps.append(
            Experiment(
                id=f"e{i}",
                domain="linear",
                values={"x": InputValue(value=x), "y": InputValue(value=y)},
            )
        )
    return exps


def test_incumbent_direction_correctness():
    dom = _simple_domain()
    exps = _linear_experiments(30)
    fitted = fit_for_domain(dom, exps)
    ys = [e.values["y"].value for e in exps]
    assert fitted.observed_extremum("y", direction="maximize") == pytest.approx(max(ys))
    assert fitted.observed_extremum("y", direction="minimize") == pytest.approx(min(ys))


def test_bo_suggestions_point_towards_optimum():
    """For y = 2x with x in [0,1], maximizing y should push suggestions to x -> 1."""
    dom = _simple_domain()
    fitted = fit_for_domain(dom, _linear_experiments(60))
    suggestions = suggest_experiments(
        fitted, Objective(target="y", direction="maximize"), n_suggestions=5,
        random_state=0,
    )
    xs = np.array([s.values["x"].value for s in suggestions])
    # Average of top-5 should be well above midpoint 0.5.
    assert xs.mean() > 0.7


def test_bo_suggestions_point_away_from_optimum_when_minimizing():
    dom = _simple_domain()
    fitted = fit_for_domain(dom, _linear_experiments(60))
    suggestions = suggest_experiments(
        fitted, Objective(target="y", direction="minimize"), n_suggestions=5,
        random_state=0,
    )
    xs = np.array([s.values["x"].value for s in suggestions])
    assert xs.mean() < 0.3


# ---------------------------------------------------------------------------
# GP interpolation property
# ---------------------------------------------------------------------------


def test_gp_interpolates_training_points_closely():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(25, 2))
    y = (X[:, 0] ** 2 + np.sin(3 * X[:, 1])).reshape(-1, 1)
    gp = SklearnGP().fit(X, y)
    mean, std = gp.predict_mean_std(X)
    # Predictions at training points close to y.
    err = np.abs(mean - y).max()
    assert err < 0.1
    # Uncertainty at training points should be small compared to out-of-sample.
    X_far = rng.uniform(5.0, 10.0, size=(10, 2))
    _, std_far = gp.predict_mean_std(X_far)
    assert std.mean() < std_far.mean()


# ---------------------------------------------------------------------------
# Composition featurizer: correctness and stability
# ---------------------------------------------------------------------------


def test_composition_is_normalized_to_sum_one():
    vec, mask = featurize_composition({"Fe": 3.0, "Ni": 1.0})
    assert vec.sum() == pytest.approx(1.0)
    assert mask.sum() == N_ELEMENTS


def test_composition_dict_and_string_agree():
    v1, _ = featurize_composition("Fe0.5Ni0.5")
    v2, _ = featurize_composition({"Fe": 0.5, "Ni": 0.5})
    assert np.allclose(v1, v2)


def test_composition_element_index_stable_for_iron():
    """Fe is Z=26, so it should live at index 25 (0-based)."""
    vec, _ = featurize_composition("Fe")
    assert vec.argmax() == 25
    assert vec[25] == pytest.approx(1.0)


def test_parse_formula_rejects_unknown_element():
    with pytest.raises(ValueError):
        parse_formula("Xx2")


# ---------------------------------------------------------------------------
# Presence-mask invariance: featurizing a row with dropped modalities leaves
# other slots untouched.
# ---------------------------------------------------------------------------


def test_dropping_modality_leaves_other_slots_bitwise_identical(
    tabular_domain, small_dataset
):
    X_full, _, layout = build_matrices(tabular_domain, small_dataset[:1])
    exp_no_atm = small_dataset[0].model_copy(
        update={"values": {
            k: v for k, v in small_dataset[0].values.items() if k != "atmosphere"
        }}
    )
    X_partial, M_partial, _ = build_matrices(tabular_domain, [exp_no_atm])
    sl_atm = layout.slices["atmosphere"]
    # Atmosphere slot should be zero + mask-zero.
    assert np.all(X_partial[0, sl_atm] == 0.0)
    assert np.all(M_partial[0, sl_atm] == 0.0)
    # Every other slot identical to the full-row version.
    other = np.ones(layout.dim, dtype=bool)
    other[sl_atm] = False
    assert np.array_equal(X_full[0, other], X_partial[0, other])


# ---------------------------------------------------------------------------
# Learning-curve sanity: more data → lower MSE (for BootstrapGBM)
# ---------------------------------------------------------------------------


def _noisy_cubic(x: np.ndarray, rng) -> np.ndarray:
    return (x ** 3 - 0.5 * x + 0.05 * rng.standard_normal(x.shape)).reshape(-1, 1)


def _fit_gbm_on(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 1))
    y = _noisy_cubic(X.ravel(), rng)
    gbm = BootstrapGBM(n_estimators=6, random_state=seed).fit(X, y)
    Xt = np.linspace(-1, 1, 200).reshape(-1, 1)
    yt = (Xt ** 3 - 0.5 * Xt)
    pred, _ = gbm.predict_mean_std(Xt)
    return float(((pred - yt) ** 2).mean())


def test_more_data_lowers_gbm_mse():
    mse_small = _fit_gbm_on(30, seed=0)
    mse_large = _fit_gbm_on(600, seed=0)
    assert mse_large < mse_small


# ---------------------------------------------------------------------------
# Cross-validation report matches the sanity floor (≥ 10% MSE improvement)
# ---------------------------------------------------------------------------


def test_cv_report_improves_on_mean_baseline(tabular_domain, medium_dataset):
    report = cross_validate(tabular_domain, medium_dataset, max_folds=5)
    assert report is not None
    # Per-target improvement ≥ 10%.
    imp = report.improvement_pct("yield_strength")
    assert imp > 10.0
    assert np.isfinite(report.r2_model).all()


def test_cv_returns_none_when_too_few_rows(tabular_domain, small_dataset):
    assert cross_validate(tabular_domain, small_dataset[:3]) is None


# ---------------------------------------------------------------------------
# MeanBaseline predictions are constant and equal to the target mean.
# ---------------------------------------------------------------------------


def test_mean_baseline_is_constant_and_correct():
    Y = np.array([[1.0], [2.0], [3.0], [4.0]])
    X = np.zeros((4, 2))
    mb = MeanBaseline().fit(X, Y)
    mean, std = mb.predict_mean_std(np.zeros((5, 2)))
    assert np.allclose(mean, 2.5)
    # std should equal the sample std (population) of Y
    assert std[0, 0] == pytest.approx(np.std(Y[:, 0]), abs=1e-6)
