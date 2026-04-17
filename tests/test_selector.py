"""Model-selector routing tests."""

from __future__ import annotations

from plantaer.models import BootstrapGBM, MeanBaseline, SklearnGP
from plantaer.models.selector import (
    N_GP_MAX,
    N_MIN_LEARN,
    fit_for_domain,
    select_estimator,
)


def test_selector_tiny_n_returns_mean_baseline():
    est = select_estimator(n=N_MIN_LEARN - 1, d=5)
    assert isinstance(est, MeanBaseline)


def test_selector_small_n_returns_gp():
    est = select_estimator(n=N_MIN_LEARN + 1, d=5)
    assert isinstance(est, SklearnGP)


def test_selector_large_n_returns_gbm():
    est = select_estimator(n=N_GP_MAX + 1, d=10)
    assert isinstance(est, BootstrapGBM)


def test_selector_high_dim_routes_to_gbm_even_at_small_n():
    # d > 5*n should skip the GP for stability.
    est = select_estimator(n=10, d=10 * 5 + 1)
    assert isinstance(est, BootstrapGBM)


def test_fit_for_domain_small(tabular_domain, small_dataset):
    fitted = fit_for_domain(tabular_domain, small_dataset)
    # tabular_domain emits d=123 (118 composition + 5 other); at n=12 the
    # selector correctly routes to BootstrapGBM because d > 5*n.
    assert isinstance(fitted.estimator, BootstrapGBM)
    assert fitted.n_rows == len(small_dataset)
    from plantaer.featurize import build_matrices

    X, _, _ = build_matrices(tabular_domain, small_dataset[:3])
    mean, std = fitted.predict(X)
    assert mean.shape == (3, 1)
    assert (std > 0).all()


def test_fit_small_lowdim_routes_to_gp():
    """When d is modest relative to n, small-n really does pick the GP."""
    from plantaer.schema import (
        Category,
        Experiment,
        InputSpec,
        InputType,
        InputValue,
        MaterialDomain,
    )

    dom = MaterialDomain(
        name="lowdim",
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
    exps = []
    for i in range(12):
        x = i / 11.0
        exps.append(
            Experiment(
                id=f"e{i}",
                domain="lowdim",
                values={
                    "x": InputValue(value=x),
                    "y": InputValue(value=2 * x + 0.1),
                },
            )
        )
    fitted = fit_for_domain(dom, exps)
    assert isinstance(fitted.estimator, SklearnGP)


def test_fit_for_domain_medium(tabular_domain, medium_dataset):
    fitted = fit_for_domain(tabular_domain, medium_dataset)
    assert isinstance(fitted.estimator, BootstrapGBM)
    assert fitted.n_rows == len(medium_dataset)


def test_fit_for_domain_zero_rows(tabular_domain):
    fitted = fit_for_domain(tabular_domain, [])
    assert isinstance(fitted.estimator, MeanBaseline)
    assert fitted.n_rows == 0
