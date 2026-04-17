"""Bayesian-optimization suggestion tests."""

from __future__ import annotations

import pytest

from plantaer.design import Objective, suggest_experiments
from plantaer.models.selector import fit_for_domain
from plantaer.schema import Category, InputSpec, InputType, MaterialDomain


def test_suggestions_respect_bounds_and_choices(tabular_domain, medium_dataset):
    fitted = fit_for_domain(tabular_domain, medium_dataset)
    suggestions = suggest_experiments(
        fitted, Objective(target="yield_strength", direction="maximize"), n_suggestions=5
    )
    assert len(suggestions) == 5
    for s in suggestions:
        t = s.values["sinter_temp_c"].value
        assert 400.0 <= t <= 1200.0
        assert s.values["atmosphere"].value in ("air", "argon", "vacuum")


def test_suggestions_tiny_n_falls_back_to_random(tabular_domain, small_dataset):
    # fit on 3 rows — below N_MIN_LEARN
    fitted = fit_for_domain(tabular_domain, small_dataset[:3])
    suggestions = suggest_experiments(
        fitted, Objective(target="yield_strength"), n_suggestions=3
    )
    assert len(suggestions) == 3
    # Fallback reports 0 acquisition (no EI available)
    assert all(s.acquisition == 0.0 for s in suggestions)


def test_domain_with_no_proposable_inputs_raises():
    dom = MaterialDomain(
        name="nope",
        inputs=[
            InputSpec(name="f", category=Category.COMPOSITION, type=InputType.COMPOSITION)
        ],
        targets=[
            InputSpec(name="y", category=Category.TARGET, type=InputType.NUMERIC)
        ],
    )
    fitted = fit_for_domain(dom, [])
    with pytest.raises(ValueError):
        suggest_experiments(fitted, Objective(target="y"))
