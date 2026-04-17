"""Experiment suggestion via Expected Improvement.

We sample a large pool of candidate experiments from the material's declared
input bounds / choices, score each with the fitted model's posterior, and
return the top-k by EI. This is the classic single-objective BO loop,
implemented without gradient-based acquisition refinement so we avoid a
BoTorch dependency for the MVP. Swap the scoring fn for BoTorch later.

Fallback: at tiny N (`fitted.n_rows < N_MIN_LEARN`) we skip EI and return a
random in-bounds design instead — the UI never dead-ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

from ..featurize import build_matrices
from ..models.selector import FittedModel, N_MIN_LEARN
from ..schema import Experiment, InputSpec, InputType, InputValue, MaterialDomain

Direction = Literal["maximize", "minimize"]


@dataclass
class Objective:
    target: str
    direction: Direction = "maximize"

    def sign(self) -> float:
        return 1.0 if self.direction == "maximize" else -1.0


@dataclass
class SuggestedExperiment:
    values: dict[str, InputValue]
    predicted_mean: float
    predicted_std: float
    acquisition: float  # expected improvement


def _proposable(spec: InputSpec) -> bool:
    if spec.type == InputType.NUMERIC:
        return spec.bounds is not None
    if spec.type == InputType.CATEGORICAL:
        return spec.choices is not None
    return False  # composition / structure / spectra — user supplies pool


def _sample_pool(
    domain: MaterialDomain, n: int, rng: np.random.Generator
) -> list[dict[str, InputValue]]:
    pool: list[dict[str, InputValue]] = []
    for _ in range(n):
        row: dict[str, InputValue] = {}
        for spec in domain.inputs:
            if spec.type == InputType.NUMERIC and spec.bounds is not None:
                lo, hi = spec.bounds
                row[spec.name] = InputValue(value=float(rng.uniform(lo, hi)), unit=spec.unit)
            elif spec.type == InputType.CATEGORICAL and spec.choices:
                row[spec.name] = InputValue(value=str(rng.choice(spec.choices)))
            # non-proposable types left unset; featurizer fills zeros + masks.
        pool.append(row)
    return pool


def _ei(mean: np.ndarray, std: np.ndarray, best: float, sign: float) -> np.ndarray:
    """Analytic EI for a scalar objective. ``sign=+1`` maximize, ``-1`` minimize."""
    improvement = sign * (mean - best)
    std = np.maximum(std, 1e-12)
    z = improvement / std
    return improvement * norm.cdf(z) + std * norm.pdf(z)


def suggest_experiments(
    fitted: FittedModel,
    objective: Objective,
    n_suggestions: int = 5,
    pool_size: int = 2048,
    random_state: int = 0,
) -> list[SuggestedExperiment]:
    """Return ``n_suggestions`` top-EI candidate experiments.

    Raises ``ValueError`` if no input in the domain is proposable (i.e. no
    numeric bounds and no categorical choices declared).
    """
    domain = fitted.domain
    rng = np.random.default_rng(random_state)
    if not any(_proposable(s) for s in domain.inputs):
        raise ValueError(
            f"material {domain.name!r} has no proposable inputs — declare numeric "
            "bounds or categorical choices on at least one input"
        )
    if objective.target not in fitted.target_names:
        raise ValueError(
            f"objective target {objective.target!r} not in targets {fitted.target_names}"
        )

    pool = _sample_pool(domain, pool_size, rng)
    fake_exps = [
        Experiment(id=f"__pool_{i}", domain=domain.name, values=v)
        for i, v in enumerate(pool)
    ]
    Xp, _, _ = build_matrices(domain, fake_exps)

    # Tiny-N fallback: return the first `n_suggestions` random candidates.
    if fitted.n_rows < N_MIN_LEARN:
        out: list[SuggestedExperiment] = []
        for i in range(min(n_suggestions, len(pool))):
            out.append(
                SuggestedExperiment(
                    values=pool[i],
                    predicted_mean=float(fitted.y_means[
                        fitted.target_names.index(objective.target)
                    ]),
                    predicted_std=float(fitted.y_stds[
                        fitted.target_names.index(objective.target)
                    ]),
                    acquisition=0.0,
                )
            )
        return out

    mean, std = fitted.predict(Xp)
    tidx = fitted.target_names.index(objective.target)
    m = mean[:, tidx]
    s = std[:, tidx]

    # EI uses the true direction-aware extremum of observed targets as the
    # incumbent — the standard definition. Falls back to the training mean
    # only if no target was ever measured (shouldn't happen post-selector).
    incumbent = fitted.observed_extremum(objective.target, direction=objective.direction)
    if incumbent is None:
        incumbent = float(fitted.y_means[tidx])
    ei = _ei(m, s, best=float(incumbent), sign=objective.sign())

    top = np.argsort(-ei)[:n_suggestions]
    return [
        SuggestedExperiment(
            values=pool[int(i)],
            predicted_mean=float(m[int(i)]),
            predicted_std=float(s[int(i)]),
            acquisition=float(ei[int(i)]),
        )
        for i in top
    ]


__all__ = ["Objective", "SuggestedExperiment", "suggest_experiments"]
