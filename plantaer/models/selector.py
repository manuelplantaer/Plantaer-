"""Model selector + per-material fitted-model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..featurize import (
    FeatureLayout,
    build_matrices,
    build_target_matrix,
    feature_names,
)
from ..schema import Experiment, MaterialDomain
from .base import Estimator, MeanBaseline
from .gbm import BootstrapGBM
from .gp import SklearnGP


# Minimum rows below which we refuse to fit a learned model and fall back to
# the MeanBaseline (UI will also offer range-based random suggestion).
N_MIN_LEARN = 5
# Cutoff between GP and GBM.
N_GP_MAX = 50
# Cutoff where "too high-dim for a GP" kicks in (d > GP_HIGH_D_FACTOR * n).
GP_HIGH_D_FACTOR = 5


def select_estimator(n: int, d: int) -> Estimator:
    """Pick the right estimator for the current (n_rows, n_features)."""
    if n < N_MIN_LEARN:
        return MeanBaseline()
    if n < N_GP_MAX and d <= GP_HIGH_D_FACTOR * n:
        return SklearnGP()
    return BootstrapGBM()


@dataclass
class FittedModel:
    domain: MaterialDomain
    estimator: Estimator
    layout: FeatureLayout
    target_names: list[str]
    # Training stats for downstream BO / UI display.
    n_rows: int
    y_means: np.ndarray
    y_stds: np.ndarray
    # Observed target values per row (shape (n_rows, n_targets)) with presence
    # mask — used by BO to compute the true direction-aware incumbent.
    Y_obs: np.ndarray
    Y_mask: np.ndarray

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.estimator.predict_mean_std(X)

    def observed_extremum(self, target: str, *, direction: str) -> float | None:
        """Return direction-aware best observed value for a target, or None.

        Ignores rows where the target was not measured.
        """
        if target not in self.target_names:
            raise KeyError(target)
        j = self.target_names.index(target)
        if self.Y_mask.shape[0] == 0:
            return None
        keep = self.Y_mask[:, j] > 0.5
        if not keep.any():
            return None
        vals = self.Y_obs[keep, j]
        return float(vals.max()) if direction == "maximize" else float(vals.min())


def fit_for_domain(
    domain: MaterialDomain, experiments: Iterable[Experiment]
) -> FittedModel:
    exps = list(experiments)
    X, _Xm, layout = build_matrices(domain, exps)
    Y, Ym, target_names = build_target_matrix(domain, exps)
    if not target_names:
        raise ValueError(f"material {domain.name!r} has no declared targets")
    if X.shape[0] == 0:
        # Empty dataset: use a MeanBaseline fit on zeros so predict still works.
        est = MeanBaseline().fit(np.zeros((1, layout.dim)), np.zeros((1, len(target_names))))
        return FittedModel(
            domain=domain,
            estimator=est,
            layout=layout,
            target_names=target_names,
            n_rows=0,
            y_means=np.zeros(len(target_names)),
            y_stds=np.ones(len(target_names)),
            Y_obs=np.zeros((0, len(target_names))),
            Y_mask=np.zeros((0, len(target_names))),
        )
    est = select_estimator(n=X.shape[0], d=layout.dim)
    est.fit(X, Y, mask=Ym)
    # Masked means/stds for downstream BO noise floor.
    tot = Ym.sum(axis=0)
    tot = np.where(tot == 0, 1.0, tot)
    y_means = (Y * Ym).sum(axis=0) / tot
    y_stds = np.sqrt(((Y - y_means) ** 2 * Ym).sum(axis=0) / tot)
    y_stds = np.where(y_stds == 0.0, 1e-6, y_stds)
    return FittedModel(
        domain=domain,
        estimator=est,
        layout=layout,
        target_names=target_names,
        n_rows=X.shape[0],
        y_means=y_means,
        y_stds=y_stds,
        Y_obs=Y,
        Y_mask=Ym,
    )


__all__ = [
    "N_MIN_LEARN",
    "N_GP_MAX",
    "GP_HIGH_D_FACTOR",
    "select_estimator",
    "FittedModel",
    "fit_for_domain",
]
