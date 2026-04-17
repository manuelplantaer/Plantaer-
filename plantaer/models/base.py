"""Estimator interface + trivial mean baseline."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Estimator(Protocol):
    """Multi-target regressor returning mean + std."""

    def fit(self, X: np.ndarray, Y: np.ndarray, mask: np.ndarray | None = None) -> "Estimator":
        ...

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...


class MeanBaseline:
    """Predicts the marginal target mean with target std as uncertainty.

    Used as the sanity floor and as a fallback when training data is empty
    or a learned model fails.
    """

    def __init__(self) -> None:
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, Y: np.ndarray, mask: np.ndarray | None = None
    ) -> "MeanBaseline":
        if Y.ndim == 1:
            Y = Y[:, None]
        if mask is None:
            self._mean = Y.mean(axis=0)
            self._std = Y.std(axis=0, ddof=0)
        else:
            if mask.ndim == 1:
                mask = mask[:, None]
            totals = mask.sum(axis=0)
            totals = np.where(totals == 0, 1.0, totals)
            self._mean = (Y * mask).sum(axis=0) / totals
            var = ((Y - self._mean) ** 2 * mask).sum(axis=0) / totals
            self._std = np.sqrt(var)
        # avoid zero-variance predictions
        self._std = np.where(self._std == 0.0, 1e-6, self._std)
        return self

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._mean is not None and self._std is not None, "fit first"
        n = X.shape[0]
        mean = np.broadcast_to(self._mean, (n, self._mean.shape[0])).copy()
        std = np.broadcast_to(self._std, (n, self._std.shape[0])).copy()
        return mean, std


__all__ = ["Estimator", "MeanBaseline"]
