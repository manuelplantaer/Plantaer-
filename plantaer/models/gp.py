"""Gaussian-Process regressor (sklearn), per target.

Small-data regime default (n < ~50). Uses an RBF + WhiteKernel so noise and
length-scale are fit jointly. Per-target independent GPs — simple and honest
for the data volumes where GPs shine.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


class SklearnGP:
    """One GP per target dimension."""

    def __init__(self, n_restarts: int = 2, random_state: int = 0):
        self.n_restarts = n_restarts
        self.random_state = random_state
        self._gps: list[GaussianProcessRegressor] = []
        self._x_scaler: StandardScaler | None = None
        self._y_means: np.ndarray | None = None
        self._y_stds: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, Y: np.ndarray, mask: np.ndarray | None = None
    ) -> "SklearnGP":
        if Y.ndim == 1:
            Y = Y[:, None]
        if mask is not None and mask.ndim == 1:
            mask = mask[:, None]

        self._x_scaler = StandardScaler().fit(X) if X.shape[0] > 1 else StandardScaler().fit(
            np.vstack([X, X])
        )
        Xs = self._x_scaler.transform(X)

        self._y_means = np.zeros(Y.shape[1])
        self._y_stds = np.ones(Y.shape[1])
        self._gps = []

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
        )

        for j in range(Y.shape[1]):
            if mask is None:
                keep = np.ones(Y.shape[0], dtype=bool)
            else:
                keep = mask[:, j] > 0.5
            if keep.sum() < 2:
                # Not enough data for a GP: store mean/std only; predict_mean_std
                # will route through the baseline branch below.
                self._gps.append(None)  # type: ignore[arg-type]
                yy = Y[keep, j] if keep.any() else np.array([0.0])
                self._y_means[j] = float(yy.mean())
                self._y_stds[j] = float(yy.std() or 1e-6)
                continue

            yj = Y[keep, j]
            self._y_means[j] = yj.mean()
            std = yj.std() or 1.0
            self._y_stds[j] = std
            yjn = (yj - self._y_means[j]) / std

            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=self.n_restarts,
                normalize_y=False,
                random_state=self.random_state,
                alpha=1e-8,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(Xs[keep], yjn)
            self._gps.append(gp)
        return self

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._x_scaler is not None, "fit first"
        Xs = self._x_scaler.transform(X)
        n = Xs.shape[0]
        d = len(self._gps)
        mean = np.zeros((n, d))
        std = np.zeros((n, d))
        assert self._y_means is not None and self._y_stds is not None
        for j, gp in enumerate(self._gps):
            if gp is None:
                mean[:, j] = self._y_means[j]
                std[:, j] = self._y_stds[j]
                continue
            m, s = gp.predict(Xs, return_std=True)
            mean[:, j] = m * self._y_stds[j] + self._y_means[j]
            std[:, j] = s * self._y_stds[j]
        return mean, np.maximum(std, 1e-6)


__all__ = ["SklearnGP"]
