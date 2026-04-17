"""Bootstrap gradient-boosting regressor.

Mid-to-large-data regime default. Trains an ensemble of sklearn
``GradientBoostingRegressor`` models on bootstrap samples; the ensemble mean
is the prediction and the ensemble std is the uncertainty. Works at any N,
handles mixed tabular inputs natively, and has zero extra dependencies.

If ``xgboost`` is installed we transparently use ``XGBRegressor`` instead —
faster and typically more accurate, same interface.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

try:  # pragma: no cover - optional
    from xgboost import XGBRegressor  # type: ignore

    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False


class BootstrapGBM:
    def __init__(
        self,
        n_estimators: int = 8,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_trees: int = 100,
        random_state: int = 0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.random_state = random_state
        self._models: list[list[object]] = []  # per target, list of estimators
        self._fallback_means: np.ndarray | None = None
        self._fallback_stds: np.ndarray | None = None

    def _make_one(self, seed: int):
        if _HAS_XGB:
            return XGBRegressor(
                n_estimators=self.n_trees,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbosity=0,
                tree_method="hist",
            )
        return GradientBoostingRegressor(
            n_estimators=self.n_trees,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            random_state=seed,
        )

    def fit(
        self, X: np.ndarray, Y: np.ndarray, mask: np.ndarray | None = None
    ) -> "BootstrapGBM":
        if Y.ndim == 1:
            Y = Y[:, None]
        if mask is not None and mask.ndim == 1:
            mask = mask[:, None]
        n, n_targets = Y.shape
        rng = np.random.default_rng(self.random_state)

        self._fallback_means = np.zeros(n_targets)
        self._fallback_stds = np.ones(n_targets)
        self._models = []

        for j in range(n_targets):
            keep = np.ones(n, dtype=bool) if mask is None else mask[:, j] > 0.5
            yj = Y[keep, j]
            Xj = X[keep]
            self._fallback_means[j] = float(yj.mean()) if yj.size else 0.0
            self._fallback_stds[j] = float(yj.std() or 1e-6) if yj.size else 1e-6
            if Xj.shape[0] < 2:
                self._models.append([])
                continue

            models = []
            for k in range(self.n_estimators):
                idx = rng.integers(0, Xj.shape[0], size=Xj.shape[0])
                m = self._make_one(seed=int(rng.integers(0, 1 << 31)))
                m.fit(Xj[idx], yj[idx])
                models.append(m)
            self._models.append(models)
        return self

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._fallback_means is not None and self._fallback_stds is not None
        n = X.shape[0]
        d = len(self._models)
        mean = np.zeros((n, d))
        std = np.zeros((n, d))
        for j, models in enumerate(self._models):
            if not models:
                mean[:, j] = self._fallback_means[j]
                std[:, j] = self._fallback_stds[j]
                continue
            preds = np.stack([m.predict(X) for m in models], axis=0)
            mean[:, j] = preds.mean(axis=0)
            std[:, j] = preds.std(axis=0, ddof=0)
        return mean, np.maximum(std, 1e-6)


__all__ = ["BootstrapGBM"]
