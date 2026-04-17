"""Leave-one-out and k-fold cross-validation for a FittedModel.

We refit the same auto-selected estimator on folds of the same domain and
report per-target MSE, R^2, and the mean-predictor baseline to make model
quality legible in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

from ..featurize import build_matrices, build_target_matrix
from ..schema import Experiment, MaterialDomain
from .selector import MeanBaseline, select_estimator


@dataclass
class CVReport:
    target_names: list[str]
    # Per target.
    mse_model: np.ndarray
    mse_baseline: np.ndarray
    r2_model: np.ndarray
    n_folds: int
    n_rows: int

    def improvement_pct(self, target: str) -> float:
        j = self.target_names.index(target)
        b = float(self.mse_baseline[j])
        if b == 0:
            return 0.0
        return 100.0 * (1.0 - float(self.mse_model[j]) / b)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def cross_validate(
    domain: MaterialDomain,
    experiments: list[Experiment],
    *,
    max_folds: int = 5,
    loo_threshold: int = 25,
) -> CVReport | None:
    """Return a ``CVReport`` or ``None`` if there aren't enough rows.

    ``n < 5`` -> None. ``5 <= n <= loo_threshold`` -> LOO. Otherwise k-fold.
    """
    n = len(experiments)
    if n < 5:
        return None

    X, _, _ = build_matrices(domain, experiments)
    Y, Ym, target_names = build_target_matrix(domain, experiments)

    if n <= loo_threshold:
        splitter = LeaveOneOut()
        splits = list(splitter.split(X))
    else:
        k = min(max_folds, n)
        splitter = KFold(n_splits=k, shuffle=True, random_state=0)
        splits = list(splitter.split(X))

    d_targets = Y.shape[1]
    pred = np.full_like(Y, np.nan)
    pred_baseline = np.full_like(Y, np.nan)
    n_folds = len(splits)

    for train_idx, test_idx in splits:
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        est = select_estimator(n=len(train_idx), d=X.shape[1])
        est.fit(X[train_idx], Y[train_idx], mask=Ym[train_idx])
        m, _ = est.predict_mean_std(X[test_idx])
        pred[test_idx] = m
        base = MeanBaseline().fit(
            X[train_idx], Y[train_idx], mask=Ym[train_idx]
        )
        bm, _ = base.predict_mean_std(X[test_idx])
        pred_baseline[test_idx] = bm

    mse_model = np.zeros(d_targets)
    mse_base = np.zeros(d_targets)
    r2 = np.zeros(d_targets)
    for j in range(d_targets):
        keep = (Ym[:, j] > 0.5) & np.isfinite(pred[:, j])
        if keep.sum() < 2:
            mse_model[j] = np.nan
            mse_base[j] = np.nan
            r2[j] = np.nan
            continue
        y_true = Y[keep, j]
        y_pred = pred[keep, j]
        y_base = pred_baseline[keep, j]
        mse_model[j] = float(((y_true - y_pred) ** 2).mean())
        mse_base[j] = float(((y_true - y_base) ** 2).mean())
        r2[j] = _r2(y_true, y_pred)

    return CVReport(
        target_names=target_names,
        mse_model=mse_model,
        mse_baseline=mse_base,
        r2_model=r2,
        n_folds=n_folds,
        n_rows=n,
    )


__all__ = ["CVReport", "cross_validate"]
