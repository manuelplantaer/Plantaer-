"""Model zoo with an auto-selector.

Every model implements the tiny ``Estimator`` protocol in ``base``:

    fit(X, Y, mask=None)
    predict_mean_std(X) -> (mean, std)

The selector in ``selector`` routes by (n_rows, feature_dim) to the right
estimator for the current material's data volume.
"""

from .base import Estimator, MeanBaseline
from .gp import SklearnGP
from .gbm import BootstrapGBM
from .selector import select_estimator, fit_for_domain, FittedModel

__all__ = [
    "Estimator",
    "MeanBaseline",
    "SklearnGP",
    "BootstrapGBM",
    "select_estimator",
    "fit_for_domain",
    "FittedModel",
]
