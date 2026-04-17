"""Featurization pipeline.

Every input is converted to a fixed-length feature vector plus a presence
mask. Missing optional inputs fill the vector with 0 and the mask with 0 —
no row is rejected for having fewer modalities than another.

Entry point: ``build_matrices(domain, experiments)``.
"""

from .dispatch import (
    build_matrices,
    build_target_matrix,
    feature_names,
    FeatureLayout,
)

__all__ = [
    "build_matrices",
    "build_target_matrix",
    "feature_names",
    "FeatureLayout",
]
