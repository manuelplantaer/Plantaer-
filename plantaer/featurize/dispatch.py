"""Featurizer dispatcher.

Walks a ``MaterialDomain`` in declaration order, asks each ``InputSpec``-
appropriate featurizer for a block of features, and stacks everything into
an ``(n_experiments, n_features)`` matrix plus a same-shape presence mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..schema import Experiment, InputSpec, InputType, MaterialDomain
from .composition import composition_feature_names, featurize_composition
from .processing import (
    categorical_feature_names,
    featurize_categorical,
    featurize_numeric,
    numeric_feature_names,
)


@dataclass(frozen=True)
class FeatureLayout:
    """Stable index over the feature dimension for a given domain."""

    names: tuple[str, ...]
    # Per-input slices into the flat feature vector.
    slices: dict[str, slice]

    @property
    def dim(self) -> int:
        return len(self.names)


# Types we know how to featurize in v1. Unknown types contribute one zero
# feature (still registered in the layout) so dimensionality stays stable.
_SUPPORTED = {
    InputType.NUMERIC,
    InputType.CATEGORICAL,
    InputType.COMPOSITION,
}


def _names_for(spec: InputSpec) -> list[str]:
    if spec.type == InputType.NUMERIC:
        return numeric_feature_names(spec)
    if spec.type == InputType.CATEGORICAL:
        return categorical_feature_names(spec)
    if spec.type == InputType.COMPOSITION:
        return composition_feature_names(spec.name)
    return [f"{spec.name}::_unfeaturized"]


def _transform_one(spec: InputSpec, value) -> tuple[np.ndarray, np.ndarray]:
    if spec.type == InputType.NUMERIC:
        return featurize_numeric(value, spec)
    if spec.type == InputType.CATEGORICAL:
        return featurize_categorical(value, spec)
    if spec.type == InputType.COMPOSITION:
        return featurize_composition(value)
    # Unfeaturized types still contribute one slot so the matrix stays stable.
    return np.zeros(1), np.zeros(1)


def feature_names(domain: MaterialDomain) -> FeatureLayout:
    """Compute the stable feature layout for a material's inputs only."""
    all_names: list[str] = []
    slices: dict[str, slice] = {}
    cursor = 0
    for spec in domain.inputs:
        block = _names_for(spec)
        slices[spec.name] = slice(cursor, cursor + len(block))
        all_names.extend(block)
        cursor += len(block)
    return FeatureLayout(names=tuple(all_names), slices=slices)


def build_matrices(
    domain: MaterialDomain,
    experiments: Sequence[Experiment] | Iterable[Experiment],
) -> tuple[np.ndarray, np.ndarray, FeatureLayout]:
    """Return (X, mask, layout).

    ``X`` and ``mask`` have shape ``(n, layout.dim)``. Rows with missing
    optional inputs are preserved (zero-imputed) so the caller can handle
    scale-invariance via the mask.
    """
    exps = list(experiments)
    layout = feature_names(domain)
    n = len(exps)
    X = np.zeros((n, layout.dim), dtype=np.float64)
    M = np.zeros((n, layout.dim), dtype=np.float64)
    for i, exp in enumerate(exps):
        for spec in domain.inputs:
            iv = exp.values.get(spec.name)
            value = iv.value if iv is not None else None
            vec, mask = _transform_one(spec, value)
            sl = layout.slices[spec.name]
            X[i, sl] = vec
            M[i, sl] = mask
    return X, M, layout


def build_target_matrix(
    domain: MaterialDomain,
    experiments: Sequence[Experiment] | Iterable[Experiment],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (Y, mask, target_names). Missing targets masked to 0."""
    exps = list(experiments)
    target_names = [t.name for t in domain.targets]
    n = len(exps)
    d = len(target_names)
    Y = np.zeros((n, d), dtype=np.float64)
    M = np.zeros((n, d), dtype=np.float64)
    for i, exp in enumerate(exps):
        for j, name in enumerate(target_names):
            iv = exp.values.get(name)
            if iv is None:
                continue
            try:
                Y[i, j] = float(iv.value)
                M[i, j] = 1.0
            except (TypeError, ValueError):
                continue
    return Y, M, target_names


__all__ = [
    "FeatureLayout",
    "feature_names",
    "build_matrices",
    "build_target_matrix",
]
