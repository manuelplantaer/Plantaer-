"""Numeric and categorical featurizers (processing, environment, geometry)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..schema import InputSpec


def numeric_feature_names(spec: InputSpec) -> list[str]:
    return [f"{spec.name}::value"]


def featurize_numeric(value: Any | None, spec: InputSpec) -> tuple[np.ndarray, np.ndarray]:
    if value is None:
        return np.zeros(1), np.zeros(1)
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{spec.name}: not numeric ({value!r})") from exc
    return np.array([v]), np.array([1.0])


def categorical_feature_names(spec: InputSpec) -> list[str]:
    if spec.choices is None:
        # Fall back to a single "hash-bucket-free" feature: a constant 0; the
        # selector will ignore it. We still emit one feature so the layout is
        # stable.
        return [f"{spec.name}::_unknown_choices"]
    return [f"{spec.name}::is::{c}" for c in spec.choices]


def featurize_categorical(
    value: Any | None, spec: InputSpec
) -> tuple[np.ndarray, np.ndarray]:
    names = categorical_feature_names(spec)
    d = len(names)
    if value is None or spec.choices is None:
        return np.zeros(d), np.zeros(d)
    if value not in spec.choices:
        raise ValueError(
            f"{spec.name}: value {value!r} not in declared choices {spec.choices}"
        )
    vec = np.zeros(d, dtype=np.float64)
    vec[spec.choices.index(value)] = 1.0
    return vec, np.ones(d, dtype=np.float64)


__all__ = [
    "numeric_feature_names",
    "featurize_numeric",
    "categorical_feature_names",
    "featurize_categorical",
]
