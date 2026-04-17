"""Experiment <-> DataFrame conversion for the per-material spreadsheet."""

from __future__ import annotations

import uuid
from typing import Iterable

import pandas as pd

from ..schema import CATEGORY_ORDER, Experiment, InputType, InputValue, MaterialDomain


def experiments_to_frame(
    domain: MaterialDomain, experiments: Iterable[Experiment]
) -> pd.DataFrame:
    """Produce one row per experiment; columns ordered by category then declaration."""
    specs = _ordered_specs(domain)
    cols = ["id"] + [s.name for s in specs]
    rows: list[dict] = []
    for exp in experiments:
        row: dict = {"id": exp.id}
        for s in specs:
            iv = exp.values.get(s.name)
            row[s.name] = None if iv is None else iv.value
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def frame_to_experiments(
    domain: MaterialDomain, frame: pd.DataFrame
) -> list[Experiment]:
    """Inverse of ``experiments_to_frame``; tolerates NaN as missing."""
    specs = _ordered_specs(domain)
    by_name = {s.name: s for s in specs}
    out: list[Experiment] = []
    for _, row in frame.iterrows():
        raw_id = row.get("id")
        exp_id = (
            str(raw_id)
            if isinstance(raw_id, str) and raw_id
            else f"exp_{uuid.uuid4().hex[:8]}"
        )
        values: dict[str, InputValue] = {}
        for name, spec in by_name.items():
            if name not in row:
                continue
            raw = row[name]
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            if isinstance(raw, str) and raw.strip() == "":
                continue
            values[name] = InputValue(
                value=_coerce(raw, spec.type), unit=spec.unit
            )
        out.append(Experiment(id=exp_id, domain=domain.name, values=values))
    return out


def _coerce(raw, t: InputType):
    if t == InputType.NUMERIC:
        return float(raw)
    if t == InputType.CATEGORICAL:
        return str(raw)
    if t == InputType.COMPOSITION:
        if isinstance(raw, dict):
            return raw
        return str(raw)
    return str(raw)


def _ordered_specs(domain: MaterialDomain):
    # Group by category in UI order, then by declaration order within a group.
    grouped = {c: [] for c in CATEGORY_ORDER}
    for s in domain.all_specs():
        grouped.setdefault(s.category, []).append(s)
    out = []
    for c in CATEGORY_ORDER:
        out.extend(grouped.get(c, []))
    return out


__all__ = ["experiments_to_frame", "frame_to_experiments"]
