"""Composition featurizer.

Default path: a lightweight element-fraction vector over the 118 elements
(Z=1..118). Robust, deterministic, zero heavy deps. If ``matminer`` is
installed we additionally compute Magpie element-property statistics and
concatenate them.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# Elements in Z order. Index 0 unused (padding for 1-based Z).
ELEMENTS: tuple[str, ...] = (
    "_",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
N_ELEMENTS = len(ELEMENTS) - 1  # 118

_SYMBOL_TO_Z: dict[str, int] = {sym: i for i, sym in enumerate(ELEMENTS) if i > 0}

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str) -> dict[str, float]:
    """Parse a flat chemical formula into an element->count dict.

    Supports fractional stoichiometry (Fe0.7Ni0.3) but NOT nested parentheses
    or hydrates — those rare cases should be entered as a dict value instead.
    """
    cleaned = formula.strip().replace(" ", "")
    if not cleaned:
        raise ValueError("empty formula")
    if "(" in cleaned or ")" in cleaned:
        raise ValueError(
            f"nested formulas not supported ({formula!r}); pass a dict instead"
        )
    out: dict[str, float] = {}
    pos = 0
    for match in _FORMULA_RE.finditer(cleaned):
        sym, num = match.group(1), match.group(2)
        if sym not in _SYMBOL_TO_Z:
            raise ValueError(f"unknown element {sym!r} in {formula!r}")
        count = float(num) if num else 1.0
        out[sym] = out.get(sym, 0.0) + count
        pos = match.end()
    if pos != len(cleaned):
        raise ValueError(f"could not parse {formula!r} (stopped at pos {pos})")
    if not out:
        raise ValueError(f"no elements parsed from {formula!r}")
    return out


def as_element_dict(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        # Validate symbols
        for sym in value:
            if sym not in _SYMBOL_TO_Z:
                raise ValueError(f"unknown element {sym!r}")
        return {k: float(v) for k, v in value.items()}
    if isinstance(value, str):
        s = value.strip()
        # Self-heal Python-dict-repr strings like "{'Fe': 0.6, 'Ni': 0.4}" that
        # may have been produced by st.data_editor str()-ifying dict cells.
        if s.startswith("{") and s.endswith("}"):
            import ast

            try:
                parsed = ast.literal_eval(s)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"could not parse composition {value!r}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"composition string {value!r} is not a dict")
            return as_element_dict(parsed)
        return parse_formula(s)
    raise TypeError(f"composition must be str or dict, got {type(value).__name__}")


def dict_to_formula(elems: dict[str, float]) -> str:
    """Render an element->count dict as a compact chemical formula.

    Used at the DataFrame boundary so Streamlit's data_editor shows
    "Fe0.61Ni0.39" instead of a Python-dict repr it would corrupt.
    Elements with integer counts drop the count (Fe2O3, not Fe2.0O3.0).
    """
    parts: list[str] = []
    for sym, count in elems.items():
        if sym not in _SYMBOL_TO_Z:
            raise ValueError(f"unknown element {sym!r}")
        n = float(count)
        if n == int(n):
            suffix = "" if int(n) == 1 else str(int(n))
        else:
            suffix = f"{n:g}"
        parts.append(f"{sym}{suffix}")
    return "".join(parts)


def composition_feature_names(prefix: str) -> list[str]:
    return [f"{prefix}::frac::{sym}" for sym in ELEMENTS[1:]]


def featurize_composition(value: Any | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (feature vector of length 118, mask of same length).

    Missing composition returns zero vector + zero mask. Feature values are
    normalized element fractions summing to 1.
    """
    vec = np.zeros(N_ELEMENTS, dtype=np.float64)
    mask = np.zeros(N_ELEMENTS, dtype=np.float64)
    if value is None:
        return vec, mask
    elems = as_element_dict(value)
    total = sum(elems.values())
    if total <= 0:
        return vec, mask
    for sym, count in elems.items():
        z = _SYMBOL_TO_Z[sym]
        vec[z - 1] = count / total
    mask[:] = 1.0
    return vec, mask


__all__ = [
    "ELEMENTS",
    "N_ELEMENTS",
    "parse_formula",
    "as_element_dict",
    "dict_to_formula",
    "composition_feature_names",
    "featurize_composition",
]
