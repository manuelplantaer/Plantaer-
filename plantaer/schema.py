"""Typed experiment schema.

Every input an experiment has is described by an `InputSpec` carrying both a
*category* (what the input represents in the lab) and a *type* (how it is
stored and featurized). Featurization, validation, and BO bounds all read
from this one source of truth, so adding a new input column to a material
propagates automatically through prediction and experiment suggestion.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Category(str, Enum):
    COMPOSITION = "composition"
    STRUCTURE = "structure"
    COMPONENT = "component"
    PROCESSING = "processing"
    ENVIRONMENT = "environment"
    GEOMETRY = "geometry"
    CHARACTERIZATION = "characterization"
    METADATA = "metadata"
    TARGET = "target"


class InputType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    COMPOSITION = "composition"
    SMILES = "smiles"
    STRUCTURE_REF = "structure_ref"
    SPECTRUM_REF = "spectrum_ref"
    IMAGE_REF = "image_ref"
    TEXT = "text"


# Categories grouped by the UI; also drives color-coding in the spreadsheet.
CATEGORY_ORDER: tuple[Category, ...] = (
    Category.COMPOSITION,
    Category.STRUCTURE,
    Category.COMPONENT,
    Category.PROCESSING,
    Category.ENVIRONMENT,
    Category.GEOMETRY,
    Category.CHARACTERIZATION,
    Category.METADATA,
    Category.TARGET,
)


class InputSpec(BaseModel):
    """Declaration of one input (or target) column for a material."""

    name: str
    category: Category
    type: InputType
    unit: str | None = None
    bounds: tuple[float, float] | None = None  # numeric only; drives BO
    choices: list[str] | None = None  # categorical only
    required: bool = False
    description: str | None = None

    @model_validator(mode="after")
    def _check_constraints(self) -> "InputSpec":
        if self.bounds is not None and self.type != InputType.NUMERIC:
            raise ValueError(f"{self.name}: bounds only valid for numeric type")
        if self.bounds is not None and self.bounds[0] >= self.bounds[1]:
            raise ValueError(f"{self.name}: bounds must satisfy lo < hi")
        if self.choices is not None and self.type != InputType.CATEGORICAL:
            raise ValueError(f"{self.name}: choices only valid for categorical type")
        if self.category == Category.TARGET and self.type != InputType.NUMERIC:
            raise ValueError(f"{self.name}: targets must be numeric in v1")
        return self


class MaterialDomain(BaseModel):
    """A user-declared material class with its inputs and targets."""

    name: str
    description: str | None = None
    inputs: list[InputSpec] = Field(default_factory=list)
    targets: list[InputSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_unique_names(self) -> "MaterialDomain":
        seen: set[str] = set()
        for spec in self.all_specs():
            if spec.name in seen:
                raise ValueError(f"duplicate column name: {spec.name}")
            seen.add(spec.name)
        for spec in self.targets:
            if spec.category != Category.TARGET:
                raise ValueError(
                    f"target '{spec.name}' must have category=target, got {spec.category}"
                )
        return self

    def all_specs(self) -> list[InputSpec]:
        return [*self.inputs, *self.targets]

    def spec(self, name: str) -> InputSpec:
        for s in self.all_specs():
            if s.name == name:
                return s
        raise KeyError(f"no spec named {name!r} in material {self.name!r}")


class InputValue(BaseModel):
    """One recorded value for an input on a specific experiment."""

    value: Any  # float | str | dict[str, float] depending on type
    unit: str | None = None


class Experiment(BaseModel):
    """One logged experiment row under a specific material domain."""

    id: str
    domain: str  # references MaterialDomain.name
    values: dict[str, InputValue] = Field(default_factory=dict)
    artifacts: dict[str, Path] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def validate_against(self, domain: MaterialDomain) -> None:
        """Raise if this experiment does not satisfy the domain's schema."""
        if self.domain != domain.name:
            raise ValueError(
                f"experiment.domain={self.domain!r} does not match {domain.name!r}"
            )
        known = {s.name for s in domain.all_specs()}
        for name in self.values:
            if name not in known:
                raise ValueError(f"unknown column {name!r} for material {domain.name!r}")
        for spec in domain.all_specs():
            if spec.required and spec.name not in self.values:
                raise ValueError(f"missing required column {spec.name!r}")
            if spec.name in self.values:
                _check_value(self.values[spec.name], spec)


def _check_value(iv: InputValue, spec: InputSpec) -> None:
    v = iv.value
    if spec.type == InputType.NUMERIC:
        if not isinstance(v, (int, float)):
            raise ValueError(f"{spec.name}: expected numeric, got {type(v).__name__}")
        if spec.bounds is not None:
            lo, hi = spec.bounds
            if not (lo <= float(v) <= hi):
                raise ValueError(
                    f"{spec.name}={v} outside bounds [{lo}, {hi}]"
                )
    elif spec.type == InputType.CATEGORICAL:
        if not isinstance(v, str):
            raise ValueError(f"{spec.name}: expected str, got {type(v).__name__}")
        if spec.choices is not None and v not in spec.choices:
            raise ValueError(f"{spec.name}={v!r} not in {spec.choices}")
    elif spec.type == InputType.COMPOSITION:
        if not isinstance(v, (str, dict)):
            raise ValueError(f"{spec.name}: composition must be formula str or dict")
    elif spec.type in (
        InputType.SMILES,
        InputType.STRUCTURE_REF,
        InputType.SPECTRUM_REF,
        InputType.IMAGE_REF,
        InputType.TEXT,
    ):
        if not isinstance(v, str):
            raise ValueError(f"{spec.name}: {spec.type} requires string value")


__all__ = [
    "Category",
    "InputType",
    "CATEGORY_ORDER",
    "InputSpec",
    "MaterialDomain",
    "InputValue",
    "Experiment",
]
