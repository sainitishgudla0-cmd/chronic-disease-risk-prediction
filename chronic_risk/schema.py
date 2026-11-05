"""Shared descriptions of inputs, datasets and per-disease pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Feature:
    """One model input, including how the app should ask for it."""

    name: str
    label: str
    help: str
    kind: Literal["numeric", "categorical"]
    group: str
    required: bool = False
    choices: tuple[tuple[Any, str], ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    step: float = 1.0
    unit: str = ""


@dataclass(frozen=True)
class DataSource:
    """A remote file pinned by checksum so training always sees the same bytes."""

    url: str
    filename: str
    sha256: str


@dataclass
class Dataset:
    """Cleaned features and binary target, plus any data kept apart from development."""

    X: pd.DataFrame
    y: np.ndarray
    external: dict[str, tuple[pd.DataFrame, np.ndarray]] = field(default_factory=dict)
    full_features: pd.DataFrame | None = None


@dataclass(frozen=True)
class DiseasePipeline:
    """Everything specific to one condition. Shared code never hard-codes a disease."""

    key: str
    title: str
    outcome_phrase: str
    target_definition: str
    population: str
    dataset_name: str
    dataset_url: str
    license_name: str
    license_url: str
    citation: str
    limitations: tuple[str, ...]
    features: tuple[Feature, ...]
    excluded_features: Mapping[str, str]
    sources: tuple[DataSource, ...]
    load_dataset: Callable[[Any], Dataset]
    subgroups: Callable[[pd.DataFrame], dict[str, np.ndarray]]
    urgent_note: str
    cv_folds: int = 5
    cv_repeats: int = 5
    n_bootstrap: int = 1000
    diagnostics: Callable[..., dict[str, Any]] | None = None
    result_caution: str = ""

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]


def coerce_frame(df: pd.DataFrame, features: tuple[Feature, ...]) -> pd.DataFrame:
    """Return exactly the model columns with the types the preprocessing expects."""
    out = pd.DataFrame(index=df.index)
    for feat in features:
        if feat.kind == "numeric":
            out[feat.name] = pd.to_numeric(df[feat.name], errors="coerce").astype(float)
        else:
            column = df[feat.name].astype(object)
            out[feat.name] = column.where(column.notna(), np.nan).astype(object)
    return out
