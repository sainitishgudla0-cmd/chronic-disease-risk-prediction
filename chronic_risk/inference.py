"""Load verified model files and produce estimates. Inputs are never stored or logged."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from chronic_risk.paths import MANIFEST_PATH, MODELS_DIR
from chronic_risk.pipelines import PIPELINES
from chronic_risk.schema import DiseasePipeline, coerce_frame

MIN_PROVIDED_FRACTION = 0.5
SETUP_HINT = "Run `python -m chronic_risk.train all` to create the model files."


class ArtifactError(RuntimeError):
    """A model file is missing, modified, or built with a different scikit-learn version."""


class InputProblem(ValueError):
    """User inputs failed validation. ``problems`` holds readable messages."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Artifact:
    pipeline: DiseasePipeline
    model: Any
    report: dict[str, Any]


@dataclass(frozen=True)
class Contribution:
    feature: str
    label: str
    points: float


@dataclass(frozen=True)
class Prediction:
    probability: float
    contributions: tuple[Contribution, ...]
    imputed: tuple[str, ...]
    unusual: tuple[str, ...]


def load_report(key: str, models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    path = Path(models_dir) / f"{key}.json"
    if not path.exists():
        raise ArtifactError(f"The evaluation report for {key} is missing. {SETUP_HINT}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"The evaluation report for {key} is unreadable. {SETUP_HINT}") from exc


def load_artifact(
    key: str, models_dir: Path = MODELS_DIR, manifest_path: Path = MANIFEST_PATH
) -> Artifact:
    """Load a model only if its checksum and scikit-learn version match the manifest."""
    pipeline = PIPELINES[key]
    model_path = Path(models_dir) / f"{key}.joblib"
    if not model_path.exists() or not Path(manifest_path).exists():
        raise ArtifactError(f"The model file for {pipeline.title} is missing. {SETUP_HINT}")
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        expected = manifest["sha256"][model_path.name]
        trained_with = manifest["sklearn_version"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise ArtifactError(f"The model manifest is incomplete. {SETUP_HINT}") from exc
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != expected:
        raise ArtifactError(
            f"The model file for {pipeline.title} failed its integrity check and was not loaded."
        )
    if trained_with != sklearn.__version__:
        raise ArtifactError(
            f"The models were built with scikit-learn {trained_with}, but {sklearn.__version__} "
            f"is installed. Install requirements.txt, or retrain. {SETUP_HINT}"
        )
    return Artifact(pipeline, joblib.load(model_path), load_report(key, models_dir))


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def validate_inputs(pipeline: DiseasePipeline, inputs: dict[str, Any]) -> list[str]:
    """Return readable problems: missing required values, invalid options, impossible numbers."""
    problems = []
    provided = 0
    for feature in pipeline.features:
        value = inputs.get(feature.name)
        if _is_blank(value):
            if feature.required:
                problems.append(f"{feature.label}: this is required.")
            continue
        provided += 1
        if feature.choices:
            if value not in [option for option, _ in feature.choices]:
                problems.append(f"{feature.label}: please choose one of the listed options.")
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            problems.append(f"{feature.label}: please enter a number.")
            continue
        if not math.isfinite(number):
            problems.append(f"{feature.label}: please enter a number.")
        elif not feature.min_value <= number <= feature.max_value:
            unit = f" {feature.unit}" if feature.unit else ""
            problems.append(
                f"{feature.label}: enter a value from {feature.min_value:g} "
                f"to {feature.max_value:g}{unit}."
            )
    needed = math.ceil(len(pipeline.features) * MIN_PROVIDED_FRACTION)
    if provided < needed:
        problems.append(
            f"Please fill in at least {needed} of the {len(pipeline.features)} items. "
            f"So far {provided} are filled in, which is too few for a useful estimate."
        )
    return problems


def _frame(pipeline: DiseasePipeline, rows: list[dict[str, Any]]) -> pd.DataFrame:
    filled = [{f.name: row.get(f.name, np.nan) for f in pipeline.features} for row in rows]
    return coerce_frame(pd.DataFrame(filled, dtype=object), pipeline.features)


def predict(artifact: Artifact, inputs: dict[str, Any]) -> Prediction:
    """Estimate the probability and show how each provided answer moved it.

    Each contribution is the change seen when that one answer is swapped for a typical training
    value. It describes this model's behavior and says nothing about cause and effect.
    """
    pipeline = artifact.pipeline
    problems = validate_inputs(pipeline, inputs)
    if problems:
        raise InputProblem(problems)

    clean = {k: (np.nan if _is_blank(v) else v) for k, v in inputs.items()}
    provided = [f for f in pipeline.features if not _is_blank(clean.get(f.name))]
    references = artifact.report["reference_values"]
    rows = [clean] + [{**clean, f.name: references[f.name]} for f in provided]
    scores = artifact.model.predict_proba(_frame(pipeline, rows))[:, 1]
    probability = float(scores[0])

    contributions = sorted(
        (
            Contribution(f.name, f.label, round((probability - float(score)) * 100, 2))
            for f, score in zip(provided, scores[1:], strict=True)
        ),
        key=lambda item: abs(item.points),
        reverse=True,
    )
    ranges = artifact.report.get("typical_ranges", {})
    unusual = tuple(
        f.label
        for f in provided
        if f.name in ranges and not ranges[f.name][0] <= float(clean[f.name]) <= ranges[f.name][1]
    )
    imputed = tuple(f.label for f in pipeline.features if _is_blank(clean.get(f.name)))
    return Prediction(probability, tuple(contributions), imputed, unusual)
