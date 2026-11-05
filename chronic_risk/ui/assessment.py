"""The shared assessment page used for all three conditions."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from chronic_risk.inference import (
    MIN_PROVIDED_FRACTION,
    Artifact,
    ArtifactError,
    InputProblem,
    load_artifact,
    predict,
)
from chronic_risk.pipelines import PIPELINES
from chronic_risk.schema import Feature
from chronic_risk.ui import charts, theme

MAX_CHART_ROWS = 8


@st.cache_resource(show_spinner="Loading the model...")
def _cached_artifact(key: str) -> Artifact:
    return load_artifact(key)


def _widget_key(disease: str, feature: Feature) -> str:
    return f"{disease}__{feature.name}"


def _set_typical(disease: str, references: dict[str, Any]) -> None:
    for feature in PIPELINES[disease].features:
        value = references[feature.name]
        if feature.choices:  # use the option object itself so the selectbox recognizes it
            value = next(option for option, _ in feature.choices if option == value)
        else:
            value = float(value)
        st.session_state[_widget_key(disease, feature)] = value


def _clear(disease: str) -> None:
    for feature in PIPELINES[disease].features:
        st.session_state[_widget_key(disease, feature)] = None


def _render_input(disease: str, feature: Feature) -> Any:
    key = _widget_key(disease, feature)
    label = feature.label if feature.required else f"{feature.label} (optional)"
    if feature.choices:
        labels = dict(feature.choices)
        return st.selectbox(
            label,
            options=list(labels),
            format_func=labels.get,
            index=None,
            placeholder="Choose an option" if feature.required else "Not available",
            help=feature.help,
            key=key,
        )
    unit = f" ({feature.unit})" if feature.unit else ""
    return st.number_input(
        f"{label}{unit}",
        min_value=float(feature.min_value),
        max_value=float(feature.max_value),
        step=float(feature.step),
        value=None,
        format="%.1f" if feature.step < 1 else "%.0f",
        placeholder="Enter a value" if feature.required else "Not available",
        help=feature.help,
        key=key,
    )


def _render_form(disease: str) -> tuple[bool, dict[str, Any]]:
    pipeline = PIPELINES[disease]
    groups: dict[str, list[Feature]] = {}
    for feature in pipeline.features:
        groups.setdefault(feature.group, []).append(feature)

    values: dict[str, Any] = {}
    with st.form(f"{disease}_form", border=True):
        for group, features in groups.items():
            st.markdown(f"**{group}**")
            # One row of columns per pair, so phones stack inputs in their intended order.
            for start in range(0, len(features), 2):
                pair = features[start : start + 2]
                for column, feature in zip(st.columns(2), pair, strict=False):
                    with column:
                        values[feature.name] = _render_input(disease, feature)
        submitted = st.form_submit_button("Get my estimate", type="primary")
    return submitted, values


def _render_result(artifact: Artifact, inputs: dict[str, Any]) -> None:
    pipeline, report = artifact.pipeline, artifact.report
    try:
        result = predict(artifact, inputs)
    except InputProblem as problem:
        st.error("Please check these items, then try again:")
        for item in problem.problems:
            st.markdown(f"- {item}")
        return
    except Exception:  # noqa: BLE001 - never expose details that could include health inputs
        st.error("Something went wrong and no estimate could be made. Nothing was saved.")
        return

    st.subheader("Your estimate")
    theme.hero_result(
        result.probability, report["counts"]["development_positive_rate"], pipeline.outcome_phrase
    )
    theme.notice(pipeline.result_caution)

    if result.imputed:
        st.warning(
            f"{len(result.imputed)} of {len(pipeline.features)} items were left blank, and the "
            "model filled them with typical training values. That makes this estimate less "
            "reliable. Blank items: " + ", ".join(result.imputed) + "."
        )
    if result.unusual:
        st.warning(
            "These values are outside the middle 98% of the training data, where the model has "
            "little to learn from. Please double-check them: " + ", ".join(result.unusual) + "."
        )

    rows = pd.DataFrame(
        [{"label": c.label, "points": c.points} for c in result.contributions if c.points != 0],
        columns=["label", "points"],
    )
    st.subheader("What moved this estimate")
    if rows.empty:
        st.info("Your answers match the typical training values, so none of them moved it.")
    else:
        st.altair_chart(charts.contributions_chart(rows.head(MAX_CHART_ROWS)), width="stretch")
        st.caption(
            "Each bar shows how the estimate changes when that answer is swapped for a typical "
            "training value. This describes how the model behaves. It does not show causes, and "
            "changing an answer may not change your actual health."
        )
        with st.expander("See this as a table"):
            st.dataframe(
                rows.rename(columns={"label": "Answer", "points": "Change (percentage points)"}),
                hide_index=True,
                width="stretch",
            )

    theme.notice(
        "A low number does not rule out a condition, and a high number does not mean you have "
        "it. Please share any concerns with a qualified healthcare professional."
    )


def _render_about_model(artifact: Artifact) -> None:
    pipeline, report = artifact.pipeline, artifact.report
    test = report["test"]
    auc = test["metrics"]["roc_auc"]
    low, high = test["ci95"]["roc_auc"]
    with st.expander("What this estimate is based on, and its limits"):
        st.markdown(f"**People in the data:** {pipeline.population}")
        st.markdown(f"**What the outcome means:** {pipeline.target_definition}")
        st.markdown(
            f"**Held-out test:** ROC AUC {auc:.3f} (95% interval {low:.3f} to {high:.3f}) on "
            f"{test['n']:,} records not used in training."
        )
        st.markdown("**Limitations:**")
        for item in pipeline.limitations:
            st.markdown(f"- {item}")
        st.page_link("app_pages/models.py", label="See full results on the Models & data page")


def render_assessment(disease: str) -> None:
    pipeline = PIPELINES[disease]
    st.title(f"{pipeline.title} estimate")
    theme.lede(
        f"Answer a few questions to see how they compare with the {pipeline.dataset_name} "
        "dataset. Your answers are not saved."
    )
    theme.notice(theme.DISCLAIMER)
    theme.notice(pipeline.urgent_note, urgent=True)

    try:
        artifact = _cached_artifact(disease)
    except ArtifactError as error:
        st.error(f"This estimate is unavailable right now. {error}")
        return

    _render_about_model(artifact)

    references = artifact.report["reference_values"]
    left, right, _ = st.columns([1, 1, 2])
    left.button(
        "Fill with typical values",
        on_click=_set_typical,
        args=(disease, references),
        help="Common values from the training data. They do not describe a real person.",
        width="stretch",
    )
    right.button("Clear answers", on_click=_clear, args=(disease,), width="stretch")

    total = len(pipeline.features)
    required = sum(f.required for f in pipeline.features)
    if required < total:
        needed = math.ceil(total * MIN_PROVIDED_FRACTION)
        st.caption(
            f"{required} items are required. Leave any others blank if you don't have them, "
            f"but fill in at least {needed} of the {total} in total."
        )

    submitted, values = _render_form(disease)
    if submitted:
        _render_result(artifact, values)
    else:
        st.info("Your estimate will appear here after you select **Get my estimate**.")
