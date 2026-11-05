"""Altair charts. Each has a tooltip and is paired with a table or written explanation."""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd

SERIES = "#2a78d6"
RAISES = "#e34948"
LOWERS = "#2a78d6"
REFERENCE = "#8a8984"
ROW_HEIGHT = 36
# Streamlit fits a chart's legend, axis and axis title inside its height, so they need room too.
# Without this allowance the rows collapse on top of each other.
CHART_CHROME_HEIGHT = 110


def row_chart_height(rows: int, row_height: int) -> int:
    """Total height for a chart with one row per item."""
    return CHART_CHROME_HEIGHT + row_height * rows


def contributions_chart(rows: pd.DataFrame) -> alt.LayerChart:
    """Diverging bars of how each answer moved this estimate, in percentage points."""
    data = rows.rename(columns={"label": "answer"}).assign(
        direction=rows["points"].map(lambda v: "Raises" if v > 0 else "Lowers")
    )
    order = data["answer"].tolist()  # largest effect first
    bars = (
        alt.Chart(data)
        .mark_bar(size=14, cornerRadiusEnd=4)
        .encode(
            x=alt.X("points:Q", title="Change in percentage points", stack=None),
            y=alt.Y(
                "answer:N",
                sort=order,
                title=None,
                scale=alt.Scale(domain=order),
                axis=alt.Axis(labelLimit=220),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Raises", "Lowers"], range=[RAISES, LOWERS]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("answer:N", title="Answer"),
                alt.Tooltip("points:Q", title="Change (points)", format="+.1f"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=REFERENCE).encode(x="x:Q")
    return (bars + zero).properties(height=row_chart_height(len(data), ROW_HEIGHT))


def _reference_line() -> alt.Chart:
    line = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    return (
        alt.Chart(line)
        .mark_line(color=REFERENCE, strokeDash=[4, 4], strokeWidth=1.5)
        .encode(x=alt.X("x:Q", title=None), y=alt.Y("y:Q", title=None))
    )


def calibration_chart(table: list[dict[str, Any]]) -> alt.LayerChart:
    data = pd.DataFrame(table)
    axis_x = alt.X(
        "mean_predicted:Q",
        title="Average model estimate",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format="%"),
    )
    observed_title = "Share who actually had the outcome"
    axis_y = alt.Y(
        "observed:Q",
        title=observed_title,
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format="%"),
    )
    tooltip = [
        alt.Tooltip("mean_predicted:Q", title="Average estimate", format=".0%"),
        alt.Tooltip("observed:Q", title="Observed", format=".0%"),
        alt.Tooltip("observed_low:Q", title="95% interval low", format=".0%"),
        alt.Tooltip("observed_high:Q", title="95% interval high", format=".0%"),
        alt.Tooltip("n:Q", title="Records in group"),
    ]
    whiskers = (
        alt.Chart(data)
        .mark_rule(color=SERIES, strokeWidth=1.5)
        .encode(
            x=axis_x,
            y=alt.Y("observed_low:Q", title=observed_title, scale=alt.Scale(domain=[0, 1])),
            y2="observed_high:Q",
        )
    )
    line = alt.Chart(data).mark_line(color=SERIES, strokeWidth=2).encode(x=axis_x, y=axis_y)
    points = (
        alt.Chart(data)
        .mark_point(color=SERIES, filled=True, size=70, opacity=1)
        .encode(x=axis_x, y=axis_y, tooltip=tooltip)
    )
    return (_reference_line() + whiskers + line + points).properties(height=320)


def roc_chart(points: list[dict[str, float]]) -> alt.LayerChart:
    data = pd.DataFrame(points)
    x = alt.X("fpr:Q", title="False positive rate", scale=alt.Scale(domain=[0, 1]))
    y = alt.Y("tpr:Q", title="True positive rate", scale=alt.Scale(domain=[0, 1]))
    line = (
        alt.Chart(data)
        .mark_line(color=SERIES, strokeWidth=2, interpolate="step-after")
        .encode(x=x, y=y)
    )
    hover = (
        alt.Chart(data)
        .mark_point(opacity=0, size=200)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("fpr:Q", title="False positive rate", format=".0%"),
                alt.Tooltip("tpr:Q", title="True positive rate", format=".0%"),
            ],
        )
    )
    return (_reference_line() + line + hover).properties(height=320)


def site_auc_chart(rows: list[dict[str, Any]]) -> alt.LayerChart:
    """Dot and 95% interval of ROC AUC at each site, against the chance line."""
    data = pd.DataFrame(rows)
    scale = alt.Scale(domain=[0.4, 1], clamp=True)
    title = "ROC AUC (0.5 is chance, 1 is perfect ranking)"
    y = alt.Y("site:N", sort=None, title=None, axis=alt.Axis(labelLimit=200))
    interval = (
        alt.Chart(data)
        .mark_rule(color=SERIES, strokeWidth=2)
        .encode(x=alt.X("low:Q", title=title, scale=scale), x2="high:Q", y=y)
    )
    dot = (
        alt.Chart(data)
        .mark_point(color=SERIES, filled=True, size=90, opacity=1)
        .encode(
            x=alt.X("auc:Q", title=title, scale=scale),
            y=y,
            tooltip=[
                alt.Tooltip("site:N", title="Data"),
                alt.Tooltip("auc:Q", title="ROC AUC", format=".3f"),
                alt.Tooltip("low:Q", title="95% interval low", format=".3f"),
                alt.Tooltip("high:Q", title="95% interval high", format=".3f"),
                alt.Tooltip("n:Q", title="Records"),
            ],
        )
    )
    chance = (
        alt.Chart(pd.DataFrame({"x": [0.5]}))
        .mark_rule(color=REFERENCE, strokeDash=[4, 4])
        .encode(x=alt.X("x:Q", title=title, scale=scale))
    )
    return (chance + interval + dot).properties(height=row_chart_height(len(data), ROW_HEIGHT))
