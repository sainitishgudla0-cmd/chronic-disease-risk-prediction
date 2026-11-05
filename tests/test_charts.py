"""Check chart specs, since a chart can render without error and still mislead."""

import pandas as pd

from chronic_risk.ui import charts


def test_contributions_give_each_answer_its_own_unstacked_row_in_effect_order():
    rows = pd.DataFrame({"label": ["Age", "Heart rate", "Pressure"], "points": [20.0, -1.5, 0.5]})
    spec = charts.contributions_chart(rows).to_dict()
    bars = spec["layer"][0]["encoding"]
    assert bars["y"]["field"] == "answer"
    assert bars["y"]["scale"]["domain"] == ["Age", "Heart rate", "Pressure"]
    assert bars["x"]["stack"] is None
    # Height is the chart's total, so it must also leave room for the legend and axis.
    assert spec["height"] >= charts.CHART_CHROME_HEIGHT + 3 * charts.ROW_HEIGHT


def test_contribution_colors_follow_direction():
    rows = pd.DataFrame({"label": ["Up", "Down"], "points": [2.0, -2.0]})
    chart = charts.contributions_chart(rows)
    values = chart.layer[0].data.set_index("answer")["direction"]
    assert values["Up"] == "Raises" and values["Down"] == "Lowers"


def _axis_titles(spec):
    return {
        channel: {layer["encoding"][channel].get("title") for layer in spec["layer"]}
        for channel in ("x", "y")
    }


def test_calibration_chart_keeps_both_axis_titles():
    table = [
        {"n": 10, "mean_predicted": 0.2, "observed": 0.3, "observed_low": 0.1, "observed_high": 0.5}
    ]
    titles = _axis_titles(charts.calibration_chart(table).to_dict())
    assert "Average model estimate" in titles["x"]
    assert "Share who actually had the outcome" in titles["y"]
    assert None not in titles["x"] - {None} and len(titles["y"] - {None}) == 1


def test_site_chart_uses_one_shared_scale_and_title():
    rows = [{"site": "A", "auc": 0.8, "low": 0.7, "high": 0.9, "n": 50}]
    spec = charts.site_auc_chart(rows).to_dict()
    scales = {str(layer["encoding"]["x"]["scale"]) for layer in spec["layer"]}
    titles = {layer["encoding"]["x"]["title"] for layer in spec["layer"]}
    assert len(scales) == 1 and len(titles) == 1
