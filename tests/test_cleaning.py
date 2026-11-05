import numpy as np
import pandas as pd
import pytest

from chronic_risk.pipelines import diabetes, heart, kidney


def _heart_raw(rows):
    return pd.DataFrame(rows, columns=heart.COLUMNS)


def test_heart_treats_zero_cholesterol_and_pressure_as_missing():
    X, _ = heart.clean(_heart_raw([[60, 1, 4, 0, 0, 0, 0, 120, 1, 1.0, 2, 0, 3, 1]]))
    assert np.isnan(X.loc[0, "chol"]) and np.isnan(X.loc[0, "trestbps"])


def test_heart_binarizes_target_and_decodes_categories():
    X, y = heart.clean(
        _heart_raw(
            [
                [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
                [67, 0, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 7, 4],
            ]
        )
    )
    assert y.tolist() == [0, 1]
    assert X.loc[0, "cp"] == "typical" and X.loc[1, "cp"] == "asymptomatic"
    assert X.loc[0, "thal"] == "fixed_defect" and X.loc[1, "sex"] == "female"


def test_heart_question_marks_become_missing_values():
    X, _ = heart.clean(_heart_raw([[60, 1, 4, 130, 200, 0, 0, 120, 1, 1.0, "?", "?", "?", 1]]))
    assert X[["slope", "ca", "thal"]].isna().all(axis=None)


def _kidney_raw():
    columns = kidney.ALL_NUMERIC + kidney.ALL_CATEGORICAL + ["class"]
    first = dict.fromkeys(columns, np.nan)
    first.update({"age": 50, "hemo": "\t12.5", "dm": "\tno", "htn": " yes", "class": "ckd\t"})
    second = dict.fromkeys(columns, np.nan)
    second.update({"age": 40, "hemo": 15.0, "dm": "yes", "htn": "no", "class": "notckd"})
    return pd.DataFrame([first, second], columns=columns)


def test_kidney_repairs_stray_whitespace():
    dataset = kidney.clean(_kidney_raw())
    assert dataset.y.tolist() == [1, 0]
    assert dataset.X.loc[0, "dm"] == "no" and dataset.X.loc[0, "htn"] == "yes"
    assert dataset.X.loc[0, "hemo"] == pytest.approx(12.5)


def test_kidney_model_excludes_circular_markers_and_adds_no_missing_flags():
    names = {f.name for f in kidney.FEATURES}
    assert not names & {"sc", "bu", "al", "rbc", "rbcc", "wbcc", "pot", "sod"}
    assert not any("missing" in name for name in kidney.clean(_kidney_raw()).X.columns)


def test_kidney_rejects_unknown_labels():
    raw = _kidney_raw()
    raw["class"] = raw["class"].astype(object)
    raw.loc[0, "class"] = "maybe"
    with pytest.raises(ValueError, match="Unexpected CKD labels"):
        kidney.clean(raw)


def test_diabetes_reports_missing_columns():
    with pytest.raises(ValueError, match="missing expected columns"):
        diabetes.clean(pd.DataFrame({"Age": [1]}))


def test_diabetes_drops_personal_socioeconomic_columns():
    names = {f.name for f in diabetes.FEATURES}
    assert not names & {"ID", "Income", "Education", "AnyHealthcare", "NoDocbcCost"}
