"""Chronic kidney disease pipeline: UCI CKD, with diagnostic and missingness leakage controls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from chronic_risk.data import ensure_source
from chronic_risk.schema import Dataset, DataSource, DiseasePipeline, Feature, coerce_frame

YES_NO = (("yes", "Yes"), ("no", "No"))
LABS = "Blood and urine tests"
HISTORY = "Health history"

FEATURES = (
    Feature("age", "Age", "Age in years. This tool is for adults.", "numeric", "About you",
            required=True, min_value=18, max_value=100, unit="years"),
    Feature("bp", "Blood pressure", "The dataset does not say which reading this is. It holds "
            "values such as 70 or 80, which resemble diastolic (bottom number) readings.",
            "numeric", "About you", min_value=40, max_value=180, step=1, unit="mm Hg"),
    Feature("htn", "Told you have high blood pressure?", "Told by a health professional.",
            "categorical", HISTORY, choices=YES_NO),
    Feature("dm", "Told you have diabetes?", "Told by a health professional.", "categorical",
            HISTORY, choices=YES_NO),
    Feature("cad", "Told you have coronary artery disease?", "Told by a health professional.",
            "categorical", HISTORY, choices=YES_NO),
    Feature("appet", "Appetite", "Your usual appetite recently.", "categorical", HISTORY,
            choices=(("good", "Good"), ("poor", "Poor"))),
    Feature("pe", "Swelling in the lower legs or feet?", "Also called pedal edema.",
            "categorical", HISTORY, choices=YES_NO),
    Feature("ane", "Told you have anemia?", "Told by a health professional.", "categorical",
            HISTORY, choices=YES_NO),
    Feature("hemo", "Hemoglobin", "From a blood count, in g/dL.", "numeric", LABS,
            min_value=3, max_value=20, step=0.1, unit="g/dL"),
    Feature("pcv", "Packed cell volume", "Also called hematocrit, from a blood count.",
            "numeric", LABS, min_value=10, max_value=60, step=1, unit="%"),
    Feature("bgr", "Random blood glucose", "A blood sugar test at any time of day.", "numeric",
            LABS, min_value=20, max_value=500, unit="mg/dL"),
    Feature("sg", "Urine specific gravity", "From a urine dipstick test.", "numeric", LABS,
            choices=((1.005, "1.005"), (1.010, "1.010"), (1.015, "1.015"), (1.020, "1.020"),
                     (1.025, "1.025"))),
    Feature("su", "Urine sugar", "Dipstick grade from 0 (none) to 5.", "numeric", LABS,
            choices=tuple((i, str(i)) for i in range(6))),
)

ALL_NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv",
               "wbcc", "rbcc"]
ALL_CATEGORICAL = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

SOURCE = DataSource(
    url="https://archive.ics.uci.edu/static/public/336/data.csv",
    filename="chronic_kidney_disease.csv",
    sha256="0eeea8d17f5ad8792d854999d6ddf1602ec4e116616f6ccdfdaef0ba8109e694",
)


def clean(raw: pd.DataFrame) -> Dataset:
    """Repair stray tabs and spaces and type each column. Full columns are kept for diagnostics."""
    frame = raw.copy()
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = (
                frame[column].astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
            )
    labels = frame["class"].map({"ckd": 1, "notckd": 0})
    if labels.isna().any():
        raise ValueError(f"Unexpected CKD labels: {sorted(frame['class'][labels.isna()].unique())}")
    for column in ALL_NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ALL_CATEGORICAL:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    full = frame[ALL_NUMERIC + ALL_CATEGORICAL].reset_index(drop=True)
    return Dataset(
        X=coerce_frame(full, FEATURES),
        y=labels.astype(int).to_numpy(),
        full_features=full,
    )


def load_dataset(raw_dir: Path) -> Dataset:
    path = ensure_source(SOURCE, Path(raw_dir) / "kidney")
    return clean(pd.read_csv(path))


def subgroups(X: pd.DataFrame) -> dict[str, np.ndarray]:
    age = X["age"].to_numpy()
    return {"Age under 60": age < 60, "Age 60+": age >= 60}


def diagnostics(
    *,
    dev_index: np.ndarray,
    test_index: np.ndarray,
    y: np.ndarray,
    full_features: pd.DataFrame,
    X_dev: pd.DataFrame,
    seed: int,
    candidate_factory: Any,
    model: Any,
    X_test: pd.DataFrame,
    **_: Any,
) -> dict[str, Any]:
    """Measure the leakage this dataset is known for, so the app can report it plainly."""
    y_dev, y_test = y[dev_index], y[test_index]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Can the label be predicted only from WHICH values are absent, ignoring the values?
    missing = full_features.isna().astype(float).to_numpy()
    probe = LogisticRegression(max_iter=2000)
    oof = cross_val_predict(
        probe, missing[dev_index], y_dev, cv=folds, method="predict_proba"
    )[:, 1]

    # What would the full 24-variable model score, including the diagnostic markers?
    all_features = tuple(
        Feature(name, name, "", "numeric", "") for name in ALL_NUMERIC
    ) + tuple(Feature(name, name, "", "categorical", "") for name in ALL_CATEGORICAL)
    full_model = candidate_factory(all_features)
    full_model.fit(full_features.iloc[dev_index], y_dev)
    full_scores = full_model.predict_proba(full_features.iloc[test_index])[:, 1]

    # Performance on rows where nothing had to be imputed for the selected inputs.
    complete = X_test.notna().all(axis=1).to_numpy()
    n_pos, n_neg = int(y_test[complete].sum()), int((1 - y_test[complete]).sum())
    complete_auc = None
    if n_pos >= 5 and n_neg >= 5:
        scores = model.predict_proba(X_test[complete])[:, 1]
        complete_auc = float(roc_auc_score(y_test[complete], scores))

    missing_rate = X_dev.isna().mean()
    return {
        "missingness_only_cv_auc": float(roc_auc_score(y_dev, oof)),
        "all_24_features_test_auc": float(roc_auc_score(y_test, full_scores)),
        "complete_case_test": {"n_positive": n_pos, "n_negative": n_neg, "auc": complete_auc},
        "selected_feature_missing_rate": {k: float(v) for k, v in missing_rate.items()},
    }


CAUTION = (
    "Treat this as a classroom example. The dataset separates people unusually cleanly, so the "
    "model can look far more certain than is realistic, especially at values near 0% or 100%. "
    "Kidney disease is assessed with blood and urine tests such as creatinine and albumin."
)

PIPELINE = DiseasePipeline(
    key="kidney",
    title="Kidney disease",
    outcome_phrase="were labelled as having chronic kidney disease",
    target_definition=(
        "The hospital dataset's label: 'ckd' or 'notckd'. The source does not document the "
        "clinical criteria behind the label."
    ),
    population=(
        "400 people seen at one hospital over about two months, according to the dataset's "
        "donors. It includes some children, although this tool only accepts adults."
    ),
    dataset_name="UCI Chronic Kidney Disease",
    dataset_url="https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease",
    license_name="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    citation=(
        "Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic Kidney Disease "
        "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5G020."
    ),
    limitations=(
        "Very small and from a single hospital over about two months. Treat every number here "
        "as a classroom example rather than evidence of real-world accuracy.",
        "High scores on this dataset are expected and should not be taken at face value. "
        "People with and without the label are unusually well separated, unlike everyday care.",
        "Missing values are concentrated in people labelled with CKD, so missingness itself can "
        "give away the answer. See the leakage checks on the Models & data page.",
        "Creatinine, urea and albumin were removed because they help define kidney disease. "
        "This tool therefore cannot stand in for those tests, which clinicians rely on.",
        "The file has no sex or ethnicity field, so performance across those groups is unknown.",
    ),
    features=FEATURES,
    excluded_features={
        "sc, bu, al (creatinine, urea, albumin)": "Direct kidney markers that help define the "
        "diagnosis. Using them would be circular.",
        "rbc, rbcc, wbcc, pot, sod": "More than 20% missing in the full file, with absences "
        "concentrated in the CKD group.",
        "pc, pcc, ba": "Urine microscopy findings, left out to keep to more commonly available "
        "results.",
    },
    sources=(SOURCE,),
    load_dataset=load_dataset,
    subgroups=subgroups,
    urgent_note=(
        "Passing little or no urine, sudden swelling, severe breathlessness or confusion can be "
        "an emergency. Contact local emergency services rather than using this tool."
    ),
    diagnostics=diagnostics,
    result_caution=CAUTION,
)
