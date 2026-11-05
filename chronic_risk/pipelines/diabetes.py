"""Diabetes pipeline: CDC Diabetes Health Indicators (BRFSS 2015 survey)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chronic_risk.data import ensure_source
from chronic_risk.schema import Dataset, DataSource, DiseasePipeline, Feature, coerce_frame

YES_NO = ((0, "No"), (1, "Yes"))

AGE_BRACKETS = (
    (1, "18-24"),
    (2, "25-29"),
    (3, "30-34"),
    (4, "35-39"),
    (5, "40-44"),
    (6, "45-49"),
    (7, "50-54"),
    (8, "55-59"),
    (9, "60-64"),
    (10, "65-69"),
    (11, "70-74"),
    (12, "75-79"),
    (13, "80 or older"),
)


def _yes_no(name: str, label: str, help_text: str, group: str) -> Feature:
    return Feature(name, label, help_text, "numeric", group, required=True, choices=YES_NO)


FEATURES = (
    Feature(
        "Age", "Age group", "Age bands as recorded in the survey.", "numeric", "About you",
        required=True, choices=AGE_BRACKETS,
    ),
    Feature(
        "Sex", "Sex", "As recorded in the survey, which offered only these two options.",
        "numeric", "About you", required=True, choices=((0, "Female"), (1, "Male")),
    ),
    Feature(
        "BMI", "Body mass index (BMI)",
        "Weight in kg divided by height in metres squared. Example: 70 kg at 1.75 m gives 22.9.",
        "numeric", "About you", required=True, min_value=12, max_value=98, step=0.1,
        unit="kg/m²",
    ),
    _yes_no("HighBP", "Told you have high blood pressure?",
            "Has a health professional ever said so?", "Health history"),
    _yes_no("HighChol", "Told you have high cholesterol?",
            "Has a health professional ever said so?", "Health history"),
    _yes_no("CholCheck", "Cholesterol checked in the past 5 years?",
            "Any cholesterol blood test within the last five years.", "Health history"),
    _yes_no("Stroke", "Ever told you had a stroke?", "Told by a health professional.",
            "Health history"),
    _yes_no("HeartDiseaseorAttack", "Coronary heart disease or heart attack?",
            "Ever reported coronary heart disease or a myocardial infarction.", "Health history"),
    _yes_no("Smoker", "Smoked at least 100 cigarettes in your life?",
            "About five packs in total, whether or not you still smoke.", "Daily habits"),
    _yes_no("PhysActivity", "Physical activity in the past 30 days?",
            "Any exercise outside your regular job.", "Daily habits"),
    _yes_no("Fruits", "Eat fruit at least once a day?", "One or more times per day.",
            "Daily habits"),
    _yes_no("Veggies", "Eat vegetables at least once a day?", "One or more times per day.",
            "Daily habits"),
    _yes_no("HvyAlcoholConsump", "Heavy alcohol use?",
            "Survey definition: more than 14 drinks a week for men, more than 7 for women.",
            "Daily habits"),
    Feature(
        "GenHlth", "How is your health in general?", "Your own overall rating.", "numeric",
        "Wellbeing", required=True,
        choices=((1, "Excellent"), (2, "Very good"), (3, "Good"), (4, "Fair"), (5, "Poor")),
    ),
    Feature(
        "PhysHlth", "Days of poor physical health, past 30 days",
        "Include illness and injury. Enter 0 if none.", "numeric", "Wellbeing", required=True,
        min_value=0, max_value=30, unit="days",
    ),
    Feature(
        "MentHlth", "Days of poor mental health, past 30 days",
        "Include stress, depression and problems with emotions. Enter 0 if none.", "numeric",
        "Wellbeing", required=True, min_value=0, max_value=30, unit="days",
    ),
    _yes_no("DiffWalk", "Serious difficulty walking or climbing stairs?",
            "As you would answer it for yourself today.", "Wellbeing"),
)

SOURCE = DataSource(
    url="https://archive.ics.uci.edu/static/public/891/data.csv",
    filename="cdc_diabetes_health_indicators.csv",
    sha256="9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964",
)


def clean(raw: pd.DataFrame) -> Dataset:
    """Select model columns from the UCI CSV and build the binary target."""
    missing = {f.name for f in FEATURES} | {"Diabetes_binary"}
    missing -= set(raw.columns)
    if missing:
        raise ValueError(f"Diabetes data is missing expected columns: {sorted(missing)}")
    X = coerce_frame(raw, FEATURES).reset_index(drop=True)
    y = raw["Diabetes_binary"].astype(int).to_numpy()
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("Diabetes_binary must contain only 0 and 1.")
    return Dataset(X=X, y=y)


def load_dataset(raw_dir: Path) -> Dataset:
    path = ensure_source(SOURCE, Path(raw_dir) / "diabetes")
    return clean(pd.read_csv(path))


def subgroups(X: pd.DataFrame) -> dict[str, np.ndarray]:
    age = X["Age"].to_numpy()
    return {
        "Female": (X["Sex"] == 0).to_numpy(),
        "Male": (X["Sex"] == 1).to_numpy(),
        "Age 18-44": age <= 5,
        "Age 45-64": (age >= 6) & (age <= 9),
        "Age 65+": age >= 10,
    }


PIPELINE = DiseasePipeline(
    key="diabetes",
    title="Diabetes",
    outcome_phrase="had reported prediabetes or diabetes",
    target_definition=(
        "Self-reported in a telephone survey: 1 means prediabetes or diabetes, 0 means neither. "
        "Diabetes only during pregnancy counts as 0. No blood tests were involved."
    ),
    population=(
        "Adults in the United States who answered the CDC's 2015 BRFSS telephone survey and "
        "whose answers were complete after the dataset curator's cleaning."
    ),
    dataset_name="CDC Diabetes Health Indicators (derived from BRFSS 2015)",
    dataset_url="https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators",
    license_name="CC0 1.0 as labelled by the Kaggle curator; UCI lists no license of its own",
    license_url="https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset",
    citation=(
        "CDC Diabetes Health Indicators [Dataset]. (2017). UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C53919. Source survey: CDC Behavioral Risk Factor "
        "Surveillance System, 2015. Cleaned dataset curated by Alex Teboul on Kaggle."
    ),
    limitations=(
        "Answers are self-reported, so undiagnosed diabetes is counted as 'no diabetes'.",
        "This is one survey at one point in time. The model estimates current reported status, "
        "not the chance of developing diabetes in the future.",
        "Some inputs, such as poor general health or difficulty walking, can be consequences of "
        "diabetes rather than causes.",
        "Survey weights are not in this file, so the dataset's rate is not a US prevalence "
        "estimate. Results may not apply outside the US or to groups underrepresented in it.",
        "Sex was recorded with only two options. Type 1 and type 2 diabetes are not separated.",
    ),
    features=FEATURES,
    excluded_features={
        "Income, Education": "Left out to collect less personal information and to avoid proxies "
        "for social and economic status.",
        "AnyHealthcare, NoDocbcCost": "Left out because they describe access to care rather than "
        "a person's health.",
    },
    sources=(SOURCE,),
    load_dataset=load_dataset,
    subgroups=subgroups,
    urgent_note=(
        "Very high thirst, frequent urination, vomiting, confusion or rapid breathing can need "
        "urgent care. Contact a clinician or local emergency services rather than using this tool."
    ),
    cv_folds=3,
    cv_repeats=1,
    n_bootstrap=300,
    result_caution=(
        "This reflects reported status in one 2015 US survey. It is not a blood test and not a "
        "forecast of future diabetes."
    ),
)
