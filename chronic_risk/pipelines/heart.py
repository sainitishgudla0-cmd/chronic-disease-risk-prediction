"""Heart disease pipeline: UCI Heart Disease, Cleveland development data, other sites external."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chronic_risk.data import ensure_source
from chronic_risk.schema import Dataset, DataSource, DiseasePipeline, Feature, coerce_frame

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num",
]
YES_NO = (("yes", "Yes"), ("no", "No"))
EXAM = "Exam and lab results"
STRESS = "Exercise stress test"
IMAGING = "Specialist imaging"

FEATURES = (
    Feature("age", "Age", "Age in years. This tool is for adults.", "numeric", "About you",
            required=True, min_value=18, max_value=100, unit="years"),
    Feature("sex", "Sex", "As recorded in the dataset, which has only these two options.",
            "categorical", "About you", required=True,
            choices=(("female", "Female"), ("male", "Male"))),
    Feature(
        "cp", "Chest pain type",
        "Your clinician's description. 'No chest pain' does not lower the estimate here, because "
        "everyone in this data had already been referred for a heart investigation.",
        "categorical", "About you", required=True,
        choices=(
            ("typical", "Typical angina"),
            ("atypical", "Atypical angina"),
            ("non_anginal", "Non-anginal pain"),
            ("asymptomatic", "No chest pain (asymptomatic)"),
        ),
    ),
    Feature("trestbps", "Resting systolic blood pressure", "The top number of a blood pressure "
            "reading at rest.", "numeric", EXAM, min_value=60, max_value=260, unit="mm Hg"),
    Feature("chol", "Serum cholesterol", "Total cholesterol from a blood test, in mg/dL. "
            "To convert from mmol/L, multiply by 38.7.", "numeric", EXAM,
            min_value=80, max_value=700, unit="mg/dL"),
    Feature("fbs", "Fasting blood sugar above 120 mg/dL?", "From a fasting blood test.",
            "categorical", EXAM, choices=YES_NO),
    Feature(
        "restecg", "Resting ECG result", "Find this in your ECG report, if you have one.",
        "categorical", EXAM,
        choices=(
            ("normal", "Normal"),
            ("st_t_abnormality", "ST-T wave abnormality"),
            ("lv_hypertrophy", "Left ventricular hypertrophy"),
        ),
    ),
    Feature("thalach", "Maximum heart rate reached", "Highest rate during an exercise test.",
            "numeric", STRESS, min_value=50, max_value=240, unit="bpm"),
    Feature("exang", "Chest pain caused by exercise?", "Angina during the exercise test.",
            "categorical", STRESS, choices=YES_NO),
    Feature("oldpeak", "ST depression on exercise",
            "A measurement from the exercise ECG, relative to rest.", "numeric", STRESS,
            min_value=-3, max_value=10, step=0.1, unit="mm"),
    Feature(
        "slope", "Slope of the peak exercise ST segment", "From the exercise ECG report.",
        "categorical", STRESS,
        choices=(("up", "Upsloping"), ("flat", "Flat"), ("down", "Downsloping")),
    ),
    Feature("ca", "Major vessels seen on fluoroscopy", "Number of vessels (0-3) colored by "
            "fluoroscopy, from a specialist procedure.", "numeric", IMAGING,
            choices=((0, "0"), (1, "1"), (2, "2"), (3, "3"))),
    Feature(
        "thal", "Thallium stress test result", "From a nuclear imaging report.",
        "categorical", IMAGING,
        choices=(
            ("normal", "Normal"),
            ("fixed_defect", "Fixed defect"),
            ("reversible_defect", "Reversible defect"),
        ),
    ),
)

SOURCE = DataSource(
    url="https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
    filename="heart_disease.zip",
    sha256="b17cd273da9ce1caa4710fce80227ea454d4dbf9fcbc8e6a9121672751563adc",
)
EXTERNAL_MEMBERS = {
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Decode documented category codes; treat impossible zero readings as missing."""
    frame = raw.apply(pd.to_numeric, errors="coerce")
    # Some sites used 0 as a placeholder: a cholesterol or systolic reading of 0 is impossible.
    for column in ("chol", "trestbps"):
        frame[column] = frame[column].where(frame[column] > 0)
    decoded = pd.DataFrame(
        {
            "age": frame["age"],
            "sex": frame["sex"].map({0: "female", 1: "male"}),
            "cp": frame["cp"].map(
                {1: "typical", 2: "atypical", 3: "non_anginal", 4: "asymptomatic"}
            ),
            "trestbps": frame["trestbps"],
            "chol": frame["chol"],
            "fbs": frame["fbs"].map({0: "no", 1: "yes"}),
            "restecg": frame["restecg"].map(
                {0: "normal", 1: "st_t_abnormality", 2: "lv_hypertrophy"}
            ),
            "thalach": frame["thalach"],
            "exang": frame["exang"].map({0: "no", 1: "yes"}),
            "oldpeak": frame["oldpeak"],
            "slope": frame["slope"].map({1: "up", 2: "flat", 3: "down"}),
            "ca": frame["ca"],
            "thal": frame["thal"].map({3: "normal", 6: "fixed_defect", 7: "reversible_defect"}),
        }
    )
    target = frame["num"]
    keep = target.notna().to_numpy()
    X = coerce_frame(decoded.loc[keep], FEATURES).reset_index(drop=True)
    return X, (target[keep] > 0).astype(int).to_numpy()


def _read_member(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    text = archive.read(member).decode("latin-1")
    return pd.read_csv(
        io.StringIO(text), header=None, names=COLUMNS, na_values="?", skipinitialspace=True
    )


def load_dataset(raw_dir: Path) -> Dataset:
    path = ensure_source(SOURCE, Path(raw_dir) / "heart")
    # Only named members are read, in memory. Nothing from the archive is extracted to disk.
    with zipfile.ZipFile(path) as archive:
        X, y = clean(_read_member(archive, "processed.cleveland.data"))
        external = {
            site: clean(_read_member(archive, member))
            for site, member in EXTERNAL_MEMBERS.items()
        }
    return Dataset(X=X, y=y, external=external)


def subgroups(X: pd.DataFrame) -> dict[str, np.ndarray]:
    age = X["age"].to_numpy()
    return {
        "Female": (X["sex"] == "female").to_numpy(),
        "Male": (X["sex"] == "male").to_numpy(),
        "Age under 55": age < 55,
        "Age 55+": age >= 55,
    }


def diagnostics(*, X_dev: pd.DataFrame, y_dev: np.ndarray, **_: Any) -> dict[str, Any]:
    """Show why the chest pain input behaves unexpectedly in this referred population."""
    rates = []
    labels = dict(next(f.choices for f in FEATURES if f.name == "cp"))
    for value, label in labels.items():
        mask = (X_dev["cp"] == value).to_numpy()
        rates.append(
            {"chest_pain_type": label, "n": int(mask.sum()), "rate": float(y_dev[mask].mean())}
        )
    return {"positive_rate_by_chest_pain": rates}


CAUTION = (
    "This only describes people already referred for angiography in the 1980s. If you have not "
    "been referred, it does not apply to you, whatever the number."
)

PIPELINE = DiseasePipeline(
    key="heart",
    title="Heart disease",
    outcome_phrase="had coronary narrowing above 50% on angiography",
    target_definition=(
        "Angiographic result: 1 means more than 50% diameter narrowing in at least one major "
        "vessel (source values 1-4), 0 means no such narrowing (source value 0)."
    ),
    population=(
        "Adults referred for coronary angiography at the Cleveland Clinic, in data collected in "
        "the 1980s. Hungary, Switzerland and VA Long Beach records are used only for outside "
        "testing."
    ),
    dataset_name="UCI Heart Disease",
    dataset_url="https://archive.ics.uci.edu/dataset/45/heart+disease",
    license_name="CC BY 4.0",
    license_url="https://creativecommons.org/licenses/by/4.0/",
    citation=(
        "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease "
        "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X."
    ),
    limitations=(
        "Everyone had been referred for angiography, so the dataset's rate is far above that of "
        "the general population. Estimates do not apply to people who have not been referred.",
        "The Cleveland development data has only 303 records, so results are imprecise.",
        "The data dates from the 1980s, before many current tests and treatments, and most "
        "participants are male.",
        "Several inputs come from specialist tests. When they are left blank the model fills in "
        "a typical training value, which makes the estimate less informative.",
        "This describes a past angiogram result among similar patients. It does not predict a "
        "future heart attack or other event.",
    ),
    features=FEATURES,
    excluded_features={},
    sources=(SOURCE,),
    load_dataset=load_dataset,
    subgroups=subgroups,
    urgent_note=(
        "Chest pain, pressure, shortness of breath or fainting can be an emergency. Call local "
        "emergency services now rather than using this tool."
    ),
    diagnostics=diagnostics,
    result_caution=CAUTION,
)
