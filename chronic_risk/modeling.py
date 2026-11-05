"""Leakage-safe splitting, candidate models and calibration decisions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from chronic_risk.schema import Feature

# Project rule, informed by Collins et al. (2016), who recommend at least 100 and ideally 200
# events for validating a model. Post-hoc calibration needs both classes well represented, so
# it is only attempted with 200 of each in a dedicated split that no other step has used.
MIN_CALIBRATION_PER_CLASS = 200
CALIBRATION_FRACTION = 0.25
PROBABILITY_EPS = 1e-6


def row_groups(X: pd.DataFrame) -> np.ndarray:
    """Give identical feature rows one group ID so they can never straddle a split."""
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def stratified_group_split(
    y: np.ndarray, groups: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices so groups stay whole and class balance is roughly preserved."""
    table = pd.DataFrame({"group": groups, "y": y})
    # Round a group's mean label: a group with mixed labels takes its majority class.
    by_group = table.groupby("group")["y"].mean().round().astype(int)
    train_groups, test_groups = train_test_split(
        by_group.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=by_group.to_numpy(),
    )
    train_index = np.flatnonzero(np.isin(groups, train_groups))
    test_index = np.flatnonzero(np.isin(groups, test_groups))
    return train_index, test_index


def build_preprocessor(features: tuple[Feature, ...]) -> ColumnTransformer:
    """Imputers, scaler and encoder. They are only ever fitted inside a Pipeline.fit call."""
    numeric = [f.name for f in features if f.kind == "numeric"]
    categorical = [f.name for f in features if f.kind == "categorical"]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def candidate_models(features: tuple[Feature, ...], seed: int) -> dict[str, Pipeline]:
    """Candidates ordered from simplest to most flexible; the order breaks near-ties."""
    estimators: dict[str, BaseEstimator] = {
        "Logistic regression (stronger regularization)": LogisticRegression(C=0.1, max_iter=2000),
        "Logistic regression (weaker regularization)": LogisticRegression(C=1.0, max_iter=2000),
        "Gradient boosting (shallow)": HistGradientBoostingClassifier(
            max_depth=2, learning_rate=0.05, max_iter=150, early_stopping=False, random_state=seed
        ),
        "Gradient boosting (deeper)": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=250, early_stopping=False, random_state=seed
        ),
    }
    return {
        name: Pipeline([("preprocess", build_preprocessor(features)), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def default_full_model(features: tuple[Feature, ...]) -> Pipeline:
    """Fixed logistic model for diagnostic comparisons, separate from model selection."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(features)),
            ("model", LogisticRegression(C=1.0, max_iter=2000)),
        ]
    )


def compare_candidates(
    candidates: dict[str, Pipeline],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    repeats: int,
    seed: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Grouped cross-validation on development rows, with a one-standard-error simplicity rule."""
    splits = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        splits.extend(splitter.split(X, y, groups))

    rows: list[dict[str, Any]] = []
    for name, candidate in candidates.items():
        losses, briers, aucs = [], [], []
        for train_index, valid_index in splits:
            fitted = clone(candidate).fit(X.iloc[train_index], y[train_index])
            scores = fitted.predict_proba(X.iloc[valid_index])[:, 1]
            losses.append(log_loss(y[valid_index], scores, labels=[0, 1]))
            briers.append(brier_score_loss(y[valid_index], scores))
            aucs.append(roc_auc_score(y[valid_index], scores))
        rows.append(
            {
                "model": name,
                "log_loss": float(np.mean(losses)),
                "log_loss_se": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
                "brier": float(np.mean(briers)),
                "roc_auc": float(np.mean(aucs)),
            }
        )

    best = min(rows, key=lambda row: row["log_loss"])
    limit = best["log_loss"] + best["log_loss_se"]
    # rows keeps simplest-first order, so this is the simplest model within one SE of the best.
    selected = next(row["model"] for row in rows if row["log_loss"] <= limit)
    for row in rows:
        row["selected"] = row["model"] == selected
    return selected, rows


def calibration_eligible(y_cal: np.ndarray) -> bool:
    return (
        int(y_cal.sum()) >= MIN_CALIBRATION_PER_CLASS
        and int((1 - y_cal).sum()) >= MIN_CALIBRATION_PER_CLASS
    )


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, PROBABILITY_EPS, 1 - PROBABILITY_EPS)
    return np.log(p / (1 - p))


def decide_calibration(
    model: Pipeline,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    groups_cal: np.ndarray,
    seed: int,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    """Adopt a calibrator only if it clearly lowers Brier score out of fold on calibration rows.

    The test set plays no part in this decision.
    """
    raw = model.predict_proba(X_cal)[:, 1]
    oof = {"sigmoid": np.zeros_like(raw), "isotonic": np.zeros_like(raw)}
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, valid_index in splitter.split(X_cal, y_cal, groups_cal):
        sigmoid = LogisticRegression(C=1e6, max_iter=1000)
        sigmoid.fit(_logit(raw[train_index]).reshape(-1, 1), y_cal[train_index])
        oof["sigmoid"][valid_index] = sigmoid.predict_proba(
            _logit(raw[valid_index]).reshape(-1, 1)
        )[:, 1]
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        isotonic.fit(raw[train_index], y_cal[train_index])
        oof["isotonic"][valid_index] = isotonic.predict(raw[valid_index])

    raw_brier = float(brier_score_loss(y_cal, raw))
    briers = {name: float(brier_score_loss(y_cal, p)) for name, p in oof.items()}
    method = min(briers, key=briers.get)

    rng = np.random.default_rng(seed)
    improvements = np.empty(n_bootstrap)
    raw_error = (raw - y_cal) ** 2
    calibrated_error = (oof[method] - y_cal) ** 2
    for i in range(n_bootstrap):
        sample = rng.integers(0, len(y_cal), len(y_cal))
        improvements[i] = raw_error[sample].mean() - calibrated_error[sample].mean()
    low, high = np.percentile(improvements, [2.5, 97.5])

    return {
        "raw_brier_out_of_fold": raw_brier,
        "calibrated_brier_out_of_fold": briers,
        "best_method": method,
        "brier_improvement_ci95": [float(low), float(high)],
        "adopted": bool(low > 0),
    }


def fit_calibrator(
    model: Pipeline, X_cal: pd.DataFrame, y_cal: np.ndarray, method: str
) -> CalibratedClassifierCV:
    """Calibrate a frozen model so its fitted preprocessing and weights stay untouched."""
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_cal, y_cal)
