import numpy as np
import pandas as pd
import pytest

from chronic_risk import evaluation, modeling
from chronic_risk.pipelines import HEART
from chronic_risk.train import train_pipeline
from tests.synthetic import synthetic_heart


class StubModel:
    """Returns the 'p' column as its probability, so calibration can be tested exactly."""

    def predict_proba(self, X):
        p = X["p"].to_numpy()
        return np.column_stack([1 - p, p])


def _simulated(transform, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.02, 0.98, n)
    y = (rng.random(n) < truth).astype(int)
    return pd.DataFrame({"p": transform(truth)}), y, np.arange(n)


def test_known_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    result = evaluation.score_with_ci(y, p, n_bootstrap=50, seed=0)
    assert result["metrics"]["roc_auc"] == 1.0
    assert result["metrics"]["brier"] == pytest.approx(0.025)
    assert result["observed_rate"] == 0.5


def test_zero_bootstraps_gives_no_interval_rather_than_failing():
    result = evaluation.score_with_ci(np.array([0, 1]), np.array([0.2, 0.8]), 0, 0)
    assert result["ci95"]["roc_auc"] is None


def test_calibration_table_covers_every_row_and_interval_holds_the_rate():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=1200)
    y = (rng.random(1200) < p).astype(int)
    table = evaluation.calibration_table(y, p)
    assert len(table) == 10 and sum(row["n"] for row in table) == 1200
    assert all(r["observed_low"] <= r["observed"] <= r["observed_high"] for r in table)
    assert len(evaluation.calibration_table(y[:100], p[:100])) == 5


def test_small_subgroups_are_not_scored():
    y = np.array([1] * 5 + [0] * 50)
    p = np.linspace(0, 1, 55)
    row = evaluation.subgroup_table(y, p, {"small": np.ones(55, bool)})[0]
    assert row["roc_auc"] is None and row["brier"] is None and row["n"] == 55


def test_cutoffs_are_the_labelled_illustrative_set_and_consistent():
    y = np.array([0, 0, 1, 1])
    table = evaluation.cutoff_table(y, np.array([0.1, 0.6, 0.4, 0.9]))
    assert [r["cutoff"] for r in table] == list(evaluation.ILLUSTRATIVE_CUTOFFS)
    at_half = next(r for r in table if r["cutoff"] == 0.5)
    assert at_half["sensitivity"] == 0.5 and at_half["specificity"] == 0.5


def test_too_little_data_blocks_calibration():
    assert not modeling.calibration_eligible(np.array([1] * 199 + [0] * 5000))
    assert not modeling.calibration_eligible(np.array([1] * 5000 + [0] * 199))
    assert modeling.calibration_eligible(np.array([1] * 200 + [0] * 200))


def test_small_dataset_trains_without_calibration():
    _, report = train_pipeline(HEART, synthetic_heart(n=300), seed=0)
    assert report["calibration"]["eligible"] is False
    assert report["calibration"]["adopted"] is False
    assert report["calibration"]["method"] is None
    assert report["counts"]["fit"] == report["counts"]["development"]


def test_badly_calibrated_model_gets_calibrated():
    X, y, groups = _simulated(lambda t: t**3)
    decision = modeling.decide_calibration(StubModel(), X, y, groups, seed=0, n_bootstrap=100)
    assert decision["adopted"] is True
    assert decision["brier_improvement_ci95"][0] > 0


def test_well_calibrated_model_is_left_alone():
    X, y, groups = _simulated(lambda t: t)
    decision = modeling.decide_calibration(StubModel(), X, y, groups, seed=0, n_bootstrap=100)
    assert decision["adopted"] is False
