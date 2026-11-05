import numpy as np
import pandas as pd

from chronic_risk import modeling
from chronic_risk.pipelines import HEART
from chronic_risk.train import train_pipeline
from tests.synthetic import synthetic_heart


def test_identical_rows_share_a_group():
    X = pd.DataFrame({"a": [1.0, 1.0, 2.0, np.nan, np.nan], "b": ["x", "x", "x", "y", "y"]})
    groups = modeling.row_groups(X)
    assert groups[0] == groups[1] != groups[2]
    assert groups[3] == groups[4]


def test_split_never_lets_a_group_straddle_and_keeps_class_balance():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 500, 3000)
    y = (groups % 4 == 0).astype(int)
    train, test = modeling.stratified_group_split(y, groups, 0.2, seed=1)
    assert not set(groups[train]) & set(groups[test])
    assert len(train) + len(test) == len(y)
    assert abs(y[train].mean() - y[test].mean()) < 0.03


def test_split_is_reproducible():
    groups = np.arange(200)
    y = groups % 2
    first = modeling.stratified_group_split(y, groups, 0.2, seed=7)
    second = modeling.stratified_group_split(y, groups, 0.2, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def test_preprocessing_statistics_come_only_from_fitting_rows():
    dataset = synthetic_heart(n=400)
    model, report = train_pipeline(HEART, dataset, seed=3)

    groups = modeling.row_groups(dataset.X)
    dev, test = modeling.stratified_group_split(dataset.y, groups, 0.2, seed=3)
    assert report["calibration"]["eligible"] is False  # so all development rows are fitting rows
    numeric = [f.name for f in HEART.features if f.kind == "numeric"]
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert np.allclose(imputer.statistics_, dataset.X.iloc[dev][numeric].median().to_numpy())
    # A sanity check that this test could fail: test rows really have different medians.
    assert not np.allclose(imputer.statistics_, dataset.X.iloc[test][numeric].median().to_numpy())


def test_reports_contain_counts_that_add_up():
    dataset = synthetic_heart(n=400)
    _, report = train_pipeline(HEART, dataset, seed=3)
    counts = report["counts"]
    assert counts["development"] + counts["test"] == counts["total"] == 400
    assert report["test"]["n"] == counts["test"]


def test_model_selection_prefers_simplest_within_one_standard_error():
    X = pd.DataFrame({"age": np.random.default_rng(0).normal(50, 10, 300)})
    y = (X["age"] > 50).astype(int).to_numpy()
    features = tuple(f for f in HEART.features if f.name == "age")
    candidates = modeling.candidate_models(features, seed=0)
    selected, rows = modeling.compare_candidates(
        candidates, X, y, np.arange(300), folds=3, repeats=1, seed=0
    )
    best = min(rows, key=lambda r: r["log_loss"])
    limit = best["log_loss"] + best["log_loss_se"]
    chosen = next(r for r in rows if r["selected"])
    assert chosen["model"] == selected
    assert chosen["log_loss"] <= limit
    assert rows.index(chosen) == next(i for i, r in enumerate(rows) if r["log_loss"] <= limit)
