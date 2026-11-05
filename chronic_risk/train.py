"""Train and evaluate the model for each condition.

Usage: python -m chronic_risk.train [diabetes|heart|kidney|all]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

from chronic_risk import evaluation, modeling
from chronic_risk.data import DataDownloadError
from chronic_risk.paths import DATA_DIR, MANIFEST_PATH, MODELS_DIR
from chronic_risk.pipelines import PIPELINES
from chronic_risk.schema import Dataset, DiseasePipeline

SEED = 42
TEST_FRACTION = 0.2


def _reference_values(pipeline: DiseasePipeline, X: pd.DataFrame) -> dict[str, Any]:
    """Typical training values, used as the comparison point in explanations and examples."""
    values: dict[str, Any] = {}
    for feature in pipeline.features:
        column = X[feature.name].dropna()
        if feature.kind == "numeric":
            median = float(column.median())
            if feature.choices:  # keep the value on one of the offered options
                options = [value for value, _ in feature.choices]
                median = min(options, key=lambda option: abs(option - median))
            values[feature.name] = median
        else:
            values[feature.name] = str(column.mode().iloc[0])
    return values


def _typical_ranges(pipeline: DiseasePipeline, X: pd.DataFrame) -> dict[str, list[float]]:
    """Central 98% of training values. Percentiles avoid storing any one person's extremes."""
    ranges = {}
    for feature in pipeline.features:
        if feature.kind == "numeric" and not feature.choices:
            low, high = np.nanpercentile(X[feature.name].to_numpy(dtype=float), [1, 99])
            ranges[feature.name] = [float(low), float(high)]
    return ranges


def _round(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_round(v) for v in value]
    if isinstance(value, float | np.floating):
        return round(float(value), 5)
    if isinstance(value, np.integer):
        return int(value)
    return value


def train_pipeline(
    pipeline: DiseasePipeline,
    dataset: Dataset,
    seed: int = SEED,
) -> tuple[Any, dict[str, Any]]:
    """Run the full leakage-safe workflow and return the final model and its report."""
    X, y = dataset.X, dataset.y
    groups = modeling.row_groups(X)

    # 1. The test set is set aside first and not touched until step 5.
    dev_index, test_index = modeling.stratified_group_split(y, groups, TEST_FRACTION, seed)
    X_dev, y_dev, groups_dev = X.iloc[dev_index], y[dev_index], groups[dev_index]
    X_test, y_test = X.iloc[test_index], y[test_index]

    # 2. A separate calibration split exists only if it would have enough of both classes.
    fit_local, cal_local = modeling.stratified_group_split(
        y_dev, groups_dev, modeling.CALIBRATION_FRACTION, seed + 1
    )
    eligible = modeling.calibration_eligible(y_dev[cal_local])
    if not eligible:
        fit_local = np.arange(len(y_dev))
    X_fit, y_fit, groups_fit = X_dev.iloc[fit_local], y_dev[fit_local], groups_dev[fit_local]

    # 3. Choose a model with cross-validation on fitting rows only.
    candidates = modeling.candidate_models(pipeline.features, seed)
    selected, comparison = modeling.compare_candidates(
        candidates, X_fit, y_fit, groups_fit, pipeline.cv_folds, pipeline.cv_repeats, seed
    )
    base_model = clone(candidates[selected]).fit(X_fit, y_fit)

    # 4. Decide on calibration using calibration rows only.
    calibration: dict[str, Any] = {
        "eligible": eligible,
        "minimum_per_class": modeling.MIN_CALIBRATION_PER_CLASS,
        "calibration_split_positive": int(y_dev[cal_local].sum()) if eligible else None,
        "calibration_split_negative": int((1 - y_dev[cal_local]).sum()) if eligible else None,
        "adopted": False,
        "method": None,
    }
    model: Any = base_model
    if eligible:
        X_cal, y_cal = X_dev.iloc[cal_local], y_dev[cal_local]
        decision = modeling.decide_calibration(
            base_model, X_cal, y_cal, groups_dev[cal_local], seed
        )
        calibration.update(decision)
        if decision["adopted"]:
            model = modeling.fit_calibrator(base_model, X_cal, y_cal, decision["best_method"])
            calibration["method"] = decision["best_method"]

    # 5. One final evaluation on the untouched test set.
    probabilities = model.predict_proba(X_test)[:, 1]
    baseline = DummyClassifier(strategy="prior").fit(X_fit, y_fit)
    report: dict[str, Any] = {
        "disease": pipeline.key,
        "seed": seed,
        "sklearn_version": sklearn.__version__,
        "selected_model": selected,
        "counts": {
            "total": len(y),
            "unique_feature_rows": len(np.unique(groups)),
            "development": len(dev_index),
            "fit": len(fit_local),
            "test": len(test_index),
            "development_positive_rate": float(y_dev.mean()),
        },
        "candidate_comparison": comparison,
        "cv": {"folds": pipeline.cv_folds, "repeats": pipeline.cv_repeats},
        "calibration": calibration,
        "test": evaluation.score_with_ci(y_test, probabilities, pipeline.n_bootstrap, seed),
        "baseline_test": {
            "brier": float(np.mean((baseline.predict_proba(X_test)[:, 1] - y_test) ** 2)),
            "constant_prediction": float(y_fit.mean()),
        },
        "calibration_table": evaluation.calibration_table(y_test, probabilities),
        "roc": evaluation.roc_points(y_test, probabilities),
        "cutoffs": evaluation.cutoff_table(y_test, probabilities),
        "subgroups": evaluation.subgroup_table(y_test, probabilities, pipeline.subgroups(X_test)),
        "reference_values": _reference_values(pipeline, X_fit),
        "typical_ranges": _typical_ranges(pipeline, X_fit),
        "missing_rate_development": {k: float(v) for k, v in X_dev.isna().mean().items()},
    }
    if model is not base_model:
        report["test_before_calibration"] = evaluation.score_with_ci(
            y_test, base_model.predict_proba(X_test)[:, 1], 0, seed
        )

    # 6. Outside sites, when a dataset has them, were not used in any earlier step.
    if dataset.external:
        report["external"] = _external_validation(pipeline, model, dataset, seed)

    # 7. Disease-specific checks, such as leakage probes.
    if pipeline.diagnostics:
        report["diagnostics"] = pipeline.diagnostics(
            dev_index=dev_index,
            test_index=test_index,
            y=y,
            full_features=dataset.full_features,
            X_dev=X_dev,
            y_dev=y_dev,
            X_test=X_test,
            model=model,
            seed=seed,
            candidate_factory=modeling.default_full_model,
        )
    return model, _round(report)


def _external_validation(
    pipeline: DiseasePipeline, model: Any, dataset: Dataset, seed: int
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    all_y, all_p = [], []
    for site, (X_site, y_site) in dataset.external.items():
        scores = model.predict_proba(X_site)[:, 1]
        all_y.append(y_site)
        all_p.append(scores)
        minority = min(int(y_site.sum()), int((1 - y_site).sum()))
        site_result = evaluation.score_with_ci(y_site, scores, pipeline.n_bootstrap, seed)
        site_result["too_few_for_stable_estimate"] = minority < 20
        site_result["missing_rate"] = {k: float(v) for k, v in X_site.isna().mean().items()}
        results[site] = site_result
    results["All outside sites"] = evaluation.score_with_ci(
        np.concatenate(all_y), np.concatenate(all_p), pipeline.n_bootstrap, seed
    )
    results["All outside sites"]["too_few_for_stable_estimate"] = False
    return results


def save_artifact(key: str, model: Any, report: dict[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / f"{key}.joblib", compress=3)
    (directory / f"{key}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


def write_manifest(directory: Path, manifest_path: Path) -> None:
    """Record checksums and versions so the app can refuse unknown or incompatible files."""
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.joblib"))
    }
    manifest = {"sklearn_version": sklearn.__version__, "sha256": files}
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def _summary(report: dict[str, Any]) -> str:
    test = report["test"]
    auc, auc_ci = test["metrics"]["roc_auc"], test["ci95"]["roc_auc"]
    return (
        f"  selected: {report['selected_model']}\n"
        f"  calibrated: {report['calibration']['adopted']} "
        f"(eligible: {report['calibration']['eligible']})\n"
        f"  test n={test['n']} positive={test['n_positive']} "
        f"ROC AUC={auc:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] "
        f"Brier={test['metrics']['brier']:.3f} "
        f"(constant-rate baseline {report['baseline_test']['brier']:.3f})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("disease", choices=[*PIPELINES, "all"], nargs="?", default="all")
    args = parser.parse_args(argv)
    keys = list(PIPELINES) if args.disease == "all" else [args.disease]
    for key in keys:
        pipeline = PIPELINES[key]
        print(f"[{key}] loading data", flush=True)
        try:
            dataset = pipeline.load_dataset(DATA_DIR)
        except (DataDownloadError, ValueError) as exc:
            print(f"[{key}] ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"[{key}] training on {len(dataset.y):,} rows", flush=True)
        model, report = train_pipeline(pipeline, dataset)
        save_artifact(key, model, report, MODELS_DIR)
        print(_summary(report), flush=True)
    write_manifest(MODELS_DIR, MANIFEST_PATH)
    print(f"Wrote models and manifest to {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
