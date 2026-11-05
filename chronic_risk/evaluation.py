"""Evaluation metrics. Everything returned is an aggregate, never a row of data."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

ILLUSTRATIVE_CUTOFFS = (0.1, 0.25, 0.5, 0.75, 0.9)
MIN_CLASS_FOR_SUBGROUP = 10
EPS = 1e-6
Z_95 = 1.959964


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float | None:
    """Slope of outcome on logit(prediction). Around 1 is good; undefined for constant scores."""
    clipped = np.clip(p, EPS, 1 - EPS)
    logit = np.log(clipped / (1 - clipped))
    if np.std(logit) < 1e-9:
        return None
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), y)
    return float(fit.coef_[0, 0])


def _point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
    }


def score_with_ci(y: np.ndarray, p: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, Any]:
    """Point estimates, percentile bootstrap intervals, and overall calibration summaries."""
    point = _point_metrics(y, p)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {key: [] for key in point}
    n = len(y)
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, n)
        if y[index].min() == y[index].max():
            continue
        for key, value in _point_metrics(y[index], p[index]).items():
            samples[key].append(value)
    intervals = {
        key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None
        for key, v in samples.items()
    }
    observed = float(y.mean())
    return {
        "n": int(n),
        "n_positive": int(y.sum()),
        "observed_rate": observed,
        "mean_predicted": float(p.mean()),
        "calibration_slope": calibration_slope(y, p),
        "brier_of_constant_rate": observed * (1 - observed),
        "metrics": point,
        "ci95": intervals,
        "bootstrap_replicates": n_bootstrap,
    }


def _wilson(positive: int, n: int) -> tuple[float, float]:
    rate = positive / n
    denominator = 1 + Z_95**2 / n
    centre = (rate + Z_95**2 / (2 * n)) / denominator
    half = Z_95 * np.sqrt(rate * (1 - rate) / n + Z_95**2 / (4 * n**2)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def calibration_table(y: np.ndarray, p: np.ndarray) -> list[dict[str, float]]:
    """Equal-count bins: ten with plenty of data, five when the test set is small."""
    bins = 10 if len(y) >= 1000 else 5
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, bins):
        positives, n = int(y[chunk].sum()), len(chunk)
        low, high = _wilson(positives, n)
        rows.append(
            {
                "n": n,
                "mean_predicted": float(p[chunk].mean()),
                "observed": positives / n,
                "observed_low": low,
                "observed_high": high,
            }
        )
    return rows


def roc_points(y: np.ndarray, p: np.ndarray, max_points: int = 60) -> list[dict[str, float]]:
    fpr, tpr, _ = roc_curve(y, p)
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return [{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr, strict=True)]


def cutoff_table(y: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    """Trade-offs at illustrative cutoffs. None of these cutoffs is medically established."""
    rows = []
    positives, negatives = y == 1, y == 0
    for cutoff in ILLUSTRATIVE_CUTOFFS:
        flagged = p >= cutoff
        tp = int((flagged & positives).sum())
        fp = int((flagged & negatives).sum())
        rows.append(
            {
                "cutoff": cutoff,
                "share_flagged": float(flagged.mean()),
                "sensitivity": tp / max(int(positives.sum()), 1),
                "specificity": int((~flagged & negatives).sum()) / max(int(negatives.sum()), 1),
                "ppv": tp / (tp + fp) if tp + fp else None,
            }
        )
    return rows


def subgroup_table(
    y: np.ndarray, p: np.ndarray, masks: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = []
    for name, mask in masks.items():
        y_group, p_group = y[mask], p[mask]
        n_positive, n = int(y_group.sum()), int(mask.sum())
        enough = min(n_positive, n - n_positive) >= MIN_CLASS_FOR_SUBGROUP
        rows.append(
            {
                "group": name,
                "n": n,
                "n_positive": n_positive,
                "observed_rate": n_positive / n if n else None,
                "mean_predicted": float(p_group.mean()) if n else None,
                "roc_auc": float(roc_auc_score(y_group, p_group)) if enough else None,
                "brier": float(brier_score_loss(y_group, p_group)) if enough else None,
            }
        )
    return rows
