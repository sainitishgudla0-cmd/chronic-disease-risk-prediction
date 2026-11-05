"""Fully artificial data for tests. No test relies on a real dataset being present."""

import numpy as np
import pandas as pd

from chronic_risk.pipelines import HEART
from chronic_risk.schema import Dataset, coerce_frame


def synthetic_heart(n: int = 400, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    frame = {}
    for feature in HEART.features:
        if feature.choices:
            options = [value for value, _ in feature.choices]
            frame[feature.name] = rng.choice(np.array(options, dtype=object), n)
        else:
            frame[feature.name] = rng.uniform(feature.min_value, feature.max_value, n)
    X = pd.DataFrame(frame)
    X.loc[rng.random(n) < 0.1, "chol"] = np.nan
    X.loc[rng.random(n) < 0.1, "thal"] = np.nan
    signal = (X["age"].to_numpy(float) - 60) / 20 + (X["cp"] == "asymptomatic").to_numpy() * 1.5
    y = (signal + rng.normal(0, 1, n) > 0).astype(int)
    return Dataset(X=coerce_frame(X, HEART.features), y=y)
