"""Interpretable WOE + logistic regression scorecard - the champion-challenger baseline.

Any complexity added by the GBM champion (models/gbm.py) must earn its place by
beating this on the OOT test set, not just on train.
"""

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from credit_risk.features.build_dataset import SCORECARD_FEATURES
from credit_risk.features.woe import WOEEncoder


def train_scorecard(
    train_df: pl.DataFrame, target: str = "default_flag", n_bins: int = 10
) -> tuple[LogisticRegression, WOEEncoder]:
    """Fit WOEEncoder on train, then fit LogisticRegression on the WOE-encoded features."""
    encoder = WOEEncoder(features=SCORECARD_FEATURES, target=target, n_bins=n_bins).fit(train_df)
    train_woe = encoder.transform(train_df)
    woe_cols = [f"{f}_woe" for f in SCORECARD_FEATURES]
    X = train_woe.select(woe_cols).to_pandas()
    y = train_woe[target].to_pandas()
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, encoder


def predict_scorecard(model: LogisticRegression, encoder: WOEEncoder, df: pl.DataFrame) -> np.ndarray:
    """Score a dataframe with a fitted encoder+model pair - returns predicted P(default)."""
    transformed = encoder.transform(df)
    woe_cols = [f"{f}_woe" for f in encoder.features]
    X = transformed.select(woe_cols).to_pandas()
    return model.predict_proba(X)[:, 1]