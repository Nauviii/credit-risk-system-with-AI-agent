"""Tests for WOE/IV correctness using synthetic data with a known ground truth."""

import numpy as np
import polars as pl
from credit_risk.features.woe import information_value, rank_features_by_iv


def _synthetic_df() -> pl.DataFrame:
    """A feature that perfectly separates classes and one that is pure random noise."""
    rng = np.random.default_rng(42)
    n = 2000
    target = rng.integers(0, 2, n)
    separating = np.where(target == 1, 1, 0)  # identical to target -> maximal separation
    noise = rng.integers(0, 2, n)  # independent of target
    return pl.DataFrame({"separating": separating, "noise": noise, "default_flag": target})


def test_separating_feature_has_higher_iv_than_noise():
    df = _synthetic_df()
    assert information_value(df, "separating") > information_value(df, "noise")


def test_rank_features_labels_noise_as_not_useful():
    df = _synthetic_df()
    ranked = rank_features_by_iv(df, ["separating", "noise"])
    noise_row = ranked.filter(pl.col("feature") == "noise")
    assert noise_row["strength"][0] == "not useful"