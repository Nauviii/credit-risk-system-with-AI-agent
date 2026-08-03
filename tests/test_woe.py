"""Tests for WOE/IV correctness using synthetic data with a known ground truth."""

import numpy as np
import pandas as pd
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


def _train_test_with_shift():
    """Train/test pair with a distribution shift, mimicking a real OOT scenario."""
    rng = np.random.default_rng(0)
    train = pl.DataFrame({
        "num_feat": rng.normal(50, 10, 3000),
        "cat_feat": rng.choice(["A", "B", "C"], 3000),
        "default_flag": rng.binomial(1, 0.2, 3000),
    })
    test = pl.DataFrame({
        "num_feat": rng.normal(70, 10, 1000),  # shifted mean, like OOT test
        "cat_feat": rng.choice(["A", "B", "D"], 1000),  # D unseen in train
        "default_flag": rng.binomial(1, 0.2, 1000),
    })
    return train, test


def test_woe_encoder_reuses_train_bins_on_test():
    from credit_risk.features.woe import WOEEncoder
    train, test = _train_test_with_shift()
    enc = WOEEncoder(features=["num_feat"], n_bins=5).fit(train)
    train_woe = set(enc.transform(train)["num_feat_woe"].unique().to_list())
    test_woe = set(enc.transform(test)["num_feat_woe"].unique().to_list())
    assert test_woe.issubset(train_woe)


def test_woe_encoder_unseen_category_falls_back_to_zero():
    from credit_risk.features.woe import WOEEncoder
    train, test = _train_test_with_shift()
    enc = WOEEncoder(features=["cat_feat"], n_bins=5).fit(train)
    result = enc.transform(test)
    unseen_woe = result.filter(pl.col("cat_feat") == "D")["cat_feat_woe"].unique().to_list()
    assert unseen_woe == [0.0]


def test_prune_correlated_features_keeps_higher_priority_of_a_duplicate_pair():
    from credit_risk.features.woe import prune_correlated_features
    # 'a' and 'b' are near-duplicates (like grade/int_rate); 'c' is independent
    corr = pd.DataFrame(
        [[1.0, 0.95, 0.1], [0.95, 1.0, 0.1], [0.1, 0.1, 1.0]],
        columns=["a", "b", "c"], index=["a", "b", "c"],
    )
    kept = prune_correlated_features(["a", "b", "c"], corr, threshold=0.6)
    assert kept == ["a", "c"]


def test_drop_until_signs_are_clean_removes_redundant_feature_and_converges():
    from credit_risk.evaluation.diagnostics import drop_until_signs_are_clean
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.uniform(0, 100, n)
    x2 = x1 + rng.normal(0, 1, n)  # near-duplicate of x1, correlation not pre-pruned here
    y = rng.binomial(1, np.clip(0.5 - 0.004 * x1, 0.02, 0.98))
    df = pl.DataFrame({"x1": x1, "x2": x2, "default_flag": y})

    kept, encoder, model = drop_until_signs_are_clean(["x1", "x2"], df)
    coefs = dict(zip(kept, model.coef_[0]))
    assert all(c < 0 for c in coefs.values())
    assert len(kept) <= 2