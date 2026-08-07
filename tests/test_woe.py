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
    train = pl.DataFrame(
        {
            "num_feat": rng.normal(50, 10, 3000),
            "cat_feat": rng.choice(["A", "B", "C"], 3000),
            "default_flag": rng.binomial(1, 0.2, 3000),
        }
    )
    test = pl.DataFrame(
        {
            "num_feat": rng.normal(70, 10, 1000),  # shifted mean, like OOT test
            "cat_feat": rng.choice(["A", "B", "D"], 1000),  # D unseen in train
            "default_flag": rng.binomial(1, 0.2, 1000),
        }
    )
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
        columns=["a", "b", "c"],
        index=["a", "b", "c"],
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
    coefs = dict(zip(kept, model.coef_[0], strict=True))
    assert all(c < 0 for c in coefs.values())
    assert len(kept) <= 2


def test_zero_bad_bin_woe_is_bounded_by_sample_size_not_by_an_epsilon():
    """A zero-bad bin must land at ln(2*n_bad), not at ln(1/eps).

    Smoothing does not make such a bin harmless - it makes it finite and sample-driven.
    The actual protection is min_bin_bads, which stops the bin existing at all by default;
    it is lowered here to force the degenerate case.
    """
    from credit_risk.features.woe import WOEEncoder

    df = pl.DataFrame(
        {
            "x": [0.0] * 500 + [1.0] * 500,
            "default_flag": [0] * 500 + [0] * 250 + [1] * 250,
        }
    )
    worst = max(abs(v) for v in WOEEncoder(["x"], min_bin_bads=1).fit(df).woe_maps_["x"].values())
    legacy = np.log((500 / 750) / 1e-6)  # what the old proportion-level epsilon produced
    assert worst < legacy / 2
    assert worst < np.log(2 * 250) + 1  # scales with the bad count, not the epsilon
    assert WOEEncoder(["x"]).fit(df).binning_report()["n_bins"][0] == 2


def test_non_monotone_feature_is_merged_until_monotone():
    from credit_risk.features.woe import WOEEncoder

    rng = np.random.default_rng(7)
    n = 40_000
    x = rng.normal(size=n)
    y = (rng.random(n) < np.clip(0.05 + 0.25 * x**2 / 6, 0, 0.9)).astype(int)
    df = pl.DataFrame({"x": x, "default_flag": y})
    report = WOEEncoder(["x"]).fit(df).binning_report()
    assert report["is_monotone"][0]
    assert report["n_bins"][0] >= 2


def test_discrete_feature_does_not_collapse_to_a_single_bin():
    from credit_risk.features.woe import WOEEncoder

    rng = np.random.default_rng(3)
    n = 20_000
    term = rng.choice([36.0, 60.0], n, p=[0.7, 0.3])
    y = (rng.random(n) < np.where(term == 60, 0.25, 0.10)).astype(int)
    df = pl.DataFrame({"term_months": term, "default_flag": y})
    encoder = WOEEncoder(["term_months"]).fit(df)
    assert encoder.binning_report()["n_bins"][0] == 2
    assert information_value(df, "term_months") > 0.05


def test_rare_categories_are_pooled_by_class_counts_and_frozen_for_transform():
    """Pooling is keyed on bad/good counts, not population share."""
    from credit_risk.features.woe import WOEEncoder

    df = pl.DataFrame(
        {
            "state": ["A"] * 500 + ["B"] * 500 + ["Y"] * 40 + ["Z"] * 40,
            "default_flag": ([0, 1] * 250) + ([0, 1] * 250) + ([0, 1] * 20) + ([0, 1] * 20),
        }
    )
    encoder = WOEEncoder(["state"]).fit(df)
    assert encoder.rare_categories_["state"] == {"Y", "Z"}

    unseen = pl.DataFrame({"state": ["Y", "Q"], "default_flag": [0, 0]})
    transformed = encoder.transform(unseen)
    assert transformed["state_woe"][0] == encoder.woe_maps_["state"]["rare"]
    assert transformed["state_woe"][1] == 0.0


def test_many_level_ordinal_survives_pooling():
    """A 35-level ordinal must not be pooled away; its IV must exceed its coarser parent.

    sub_grade (35 levels) refines grade (7 levels), so a finer partition can only add
    information. IV(sub_grade) < IV(grade) is mathematically impossible and was the
    signature of the population-share pooling bug.
    """
    from credit_risk.features.woe import WOEEncoder

    rng = np.random.default_rng(1)
    n = 60_000
    level = rng.integers(0, 35, n)
    labels = np.array([f"{chr(65 + i // 5)}{i % 5 + 1}" for i in range(35)])
    sub_grade = labels[level]
    df = pl.DataFrame(
        {
            "sub_grade": sub_grade,
            "grade": np.array([s[0] for s in sub_grade]),
            "default_flag": (rng.random(n) < np.clip(0.02 + 0.007 * level, 0, 0.6)).astype(int),
        }
    )
    assert len(WOEEncoder(["sub_grade"]).fit(df).woe_maps_["sub_grade"]) >= 30
    assert information_value(df, "sub_grade") > information_value(df, "grade")


def test_thin_missing_bin_is_neutralised_not_given_an_extreme_woe():
    from credit_risk.features.woe import WOEEncoder

    rng = np.random.default_rng(11)
    n = 20_000
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.2 + 0.9 * x)))).astype(int)
    x[:8] = np.nan  # a handful of nulls, all of them bad
    df = pl.DataFrame({"x": x, "default_flag": y}).with_columns(
        pl.when(pl.col("x").is_nan()).then(None).otherwise(pl.col("x")).alias("x")
    )
    encoder = WOEEncoder(["x"]).fit(df)
    assert encoder.woe_maps_["x"]["missing"] == 0.0
    assert encoder.binning_report()["n_neutralised"][0] == 1


def test_monotonicity_is_judged_on_numeric_bin_order_not_label_text():
    """Bin labels sort "(12.3," before "(5.2," as strings; ordering must not rely on that."""
    from credit_risk.features.woe import WOEEncoder

    rng = np.random.default_rng(5)
    n = 60_000
    income = rng.lognormal(mean=11.0, sigma=0.6, size=n)  # spans 4-6 digits
    y = (rng.random(n) < np.clip(0.30 - 0.000_0015 * income, 0.01, 0.5)).astype(int)
    df = pl.DataFrame({"annual_inc": income, "default_flag": y})
    report = WOEEncoder(["annual_inc"]).fit(df).binning_report()
    assert report["is_numeric"][0]
    assert report["is_monotone"][0]
