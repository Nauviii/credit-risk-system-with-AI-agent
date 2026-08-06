"""Tests for explainability: SHAP aggregation, direction detection, and points scaling."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from credit_risk.explainability.attribution import (
    compute_shap_values,
    points_range,
    rank_agreement,
    scorecard_points,
    shap_direction_report,
    shap_global_importance,
)


@pytest.fixture(scope="module")
def fitted():
    """A GBM where one feature raises risk, one lowers it, and one is pure noise."""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    n = 4_000
    frame = pd.DataFrame(
        {
            "raises_risk": rng.normal(size=n),
            "lowers_risk": rng.normal(size=n),
            "noise": rng.normal(size=n),
        }
    )
    logit = -2.0 + 1.5 * frame["raises_risk"] - 1.5 * frame["lowers_risk"]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = lgb.train(
        {"objective": "binary", "verbosity": -1, "seed": 0, "num_leaves": 15},
        lgb.Dataset(frame, label=y),
        num_boost_round=60,
    )
    return model, frame


def test_shap_importance_ranks_signal_above_noise(fitted):
    model, frame = fitted
    values, used = compute_shap_values(model, frame, sample_size=2_000)
    table = shap_global_importance(values, used)
    assert table["feature"][0] in ("raises_risk", "lowers_risk")
    assert table["feature"][-1] == "noise"
    assert table["share"].sum() == pytest.approx(1.0)


def test_direction_report_recovers_the_sign_of_each_driver(fitted):
    model, frame = fitted
    values, used = compute_shap_values(model, frame, sample_size=2_000)
    direction = dict(
        zip(*shap_direction_report(values, used)[["feature", "raises_risk"]], strict=True)
    )
    assert direction["raises_risk"] is True
    assert direction["lowers_risk"] is False


def test_direction_report_skips_categoricals_rather_than_guessing():
    frame = pd.DataFrame({"cat": pd.Categorical(["a", "b"] * 100), "num": np.arange(200.0)})
    values = np.random.default_rng(0).normal(size=(200, 2))
    report = shap_direction_report(values, frame)
    assert report.filter(pl.col("feature") == "cat")["direction"][0] is None
    assert report.filter(pl.col("feature") == "num")["direction"][0] is not None


def test_rank_agreement_is_one_for_identical_orderings():
    a = pl.DataFrame({"feature": ["x", "y", "z", "w"]})
    assert rank_agreement(a, a)["spearman"] == pytest.approx(1.0)
    reversed_order = pl.DataFrame({"feature": ["w", "z", "y", "x"]})
    assert rank_agreement(a, reversed_order)["spearman"] == pytest.approx(-1.0)


def _encoder_and_model():
    """Minimal fitted scorecard: one feature, two bins, known coefficient."""
    from sklearn.linear_model import LogisticRegression

    from credit_risk.features.woe import WOEEncoder

    rng = np.random.default_rng(1)
    n = 4_000
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.0 + 1.2 * x)))).astype(int)
    df = pl.DataFrame({"x": x, "default_flag": y})
    encoder = WOEEncoder(["x"]).fit(df)
    woe = encoder.transform(df)
    model = LogisticRegression(max_iter=1000).fit(
        woe.select(["x_woe"]).to_pandas(), woe["default_flag"].to_pandas()
    )
    return encoder, model


def test_safer_bins_earn_more_points_than_riskier_ones():
    encoder, model = _encoder_and_model()
    table = scorecard_points(encoder, model, ["x"]).sort("bad_rate")
    assert table["points"].to_list() == sorted(table["points"].to_list(), reverse=True)


def test_points_reproduce_the_pdo_scale():
    """A one-unit change in WOE must move points by factor * |coefficient|."""
    encoder, model = _encoder_and_model()
    table = scorecard_points(encoder, model, ["x"], pdo=20)
    factor = 20 / np.log(2)
    delta_points = table["points"].max() - table["points"].min()
    delta_woe = table["woe"].max() - table["woe"].min()
    assert delta_points == pytest.approx(factor * abs(model.coef_[0][0]) * delta_woe, abs=0.2)


def test_points_range_ranks_by_swing_not_by_coefficient():
    encoder, model = _encoder_and_model()
    ranges = points_range(scorecard_points(encoder, model, ["x"]))
    assert ranges["swing"][0] > 0
    assert ranges["n_bins"][0] == encoder.bin_tables_["x"].height
