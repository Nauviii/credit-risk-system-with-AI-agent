"""Tests for monitoring: distribution drift, and the outcome check PSI cannot perform."""

import numpy as np
import polars as pl
import pytest

from credit_risk.evaluation.stability import (
    outside_reference_range,
    psi_against_reference,
    reference_profile,
)
from credit_risk.monitoring.drift import DriftMonitor
from credit_risk.monitoring.performance import (
    early_warning,
    expected_vs_actual,
    vintage_performance,
)


class _FakeBundle:
    def __init__(self, reference):
        self.reference = reference


def _reference(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "score": reference_profile(pl.Series(rng.normal(550, 30, 20_000))),
        "features": {
            "annual_inc": reference_profile(pl.Series(rng.normal(60_000, 15_000, 20_000)))
        },
    }


def test_frozen_reference_reproduces_a_two_sample_psi():
    rng = np.random.default_rng(1)
    train = pl.Series(rng.normal(size=20_000))
    live = pl.Series(rng.normal(loc=0.8, size=20_000))
    assert psi_against_reference(live, reference_profile(train)) > 0.25


def test_identical_population_scores_stable():
    rng = np.random.default_rng(2)
    monitor = DriftMonitor(_FakeBundle(_reference()))
    assert monitor.score_drift(rng.normal(550, 30, 20_000))["band"] == "stable"


def test_shifted_scores_are_flagged_material():
    rng = np.random.default_rng(3)
    monitor = DriftMonitor(_FakeBundle(_reference()))
    assert monitor.score_drift(rng.normal(510, 30, 20_000))["band"] == "material shift"


def test_a_column_that_stops_arriving_is_reported_not_skipped():
    """The dangerous case: the model scores a missing column as null without complaining."""
    monitor = DriftMonitor(_FakeBundle(_reference()))
    report = monitor.feature_drift(pl.DataFrame({"other": [1.0, 2.0]}))
    assert report["band"].to_list() == ["column missing"]


def test_bundle_without_reference_profiles_is_refused():
    with pytest.raises(ValueError, match="reference profiles"):
        DriftMonitor(_FakeBundle({}))


def test_a_reference_on_the_wrong_quantity_is_named_not_reported_as_drift():
    """The 12.4339 bug: PD-range reference, point-range data, PSI large and meaningless."""
    rng = np.random.default_rng(11)
    pd_reference = reference_profile(pl.Series(rng.beta(2, 20, 20_000)))
    points = pl.Series(rng.normal(550, 30, 20_000))
    assert outside_reference_range(points, pd_reference) > 0.75

    monitor = DriftMonitor(_FakeBundle({"score": pd_reference, "features": {}}))
    result = monitor.score_drift(points)
    assert result["band"] == "reference mismatch"
    assert result["out_of_range"] > 0.75


def test_matching_reference_reports_almost_nothing_out_of_range():
    rng = np.random.default_rng(12)
    monitor = DriftMonitor(_FakeBundle(_reference()))
    assert monitor.score_drift(rng.normal(550, 30, 20_000))["out_of_range"] < 0.05


def test_significant_but_immaterial_deviation_does_not_alert():
    """At 400k rows a 1.4% miss reaches z = -3.3 and is of no interest to anyone."""
    rng = np.random.default_rng(13)
    n = 400_000
    pd_values = rng.uniform(0.05, 0.15, n)
    actual = (rng.random(n) < pd_values * 0.985).astype(int)
    result = expected_vs_actual(pd_values, actual)
    assert result["significant"]
    assert result["ratio_band"] == "stable"
    assert not result["alert"]


def test_matching_outcomes_produce_no_alert():
    rng = np.random.default_rng(4)
    pd_values = rng.uniform(0.03, 0.25, 30_000)
    actual = (rng.random(30_000) < pd_values).astype(int)
    result = expected_vs_actual(pd_values, actual)
    assert not result["alert"]
    assert result["ratio"] == pytest.approx(1.0, abs=0.05)


def test_concept_drift_is_caught_where_distributions_would_look_stable():
    """The 2016 failure in miniature: same inputs, same scores, worse outcomes."""
    rng = np.random.default_rng(5)
    pd_values = rng.uniform(0.03, 0.25, 30_000)
    worse = (rng.random(30_000) < pd_values * 1.3).astype(int)
    result = expected_vs_actual(pd_values, worse)
    assert result["alert"]
    assert result["z_score"] > 2
    assert result["ratio"] > 1.2
    assert result["ratio_band"] == "material"


def test_a_deviation_between_the_two_thresholds_lands_in_watch():
    """The 2016 case: 9.97% off is not "stable" merely because the line was drawn at 10%."""
    rng = np.random.default_rng(15)
    n = 300_000
    pd_values = rng.uniform(0.05, 0.15, n)
    actual = (rng.random(n) < pd_values * 1.08).astype(int)
    result = expected_vs_actual(pd_values, actual)
    assert result["ratio_band"] == "watch"
    assert result["alert"]


def test_vintage_table_isolates_one_bad_cohort_inside_a_stable_book():
    rng = np.random.default_rng(6)
    n = 20_000
    pd_values = rng.uniform(0.05, 0.15, n)
    vintage = np.where(np.arange(n) < n // 2, 2015, 2016)
    multiplier = np.where(vintage == 2016, 1.5, 1.0)
    df = pl.DataFrame(
        {
            "pd": pd_values,
            "vintage": vintage,
            "default_flag": (rng.random(n) < pd_values * multiplier).astype(int),
        }
    )
    table = vintage_performance(df, "pd", "vintage")
    alerts = dict(zip(table["vintage"], table["alert"], strict=True))
    assert alerts["2015"] is False
    assert alerts["2016"] is True


def test_early_warning_scales_the_expectation_by_hazard_coverage():
    """At MOB 6 only a fraction of eventual defaults has arrived; expecting all of them alerts."""
    rng = np.random.default_rng(7)
    n = 20_000
    pd_values = rng.uniform(0.05, 0.20, n)
    ever = rng.random(n) < pd_values
    # a quarter of eventual defaults land by month 6
    mob = np.where(ever & (rng.random(n) < 0.25), 4.0, np.nan)
    df = pl.DataFrame(
        {"pd": pd_values, "mob_event": mob, "default_flag": ever.astype(np.int8)}
    ).with_columns(
        pl.when(pl.col("mob_event").is_nan())
        .then(None)
        .otherwise(pl.col("mob_event"))
        .alias("mob_event")
    )
    correct = early_warning(df, "pd", "mob_event", 6, {6: 0.25}).row(0, named=True)
    naive = early_warning(df, "pd", "mob_event", 6, {6: 1.0}).row(0, named=True)
    assert not correct["alert"]
    assert naive["alert"] and naive["z_score"] < -2  # looks like far fewer defaults than expected


def test_early_warning_rejects_an_unmeasured_month():
    df = pl.DataFrame(
        {"pd": [0.1], "mob_event": [None]}, schema_overrides={"mob_event": pl.Float64}
    )
    with pytest.raises(KeyError, match="MOB 9"):
        early_warning(df, "pd", "mob_event", 9, {6: 0.25})
