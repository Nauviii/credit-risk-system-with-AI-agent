"""Tests for reference baselines: single-feature AUC and per-vintage decomposition."""

import numpy as np
import polars as pl

from credit_risk.evaluation.baselines import (
    auc_by_vintage,
    reference_baseline_table,
    score_only_auc,
)


def _frame() -> pl.DataFrame:
    """Two vintages where sub_grade ranks risk perfectly within each year."""
    return pl.DataFrame({
        "issue_d": ["Jan-2013"] * 4 + ["Jan-2014"] * 4,
        "sub_grade": ["A1", "B2", "C3", "D4"] * 2,
        "int_rate": [5.0, 10.0, 15.0, 20.0] * 2,
        "default_flag": [0, 0, 1, 1, 0, 1, 1, 1],
    })


def test_ordinal_string_feature_is_ranked_in_risk_order():
    single_vintage = _frame().filter(pl.col("issue_d") == "Jan-2013")
    assert score_only_auc(single_vintage, "sub_grade")["auc"] == 1.0
    # pooled across two vintages the same ordering is no longer perfect - the
    # aggregation effect auc_by_vintage exists to separate out
    assert score_only_auc(_frame(), "sub_grade")["auc"] < 1.0


def test_coverage_reported_when_feature_has_nulls():
    df = _frame().with_columns(
        pl.when(pl.col("sub_grade") == "A1").then(None).otherwise(pl.col("int_rate")).alias("int_rate")
    )
    result = score_only_auc(df, "int_rate")
    assert result["n"] == 6
    assert result["coverage"] == 0.75


def test_missing_columns_are_skipped_not_fabricated():
    table = reference_baseline_table(_frame(), ["sub_grade", "int_rate", "not_a_column"])
    assert table["feature"].to_list() == ["sub_grade", "int_rate"]


def test_auc_by_vintage_returns_each_year_plus_pooled_row():
    score = np.array([1.0, 2.0, 3.0, 4.0] * 2)
    result = auc_by_vintage(_frame(), score)
    assert result["issue_year"].to_list() == ["2013", "2014", None]
    assert result.filter(pl.col("issue_year").is_not_null())["auc"].to_list() == [1.0, 1.0]


def test_pooled_auc_falls_below_per_year_auc_when_base_rates_differ():
    """The aggregation artefact this function exists to expose."""
    result = auc_by_vintage(_frame(), np.array([1.0, 2.0, 3.0, 4.0] * 2))
    pooled = result.filter(pl.col("issue_year").is_null())["auc"][0]
    assert pooled < 1.0


def test_inverse_risk_feature_is_oriented_above_half():
    """fico_range_low falls with risk; reported AUC must be oriented, not left below 0.5."""
    df = pl.DataFrame({
        "issue_d": ["Jan-2013"] * 4,
        "fico_range_low": [800.0, 750.0, 650.0, 600.0],
        "default_flag": [0, 0, 1, 1],
    })
    result = score_only_auc(df, "fico_range_low")
    assert result["raw_auc"] == 0.0
    assert result["auc"] == 1.0
    assert result["risk_increases_with_value"] is False


def test_auc_by_segment_splits_on_an_arbitrary_column():
    from credit_risk.evaluation.baselines import auc_by_segment

    df = pl.DataFrame({
        "term_months": ["36", "36", "36", "36", "60", "60", "60", "60"],
        "default_flag": [0, 0, 1, 1, 0, 1, 1, 1],
    })
    result = auc_by_segment(df, np.array([1.0, 2.0, 3.0, 4.0] * 2), "term_months")
    assert result["segment"].to_list() == ["36", "60", None]
    assert result.filter(pl.col("segment") == "36")["auc"][0] == 1.0