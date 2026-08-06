"""Tests for vintage diagnostics: censoring bias, feature availability, hazard timing."""

import polars as pl

from credit_risk.data.diagnostics import (
    censoring_by_vintage,
    charge_off_proxy_check,
    default_hazard_by_mob,
    feature_availability,
    first_reliable_year,
    prepayment_risk_link,
)


def _frame() -> pl.DataFrame:
    """Two fully matured vintages plus one heavily censored vintage."""
    return pl.DataFrame(
        {
            "issue_d": ["Jan-2012", "Jan-2012", "Jan-2013", "Jun-2016", "Jun-2016", "Jun-2016"],
            "loan_status": [
                "Fully Paid",
                "Does not meet the credit policy. Status:Charged Off",
                "Charged Off",
                "Current",
                "Charged Off",
                "Fully Paid",
            ],
            "last_pymnt_d": ["Dec-2014", "Sep-2012", "Mar-2014", None, "Feb-2017", "Aug-2017"],
            "term": [
                " 36 months",
                " 36 months",
                " 60 months",
                " 36 months",
                " 36 months",
                " 36 months",
            ],
            "fico_range_low": [700.0, 660.0, 680.0, 690.0, 665.0, 730.0],
            "dti": [10.0, 22.0, 18.0, 15.0, 25.0, 8.0],
            "mort_acc": [None, None, 2.0, 1.0, 0.0, 3.0],
        }
    )


def test_legacy_prefix_statuses_count_as_defaults():
    row = censoring_by_vintage(_frame()).filter(pl.col("issue_year") == 2012)
    assert row["n_default"][0] == 1


def test_censored_vintage_shows_bias_gap_matured_does_not():
    result = censoring_by_vintage(_frame())
    mature = result.filter(pl.col("issue_year") == 2012)
    censored = result.filter(pl.col("issue_year") == 2016)
    assert mature["bias_gap"][0] == 0.0
    assert censored["dr_observed"][0] > censored["dr_lower_bnd"][0]


def test_feature_availability_flags_vintage_dependent_column():
    avail = feature_availability(_frame(), ["mort_acc", "fico_range_low", "not_a_column"])
    assert avail.filter(pl.col("issue_year") == 2012)["mort_acc"][0] == 1.0
    assert avail.filter(pl.col("issue_year") == 2016)["mort_acc"][0] == 0.0
    assert "not_a_column" not in avail.columns


def test_first_reliable_year_picks_earliest_year_below_threshold():
    result = first_reliable_year(feature_availability(_frame(), ["mort_acc", "fico_range_low"]))
    lookup = dict(zip(result["feature"], result["first_year"], strict=True))
    assert lookup["fico_range_low"] == 2012
    assert lookup["mort_acc"] == 2013


def test_hazard_cumulative_share_reaches_one_per_term():
    result = default_hazard_by_mob(_frame(), vintages=[2012, 2013])
    assert result["mob"].min() > 0
    assert result.group_by("term_months").agg(pl.col("cum_share").max())["cum_share"].to_list() == [
        1.0,
        1.0,
    ]


def test_prepayment_split_excludes_defaults_and_censored():
    result = prepayment_risk_link(_frame(), vintages=[2012, 2013])
    assert result["n"].sum() == 1  # only the 2012 Fully Paid loan qualifies


def _payment_frame() -> pl.DataFrame:
    """Two charged-off loans that stopped paying at the same point; one has later recoveries."""
    return pl.DataFrame(
        {
            "issue_d": ["Jan-2013", "Jan-2013", "Jan-2013"],
            "loan_status": ["Charged Off", "Charged Off", "Fully Paid"],
            "last_pymnt_d": ["Mar-2014", "Nov-2014", "Dec-2015"],
            "installment": [300.0, 300.0, 300.0],
            "total_pymnt": [3000.0, 3600.0, 10800.0],
            "recoveries": [0.0, 600.0, 0.0],
        }
    )


def test_proxy_check_excludes_non_defaults():
    assert charge_off_proxy_check(_payment_frame(), [2013])["n"].sum() == 2


def test_proxy_check_detects_recovery_contamination():
    result = charge_off_proxy_check(_payment_frame(), [2013])
    gap = dict(zip(result["has_recovery"], result["med_gap"], strict=True))
    assert gap[True] > gap[False] + 3.0
