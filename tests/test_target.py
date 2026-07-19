"""Tests for censoring exclusion and default_flag construction."""

import polars as pl
from credit_risk.data.target import build_target


def test_build_target_strips_legacy_prefix():
    df = pl.DataFrame({"loan_status": ["Does not meet the credit policy. Status:Fully Paid"]})
    result = build_target(df)
    assert result["loan_status"][0] == "Fully Paid"


def test_build_target_drops_censored_and_flags_default():
    df = pl.DataFrame({
        "loan_status": ["Fully Paid", "Charged Off", "Current", "Late (31-120 days)"],
    })
    result = build_target(df)
    assert result.height == 2
    assert result["default_flag"].to_list() == [0, 1]