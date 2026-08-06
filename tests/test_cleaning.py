"""Tests for cleaning.py - each test locks in one decision from docs/eda_findings.md."""

import polars as pl

from credit_risk.features.cleaning import (
    add_has_history_flags,
    clean_sentinels,
    derive_credit_history_months,
    parse_term_months,
    winsorize,
)


def test_clean_sentinels_nulls_dti_999():
    df = pl.DataFrame({"dti": [15.0, 999.0, 22.5]})
    result = clean_sentinels(df)
    assert result["dti"].to_list() == [15.0, None, 22.5]


def test_winsorize_caps_annual_inc_at_350k():
    df = pl.DataFrame({"annual_inc": [50_000.0, 61_000_000.0]})
    result = winsorize(df)
    assert result["annual_inc"].to_list() == [50_000.0, 350_000.0]


def test_parse_term_months():
    df = pl.DataFrame({"term": [" 36 months", " 60 months"]})
    result = parse_term_months(df)
    assert result["term_months"].to_list() == [36, 60]


def test_credit_history_months_computed_correctly():
    df = pl.DataFrame({"issue_d": ["Dec-2015"], "earliest_cr_line": ["Jan-2000"]})
    result = derive_credit_history_months(df)
    assert result["credit_history_months"][0] == 191


def test_has_history_flag_false_when_null():
    df = pl.DataFrame({"mths_since_last_delinq": [12.0, None]})
    result = add_has_history_flags(df.select("mths_since_last_delinq"))
    assert result["has_mths_since_last_delinq"].to_list() == [True, False]
