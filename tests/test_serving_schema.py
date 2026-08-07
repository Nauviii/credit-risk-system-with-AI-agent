"""Tests for the request contract: derived from the bundle, never hand-listed."""

import polars as pl
import pytest

from credit_risk.serving.schema import build_frame, required_raw_fields


def test_derived_features_ask_for_their_sources_not_themselves():
    required = required_raw_fields(["term_months", "credit_history_months", "annual_inc"])
    assert "term" in required and "term_months" not in required
    assert "earliest_cr_line" in required and "credit_history_months" not in required
    assert "annual_inc" in required


def test_has_flags_ask_for_the_underlying_column():
    required = required_raw_fields(["has_mths_since_last_delinq"])
    assert "mths_since_last_delinq" in required
    assert "has_mths_since_last_delinq" not in required


def test_absent_optional_fields_become_null_not_an_error():
    required = required_raw_fields(["annual_inc", "dti"])
    frame = build_frame({"annual_inc": 50_000.0}, required)
    assert frame.height == 1
    assert frame["dti"][0] is None
    assert frame["annual_inc"][0] == 50_000.0


def test_string_fields_keep_utf8_dtype_when_null():
    """A null date column typed Null would break strptime in clean_features."""
    frame = build_frame({}, required_raw_fields(["term_months"]))
    assert frame.schema["term"] == pl.Utf8
    assert frame.schema["earliest_cr_line"] == pl.Utf8


def test_pipeline_produces_the_requested_features_end_to_end():
    from credit_risk.features.cleaning import clean_features

    features = ["term_months", "credit_history_months", "annual_inc", "has_mths_since_last_delinq"]
    payload = {
        "term": " 36 months",
        "earliest_cr_line": "Jan-2005",
        "issue_d": "Jan-2016",
        "annual_inc": 65_000.0,
        "mths_since_last_delinq": None,
    }
    cleaned = clean_features(build_frame(payload, required_raw_fields(features)))
    assert set(features).issubset(cleaned.columns)
    assert cleaned["term_months"][0] == 36
    assert cleaned["credit_history_months"][0] == 132
    assert cleaned["has_mths_since_last_delinq"][0] is False


def test_sentinel_and_winsorize_rules_apply_at_serving_time():
    """The same cleaning the training path applies, or the score is not comparable."""
    from credit_risk.features.cleaning import clean_features

    required = required_raw_fields(["dti", "annual_inc"])
    payload = {
        "dti": 999.0,
        "annual_inc": 61_000_000.0,
        "term": " 36 months",
        "earliest_cr_line": "Jan-2005",
        "issue_d": "Jan-2016",
    }
    cleaned = clean_features(build_frame(payload, required))
    assert cleaned["dti"][0] is None
    assert cleaned["annual_inc"][0] == pytest.approx(350_000.0)
