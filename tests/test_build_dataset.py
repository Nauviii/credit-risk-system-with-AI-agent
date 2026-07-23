"""Tests for build_dataset.py - split boundaries are exact-date sensitive, worth locking down."""

import polars as pl
from credit_risk.features.build_dataset import tag_split, gbm_features, SCORECARD_FEATURES


def test_split_boundaries_are_inclusive_and_exclusive_correctly():
    df = pl.DataFrame({"issue_d": ["Dec-2014", "Jan-2015", "Dec-2015", "Jan-2016", "Dec-2016", "Jan-2017"]})
    config = {
        "train_end": "2014-12", "validation_start": "2015-01", "validation_end": "2015-12",
        "oot_test_start": "2016-01", "oot_test_end": "2016-12",
    }
    result = tag_split(df, config)
    assert result["split"].to_list() == ["train", "validation", "validation", "oot_test", "oot_test", "excluded"]


def test_gbm_features_excludes_leakage_and_non_feature_columns():
    df = pl.DataFrame({
        "loan_amnt": [1.0], "recoveries": [0.0], "id": ["1"],
        "default_flag": [0], "split": ["train"],
    })
    cols = gbm_features(df)
    assert "loan_amnt" in cols
    assert "recoveries" not in cols
    assert "id" not in cols
    assert "default_flag" not in cols


def test_scorecard_features_has_no_duplicates():
    assert len(SCORECARD_FEATURES) == len(set(SCORECARD_FEATURES))