"""Tests for build_dataset.py - split boundaries are exact-date sensitive, worth locking down."""

import polars as pl
from credit_risk.features.build_dataset import tag_split, gbm_features, SCORECARD_FEATURES


def test_split_boundaries_are_inclusive_and_exclusive_correctly():
    df = pl.DataFrame({"issue_d": ["Dec-2014", "Jan-2015", "Dec-2015", "Jan-2016", "Dec-2016", "Jan-2017"]})
    config = {
        "train_start": "2013-01", "train_end": "2014-12", "validation_start": "2015-01", "validation_end": "2015-12",
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

def test_target_timing_and_platform_columns_never_become_features():
    """Regression guard: label-timing and platform columns must stay out of any feature list."""
    from credit_risk.data.schema import PLATFORM_COLUMNS, TARGET_TIMING_COLUMNS, LEAKAGE_COLUMNS

    banned = set(PLATFORM_COLUMNS + TARGET_TIMING_COLUMNS + LEAKAGE_COLUMNS)
    df = pl.DataFrame({c: [0] for c in list(banned) + ["annual_inc", "dti"]})
    assert set(gbm_features(df)).isdisjoint(banned)
    assert set(SCORECARD_FEATURES).isdisjoint(banned)


def test_application_features_excludes_lender_derived_columns():
    """The deployable candidate set must not contain LendingClub's own risk output."""
    from credit_risk.data.schema import LENDER_DERIVED_COLUMNS
    from credit_risk.features.build_dataset import application_features

    df = pl.DataFrame({c: [0] for c in LENDER_DERIVED_COLUMNS + ["annual_inc", "dti"]})
    assert set(application_features(df)) == {"annual_inc", "dti"}
    assert set(gbm_features(df)) > set(application_features(df))


def test_application_feature_set_carries_no_lender_derived_column():
    """The two derived lists must differ exactly by LendingClub's own risk output."""
    from credit_risk.data.schema import LENDER_DERIVED_COLUMNS
    from credit_risk.features.build_dataset import APPLICATION_FEATURES

    assert set(APPLICATION_FEATURES).isdisjoint(LENDER_DERIVED_COLUMNS)
    assert set(SCORECARD_FEATURES) & set(LENDER_DERIVED_COLUMNS) == {"sub_grade"}
    assert "term_months" in APPLICATION_FEATURES  # dropped from the full list as a suppressor