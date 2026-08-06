"""Assemble the final feature matrix - the single place where cleaning, exclusion
lists, and the OOT split from configs/base.yaml all come together.

Two feature lists reflect the champion-challenger design: SCORECARD_FEATURES is
curated (IV > 0.02, no redundant pairs) for the interpretable logistic baseline;
GBM_FEATURES is broader since the champion model can exploit weak/interacting
signals that a WOE scorecard cannot represent well.
"""

from pathlib import Path

import polars as pl
import yaml

from credit_risk.data.schema import (
    ALWAYS_MISSING_COLUMNS,
    EXCLUDED_VINTAGE_COLUMNS,
    FAIR_LENDING_EXCLUDED_COLUMNS,
    HIGH_CARDINALITY_COLUMNS,
    JOINT_APPLICATION_COLUMNS,
    LEAKAGE_COLUMNS,
    LENDER_DERIVED_COLUMNS,
    PLATFORM_COLUMNS,
    REDUNDANT_OR_CONSTANT_COLUMNS,
    TARGET_TIMING_COLUMNS,
)
from credit_risk.features.cleaning import clean_features

_RAW_COLUMNS_REPLACED_BY_DERIVED = ["term", "earliest_cr_line"]
_NON_FEATURE_COLUMNS = ["id", "loan_status", "issue_d", "issue_date"]

# Derived by notebooks/feature_selection_scorecard.ipynb on the H=24 target, train
# window 2013-2014, and monotonic binning. Rerun that notebook if any of the three
# changes; never re-derive at runtime.
#
# term_months is absent on purpose and it is the most informative absence here. It is
# dropped by drop_until_signs_are_clean at coefficient +0.0115 because sub_grade already
# prices term (LendingClub charges more for 60-month loans), so term carries no
# independent signal once sub_grade is in the model. In APPLICATION_FEATURES below,
# where sub_grade is gone, term_months is the STRONGEST feature at -1.05. The gap
# between the two lists is a direct measurement of how much of this scorecard is really
# LendingClub's scorecard.
SCORECARD_FEATURES = [
    "sub_grade",
    "fico_range_low",
    "acc_open_past_24mths",
    "annual_inc",
    "dti",
    "tot_hi_cred_lim",
    "mo_sin_rcnt_tl",
    "mths_since_recent_inq",
    "mo_sin_old_rev_tl_op",
    "mths_since_recent_bc",
    "home_ownership",
    "purpose",
    "percent_bc_gt_75",
    "verification_status",
]

# Same pipeline, lender-derived columns removed. This is the set whose performance
# transfers to production, since a lender scoring its own applicants has no sub_grade.
APPLICATION_FEATURES = [
    "fico_range_low",
    "acc_open_past_24mths",
    "annual_inc",
    "dti",
    "bc_open_to_buy",
    "tot_hi_cred_lim",
    "mo_sin_rcnt_tl",
    "mths_since_recent_inq",
    "term_months",
    "mo_sin_old_rev_tl_op",
    "mths_since_recent_bc",
    "home_ownership",
    "purpose",
    "percent_bc_gt_75",
    "verification_status",
]


def _excluded_columns() -> set[str]:
    """Every column that must never reach a feature matrix, for any reason."""
    return set(
        LEAKAGE_COLUMNS
        + EXCLUDED_VINTAGE_COLUMNS
        + HIGH_CARDINALITY_COLUMNS
        + FAIR_LENDING_EXCLUDED_COLUMNS
        + ALWAYS_MISSING_COLUMNS
        + JOINT_APPLICATION_COLUMNS
        + REDUNDANT_OR_CONSTANT_COLUMNS
        + PLATFORM_COLUMNS
        + TARGET_TIMING_COLUMNS
        + _RAW_COLUMNS_REPLACED_BY_DERIVED
        + _NON_FEATURE_COLUMNS
        + ["default_flag", "split"]
    )


def gbm_features(df: pl.DataFrame) -> list[str]:
    """All columns not explicitly excluded - the broader candidate set for the GBM champion."""
    return [c for c in df.columns if c not in _excluded_columns()]


def application_features(df: pl.DataFrame) -> list[str]:
    """gbm_features minus LendingClub's own grade/rate output - the deployable candidate set.

    A lender scoring its own applicants has no counterpart to sub_grade or int_rate, so
    this is the only set whose measured performance transfers to production.
    """
    lender_derived = set(LENDER_DERIVED_COLUMNS)
    return [c for c in gbm_features(df) if c not in lender_derived]


def load_split_config(config_path: Path) -> dict:
    """Load the OOT split cutoffs from configs/base.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)["split"]


def tag_split(df: pl.DataFrame, split_config: dict) -> pl.DataFrame:
    """Add a `split` column (train / validation / oot_test / excluded) from issue_d and configured cutoffs."""
    issue_date = pl.col("issue_d").str.strptime(pl.Date, "%b-%Y")
    train_start = pl.lit(split_config["train_start"]).str.strptime(pl.Date, "%Y-%m")
    train_end = pl.lit(split_config["train_end"]).str.strptime(pl.Date, "%Y-%m")
    val_start = pl.lit(split_config["validation_start"]).str.strptime(pl.Date, "%Y-%m")
    val_end = pl.lit(split_config["validation_end"]).str.strptime(pl.Date, "%Y-%m")
    oot_start = pl.lit(split_config["oot_test_start"]).str.strptime(pl.Date, "%Y-%m")
    oot_end = pl.lit(split_config["oot_test_end"]).str.strptime(pl.Date, "%Y-%m")

    return df.with_columns(
        pl.when((issue_date >= train_start) & (issue_date <= train_end))
        .then(pl.lit("train"))
        .when((issue_date >= val_start) & (issue_date <= val_end))
        .then(pl.lit("validation"))
        .when((issue_date >= oot_start) & (issue_date <= oot_end))
        .then(pl.lit("oot_test"))
        .otherwise(pl.lit("excluded"))
        .alias("split")
    )


def assemble_feature_matrix(labeled_df: pl.DataFrame, config_path: Path) -> pl.DataFrame:
    """Full assembly: clean -> tag OOT split -> drop rows outside train/oot_test.

    labeled_df must already be the output of data.target.build_target() (fully
    observable loans, default_flag present).
    """
    split_config = load_split_config(config_path)
    cleaned = clean_features(labeled_df)
    tagged = tag_split(cleaned, split_config)
    return tagged.filter(pl.col("split") != "excluded")
