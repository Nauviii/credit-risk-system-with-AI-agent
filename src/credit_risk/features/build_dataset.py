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
    LEAKAGE_COLUMNS, EXCLUDED_VINTAGE_COLUMNS, HIGH_CARDINALITY_COLUMNS,
    ALWAYS_MISSING_COLUMNS, JOINT_APPLICATION_COLUMNS, REDUNDANT_OR_CONSTANT_COLUMNS,
)
from credit_risk.features.cleaning import clean_features

_RAW_COLUMNS_REPLACED_BY_DERIVED = ["term", "earliest_cr_line"]
_NON_FEATURE_COLUMNS = ["id", "loan_status", "issue_d", "issue_date"]

# IV >= 0.02 in the phase-3/4 screen (docs/eda_findings.md section 8, "weak" or better),
# minus anything in REDUNDANT_OR_CONSTANT_COLUMNS.
SCORECARD_FEATURES = [
    "sub_grade", "grade", "int_rate", "term_months", "fico_range_low",
    "dti", "verification_status", "loan_amnt", "home_ownership", "annual_inc",
    "revol_util", "inq_last_6mths", "purpose",
    "acc_open_past_24mths", "bc_open_to_buy", "avg_cur_bal", "tot_hi_cred_lim",
    "tot_cur_bal", "num_tl_op_past_12m", "total_bc_limit", "mort_acc", "installment",
    "percent_bc_gt_75", "num_actv_rev_tl", "bc_util", "num_rev_tl_bal_gt_0",
    "mo_sin_rcnt_tl", "total_rev_hi_lim", "mo_sin_rcnt_rev_tl_op",
    "mths_since_recent_bc", "mo_sin_old_rev_tl_op",
]


def _excluded_columns() -> set[str]:
    """Every column that must never reach a feature matrix, for any reason."""
    return set(
        LEAKAGE_COLUMNS + EXCLUDED_VINTAGE_COLUMNS + HIGH_CARDINALITY_COLUMNS
        + ALWAYS_MISSING_COLUMNS + JOINT_APPLICATION_COLUMNS + REDUNDANT_OR_CONSTANT_COLUMNS
        + _RAW_COLUMNS_REPLACED_BY_DERIVED + _NON_FEATURE_COLUMNS
        + ["default_flag", "split"]
    )


def gbm_features(df: pl.DataFrame) -> list[str]:
    """All columns not explicitly excluded - the broader candidate set for the GBM champion."""
    return [c for c in df.columns if c not in _excluded_columns()]


def load_split_config(config_path: Path) -> dict:
    """Load the OOT split cutoffs from configs/base.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)["split"]


def tag_split(df: pl.DataFrame, split_config: dict) -> pl.DataFrame:
    """Add a `split` column (train / oot_test / excluded) from issue_d and the configured cutoffs."""
    issue_date = pl.col("issue_d").str.strptime(pl.Date, "%b-%Y")
    train_end = pl.lit(split_config["train_end"]).str.strptime(pl.Date, "%Y-%m")
    oot_start = pl.lit(split_config["oot_test_start"]).str.strptime(pl.Date, "%Y-%m")
    oot_end = pl.lit(split_config["oot_test_end"]).str.strptime(pl.Date, "%Y-%m")

    return df.with_columns(
        pl.when(issue_date <= train_end).then(pl.lit("train"))
        .when((issue_date >= oot_start) & (issue_date <= oot_end)).then(pl.lit("oot_test"))
        .otherwise(pl.lit("excluded"))
        .alias("split")
    )


def assemble_feature_matrix(labeled_df: pl.DataFrame, config_path: Path) -> pl.DataFrame:
    """Full assembly: clean -> tag OOT split -> drop rows outside train/oot_test.

    labeled_df must already be the output of data.target.build_target() (matured
    loans only, default_flag present).
    """
    split_config = load_split_config(config_path)
    cleaned = clean_features(labeled_df)
    tagged = tag_split(cleaned, split_config)
    return tagged.filter(pl.col("split") != "excluded")