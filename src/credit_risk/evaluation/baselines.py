"""Reference baselines that bound how much discrimination the model actually adds.

Two questions a PD model on Lending Club data cannot leave unanswered:

1. `grade`, `sub_grade` and `int_rate` are LendingClub's OWN risk assessment, not raw
   applicant data. A model built on top of them inherits their discrimination for free.
   score_only_auc measures each one used directly as a score, with no model fitted, so
   every reported model AUC can be read against that floor.

2. A pooled AUC over several vintages is not the same quantity as a single-vintage AUC:
   mixing populations with different base rates depresses the pooled figure even when
   within-vintage discrimination is unchanged. auc_by_vintage separates the two, which
   matters whenever a train AUC computed over several years is compared against a
   validation AUC computed over one.
"""

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

_TARGET = "default_flag"


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    """AUC, or None when the slice has only one class."""
    return float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None


def score_only_auc(df: pl.DataFrame, feature: str) -> dict:
    """AUC of one ordinal or continuous column used directly as a score, no model fitted.

    String columns are ranked lexicographically, which is the correct risk order for
    Lending Club's sub_grade (A1 < A2 < ... < G5) and grade. Rows where the feature is
    null are excluded, so `coverage` must be read alongside the AUC. `auc` is oriented
    so 0.5 is always the no-signal floor; `raw_auc` keeps the unoriented value.
    """
    subset = df.select(feature, _TARGET).drop_nulls()
    if subset.height == 0:
        return {
            "feature": feature,
            "n": 0,
            "coverage": 0.0,
            "auc": None,
            "raw_auc": None,
            "risk_increases_with_value": None,
        }

    column = subset[feature]
    score = (column.rank("dense") if column.dtype == pl.Utf8 else column).to_numpy()
    raw = _auc(subset[_TARGET].to_numpy(), score.astype(float))
    # A feature whose risk DECREASES with its value (fico_range_low) scores below 0.5 when
    # read directly. Reporting that number as-is understates it and looks like a data fault,
    # so orient it and record which way the feature points.
    oriented = max(raw, 1.0 - raw) if raw is not None else None
    return {
        "feature": feature,
        "n": subset.height,
        "coverage": subset.height / df.height,
        "auc": oriented,
        "raw_auc": raw,
        "risk_increases_with_value": None if raw is None else bool(raw >= 0.5),
    }


def reference_baseline_table(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """score_only_auc for several features at once, skipping ones absent from df."""
    rows = [score_only_auc(df, f) for f in features if f in df.columns]
    return pl.DataFrame(rows).sort("auc", descending=True, nulls_last=True)


def auc_by_segment(df: pl.DataFrame, score: np.ndarray, segment: str) -> pl.DataFrame:
    """AUC, size and default rate per segment value, plus a pooled row (segment = null).

    Two distinct uses. By issue year, a pooled AUC below every per-year AUC is an
    aggregation artefact of mixing base rates rather than poor within-year performance.
    By term, it tests whether one model can serve both populations: under a fixed 24-month
    horizon the target captures ~60% of eventual 36-month defaults but only ~42% of
    60-month ones, so the label does not mean the same thing on each side.
    """
    frame = df.select(pl.col(segment).alias("_segment"), pl.col(_TARGET)).with_columns(
        pl.Series("score", score)
    )
    rows = [
        {
            "segment": value,
            "n": part.height,
            "default_rate": float(part[_TARGET].mean()),
            "auc": _auc(part[_TARGET].to_numpy(), part["score"].to_numpy()),
        }
        for value in sorted(frame["_segment"].unique().drop_nulls().to_list())
        for part in [frame.filter(pl.col("_segment") == value)]
    ]
    rows.append(
        {
            "segment": None,
            "n": frame.height,
            "default_rate": float(frame[_TARGET].mean()),
            "auc": _auc(frame[_TARGET].to_numpy(), frame["score"].to_numpy()),
        }
    )
    return pl.DataFrame(
        rows,
        schema={"segment": pl.Utf8, "n": pl.Int64, "default_rate": pl.Float64, "auc": pl.Float64},
    )


def auc_by_vintage(df: pl.DataFrame, score: np.ndarray) -> pl.DataFrame:
    """AUC and default rate per issue year, plus the pooled figure for comparison."""
    with_year = df.with_columns(
        pl.col("issue_d")
        .cast(pl.Utf8)
        .str.strptime(pl.Date, "%b-%Y", strict=False)
        .dt.year()
        .cast(pl.Utf8)
        .alias("_issue_year")
    )
    return auc_by_segment(with_year, score, "_issue_year").rename({"segment": "issue_year"})
