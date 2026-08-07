"""Clean raw fields per EDA findings before any binning/scaling happens downstream."""

import polars as pl

from credit_risk.data.schema import SENTINEL_VALUES, STRUCTURALLY_MISSING_COLUMNS, WINSORIZE_CAPS


def clean_sentinels(df: pl.DataFrame) -> pl.DataFrame:
    """Replace known placeholder values (e.g. dti==999) with null, per SENTINEL_VALUES.

    Columns absent from df are skipped rather than raising, matching add_has_history_flags.
    Training passes the full raw frame; serving passes only the fields the champion needs,
    and the same cleaning rules must apply to both without a second code path.
    """
    for col, bad_values in SENTINEL_VALUES.items():
        if col not in df.columns:
            continue
        df = df.with_columns(
            pl.when(pl.col(col).is_in(bad_values)).then(None).otherwise(pl.col(col)).alias(col)
        )
    return df


def winsorize(df: pl.DataFrame) -> pl.DataFrame:
    """Cap columns at their documented upper bound (WINSORIZE_CAPS) to blunt isolated outliers.

    Absent columns are skipped, for the same reason as clean_sentinels.
    """
    for col, cap in WINSORIZE_CAPS.items():
        if col not in df.columns:
            continue
        df = df.with_columns(pl.min_horizontal(pl.col(col), pl.lit(cap)).alias(col))
    return df


def parse_term_months(df: pl.DataFrame) -> pl.DataFrame:
    """Convert term ' 36 months' / ' 60 months' to an integer term_months column."""
    if "term" not in df.columns:
        return df
    return df.with_columns(
        pl.col("term").str.extract(r"(\d+)", 1).cast(pl.Int32).alias("term_months")
    )


def derive_credit_history_months(df: pl.DataFrame) -> pl.DataFrame:
    """Derive credit_history_months = months between earliest_cr_line and issue_d.

    Replaces earliest_cr_line as a raw date string, which must never be used
    as a categorical feature (686 near-unique values, no reusable signal as-is).
    """
    if "earliest_cr_line" not in df.columns or "issue_d" not in df.columns:
        return df
    issue_date = pl.col("issue_d").str.strptime(pl.Date, "%b-%Y", strict=False)
    earliest_date = pl.col("earliest_cr_line").str.strptime(pl.Date, "%b-%Y", strict=False)
    months = (issue_date.dt.year() - earliest_date.dt.year()) * 12 + (
        issue_date.dt.month() - earliest_date.dt.month()
    )
    return df.with_columns(months.alias("credit_history_months"))


def add_has_history_flags(df: pl.DataFrame) -> pl.DataFrame:
    """For STRUCTURALLY_MISSING_COLUMNS present in df, add a has_history flag; null means 'never happened'.

    The numeric column itself is left null (not sentinel-filled) - WOE binning treats
    null as its own explicit bin (see features/woe.py), so the "never happened" group
    keeps its own default rate instead of being merged into an arbitrary bucket.
    """
    present = [c for c in STRUCTURALLY_MISSING_COLUMNS if c in df.columns]
    flags = [pl.col(col).is_not_null().alias(f"has_{col}") for col in present]
    return df.with_columns(flags)


def clean_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the full cleaning sequence: sentinels -> winsorize -> term -> credit history -> flags."""
    df = clean_sentinels(df)
    df = winsorize(df)
    df = parse_term_months(df)
    df = derive_credit_history_months(df)
    df = add_has_history_flags(df)
    return df
