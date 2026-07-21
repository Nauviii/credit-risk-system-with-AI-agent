"""Weight of Evidence and Information Value for feature screening and interpretable binning.

IV rule of thumb (standard credit scoring convention): <0.02 not useful,
0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious - re-check for leakage.
"""

import polars as pl

_NUMERIC_DTYPES = (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
_EPS = 1e-6


def woe_iv_table(df: pl.DataFrame, feature: str, target: str = "default_flag", n_bins: int = 10) -> pl.DataFrame:
    """Return per-bin WOE/IV for a feature; numeric features are quantile-binned, categorical used as-is.

    Nulls form their own explicit "missing" bin rather than being dropped, since
    missingness is often informative (see STRUCTURALLY_MISSING_COLUMNS in schema.py).
    """
    working = df.select([feature, target]).filter(pl.col(target).is_not_null())
    is_numeric = working.schema[feature] in _NUMERIC_DTYPES

    if is_numeric:
        non_null = working.filter(pl.col(feature).is_not_null())
        null_part = working.filter(pl.col(feature).is_null())
        binned = non_null.with_columns(
            pl.col(feature).qcut(n_bins, allow_duplicates=True).cast(pl.Utf8).alias("_bin")
        ).select(["_bin", target])
        if null_part.height > 0:
            null_part = null_part.select(pl.lit("missing").alias("_bin"), target)
            working = pl.concat([binned, null_part])
        else:
            working = binned
        group_col = "_bin"
    else:
        working = working.with_columns(pl.col(feature).fill_null("missing"))
        group_col = feature

    total_good = (working[target] == 0).sum()
    total_bad = (working[target] == 1).sum()

    grouped = working.group_by(group_col).agg(
        pl.len().alias("n"),
        (pl.col(target) == 0).sum().alias("n_good"),
        (pl.col(target) == 1).sum().alias("n_bad"),
    )
    grouped = grouped.with_columns(
        (pl.col("n_good") / total_good).alias("dist_good"),
        (pl.col("n_bad") / total_bad).alias("dist_bad"),
    )
    grouped = grouped.with_columns(
        ((pl.col("dist_good") + _EPS) / (pl.col("dist_bad") + _EPS)).log().alias("woe")
    )
    grouped = grouped.with_columns(
        ((pl.col("dist_good") - pl.col("dist_bad")) * pl.col("woe")).alias("iv_contribution")
    )
    return grouped.sort(group_col)


def information_value(df: pl.DataFrame, feature: str, target: str = "default_flag", n_bins: int = 10) -> float:
    """Total IV for a feature - the standard first-pass screening metric before deeper feature engineering."""
    return float(woe_iv_table(df, feature, target, n_bins)["iv_contribution"].sum())


def rank_features_by_iv(
    df: pl.DataFrame, features: list[str], target: str = "default_flag", n_bins: int = 10
) -> pl.DataFrame:
    """Rank candidate features by IV, flagging weak (<0.02) and suspiciously strong (>0.5) ones."""
    rows = []
    for f in features:
        try:
            iv = information_value(df, f, target, n_bins)
        except Exception as e:
            iv = None
            print(f"skipped {f}: {e}")
        rows.append({"feature": f, "iv": iv})
    result = pl.DataFrame(rows).sort("iv", descending=True, nulls_last=True)
    return result.with_columns(
        pl.when(pl.col("iv") > 0.5).then(pl.lit("suspicious - check leakage"))
        .when(pl.col("iv") > 0.3).then(pl.lit("strong"))
        .when(pl.col("iv") > 0.1).then(pl.lit("medium"))
        .when(pl.col("iv") > 0.02).then(pl.lit("weak"))
        .otherwise(pl.lit("not useful"))
        .alias("strength")
    )