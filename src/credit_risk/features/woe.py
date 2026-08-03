"""Weight of Evidence and Information Value for feature screening and interpretable binning.

IV rule of thumb (standard credit scoring convention): <0.02 not useful,
0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious - re-check for leakage.
"""

import pandas as pd
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


class WOEEncoder:
    """Fit WOE bins/values on train only, then apply the frozen mapping anywhere else.

    Convention: WOE = ln(dist_good / dist_bad). Higher WOE means safer. When used
    as input to a model predicting P(default=1), well-specified coefficients come
    out NEGATIVE (higher WOE -> lower predicted default probability) - this is
    correct, not a sign-flip bug. Verified with a single clean feature (no
    multicollinearity possible) before trusting this in tests/test_woe.py.

    Prevents the classic scorecard bug: recomputing bins per-dataset makes train and
    OOT test (or serving) use different, incomparable encodings of the same feature.
    Bin edges are computed once in fit() and reused via cut() in both fit and
    transform, so labels always match exactly (qcut vs cut recompute edges
    independently and can disagree at floating-point precision otherwise).
    """

    def __init__(self, features: list[str], target: str = "default_flag", n_bins: int = 10):
        self.features = features
        self.target = target
        self.n_bins = n_bins
        self.bin_edges_: dict[str, list[float]] = {}
        self.woe_maps_: dict[str, dict[str, float]] = {}

    def _bin_column(self, df: pl.DataFrame, feature: str) -> pl.Expr:
        """Bin feature using this encoder's stored edges (numeric) or as-is (categorical)."""
        is_numeric = df.schema[feature] in _NUMERIC_DTYPES
        if is_numeric:
            edges = self.bin_edges_[feature]
            binned = pl.col(feature).cut(edges).cast(pl.Utf8) if edges else pl.lit("single_value")
            return pl.when(pl.col(feature).is_null()).then(pl.lit("missing")).otherwise(binned)
        return pl.col(feature).fill_null("missing")

    def fit(self, df: pl.DataFrame) -> "WOEEncoder":
        """Learn bin edges (numeric) and per-bin WOE values from df - call on train only."""
        for feature in self.features:
            is_numeric = df.schema[feature] in _NUMERIC_DTYPES
            if is_numeric:
                non_null = df[feature].drop_nulls()
                qs = [i / self.n_bins for i in range(1, self.n_bins)]
                self.bin_edges_[feature] = sorted(set(non_null.quantile(q) for q in qs))

            working = df.select([feature, self.target]).filter(pl.col(self.target).is_not_null())
            binned = working.with_columns(self._bin_column(working, feature).alias("_bin"))
            total_good = (binned[self.target] == 0).sum()
            total_bad = (binned[self.target] == 1).sum()
            grouped = binned.group_by("_bin").agg(
                (pl.col(self.target) == 0).sum().alias("n_good"),
                (pl.col(self.target) == 1).sum().alias("n_bad"),
            )
            grouped = grouped.with_columns(
                (((pl.col("n_good") / total_good) + _EPS) / ((pl.col("n_bad") / total_bad) + _EPS))
                .log().alias("woe")
            )
            self.woe_maps_[feature] = dict(zip(grouped["_bin"].cast(pl.Utf8), grouped["woe"]))
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the frozen bin edges/WOE map from fit() - safe on train, OOT test, or serving data.

        Categories/bins unseen during fit (e.g. a new state, or a bin only present
        in OOT test) fall back to WOE 0.0 - neutral, not an error, but worth
        monitoring: a high rate of fallback hits signals population drift.
        """
        out = df
        for feature in self.features:
            binned = self._bin_column(df, feature)
            woe_map = self.woe_maps_[feature]
            out = out.with_columns(
                binned.replace_strict(woe_map, default=0.0, return_dtype=pl.Float64).alias(f"{feature}_woe")
            )
        return out


def prune_correlated_features(features_by_priority: list[str], corr: pd.DataFrame, threshold: float = 0.6) -> list[str]:
    """Greedily keep each feature (in priority order, e.g. IV descending) unless it
    correlates above threshold with an already-kept one - avoids near-duplicate
    features destabilizing a logistic regression (e.g. grade/sub_grade/int_rate)."""
    kept: list[str] = []
    for f in features_by_priority:
        if all(abs(corr.loc[f, k]) < threshold for k in kept):
            kept.append(f)
    return kept