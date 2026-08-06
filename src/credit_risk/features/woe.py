"""Weight of Evidence and Information Value for feature screening and interpretable binning.

Three binning rules, all standard scorecard practice, all enforced here rather than
assumed (Siddiqi, Credit Risk Scorecards, 2017, ch. 6):

1. Smoothing at COUNT level, not on distributions. The previous version added a 1e-6
   epsilon to dist_good/dist_bad, so a bin holding zero bads produced WOE up to +13 -
   an order of magnitude beyond the -1.5..+1.5 range real bins occupy - which then
   dominated the logistic fit and inflated that feature's IV. Haldane-Anscombe
   correction (+0.5 per count) bounds the value at something interpretable instead.

2. Monotonic bad rate across ordered bins. A non-monotone numeric feature makes the
   resulting scorecard unexplainable to a reviewer and unstable out of time. Adjacent
   bins violating the feature's dominant direction are merged until monotone.

3. Minimum bin size, by both population share and absolute bad count. Quantile edges
   on discrete features silently collapse (term_months takes 2 values; mort_acc and
   num_rev_tl_bal_gt_0 are small integers), leaving bins with too few bads to estimate
   a stable WOE. binning_report() exposes what each feature actually ended up with.

The "missing" bin is never merged into the monotone sequence: missingness has no
position in the feature's ordering, and for STRUCTURALLY_MISSING_COLUMNS it carries
its own meaning.

IV rule of thumb: <0.02 not useful, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong,
>0.5 suspicious - re-check for leakage rather than celebrating.
"""

import numpy as np
import pandas as pd
import polars as pl

_NUMERIC_DTYPES = (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
_MISSING = "missing"
_RARE = "rare"
_SMOOTH = 0.5

DEFAULT_INITIAL_BINS = 20
DEFAULT_MIN_BIN_FRACTION = 0.05
DEFAULT_MIN_BIN_BADS = 30


def _quantile_edges(values: pl.Series, n_bins: int) -> list[float]:
    """Starting cut points before any merging; duplicates collapse on discrete features."""
    qs = [i / n_bins for i in range(1, n_bins)]
    edges = {values.quantile(q) for q in qs}
    return sorted(e for e in edges if e is not None)


def _aggregate(indices: np.ndarray, target: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Total and bad counts per ordered bin index."""
    n = np.bincount(indices, minlength=n_groups).astype(float)
    bad = np.bincount(indices, weights=target, minlength=n_groups).astype(float)
    return n, bad


def _dominant_direction(n: np.ndarray, bad: np.ndarray) -> int:
    """+1 if bad rate rises with the feature, -1 if it falls - fixes the monotonicity target."""
    occupied = n > 0
    if occupied.sum() < 2:
        return 1
    rate = np.divide(bad, n, out=np.zeros_like(n), where=n > 0)
    slope = np.polyfit(np.arange(len(n))[occupied], rate[occupied], 1, w=n[occupied])[0]
    return 1 if slope >= 0 else -1


def _smallest_neighbour(n: np.ndarray, undersized: np.ndarray) -> int:
    """Index to merge at: take the smallest offending bin, fold it into its smaller neighbour.

    Merging the first offending bin instead would fold a well-populated bin into an
    empty trailing one - which happens on discrete features, where the top quantile
    edge equals the maximum value and leaves the last interval empty.
    """
    offenders = np.where(undersized)[0]
    j = int(offenders[np.argmin(n[offenders])])
    if j == 0:
        return 0
    if j == len(n) - 1:
        return j - 1
    return j if n[j + 1] <= n[j - 1] else j - 1


def _merge(n: np.ndarray, bad: np.ndarray, edges: list[float], i: int):
    """Merge bin i into bin i+1, dropping the edge between them."""
    n[i + 1] += n[i]
    bad[i + 1] += bad[i]
    return np.delete(n, i), np.delete(bad, i), edges[:i] + edges[i + 1:]


def monotone_edges(
    values: pl.Series,
    target: np.ndarray,
    n_bins: int = DEFAULT_INITIAL_BINS,
    min_bin_fraction: float = DEFAULT_MIN_BIN_FRACTION,
    min_bin_bads: int = DEFAULT_MIN_BIN_BADS,
) -> list[float]:
    """Quantile-cut then merge adjacent bins until every bin is large enough and monotone.

    values/target must already exclude nulls and be aligned. Returns the final interior
    cut points, which may be empty when the feature cannot support more than one bin.
    """
    edges = _quantile_edges(values, n_bins)
    if not edges:
        return []

    array = values.to_numpy().astype(float)
    # side="left" matches Polars cut(), whose intervals are (lower, edge]. Using "right"
    # shifts every value sitting exactly on an edge into the next bin, which silently
    # empties the first bin on discrete features and drops the wrong cut point.
    indices = np.searchsorted(np.asarray(edges), array, side="left")
    n, bad = _aggregate(indices, target, len(edges) + 1)
    direction = _dominant_direction(n, bad)
    min_n = max(min_bin_fraction * n.sum(), 1.0)

    while len(n) > 1:
        good = n - bad
        undersized = (n < min_n) | (bad < min_bin_bads) | (good < min_bin_bads)
        # Never merge down to a single bin: that would silently zero out a feature's IV,
        # including for a leaking feature that the >0.5 IV guard is supposed to catch.
        # Keep the two-bin split and let binning_report() expose the undersized bin.
        if undersized.any() and len(n) > 2:
            n, bad, edges = _merge(n, bad, edges, _smallest_neighbour(n, undersized))
            continue

        rate = bad / n
        deltas = np.diff(rate)
        violation = deltas * direction < 0
        if not violation.any():
            break
        # merge the shallowest violation first, so genuine structure survives longest
        candidates = np.where(violation)[0]
        i = int(candidates[np.argmin(np.abs(deltas[candidates]))])
        n, bad, edges = _merge(n, bad, edges, i)

    return edges


def _bin_expr(df: pl.DataFrame, feature: str, edges: list[float], rare: set[str] | None = None) -> pl.Expr:
    """Map a feature to string bin labels using fixed edges (numeric) or its own values."""
    if df.schema[feature] in _NUMERIC_DTYPES:
        binned = pl.col(feature).cut(edges).cast(pl.Utf8) if edges else pl.lit("all")
        return pl.when(pl.col(feature).is_null()).then(pl.lit(_MISSING)).otherwise(binned)
    value = pl.col(feature).cast(pl.Utf8).fill_null(_MISSING)
    return value.replace(dict.fromkeys(rare or set(), _RARE))


def _bin_order_expr(df: pl.DataFrame, feature: str, edges: list[float]) -> pl.Expr:
    """Numeric bin index, matching cut()'s (lower, edge] intervals; -1 for missing/categorical.

    Bin labels look like "(-inf, 5.0]" and "(12.3, 18.7]", so sorting them as strings
    orders 12.3 before 5.2 and makes any monotonicity check meaningless.
    """
    if df.schema[feature] not in _NUMERIC_DTYPES or not edges:
        return pl.lit(-1, dtype=pl.Int32)
    index = pl.sum_horizontal([(pl.col(feature) > e).cast(pl.Int32) for e in edges])
    return pl.when(pl.col(feature).is_null()).then(pl.lit(-1, dtype=pl.Int32)).otherwise(index)


def _neutralise_thin_special_bins(table: pl.DataFrame, min_bads: int) -> pl.DataFrame:
    """Force WOE to 0 for a missing/rare bin too thin to estimate.

    These two bins are exempt from the merging loop - missingness has no position in the
    feature's ordering - which left them with no size guard at all. A 12-row missing bin
    was producing |WOE| 1.69, pure noise weighted like real signal. Neutral is the
    conservative reading: the bin exists, but the data cannot say which way it points.
    """
    thin = (
        pl.col("_bin").is_in([_MISSING, _RARE])
        & ((pl.col("n_bad") < min_bads) | (pl.col("n_good") < min_bads))
    )
    return table.with_columns(
        pl.when(thin).then(pl.lit(0.0)).otherwise(pl.col("woe")).alias("woe"),
        thin.alias("neutralised"),
    )


def _woe_from_counts(grouped: pl.DataFrame, total_good: int, total_bad: int) -> pl.DataFrame:
    """Haldane-Anscombe smoothed WOE and IV contribution; bounded even for empty cells."""
    return grouped.with_columns(
        (((pl.col("n_good") + _SMOOTH) / (total_good + 2 * _SMOOTH))
         / ((pl.col("n_bad") + _SMOOTH) / (total_bad + 2 * _SMOOTH))).log().alias("woe"),
        (pl.col("n_good") / total_good).alias("dist_good"),
        (pl.col("n_bad") / total_bad).alias("dist_bad"),
    ).with_columns(
        ((pl.col("dist_good") - pl.col("dist_bad")) * pl.col("woe")).alias("iv_contribution")
    )


def _rare_categories(df: pl.DataFrame, feature: str, target: str, min_bads: int) -> set[str]:
    """Category values with too few bads or goods to estimate a WOE from.

    Deliberately NOT keyed on min_bin_fraction. The 5% population floor is a rule for
    controlling how many bins a NUMERIC feature is cut into; applied to a categorical it
    destroys any feature with many levels. sub_grade has 35 levels averaging 2.9% of the
    population each, so a 5% floor pooled every one of them into a single "rare" bin and
    drove its IV to zero - below its own coarser parent, `grade`, which is impossible.
    What actually matters statistically is having enough of both classes in the cell.
    """
    counts = df.group_by(feature).agg(
        (pl.col(target) == 1).sum().alias("bad"),
        (pl.col(target) == 0).sum().alias("good"),
    ).filter((pl.col("bad") < min_bads) | (pl.col("good") < min_bads))
    return {v for v in counts[feature].cast(pl.Utf8).to_list() if v is not None}


def woe_iv_table(
    df: pl.DataFrame,
    feature: str,
    target: str = "default_flag",
    n_bins: int = DEFAULT_INITIAL_BINS,
    min_bin_fraction: float = DEFAULT_MIN_BIN_FRACTION,
    min_bin_bads: int = DEFAULT_MIN_BIN_BADS,
) -> pl.DataFrame:
    """Per-bin counts, bad rate, WOE and IV contribution for one feature."""
    working = df.select(feature, target).filter(pl.col(target).is_not_null())
    edges: list[float] = []
    rare: set[str] = set()

    if working.schema[feature] in _NUMERIC_DTYPES:
        non_null = working.filter(pl.col(feature).is_not_null())
        if non_null.height > 0:
            edges = monotone_edges(
                non_null[feature], non_null[target].to_numpy().astype(float),
                n_bins, min_bin_fraction, min_bin_bads,
            )
    else:
        rare = _rare_categories(working, feature, target, min_bin_bads)

    binned = working.with_columns(
        _bin_expr(working, feature, edges, rare).alias("_bin"),
        _bin_order_expr(working, feature, edges).alias("_order"),
    )
    grouped = binned.group_by("_bin", "_order").agg(
        pl.len().alias("n"),
        (pl.col(target) == 0).sum().alias("n_good"),
        (pl.col(target) == 1).sum().alias("n_bad"),
    )
    total_good = int((binned[target] == 0).sum())
    total_bad = int((binned[target] == 1).sum())
    table = _woe_from_counts(grouped, total_good, total_bad)
    return (
        _neutralise_thin_special_bins(table, min_bin_bads)
        .with_columns((pl.col("n_bad") / pl.col("n")).alias("bad_rate"))
        .sort("_order", "_bin")
    )


def information_value(df: pl.DataFrame, feature: str, target: str = "default_flag", **kwargs) -> float:
    """Total IV for a feature - the standard first-pass screening metric."""
    return float(woe_iv_table(df, feature, target, **kwargs)["iv_contribution"].sum())


def rank_features_by_iv(
    df: pl.DataFrame, features: list[str], target: str = "default_flag", **kwargs
) -> pl.DataFrame:
    """Rank candidate features by IV, flagging weak (<0.02) and suspiciously strong (>0.5) ones."""
    rows = []
    for f in features:
        try:
            iv = information_value(df, f, target, **kwargs)
        except Exception as exc:
            iv = None
            print(f"skipped {f}: {exc}")
        rows.append({"feature": f, "iv": iv})
    return pl.DataFrame(rows).sort("iv", descending=True, nulls_last=True).with_columns(
        pl.when(pl.col("iv") > 0.5).then(pl.lit("suspicious - check leakage"))
        .when(pl.col("iv") > 0.3).then(pl.lit("strong"))
        .when(pl.col("iv") > 0.1).then(pl.lit("medium"))
        .when(pl.col("iv") > 0.02).then(pl.lit("weak"))
        .otherwise(pl.lit("not useful"))
        .alias("strength")
    )


class WOEEncoder:
    """Fit WOE bins/values on train only, then apply the frozen mapping anywhere else.

    Convention: WOE = ln(dist_good / dist_bad). Higher WOE means safer. When used as
    input to a model predicting P(default=1), well-specified coefficients come out
    NEGATIVE - this is correct, not a sign-flip bug.

    Prevents the classic scorecard bug of recomputing bins per dataset, which makes
    train and OOT test (or serving) use different, incomparable encodings of the same
    feature. Edges and rare-category sets are computed once in fit() and reused in
    transform(), so bin labels always match exactly.
    """

    def __init__(
        self,
        features: list[str],
        target: str = "default_flag",
        n_bins: int = DEFAULT_INITIAL_BINS,
        min_bin_fraction: float = DEFAULT_MIN_BIN_FRACTION,
        min_bin_bads: int = DEFAULT_MIN_BIN_BADS,
    ):
        self.features = features
        self.target = target
        self.n_bins = n_bins
        self.min_bin_fraction = min_bin_fraction
        self.min_bin_bads = min_bin_bads
        self.bin_edges_: dict[str, list[float]] = {}
        self.rare_categories_: dict[str, set[str]] = {}
        self.woe_maps_: dict[str, dict[str, float]] = {}
        self.bin_tables_: dict[str, pl.DataFrame] = {}

    def _bin_column(self, df: pl.DataFrame, feature: str) -> pl.Expr:
        return _bin_expr(df, feature, self.bin_edges_.get(feature, []), self.rare_categories_.get(feature))

    def fit(self, df: pl.DataFrame) -> "WOEEncoder":
        """Learn bin edges, rare-category pooling and per-bin WOE from df - train only."""
        base = df.filter(pl.col(self.target).is_not_null())

        for feature in self.features:
            working = base.select(feature, self.target)
            if working.schema[feature] in _NUMERIC_DTYPES:
                non_null = working.filter(pl.col(feature).is_not_null())
                self.bin_edges_[feature] = monotone_edges(
                    non_null[feature], non_null[self.target].to_numpy().astype(float),
                    self.n_bins, self.min_bin_fraction, self.min_bin_bads,
                ) if non_null.height > 0 else []
            else:
                self.rare_categories_[feature] = _rare_categories(
                    working, feature, self.target, self.min_bin_bads
                )

            binned = working.with_columns(
                self._bin_column(working, feature).alias("_bin"),
                _bin_order_expr(working, feature, self.bin_edges_.get(feature, [])).alias("_order"),
            )
            grouped = binned.group_by("_bin", "_order").agg(
                pl.len().alias("n"),
                (pl.col(self.target) == 0).sum().alias("n_good"),
                (pl.col(self.target) == 1).sum().alias("n_bad"),
            )
            table = _neutralise_thin_special_bins(
                _woe_from_counts(
                    grouped,
                    int((binned[self.target] == 0).sum()),
                    int((binned[self.target] == 1).sum()),
                ),
                self.min_bin_bads,
            ).with_columns((pl.col("n_bad") / pl.col("n")).alias("bad_rate")).sort("_order", "_bin")
            self.bin_tables_[feature] = table
            self.woe_maps_[feature] = dict(zip(table["_bin"].cast(pl.Utf8), table["woe"]))
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the frozen bin edges/WOE map from fit() - safe on OOT test or serving data.

        Bins unseen during fit fall back to WOE 0.0 - neutral, not an error, but worth
        monitoring: a high fallback rate signals population drift.
        """
        out = df
        for feature in self.features:
            out = out.with_columns(
                self._bin_column(df, feature)
                .replace_strict(self.woe_maps_[feature], default=0.0, return_dtype=pl.Float64)
                .alias(f"{feature}_woe")
            )
        return out

    def binning_report(self) -> pl.DataFrame:
        """One row per feature: bin count, smallest bin, fewest bads, monotonicity.

        This is the artefact that makes binning decisions auditable instead of implicit.
        A feature reduced to one bin carries no signal and should be dropped; is_monotone
        False is expected only for categorical features, which have no natural ordering.
        """
        rows = []
        for feature, table in self.bin_tables_.items():
            is_numeric = feature in self.bin_edges_
            ordered = table.filter(pl.col("_order") >= 0).sort("_order")
            rates = ordered["bad_rate"].to_list()
            diffs = np.diff(rates) if len(rates) > 1 else np.array([0.0])
            rows.append({
                "feature": feature,
                "is_numeric": is_numeric,
                "n_bins": table.height,
                "has_missing_bin": _MISSING in table["_bin"].to_list(),
                "n_neutralised": int(table["neutralised"].sum()),
                "min_bin_n": int(table["n"].min()),
                "min_bin_bads": int(table["n_bad"].min()),
                # only meaningful for numeric features: categories have no ordering
                "is_monotone": bool((diffs >= 0).all() or (diffs <= 0).all()) if is_numeric else None,
                "max_abs_woe": float(table["woe"].abs().max()),
            })
        return pl.DataFrame(rows).sort("feature")


def prune_correlated_features(features_by_priority: list[str], corr: pd.DataFrame, threshold: float = 0.6) -> list[str]:
    """Greedily keep each feature (in priority order, e.g. IV descending) unless it
    correlates above threshold with an already-kept one - avoids near-duplicate
    features destabilizing a logistic regression (e.g. grade/sub_grade/int_rate)."""
    kept: list[str] = []
    for f in features_by_priority:
        if all(abs(corr.loc[f, k]) < threshold for k in kept):
            kept.append(f)
    return kept