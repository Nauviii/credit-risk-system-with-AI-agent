"""Population Stability Index: how far a distribution has moved from its reference.

Same fit-on-reference / transform-elsewhere discipline as WOEEncoder. Bin edges come
from the reference population (train) and are frozen; the comparison population is
bucketed with those same edges. Recomputing edges per population would compare each
distribution against itself and report stability that is not there.

PSI = sum over bins of (actual_share - reference_share) * ln(actual_share / reference_share)

Conventional reading: < 0.10 stable, 0.10-0.25 worth watching, > 0.25 material shift.
These are rules of thumb, not tests - a large PSI on a low-IV feature rarely matters,
and a small PSI hides a lot when the population is large.

Two implementation choices worth knowing, since both silently change the number:
- Outer edges are -inf/+inf, so values outside the reference range land in the end bins
  rather than becoming nulls.
- Empty bins would make ln(0) undefined. Shares are floored at `_EPS` instead of merging
  bins, so bin geometry stays comparable across features. A bin that is empty in one
  population and populated in the other therefore contributes a large but finite term.
"""

import numpy as np
import polars as pl

_EPS = 1e-6
_MISSING = "missing"


def psi_bin_edges(reference: pl.Series, n_bins: int = 10) -> list[float]:
    """Quantile cut points from the reference population, to be frozen and reused."""
    non_null = reference.drop_nulls()
    if non_null.len() == 0:
        return []
    qs = [i / n_bins for i in range(1, n_bins)]
    edges = {non_null.quantile(q) for q in qs}
    return sorted(e for e in edges if e is not None)


def _shares(values: pl.Series, edges: list[float], n_bins: int) -> np.ndarray:
    """Share of the population in each bin, with nulls as their own final bin."""
    array = values.to_numpy()
    null_mask = values.is_null().to_numpy()
    counts = np.zeros(n_bins + 1)
    if (~null_mask).any():
        numeric = array[~null_mask].astype(float)
        index = np.searchsorted(np.asarray(edges), numeric, side="left")
        counts[: n_bins] = np.bincount(index, minlength=n_bins)
    counts[n_bins] = null_mask.sum()
    total = counts.sum()
    return counts / total if total else counts


def psi_table(reference: pl.Series, actual: pl.Series, edges: list[float]) -> pl.DataFrame:
    """Per-bin reference share, actual share and PSI contribution."""
    n_bins = len(edges) + 1
    ref, act = _shares(reference, edges, n_bins), _shares(actual, edges, n_bins)
    ref_f, act_f = np.maximum(ref, _EPS), np.maximum(act, _EPS)
    labels = [f"<= {e}" for e in edges] + [f"> {edges[-1]}" if edges else "all"] + [_MISSING]
    return pl.DataFrame({
        "bin": labels,
        "reference_share": ref,
        "actual_share": act,
        "psi_contribution": (act_f - ref_f) * np.log(act_f / ref_f),
    })


def population_stability_index(
    reference: pl.Series, actual: pl.Series, n_bins: int = 10, edges: list[float] | None = None
) -> float:
    """PSI of actual against reference, using reference-derived (or supplied) edges."""
    edges = psi_bin_edges(reference, n_bins) if edges is None else edges
    return float(psi_table(reference, actual, edges)["psi_contribution"].sum())


def psi_report(
    reference_df: pl.DataFrame, actual_df: pl.DataFrame, features: list[str], n_bins: int = 10
) -> pl.DataFrame:
    """PSI per feature with the conventional interpretation band, worst first."""
    rows = []
    for feature in features:
        if feature not in reference_df.columns or feature not in actual_df.columns:
            continue
        if not reference_df[feature].dtype.is_numeric():
            continue  # categorical stability needs share-by-level, not quantile bins
        rows.append({
            "feature": feature,
            "psi": population_stability_index(reference_df[feature], actual_df[feature], n_bins),
        })
    return pl.DataFrame(rows).sort("psi", descending=True).with_columns(
        pl.when(pl.col("psi") > 0.25).then(pl.lit("material shift"))
        .when(pl.col("psi") > 0.10).then(pl.lit("watch"))
        .otherwise(pl.lit("stable"))
        .alias("band")
    )