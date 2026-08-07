"""Is a measured AUC difference real, and is it stable? The two questions this project skipped.

Every conclusion here rests on small differences: the champion beats `sub_grade` used alone
by +0.0076 AUC, and LendingClub's grade and rate contribute +0.0171 within the same model
class. Those numbers have decided which model is the champion and how the project's headline
finding is phrased, and neither has been tested.

Two instruments, answering two different questions:

- `delong_auc_test` asks whether a difference is distinguishable from zero. The models are
  scored on the SAME loans, so their AUCs are correlated and an unpaired comparison would
  badly overstate the standard error. DeLong (1988), computed by the O(n log n) midrank
  algorithm of Sun and Xu (2014).
- `gini_by_period` asks whether the difference matters. At 434,407 observations almost
  anything is significant; if Gini swings by 0.06 between quarters, a 0.015 Gini edge is
  real and practically invisible. Significance without stability is not evidence of much.
"""

import numpy as np
import polars as pl
from scipy import stats

_TARGET = "default_flag"


def _midrank(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, which is what makes the DeLong variance exact under ties."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_components(scores: np.ndarray, n_positive: int) -> tuple[np.ndarray, np.ndarray]:
    """AUCs and their covariance matrix for k scorers evaluated on one sorted sample.

    `scores` is (k, n) with the positive cases first. Returns AUC per scorer and the k x k
    covariance, whose off-diagonal is exactly the term an unpaired test throws away.
    """
    m, n = n_positive, scores.shape[1] - n_positive
    positive, negative = scores[:, :m], scores[:, m:]
    k = scores.shape[0]

    tx = np.stack([_midrank(positive[r]) for r in range(k)])
    ty = np.stack([_midrank(negative[r]) for r in range(k)])
    tz = np.stack([_midrank(scores[r]) for r in range(k)])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01, ddof=1).reshape(k, k) / m + np.cov(v10, ddof=1).reshape(k, k) / n
    return aucs, covariance


def _prepare(y: np.ndarray, *scores: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.argsort(-np.asarray(y, dtype=float), kind="mergesort")  # positives first
    return np.stack([np.asarray(s, dtype=float)[order] for s in scores]), int(np.sum(y))


def auc_confidence_interval(y: np.ndarray, score: np.ndarray, alpha: float = 0.05) -> dict:
    """AUC with a DeLong confidence interval.

    Reported alongside every headline AUC. An interval makes the sample size visible, which a
    point estimate hides: the same 0.6964 means very different things on 400,000 loans and on
    4,000.
    """
    prepared, n_positive = _prepare(y, score)
    aucs, covariance = _delong_components(prepared, n_positive)
    std = float(np.sqrt(covariance[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    return {
        "auc": float(aucs[0]),
        "std_error": std,
        "ci_lower": float(aucs[0] - z * std),
        "ci_upper": float(aucs[0] + z * std),
        "n": int(len(y)),
        "n_positive": n_positive,
    }


def delong_auc_test(
    y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, alpha: float = 0.05
) -> dict:
    """Test whether two AUCs measured on the SAME sample differ.

    Correlation between the two scorers is estimated and subtracted, so the standard error is
    of the DIFFERENCE, not of each AUC separately. Two models sharing most of their signal -
    which is exactly the case for a GBM with and without `sub_grade` - are far more comparable
    than their individual intervals suggest, and an unpaired test would miss a real difference.
    """
    prepared, n_positive = _prepare(y, score_a, score_b)
    aucs, covariance = _delong_components(prepared, n_positive)
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    std = float(np.sqrt(max(variance, 0.0)))
    difference = float(aucs[0] - aucs[1])
    z = difference / std if std > 0 else 0.0
    critical = stats.norm.ppf(1 - alpha / 2)
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "difference": difference,
        "std_error": std,
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
        "ci_lower": difference - critical * std,
        "ci_upper": difference + critical * std,
        "significant": bool(2 * stats.norm.sf(abs(z)) < alpha),
    }


def gini_by_period(
    df: pl.DataFrame, score: np.ndarray, period_column: str, alpha: float = 0.05
) -> pl.DataFrame:
    """Gini with a confidence interval per period, plus the pooled figure.

    Gini = 2*AUC - 1, so the interval transforms directly. Read the spread across periods
    against the model difference being claimed: an edge smaller than the quarter-to-quarter
    swing is real but not something a portfolio would notice.
    """
    frame = df.select(pl.col(period_column).alias("_period"), pl.col(_TARGET)).with_columns(
        pl.Series("_score", score)
    )
    rows = []
    for period in sorted(frame["_period"].unique().drop_nulls().to_list()):
        part = frame.filter(pl.col("_period") == period)
        y = part[_TARGET].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        result = auc_confidence_interval(y, part["_score"].to_numpy(), alpha)
        rows.append(
            {
                "period": str(period),
                "n": result["n"],
                "n_default": result["n_positive"],
                "gini": 2 * result["auc"] - 1,
                "ci_lower": 2 * result["ci_lower"] - 1,
                "ci_upper": 2 * result["ci_upper"] - 1,
            }
        )
    pooled = auc_confidence_interval(frame[_TARGET].to_numpy(), frame["_score"].to_numpy(), alpha)
    rows.append(
        {
            "period": "pooled",
            "n": pooled["n"],
            "n_default": pooled["n_positive"],
            "gini": 2 * pooled["auc"] - 1,
            "ci_lower": 2 * pooled["ci_lower"] - 1,
            "ci_upper": 2 * pooled["ci_upper"] - 1,
        }
    )
    return pl.DataFrame(rows).with_columns(
        (pl.col("ci_upper") - pl.col("ci_lower")).alias("ci_width")
    )
