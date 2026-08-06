"""Calibration: is a predicted PD of 0.12 actually followed by 12% defaults?

Discrimination (AUC, Gini, KS) only cares about ranking. Every business decision built
on top of a PD model - expected loss, pricing, provisioning, cutoffs - needs the level
to be right as well, and a model can rank perfectly while being systematically wrong
about magnitude. Nothing in this project has checked that until now.

Two project-specific facts shape how these are used:

1. The target is a 24-MONTH PD, not lifetime. The horizon truncates term-60 loans harder
   than term-36 ones (it captures roughly 42% of eventual 60-month defaults against 60%
   for 36-month), so a single calibration applied to both terms understates 60-month risk
   by more than it understates 36-month risk. Calibrate per term, or state the bias.

2. The calibrator must be fitted on VALIDATION, never on train. Fitted on train it simply
   re-learns the fit the model already has and reports near-perfect calibration.
"""

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-9


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds, clipped away from 0 and 1 so the transform stays finite."""
    clipped = np.clip(p, _EPS, 1 - _EPS)
    return np.log(clipped / (1 - clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pl.DataFrame:
    """Predicted vs observed default rate per score band - the reliability diagram as data.

    Bins are quantiles of the prediction, so each row carries a similar population rather
    than a similar width; equal-width bins on a skewed PD distribution leave the top bands
    nearly empty and make the tail look miscalibrated when it is merely thin.
    """
    order = np.argsort(p)
    groups = np.array_split(order, n_bins)
    rows = [
        {
            "bin": i,
            "n": len(g),
            "mean_predicted": float(p[g].mean()),
            "observed_rate": float(y[g].mean()),
            "gap": float(p[g].mean() - y[g].mean()),
        }
        for i, g in enumerate(groups) if len(g) > 0
    ]
    return pl.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Population-weighted mean absolute gap between predicted and observed rates."""
    table = reliability_table(y, p, n_bins)
    weights = table["n"].to_numpy() / table["n"].sum()
    return float((weights * table["gap"].abs().to_numpy()).sum())


def brier_decomposition(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Split the Brier score into reliability, resolution and uncertainty.

    brier = reliability - resolution + uncertainty. Reliability is the calibration error
    (lower is better); resolution is how far the bins' observed rates spread from the base
    rate (higher is better, and is the discrimination part); uncertainty is the base rate's
    own variance, fixed by the data. A single Brier number confounds all three, which is
    why a model can look worse on Brier purely because its population is less risky.
    """
    table = reliability_table(y, p, n_bins)
    n = len(y)
    base = float(y.mean())
    weights = table["n"].to_numpy() / n
    predicted = table["mean_predicted"].to_numpy()
    observed = table["observed_rate"].to_numpy()

    reliability = float((weights * (predicted - observed) ** 2).sum())
    resolution = float((weights * (observed - base) ** 2).sum())
    uncertainty = base * (1 - base)
    return {
        "brier": float(((p - y) ** 2).mean()),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "decomposition_check": reliability - resolution + uncertainty,
    }


class Calibrator:
    """Map raw model scores to calibrated probabilities. Fit on validation, apply anywhere.

    method="platt" is the default and fits a single logistic on the log-odds, so it only
    shifts and rescales. method="isotonic" is non-parametric and can correct any monotone
    distortion, but on this data it bought nothing (OOT ECE 0.0117 vs 0.0118) while costing
    three things: it is a step function, so it creates ties and shaves AUC (0.6972 -> 0.6970);
    it cannot extrapolate past its fitted range; and it assigned PD exactly 0 to 989 OOT
    loans, six of which defaulted. A PD of zero implies infinite odds, breaks any expected
    loss calculation and sends the point score to its clipping bound. Prefer isotonic only
    where a reliability plot shows distortion a two-parameter fit demonstrably cannot follow.

    `floor` and `cap` bound the output regardless of method. They are a safety rail, not a
    correction: a model should not be claiming certainty at either end.
    """

    def __init__(self, method: str = "platt", floor: float = 1e-4, cap: float = 1 - 1e-4):
        if method not in ("isotonic", "platt"):
            raise ValueError(f"unknown method: {method}")
        self.method = method
        self.floor = floor
        self.cap = cap
        self.model_ = None

    def fit(self, y: np.ndarray, p: np.ndarray) -> "Calibrator":
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        else:
            self.model_ = LogisticRegression().fit(_logit(p).reshape(-1, 1), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Calibrator must be fitted before transform")
        raw = (
            self.model_.predict(p) if self.method == "isotonic"
            else self.model_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        )
        return np.clip(raw, self.floor, self.cap)


def central_tendency_shift(p: np.ndarray, target_rate: float, tolerance: float = 1e-8) -> float:
    """Log-odds offset c such that mean(sigmoid(logit(p) + c)) equals target_rate.

    Used to anchor a model trained on one vintage's base rate to a long-run average - the
    standard regulatory adjustment separating rank ordering (kept) from level (re-set).
    Solved by bisection because the mean of a sigmoid has no closed-form inverse.
    """
    logits = _logit(p)
    low, high = -20.0, 20.0
    while high - low > tolerance:
        mid = (low + high) / 2
        if _sigmoid(logits + mid).mean() < target_rate:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def pd_to_score(p: np.ndarray, pdo: int = 20, base_score: int = 600, base_odds: float = 50.0) -> np.ndarray:
    """Convert PD to scorecard points: Score = offset + factor * ln(odds of good).

    factor = pdo / ln(2), so every `pdo` points doubles the good:bad odds. Defaults put
    600 points at 50:1 odds. A PD without a point scale is not yet a scorecard - cutoffs,
    overrides and monitoring in production are all expressed in points.
    """
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    odds_good = (1 - np.clip(p, _EPS, 1 - _EPS)) / np.clip(p, _EPS, 1 - _EPS)
    return offset + factor * np.log(odds_good)