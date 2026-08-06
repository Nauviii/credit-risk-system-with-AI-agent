"""Tests for calibration: detect miscalibration, correct it, and scale to points."""

import numpy as np
import pytest

from credit_risk.evaluation.calibration import (
    Calibrator,
    brier_decomposition,
    central_tendency_shift,
    expected_calibration_error,
    pd_to_score,
    reliability_table,
)


def _well_calibrated(n: int = 40_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.5, n)
    return (rng.random(n) < p).astype(int), p


def test_well_calibrated_predictions_have_small_ece():
    y, p = _well_calibrated()
    assert expected_calibration_error(y, p) < 0.01


def test_systematically_inflated_predictions_are_caught():
    y, p = _well_calibrated()
    inflated = np.clip(p * 1.6, 0, 1)
    assert expected_calibration_error(y, inflated) > 0.05
    assert reliability_table(y, inflated)["gap"].mean() > 0


def test_brier_decomposition_reconstructs_the_score():
    y, p = _well_calibrated()
    parts = brier_decomposition(y, p, n_bins=20)
    assert parts["decomposition_check"] == pytest.approx(parts["brier"], abs=5e-3)
    assert parts["uncertainty"] == pytest.approx(y.mean() * (1 - y.mean()), abs=1e-12)


def test_calibrator_fitted_on_one_sample_corrects_another():
    y_fit, p_fit = _well_calibrated(seed=1)
    y_apply, p_apply = _well_calibrated(seed=2)

    def inflate(p):
        return np.clip(p * 1.6, 0, 1)

    before = expected_calibration_error(y_apply, inflate(p_apply))
    for method in ("isotonic", "platt"):
        calibrator = Calibrator(method).fit(y_fit, inflate(p_fit))
        after = expected_calibration_error(y_apply, calibrator.transform(inflate(p_apply)))
        assert after < before / 2


def test_central_tendency_shift_hits_the_target_mean():
    _, p = _well_calibrated()
    for target in (0.05, 0.30):
        shift = central_tendency_shift(p, target)
        logits = np.log(p / (1 - p)) + shift
        assert (1 / (1 + np.exp(-logits))).mean() == pytest.approx(target, abs=1e-6)


def test_central_tendency_shift_preserves_ranking():
    _, p = _well_calibrated()
    shifted = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + central_tendency_shift(p, 0.05))))
    assert (np.argsort(p) == np.argsort(shifted)).all()


def test_pdo_scaling_doubles_odds_every_pdo_points():
    low_risk, high_risk = pd_to_score(np.array([0.02])), pd_to_score(np.array([0.5]))
    assert low_risk[0] > high_risk[0]
    # 50:1 odds must land exactly on the base score by construction
    assert pd_to_score(np.array([1 / 51]), pdo=20, base_score=600, base_odds=50.0)[
        0
    ] == pytest.approx(600.0)
    # halving the good:bad odds must cost exactly one PDO
    assert pd_to_score(np.array([1 / 26]))[0] == pytest.approx(580.0, abs=1e-6)


def test_calibrator_never_returns_a_zero_or_one_probability():
    """Isotonic can map a whole block to exactly 0; a PD of 0 implies infinite odds."""
    y_fit, p_fit = _well_calibrated(seed=4)
    y_fit[p_fit < 0.10] = 0  # force a clean all-good region for isotonic to collapse
    for method in ("isotonic", "platt"):
        out = Calibrator(method).fit(y_fit, p_fit).transform(p_fit)
        assert out.min() > 0.0
        assert out.max() < 1.0
        assert np.isfinite(pd_to_score(out)).all()


def test_platt_preserves_ranking_exactly():
    """A strictly monotone calibrator cannot change AUC; isotonic can, via ties."""
    y_fit, p_fit = _well_calibrated(seed=5)
    out = Calibrator("platt").fit(y_fit, p_fit).transform(p_fit)
    assert len(np.unique(out)) == len(np.unique(p_fit))
