"""Tests for PSI: identical distributions score ~0, shifted ones score large."""

import numpy as np
import polars as pl

from credit_risk.evaluation.stability import (
    population_stability_index,
    psi_bin_edges,
    psi_report,
    psi_table,
)


def test_identical_distributions_score_near_zero():
    rng = np.random.default_rng(0)
    reference = pl.Series(rng.normal(size=50_000))
    actual = pl.Series(rng.normal(size=50_000))
    assert population_stability_index(reference, actual) < 0.01


def test_shifted_distribution_scores_material():
    rng = np.random.default_rng(0)
    reference = pl.Series(rng.normal(size=50_000))
    actual = pl.Series(rng.normal(loc=1.0, size=50_000))
    assert population_stability_index(reference, actual) > 0.25


def test_psi_is_symmetric_in_its_two_populations():
    rng = np.random.default_rng(2)
    a = pl.Series(rng.normal(size=20_000))
    b = pl.Series(rng.normal(loc=0.4, size=20_000))
    edges = psi_bin_edges(a)
    forward = population_stability_index(a, b, edges=edges)
    backward = float(psi_table(b, a, edges)["psi_contribution"].sum())
    assert abs(forward - backward) < 1e-9


def test_values_outside_the_reference_range_land_in_end_bins_not_nulls():
    reference = pl.Series(np.linspace(0, 1, 10_000))
    actual = pl.Series(np.linspace(-5, 6, 10_000))
    table = psi_table(reference, actual, psi_bin_edges(reference))
    assert abs(table["actual_share"].sum() - 1.0) < 1e-9
    assert table.filter(pl.col("bin") == "missing")["actual_share"][0] == 0.0


def test_nulls_form_their_own_bin_and_drive_psi():
    reference = pl.Series([1.0] * 5_000 + [2.0] * 5_000)
    actual = pl.Series([1.0] * 5_000 + [None] * 5_000, dtype=pl.Float64)
    assert population_stability_index(reference, actual) > 0.25


def test_report_skips_categoricals_and_bands_the_result():
    rng = np.random.default_rng(3)
    ref = pl.DataFrame({"x": rng.normal(size=20_000), "grade": ["A"] * 20_000})
    act = pl.DataFrame({"x": rng.normal(loc=2.0, size=20_000), "grade": ["B"] * 20_000})
    report = psi_report(ref, act, ["x", "grade", "absent"])
    assert report["feature"].to_list() == ["x"]
    assert report["band"][0] == "material shift"
