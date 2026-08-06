"""Tests for the decision layer: approval curves, LGD, horizon scaling, cutoff economics."""

import numpy as np
import polars as pl
import pytest

from credit_risk.evaluation.business import (
    approval_curve,
    cutoff_table,
    empirical_lgd,
    to_lifetime_pd,
)


def _book(n: int = 10_000, seed: int = 0):
    """Scores that genuinely rank risk, so a tighter cutoff must lower the bad rate."""
    rng = np.random.default_rng(seed)
    score = rng.normal(550, 30, n)
    p = 1 / (1 + np.exp((score - 550) / 25))
    return score, (rng.random(n) < p).astype(int), p


def test_tighter_cutoff_lowers_the_approved_bad_rate():
    score, y, _ = _book()
    curve = approval_curve(score, y).sort("approval_rate")
    assert curve["bad_rate"].to_list() == sorted(curve["bad_rate"].to_list())


def test_approval_rate_is_monotone_in_the_cutoff():
    score, y, _ = _book()
    curve = approval_curve(score, y).sort("cutoff", descending=True)
    assert curve["approval_rate"].to_list() == sorted(curve["approval_rate"].to_list())


def test_declined_population_is_worse_than_the_approved_one():
    score, y, _ = _book()
    curve = approval_curve(score, y).drop_nulls("bad_rate_declined")
    assert (curve["bad_rate_declined"] > curve["bad_rate"]).all()


def test_lifetime_scaling_raises_pd_more_for_lower_coverage():
    pd_horizon = np.array([0.10, 0.10])
    lifetime = to_lifetime_pd(pd_horizon, np.array([0.60, 0.42]))
    assert lifetime[1] > lifetime[0]
    assert lifetime[0] == pytest.approx(0.10 / 0.60)
    assert (to_lifetime_pd(np.array([0.9]), np.array([0.4])) <= 1.0).all()


def test_empirical_lgd_ignores_fully_repaid_principal():
    df = pl.DataFrame({
        "loan_status": ["Charged Off", "Charged Off", "Fully Paid"],
        "loan_amnt": [10_000.0, 10_000.0, 10_000.0],
        "total_rec_prncp": [2_000.0, 10_000.0, 10_000.0],  # second has nothing outstanding
        "recoveries": [800.0, 0.0, 0.0],
    })
    result = empirical_lgd(df)
    assert result["n"] == 1
    assert result["lgd_mean"] == pytest.approx(1 - 800 / 8_000)


def test_cutoff_table_expected_loss_falls_as_the_cutoff_tightens():
    score, y, p = _book()
    ead = np.full_like(score, 10_000.0)
    table = cutoff_table(score, y, to_lifetime_pd(p, np.full_like(p, 0.6)), ead, lgd=0.85)
    ordered = table.sort("approval_rate")
    assert ordered["expected_loss_rate"].to_list() == sorted(ordered["expected_loss_rate"].to_list())
    assert (ordered["expected_loss"].diff().drop_nulls() > 0).all()


def test_horizon_coverage_reads_term_months_from_a_cleaned_frame():
    """Regression: term_months comes from clean_features, not from build_target."""
    from credit_risk.evaluation.business import horizon_coverage
    from credit_risk.features.cleaning import parse_term_months

    labeled = pl.DataFrame({
        "issue_d": ["Jan-2013"] * 4,
        "term": [" 36 months", " 36 months", " 60 months", " 60 months"],
        "default_flag": [1, 0, 1, 0],
        "mob_event": [12.0, 40.0, 20.0, 40.0],
    })
    assert "term_months" not in labeled.columns
    coverage = horizon_coverage(parse_term_months(labeled), horizon=24, vintages=[2013])
    assert coverage["term_months"].to_list() == [36, 60]
    assert coverage["n_default_in_horizon"].to_list() == [1, 1]


def test_scheduled_gross_yield_matches_the_loan_terms():
    from credit_risk.evaluation.business import scheduled_gross_yield

    df = pl.DataFrame({
        "installment": [333.0, 200.0],
        "term_months": [36, 60],
        "loan_amnt": [10_000.0, 10_000.0],
    })
    yields = scheduled_gross_yield(df)
    assert yields[0] == pytest.approx((333.0 * 36 - 10_000) / 10_000)
    assert yields[1] > yields[0]  # longer term collects more interest on the same principal


def test_net_margin_falls_as_the_cutoff_loosens():
    from credit_risk.evaluation.business import cutoff_table as ct

    score, y, p = _book()
    ead = np.full_like(score, 10_000.0)
    table = ct(
        score, y, to_lifetime_pd(p, np.full_like(p, 0.6)), ead, lgd=0.85,
        gross_yield=np.full_like(score, 0.25),
    ).sort("approval_rate")
    assert "net_margin_rate" in table.columns
    assert table["net_margin_rate"].to_list() == sorted(table["net_margin_rate"].to_list(), reverse=True)


def test_total_margin_peaks_looser_than_the_margin_rate():
    """Maximising margin per unit of exposure ignores volume and lands too tight."""
    from credit_risk.evaluation.business import cutoff_table as ct

    score, y, p = _book(n=40_000)
    ead = np.full_like(score, 10_000.0)
    table = ct(
        score, y, to_lifetime_pd(p, np.full_like(p, 0.6)), ead, lgd=0.85,
        gross_yield=np.full_like(score, 0.25), n_points=20,
    )
    best_rate = table.sort("net_margin_rate", descending=True)["approval_rate"][0]
    best_total = table.sort("net_margin_total", descending=True)["approval_rate"][0]
    assert best_total >= best_rate


def test_marginal_margin_turns_negative_before_the_average_does():
    from credit_risk.evaluation.business import cutoff_table as ct

    score, y, p = _book(n=40_000)
    ead = np.full_like(score, 10_000.0)
    table = ct(
        score, y, to_lifetime_pd(p, np.full_like(p, 0.6)), ead, lgd=0.85,
        gross_yield=np.full_like(score, 0.25), n_points=20,
    ).drop_nulls("marginal_margin_rate")
    assert (table["marginal_margin_rate"] < table["net_margin_rate"]).any()