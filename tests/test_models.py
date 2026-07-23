"""Smoke tests for model training - confirm both pipelines run end-to-end and beat chance."""

import numpy as np
import polars as pl
from credit_risk.evaluation.metrics import auc


def _synthetic_labeled_df(n: int, seed: int) -> pl.DataFrame:
    """Minimal frame covering every SCORECARD_FEATURES/gbm_features column with real signal."""
    rng = np.random.default_rng(seed)
    grade = rng.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1])
    risk = {"A": 0.05, "B": 0.15, "C": 0.30, "D": 0.45}
    default_flag = np.array([rng.binomial(1, risk[g]) for g in grade])
    return pl.DataFrame({
        "sub_grade": grade, "grade": grade,
        "int_rate": np.where(grade == "A", 8.0, np.where(grade == "B", 12.0, np.where(grade == "C", 18.0, 24.0))),
        "term_months": rng.choice([36, 60], n), "fico_range_low": rng.uniform(660, 750, n),
        "dti": rng.uniform(0, 40, n), "verification_status": rng.choice(["Verified", "Not Verified"], n),
        "loan_amnt": rng.uniform(1000, 35000, n), "home_ownership": rng.choice(["RENT", "OWN"], n),
        "annual_inc": rng.uniform(20000, 150000, n), "revol_util": rng.uniform(0, 100, n),
        "inq_last_6mths": rng.integers(0, 5, n), "purpose": rng.choice(["debt_consolidation", "credit_card"], n),
        "acc_open_past_24mths": rng.integers(0, 10, n), "bc_open_to_buy": rng.uniform(0, 20000, n),
        "avg_cur_bal": rng.uniform(0, 50000, n), "tot_hi_cred_lim": rng.uniform(0, 200000, n),
        "tot_cur_bal": rng.uniform(0, 100000, n), "num_tl_op_past_12m": rng.integers(0, 5, n),
        "total_bc_limit": rng.uniform(0, 30000, n), "mort_acc": rng.integers(0, 3, n),
        "installment": rng.uniform(50, 1200, n), "percent_bc_gt_75": rng.uniform(0, 100, n),
        "num_actv_rev_tl": rng.integers(0, 10, n), "bc_util": rng.uniform(0, 100, n),
        "num_rev_tl_bal_gt_0": rng.integers(0, 10, n), "mo_sin_rcnt_tl": rng.integers(0, 60, n),
        "total_rev_hi_lim": rng.uniform(0, 100000, n), "mo_sin_rcnt_rev_tl_op": rng.integers(0, 60, n),
        "mths_since_recent_bc": rng.integers(0, 60, n), "mo_sin_old_rev_tl_op": rng.integers(0, 200, n),
        "mths_since_recent_inq": rng.integers(0, 24, n),
        "default_flag": default_flag,
    })


def test_scorecard_beats_chance_on_held_out_data():
    from credit_risk.models.scorecard import train_scorecard, predict_scorecard
    train = _synthetic_labeled_df(3000, seed=0)
    test = _synthetic_labeled_df(1000, seed=1)
    model, encoder = train_scorecard(train, n_bins=5)
    pred = predict_scorecard(model, encoder, test)
    assert auc(test["default_flag"].to_pandas(), pred) > 0.6


def test_gbm_beats_chance_on_held_out_data():
    from credit_risk.models.gbm import train_gbm, predict_gbm
    train = _synthetic_labeled_df(3000, seed=0)
    valid = _synthetic_labeled_df(500, seed=2)
    test = _synthetic_labeled_df(1000, seed=1)
    model, features = train_gbm(train, valid)
    pred = predict_gbm(model, features, test)
    assert auc(test["default_flag"].to_pandas(), pred) > 0.6