"""Tests for DeLong: AUCs must match sklearn, and the paired test must beat an unpaired one."""

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk.evaluation.significance import (
    auc_confidence_interval,
    delong_auc_test,
    gini_by_period,
)


def _sample(n: int = 20_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-2.0 + 1.2 * x)))).astype(int)
    return y, x, rng


def test_auc_matches_sklearn():
    """The DeLong estimator must reproduce the AUC, not merely approximate it."""
    y, x, _ = _sample()
    assert auc_confidence_interval(y, x)["auc"] == pytest.approx(roc_auc_score(y, x), abs=1e-9)


def test_auc_matches_sklearn_with_heavy_ties():
    """Midrank handling is the whole reason the variance is exact under ties."""
    y, x, _ = _sample()
    tied = np.round(x, 1)
    assert auc_confidence_interval(y, tied)["auc"] == pytest.approx(
        roc_auc_score(y, tied), abs=1e-9
    )


def test_confidence_interval_brackets_the_estimate_and_narrows_with_n():
    y, x, _ = _sample(n=4_000, seed=1)
    small = auc_confidence_interval(y, x)
    y, x, _ = _sample(n=64_000, seed=1)
    large = auc_confidence_interval(y, x)
    assert small["ci_lower"] < small["auc"] < small["ci_upper"]
    assert (large["ci_upper"] - large["ci_lower"]) < (small["ci_upper"] - small["ci_lower"]) / 2


def test_identical_scores_produce_no_difference():
    y, x, _ = _sample()
    result = delong_auc_test(y, x, x.copy())
    assert result["difference"] == pytest.approx(0.0, abs=1e-12)
    assert not result["significant"]


def test_a_real_difference_is_detected():
    y, x, rng = _sample()
    noise = rng.normal(size=len(y))
    result = delong_auc_test(y, x, noise)
    assert result["auc_a"] > result["auc_b"]
    assert result["significant"] and result["p_value"] < 1e-10


def test_paired_test_is_sharper_than_treating_the_aucs_as_independent():
    """The point of DeLong: two scorers sharing signal are more comparable than their own
    intervals suggest, and an unpaired standard error would hide a real difference."""
    y, x, rng = _sample(n=40_000, seed=2)
    slightly_better = x + 0.15 * rng.normal(size=len(y))

    paired = delong_auc_test(y, x, slightly_better)
    independent = np.sqrt(
        auc_confidence_interval(y, x)["std_error"] ** 2
        + auc_confidence_interval(y, slightly_better)["std_error"] ** 2
    )
    assert paired["std_error"] < independent


def test_difference_interval_excludes_zero_exactly_when_significant():
    y, x, rng = _sample(n=30_000, seed=3)
    for other in (x + 0.5 * rng.normal(size=len(y)), rng.normal(size=len(y))):
        result = delong_auc_test(y, x, other)
        excludes_zero = result["ci_lower"] > 0 or result["ci_upper"] < 0
        assert excludes_zero == result["significant"]


def test_gini_by_period_reports_each_period_and_a_pooled_row():
    y, x, _ = _sample(n=20_000, seed=4)
    period = np.where(np.arange(len(y)) < len(y) // 2, "2016Q1", "2016Q2")
    df = pl.DataFrame({"default_flag": y, "period": period})
    table = gini_by_period(df, x, "period")
    assert table["period"].to_list() == ["2016Q1", "2016Q2", "pooled"]
    assert (table["ci_lower"] < table["gini"]).all()
    assert (table["gini"] < table["ci_upper"]).all()
    # pooled has the most data, so the tightest interval
    assert table["ci_width"][-1] < table["ci_width"][0]


def test_period_with_a_single_class_is_skipped_not_crashed():
    y, x, _ = _sample(n=10_000, seed=5)
    y[: len(y) // 2] = 0  # first period all good
    period = np.where(np.arange(len(y)) < len(y) // 2, "empty", "real")
    df = pl.DataFrame({"default_flag": y, "period": period})
    assert gini_by_period(df, x, "period")["period"].to_list() == ["real", "pooled"]


def test_mlflow_metric_names_are_sanitised():
    """MLflow rejects parentheses, and it rejects them after the run has done its work."""
    import re
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path("scripts").resolve()))
    source = Path("scripts/train_baseline.py").read_text()
    namespace: dict = {"re": re}
    exec(  # noqa: S102 - importing the script would pull in mlflow and the whole pipeline
        source[source.index("def _metric_name") : source.index("def _significance_report")],
        namespace,
    )
    name = namespace["_metric_name"]("champion vs sub_grade alone")
    assert name == "delong_champion_vs_sub_grade_alone"
    assert re.fullmatch(r"[A-Za-z0-9_\-./ ]+", name)
    assert re.fullmatch(r"[A-Za-z0-9_\-./ ]+", namespace["_metric_name"]("GBM (application-only)"))
