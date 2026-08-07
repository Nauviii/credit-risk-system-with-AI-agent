"""Actual versus expected default rate. The check PSI structurally cannot perform.

Phase 5 measured the failure this module exists for. Between train (2013-2014) and the 2016
out-of-time set, every distribution held: score PSI 0.0004, highest feature PSI 0.081, Brier
resolution identical at 0.00505 and 0.00506. Yet the default rate rose from 8.9% to 11.4%.
The same borrower profile simply defaulted more often - concept drift, not covariate shift.

A monitoring system watching only distributions would have shown all green while realised
losses rose 27% relative. Distribution stability and outcome stability are different
questions and need separate instruments.

The complication is timing: the model predicts default within 24 months, so a vintage's
verdict arrives two years after it is written. `early_warning` closes that gap by comparing
observed defaults at month-on-book M against the share of the eventual 24-month PD that the
hazard curve says should have arrived by M.
"""

import numpy as np
import polars as pl

_TARGET = "default_flag"


# Graded, not a single cliff. The 2016 vintage backtest landed at a ratio of 1.0997 against
# a 0.10 threshold and was classified "not material" by three thousandths - which is an
# artefact of where the line was drawn, not a finding about 2016. Two levels plus the raw
# ratio keeps the judgement with the reader instead of hiding it behind one boolean.
WATCH_RATIO_TOLERANCE = 0.05
MATERIAL_RATIO_TOLERANCE = 0.10


def expected_vs_actual(
    predicted_pd: np.ndarray,
    actual: np.ndarray,
    coverage: float = 1.0,
    tolerance: float = MATERIAL_RATIO_TOLERANCE,
) -> dict:
    """Compare realised defaults against what the model predicted, with a significance test.

    Defaults are independent Bernoulli draws with DIFFERENT probabilities, so the count is
    Poisson-binomial: mean = sum(p), variance = sum(p*(1-p)). Using a single pooled rate
    instead would overstate the variance and hide real deterioration.

    `coverage` scales the expectation when the observation window is shorter than the model's
    horizon - pass the share of eventual defaults expected by that point (see early_warning).

    Significance is NOT materiality, and at portfolio scale the two come apart badly. On
    434,407 loans a 1.4% deviation reaches z = -3.3 - unambiguously real, and of no interest
    to anyone. Alerting on z alone makes every vintage red and the dashboard useless. So an
    alert requires BOTH: statistically distinguishable (|z| > 2) and economically meaningful
    (`ratio_band` above "stable"). Every component is returned so the distinction stays
    visible rather than being collapsed into one flag.

    One caveat on interpretation: if the PD being tested has been anchored to a long-run
    average, it is deliberately not the point-in-time rate of any single vintage, so a steady
    gap against one cohort is the anchor working, not the model failing. Monitor a
    point-in-time calibrated PD if the question is "was this vintage as predicted".
    """
    scaled = np.clip(predicted_pd * coverage, 1e-9, 1 - 1e-9)
    expected = float(scaled.sum())
    variance = float((scaled * (1 - scaled)).sum())
    observed = float(actual.sum())
    z = float((observed - expected) / np.sqrt(variance)) if variance > 0 else 0.0
    deviation = abs(observed / expected - 1.0) if expected > 0 else 0.0
    band = (
        "material"
        if deviation > tolerance
        else "watch" if deviation > WATCH_RATIO_TOLERANCE else "stable"
    )
    return {
        "n": int(len(actual)),
        "expected_defaults": round(expected, 1),
        "actual_defaults": int(observed),
        "expected_rate": expected / len(actual) if len(actual) else None,
        "actual_rate": observed / len(actual) if len(actual) else None,
        "ratio": observed / expected if expected > 0 else None,
        "z_score": round(z, 2),
        "significant": bool(abs(z) > 2.0),
        "ratio_band": band,
        "alert": bool(abs(z) > 2.0 and band != "stable"),
    }


def vintage_performance(
    df: pl.DataFrame, pd_column: str, vintage_column: str, coverage: float = 1.0
) -> pl.DataFrame:
    """expected_vs_actual per vintage, which is the unit a credit portfolio is managed in.

    A pooled figure hides the pattern that matters: a single deteriorating cohort inside a
    stable book, which is exactly what 2016 was.
    """
    rows = []
    for vintage in sorted(df[vintage_column].unique().drop_nulls().to_list()):
        part = df.filter(pl.col(vintage_column) == vintage)
        rows.append(
            {
                "vintage": str(vintage),
                **expected_vs_actual(
                    part[pd_column].to_numpy(), part[_TARGET].to_numpy(), coverage=coverage
                ),
            }
        )
    return pl.DataFrame(rows)


def early_warning(
    df: pl.DataFrame,
    pd_column: str,
    mob_column: str,
    observed_mob: int,
    hazard_coverage: dict[int, float],
    vintage_column: str | None = None,
    default_column: str = _TARGET,
) -> pl.DataFrame:
    """Compare defaults seen by month `observed_mob` against what should have arrived by then.

    Waiting for the 24-month window to close means learning about a bad vintage two years
    late. The hazard curve measured on a fully matured vintage says what share of eventual
    defaults lands by each month on book, so a cohort only six months old can already be
    tested against a properly scaled expectation.

    `hazard_coverage` maps month-on-book to cumulative share of defaults WITHIN THE MODEL'S
    HORIZON - not of lifetime defaults, which is the different quantity business.py needs.
    Take it from a fully matured vintage. Its accuracy is the limit of this method: it assumes
    default timing is stable across vintages, which Phase 5 verified between 2013-2014 and
    2016 but which is itself worth monitoring.

    `mob_column` alone is not enough to identify a default. It carries a month for every
    resolved loan, including one paid off early, so counting `mob <= M` alone counted early
    payers as defaults - at MOB 6 that turned 1,100 real defaults into 8,834 and produced a
    z of +243. Both conditions are required: the loan is bad AND it went bad by month M. In
    production `default_column` is "charged off as of today", which has the same meaning.
    """
    coverage = hazard_coverage.get(observed_mob)
    if coverage is None:
        raise KeyError(f"no hazard coverage for MOB {observed_mob}; have {sorted(hazard_coverage)}")

    if default_column not in df.columns:
        raise KeyError(f"early_warning needs a default indicator column '{default_column}'")

    observed = df.with_columns(
        (
            (pl.col(default_column) == 1)
            & pl.col(mob_column).is_not_null()
            & (pl.col(mob_column) <= observed_mob)
        )
        .cast(pl.Int8)
        .alias(_TARGET)
    )
    if vintage_column is None:
        return pl.DataFrame(
            [
                {
                    "observed_mob": observed_mob,
                    "coverage": coverage,
                    **expected_vs_actual(
                        observed[pd_column].to_numpy(), observed[_TARGET].to_numpy(), coverage
                    ),
                }
            ]
        )
    return vintage_performance(observed, pd_column, vintage_column, coverage).with_columns(
        pl.lit(observed_mob).alias("observed_mob"), pl.lit(coverage).alias("coverage")
    )