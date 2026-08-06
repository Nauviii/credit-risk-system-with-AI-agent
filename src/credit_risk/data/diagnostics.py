"""Vintage-level data diagnostics run BEFORE target definition and feature selection.

These four tables decide three parameters that are currently assumed rather than
measured: the outcome horizon H, the train window start year, and whether the
2016 OOT set is trustworthy at all.

Unlike evaluation/diagnostics.py (which inspects a fitted scorecard), everything
here operates on the RAW ingested frame - target.build_target() drops exactly the
censored rows we need to count, so it must not run first.
"""

import polars as pl

_BAD_STATUSES = ["Charged Off", "Default"]
_CENSORED_STATUSES = [
    "Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)",
]
_LEGACY_PREFIX = "Does not meet the credit policy. Status:"


def _clean_status() -> pl.Expr:
    """Strip the legacy policy prefix so old vintages' statuses match the modern labels."""
    return (
        pl.col("loan_status")
        .str.replace(_LEGACY_PREFIX, "", literal=True)
        .str.strip_chars()
    )


def _issue_date() -> pl.Expr:
    """Parse issue_d ('Dec-2015') to a Date."""
    return pl.col("issue_d").str.strptime(pl.Date, "%b-%Y", strict=False)


def _months(col: str) -> pl.Expr:
    """Absolute month index of a Date column, for month arithmetic without day noise."""
    return pl.col(col).dt.year() * 12 + pl.col(col).dt.month()


def censoring_by_vintage(df: pl.DataFrame) -> pl.DataFrame:
    """Observed vs lower-bound default rate per issue year, exposing upward censoring bias.

    dr_observed  = defaults / matured  (what build_target currently reports)
    dr_lower_bnd = defaults / issued   (true rate if no censored loan ever defaults)
    The true vintage rate lies between the two; the gap is the size of the bias.
    """
    return (
        df.with_columns(_clean_status().alias("status"), _issue_date().alias("d0"))
        .with_columns(pl.col("d0").dt.year().alias("issue_year"))
        .group_by("issue_year")
        .agg(
            pl.len().alias("n_issued"),
            (~pl.col("status").is_in(_CENSORED_STATUSES)).sum().alias("n_matured"),
            pl.col("status").is_in(_BAD_STATUSES).sum().alias("n_default"),
            pl.col("status").is_in(_CENSORED_STATUSES).mean().alias("pct_censored"),
        )
        .with_columns(
            (pl.col("n_default") / pl.col("n_matured")).alias("dr_observed"),
            (pl.col("n_default") / pl.col("n_issued")).alias("dr_lower_bnd"),
        )
        .with_columns(
            (pl.col("dr_observed") - pl.col("dr_lower_bnd")).alias("bias_gap")
        )
        .sort("issue_year")
    )


def feature_availability(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Null rate per feature per issue year - finds the year each feature became collectable.

    A feature whose null rate collapses from ~1.0 to ~0.0 at some year is vintage
    dependent: its WOE 'missing' bin encodes calendar time, not credit risk.
    """
    present = [f for f in features if f in df.columns]
    return (
        df.with_columns(_issue_date().dt.year().alias("issue_year"))
        .group_by("issue_year")
        .agg(
            pl.len().alias("n"),
            *[pl.col(f).is_null().mean().alias(f) for f in present],
        )
        .sort("issue_year")
    )


def first_reliable_year(availability: pl.DataFrame, max_null_rate: float = 0.05) -> pl.DataFrame:
    """Earliest issue year where each feature's null rate falls below max_null_rate."""
    features = [c for c in availability.columns if c not in ("issue_year", "n")]
    rows = [
        {
            "feature": f,
            "first_year": (
                availability.filter(pl.col(f) < max_null_rate)["issue_year"].min()
            ),
        }
        for f in features
    ]
    return pl.DataFrame(rows, schema={"feature": pl.Utf8, "first_year": pl.Int32}).sort(
        "first_year", "feature", nulls_last=True
    )


def default_hazard_by_mob(
    df: pl.DataFrame, vintages: list[int], co_lag: int = 5
) -> pl.DataFrame:
    """Cumulative share of a vintage's defaults resolved by month-on-book, per term.

    Run on fully matured vintages only. Read off the MOB where cum_share reaches
    ~0.90 to choose the outcome horizon H. Also validates the charge-off proxy:
    the per-MOB counts should peak around month 8-18, not be flat.
    """
    return (
        df.with_columns(
            _clean_status().alias("status"),
            _issue_date().alias("d0"),
            pl.col("last_pymnt_d").str.strptime(pl.Date, "%b-%Y", strict=False).alias("d1"),
            pl.col("term").str.extract(r"(\d+)", 1).cast(pl.Int32).alias("term_months"),
        )
        .filter(
            pl.col("d0").dt.year().is_in(vintages)
            & pl.col("status").is_in(_BAD_STATUSES)
            & pl.col("d1").is_not_null()
        )
        .with_columns((_months("d1") + co_lag - _months("d0")).alias("mob"))
        .filter(pl.col("mob") > 0)
        .group_by("term_months", "mob")
        .agg(pl.len().alias("n_default"))
        .sort("term_months", "mob")
        .with_columns(
            (pl.col("n_default").cum_sum() / pl.col("n_default").sum())
            .over("term_months")
            .alias("cum_share")
        )
    )


def prepayment_risk_link(
    df: pl.DataFrame, vintages: list[int], early_margin: int = 6
) -> pl.DataFrame:
    """Compare early payers vs on-schedule payers on FICO - tests whether censoring inflates AUC.

    Censoring only distorts measured discrimination if who-prepays correlates with
    risk. Run on fully matured vintages. A materially higher mean FICO among early
    payers means the OOT 'matured' subset is a risk-widened sample and its AUC is
    optimistic, not just its default rate.
    """
    return (
        df.with_columns(
            _clean_status().alias("status"),
            _issue_date().alias("d0"),
            pl.col("last_pymnt_d").str.strptime(pl.Date, "%b-%Y", strict=False).alias("d1"),
            pl.col("term").str.extract(r"(\d+)", 1).cast(pl.Int32).alias("term_months"),
        )
        .filter(
            pl.col("d0").dt.year().is_in(vintages)
            & (pl.col("status") == "Fully Paid")
            & pl.col("d1").is_not_null()
        )
        .with_columns(
            (
                (_months("d1") - _months("d0"))
                < (pl.col("term_months") - early_margin)
            ).alias("early_payer")
        )
        .group_by("term_months", "early_payer")
        .agg(
            pl.len().alias("n"),
            pl.col("fico_range_low").mean().alias("mean_fico"),
            pl.col("dti").mean().alias("mean_dti"),
        )
        .sort("term_months", "early_payer")
    )


def charge_off_proxy_check(df: pl.DataFrame, vintages: list[int]) -> pl.DataFrame:
    """Test whether last_pymnt_d is contaminated by post-charge-off recovery payments.

    build_target's fixed-horizon logic needs default TIMING, not just occurrence, so the
    `last_pymnt_d + co_lag` proxy has to be validated before H can be trusted.

    Reads: `gap` = mob_last_pymnt - n_pymt_implied. A small positive gap is expected in
    both groups (a defaulter misses ~4 payments before charge-off). What matters is the
    DIFFERENCE between groups: if the recoveries>0 group's median gap exceeds the
    recoveries==0 group's by more than ~3 months, last_pymnt_d is tracking recoveries
    and the proxy shifts default timing too late.

    Uses total_pymnt/recoveries for TARGET diagnostics only; both stay banned as features.
    """
    return (
        df.with_columns(
            _clean_status().alias("status"),
            _issue_date().alias("d0"),
            pl.col("last_pymnt_d").str.strptime(pl.Date, "%b-%Y", strict=False).alias("d1"),
        )
        .filter(
            pl.col("d0").dt.year().is_in(vintages)
            & pl.col("status").is_in(_BAD_STATUSES)
            & pl.col("d1").is_not_null()
            & (pl.col("installment") > 0)
            & pl.col("total_pymnt").is_not_null()
        )
        .with_columns(
            (_months("d1") - _months("d0")).alias("mob_last_pymnt"),
            (
                (pl.col("total_pymnt") - pl.col("recoveries").fill_null(0.0))
                / pl.col("installment")
            ).alias("n_pymt_implied"),
            (pl.col("recoveries").fill_null(0.0) > 0).alias("has_recovery"),
        )
        .with_columns(
            (pl.col("mob_last_pymnt") - pl.col("n_pymt_implied")).alias("gap")
        )
        .group_by("has_recovery")
        .agg(
            pl.len().alias("n"),
            pl.col("mob_last_pymnt").median().alias("med_mob_last_pymnt"),
            pl.col("n_pymt_implied").median().alias("med_n_pymt_implied"),
            pl.col("gap").median().alias("med_gap"),
            pl.col("gap").quantile(0.25).alias("gap_p25"),
            pl.col("gap").quantile(0.75).alias("gap_p75"),
        )
        .sort("has_recovery")
    )