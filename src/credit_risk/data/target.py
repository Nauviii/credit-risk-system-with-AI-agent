"""Fixed-horizon PD target: default within H months of origination.

Replaces the previous maturity-filter definition. That version kept only loans whose
outcome had resolved by the data cutoff, which conditions on a post-origination event
and makes the measured default rate a function of vintage age rather than credit risk.
Measured bias (docs/vintage_censoring.csv, full data): 2013 +0.0pp, 2014 +1.0pp,
2015 +2.2pp, 2016 +7.6pp between defaults/matured and defaults/issued.

Under a fixed horizon every loan old enough to be observed for H months gets a label,
so no loan is dropped for being healthy, the definition means the same thing in every
vintage, and train/validation/OOT default rates become directly comparable.

Default timing comes from `last_pymnt_d + charge_off_lag_months`. Validated on 2012-2013
vintages (docs/charge_off_proxy_check.csv): median gap between months-to-last-payment and
payments-actually-made is 0.00 for loans without recoveries and 0.01 for loans with them,
so `last_pymnt_d` marks the true final scheduled payment and is not moved by post
charge-off collections.
"""

from pathlib import Path

import polars as pl
import yaml

_LEGACY_PREFIX = "Does not meet the credit policy. Status:"


def load_target_config(config_path: Path) -> dict:
    """Load the target definition block from configs/base.yaml."""
    with open(config_path) as f:
        return yaml.safe_load(f)["target"]


def _month_index(date_expr: pl.Expr) -> pl.Expr:
    """Absolute month number of a Date, for month arithmetic free of day-level noise."""
    return date_expr.dt.year() * 12 + date_expr.dt.month()


def build_target(df: pl.DataFrame, config_path: Path) -> pl.DataFrame:
    """Add default_flag = default within horizon_months, keeping only fully observable loans.

    Adds three columns:
      mob_observable - months of performance history available at the data cutoff
      mob_event      - month-on-book of the charge-off event (null status -> never paid)
      default_flag   - 1 if a bad status occurred with mob_event <= horizon, else 0

    mob_observable and mob_event describe the label itself and are registered in
    schema.TARGET_TIMING_COLUMNS so they can never reach a feature matrix.
    """
    cfg = load_target_config(config_path)
    horizon = int(cfg["horizon_months"])
    lag = int(cfg["charge_off_lag_months"])
    bad_statuses = list(cfg["bad_statuses"])
    year, month = (int(part) for part in str(cfg["data_cutoff"]).split("-"))
    cutoff_index = year * 12 + month

    issued = _month_index(
        pl.col("issue_d").cast(pl.Utf8).str.strptime(pl.Date, "%b-%Y", strict=False)
    )
    last_paid = _month_index(
        pl.col("last_pymnt_d").cast(pl.Utf8).str.strptime(pl.Date, "%b-%Y", strict=False)
    )

    out = df.with_columns(
        pl.col("loan_status")
        .str.replace(_LEGACY_PREFIX, "", literal=True)
        .str.strip_chars()
        .alias("loan_status"),
        (cutoff_index - issued).alias("mob_observable"),
        (last_paid - issued + lag).fill_null(lag).alias("mob_event"),
    )
    out = out.filter(pl.col("mob_observable") >= horizon)
    return out.with_columns(
        (pl.col("loan_status").is_in(bad_statuses) & (pl.col("mob_event") <= horizon))
        .cast(pl.Int8)
        .alias("default_flag")
    )
