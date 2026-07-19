"""Define the binary PD target from loan_status, handling censoring explicitly."""

import polars as pl
from credit_risk.data.schema import INDETERMINATE_STATUSES

_BAD_STATUSES = ["Charged Off", "Default"]
_GOOD_STATUSES = ["Fully Paid"]
_LEGACY_PREFIX = "Does not meet the credit policy. Status:"


def build_target(df: pl.DataFrame) -> pl.DataFrame:
    """Return df restricted to matured loans (Fully Paid/Charged Off) with default_flag column.

    Loans still Current/Late/In Grace Period are right-censored (outcome unknown) and dropped.
    """
    out = df.with_columns(
        pl.col("loan_status")
        .str.replace(_LEGACY_PREFIX, "", literal=True)
        .str.strip_chars()
        .alias("loan_status")
    )
    out = out.filter(~pl.col("loan_status").is_in(INDETERMINATE_STATUSES))
    out = out.filter(pl.col("loan_status").is_in(_BAD_STATUSES + _GOOD_STATUSES))
    out = out.with_columns(
        pl.col("loan_status").is_in(_BAD_STATUSES).cast(pl.Int8).alias("default_flag")
    )
    return out