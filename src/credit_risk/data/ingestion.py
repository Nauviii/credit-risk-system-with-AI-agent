"""Load raw Lending Club CSV/CSV.GZ into a validated Polars DataFrame."""

from pathlib import Path
import polars as pl

from credit_risk.data.schema import RAW_ACCEPTED_SCHEMA

_ID_PATTERN = r"^\d+$"


def load_raw_accepted_loans(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """Read accepted_*.csv(.gz), restrict to columns if given, and validate against schema.

    The raw file concatenates Lending Club's per-quarter exports, which occasionally
    leave a non-data summary row (e.g. "Total amount funded in policy code 1: ...")
    at chunk boundaries. `id` is read as string (never inferred as int) to avoid a
    parse crash, then rows where `id` isn't purely numeric are dropped as non-data.
    """
    read_columns = list(dict.fromkeys(["id", *columns])) if columns else None
    df = pl.read_csv(
        path,
        columns=read_columns,
        infer_schema_length=100_000,
        null_values=["", "n/a"],
        schema_overrides={"id": pl.Utf8},
    )
    df = df.filter(pl.col("id").str.contains(_ID_PATTERN))
    return RAW_ACCEPTED_SCHEMA.validate(df)