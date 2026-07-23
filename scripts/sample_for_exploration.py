import argparse
import gzip
import shutil
from pathlib import Path

import polars as pl

from credit_risk.data.ingestion import load_raw_accepted_loans


def stratified_sample(df: pl.DataFrame, n: int, seed: int = 42) -> pl.DataFrame:
    """Sample n rows preserving loan_status x issue-year proportions so the subset stays representative."""
    df = df.with_columns(
        (pl.col("loan_status") + "_" + pl.col("issue_d").str.split("-").list.last()).alias("_stratum")
    )
    frac = min(n / df.height, 1.0)
    parts = [group.sample(fraction=frac, seed=seed) for _, group in df.group_by("_stratum")]
    return pl.concat(parts).drop("_stratum")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/interim/accepted_sample.csv.gz")
    parser.add_argument("--n", type=int, default=300_000)
    args = parser.parse_args()

    df = load_raw_accepted_loans(Path(args.input))
    sample = stratified_sample(df, args.n)

    output_path = Path(args.output)
    tmp_csv = output_path.with_suffix("")
    sample.write_csv(tmp_csv)
    with open(tmp_csv, "rb") as f_in, gzip.open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    tmp_csv.unlink()

    print(f"sampled {sample.height} rows from {df.height} total, written to {output_path}")


if __name__ == "__main__":
    main()