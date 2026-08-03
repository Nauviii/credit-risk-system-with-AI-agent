"""Train scorecard baseline and GBM champion, evaluate on train/validation/oot_test, log to MLflow.

This is the script behind every number in docs/modeling_findings.md - run it to
reproduce them exactly, not just to read about them.

Usage:
    python scripts/train_baseline.py --input data/raw/accepted_2007_to_2018Q4.csv
"""

import argparse
from pathlib import Path

import mlflow
import polars as pl
import yaml

from credit_risk.data.ingestion import load_raw_accepted_loans
from credit_risk.data.target import build_target
from credit_risk.evaluation.metrics import discrimination_report
from credit_risk.features.build_dataset import assemble_feature_matrix, SCORECARD_FEATURES, gbm_features
from credit_risk.models.gbm import predict_gbm, train_gbm
from credit_risk.models.scorecard import predict_scorecard, train_scorecard
from credit_risk.tracking import start_run

_CONFIG_PATH = Path("configs/base.yaml")
_TUNED_PARAMS_PATH = Path("configs/gbm_best_params.yaml")
_SPLITS = ["train", "validation", "oot_test"]


def _load_tuned_gbm_params() -> dict | None:
    """Load hyperparameters from scripts/tune_gbm.py's output, if it has been run."""
    if not _TUNED_PARAMS_PATH.exists():
        return None
    with open(_TUNED_PARAMS_PATH) as f:
        return yaml.safe_load(f)


def _evaluate_and_log(name: str, splits: dict[str, pl.DataFrame], predict_fn) -> None:
    """Print and mlflow-log discrimination metrics for one model across every split."""
    print(f"\n=== {name} ===")
    for split_name in _SPLITS:
        df = splits[split_name]
        pred = predict_fn(df)
        metrics = discrimination_report(df["default_flag"].to_pandas(), pred)
        print(split_name, {k: round(v, 4) for k, v in metrics.items()})
        for k, v in metrics.items():
            mlflow.log_metric(f"{split_name}_{k}", v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to accepted loans CSV/CSV.GZ")
    args = parser.parse_args()

    df = load_raw_accepted_loans(Path(args.input))
    labeled = build_target(df)
    final = assemble_feature_matrix(labeled, _CONFIG_PATH)
    splits = {name: final.filter(pl.col("split") == name) for name in _SPLITS}

    print("=== split sizes ===")
    for name, split_df in splits.items():
        print(f"{name}: n={split_df.height}, default_rate={split_df['default_flag'].mean():.4f}")

    with start_run("scorecard_baseline_v1", {"model_type": "logistic_regression", "n_features": len(SCORECARD_FEATURES)}):
        model, encoder = train_scorecard(splits["train"])
        _evaluate_and_log("Scorecard", splits, lambda d: predict_scorecard(model, encoder, d))

    tuned_params = _load_tuned_gbm_params()
    gbm_params_source = "tuned (scripts/tune_gbm.py)" if tuned_params else "default (untuned)"
    print(f"\nGBM hyperparameters: {gbm_params_source}")

    with start_run("gbm_champion_v1", {
        "model_type": "lightgbm", "n_features": len(gbm_features(final)), "params_source": gbm_params_source,
    }):
        gbm_model, features = train_gbm(splits["train"], splits["validation"], params=tuned_params)
        mlflow.log_metric("best_iteration", gbm_model.best_iteration)
        _evaluate_and_log("GBM", splits, lambda d: predict_gbm(gbm_model, features, d))


if __name__ == "__main__":
    main()