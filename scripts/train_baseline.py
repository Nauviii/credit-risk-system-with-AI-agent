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
from credit_risk.evaluation.baselines import (
    auc_by_segment,
    auc_by_vintage,
    reference_baseline_table,
)
from credit_risk.evaluation.metrics import discrimination_report
from credit_risk.features.build_dataset import (
    APPLICATION_FEATURES,
    SCORECARD_FEATURES,
    application_features,
    assemble_feature_matrix,
    gbm_features,
)
from credit_risk.models.gbm import predict_gbm, train_gbm
from credit_risk.models.scorecard import predict_scorecard, train_scorecard
from credit_risk.tracking import start_run

_CONFIG_PATH = Path("configs/base.yaml")
_TUNED_PARAMS_PATHS = {
    "GBM (full)": Path("configs/gbm_best_params.yaml"),
    "GBM (application-only)": Path("configs/gbm_best_params_application.yaml"),
}
_SPLITS = ["train", "validation", "oot_test"]
# LendingClub's own risk output. Any model AUC must be read against these floors.
_REFERENCE_FEATURES = ["sub_grade", "grade", "int_rate", "fico_range_low"]


def _load_tuned_gbm_params(path: Path) -> dict | None:
    """Load hyperparameters from scripts/tune_gbm.py's output, if it has been run."""
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _print_reference_baselines(splits: dict[str, pl.DataFrame]) -> dict[str, float]:
    """Single-feature AUC floors; returns sub_grade's AUC per split for lift reporting."""
    print("\n=== reference baselines (no model fitted) ===")
    floors = {}
    for split_name, df in splits.items():
        rows = reference_baseline_table(df, _REFERENCE_FEATURES).to_dicts()
        print(split_name, {r["feature"]: round(r["auc"], 4) for r in rows if r["auc"] is not None})
        floors[split_name] = next((r["auc"] for r in rows if r["feature"] == "sub_grade"), None)
    return floors


def _evaluate_and_log(
    name: str, splits: dict[str, pl.DataFrame], predict_fn, floors: dict[str, float] | None = None
) -> None:
    """Print and mlflow-log discrimination metrics for one model across every split.

    `lift` is AUC above sub_grade used alone. It is the headline number: raw AUC is not
    comparable across splits because later vintages are intrinsically easier to
    discriminate for every scorer, LendingClub's own grade included.
    """
    print(f"\n=== {name} ===")
    for split_name in _SPLITS:
        df = splits[split_name]
        pred = predict_fn(df)
        metrics = discrimination_report(df["default_flag"].to_pandas(), pred)
        line = {k: round(v, 4) for k, v in metrics.items()}
        if floors and floors.get(split_name):
            line["lift_over_sub_grade"] = round(metrics["auc"] - floors[split_name], 4)
            mlflow.log_metric(f"{split_name}_lift_over_sub_grade", line["lift_over_sub_grade"])
        print(split_name, line)
        for k, v in metrics.items():
            mlflow.log_metric(f"{split_name}_{k}", v)
        if split_name == "train":
            # Train spans two vintages; a pooled AUC below every per-year AUC is an
            # aggregation artefact, not evidence of poor within-year discrimination.
            print("  per-vintage:", auc_by_vintage(df, pred).to_dicts())
        if split_name == "oot_test":
            # Decides whether one model can serve both terms, or whether the 24-month
            # horizon has made the label mean different things on each side.
            segment = auc_by_segment(
                df.with_columns(pl.col("term_months").cast(pl.Utf8)), pred, "term_months"
            )
            print("  per-term:", segment.to_dicts())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to accepted loans CSV/CSV.GZ")
    args = parser.parse_args()

    df = load_raw_accepted_loans(Path(args.input))
    labeled = build_target(df, _CONFIG_PATH)
    final = assemble_feature_matrix(labeled, _CONFIG_PATH)
    splits = {name: final.filter(pl.col("split") == name) for name in _SPLITS}

    print("=== split sizes ===")
    for name, split_df in splits.items():
        print(f"{name}: n={split_df.height}, default_rate={split_df['default_flag'].mean():.4f}")

    floors = _print_reference_baselines(splits)

    for label, feature_set, run_name in (
        ("Scorecard (full)", SCORECARD_FEATURES, "scorecard_full_v2"),
        ("Scorecard (application-only)", APPLICATION_FEATURES, "scorecard_application_v2"),
    ):
        with start_run(
            run_name, {"model_type": "logistic_regression", "n_features": len(feature_set)}
        ):
            model, encoder = train_scorecard(splits["train"], features=feature_set)
            _evaluate_and_log(
                label, splits, lambda d, m=model, e=encoder: predict_scorecard(m, e, d), floors
            )

    # Both pools, so the 2x2 (linear vs GBM) x (full vs application-only) grid is complete.
    # Without the application-only GBM, a weak application-only scorecard cannot be told
    # apart from application data genuinely carrying less signal than LendingClub's grade.
    for label, pool, run_name in (
        ("GBM (full)", gbm_features(final), "gbm_full_v2"),
        ("GBM (application-only)", application_features(final), "gbm_application_v2"),
    ):
        tuned_params = _load_tuned_gbm_params(_TUNED_PARAMS_PATHS[label])
        gbm_params_source = "tuned" if tuned_params else "default (untuned)"
        print(f"\n{label} hyperparameters: {gbm_params_source}")
        with start_run(
            run_name,
            {
                "model_type": "lightgbm",
                "n_features": len(pool),
                "params_source": gbm_params_source,
            },
        ):
            gbm_model, features = train_gbm(
                splits["train"], splits["validation"], params=tuned_params, features=pool
            )
            mlflow.log_metric("best_iteration", gbm_model.best_iteration)
            _evaluate_and_log(
                label, splits, lambda d, m=gbm_model, f=features: predict_gbm(m, f, d), floors
            )


if __name__ == "__main__":
    main()
