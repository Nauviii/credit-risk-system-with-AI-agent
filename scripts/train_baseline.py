"""Train scorecard baseline and GBM champion, evaluate on train/validation/oot_test, log to MLflow.

This is the script behind every number in docs/modeling_findings.md - run it to
reproduce them exactly, not just to read about them.

Usage:
    python scripts/train_baseline.py --input data/raw/accepted_2007_to_2018Q4.csv
"""

import argparse
import re
from pathlib import Path

import mlflow
import numpy as np
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
from credit_risk.evaluation.significance import (
    auc_confidence_interval,
    delong_auc_test,
    gini_by_period,
)
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


def _metric_name(label: str) -> str:
    """MLflow metric names allow only alphanumerics, _ - . / and spaces.

    Model labels here carry parentheses ("GBM (application-only)"), which MLflow rejects at
    write time - after the run has already done its work. Derived from the comparison label
    rather than the model names so the logged key stays readable.
    """
    return "delong_" + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def _significance_report(oot: pl.DataFrame, predictions: dict[str, np.ndarray]) -> None:
    """Test the differences the project's conclusions rest on, instead of asserting them.

    Every model is scored on the same 434,407 loans, so their AUCs are correlated and DeLong
    is the correct test - an unpaired comparison would overstate the standard error of the
    difference. Per-quarter Gini is printed alongside because at this sample size almost any
    difference is significant, and an edge smaller than the quarter-to-quarter swing is real
    without being something a portfolio would notice.
    """
    y = oot["default_flag"].to_numpy()

    print("\n=== OOT AUC with DeLong confidence intervals ===")
    for name, scores in predictions.items():
        r = auc_confidence_interval(y, scores)
        print(f"{name:<30} {r['auc']:.4f}  [{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")

    # The three comparisons that decided the champion and the project's headline claim.
    comparisons = [
        ("champion vs sub_grade alone", "GBM (application-only)", "sub_grade"),
        ("lender-derived contribution", "GBM (full)", "GBM (application-only)"),
        ("GBM vs scorecard, app-only", "GBM (application-only)", "Scorecard (application-only)"),
    ]
    print("\n=== paired AUC differences (DeLong) ===")
    for label, a, b in comparisons:
        if a not in predictions or b not in predictions:
            continue
        r = delong_auc_test(y, predictions[a], predictions[b])
        print(
            f"{label:<30} {r['difference']:+.4f}  "
            f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]  "
            f"p={r['p_value']:.2e}  {'significant' if r['significant'] else 'not significant'}"
        )
        mlflow.log_metric(_metric_name(label), r["difference"])
        mlflow.log_metric(f"{_metric_name(label)}_p_value", r["p_value"])

    quarters = oot.with_columns(
        (
            pl.col("issue_d")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%b-%Y", strict=False)
            .dt.year()
            .cast(pl.Utf8)
            + "Q"
            + (
                (
                    pl.col("issue_d")
                    .cast(pl.Utf8)
                    .str.strptime(pl.Date, "%b-%Y", strict=False)
                    .dt.month()
                    - 1
                )
                // 3
                + 1
            ).cast(pl.Utf8)
        ).alias("quarter")
    )
    champion = "GBM (application-only)"
    if champion in predictions:
        print(f"\n=== per-quarter Gini, {champion} ===")
        table = gini_by_period(quarters, predictions[champion], "quarter")
        print(table.to_pandas().to_string(index=False))
        within = table.filter(pl.col("period") != "pooled")
        print(
            f"quarter-to-quarter Gini swing: " f"{within['gini'].max() - within['gini'].min():.4f}"
        )


def _evaluate_and_log(
    name: str,
    splits: dict[str, pl.DataFrame],
    predict_fn,
    floors: dict[str, float] | None = None,
    collect: dict[str, np.ndarray] | None = None,
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
        if collect is not None and split_name == "oot_test":
            collect[name] = np.asarray(pred, dtype=float)
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
    oot_predictions: dict[str, np.ndarray] = {}

    for label, feature_set, run_name in (
        ("Scorecard (full)", SCORECARD_FEATURES, "scorecard_full_v2"),
        ("Scorecard (application-only)", APPLICATION_FEATURES, "scorecard_application_v2"),
    ):
        with start_run(
            run_name, {"model_type": "logistic_regression", "n_features": len(feature_set)}
        ):
            model, encoder = train_scorecard(splits["train"], features=feature_set)
            _evaluate_and_log(
                label,
                splits,
                lambda d, m=model, e=encoder: predict_scorecard(m, e, d),
                floors,
                collect=oot_predictions,
            )

    # sub_grade used directly is the floor every model is tested against, so it belongs in
    # the same collection - ranked lexicographically, which is its true risk order.
    oot = splits["oot_test"]
    if "sub_grade" in oot.columns:
        oot_predictions["sub_grade"] = oot["sub_grade"].rank("dense").to_numpy().astype(float)

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
                label,
                splits,
                lambda d, m=gbm_model, f=features: predict_gbm(m, f, d),
                floors,
                collect=oot_predictions,
            )

    with start_run("significance", {"n_models": len(oot_predictions)}):
        _significance_report(splits["oot_test"], oot_predictions)


if __name__ == "__main__":
    main()