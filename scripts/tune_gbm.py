"""Tune GBM hyperparameters. Produces a decision file; never trains the reported model.

Two protocol fixes over the previous version, both forced by measured results:

1. NO DUAL USE OF 2015. Previously the same validation set drove early stopping AND was
   the Optuna objective, so every trial could stop at the point that flattered its own
   score. Early stopping now runs on an inner temporal slice of train (the last quarter
   of 2014); 2015 is scored once per trial and nothing else. The residual limitation is
   inherent to model selection: after N trials the winner's 2015 AUC is optimistically
   biased. OOT 2016 stays untouched throughout, which is what protects the final number.

2. SEARCH SPACE RE-CENTRED FOR THE NEW BASE RATE. The previous best params were tuned
   against the old censored target at a 17% default rate. At 8.9%, min_child_samples=155
   leaves roughly 14 defaults per leaf - far weaker regularization than the same number
   used to buy. Measured consequence: OOT lift over sub_grade collapses from +0.104 on
   train to +0.025 (full pool) and from +0.098 to +0.007 (application-only). The bounds
   below scale with the bad count instead of the row count, and num_boost_round rises to
   4000 because the previous run stopped at 1911/2000 - against the cap, not converged.

Usage:
    python scripts/tune_gbm.py --input data/raw/accepted_2007_to_2018Q4.csv --pool full
    python scripts/tune_gbm.py --input data/raw/accepted_2007_to_2018Q4.csv --pool application
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import mlflow
import optuna
import polars as pl
import yaml
from optuna_integration import LightGBMPruningCallback

from credit_risk.data.ingestion import load_raw_accepted_loans
from credit_risk.data.target import build_target
from credit_risk.evaluation.baselines import score_only_auc
from credit_risk.evaluation.metrics import discrimination_report
from credit_risk.features.build_dataset import (
    application_features, assemble_feature_matrix, gbm_features,
)
from credit_risk.models.gbm import predict_gbm, prepare_lgb_frame, train_gbm
from credit_risk.tracking import start_run

_CONFIG_PATH = Path("configs/base.yaml")
_OUTPUT_PATHS = {
    "full": Path("configs/gbm_best_params.yaml"),
    "application": Path("configs/gbm_best_params_application.yaml"),
}
# Last quarter of train, held out inside each trial purely for early stopping.
_INNER_VALID_START = "Oct-2014"
_NUM_BOOST_ROUND = 4000


def _split_inner(train: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Temporal inner split of train, so early stopping never touches the scoring set."""
    cutoff = pl.lit(_INNER_VALID_START).str.strptime(pl.Date, "%b-%Y")
    issued = pl.col("issue_d").cast(pl.Utf8).str.strptime(pl.Date, "%b-%Y", strict=False)
    return train.filter(issued < cutoff), train.filter(issued >= cutoff)


def _objective(trial, sets: dict, n_bad: int) -> float:
    """One trial: early-stop on the inner slice, score on 2015. Returns 2015 AUC."""
    params = {
        "objective": "binary", "metric": "auc", "verbosity": -1, "seed": 42,
        "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        # expressed as a share of the BAD count, not the row count: a leaf needs enough
        # defaults to estimate a rate, and that is what changed when the target did
        "min_child_samples": trial.suggest_int(
            "min_child_samples", max(20, n_bad // 500), max(200, n_bad // 20), log=True
        ),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 50.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 50.0, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    }
    inner_train = lgb.Dataset(sets["X_inner_train"], label=sets["y_inner_train"])
    inner_valid = lgb.Dataset(sets["X_inner_valid"], label=sets["y_inner_valid"], reference=inner_train)

    model = lgb.train(
        params, inner_train, num_boost_round=_NUM_BOOST_ROUND, valid_sets=[inner_valid],
        callbacks=[
            lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0),
            LightGBMPruningCallback(trial, "auc", valid_name="valid_0"),
        ],
    )
    trial.set_user_attr("best_iteration", model.best_iteration)
    scores = model.predict(sets["X_score"], num_iteration=model.best_iteration)
    return discrimination_report(sets["y_score"], scores)["auc"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--pool", choices=["full", "application"], default="full")
    args = parser.parse_args()

    df = load_raw_accepted_loans(Path(args.input))
    final = assemble_feature_matrix(build_target(df, _CONFIG_PATH), _CONFIG_PATH)
    train = final.filter(pl.col("split") == "train")
    valid = final.filter(pl.col("split") == "validation")
    oot = final.filter(pl.col("split") == "oot_test")

    features = gbm_features(final) if args.pool == "full" else application_features(final)
    inner_train, inner_valid = _split_inner(train)
    n_bad = int(train["default_flag"].sum())
    print(f"pool={args.pool} ({len(features)} features)  inner_train={inner_train.height} "
          f"inner_valid={inner_valid.height}  train_bads={n_bad}")

    sets = {
        "X_inner_train": prepare_lgb_frame(inner_train, features),
        "y_inner_train": inner_train["default_flag"].to_pandas(),
        "X_inner_valid": prepare_lgb_frame(inner_valid, features),
        "y_inner_valid": inner_valid["default_flag"].to_pandas(),
        "X_score": prepare_lgb_frame(valid, features),
        "y_score": valid["default_flag"].to_pandas(),
    }

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    with start_run(f"gbm_tuning_{args.pool}", {"n_trials": args.n_trials, "pool": args.pool}):
        study.optimize(lambda t: _objective(t, sets, n_bad), n_trials=args.n_trials)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_validation_auc", study.best_value)
        print(f"\nbest validation AUC: {study.best_value:.4f}")
        print("best params:", study.best_params)

        # Retrain under the protocol train_baseline.py uses, then report against the
        # sub_grade floor. Raw AUC is not comparable across splits; lift is.
        best_params = {"objective": "binary", "metric": "auc", "verbosity": -1, "seed": 42, **study.best_params}
        model, _ = train_gbm(train, valid, params=best_params, features=features)
        print(f"\n=== Tuned GBM ({args.pool}) ===  best_iteration={model.best_iteration}/{_NUM_BOOST_ROUND}")
        lifts = {}
        for name, split_df in [("train", train), ("validation", valid), ("oot_test", oot)]:
            metrics = discrimination_report(
                split_df["default_flag"].to_pandas(), predict_gbm(model, features, split_df)
            )
            floor = score_only_auc(split_df, "sub_grade")["auc"]
            lifts[name] = metrics["auc"] - floor
            print(name, {k: round(v, 4) for k, v in metrics.items()}, "lift", round(lifts[name], 4))
            for k, v in metrics.items():
                mlflow.log_metric(f"tuned_{name}_{k}", v)
        # The number that decides whether this run improved anything.
        print(f"\nlift collapse train -> oot: {lifts['train']:.4f} -> {lifts['oot_test']:.4f}")
        mlflow.log_metric("lift_collapse", lifts["train"] - lifts["oot_test"])

    output_path = _OUTPUT_PATHS[args.pool]
    with open(output_path, "w") as f:
        yaml.safe_dump({**study.best_params, "seed": 42}, f)
    print(f"best params written to {output_path}")


if __name__ == "__main__":
    main()