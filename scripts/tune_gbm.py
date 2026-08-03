"""Tune GBM hyperparameters against the time-based validation set (2015), never OOT test.

Produces a decision (best hyperparameters), written to configs/gbm_best_params.yaml.
scripts/train_baseline.py consumes that file if present - this script never trains
the model that gets reported as final, matching the same separation used for
scorecard feature selection (notebooks/feature_selection_scorecard.py).

Objective is validation AUC, but the search space is centered on regularization
(num_leaves, min_child_samples, feature/bagging_fraction, lambda_l1/l2) because
docs/modeling_findings.md already shows default-param GBM overfitting (train-OOT
AUC gap 0.035) - the goal here is a better-generalizing model, not a higher
train-set score.

Usage:
    python scripts/tune_gbm.py --input data/raw/accepted_2007_to_2018Q4.csv --n-trials 50
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import optuna
import polars as pl
import yaml
from optuna_integration import LightGBMPruningCallback

from credit_risk.data.ingestion import load_raw_accepted_loans
from credit_risk.data.target import build_target
from credit_risk.evaluation.metrics import discrimination_report
from credit_risk.features.build_dataset import assemble_feature_matrix, gbm_features
from credit_risk.models.gbm import prepare_lgb_frame, predict_gbm, train_gbm
from credit_risk.tracking import start_run
import mlflow

_OUTPUT_PATH = Path("configs/gbm_best_params.yaml")


def _objective(trial: optuna.Trial, X_train, y_train, X_valid, y_valid) -> float:
    """One trial: sample regularization-focused params, train with pruning, return validation AUC."""
    params = {
        "objective": "binary", "metric": "auc", "verbosity": -1, "seed": 42,
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_valid, label=y_valid, reference=train_set)
    pruning_callback = LightGBMPruningCallback(trial, "auc", valid_name="valid_0")

    model = lgb.train(
        params, train_set, num_boost_round=2000, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0), pruning_callback],
    )
    return model.best_score["valid_0"]["auc"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    df = load_raw_accepted_loans(Path(args.input))
    labeled = build_target(df)
    final = assemble_feature_matrix(labeled, Path("configs/base.yaml"))
    train = final.filter(pl.col("split") == "train")
    valid = final.filter(pl.col("split") == "validation")
    oot = final.filter(pl.col("split") == "oot_test")

    features = gbm_features(final)
    X_train = prepare_lgb_frame(train, features)
    y_train = train["default_flag"].to_pandas()
    X_valid = prepare_lgb_frame(valid, features)
    y_valid = valid["default_flag"].to_pandas()

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    with start_run("gbm_tuning", {"n_trials": args.n_trials}):
        study.optimize(
            lambda t: _objective(t, X_train, y_train, X_valid, y_valid),
            n_trials=args.n_trials,
        )
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_validation_auc", study.best_value)

        print(f"\nbest validation AUC: {study.best_value:.4f}")
        print("best params:", study.best_params)

        # Retrain with best params and report on OOT test - never used during tuning.
        best_params = {"objective": "binary", "metric": "auc", "verbosity": -1, "seed": 42, **study.best_params}
        final_model, _ = train_gbm(train, valid, params=best_params)
        print("\n=== Tuned GBM, evaluated on every split ===")
        for name, split_df in [("train", train), ("validation", valid), ("oot_test", oot)]:
            pred = predict_gbm(final_model, features, split_df)
            metrics = discrimination_report(split_df["default_flag"].to_pandas(), pred)
            print(name, {k: round(v, 4) for k, v in metrics.items()})
            for k, v in metrics.items():
                mlflow.log_metric(f"tuned_{name}_{k}", v)

    with open(_OUTPUT_PATH, "w") as f:
        yaml.safe_dump({**study.best_params, "seed": 42}, f)
    print(f"\nbest params written to {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()