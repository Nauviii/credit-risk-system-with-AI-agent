"""LightGBM champion - handles nulls/categoricals natively, so no WOE encoding needed here."""

import lightgbm as lgb
import pandas as pd
import polars as pl

from credit_risk.features.build_dataset import gbm_features


def prepare_lgb_frame(df: pl.DataFrame, features: list[str]) -> pd.DataFrame:
    """Convert to pandas with string columns as category dtype for LightGBM's native handling."""
    pdf = df.select(features).to_pandas()
    for col in pdf.select_dtypes(include="object").columns:
        pdf[col] = pdf[col].astype("category")
    return pdf


def train_gbm(
    train_df: pl.DataFrame, valid_df: pl.DataFrame, target: str = "default_flag", params: dict | None = None
) -> tuple[lgb.Booster, list[str]]:
    """Train LightGBM with early stopping against a time-based validation set (not random)."""
    features = gbm_features(train_df)
    X_train = prepare_lgb_frame(train_df, features)
    X_valid = prepare_lgb_frame(valid_df, features)

    default_params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05,
        "num_leaves": 31, "verbosity": -1, "seed": 42,
    }
    train_set = lgb.Dataset(X_train, label=train_df[target].to_pandas())
    valid_set = lgb.Dataset(X_valid, label=valid_df[target].to_pandas(), reference=train_set)

    model = lgb.train(
        {**default_params, **(params or {})}, train_set, num_boost_round=2000,
        valid_sets=[valid_set], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    return model, features


def predict_gbm(model: lgb.Booster, features: list[str], df: pl.DataFrame):
    """Score a dataframe with a fitted GBM - returns predicted P(default)."""
    X = prepare_lgb_frame(df, features)
    return model.predict(X, num_iteration=model.best_iteration)