"""Tests for champion persistence: a loaded bundle must score identically to the fitted one."""

import json

import numpy as np
import polars as pl
import pytest

from credit_risk.evaluation.calibration import Calibrator
from credit_risk.serving.artifacts import (
    ChampionBundle,
    categorical_levels,
    file_sha256,
    save_bundle,
)


@pytest.fixture(scope="module")
def trained():
    """A small GBM with one categorical feature, plus a Platt calibrator fitted on it."""
    import lightgbm as lgb

    from credit_risk.models.gbm import prepare_lgb_frame

    rng = np.random.default_rng(0)
    n = 3_000
    df = pl.DataFrame(
        {
            "annual_inc": rng.normal(size=n),
            "dti": rng.normal(size=n),
            "purpose": rng.choice(["car", "house", "medical"], n),
        }
    )
    logit = -2.0 + 1.4 * df["annual_inc"].to_numpy()
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = df.with_columns(pl.Series("default_flag", y))

    features = ["annual_inc", "dti", "purpose"]
    frame = prepare_lgb_frame(df, features)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "seed": 0, "num_leaves": 8},
        lgb.Dataset(frame, label=y),
        num_boost_round=40,
    )
    calibrator = Calibrator("platt").fit(y, booster.predict(frame))
    return booster, features, frame, calibrator, df


def _save(tmp_path, trained, **kwargs):
    booster, features, frame, calibrator, _ = trained
    return save_bundle(
        tmp_path / "champion", booster, features, categorical_levels(frame), calibrator, **kwargs
    )


def test_round_trip_reproduces_raw_predictions_exactly(tmp_path, trained):
    _, _, _, _, df = trained
    bundle = ChampionBundle.load(_save(tmp_path, trained))
    booster, features, frame, _, _ = trained
    np.testing.assert_allclose(
        bundle.predict_raw(df), booster.predict(frame, num_iteration=booster.best_iteration)
    )


def test_calibrated_output_matches_the_fitted_calibrator(tmp_path, trained):
    booster, features, frame, calibrator, df = trained
    bundle = ChampionBundle.load(_save(tmp_path, trained))
    np.testing.assert_allclose(
        bundle.predict_pd(df), calibrator.transform(booster.predict(frame)), rtol=1e-9
    )


def test_central_tendency_shift_is_applied_on_load(tmp_path, trained):
    _, _, _, _, df = trained
    plain = ChampionBundle.load(_save(tmp_path, trained)).predict_pd(df)
    shifted = ChampionBundle.load(
        _save(tmp_path / "b", trained, central_tendency_shift=0.5)
    ).predict_pd(df)
    assert (shifted > plain).all()
    assert (np.argsort(shifted) == np.argsort(plain)).all()  # ranking preserved


def test_feature_order_is_restored_not_inferred(tmp_path, trained):
    """Scoring a frame whose columns are in a different order must not change the answer."""
    _, _, _, _, df = trained
    bundle = ChampionBundle.load(_save(tmp_path, trained))
    shuffled = df.select(["purpose", "dti", "annual_inc", "default_flag"])
    np.testing.assert_allclose(bundle.predict_pd(shuffled), bundle.predict_pd(df))


def test_categorical_levels_survive_a_single_row_request(tmp_path, trained):
    """One row cannot carry every category; the stored levels must supply them."""
    _, _, _, _, df = trained
    bundle = ChampionBundle.load(_save(tmp_path, trained))
    full = bundle.predict_pd(df)
    single = bundle.predict_pd(df.head(1))
    np.testing.assert_allclose(single, full[:1])


def test_missing_feature_raises_instead_of_scoring_on_nulls(tmp_path, trained):
    _, _, _, _, df = trained
    bundle = ChampionBundle.load(_save(tmp_path, trained))
    with pytest.raises(KeyError, match="dti"):
        bundle.predict_pd(df.drop("dti"))


def test_isotonic_calibrator_is_refused_with_a_clear_reason(tmp_path, trained):
    booster, features, frame, _, _ = trained
    isotonic = Calibrator("isotonic").fit(
        np.random.default_rng(0).integers(0, 2, len(frame)), booster.predict(frame)
    )
    with pytest.raises(ValueError, match="platt"):
        save_bundle(tmp_path / "iso", booster, features, categorical_levels(frame), isotonic)


def test_metadata_and_schema_version_are_persisted(tmp_path, trained):
    path = _save(tmp_path, trained, metadata={"oot_auc": 0.6964, "train_rows": 370443})
    written = json.loads((path / "bundle.json").read_text())
    assert written["schema_version"] == 2
    assert written["metadata"]["oot_auc"] == 0.6964
    assert ChampionBundle.load(path).metadata["train_rows"] == 370443


def test_score_is_monotone_decreasing_in_pd(tmp_path, trained):
    _, _, _, _, df = trained
    bundle = ChampionBundle.load(_save(tmp_path, trained))
    order_pd = np.argsort(bundle.predict_pd(df))
    order_score = np.argsort(-bundle.predict_score(df))
    assert (order_pd == order_score).all()


def test_file_hash_is_stable_and_content_sensitive(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("id,x\n1,2\n")
    b.write_text("id,x\n1,3\n")
    assert file_sha256(a) == file_sha256(a)
    assert file_sha256(a) != file_sha256(b)