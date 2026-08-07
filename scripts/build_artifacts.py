"""Train the champion and write a servable bundle. The only place artefacts are produced.

Separate from train_baseline.py by design. That script is the evaluation harness: it trains
four models to compare them and keeps nothing. This one trains exactly the champion, applies
the calibration and anchoring decided in Phase 5, and writes something serving can load.

Champion: GBM on application-only features, Platt-calibrated on validation (2015), anchored
to the long-run rate across the three observed vintages. Rationale in PROJECT_HANDOFF.md
section 1 and docs/evaluation_findings.md section 1.
"""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import yaml

from credit_risk.data.ingestion import load_raw_accepted_loans
from credit_risk.data.target import build_target
from credit_risk.evaluation.calibration import (
    Calibrator,
    central_tendency_shift,
    expected_calibration_error,
    pd_to_score,
)
from credit_risk.evaluation.metrics import discrimination_report
from credit_risk.evaluation.stability import reference_profile
from credit_risk.features.build_dataset import application_features, assemble_feature_matrix
from credit_risk.models.gbm import prepare_lgb_frame, train_gbm
from credit_risk.serving.artifacts import categorical_levels, file_sha256, save_bundle
from credit_risk.tracking import start_run

_CONFIG_PATH = Path("configs/base.yaml")
_PARAMS_PATH = Path("configs/gbm_best_params_application.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/champion")
    args = parser.parse_args()

    input_path = Path(args.input)
    final = assemble_feature_matrix(
        build_target(load_raw_accepted_loans(input_path), _CONFIG_PATH), _CONFIG_PATH
    )
    splits = {n: final.filter(pl.col("split") == n) for n in ("train", "validation", "oot_test")}

    features = application_features(final)
    params = yaml.safe_load(_PARAMS_PATH.read_text())
    booster, features = train_gbm(
        splits["train"], splits["validation"], params=params, features=features
    )

    raw = {n: booster.predict(prepare_lgb_frame(df, features)) for n, df in splits.items()}
    y = {n: df["default_flag"].to_numpy() for n, df in splits.items()}

    # Fitted on validation, never on train: on train it would re-learn the fit the model
    # already has and report a calibration quality that does not exist out of sample.
    calibrator = Calibrator("platt").fit(y["validation"], raw["validation"])
    calibrated_oot = calibrator.transform(raw["oot_test"])

    long_run_rate = float(np.mean([y[n].mean() for n in splits]))
    shift = central_tendency_shift(calibrated_oot, long_run_rate)

    # Frozen training distributions, so production drift monitoring has something to
    # compare against once the train set is no longer around.
    # Profiled on POINTS, because points are what DriftMonitor compares. Profiling the
    # calibrated probability instead put the reference on a 0-0.17 range while monitoring
    # passed 473-649, so every live value fell in the final bin and PSI became a constant
    # 12.4339 for every population - large, stable, and completely uninformative.
    train_pd = calibrator.transform(raw["train"])
    train_logits = np.log(train_pd / (1 - train_pd)) + shift
    train_scores = pd_to_score(1 / (1 + np.exp(-train_logits)))
    reference = {
        "score": reference_profile(pl.Series(train_scores)),
        "features": {
            f: reference_profile(splits["train"][f])
            for f in features
            if splits["train"][f].dtype.is_numeric()
        },
    }

    oot = discrimination_report(y["oot_test"], raw["oot_test"])
    metadata = {
        "champion": "gbm_application_only",
        "n_features": len(features),
        "input_file": input_path.name,
        "input_sha256": file_sha256(input_path),
        "config_sha256": file_sha256(_CONFIG_PATH),
        "params_sha256": file_sha256(_PARAMS_PATH),
        "best_iteration": booster.best_iteration,
        "long_run_rate": long_run_rate,
        "split_rows": {n: df.height for n, df in splits.items()},
        "split_default_rate": {n: float(y[n].mean()) for n in splits},
        "oot_auc": oot["auc"],
        "oot_gini": oot["gini"],
        "oot_ece_uncalibrated": expected_calibration_error(y["oot_test"], raw["oot_test"]),
        "oot_ece_calibrated": expected_calibration_error(y["oot_test"], calibrated_oot),
    }

    with start_run("champion_artifact", {"champion": "gbm_application_only"}):
        for key in ("oot_auc", "oot_gini", "oot_ece_uncalibrated", "oot_ece_calibrated"):
            mlflow.log_metric(key, metadata[key])
        path = save_bundle(
            Path(args.output),
            booster,
            features,
            categorical_levels(prepare_lgb_frame(splits["train"], features)),
            calibrator,
            central_tendency_shift=shift,
            metadata=metadata,
            reference=reference,
        )
        mlflow.log_param("artifact_path", str(path))

    print(f"\nbundle written to {path}")
    for key in (
        "n_features",
        "best_iteration",
        "oot_auc",
        "oot_gini",
        "oot_ece_uncalibrated",
        "oot_ece_calibrated",
        "long_run_rate",
    ):
        print(f"  {key}: {metadata[key]}")
    print(f"  reference_profiles: score + {len(reference['features'])} numeric features")
    print(f"  input_sha256: {metadata['input_sha256'][:16]}...")


if __name__ == "__main__":
    main()
