# Modeling Findings - Phase 4

Basis: **full dataset** (2,260,668 rows), reproduced via `scripts/train_baseline.py`.
Train 453,809 / validation 375,546 / OOT test 293,105 (matured loans only, after
the 3-way time-based split in `configs/base.yaml`). An earlier checkpoint on the
299,970-row sample is kept in git history for reference but is superseded below -
full data is the more reliable evidence, not the sample.

## Validation split
Carved out of the former train window rather than randomly, applying the same
OOT logic one level down: train ≤2014, validation 2015 (used only for GBM early
stopping), OOT test 2016 unchanged. Default rate rises across all three splits
(17.0% -> 20.2% -> 23.3%), consistent with the drift documented in
`eda_findings.md` section 11.

## Results (full dataset)

| Split | Scorecard AUC | Scorecard Gini | Scorecard KS | GBM AUC | GBM Gini | GBM KS |
|---|---|---|---|---|---|---|
| train | 0.704 | 0.408 | 0.297 | 0.756 | 0.513 | 0.377 |
| validation | 0.733 | 0.466 | 0.340 | 0.743 | 0.485 | 0.353 |
| oot_test | 0.708 | 0.417 | 0.301 | 0.721 | 0.442 | 0.318 |

## Finding: GBM earns its complexity on full data (revised from the sample checkpoint)
On the 300k sample, GBM only tied the scorecard on OOT test (+0.001 AUC) while
overfitting badly (train-OOT gap 0.086). On the full 2.26M-row dataset, that
gap shrinks to 0.035, and GBM genuinely beats the scorecard on OOT test: +0.013
AUC, +0.025 Gini, +0.017 KS - no longer a statistical tie. This matches the
expected pattern: a higher-capacity model like GBM needs enough data to learn
real signal rather than train-set noise, and the earlier near-tie was a data
volume limitation, not evidence the extra complexity was unjustified. Still,
this uses **untuned default hyperparameters** for GBM - the comparison is not
final until Optuna tuning is done.

## What's not yet done (before picking a champion)
- Hyperparameter tuning (Optuna) for GBM.
- Full Phase 5 evaluation: PSI (train vs OOT population stability), calibration
  (Brier/ECE - discrimination held up on OOT despite the default-rate shift, but
  calibration has not been checked and likely needs recalibration given that shift).
- SHAP explainability (Phase 6) for the GBM, and coefficient/WOE inspection for
  the scorecard, to confirm both models rely on sensible drivers (grade/int_rate
  as expected) rather than something spurious.

## Infrastructure
Both runs logged to MLflow (`tracking.start_run`), local SQLite backend
(`mlruns.db`) by default - view with `mlflow ui --backend-store-uri sqlite:///mlruns.db`.
Switches to a remote DagsHub-hosted MLflow server by setting
`CREDIT_RISK_MLFLOW_TRACKING_URI` in `.env`, no code change needed. Fixed two
real environment issues along the way: `mlflow`'s `file://` backend is now
deprecated (switched default to `sqlite`), and `pydantic-settings` was rejecting
`MLFLOW_TRACKING_USERNAME`/`PASSWORD` in `.env` as unrecognized fields (added
`extra: "ignore"` to `Settings.model_config`, since those two are read directly
by `mlflow`, not by our settings class).