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

## Results (FINAL - full dataset, confirmed feature sets)

Scorecard: 16 features (`SCORECARD_FEATURES`, clean - zero multicollinearity above
0.6, all coefficients negative as expected). GBM: 76 features (`gbm_features()`).

| Split | Scorecard AUC | Scorecard Gini | Scorecard KS | GBM AUC | GBM Gini | GBM KS |
|---|---|---|---|---|---|---|
| train | 0.702 | 0.404 | 0.293 | 0.756 | 0.513 | 0.377 |
| validation | 0.732 | 0.463 | 0.336 | 0.743 | 0.485 | 0.353 |
| oot_test | 0.707 | 0.414 | 0.297 | 0.721 | 0.442 | 0.318 |

Reproduced via `scripts/train_baseline.py --input data/raw/accepted_2007_to_2018Q4.csv`.
Cutting the scorecard from 31 to 16 features (removing redundant/collinear ones -
see section below) cost essentially nothing: OOT AUC moved from 0.708 to 0.707,
noise-level. The dropped features were genuinely uninformative, not a tradeoff.

GBM continues to beat the scorecard on OOT test (+0.014 AUC, +0.028 Gini,
+0.021 KS) - consistent with the earlier full-data finding, unaffected by the
scorecard-only feature pruning above.

## Feature selection process (produced the 16 SCORECARD_FEATURES above)
Full process lives in `notebooks/feature_selection_scorecard.py`, run
independently by the user on full data and cross-checked against a sample-data
run - not just asserted. Summary:
1. IV ranking on the full 76-feature candidate pool -> 29 features with IV >= 0.02.
2. `woe.prune_correlated_features()` (pairwise, threshold 0.6) -> 17 features.
   Caught severe pairs like `sub_grade`/`int_rate`/`grade` (corr 0.93-0.97,
   near-duplicate information) and `num_rev_tl_bal_gt_0`/`num_actv_rev_tl` (0.99).
3. `evaluation.diagnostics.drop_until_signs_are_clean()` -> 16 features. Pairwise
   pruning alone left `percent_bc_gt_75` with an unexpected positive coefficient
   (multi-feature collinearity, not caught by pairwise correlation) - this
   iterative step refits and drops until every coefficient sign is correct.
4. Full-data run matched the sample-data run on every feature except one pair of
   near-twins (corr 0.99) where full data's IV tiebreak differed - strong
   evidence the selection is stable, not sample-specific noise.

## Pre-tuning sanity check: scorecard coefficient signs (historical note)
Before the process above existed, a 31-feature scorecard showed 24/31 negative
coefficients, which looked like a bug. Root-caused via a clean single-feature
synthetic test: negative is mathematically correct given our WOE convention
(`WOE = ln(dist_good/dist_bad)`, higher = safer) and target definition
(`default_flag=1` = bad) - documented directly in `WOEEncoder`'s docstring so
it isn't re-litigated. The actual anomaly is a POSITIVE coefficient, which is
exactly what steps 2-3 above are designed to catch and remove.
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