# Credit Risk PD System

End-to-end PD (probability of default) system on Lending Club Loan Data
(accepted + rejected, 2007-2018Q4), built with industry validation practices
rather than pure model-fitting.

## Status: Phase 0-1 (scaffolding)

## Key design decisions
- Target: `default_flag` built only from matured loans (Fully Paid / Charged Off).
  Current/Late/In Grace Period loans are right-censored and excluded — see
  `src/credit_risk/data/target.py`.
- Leakage: columns populated only post-origination (`total_pymnt`, `recoveries`,
  `last_pymnt_amnt`, etc.) are enumerated in `src/credit_risk/data/schema.py`
  and must never enter the feature set.
- Split: out-of-time by `issue_d`, not random K-fold — see `configs/base.yaml`.

## Structure
```
src/credit_risk/
  data/            ingestion, schema contract, target construction
  features/        aggregation, WOE/IV binning (phase 3)
  models/          scorecard baseline + GBM champion (phase 4)
  evaluation/       AUC/Gini/KS/PSI/calibration/cost-based threshold (phase 5)
  explainability/  SHAP (phase 6)
  serving/         FastAPI real-time inference (phase 8)
  monitoring/      drift + performance decay (phase 9)
```

## Setup
```
poetry install
pre-commit install
pytest
```

## Next step
Place `accepted_2007_to_2018Q4.csv.gz` under `data/raw/` (or a stratified sample
for local exploration) to begin Phase 2 (validation) and Phase 3 (EDA).
