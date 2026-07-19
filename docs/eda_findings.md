# EDA Findings - Phase 3 Close-out

Basis: 299,970-row stratified sample (verified representative against the full
2,260,668-row dataset - default rate, grade monotonicity, and leakage checks all
matched within ~0.3pp). All decisions below are enforced in code, not just documented
here - see `data/schema.py` and `features/woe.py`.

## 1. Target & censoring
- Overall: 40.4% of loans are censored (Current/Late/Grace Period) and excluded by `build_target()`.
- Matured population: default rate 19.98% (sample) / 20.0% (full data).

## 2. OOT split (already applied in `configs/base.yaml`)
Censoring rate by issue year: 2007-2013 0% -> 2014 5.3% -> 2015 10.8% -> 2016 32.5%
-> 2017 61.8% -> 2018 88.6%. Recent vintages' "matured" subset is a biased sample
(fast-resolving loans only). **Train: through 2015-12. OOT test: 2016. 2017-2018:
excluded from label-based work, reserved for monitoring-drift simulation (phase 9).**

## 3. Label validity check
Default rate is monotonic across every grade, A (6.0%) through G (49.7%), on both
sample and full data. Grade behaves as a legitimate incumbent risk score.

## 4. Leakage columns (encoded in `LEAKAGE_COLUMNS`)
Original list plus, added after this EDA pass: full `hardship_*` group (99.5% missing
- populated only when a borrower already enrolled in a distress program, a clear
post-origination signal), `settlement_percentage`/`settlement_term`/
`debt_settlement_flag_date`, and `last_fico_range_low/high` (FICO refreshed during
the loan's life, not at origination). Verified concretely: mean `recoveries` is 0 for
every status except Charged Off (~1209) - textbook leakage signature.

## 5. Missing data - categorized, not treated uniformly
- **`ALWAYS_MISSING_COLUMNS`**: `member_id` (100% missing) - drop.
- **`STRUCTURALLY_MISSING_COLUMNS`**: `mths_since_last_delinq` and 5 related fields
  (51-84% missing) - missing means the event never happened, not unknown. Needs a
  `has_history` binary flag + sentinel value in `features/`, never mean/median fill.
- **`JOINT_APPLICATION_COLUMNS`**: `annual_inc_joint`, `sec_app_*`, etc. (~95-99%
  missing) - missing because `application_type != "Joint App"` (~5% of loans), not
  unknown. Deferred to a later model iteration; V1 model uses individual-application
  fields only.
- **Vintage-dependent bureau fields** (`il_util`, `open_acc_6m`, `all_util`, and 10
  related fields, ~38-47% missing): LendingClub started collecting these partway
  through the data window. Not yet decided whether to use them - needs a missing-by-year
  check before Phase 4 to confirm they're populated within our 2012-2015 train window,
  otherwise they're unusable for training even though not technically "leakage."

## 6. High cardinality (`HIGH_CARDINALITY_COLUMNS`)
`emp_title` (102,369 unique, free text) and `desc`/`title`/`url` - drop for V1.
`zip_code` (900 unique, 3-digit) - use `addr_state` (51 unique) instead for geography.
`earliest_cr_line` (686 unique date strings) - convert to a derived numeric feature
(credit history length in months from `issue_d`), never used as a raw category.

## 7. Data quality issues found (must fix before modeling)
- `dti`: max observed = 999.0 against a realistic 0-40 range - placeholder/sentinel,
  not a real value. Encoded in `SENTINEL_VALUES`; must be nulled before binning.
- `annual_inc`: max observed = 61,000,000 against a median of 65,000 - self-reported
  and unverified for many rows; needs capping/winsorization or log transform.
- `revol_util`: max observed = 172% - plausible (utilization can exceed 100% via
  accrued interest/fees), not an error; keep as-is.
- `fico_range_low <= fico_range_high`: 0 violations - consistent, no fix needed.
- `term`: stored as `" 36 months"` / `" 60 months"` (leading space, string) - parse
  to integer months in feature engineering.

## 8. Preliminary IV ranking (`features/woe.py`, on matured/labeled sample)
| Feature | IV | Strength |
|---|---|---|
| sub_grade | 0.492 | strong |
| grade | 0.455 | strong |
| int_rate | 0.446 | strong |
| term | 0.177 | medium |
| fico_range_low | 0.124 | medium |
| dti | 0.076 | weak |
| verification_status | 0.054 | weak |
| loan_amnt | 0.039 | weak |
| home_ownership | 0.030 | weak |
| annual_inc | 0.029 | weak |
| revol_util | 0.028 | weak |
| inq_last_6mths | 0.024 | weak |
| purpose | 0.022 | weak |
| addr_state, pub_rec, open_acc, revol_bal, delinq_2yrs, total_acc | <0.02 | not useful individually |

No feature exceeded the 0.5 "suspicious" threshold - the top 3 are strong because
they represent LendingClub's own pre-existing risk pricing (known at origination,
not leakage), consistent with the grade/int_rate relationship confirmed in point 3.
Note: IV is univariate: "not useful individually" doesn't rule out value via
interactions in the GBM champion model - only rules them out of the WOE scorecard baseline.

## 9. Ready for Phase 4
Open items to resolve before feature engineering is "done": (a) confirm vintage-dependent
bureau fields' coverage within the 2012-2015 train window, (b) decide `annual_inc`
treatment (cap vs. log), (c) build `has_history` flags for structurally-missing fields.
Everything else above is finalized and enforced in code.
