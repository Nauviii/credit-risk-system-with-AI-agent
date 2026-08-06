# Modeling Findings — Phase 4

Basis: **full dataset** (2,260,668 rows), reproduced via `scripts/train_baseline.py`.

> **This file was rewritten.** Its previous contents were produced under a maturity-filter
> target definition, a train window starting in 2007, and a WOE implementation with three
> encoding bugs. Every number in that version is void. See `PROJECT_HANDOFF.md` section 9
> for the full before/after and the bug list.

## Population

| Split | Window | n | 24-month default rate |
|---|---|---|---|
| train | 2013-01 – 2014-12 | 370,443 | 0.0892 |
| validation | 2015 | 421,095 | 0.1070 |
| oot_test | 2016 | 434,407 | 0.1136 |

Every loan in the window receives a label — the fixed 24-month horizon means nothing is
dropped for being healthy, so `n` equals loans issued. Train starts in 2013 because seven
candidate bureau features are 100% null before 2012 (`docs/vintage_feature_availability.csv`);
keeping earlier vintages puts 21% of train into a WOE `missing` bin that encodes calendar
time rather than risk.

Drift across splits is real but smaller than previously reported: the OOT/train ratio is
**1.274**, against 1.368 under the old censored definition.

## Reference baselines — read these before any model number

Printed on every run of `train_baseline.py`. A single feature used directly as a score,
no model fitted, orientation corrected so 0.5 is always the no-signal floor.

| Feature | train | validation | oot_test |
|---|---|---|---|
| `sub_grade` | 0.6690 | 0.6938 | **0.6889** |
| `int_rate` | 0.6656 | 0.6936 | 0.6878 |
| `grade` | 0.6618 | 0.6851 | 0.6795 |
| `fico_range_low` | 0.5844 | 0.5893 | 0.5886 |

`sub_grade` alone reaches Gini 0.3778 on OOT. Any model AUC quoted without this floor
alongside it is uninterpretable.

Note the rise in `sub_grade`'s AUC from train to later vintages. This is **not** the
population becoming easier to discriminate — the application-only model is flat across the
same vintages. LendingClub's grading model improved over time, and any model containing
`sub_grade` inherits that improvement.

## Results (OOT test, 2016)

| | Full pool (74 features) | Application-only (70 features) |
|---|---|---|
| Linear (WOE + LR) | 0.7015 / Gini 0.4031 / KS 0.2913 | 0.6665 / Gini 0.3331 / KS 0.2390 |
| GBM (tuned) | 0.7145 / Gini 0.4289 / KS 0.3132 | **0.6972 / Gini 0.3945 / KS 0.2871** |

**Champion: GBM application-only.** The full-pool model is a benchmark, not a candidate —
under the originator framing, `grade`/`sub_grade`/`int_rate`/`installment` are
LendingClub's own risk output and unavailable at decision time.

Four readings from this 2×2:

1. Application data alone **beats** LendingClub's grade (+0.0084 AUC over `sub_grade`),
   but only with a GBM. The linear scorecard cannot (−0.0223). What was missing was model
   capacity, not signal.
2. LendingClub's grade and rate contribute **+0.0173 AUC** measured within the same model
   class. The linear comparison (+0.0350) overstates the dependence by half.
3. The GBM's advantage over the scorecard is +0.0122 with `sub_grade` present, +0.0294
   without it. Most of what the GBM adds is work `sub_grade` was already doing — it is
   itself a fitted model output carrying non-linearities and interactions.
4. Discrimination held on OOT for every model. The 2016 problem is a level shift, not a
   ranking failure — see `evaluation_findings.md`.

## Hyperparameter tuning is at its ceiling — do not continue

Two very different parameter sets produce the same generalisation:

| | `min_child_samples` | learning rate | best iteration | train AUC | OOT AUC |
|---|---|---|---|---|---|
| Tuned on the old biased target | 155 | 0.0097 | 1911 / 2000 | 0.7730 | 0.7137 |
| Retuned, protocol fixed | 948 | 0.0373 | 448 / 4000 | 0.7823 | **0.7145** |

Regularization was tightened by a factor of six and OOT moved by 0.0008. Train AUC rose, so
the train–OOT gap *widened* — but that gap is an artefact of boosting fitting the training
data directly, not a defect. **What is diagnostic is whether OOT improves under
regularization, and it does not.** Do not read the train–OOT gap as an overfitting alarm on
this project; an earlier session did, and 100 Optuna trials bought 0.0008 AUC.

Untried lever, if completeness is wanted: LightGBM's `cat_smooth`, `cat_l2` and
`max_cat_threshold` for high-cardinality categoricals (`addr_state` 50 levels, `sub_grade`
35). Expectations should be low given the above.

### Tuning protocol

`scripts/tune_gbm.py`, 50 trials per pool, separate params files per pool.

Early stopping runs on an inner temporal slice of train (2014-10 onward, 74,144 rows);
2015 is scored once per trial and used for nothing else. The previous version used 2015 for
both, letting every trial stop at the point that flattered its own score. Residual
limitation: after 50 trials the winner's 2015 AUC is optimistically biased. OOT 2016 is
untouched throughout, which is what protects the headline number.

Search bounds for `min_child_samples` are expressed as a share of the **bad count**, not the
row count — that is what changed when the target changed, and it is why the old bounds
bought far less regularization than they appeared to.

## Segmentation and vintage decomposition

Per-vintage AUC on train sits either side of the pooled figure (2013 = 0.7635, 2014 = 0.7739,
pooled = 0.7735 for the champion), so there is no aggregation artefact from mixing base rates.

Per-term OOT AUC for the champion: term 36 = 0.6934, term 60 = 0.6924, pooled = 0.6972.
Within-term discrimination is identical; pooled sits above both because term itself carries
signal (OOT default rate 10.4% vs 14.2%). **Segmented models are not warranted.** The term
asymmetry created by the 24-month horizon is a calibration and expected-loss problem, not a
ranking problem — handled in `evaluation_findings.md`.

## Feature selection

`notebooks/feature_selection_scorecard.ipynb`, run on full data. Rerun it if the target,
train window, or binning rules change — all three changed once already and voided the
previous 16-feature list.

Process: IV ≥ 0.02 → drop single-bin features → pairwise correlation prune at 0.6 →
`drop_until_signs_are_clean` (iterative refit catching multi-feature collinearity that
pairwise pruning misses).

**`SCORECARD_FEATURES` (14):** `sub_grade`, `fico_range_low`, `acc_open_past_24mths`,
`annual_inc`, `dti`, `tot_hi_cred_lim`, `mo_sin_rcnt_tl`, `mths_since_recent_inq`,
`mo_sin_old_rev_tl_op`, `mths_since_recent_bc`, `home_ownership`, `purpose`,
`percent_bc_gt_75`, `verification_status`.

**`APPLICATION_FEATURES` (15):** the same minus `sub_grade`, plus `bc_open_to_buy` and
`term_months`.

### The most informative result here is an absence

`term_months` is dropped from the full list at coefficient +0.0115, and is the **strongest
feature in the application-only list at −1.05**.

Term is one of the most fundamental risk drivers in installment lending. It disappears from
the full model because `sub_grade` already prices it — LendingClub charges more for 60-month
loans — so term carries no independent signal once `sub_grade` is present. Remove
`sub_grade` and term immediately returns to the top.

The gap between the two lists is a direct measurement of how much of the "full" scorecard is
really LendingClub's scorecard. `term_months` was deliberately **not** forced back in:
adding a positive-coefficient feature would break the sign convention the entire diagnostic
machinery rests on. The right answer is the application-only model, not a patched list.

### WOE binning rules

Three rules enforced in `features/woe.py`, all auditable via `WOEEncoder.binning_report()`:
Haldane–Anscombe smoothing at count level, monotonic bad rate across ordered bins (adjacent
violations merged), and minimum bin size by both population share (5%) and absolute counts
(30 bads and 30 goods). Categorical pooling is keyed on class counts, never on population
share. Missing and rare bins too thin to estimate are set to WOE 0 — neutral, not extreme.

Sanity check that must hold after any change: **IV(`sub_grade`) > IV(`grade`)**. A finer
partition cannot carry less information, and the reverse was the signature of a pooling bug
that collapsed all 35 `sub_grade` levels into one bin.

Known trade-off: monotonicity is enforced, which damages genuinely U-shaped features. Any
such feature must be transformed explicitly (e.g. distance from an optimum) rather than
left for the binner to discover.

## Coefficient signs (retained note)

`WOE = ln(dist_good/dist_bad)`, so higher WOE means safer, and a model predicting
P(default=1) must show **negative** coefficients. Negative is correct; **positive is the
anomaly**, and is what the pruning steps exist to catch. Documented in `WOEEncoder`'s
docstring so it is not re-litigated — an earlier session flagged the negatives as a bug.

## Infrastructure

Runs logged to MLflow via `tracking.start_run`, local SQLite backend (`mlruns.db`) by
default — view with `mlflow ui --backend-store-uri sqlite:///mlruns.db`. Switches to a
remote DagsHub-hosted server by setting `CREDIT_RISK_MLFLOW_TRACKING_URI` in `.env`, no code
change. Two environment issues fixed along the way: `mlflow`'s `file://` backend is
deprecated (default switched to `sqlite`), and `pydantic-settings` rejected
`MLFLOW_TRACKING_USERNAME`/`PASSWORD` in `.env` (added `extra: "ignore"`, since those are
read by `mlflow` directly rather than by our `Settings` class).

`lift_over_sub_grade` is logged per split alongside the raw metrics. Use it in preference to
raw AUC when comparing across splits, since later vintages are easier for `sub_grade` too.