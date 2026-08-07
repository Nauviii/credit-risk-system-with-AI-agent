# Credit Risk PD System — Project Handoff

**Repo:** https://github.com/Nauviii/credit-risk-system-with-AI-agent.git
**Environment:** Windows, PowerShell, `uv`-managed venv, Python 3.12, local dev at `D:\Project_DS\credit-risk-system`
**Status:** Phases 0–9 complete. 127 tests passing, CI green, service containerised.

> **This document was rewritten from scratch.** An earlier version described a different
> target definition, train window, feature set and results table, all of which were found
> to be wrong and have been corrected. Nothing in the previous handoff should be trusted.
> Section 9 lists what changed and why, so the earlier conclusions are not re-derived.

---

## 1. Objective and framing

End-to-end Probability of Default system on real loan data, built to replicate industry
practice rather than a Kaggle-style leaderboard chase. Core principle: every technical
decision must be traceable from business logic through code, with reproducible evidence —
not asserted and moved past.

**Perspective: originator.** The system models a lender scoring its own applicants. This
is a decision, not a detail, and it determines the champion model. LendingClub's `grade`,
`sub_grade` and `int_rate` are that platform's own fitted risk output; an originator has
no counterpart to them at decision time, so they are excluded from the champion. The
alternative framing (an investor selecting from listed loans, where `sub_grade` is
legitimately available) was considered and rejected: it also requires a return model —
interest income, recoveries, prepayment — and PD alone cannot support any decision under it.

Planned layers, in order:
1. Business framing → EDA → preprocessing → modeling → evaluation → decision layer (**complete**)
2. Explainability (Phase 6), MLOps (7), serving (8), monitoring (9) — **complete**
3. AI agent orchestration + LLMOps (Phase 10) — the only phase not started. It must be a
   separate service calling the PD serving API, never embedded in `credit_risk`, to preserve
   the core service's latency guarantees (measured p95: 27.5 ms against a 200 ms budget).

Standing constraints:
- Sample-data results are hypotheses; every material finding is re-confirmed on full data
  (2.26M rows) before being treated as final.
- Every deliverable handed over as an individual file with an exact target path.
- Analysis that decides something lives in a notebook; production code consumes the
  decision and never re-derives it at runtime.

---

## 2. Dataset

**Lending Club Loan Data**, accepted + rejected, 2007–2018Q4, from Kaggle
`wordsforthewise/lending-club`. Chosen over Home Credit and Amex because it carries real
timestamps (enabling genuine out-of-time validation) and un-anonymised features (enabling
the interpretability goal).

- `accepted_2007_to_2018Q4.csv`: 2,260,668 rows × 151 columns — the modeling dataset.
- `rejected_*.csv`: retained for a possible reject-inference exercise. Not started. See
  Section 8 — this is now a stated limitation of the decision layer, not just a nice-to-have.
- A 299,970-row stratified sample (`scripts/sample_for_exploration.py`) exists for quick
  iteration. **Do not use it for anything involving base rates or vintage composition** —
  it is stratified by `(loan_status, issue_year)`, so between-year proportions are not real.

### Data-quality findings

All encoded as named constants in `src/credit_risk/data/schema.py`, evidence in `docs/eda_findings.md`.

- Raw file contains occasional non-data summary rows from concatenated quarterly exports;
  handled in `ingestion.py` by reading `id` as string and filtering to numeric-only values.
- `member_id`: 100% missing, dropped.
- `hardship_*`, `settlement_*`, and 30 further post-origination columns: populated only
  after a borrower is already in distress. Enumerated in `LEAKAGE_COLUMNS` and guarded by
  a regression test — an incomplete list silently inflates in-sample performance.
- `mths_since_last_delinq` and 5 related fields: missing means "never happened", not
  unknown. Handled with `has_history` flags, not mean-imputation.
- `dti` has a sentinel value of 999 (vs a realistic 0–40 range) — nulled before binning.
- `annual_inc` has an erroneous outlier (max 61,000,000 vs p99.9 of 600,000) — winsorized
  at p99.5 (350,000).
- Redundant/constant columns dropped: `funded_amnt`/`funded_amnt_inv`, `fico_range_high`,
  `policy_code`, `pymnt_plan`.
- `initial_list_status` and `disbursement_method` (`PLATFORM_COLUMNS`): platform funding
  mechanics, not applicant attributes, and effectively constant inside the train window.
- `addr_state` and `zip_code` (`FAIR_LENDING_EXCLUDED_COLUMNS`): excluded on fair-lending
  grounds, not statistical ones. Location is a well-established proxy for protected
  characteristics. SHAP put `addr_state` ninth by attribution (3.4%) on the champion even
  though its IV of 0.0135 had kept it out of the scorecard — the GBM was using it materially
  while the univariate screen called it uninformative. The reason is recorded next to the
  constant so the column is not reinstated the next time someone finds signal in it. This
  does not make the model neutral; explicit disparate-impact testing is still required
  before deployment.
- `mob_event` and `mob_observable` (`TARGET_TIMING_COLUMNS`): derived by `build_target`,
  they encode *when* the label event happened. Excluded from every feature list and
  guarded by a test — without it, `gbm_features()` would have picked them up and leaked
  the target outright.

### Vintage-dependent feature availability — why train starts in 2013

Measured in `notebooks/vintage_diagnostics.ipynb`, output in `docs/vintage_feature_availability.csv`.

LendingClub expanded bureau collection in stages. Seven of the candidate features
(`acc_open_past_24mths`, `mort_acc`, `mths_since_recent_bc`, `avg_cur_bal`,
`bc_open_to_buy`, `num_rev_tl_bal_gt_0`, `mo_sin_rcnt_tl`) are **100% null for 2007–2011**,
14–52% null in 2012, and drop below 5% only from **2013**.

Keeping those vintages puts 96,502 rows (21% of the old train set) into a WOE `missing`
bin that encodes calendar time rather than credit risk, and that bin never fires at
validation or serving time. Hence `train_start: 2013-01`. A second group (`il_util`,
`open_acc_6m`, etc.) is 95–100% missing across the whole train window and stays excluded
entirely (`EXCLUDED_VINTAGE_COLUMNS`).

---

## 3. Target definition — fixed 24-month horizon

`data/target.py::build_target`. **This replaced a maturity-filter definition that was
producing a biased default rate.** The old version kept only loans whose outcome had
resolved by the data cutoff, which conditions on a post-origination event and makes the
measured default rate a function of vintage age.

Measured bias, from `docs/vintage_censoring.csv` (full data), between defaults/matured and
defaults/issued: 2013 +0.0pp, 2014 +1.0pp, 2015 +2.2pp, **2016 +7.6pp**.

Under a fixed horizon every loan old enough to be observed for H months gets a label:
no loan is dropped for being healthy, the definition means the same thing in every vintage,
and default rates become directly comparable across splits.

```yaml
target:
  definition: fixed_horizon
  data_cutoff: "2018-12"
  horizon_months: 24
  charge_off_lag_months: 5
  bad_statuses: ["Charged Off", "Default"]
```

**Why H = 24.** It is the largest horizon under which the 2016 vintage is still fully
observable (2016-12 + 24m = 2018-12). The cost, measured on the fully matured 2013 vintage:
H=24 captures **60.1%** of eventual 36-month defaults and **43.7%** of 60-month ones. The
under-count is identical across every split, so it shifts the level but not the comparison.
Consequences carried downstream: this is a 24-month PD, not lifetime, and the term
asymmetry must be corrected before any expected-loss calculation (Section 7).

**Default timing proxy.** Charge-off month is `last_pymnt_d + 5` (LendingClub charges off
at ~121 days delinquent). Validated on 2012–2013 in `docs/charge_off_proxy_check.csv`: the
median gap between months-to-last-payment and payments-actually-made is 0.000 for loans
without recoveries and 0.012 for loans with them. `last_pymnt_d` marks the true final
scheduled payment and is not moved by post-charge-off collections. A loan with a bad status
and a null `last_pymnt_d` never paid at all and is labelled a default at month `co_lag`.

### Splits

| Split | Window | n | 24-month default rate |
|---|---|---|---|
| train | 2013-01 – 2014-12 | 370,443 | 0.0892 |
| validation | 2015 | 421,095 | 0.1070 |
| oot_test | 2016 | 434,407 | 0.1136 |

2017-01 onward cannot reach 24 months of observation before the cutoff. Reserved for
feature-drift simulation (Phase 9), never for labels.

Drift is real but smaller than previously reported: the OOT/train ratio is **1.274**,
against 1.368 under the old biased definition.

---

## 4. Results

All figures on the 2016 out-of-time test set. Discrimination is settled.

### Reference baselines — no model fitted

| Feature used directly as a score | train | validation | oot_test |
|---|---|---|---|
| `sub_grade` | 0.6690 | 0.6938 | **0.6889** |
| `int_rate` | 0.6656 | 0.6936 | 0.6878 |
| `grade` | 0.6618 | 0.6851 | 0.6795 |
| `fico_range_low` | 0.5844 | 0.5893 | 0.5886 |

These are printed on every run of `train_baseline.py`. A model AUC read without them is
uninterpretable: `sub_grade` alone reaches Gini 0.3778 on OOT.

### Models

| | Full pool (73 features) | Application-only (69 features) |
|---|---|---|
| Linear (WOE + LR) | 0.7015 / Gini 0.4031 | 0.6665 / Gini 0.3331 |
| GBM (tuned) | 0.7135 / Gini 0.4270 | **0.6964 / Gini 0.3929** ← champion |

- **Champion: GBM application-only.** Beats `sub_grade` used alone by +0.0076 AUC (95% CI
  [+0.0054, +0.0097], DeLong p = 5.3e-12), without seeing it. Note this is a framing
  decision, not a performance claim: Scorecard (full) at [0.6992, 0.7039] outperforms it
  and does not overlap. The champion is the best model that does not consume LendingClub's
  own risk output. Full pool is reported as a benchmark, not taken forward.
- LendingClub's grade/rate contribute **+0.0171 AUC** (95% CI [+0.0158, +0.0183]) measured
  within the same model class
  (GBM full minus GBM application-only). The linear comparison (+0.0350) overstates it.
- The GBM's advantage over the scorecard is +0.0122 with `sub_grade` present but +0.0294
  without it. Most of what the GBM adds is work `sub_grade` was already doing — it is
  itself a fitted model output carrying non-linearities and interactions.
- The linear scorecard **cannot** match `sub_grade` on application data alone (−0.0223).
  A GBM can. What was missing was model capacity, not signal.

### Hyperparameters are at their ceiling — do not tune further

Two very different parameter sets give the same OOT result:

| | `min_child_samples` | learning rate | iterations | OOT AUC |
|---|---|---|---|---|
| Old (tuned on the biased target) | 155 | 0.0097 | 1911 | 0.7137 |
| New (retuned, protocol fixed) | 948 | 0.0373 | 448 | 0.7145 |

Regularization was tightened massively and generalisation moved by 0.0008. Train AUC rose
(0.7730 → 0.7823), so the train–OOT gap *widened*, but that gap is a boosting fit artefact,
not a defect: what is diagnostic is whether OOT improves under regularization. It does not.
**Do not treat the train–OOT gap as an overfitting alarm on this project.** A lever not yet
tried, if completeness is wanted: LightGBM's `cat_smooth`, `cat_l2`, `max_cat_threshold`
for high-cardinality categoricals. Expectations should be low.

### Per-vintage and per-term

Per-vintage AUC on train (2013 vs 2014) sits either side of the pooled figure, so there is
no aggregation artefact. The rise in AUC across vintages is specific to `sub_grade` (0.669 →
0.694): **LendingClub's grading model improved over time**, and any model containing it
inherits that rise. The application-only model is flat to slightly declining across vintages.

**Discrimination decays inside the OOT year.** Champion Gini by 2016 issue quarter:
0.4175, 0.3935, 0.3785, 0.3713 — monotone, with non-overlapping Q1/Q4 confidence intervals.
The quarter-to-quarter swing of 0.0462 Gini is three times the champion's entire edge over
`sub_grade` (0.0152 Gini). Two consequences: the pooled OOT figure averages over a declining
trend and overstates the model's end-of-period state, and the Phase 9 concept drift eroded
ranking power, not only level. Per-cohort Gini belongs in the monitoring set alongside PSI
and expected-versus-actual.

Per-term OOT AUC, GBM application-only: term 36 = 0.6934, term 60 = 0.6924, pooled = 0.6972.
Within-term discrimination is identical, and pooled sits above both because term itself
carries signal. **Segmented models are not warranted.** The term asymmetry is a calibration
and expected-loss problem, not a ranking problem.

### Feature sets

Derived reproducibly in `notebooks/feature_selection_scorecard.ipynb`
(IV ≥ 0.02 → drop single-bin features → pairwise correlation prune at 0.6 → iterative
sign-based refinement). Rerun it if the target, train window or binning rules change.

`SCORECARD_FEATURES` (14): `sub_grade`, `fico_range_low`, `acc_open_past_24mths`,
`annual_inc`, `dti`, `tot_hi_cred_lim`, `mo_sin_rcnt_tl`, `mths_since_recent_inq`,
`mo_sin_old_rev_tl_op`, `mths_since_recent_bc`, `home_ownership`, `purpose`,
`percent_bc_gt_75`, `verification_status`.

`APPLICATION_FEATURES` (15): the same minus `sub_grade`, plus `bc_open_to_buy` and
`term_months`.

**`term_months` is absent from the full list, and that absence is the most informative
result in this section.** It is dropped at coefficient +0.0115 because `sub_grade` already
prices term — LendingClub charges more for 60-month loans — so term carries no independent
signal once `sub_grade` is present. In the application-only list it is the **strongest
feature at −1.05**. The gap between the two lists is a direct measurement of how much of
the full scorecard is really LendingClub's scorecard. It was deliberately not forced back
in: adding a positive-coefficient feature would break the sign convention the whole
diagnostic machinery rests on.

---

## 5. Calibration

`notebooks/evaluation_phase5.ipynb`, module `evaluation/calibration.py`.

Uncalibrated, the champion is **systematically optimistic**: mean predicted PD on OOT is
0.0892 (exactly train's base rate) against an actual 0.1136. All ten reliability bins run
negative, from −0.004 to −0.043. ECE 0.0244.

**Brier decomposition isolates the problem cleanly.** Resolution is 0.00505 on validation
and 0.00506 on OOT — identical, so discrimination transfers perfectly. Only reliability
worsens, 0.00023 → 0.00074. The model's OOT problem is purely level, not ranking, which is
exactly what calibration fixes. Note also that uncertainty (0.1007) dominates the total
Brier of 0.0961: a raw Brier score on this data is ~95% base-rate variance and is useless
for comparing models across populations.

**Platt, not isotonic**, fitted on validation:

| | ECE | AUC | min PD | distinct values |
|---|---|---|---|---|
| Platt | 0.0118 | 0.6972 | 3.6e-03 | 434,407 |
| Isotonic | 0.0117 | 0.6970 | 1.0e-04 (floor) | **212** |
| Uncalibrated | 0.0244 | 0.6972 | — | — |

Isotonic's 0.0001 ECE advantage is noise. It collapsed 434,407 predictions onto 212 levels
and assigned PD exactly 0 to a block of loans, six of which defaulted — a PD of zero implies
infinite odds, zeroes expected loss and sends the point score to its clipping bound.
`Calibrator` now defaults to Platt and applies a `floor`/`cap` of 1e-4 regardless of method.

Residual: after calibration mean PD is 0.1017 against an actual 0.1136. This is correct
behaviour, not a failure — the calibrator was fitted on 2015 and cannot anticipate the
2015→2016 drift. It is the argument for scheduled recalibration.

**Per-term calibration is not needed.** Gaps under a single calibrator: term 36 −0.0112,
term 60 −0.0136. Splitting the calibrator changes nothing (−0.0116 / −0.0129). Both terms
are correctly calibrated to their own 24-month rate; the term asymmetry bites only at the
lifetime conversion (Section 7).

**Score scale.** `pd_to_score`, PDO 20, base 600 points at 50:1 odds. Range 473–649,
median 555. Verified: bad rate by band runs 0.0324 → 0.0640 → 0.1163 → 0.1927 → 0.2999,
giving odds ratios of 2.04, 1.92, 1.81, 1.79 — the doubling per 20 points the scale promises.

**Central tendency.** Long-run rate 0.1033 (mean of three observed vintages), log-odds shift
+0.0176, AUC unchanged. Three vintages is not a cycle; this anchor is a placeholder.

---

## 6. Stability — and why PSI would have missed the problem

> Phase 9 confirmed this on real vintages. Score PSI on the 2016 out-of-time set is
> **0.0007** while realised defaults ran 10% above prediction. Full results in
> `docs/monitoring_findings.md`.

`evaluation/stability.py`. Same fit-on-reference discipline as `WOEEncoder`: bin edges come
from train and are frozen.

- Score PSI, train → OOT: **0.0004**. Train → validation: 0.0061.
- Highest feature PSI across all 12 numeric champion features: **0.081** (`percent_bc_gt_75`).
  Every feature lands in the "stable" band.

The population did not move. Combined with resolution being identical between validation
and OOT, and AUC holding, the entire OOT degradation is a shift in the outcome relationship
at a fixed population — **concept drift, not covariate shift**. The same borrower profile
defaulted more often in 2016.

**This has a direct consequence for Phase 9.** A monitoring system watching feature and
score PSI would have shown all green while the realised default rate rose 27% relative.
Monitoring must track actual-versus-expected default rate by vintage, not distribution
stability alone. (Worth verifying independently: LendingClub's 2015–2016 vintages are
widely reported to have underperformed, and the platform tightened credit in 2016.)

---

## 7. Decision layer

`evaluation/business.py`, `notebooks/decision_layer.ipynb`. This is what turns a PD into a
decision, and it is where most of the project's business content now lives.

**LGD, estimated not assumed.** From 221,290 charged-off loans: exposure-weighted 0.882,
median 0.896. Collections return roughly 12% of outstanding principal. Uses
post-origination columns to estimate a portfolio parameter from history — legitimate here,
still banned as model input.

**Horizon coverage, measured on the fully matured 2013 vintage.**

| Term | Defaults within 24m | Eventual defaults | Coverage |
|---|---|---|---|
| 36 | 7,440 | 12,378 | 0.6011 |
| 60 | 3,780 | 8,646 | 0.4372 |

Lifetime PD after correction: term 36 rises 0.0941 → 0.1566; term 60 rises 0.1299 → 0.2968.
The term risk ratio is 1.380 on a 24-month basis but **1.895 on a lifetime basis** — a single
blended factor would understate 60-month exposure by a factor of 1.37.

Consistency check: expected loss rate 0.1807 against realised 0.1056 at full approval,
ratio 1.71, which sits inside the 1.66–2.29 range implied by the two coverage factors.

**Approval curve and cutoff economics** (OOT, `docs/cutoff_table.csv`):

| Approval | Bad rate | Expected loss rate | Gross yield | Net margin rate | Net margin | Marginal margin |
|---|---|---|---|---|---|---|
| 25% | 3.67% | 5.29% | 17.54% | 11.06% | $176M | +11.82% |
| 35% | 4.53% | 6.47% | 19.24% | **11.16%** | $248M | +11.27% |
| 50% | 5.75% | 8.21% | 21.38% | 10.87% | $343M | +9.43% |
| 80% | 8.47% | 12.41% | 25.41% | 8.71% | **$439M** | +0.57% |
| 85% | 9.03% | 13.34% | 26.15% | 8.02% | $430M | −2.75% |
| 100% | 11.36% | 18.07% | 29.42% | 3.40% | $218M | −36.89% |

Three framings, and picking the wrong column picks the wrong cutoff:

- **Risk appetite** (bad rate ≤ 6%): approve 50%, bad rate 5.75%, $343M. In practice this
  binds well before profit maximisation, and the ceiling is a policy input the data cannot supply.
- **Marginal zero**: approve 80%. This is the profit-maximising answer — the added tranche
  still earns +0.57% and turns negative immediately after.
- **Max margin rate**: approve 35%. Shown for contrast, **not** to be followed: it optimises
  margin per unit of exposure while ignoring volume, and leaves $191M on the table.

Note the average-versus-marginal distinction. At 80% approval the average margin is still a
healthy 8.71% while each additional approval has stopped adding value. This is the usual way
cutoff decisions go wrong.

**Risk-based pricing does most of the work.** Gross yield rises monotonically as the cutoff
loosens (12.1% → 29.4%), because LendingClub charges more to riskier borrowers. Margin stays
positive even at 100% approval. So the bad-rate curve overstates the model's value: bad rate
halves between 100% and 50% approval, but total margin *falls* from its peak. The model's
real contribution is finding mispricing relative to grade, not screening out bad loans —
which is consistent with the champion beating `sub_grade` by only +0.0084 AUC.

---

## 8. Known limitations — state these, do not hide them

1. **No reject inference.** Every figure is measured on LendingClub's *accepted* population.
   "Approve 80%" means 80% of applicants the platform already accepted. Bad rates are valid;
   approval rates are not readable as real-world acceptance policy. `rejected_*.csv` is on
   disk if this is ever taken up.
2. **Prepayment is not modelled.** `net_margin_rate` assumes scheduled interest with only a
   `(1 − PD)` survival haircut. Roughly half of LendingClub borrowers prepay, which truncates
   interest on the *good* loans. The true optimum is tighter than 80%. `last_pymnt_d` has
   already been validated as reliable, so this is tractable follow-up work.
3. **24-month PD, not lifetime.** `to_lifetime_pd` applies one factor per term, assuming
   every borrower inside a term defaults on the same schedule. High-risk borrowers default
   earlier, so their lifetime PD is over-scaled. A discrete-time hazard model is the proper fix.
4. **LGD is a single portfolio constant** where it genuinely varies by term and grade.
5. **Long-run central tendency uses three vintages**, which is not a credit cycle.
6. **Tuning still touches 2015 twice.** Early stopping now runs on an inner temporal slice of
   train, but 2015 is the Optuna objective across 50 trials *and* the early-stopping set for
   the final model. OOT 2016 is untouched, which is what protects the headline number.
7. **The whole evaluation rests on one OOT vintage**, and 2016 ran materially worse than
   train — and degraded monotonically within it (Section 4).
8. `notebooks/eda.ipynb` still calls `build_target(df)` with one argument and will fail. Low
   priority but not forgotten.
11. **Monitoring has no alerting infrastructure.** Phase 9 delivered tested functions and a
    simulation; scheduling, per-environment thresholds, notification routing and an alert
    audit trail are not built.
12. **Serving carries no authentication, rate limiting or request logging.** The service is
    correct and fast, not hardened.
9. `README.md` is intentionally not up to date; scheduled for the end of the project.
10. Every commit message so far is `"updated"`. Flagged once, deprioritised by choice.

---

## 9. What changed from the previous handoff

Read this before reusing anything from an older document or notebook output.

| Area | Was | Now |
|---|---|---|
| Target | Matured loans only, "ever default" | Fixed 24-month horizon, all observable loans |
| Train window | ≤ 2014 (all vintages) | 2013-01 – 2014-12 |
| Default rates | 17.0 / 20.2 / 23.3% | 8.9 / 10.7 / 11.4% |
| OOT drift claim | "+4.8pp, real drift" | Ratio 1.274 vs 1.368; overstated, but real |
| Feature count | 16 scorecard / 76 GBM | 14 + 15 / 74 + 70 |
| Champion | GBM, full pool | GBM, application-only |
| OOT AUC | 0.723 | 0.6972 (champion), 0.7145 (full benchmark) |

Bugs found and fixed along the way, each with a regression test:
- WOE smoothing was applied to proportions with a 1e-6 epsilon, producing WOE up to +13 for
  a zero-bad bin. Now Haldane–Anscombe at count level.
- WOE categorical pooling used a 5% population floor, which collapsed all 35 `sub_grade`
  levels into one `rare` bin and drove its IV *below* its own coarser parent `grade` —
  mathematically impossible, and the signature of the bug. Now keyed on bad/good counts.
- `np.searchsorted(side="right")` did not match Polars `cut`'s `(lower, edge]` intervals,
  emptying the first bin on discrete features.
- `binning_report` sorted bins as strings, so `"(12.3,"` ordered before `"(5.2,"` and the
  monotonicity flag was meaningless.
- Missing/rare bins had no minimum size guard; a 12-row bin was carrying |WOE| 1.69.
- `score_only_auc` did not orient features, reporting `fico_range_low` at 0.41 instead of 0.59.
- `horizon_coverage` was called on the pre-cleaning frame, where `term_months` does not exist.

---

## 10. Codebase

```
src/credit_risk/
  data/
    schema.py         # ALL exclusion/cleaning decisions as named constants
    ingestion.py      # Polars raw CSV loader, junk-row fix
    target.py         # build_target() - fixed-horizon label, observability filter
    diagnostics.py    # vintage censoring, feature availability, hazard, proxy validation
  features/
    cleaning.py       # sentinels, winsorize, term/credit-history parsing, has_history flags
    woe.py            # WOEEncoder (monotonic binning, count smoothing, min bin size,
                      # binning_report), rank_features_by_iv, prune_correlated_features
    build_dataset.py  # assemble_feature_matrix, tag_split, gbm_features,
                      # application_features, SCORECARD_FEATURES, APPLICATION_FEATURES
  models/
    scorecard.py      # train_scorecard/predict_scorecard - accepts any feature list
    gbm.py            # train_gbm/predict_gbm/prepare_lgb_frame - accepts any feature list
  evaluation/
    metrics.py        # auc, gini, ks_statistic, discrimination_report
    baselines.py      # score_only_auc, reference_baseline_table, auc_by_segment/vintage
    diagnostics.py    # coefficient_sign_report, multicollinearity, drop_until_signs_are_clean
    stability.py      # PSI: frozen reference bins, psi_table, psi_report
    calibration.py    # reliability, ECE, Brier decomposition, Calibrator, central tendency, PDO
    business.py       # approval_curve, horizon_coverage, to_lifetime_pd, empirical_lgd,
                      # scheduled_gross_yield, cutoff_table
  explainability/, serving/, monitoring/   # empty, Phases 6/8/9
  tracking.py         # MLflow wrapper - local SQLite by default

scripts/
  train_baseline.py   # trains all four models, prints baselines, per-vintage, per-term, lift
  tune_gbm.py         # Optuna, --pool full|application, writes separate params files
  sample_for_exploration.py

notebooks/
  vintage_diagnostics.ipynb          # decided H, train_start, and whether OOT is trustworthy
  feature_selection_scorecard.ipynb  # derives both feature lists
  evaluation_phase5.ipynb            # calibration + PSI
  decision_layer.ipynb               # LGD, lifetime PD, approval curve, cutoff economics
  eda.ipynb                          # Phase 3 (currently broken, see Section 8)

configs/
  base.yaml                          # target + split, each value documented with its evidence
  gbm_best_params.yaml               # full pool
  gbm_best_params_application.yaml   # application-only pool

tests/   # 75 tests, all passing
docs/    # eda_findings.md, modeling_findings.md, plus generated CSVs
```

### Engineering decisions worth knowing before touching this code

- **Polars**, not pandas, for all loading/transforms at this scale; convert to pandas only at
  the sklearn/LightGBM/SHAP boundary.
- **`uv`** with a PEP 621 `[project]` `pyproject.toml`. Do not reintroduce `[tool.poetry]` —
  this repo was migrated off it mid-project after real `uv sync` failures.
- `pytest` finds the `src`-layout package via `[tool.pytest.ini_options] pythonpath = ["src"]`.
- MLflow default tracking URI is `sqlite:///mlruns.db`; the `file://` backend is deprecated.
  `Settings` uses `extra: "ignore"` so `MLFLOW_TRACKING_*` env vars do not fail validation.
- `numba>=0.60` pinned explicitly; without it `uv` can resolve an ancient `numba` via `shap`.
- **The WOE sign convention is not a bug.** `WOE = ln(dist_good/dist_bad)`, so higher means
  safer, and a model predicting P(default=1) must show *negative* coefficients. A positive
  coefficient is the anomaly. Documented in the `WOEEncoder` docstring so it is not
  re-litigated — an earlier session flagged the negatives as a defect.

---

## 11. How to resume

```powershell
cd D:\Project_DS\credit-risk-system
uv sync
uv run pytest tests/ -v          # 75 passed
uv run python scripts/train_baseline.py --input data/raw/accepted_2007_to_2018Q4.csv
```

```powershell
# Build a servable bundle (writes artifacts/champion, ~10 s of that is hashing the input)
uv run python scripts/build_artifacts.py --input data/raw/accepted_2007_to_2018Q4.csv

# Serve it
uv run uvicorn credit_risk.serving.app:app --reload    # http://localhost:8000/docs
docker run -p 8000:8000 -v ./artifacts:/app/artifacts:ro credit-risk-system:local
```

Immediate next task: **Phase 10, the agent layer.** Everything it needs is in place — a
versioned artefact, a scoring API with provenance on `/model`, a decision layer that turns a
PD into a cutoff, and monitoring that can tell it whether the model is still trustworthy.

Build it as a separate service. The PD path is measured at p95 27.5 ms and that budget cannot
absorb an LLM call. The agent consumes `/score`, `/model` and the monitoring outputs over
HTTP; nothing about it belongs inside `credit_risk`.

Documentation is current as of this handoff: `eda_findings.md`, `modeling_findings.md`,
`evaluation_findings.md`, `explainability_findings.md`, `monitoring_findings.md`. `README.md`
is the one remaining gap and was deliberately left for last.