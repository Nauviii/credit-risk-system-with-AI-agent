# Explainability Findings — Phase 6

Basis: champion model (GBM, application-only, 70 features, 659 iterations). SHAP computed on
20,000 sampled rows of the 2016 OOT set — what drives the model on data it has never seen.
Scorecard points derived from the same `APPLICATION_FEATURES` list on train.

Reproduced by `notebooks/explainability.ipynb`. Outputs: `docs/shap_importance.csv`,
`docs/shap_direction.csv`, `docs/scorecard_points.csv`.

---

## 1. Every direction check passes

| Feature | Expected to raise risk | Observed | Spearman |
|---|---|---|---|
| `dti` | yes | yes | +0.979 |
| `annual_inc` | no | no | −0.980 |
| `fico_range_low` | no | no | −0.981 |
| `mths_since_recent_inq` | no | no | — |
| `acc_open_past_24mths` | yes | yes | — |
| `percent_bc_gt_75` | yes | yes | — |
| `term_months` | yes | yes | — |
| `tot_hi_cred_lim` | no | no | — |
| `bc_open_to_buy` | no | no | — |

No contradictions among the high-importance features. The correlations on the top three are
near-perfectly monotone (|ρ| > 0.97), so these are not weak tendencies — the model has
learned the relationships a credit officer would state without looking at data.

**One minor inconsistency worth a look.** `mths_since_last_delinq` behaves correctly
(ρ = −0.906: more months since a delinquency is safer), but the analogous
`mths_since_last_record` runs the opposite way (ρ = +0.678: more months since a public record
reads as riskier). Both carry under 1% of attribution so nothing downstream depends on it,
but two features of identical construction pointing opposite ways suggests the public-record
variant is picking up something other than recency — most likely an interaction with the
`has_history` flag, since the correlation is computed only over rows where the value is present.

---

## 2. Attribution is diffuse, not concentrated

| | share of total attribution |
|---|---|
| Top 5 features | **31.7%** |
| Top 15 features | 64.1% |

Top features: `annual_inc` (7.0%), `fico_range_low` (7.0%), `loan_amnt` (6.4%),
`acc_open_past_24mths` (5.7%), `dti` (5.5%), `term_months` (5.0%).

No single driver dominates. The model genuinely uses its 70 features rather than hiding a
three-feature model inside a wide one. That is good for robustness — no one input can break
the score — but it carries a real operational cost, and this is the finding to carry into
Phase 8:

**Serving must compute and validate 70 features per request, each a potential null, schema
change or drift point, to buy +0.0084 AUC over a single column (`sub_grade`).** With
concentrated attribution a trimmed model would be an easy win; with 64% spread across 15
features and the rest across 55, trimming will cost measurable performance. The trade-off
between accuracy and serving fragility should be decided deliberately, not inherited.

---

## 3. Where the GBM beats the linear model — `loan_amnt`

SHAP versus IV rank agreement: **Spearman 0.834** across all 70 features. Most of the
ranking is recoverable from a univariate screen. The interesting part is the exceptions.

Features with high SHAP attribution that the IV screen rated unusable (< 0.02):

| Feature | SHAP share | IV |
|---|---|---|
| `loan_amnt` | **6.4%** | 0.0019 |
| `addr_state` | 3.4% | 0.0135 |
| `total_il_high_credit_limit` | 2.3% | 0.0019 |
| `pct_tl_nvr_dlq` | 2.1% | 0.0017 |
| `emp_length` | 1.9% | 0.0124 |

**`loan_amnt` is the third most important feature to the GBM and has essentially zero
univariate signal.** Its effect is entirely interactive, and the interaction is obvious once
stated: $30,000 is unremarkable against a $200,000 income and dangerous against $40,000.
Pooled across all incomes the relationship is not monotone, so IV measures nothing and the
screen discards it. Its SHAP direction is strongly monotone (ρ = +0.948) *conditional* on the
rest of the model.

This is a concrete, explainable mechanism for a measured result: the GBM beats the linear
scorecard by +0.0122 AUC with `sub_grade` present but **+0.0294 without it**. When
`sub_grade` is available it already prices loan size against capacity; when it is removed,
only a model that can represent interactions recovers that.

**Actionable consequence.** The scorecard has no capacity-to-repay feature. A
`loan_amnt / annual_inc` ratio is pure application data, is exactly the interaction the GBM
is exploiting, and would be readable by a credit committee. Adding it as a candidate and
re-running feature selection is the single most promising route to closing part of that
0.0294 gap. Not done — it reopens feature selection and voids the current lists.

---

## 4. `addr_state` — a compliance question, not a modelling one

`addr_state` carries 3.4% of attribution, the ninth largest, despite an IV of 0.0135 that put
it below the selection threshold.

Geography is a well-established proxy for protected characteristics in US consumer lending.
It is not itself a prohibited basis under ECOA / Regulation B, but models using location have
long been subject to redlining and disparate-impact scrutiny, and most originators either
exclude it or run explicit disparate-impact testing before deploying with it.

SHAP answers what the model does. It cannot answer whether a driver is one the lender may
legitimately use, and this is the case where that distinction bites.

**Decision taken: `addr_state` is excluded.** Recorded in
`schema.FAIR_LENDING_EXCLUDED_COLUMNS` with the reason written next to it, wired into
`_excluded_columns()` so it cannot reach any feature list, and guarded by a test. The cost is
3.4% of attribution — real but affordable, and far cheaper than discovering the problem after
deployment.

`zip_code` was already excluded, but as `HIGH_CARDINALITY_COLUMNS` — accidental compliance:
right outcome, wrong recorded reason. A column dropped for the wrong reason gets added back
the moment someone finds signal in it, so it has been moved to the same constant.

Neither feature list changes: `addr_state` never survived the IV screen, so `SCORECARD_FEATURES`
and `APPLICATION_FEATURES` are untouched and feature selection does not need re-running. Only
the two GBM pools shrink, 74 to 73 and 70 to 69.

### Measured cost: 0.0008 AUC

| Model | OOT AUC with `addr_state` | without | cost |
|---|---|---|---|
| GBM application-only (champion) | 0.6972 | **0.6964** | 0.0008 |
| GBM full (benchmark) | 0.7145 | 0.7135 | 0.0010 |

Both scorecards are unchanged, as expected. Lift over `sub_grade` falls from +0.0084 to
+0.0076 — the champion still beats LendingClub's grade without seeing it.

**Attribution is not the same as necessity.** `addr_state` carried 3.4% of SHAP attribution
yet removing it cost 0.08% of AUC. SHAP measures how much a model *uses* a feature given the
others; with correlated inputs, dropping one lets the rest absorb almost all of its signal.
Read attribution as a description of the fitted model, never as an estimate of what a feature
is worth — for that, refit without it, which is cheap and unambiguous.

Note this cuts the other way too: excluding `addr_state` does not make the model neutral.
Other retained features carry geographic and socioeconomic signal indirectly. Exclusion is a
first step, not a proof.

---

## 5. The scorecard as a points table

Full lookup in `docs/scorecard_points.csv`. PDO 20, base 600 points at 50:1 odds, 15
features. Implied score range ≈ 471–642, consistent with the calibrated GBM's 473–649.

Points by swing — the difference between a feature's worst and best bin, which is what
actually moves an applicant's score:

| Feature | Swing | Bins | Range |
|---|---|---|---|
| `purpose` | **22.1** | 13 | 20.9 – 43.0 |
| `acc_open_past_24mths` | 16.3 | 8 | 29.0 – 45.3 |
| `annual_inc` | 16.0 | 11 | 29.5 – 45.5 |
| `mths_since_recent_inq` | 14.8 | 12 | 30.9 – 45.7 |
| `fico_range_low` | 14.4 | 12 | 32.3 – 46.7 |
| `term_months` | 12.7 | 2 | 28.4 – 41.1 |
| `dti` | 11.8 | 11 | 30.9 – 42.7 |

Reading these: 20 points is one PDO, so a doubling of good:bad odds. `purpose` alone can
swing an applicant by more than a full PDO; moving from the worst FICO band to the best is
worth 14.4 points, less than three quarters of a doubling. `term_months` extracts 12.7 points
from a binary choice, which is why it was the strongest coefficient in the
application-only fit.

**Agreement with the GBM's SHAP ranking: Spearman 0.725** across the 15 shared features. Both
models read the population broadly the same way, which is the reassuring outcome. The main
divergence is `purpose`, which the scorecard leans on far harder (largest swing) than the GBM
does (3.7% attribution, eighth) — consistent with a linear model needing a coarse categorical
to stand in for structure the GBM represents directly.

---

## 6. Limitations

1. **SHAP direction is measured marginally.** A feature whose effect genuinely reverses
   across its range averages toward zero and reads as *no* direction rather than as a
   warning. Anything reported near zero (`revol_bal`, ρ = +0.02) needs a dependence plot
   before being called uninformative.
2. **20,000 sampled OOT rows.** The global ranking is stable at that size; individual tail
   features are not, and the ordering below roughly 1% attribution should not be relied on.
3. **SHAP explains the model, never whether the model is right.** A driver can be perfectly
   sensible and still be a proxy for something the lender may not legally or ethically use —
   see section 4.
4. **Categorical direction is not reported.** LightGBM category codes have no order, so a
   correlation against them would be meaningless rather than merely weak. `purpose`,
   `home_ownership`, `addr_state`, `verification_status` and `emp_length` show null.
5. **Two models, two populations.** SHAP is computed on OOT; scorecard points are fitted on
   train. The rank agreement in section 5 mixes those, so treat 0.725 as indicative.