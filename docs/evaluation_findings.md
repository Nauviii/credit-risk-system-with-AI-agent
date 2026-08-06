# Evaluation Findings — Phase 5

Basis: **full dataset**, champion model (GBM application-only), OOT test = 2016 vintage,
434,407 loans, observed 24-month default rate 0.1136.

Reproduced by `notebooks/evaluation_phase5.ipynb` (calibration, stability) and
`notebooks/decision_layer.ipynb` (LGD, lifetime PD, cutoff economics).

Discrimination is settled in `modeling_findings.md` (OOT AUC 0.6972, Gini 0.3945). This
document covers everything that ranking does not answer: whether the predicted level is
right, whether the population moved, and what the model is worth as a decision rule.

---

## 1. Calibration

### The model is systematically optimistic

Mean predicted PD on OOT is **0.0892** — exactly train's base rate — against an actual
**0.1136**. All ten reliability bins run negative, widening monotonically from −0.004 in the
safest decile to −0.043 in the riskiest. ECE 0.0244 on OOT, 0.0131 on validation.

The model reproduces the level it was trained on and nothing else. This is expected and is
precisely what calibration exists to fix.

### Brier decomposition isolates the problem

`brier = reliability − resolution + uncertainty`

| | validation | oot_test |
|---|---|---|
| brier | 0.09075 | 0.09609 |
| reliability (calibration error, lower better) | 0.00023 | **0.00074** |
| resolution (discrimination, higher better) | 0.00505 | **0.00506** |
| uncertainty (base-rate variance, fixed) | 0.09554 | 0.10066 |

**Resolution is identical between validation and OOT.** Discrimination transfers perfectly.
Only reliability worsens, by roughly 3×. The OOT problem is purely level, not ranking.

Second reading: uncertainty (0.1007) is ~95% of the total Brier score. A raw Brier number on
this data is almost entirely base-rate variance and is useless for comparing models across
populations — always decompose it.

### Platt, not isotonic

Both fitted on validation (2015), applied to OOT (2016). Fitting on train would re-learn the
fit the model already has and report near-perfect calibration that does not exist.

| | ECE | AUC | min PD | distinct values |
|---|---|---|---|---|
| Platt | 0.0118 | **0.6972** | 3.6e-03 | 434,407 |
| Isotonic | 0.0117 | 0.6970 | 1.0e-04 (hit floor) | **212** |
| Uncalibrated | 0.0244 | 0.6972 | — | — |

Isotonic's 0.0001 ECE advantage is noise. Its costs are not:

- It collapsed 434,407 predictions onto **212 distinct levels**.
- It assigned **PD exactly 0** to a block of loans, six of which defaulted. A PD of zero
  implies infinite odds, zeroes expected loss, and sends the point score to its clipping
  bound — this produced a spurious 989-loan spike at score 1085 before the floor was added.
- Being a step function it creates ties and shaves AUC (0.6972 → 0.6970).
- It cannot extrapolate beyond its fitted range, which is a serving hazard.

`Calibrator` now defaults to Platt and applies a `floor`/`cap` of 1e-4 regardless of method.
Prefer isotonic only where a reliability plot shows distortion a two-parameter fit
demonstrably cannot follow.

### Residual gap is correct behaviour

After calibration, mean PD is 0.1017 against an actual 0.1136 — still 1.2pp short. The
calibrator was fitted on 2015 and cannot anticipate the 2015→2016 drift. This is the
argument for scheduled recalibration, not evidence of a broken calibrator.

### Per-term calibration is not needed

| | single calibrator | per-term calibrator |
|---|---|---|
| term 36 gap | −0.0112 | −0.0116 |
| term 60 gap | −0.0136 | −0.0129 |

The difference between terms is 0.0024, and splitting the calibrator does not reduce it.
Both terms are correctly calibrated against their own **24-month** rate.

This does not retire the horizon-asymmetry issue — it relocates it. The asymmetry bites at
the lifetime conversion (section 3), not at 24-month calibration.

### Score scale

`pd_to_score`, PDO 20, base score 600 at 50:1 odds. Range 473–649, median 555.

Verified against the data: bad rate by 20-point band runs 0.0324 → 0.0640 → 0.1163 →
0.1927 → 0.2999, giving successive odds ratios of 2.04, 1.92, 1.81, 1.79. The scale delivers
the doubling of good:bad odds every 20 points that it promises, drifting slightly below 2 at
the risky end.

### Central tendency

Long-run rate 0.1033 (mean of the three observed vintages), log-odds shift +0.0176, AUC
unchanged as required. **Three vintages is not a credit cycle** — this anchor is a
placeholder and should be replaced when a longer history is available.

---

## 2. Stability — and why PSI would have missed this entirely

`evaluation/stability.py`. Bin edges taken from train and frozen; recomputing them per
population would compare each distribution against itself.

- Score PSI, train → OOT: **0.0004**
- Score PSI, train → validation: 0.0061
- Highest feature PSI across all 12 numeric champion features: **0.081** (`percent_bc_gt_75`)

Every feature lands in the "stable" band (< 0.10). Full table in
`docs/psi_application_features.csv`.

### The population did not move

Combine three facts: feature distributions unchanged (PSI ≈ 0), score distribution unchanged
(PSI 0.0004), resolution unchanged (0.00505 → 0.00506) — yet the default rate rose from
8.9% to 11.4%.

The same borrower profile defaulted more often in 2016. This is **concept drift, not
covariate shift**.

### Consequence for Phase 9 monitoring

A monitoring system watching feature and score PSI would have shown **all green** while
realised losses rose 27% relative. PSI cannot detect this class of failure by construction.

Monitoring must track **actual-versus-expected default rate by vintage**, with vintage
performance curves, alongside distribution stability. This is a design requirement for
Phase 9, not a nice-to-have.

Worth verifying independently: LendingClub's 2015–2016 vintages are widely reported to have
underperformed and the platform tightened credit in 2016. Treat as a lead, not a fact.

---

## 3. From 24-month PD to lifetime PD

Measured on the fully matured 2013 vintage (a 2013-12 loan on a 60-month term finishes in
2018-12, the data cutoff).

| Term | Defaults within 24m | Eventual defaults | Coverage |
|---|---|---|---|
| 36 | 7,440 | 12,378 | **0.6011** |
| 60 | 3,780 | 8,646 | **0.4372** |

Mean PD after correction:

| Term | PD (24-month) | PD (lifetime) |
|---|---|---|
| 36 | 0.0941 | 0.1566 |
| 60 | 0.1299 | 0.2968 |

The term risk ratio is **1.380 on a 24-month basis but 1.895 on a lifetime basis**. A single
blended coverage factor would understate 60-month exposure by a factor of 1.37.

Consistency check: expected loss rate 0.1807 against realised 0.1056 at full approval, ratio
1.71, which sits inside the 1.66–2.29 range implied by the two coverage factors. The horizon
correction is the right order of magnitude.

**Limitation.** `to_lifetime_pd` applies one factor per term, assuming every borrower within
a term defaults on the same schedule. High-risk borrowers default earlier, so their lifetime
PD is over-scaled and low-risk borrowers' under-scaled. A discrete-time hazard model is the
proper fix; this is good enough to stop expected loss being wrong by a factor, not good
enough to price with.

---

## 4. LGD, estimated not assumed

From 221,290 charged-off loans, LGD = unrecovered share of principal still outstanding:

| | value |
|---|---|
| mean | 0.8827 |
| median | 0.8960 |
| **exposure-weighted (used)** | **0.8820** |

Collections return roughly 12% of outstanding principal — in line with unsecured consumer
lending. Uses post-origination columns to estimate a portfolio parameter from history:
legitimate for this purpose, still banned as model input (`schema.LEAKAGE_COLUMNS`).

Limitation: treated as a single portfolio constant where it genuinely varies by term and grade.

---

## 5. Decision layer — approval, cutoff, and economics

`evaluation/business.py`. Full table in `docs/cutoff_table.csv`, curve in
`docs/approval_curve.csv`.

| Approval | Bad rate | Expected loss rate | Gross yield | Net margin rate | Net margin | Marginal margin |
|---|---|---|---|---|---|---|
| 5% | 1.56% | 2.46% | 12.06% | 9.25% | $29M | — |
| 25% | 3.67% | 5.29% | 17.54% | 11.06% | $176M | +11.82% |
| 35% | 4.53% | 6.47% | 19.24% | **11.16%** | $248M | +11.27% |
| 50% | 5.75% | 8.21% | 21.38% | 10.87% | $343M | +9.43% |
| 65% | 7.00% | 10.13% | 23.37% | 10.09% | $413M | +6.16% |
| 80% | 8.47% | 12.41% | 25.41% | 8.71% | **$439M** | **+0.57%** |
| 85% | 9.03% | 13.34% | 26.15% | 8.02% | $430M | −2.75% |
| 100% | 11.36% | 18.07% | 29.42% | 3.40% | $218M | −36.89% |

`net_margin_rate = gross_yield × (1 − PD_lifetime) − expected_loss_rate`.

### Three framings, and picking the wrong column picks the wrong cutoff

| Framing | Cutoff | Approval | Bad rate | Net margin |
|---|---|---|---|---|
| Risk appetite (bad rate ≤ 6%) | 555 | 50% | 5.75% | $343M |
| **Marginal zero** (profit-maximising) | 536 | **80%** | 8.47% | **$439M** |
| Max margin *rate* | 564 | 35% | 4.53% | $248M |

Maximising `net_margin_rate` optimises margin per unit of exposure while ignoring volume,
and lands at 35% approval — leaving **$191M** on the table. A lender with capital to deploy
maximises currency, not a ratio.

The decisive column is **marginal margin**: what the tranche added by loosening actually
earns. It crosses zero between 80% and 85%. At that point the *average* margin is still a
healthy 8.71% while each additional approval has already stopped adding value. Confusing the
average with the marginal is the standard way cutoff decisions go wrong.

In practice the risk-appetite constraint binds well before profit maximisation, and the 6%
ceiling used above is a policy input the data cannot supply. Stating the target *is* the
decision.

### Risk-based pricing does most of the work

Gross yield rises monotonically as the cutoff loosens, 12.06% → 29.42%, because LendingClub
charges more to riskier borrowers. Margin stays positive even at 100% approval (+3.40%).

This reframes the model's value. Bad rate halves between 100% and 50% approval, which sounds
decisive — but total margin *falls* from its peak over that same range. The model's real
contribution is finding **mispricing relative to grade**, not screening out bad loans. That
is consistent with the champion beating `sub_grade` by only +0.0084 AUC: if the platform's
pricing is already broadly right, the remaining headroom is thin.

---

## 6. Limitations attached to these numbers

1. **No reject inference.** Everything is measured on LendingClub's *accepted* population.
   "Approve 80%" means 80% of applicants the platform already accepted. Bad rates are valid;
   approval rates are not readable as real-world acceptance policy.
2. **Prepayment is not modelled.** `net_margin_rate` assumes scheduled interest with only a
   `(1 − PD)` haircut. Roughly half of LendingClub borrowers prepay, truncating interest on
   the *good* loans — which is where loose cutoffs get their margin. The true optimum is
   tighter than 80%. `last_pymnt_d` is already validated as reliable, so this is tractable.
3. **24-month PD, not lifetime**, corrected by a single factor per term (section 3).
4. **LGD is one portfolio constant** (section 4).
5. **Long-run central tendency uses three vintages**, not a cycle.
6. **One OOT vintage**, and 2016 ran materially worse than train. Per-quarter stability and
   confidence intervals on the AUC differences have not been computed.
