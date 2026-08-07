# Monitoring Findings — Phase 9

Basis: champion bundle `3eae03c28fd9`, 69 features. Drift measured on the 2017 (443,579) and
2018 (495,242) vintages, which have no labels — they cannot reach 24 months of observation
before the 2018-12 cutoff. Outcome checks backtested on 2013–2016, which do.

Reproduced by `notebooks/monitoring_simulation.ipynb`.

---

## 1. The case for Phase 9, measured rather than argued

| Instrument | 2016 vintage |
|---|---|
| Score PSI against train | **0.0007** |
| Feature PSI, worst of 58 | stable |
| Outcome: expected vs actual defaults | **49,327 vs 44,855, ratio 1.100** |

The score distribution on 2016 did not move at all — 0.0007 is zero for practical purposes —
while 10% more loans defaulted than the model predicted. A monitoring system watching
distributions would have shown green for the entire year.

Distribution stability and outcome stability are different questions. Neither instrument
substitutes for the other, and this table is the evidence.

---

## 2. Score drift: stable everywhere

| Population | Score PSI | Out of range | Band |
|---|---|---|---|
| 2017 | 0.0111 | 0.010 | stable |
| 2018 | 0.0292 | 0.026 | stable |
| validation (2015) | 0.0066 | — | stable |
| oot_test (2016) | 0.0007 | — | stable |

Per quarter, 2017Q1 through 2018Q4: 0.0169, 0.0116, 0.0049, 0.0183, 0.0272, 0.0205, 0.0395,
0.0399. Every quarter stable, but note the direction — the last four quarters run roughly
double the first four. Far below the 0.10 watch band, and worth keeping an eye on because it
moves with the feature drift in section 3.

`out_of_range` is reported alongside every PSI and belongs in any dashboard built from this.
It is the diagnostic that caught a real bug: the reference had been profiled on calibrated
probabilities (0–0.17) while monitoring passed point scores (473–649), and PSI returned a
constant 12.4339 for three unrelated populations. PSI cannot signal that failure — the number
is perfectly well-defined for a distribution sitting entirely outside its reference, and it
looks exactly like severe drift.

---

## 3. Feature drift: the population moved, and the score did not

2018 against the training reference:

| Feature | PSI | Band |
|---|---|---|
| `percent_bc_gt_75` | 0.3359 | material |
| `bc_util` | 0.3084 | material |
| `revol_util` | 0.2960 | material |
| `bc_open_to_buy` | 0.2541 | material |
| `fico_range_low` | 0.1605 | watch |
| `num_bc_tl` | 0.1527 | watch |
| `inq_last_6mths` | 0.1316 | watch |
| `num_rev_accts` | 0.1262 | watch |

In 2017 the same four utilisation features were already flagged, all in the watch band
(0.1171–0.1711). The shift roughly doubled over the following year. LendingClub's borrower
population genuinely moved on revolving-utilisation, and the movement is directional and
sustained, not noise.

**The finding is what did *not* happen.** Four inputs at material shift produced a score PSI
of 0.0292 — completely stable. The model absorbed the movement without its output moving.
That follows from Phase 6: those four features carry roughly 7% of SHAP attribution between
them, and attribution on this model is diffuse (top 5 features = 31.7%), so no single input
can move the score on its own.

The practical consequence sharpens the monitoring design: **score PSI alone would also have
missed this.** Three layers are needed and each catches something the other two cannot —
features catch population movement, score catches movement that reaches the decision,
outcomes catch a changed relationship invisible to both.

---

## 4. Outcome performance, and what it actually measures

Point-in-time calibrated PD (Platt fitted on 2015, no central tendency anchor):

| Vintage | n | Expected | Actual | Ratio | z | Band | Alert |
|---|---|---|---|---|---|---|---|
| 2013 | 134,814 | 13,107 | 11,220 | 0.856 | −17.79 | material | yes |
| 2014 | 235,629 | 24,594 | 21,835 | 0.888 | −19.09 | material | yes |
| 2015 | 421,095 | 45,053 | 45,051 | **1.0000** | −0.01 | stable | no |
| 2016 | 434,407 | 44,215 | 49,327 | **1.116** | 26.35 | material | yes |

2015 lands at a ratio of 0.99997 with z = −0.01. That is not a coincidence and it is the
strongest available check that the calibration works: 2015 is the vintage the calibrator was
fitted on, and it reproduces it essentially exactly.

**A prediction that was wrong, and the correction matters.** The anchored PD showed 2013 and
2014 at ratios of 0.844 and 0.875, and the expectation was that removing the central tendency
anchor would move them close to 1.0. It did not — they moved only to 0.856 and 0.888. The
anchor was not the cause.

The cause is the calibration vintage. A single calibrator fitted on 2015 systematically
mis-states every other cohort: 2013 and 2014 genuinely ran safer than 2015, so the model
over-predicts them by 11–14%; 2016 ran worse, so it under-predicts by 12%.

So expected-vs-actual measures **deviation from the calibration baseline, not from truth**.
That is exactly the right instrument for its job — detecting when a new cohort departs from
the calibrated expectation — provided nobody reads a red row as proof the model is broken.
It also sets a recalibration cadence: one year after fitting, 2016 already deviates 11.6%.

Note also why significance alone is useless here. Every row is significant; 2015 sits at
z = −0.01 only because it is the fitting vintage. On 400,000 loans a 1.4% deviation reaches
z = −3.3. Bands are graded — stable below 5%, watch to 10%, material above — because a single
threshold produced a cliff: the 2016 vintage at MOB 24 sat at a ratio of 1.0997 against a
0.10 line and was classified "not material" by three thousandths.

---

## 5. Early warning: a verdict 19 months early

Hazard coverage from the fully matured 2013 vintage — cumulative share of *in-horizon*
defaults arriving by each month on book (not lifetime, which is the different quantity
`business.py` uses): MOB 6 = 0.023, MOB 12 = 0.260, MOB 18 = 0.640, MOB 24 = 1.000.

2016 judged against a scaled expectation:

| MOB | Coverage | Expected | Actual | Ratio | z | Alert |
|---|---|---|---|---|---|---|
| 6 | 0.023 | 1,035 | 1,171 | 1.131 | 4.2 | yes |
| 12 | 0.260 | 11,674 | 14,265 | 1.222 | 24.5 | yes |
| 18 | 0.640 | 28,716 | 33,051 | 1.151 | 26.9 | yes |
| 24 | 1.000 | 44,855 | 49,327 | 1.100 | 22.9 | yes |

**Earliest alerting month on book: 5** — nineteen months before the outcome window closes.

The ratio holds between 1.10 and 1.22 across every observation point, so this is a stable
deterioration signal rather than an artefact of one measurement. Two bugs had to be fixed
before these numbers meant anything: `mob_event` is populated for every resolved loan, so
counting `mob <= M` alone treated early payers as defaults and turned 1,171 real defaults
into 8,834 at MOB 6; and the expectation must be scaled by hazard coverage, without which a
perfectly healthy cohort registers z = −184 for "far fewer defaults than predicted".

---

## 6. What a production job runs

Two cadences, because the two questions resolve on different timescales.

**Per batch of new applications** — answerable immediately, no outcome needed:
score PSI with `out_of_range`, feature PSI per numeric feature, and any feature whose column
stopped arriving. That last one is the most dangerous case: the model scores a missing column
as null without complaining, so every score shifts and nothing fails.

**Monthly, per cohort** — expected vs actual at whatever month on book the cohort has reached,
scaled by hazard coverage. Alert on significant *and* materially deviating.

`monitoring_cycle` in the notebook is the shape of it. On the 2018 batch plus the matured
2016 cohort it returns: drift stable, 8 features flagged, performance ratio 1.100, band
`watch`, alert raised.

---

## 7. Limitations

1. **Hazard coverage comes from one vintage** (2013) and assumes default timing is stable
   across cohorts. Verified between 2013–2014 and 2016 in Phase 5, but an assumption that
   itself needs monitoring — a vintage that defaults *faster* would trip early warning for
   the wrong reason.
2. **PSI bands are conventions, not tests.** They ignore sample size entirely, and a large PSI
   on a low-attribution feature rarely matters — as section 3 shows directly.
3. **2017–2018 are LendingClub's accepted loans**, so the drift measured is drift in the
   accepted population, not in the applicant population a live model would face.
4. **Outcome checks are anchored to one calibration vintage** (section 4). Deviation is
   relative to that baseline.
5. **No alerting infrastructure.** These are tested functions and a simulation. Scheduling,
   thresholds per environment, notification routing and an alert audit trail are not built.
