# Analysis Registry — dairy CAFO → mental health / crime outcomes
**Date:** 2026-09-17 | **Status:** FOR MANUAL REVIEW | **Source:** `script4-model-test.py` (1,550 lines, last run 2026-09-10)

Purpose of this document: enumerate EVERY analytical approach explicitly, one per row,
so that (i) nothing is run that isn't on this list, (ii) nothing on this list is run
without a stated identifying assumption and a stated failure mode, and (iii) the
presentation can cite a registry ID rather than "the regression in the script."

**Nothing here is new estimation.** A1–A17 already exist in `script4-model-test.py`.
A18–A22 are candidates, NOT built, listed so the decision to run or skip them is explicit.

---

## Fixed elements, shared by every analysis below

| Element | Value | Set at |
|---|---|---|
| Panel | latest `Data/merged/*_panel.csv` | script4 L77 |
| Sample filter | `rural == 1` (NCHS 3–6, non-large-metro) | script4 L78 |
| Unit | county-year, FIPS 5-digit zero-padded | — |
| Years available | 2000–2023 (CAFO forward-filled from census years 2002/07/12/17/22) | script1b |
| FSIS | merged separately from `*_panel_fsis.csv`, 2017–2023 only, ~46–50% county coverage | script4 L86–99 |
| Default SE | cluster-robust by **state** (`state_fips`), not county | all blocks |
| Default CI | 95% (z = 1.96), EXCEPT A7 event study which uses 90% (z = 1.645) to match the external memo | — |

### Outcome set (6) — `OUTCOMES`, script4 L214
| ID | Label | Column | Known coverage issue |
|---|---|---|---|
| O1 | Poor MH Days | `poor_mental_health_days` | CHR, near-complete |
| O2 | Frequent Mental Distress | `frequent_mental_distress_per100k` | CHR, later years only |
| O3 | Deaths of Despair | `crude_rate_from_census_pop` | **hard 0% coverage after 2020** — truncated at L206; ~27% coverage |
| O4 | Violent Crime (CHR) | `violent_crime` | CHR |
| O5 | Assault (Agg+Simple) | `crime_assault` = aggravated + simple, NaN-propagated from `total_incidents_per100k` | NIBRS coverage |
| O6 | Total Incidents (NIBRS) | `total_incidents_per100k` | NIBRS coverage |

### Treatment definitions (the thing that varies most)
| ID | Variable | Meaning |
|---|---|---|
| T1 | `any_large_dairy` / `dairy_large_present` | binary: ≥1 LARGE dairy CAFO |
| T2 | `any_medlarge_dairy` | binary: ≥1 medium-or-large |
| T3 | `any_medium_dairy` | binary: ≥1 medium |
| T4 | `any_dairy` | binary: ≥1 of any size |
| T5 | `cafo_dairy_large` | raw COUNT, with `log_pop` as separate control |
| T6 | `cafo_dairy_large_log_ct` | log1p(raw count) |
| T7 | `cafo_dairy_large_raw` | count / pop × 10k, unlogged |
| T8 | `cafo_dairy_large_log` | log1p(per-10k rate) |

---

## PART A — Design-based estimates → `script4a-twfe-eventstudy.py`

### A1 — Pooled cross-section OLS
- **Question:** Across counties, are counties WITH a large dairy CAFO different on the outcome?
- **Spec:** `Y_it = β·T1_it + X_it·γ + state_FE + year_FE + ε`, state-clustered
- **Sample:** full panel, all rural county-years | **Controls:** `CONTROL_COLS` (24 vars)
- **Identifying assumption:** selection into CAFO presence is fully captured by X + state + year. **This is not credible** and is only run as the memo's starting point.
- **Failure mode:** any cross-county confounder. Cannot be presented as causal.
- **Built:** script4 L521 | **Output:** `Block1_dairy_levels_vs_within.csv`, `Y1_*.png`
- **2026-09-10 result:** O3 β = −2.07, p < 1e−6 (significant, WRONG SIGN vs hypothesis); all others n.s.

### A2 — TWFE within-county
- **Question:** Within a county over time, does gaining a large dairy CAFO move the outcome?
- **Spec:** same, but `county_FE + year_FE`
- **Identifying assumption:** parallel trends conditional on X; no time-varying county confounder correlated with CAFO entry.
- **Failure mode:** staggered-adoption bias (Goodman-Bacon) — treated-vs-already-treated comparisons contaminate β. **This is the specific reason A18 exists.**
- **Built:** script4 L521 | **Same output as A1**
- **2026-09-10 result:** only O4 significant (β = +22.1, p = 0.022). O3 collapses to −0.64 n.s. — the sign flip vs A1.

### A3 — Horse race (dairy conditional on other animal types)
- **Question:** Is the dairy coefficient actually dairy, or is it "any large CAFO"?
- **Spec:** A2 + `any_large_beef`, `any_large_hogs`, `any_large_chickens` entered jointly. Controls held identical to A2 so the ONLY change is conditioning on other animals.
- **Built:** script4 L584 | **Output:** `Block1b_*.csv`, `Y1b_*.png`

### A4 — Treatment size-threshold sensitivity
- **Question:** Does "large" specifically matter, or would any dairy CAFO do?
- **Spec:** A2 re-run 4× with T1 / T2 / T3 / T4
- **Built:** script4 L658 | **Output:** `Block1c_*.csv`, `Y1c_*.png`

### A5 — Functional-form sensitivity
- **Question:** Does it matter whether exposure is presence / count / per-capita, logged or not?
- **Spec:** A2 re-run 5× with T1 / T5 / T6 / T7 / T8, `log_pop` in X for all five so it's apples-to-apples
- **Built:** script4 L718 | **Output:** `Block1d_*.csv`

### A6 — Dairy × FSIS interaction
- **Question:** Is the dairy association concentrated where a slaughterhouse is also present?
- **Spec:** single regression per outcome with `any_large_dairy`, `any_fsis`, and their interaction — not three separate regressions
- **Sample:** 2017–2023 ONLY, ~46–50% coverage. **Underpowered by construction.**
- **Built:** script4 L767 | **Output:** `Block1e_*.csv`
- **Flag:** currently labeled "exploratory." Decide explicitly whether it appears in the deck.

### A7 — Event study around first large-dairy entry
- **Question:** Does the outcome move at CAFO entry, and were pre-entry trends flat?
- **Spec:** ±10 leads / 9 lags around first entry (cohorts from census years only), state-clustered, **90% CI**, plus joint Wald tests on pre and post blocks using the same cluster-robust covariance
- **Built:** script4 L796, helper L376 | **Output:** `Block2_*.csv`, `Block2b_*joint_tests.csv`, `Y2_*.png`
- **2026-09-10 result:** pre-trend **FAILS** for O3 and O4 — i.e. the two outcomes with the strongest headline numbers are the two whose event studies cannot be read causally.
- **OPEN:** control group is not documented anywhere. Never-treated vs. not-yet-treated is currently implicit. Must be stated before this goes in a deck. (→ A22)

---

## PART B — Regularized / ML estimates → `script4b-ml-selection.py`

### A8 — Ridge, pooled (raw)
- **Question:** Does an atheoretical regularized fit over the full treatment+control pool pick dairy out?
- **Spec:** `RidgeCV`, alphas 1e-3…1e3 (25 grid), standardized, over CAFO log-per-10k pool + demographics
- **Built:** script4 L872/L931 | **Output:** `Block3_*.csv`, `Y3_*.png`
- **Limitation (already noted in code):** produces no valid p-value. Deliberately excluded from A16.

### A9 — Ridge, within-transformed
- **Spec:** same as A8 on county+year demeaned data — the pooled/within contrast is the memo's own point
- **Built:** same block

### A10 — Control specification curve
- **Question:** How much does β move purely as a function of which controls you chose?
- **Spec:** A2 with `log_pop` always in, plus **every subset** of `CORE_9` → 2⁹ = 512 regressions per outcome × 6 outcomes = **3,072 fits**
- **CORE_9 chosen for ≥92% coverage in 2010–2023** so combinations aren't driven by missingness
- **Built:** script4 L1050 | **Output:** `Block4_specification_curve.csv` (741 KB), `Y4_*.png`
- **This is the single most defensible robustness exhibit in the script.**

### A11 — Covariate importance ranking (Ridge + RF), dairy included
- **Question:** How much predictive weight does dairy carry relative to standard controls?
- **Spec:** `WIDE_20` (22 vars incl. `any_large_dairy`) → Ridge coefficients + RF importances
- **Built:** script4 L1134 | **Output:** `Block5_covariate_ranking.csv`, `Y5_*.png`
- **Not causal.** Prediction only.

### A12 — Random Forest variable importance, 2010–2020
- **Spec:** RF over `ML_CONTROLS`, run separately for T7 (per-capita) and T5 (raw count)
- **Built:** script4 L1278 | **Output:** `Block6a_*.csv`

### A13 — Double Lasso / DoubleML (partially linear)
- **Question:** Does a Neyman-orthogonal, cross-fitted estimator agree, without hand-picking X?
- **Spec:** `DoubleMLPLR`, LassoCV nuisance for both E[Y|X] and E[D|X], n_folds = 5
- **Sample:** 2010–2020, min 500 obs
- **Built:** script4 L1315 | **Output:** `Block6b_*.csv`
- **Note:** this is a *selection-on-observables* estimator. It does NOT solve the staggered-DiD problem A2 has; it solves a different problem (control choice). Must not be presented as a causal upgrade over A2.

### A14 — TWFE re-run, 2010–2020 window + interactions
- **Spec:** A2 restricted to 2010–2020 with `ML_CONTROLS`, both T5 and T7, plus `dairy_x_incineq`, `dairy_x_hispanic`, `any_large_dairy_x_fsis` each entered jointly with dairy presence
- **Built:** script4 L1353 | **Output:** `Block6c_*.csv`
- **CRITICAL OPEN ISSUE:** A2 (full panel) is null for 5/6 outcomes; A14 (2010–2020) is significant for 5/6. **There is currently no documented justification for the 2010–2020 window.** This is the first thing a discussant will attack. Must be resolved before the deck. (→ A21)

---

## PART C — Synthesis, no new estimation → `script4c-robustness-summary.py`

### A15 — Causal vs ML comparison table
- Reads A12/A13/A14 only — the one place TWFE, DML and RF ran on the identical sample and controls. Full-panel A2 deliberately NOT mixed in.
- **Built:** script4 L1395 | **Output:** `Block7a_*.csv`

### A16 — Vibration of effects
- Every dairy β in the script on one plot: thresholds, forms, windows, methods. x-axis units deliberately NOT comparable across rows.
- **Built:** script4 L1425 | **Output:** `Block7b_*.csv`, `Y6_*.png`

### A17 — Pass/fail matrix
- One row per outcome × 5 tests. Pre-trend PASS = cannot reject flat (p ≥ 0.05).
- **Built:** script4 L1499 | **Output:** `Block7c_passfail_summary.csv`

---

## CANDIDATES — not built, decide explicitly

### A18 — Callaway–Sant'Anna staggered DiD
- **Why:** A2 and A7 are both vulnerable to negative-weight / forbidden-comparison bias under staggered adoption. CS gives clean ATT(g,t) with a stated control group.
- **Flagged in script4's own docstring as the open follow-on.**
- **Cost:** moderate. Cohorts already exist via `build_entry_cohort()` (L351). Needs `differences` or `csdid` equivalent in Python.
- **Caveat:** only 5 possible entry cohorts (census years), so g is coarse.

### A19 — Heterogeneous treatment, done properly
- A6 and A14's interactions are currently bolt-ons in two different samples with two different control sets. If heterogeneity is a claim in the deck, it needs one consistent spec.

### A20 — Permutation / placebo inference
- Randomize entry year within counties, re-run A7, compare observed joint statistic to the null distribution. Directly addresses "is this just noise given how many specs we ran."

### A21 — Window justification (**blocking for the deck**)
- Not an estimator: a documented, pre-stated reason for full-panel vs 2010–2020, or a decision to report both side by side and stop treating the sub-window as the headline.

### A22 — Control-group documentation for A7 (**blocking for the deck**)
- State and implement never-treated vs not-yet-treated explicitly.

---

## Proposed file split

| New file | Contains | Approx. lines |
|---|---|---|
| `script4_common.py` (importable) | panel load, rural filter, FSIS merge, treatment dictionary T1–T8, `OUTCOMES`, `CONTROL_COLS`, FE helpers | ~250 |
| `script4a-twfe-eventstudy.py` | A1–A7 | ~500 |
| `script4b-ml-selection.py` | A8–A14 | ~500 |
| `script4c-robustness-summary.py` | A15–A17, reads 4a/4b CSVs | ~250 |

Naming note: stage-numbered scripts use dashes and are therefore not importable.
Shared code must live in an underscore module. `functions.py` is the existing precedent —
FE helpers could go there instead of a new `script4_common.py`. **Your call.**

## Review checklist for today
- [ ] Is the outcome set right — all 6, or drop the ones with coverage cliffs (O3 post-2020)?
- [ ] State-clustered SEs: is state the right cluster level, or county?
- [ ] A7 at 90% CI while everything else is 95% — keep the memo-matching asymmetry or standardize?
- [ ] Which of A18–A20 get built this week?
- [ ] A21 and A22 — who decides, and what's the answer?
- [ ] Which registry IDs are presentation exhibits vs. appendix vs. internal only?

---

# TREATMENT VARIABLE FAMILY (added 2026-09-18)

Built once in `script4_treatment.py`, persisted into the panel, crossed with EVERY spec.
One missingness rule throughout: **NaN stays NaN. No `fillna(0)` in treatment construction.**

## Size definition (applies to all)
`cafo_dairy_large` = dairy operations with **500+ milk cows** (USDA NASS top inventory bin,
code 7; see script0b-usda-raw.py:663). Medium = 200-499, small = <200.
NOTE: EPA's regulatory Large CAFO threshold for dairy is 700+ mature cows. USDA has no 700
cutpoint. 500+ is the closest available bin — defensible, but NOT the regulatory definition.
Must be stated in the data section.

## Group E — extensive margin (presence / entry)
| ID | Column | Definition | Events |
|---|---|---|---|
| E1 | `treat_dairy_lg_bin` | >=1 large dairy, NaN-preserving | — |
| E2 | `treat_add_absorb` | absorbing: 1 from first positive-change wave onward | 277 counties |
| E3 | `dairy_entry_year`, `dairy_t_rel` | cohort + relative event time | 4 cohorts |

Exits and contractions never turn treatment back off (team decision, 2026-09-18).
Do NOT drop exit counties outright — that selects on a treatment path correlated with outcome.

## Group I — intensive margin (dose)
| ID | Column | Definition |
|---|---|---|
| I1 | `treat_add_cum` | cumulative net additions of large ops |
| I2 | `treat_add_nevents` | count of positive-change events so far |
| I3 | `cafo_dairy_large` | raw count (with log_pop separate) |
| I4 | `cafo_dairy_large_p10k` | count / pop x 10k |

## Group M — mechanism split (build vs consolidation)
Motivated by: 74.8% of "expansion" events occur while TOTAL dairy ops are FALLING
(median: small -13, total -12). Delta(large)>0 predominantly measures SECTOR CONSOLIDATION,
not new construction. Entry events show the same pattern (65.9% with total falling).

### WHAT "DELTA" MEANS (was never stated — fixed 2026-09-18)
Delta is the change between CONSECUTIVE AG CENSUS WAVES for the same county —
a FIVE-YEAR difference, not year-over-year. Only four wave-pairs exist:
2002->2007, 2007->2012, 2012->2017, 2017->2022.

    d_large = cafo_dairy_large[wave t] - cafo_dairy_large[wave t-1]   (500+ cow ops)
    d_total = cafo_dairy_total[wave t] - cafo_dairy_total[wave t-1]   (ALL dairy ops)

| ID | Column | Definition in words | Events |
|---|---|---|---|
| M1 | `tr_m1_build` | Large ops UP **and** total dairy ops UP — consistent with genuinely new operations | 222 |
| M2 | `tr_m2_consolidate` | Large ops UP **but** total dairy ops DOWN — small farms vanishing, surviving herds crossing 500 cows. CONSOLIDATION, not construction | 676 |
| M3 | `tr_m3_add_any` | Large ops UP regardless of total | 938 |
| — | (neither) | d_large>0 and d_total==0 | 40 |

Once a county has a qualifying event, treatment is ABSORBING: 1 from that wave
onward in every later year, including forward-filled inter-census years.

**Worked example — fips 42071 (Lancaster County, PA), a consolidation county:**

| wave | small | medium | LARGE | TOTAL | d_LARGE | d_TOTAL |
|---|---|---|---|---|---|---|
| 2002 | 1870 | 32 | 9 | 1911 | . | . |
| 2007 | 1889 | 26 | 13 | 1928 | +4 | +17 |
| 2012 | 1841 | 17 | 20 | 1878 | +7 | −50 |
| 2017 | 1572 | 22 | 19 | 1613 | −1 | −265 |
| 2022 | 805 | 42 | 20 | 867 | +1 | −746 |

Large operations roughly double while the county loses over a thousand dairy farms.
A treatment defined only on d_large>0 calls this "a CAFO was added". What actually
happened is that the local dairy sector collapsed and concentrated.

**M3 NEVER RAN.** It is byte-identical to E2 (`tr_e2_add_absorb`), so it was left
out of the TREATMENTS dict. The column exists; no separate estimate does.

**SUPPRESSION CHECK — RESOLVED 2026-09-18. Consolidation is REAL, not an artifact.**
Evidence (`Data/clean/diagnostic/2026-06-09_qa_suppressed_bin_imputation.csv`):
- `gap` = nass_total_ops - (small+medium+large) is **exactly 0 for all 10,110 dairy rows**
  where it is computable (min=0, max=0, no negatives). The observed size bins fully account
  for NASS's own separately-queried total operations. No suppressed/omitted dairy bin rows.
- Zero `clean`-tier rows across ALL 5 animal classes and all years -> the `large_imputed`
  branch in script0b never fires. `large_imputed` == `large` everywhere. Using `large`
  directly (per existing convention) is confirmed harmless.
- Only gap in coverage: `nass_total_ops` is missing (`dark` tier) in **2022 for the three
  cattle classes** (cows milk, cows beef, incl calves). Hogs and layers are fully covered.
  This is a totals-query coverage gap, not suppression.
- Pattern is robust to dropping the unverifiable wave: on VERIFIED waves only (2002-2017),
  70.1% of positive-change events still occur while total dairy ops FALL (vs 72.1% with 2022),
  median d_small = -8, d_total = -7. Essentially unchanged.

Residual caveat: counties absent from the compact entirely cannot be validated this way;
the design zero-fills them (see Zero-fill decision in project memory).

## Group C — composition / consolidation (added at user request, 2026-09-18)
| ID | Column(s) | Definition | Within-variation (CHR sample, CORE_9) |
|---|---|---|---|
| C1 | `dairy_large_share` | large / total, [0,1] | 695 counties (27.0%) |
| C2 | `dairy_large_share_scr` | C1, screened to total>=10 | 487 of 1,301 (37.4%) |
| C3 | `log_lg` + `log_sm` **jointly** | conditional model, contemporaneous | 602 (22.8%) |
| C4 | `log_lg` + `log_sm_baseline` | conditional on PRE-DETERMINED small-farm structure | 602 (22.8%) |
| C5 | `hhi_size` | HHI across the 3 size bins | 1,106 (43.0%) |

### Notes on Group C
- **C1 caveats:** median county has 7 total dairy ops; 40% have <=5, so the ratio is unstable.
  70.2% of share values are exactly 0 (median 0, p75 0) — a linear-in-share model is mostly
  fitting the 0->positive margin, i.e. close to E1 in disguise. C1 also conflates
  "large ops arrive" with "small ops disappear" by construction.
- **C3 is the preferred consolidation spec.** Two coefficients instead of one: large arriving
  (holding small fixed) vs small disappearing (holding large fixed) — the two rival mechanisms
  estimated against each other. C1 is the RESTRICTED version of C3 (imposes beta_large = -beta_total);
  test that restriction with an F-test rather than assuming it.
- **C3 bad-control warning:** contemporaneous `small` is post-treatment if large CAFOs drive
  small farms out. C4 conditions on baseline (pre-determined) small-farm structure instead.
  Build both; the difference is itself informative.
- **C5 caveat:** HHI moves when small farms consolidate among themselves with no large op
  involved. Its extra variation is partly mechanical and not CAFO-specific. Supporting
  exhibit, not a headline.

## Standing finding that constrains all of the above
`CONTROL_COLS` (25 vars) destroys the design via listwise deletion:
no controls / CORE_9 -> 284-602 switching counties; FULL 25 -> 96.
Replace with a coverage-screened, pre-treatment-only control set BEFORE re-estimating anything.

---

# CONTROL-SET DIAGNOSTICS (2026-09-18)

## VIF — collinearity is NOT a problem
Sample: Poor MH Days estimating sample, 10,958 rows / 2,324 counties, FULL 25 + log_pop.

| | VIF>10 | VIF>5 | max |
|---|---|---|---|
| raw levels | 0 | 2 | children_in_poverty 9.10 |
| **within-transformed (county+year demeaned)** | **0** | **0** | **log_pop 3.24** |

Condition number (within): 12.7.
Collinearity among these controls is almost entirely CROSS-SECTIONAL and is removed by
county FE (children_in_poverty: VIF 9.10 raw -> 1.13 within). In the specification actually
estimated, the controls are near-orthogonal.

**Report this as a PASSED diagnostic. Do NOT cite collinearity as a reason to trim controls —
the VIF table contradicts that claim.**

Watch item: `%_hispanic` <-> `log_pop` correlate -0.775 within-transformed (individual VIFs
still fine at 2.86 / 3.24). Relevant because `dairy_x_hispanic` is an interaction term in
Part (f) — that interaction is hard to interpret cleanly if the two are this tangled.

## Do controls absorb the treatment? No.
Within-transformed, treatment = large-dairy presence:
- VIF of treatment against all 25 controls: **1.010**
- R^2 of treatment on controls: **0.0096**
- largest single correlation: `%_hispanic` at **-0.046**; `poor_physical_health_days` at +0.025

## REVISION to the 2026-09-17 post-treatment/bad-control concern
That concern was overstated. Bad-control bias requires controls to be AFFECTED by treatment;
within-county correlations max out at 0.046. Drop the clearest mediators as housekeeping,
but this is second-order, not a fix for something large.
Caveats: tests contemporaneous mediation only; computed on the noisy binary treatment;
imprecise with ~93 switchers.

## Control screen should be built on COVERAGE, ranked by actual cost
1. **Coverage / listwise deletion — DOMINANT.** FULL 25 -> 96 switchers; CORE_9 -> 602.
2. Post-treatment status — second order (see revision above).
3. Collinearity — not a problem at all (see VIF table).
