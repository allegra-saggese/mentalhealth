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
