# Plan — estimator triangulation (pre-specified, nothing run)
**2026-10-22** · status: DRAFT, awaiting the owner's decisions marked **[DECIDE]**. Written before any
result from these estimators is read. If a choice is changed after seeing results, the decision register says so.

## Why
The reported evidence rests on two estimators: TWFE (county + year FE) and Callaway–Sant'Anna (CS). They disagree
for the mental-health outcomes: Poor MH Days E2 0.104 (TWFE) vs 0.037 (CS), see `2026-10-22_ANALYSIS-ASSESSMENT.md`.
Is the CS number a property of the data or of CS's particular choices (cohort weighting, comparison group,
covariate adjustment)? Estimating the same effect with other heterogeneity-robust estimators answers that. LP-DiD is
already computed (`estimators.lpdid`; `script4a` A7, `script4r` R4) but not reported and not reviewed; it is the first thing to read.

## Fixed elements (shared by every estimator below, as in METHODS.md)
Rural county-years 2000–2023 via `script4_treatment.load_panel` (D-34 zero rule); treatments **E2** (first expansion) as the
main design and **E3** (first entry, clean controls) as the second; the pre-committed 8 controls (D-16, D-31); 7 outcomes
(Poor MH Days and Frequent Mental Distress are the mental-health pair; the other five are shown but do not carry the claim,
D-23); standard errors by resampling whole states (as `did_cs.py`, 300 draws, fixed seed) for comparability; event window e ∈ [−8, 8], reference e = −1.
Notation: `i` county, `t` year, `g_i` first treated year (2007, 2012, 2017, 2022), `D_it = 1[t ≥ g_i]`, never-treated `g_i = ∞`.

## Estimators and the package that runs each (versions checked 2026-10-22 against PyPI/CRAN and the package source)

| # | Estimator | What it does | Equation (core) | Comparison group | Package to run it |
|---|---|---|---|---|---|
| 1 | **LP-DiD** (already computed) | Local projection DiD with clean controls | `Y_{i,t+h} − Y_{i,t−1} = β_h·ΔD_it + e_it`, sample restricted to newly treated vs not-yet-/never-treated at each `t` | clean controls only | pyfixest `lpdid` (0.60.0, installed) and the project's `estimators.lpdid`; compare the two before trusting either. Read existing output first |
| 2 | **Sun–Abraham (interaction-weighted)** | Cohort × relative-time interactions, then a weighted average | `Y_it = α_i + λ_t + Σ_g Σ_{e≠−1} δ_{g,e}·1[g_i = g]·1[t − g = e] + ε_it`; `θ_e = Σ_g Pr(g_i = g \| observed at e)·δ̂_{g,e}` | never-treated counties | pyfixest `event_study(estimator="saturated")` / `SaturatedEventStudy`, documented as Sun–Abraham, **marked beta** (installed). Cross-check on a simulated panel before use |
| 3 | **Imputation (Borusyak–Jaravel–Spiess / Gardner)** | Predict the untreated outcome for treated observations, then average the gaps | step 1 fit `Y_it = α_i + λ_t + X_it'γ + ε` on observations with `D_it = 0` only; step 2 `τ̂_it = Y_it − Ŷ_it(0)`; step 3 average `τ̂_it` by horizon or overall | all untreated county-years | pyfixest `did2s` (Gardner two-step; point estimates numerically identical to BJS imputation per Gardner et al. 2024; its standard errors are not BJS's, so use the state bootstrap). BJS efficiency holds under spherical errors (homoskedastic and serially uncorrelated), which panel outcomes rarely satisfy |
| 4 | **de Chaisemartin–D'Haultfoeuille (DID_ℓ)** | At each year compares counties whose treatment switches to counties whose treatment does not | `DID_ℓ = average over switchers of [Y_{i,t+ℓ} − Y_{i,t−1}] − average over stayers of the same change`; stayers have the same period-one treatment as the switcher | counties unchanged between `t−1` and `t+ℓ` | R `DIDmultiplegtDYN` 2.4.0 on CRAN (authors' package; **not installed here**; bridge from Python as `script4w` does for `HonestDiD`). Python port `py-did-multiplegt-dyn` 0.1.9 exists (0.x release) as a cross-check. E2/E3 are binary and absorbing, so the plain DID_ℓ applies; for the continuous T3/T4 the baseline-treatment condition needs the paper's web-appendix extension, which the R package advertises |
| 5 | **Stacked DiD** | One sub-experiment per cohort, stacked | for cohort `c`: treated cohort `c` plus clean controls in window `[−8, +8]`; `Y_ict = α_ic + λ_tc + Σ_{e≠−1} θ_e·1[treated_i]·1[t − c = e] + ε_ict`; stack with corrective weights (Wing–Freedman–Hollingsworth) | clean controls per sub-experiment | No dedicated package was verified: implement from the equation (about 60 lines on top of `pf.feols`) and test it on the simulated panel |

**Independent check of our own Callaway–Sant'Anna code (`did_cs.py`).** Four independent implementations exist: R `did` 2.5.1 (CRAN; not installed here) and, on PyPI, `csdid` 0.4.2, `differences` 0.3.0, `moderndid` 0.3.1, `diff-diff` 3.13.0. Run one on the same panel and compare ATT(g,t) cell by cell (`/differential-audit`); disagreement would matter more than any new estimator. pyfixest 0.60.0 has no Callaway–Sant'Anna estimator.

Existing arms kept for comparison: TWFE, CS covariate-adjusted, CS unconditional, CS with past outcomes (`did_cs.py`, `script4u`).

## What is compared and how agreement is defined **[DECIDE]**
- Primary exhibit: one forest-style figure, rows = estimators, panels = outcomes, E2 and E3 designs, each estimate with its 95% interval, scaled by the outcome's within-county SD (as `script4x`), made with `figstyle.py`.
- Proposed agreement rule, fixed now: an estimator "agrees" with CS if its point estimate has the same sign and lies inside CS's 95% interval; the mental-health result is called **estimator-robust** only if at least four of the five new/extra estimators (LP-DiD, SA, imputation, DID_ℓ, stacked) agree for Poor MH Days. Thresholds are the owner's to change **before** the run.
- Pre-trends: each estimator's own pre-period test is reported next to its estimate (imputation: leads regression on the untreated sample; SA and stacked: pre-period δ; DID_ℓ: placebo horizons).
- Every run is a row in the specification ledger (`/spec-register`), kept or not.

## Implementation (nothing built yet)
- New script `script4z-estimator-triangulation.py` (next free stage letter), importing treatments, controls and the sample from `script4_treatment.py`, existing estimators from `did_tools.py`, `did_cs.py`, `estimators.py`; new estimator functions go in `estimators.py` (shared module change: **[DECIDE]** to approve).
- No new estimator is trusted until it reproduces a known answer: (i) on a simulated panel with a known effect and heterogeneous timing (seeded, built with the `power-analysis` Monte Carlo pattern) it must recover the truth where TWFE does not; (ii) with homogeneous effects it must agree with TWFE; (iii) LP-DiD and CS outputs reproduced from the existing files to 1e-6.
- `PYTHONHASHSEED=0`, `default_rng(seed)` once; outputs dated; `tests/compare_outputs.py` unaffected by additions.

## Verification before anything is cited
Simulation recovery table, self-check lines printing zero difference against `script4a`/`script4u` where overlaps exist, `/figure-check` on the exhibit, `/audit-reproducibility` on any number that enters the paper, and a `/robustness-suite` pass on the one estimate chosen as headline.

## Risks
Few treated cohorts (3; Frequent Mental Distress 2) make every estimator noisy: agreement may be uninformative rather than reassuring. Timing is five-year coarse (D-40). Imputation and DID_ℓ standard errors need care with state clustering; the state resampling bootstrap is the common yardstick.

## Verified references (fresh-context check, 2026-10-22; Crossref, RePEc, NBER, arXiv and the papers' own text)

| Estimator | Reference | Notes |
|---|---|---|
| Sun–Abraham | Sun & Abraham, *Journal of Econometrics* 225(2):175–199, 2021, doi:10.1016/j.jeconom.2020.09.006 | Control cohort is never-treated or last-treated |
| Imputation | Borusyak, Jaravel & Spiess, *Review of Economic Studies* 91(6):3253–3285, 2024, doi:10.1093/restud/rdae007 | Efficiency under spherical errors; pre-trend test is a separate regression on untreated observations |
| Two-stage | Gardner, "Two-stage differences in differences", arXiv:2207.05943 (working paper, version dated April 2021); current version Gardner, Thakral, Tô & Yap (2024), working paper | No journal version found; 2024 version states point estimates identical to BJS |
| DID_M | de Chaisemartin & D'Haultfœuille, *American Economic Review* 110(9):2964–2996, 2020, doi:10.1257/aer.20181169 | Binary treatment, no dynamic effects |
| DID_ℓ | de Chaisemartin & D'Haultfœuille, "Difference-in-Differences Estimators of Intertemporal Treatment Effects", *Review of Economics and Statistics* 108(4):863–880, 2026, doi:10.1162/rest_a_01414 (NBER WP 29873, 2022; arXiv 2007.04267) | Compares with units of the same baseline treatment |
| LP-DiD | Dube, Girardi, Jordà & Taylor, "A Local Projections Approach to Difference-in-Differences", *Journal of Applied Econometrics* 40(7):741–758, 2025, doi:10.1002/jae.70000 (NBER WP 31184) | |
| Stacked DiD | Wing, Freedman & Hollingsworth, "Stacked Difference-in-Differences", NBER WP 32054, 2024 (working paper); stacked designs also in Cengiz, Dube, Lindner & Zipperer, *QJE* 134(3):1405–1454, 2019 | Cengiz et al. is a prominent early application, not the sole origin (Callison & Kaestner 2014 and others precede it) |
| Callaway–Sant'Anna | Callaway & Sant'Anna, *Journal of Econometrics* 225(2):200–230, 2021, doi:10.1016/j.jeconom.2020.12.001 | |
