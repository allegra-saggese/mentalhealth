# Original header of `script4a-design-estimates.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4a-design-estimates.py  --  PART A: the design-based estimates.

REBUILT 2026-09-28 on the corrected panel. The previous file
(z-archive/script4a-twfe-eventstudy_PRE-REBUILD_2026-09-18.py) ran on the
pre-CHR-fix panel and used the old A-numbering. Two things changed:

  1. PANEL. Ten control variables were re-sourced to their correct data years
     (SAIPE / BLS LAUS / SAHIE / Census PEP). See 2026-09-21_CHR-YEAR-MISALIGNMENT.md.
  2. NAMING. A = built and runnable, B = specified only. See 2026-09-27_MODEL-NAMING.md.

MODELS IN THIS FILE
-------------------
    A1   Pooled OLS                 state + year FE          benchmark
    A2   TWFE within-county         county + year FE         HEADLINE
    A4   Dairy x FSIS interaction   2017-2023, thin sample   exploratory
    A5   Event study                leads/lags around entry
    A6   Callaway-Sant'Anna         staggered DiD, ATT(g,t)

    A3 (other-animal horse race) was DROPPED 2026-10-12 (Sept 30 feedback).
    script4l-a3-horserace.py is retired and no longer part of the pipeline.

RUN 2 ADDITIONS (2026-10-12)
----------------------------
* Controls: step-8 set revised in script4_treatment (no %_female; combined
  Asian+AIAN+NHPI share; %_black).
* A4 saves SEs for EVERY term (dairy, FSIS, interaction), not just the interaction.
* A5 also run (a) with baseline (2000) log population x year, and (b) with log
  population as the OUTCOME -- the depopulation placebo.
* A6 event-time ATTs now carry bootstrap SEs (aggregated inside each draw).
* A7 LP-DiD (estimators.lpdid) on the same E2 cohorts as A5/A6.
* Conley spatial SEs on the A2 grid when county centroids are available
  (Data/raw/gazetteer/; see estimators.load_centroids).

WHAT CHANGED SUBSTANTIVELY FROM THE 09-18 VERSION
--------------------------------------------------
* TREATMENTS. A1/A2 now run over CORE_TREATMENTS (T1-T4) -- the decided
  treatment set -- not the legacy 12-definition TREATMENTS dict. T5 was dropped
  and T6 retired (D-17), so T1-T4 is the complete set with no unrun members.
  The legacy grid is still available via --legacy-grid for continuity checks.
* OUTCOMES. 7, not 6 (the four crime outcomes were settled 2026-09-21, D-11).
* CONTROLS. ONE control set: the D-16 PRE-COMMITTED STEP-8 SET
  (COVARIATE_ORDER[:8]). All eight are annually measured at 99.7-99.9% coverage,
  the estimating sample stays intact (99.7% of outcome rows), and every one can
  be tested for pre-trends.
  CONTROL_PRETREAT (18) was REMOVED as a robustness arm on 2026-09-29 (D-29).
  Its ten extra members are each excluded for a stated substantive reason --
  two colliders (adult_smoking, teen_births), a spliced measure break
  (access_to_healthy_foods), an unusable series (driving_alone_to_work), a
  variable with no usable variation in non-metro counties
  (%_native_hawaiian/other_pacific_islander), and five that cannot be tested for
  pre-trends because they are rolling or model-smoothed. Running that set as a
  "robustness check" would imply the exclusions were arbitrary. They are not.
  THE STEP-8 SET IS PRE-COMMITTED. It was fixed in D-16 before the stability
  table in script4c was read. Choosing it afterwards would have been
  post-selection inference and would invalidate every p-value below.

STANDARD ERRORS
---------------
Three variance estimators side by side on every estimate:
    se_state   CRV1 on state_fips   <- HEADLINE. Treatment assignment is
                                       spatially correlated within state (dairy
                                       regions), and it is the most conservative
                                       of the three here.
    se_county  CRV1 on fips
    se_hetero  heteroskedasticity-robust
Reporting all three is the answer to the SE question raised in review; the
headline choice is not an understatement of uncertainty.

ESTIMATION uses pyfixest.feols, never the legacy within_transform() helper --
that applied a SINGLE demeaning pass, exact only for balanced panels. This panel
is unbalanced (77.3% have the full span) and one-shot demeaning gave
beta = +0.08466 where correct alternating projections give +0.07970: a 6.2%
error in the POINT ESTIMATE, not just the SE.

IDENTIFYING VARIATION is reported inline (n_switchers, n_transitions). Under
county FE only counties whose treatment CHANGES contribute to beta, so a row
with 30 switchers is not the same evidence as one with 300 even at equal N.

Outputs -> Data/output/tables/script4a/  and  figs/script4a/

````
