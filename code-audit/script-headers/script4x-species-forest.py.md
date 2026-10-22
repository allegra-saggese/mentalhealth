# Original header of `script4x-species-forest.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4x-species-forest.py  --  PER-SPECIES RESULTS AND FOREST PLOTS  (added 2026-10-17; team request 10-15)

For each CAFO species (dairy, beef, hogs, chickens) and each exposure scope
    large       operations in the top size bin only
    all sizes   operations of every size (small + medium + large)
estimate, for each of the 7 outcomes, five estimates:
    T1   presence               1 if the county has >= 1 operation                 TWFE
    T3   per 10,000 residents   count / population x 10,000                        TWFE
    T4   log(1 + count)         (log population also enters)                       TWFE
    E2   first expansion        0 until the first census wave at which the count   TWFE
                                rises, 1 from then on
    CS   Callaway-Sant'Anna     cohorts = the same first-expansion year; covariate-adjusted overall ATT
TWFE: county FE + year FE, the 8 controls, SE clustered by state. CS: did_cs.py, 300 bootstrap draws over states.

COUNTS. Same rule as script4_treatment.load_panel (D-34): from 2002 on a missing count is a zero; 2000-01 are
missing. The rule was verified against NASS for dairy (all waves) and for beef cows / incl. calves (2022);
hogs and chickens rely on the same logic and are NOT yet checked against NASS totals (QA doc).

VALIDATION. For dairy / large the T1, T3, T4, E2 variables must equal script4_treatment's, and the TWFE and CS
results must equal script4a's. The script asserts this (full run only for the CS standard errors).

FIGURES. One forest plot per species, large | all sizes side by side, 7 outcomes x 5 estimates. x-axis = the
coefficient divided by the outcome's within-county SD, so all rows share one scale; the treatment unit differs by
row (see labels): T1, E2 and CS compare treated with not treated, T3 is per 10,000 residents, T4 per unit of
log(1+count). Bars = 95% intervals, state-clustered (CS: bootstrap over states).

OUTPUTS -> Data/output/tables/script4x/<date>_species_results.csv ; figs/script4x/<date>_forest_<species>.png
Usage:  python3 script4x-species-forest.py            full (about 20-30 minutes, runs species x scope in parallel)
        python3 script4x-species-forest.py --plot-only redraw the figures from the latest saved results (no estimation)
        python3 script4x-species-forest.py --quick    dairy only, 2 outcomes, 20 bootstrap draws

````
