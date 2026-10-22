# Original header of `script4u-did-by-treatment.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4u-did-by-treatment.py  --  DiD ESTIMATES FOR TWO "FIRST TREATED YEAR" DEFINITIONS (added 2026-10-17)

Runs the same three estimators for two definitions of when a county becomes treated, and adds the
pre-trend adjustments the team asked about. Written as a new script (script4a untouched) with the CS
code in did_cs.py; a built-in check reproduces script4a's results for the first definition.

TREATMENT DEFINITIONS   (g = a county's first treated year; always a census wave: 2007/12/17/22)
    first expansion   tr_e2_add_absorb  g = first wave the county's number of large dairies goes UP.
                      Sample: every rural county (same as script4a).
    first entry       tr_e3_entry_absorb g = first wave the count goes from 0 to 1+.
                      Sample: entry counties + counties that NEVER have a large dairy in any census
                      wave (clean comparison group). Counties that only expand an existing large dairy
                      or have one throughout are LEFT OUT, because they are not valid "untreated"
                      comparisons for an entry design (decision Q-A, 2026-10-17).

ESTIMATORS    (all: 8 controls from COVARIATE_ORDER[:8], county + year fixed effects, SE clustered by state)
    static TWFE      Y_it = b D_it + X_it'c + a_i + d_t + e_it
    event study      Y_it = sum_{e != -1} theta_e 1[t - g_i = e] + X_it'c + a_i + d_t + e_it,
                     e = year - g clipped to [-8, 8], reference e = -1, never-treated in the reference.
                     Leads (e < -1) are the pre-trend test; joint Wald chi-square that all leads are 0.
    Callaway-Sant'Anna, covariate-adjusted (see did_cs.py), bootstrap over states.

PRE-TREND ADJUSTMENTS    (each is a second specification next to the baseline)
    county trends    adds a county-specific linear time trend l_i * t to the static TWFE and the event study.
                     NOT done with pyfixest's  fips[year]  syntax: in pyfixest 0.60 that term is silently
                     ignored (tested 2026-10-17: identical coefficient to the model without it). Instead the
                     outcome and every regressor are residualized on county intercepts + county linear trends +
                     year effects by alternating projections (Frisch-Waugh), then OLS with state-clustered
                     SEs (fe_ols below). fe_ols with no trend is checked to equal pyfixest.
    past outcomes    adds the outcome's own levels in years g-1 and g-2 to the CS covariates, so comparison
                     counties are matched on their pre-treatment outcome path.
    state x year FE  replaces the year effects with state-by-year effects: treated counties are compared only
                     with comparison counties in the SAME STATE and year, which removes regional drift.
                     (checked: every treated county sits in a state that also has comparison counties.)
    baseline x time  adds baseline (2002) county characteristics x (year-2002) and x (year-2002)^2:
                     log population, median household income, % Hispanic, % 65+, unemployment, and the 2002
                     number of dairy operations (all sizes, and large). Lets different kinds of counties drift
                     differently using only pre-treatment information, without a free slope per county.
    (the HonestDiD sensitivity analysis is a separate step, not in this script.)

OUTPUTS -> Data/output/tables/script4u/
    <date>_static_twfe.csv           design, spec, outcome, beta, se, p, N, treated counties
    <date>_event_study.csv           design, spec, outcome, event_time, beta, se, p
    <date>_event_study_leads_joint.csv   joint Wald test of the leads
    <date>_cs_att_gt.csv, <date>_cs_aggregated.csv   design, spec, outcome, ATT(g,t) / event-time / overall
    <date>_validation_vs_script4a.csv   max abs difference from script4a for the first-expansion baseline

Usage:  python3 script4u-did-by-treatment.py            full run (300 bootstrap draws; about 30 minutes)
        python3 script4u-did-by-treatment.py --no-cs    TWFE and event studies only (minutes); the CS files are left as they were
        python3 script4u-did-by-treatment.py --quick    20 draws, two outcomes (for testing)

````
