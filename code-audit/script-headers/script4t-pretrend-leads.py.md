# Original header of `script4t-pretrend-leads.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4t-pretrend-leads.py  --  OUTCOME PRE-TRENDS (LEADS) FOR EACH REPORTED TREATMENT (added 2026-10-17)

QUESTION
--------
Before treatment, do outcomes already move differently in counties that go on to be treated?
Answered for every reported treatment definition, next to the covariate pre-trends from
script4j. DESCRIPTIVE TESTS ONLY: no verdict labels are assigned here.

TWO DESIGNS, because the treatments are of two kinds
----------------------------------------------------
Absorbing event treatments (E2 first positive change; E3 first entry, 0 -> 1+ large dairy):
    EVENT STUDY. Y_it = sum_{e != -1} theta_e 1[t - g_i = e] + X_it'gamma + county FE + year FE.
    Event time e = year - cohort year, clipped to [-8, 8]; reference e = -1; never-treated
    counties sit in the reference category. Leads are e = -8 ... -2.

Non-absorbing / continuous treatments (T1 presence, T3 per 10k, T4 log count):
    LEAD TEST. Y_it = b0 D_it + sum_{k=1..4} b_k D_{i,t+k} + [log_pop for T4] + X_it'gamma
    + county FE + year FE. D_{i,t+k} is the county's own treatment k years AHEAD. If treatment
    does not predict the outcome before it changes, b_1 ... b_4 are zero. Because treatment moves
    only at census waves (2007, 2012, 2017, 2022) and is forward-filled, a lead differs from D_it
    only in the k years before a wave. The last four panel years drop out (no lead available).

Both use the pre-committed control set (COVARIATE_ORDER[:8]), state-clustered SEs.
JOINT TEST: Wald chi-square that all leads are zero, with the state-clustered covariance.
With about 49 clusters and several leads, the chi-square approximation is rough: read the
joint p alongside the individual leads.

OUTPUTS -> Data/output/tables/script4t/
    <date>_leads_coefficients.csv    every lead coefficient (treatment, outcome, lead, beta, se, p)
    <date>_leads_joint_tests.csv     one row per treatment x outcome: n leads, Wald chi2, p, N
    (the covariate pre-trends are read from script4j's <date>_control_pretrends.csv and printed alongside)

````
