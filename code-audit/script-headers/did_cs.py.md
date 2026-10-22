# Original header of `did_cs.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

did_cs.py  --  Callaway & Sant'Anna (2021) staggered DiD, covariate-adjusted, shared module.

Added 2026-10-17 for script4u. The two core functions (att_gt_table, base_period_covariates) are
copied from script4a-design-estimates.py without change to the estimator, so that
script4u reproduces script4a's E2 results exactly (it asserts this). script4a still holds its own
copy: merging the two is a planned clean-up, not done yet.

WHAT IS ESTIMATED
-----------------
Cohort g = the first census wave at which a county is treated. For each cohort g and calendar year t
    ATT(g,t) = E[ Y_t - Y_{g-1} | cohort g ]  -  E[ Y_t - Y_{g-1} | comparison counties ]
Comparison counties = never treated, or not yet treated by year max(t, g). Counties already treated
are never comparisons.
Covariate adjustment (outcome regression): fit  dY = X'b  on the comparison counties only, with X
measured in the BASE YEAR g-1 (so it is pre-treatment by construction), then
    ATT_adj(g,t) = mean over cohort-g counties of [ dY_i - X_i'b ].
Optional `outcome_lags`: also put the outcome's own levels in years g-1 ... g-lags into X, so
comparison counties are matched on their pre-treatment outcome path.
Aggregation: event time e = t - g ->  ATT(e) = weighted mean of ATT(g, g+e), weights = number of
treated counties; overall ATT = weighted mean over all post-treatment cells (t >= g).
Inference: nonparametric bootstrap resampling whole STATES with replacement (state clustering).

````
