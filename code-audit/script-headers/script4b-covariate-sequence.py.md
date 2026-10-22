# Original header of `script4b-covariate-sequence.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4b-covariate-sequence.py  --  A1/A2 with covariates added one at a time.

DESIGN
------
For every combination of
    treatment   T1-T4        (CORE_TREATMENTS: presence, count, per capita, log)
    outcome     7            (3 mental health / mortality + 4 crime)
    FE          2            A1 pooled state+year, A2 within county+year
    step        0..14        covariates added ONE AT A TIME in COVARIATE_ORDER
estimate the regression and record what changed.

  4 x 7 x 2 x 15 = 840 regressions.

Step 0 is the treatment with no covariates (plus log_pop for T2/T4, which is part
of the SPECIFICATION, not a covariate -- see SPEC_EXTRA_REGRESSORS).

WHAT IS RECORDED, AND WHY
-------------------------
  beta, se_state/county/hetero   the estimate under three variance estimators
  d_beta_pct                     % change in beta from the PREVIOUS step. This is
                                 the diagnostic for "does this covariate matter" --
                                 a control earns its place by MOVING the treatment
                                 coefficient, which is what confounding control means.
  vif_max, vif_treatment         collinearity. NOTE: VIF answers "is this redundant
                                 given the others", NOT "is this useful". A variable
                                 can have VIF 1.0 and be useless, or VIF 4 and be
                                 essential. Read it alongside d_beta_pct, not instead.
  partial_r2                     share of residual outcome variance the covariate
                                 block explains
  N, n_counties, n_switchers     what the step costs in sample and in identifying
                                 variation. Under county FE only switchers identify
                                 beta, so a step that halves switchers has halved
                                 the evidence even if N looks fine.

ORDER DEPENDENCE -- read this before interpreting the path
-----------------------------------------------------------
Forward stepwise results depend entirely on the order. The order is FIXED in
script4_treatment.COVARIATE_ORDER and grouped so the eight full-coverage
covariates enter first: through step 8 the sample is intact (N ~ 65,578) and a
change in beta is attributable to the covariate alone. From step 9 the sample
falls -- largest at step 9, where SAHIE's 2008 start truncates the window -- so
later steps confound the covariate with the sample restriction.

THIS IS A DIAGNOSTIC, NOT A SELECTION PROCEDURE. Choosing the final specification
because of what this path shows, then reporting conventional standard errors on
it, is post-selection inference and the p-values would be invalid. The reported
specification stays pre-committed; this run is an appendix exhibit.

Output -> Data/output/tables/script4b/

````
