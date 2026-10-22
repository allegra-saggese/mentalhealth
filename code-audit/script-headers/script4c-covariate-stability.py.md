# Original header of `script4c-covariate-stability.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4c-covariate-stability.py  --  synthesis of script4b. NO new estimation.

PURPOSE
-------
script4b ran 840 regressions: for each treatment x outcome x FE spec it walked
the covariate set in a fixed order, one variable at a time, step 0 -> step 14.
That produced 56 "paths" (4 treatments x 7 outcomes x 2 FE specs).

This script asks one question of each path: DOES THE ANSWER DEPEND ON WHERE YOU
STOP ADDING COVARIATES?

A path is FRAGILE if any of the following holds between step 0 and step 14:
    - the sign of beta flips
    - the 5% significance verdict changes (either direction)
    - |total change in beta| exceeds 25%

WHY THIS MATTERS, AND WHAT IT IS NOT
------------------------------------
This is a measure of researcher degrees of freedom, not a model-selection rule.
If a path flips sign or gains significance only once controls are added, then
"the effect" is a statement about the control set as much as about the treatment.
That must be disclosed, not resolved by picking the step with the nicest result.

Choosing a specification BECAUSE of this table and then reporting conventional
p-values on it is post-selection inference. The reported specification stays
pre-committed (see CONTROL-DECISION-REGISTER.md). This is an appendix exhibit.

STEP 8 vs STEP 14
-----------------
Both are reported. Step 8 is the last step with the sample intact (all eight
full-coverage covariates in, N ~ 65,578). Step 9 brings SAHIE, whose 2008 start
truncates the window, so from there the covariate is confounded with the sample
restriction. A path that is stable to step 8 but moves after it is telling you
about SAMPLE, not about confounding.

Input  <- Data/output/tables/script4b/<date>_covariate_sequence.csv
Output -> Data/output/tables/script4c/
            <date>_path_stability.csv     one row per path (56)
            <date>_stability_summary.csv  aggregate by FE spec / outcome

````
