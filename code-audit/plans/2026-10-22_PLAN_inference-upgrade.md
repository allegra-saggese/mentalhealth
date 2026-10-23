# Plan — inference upgrade (pre-specified, nothing run)
**2026-10-22** · status: DRAFT, awaiting the owner's decisions marked **[DECIDE]**. Written before any of these p-values is read.

## Why
Reported inference is state-clustered standard errors with 48 clusters, plain z/t critical values, no adjustment for 7 outcomes × several treatments.
In `2026-10-22_ANALYSIS-ASSESSMENT.md`: 13 of 28 reported A2 cells are significant unadjusted, 3 survive Holm/Benjamini–Hochberg, and for E2 across the 7 outcomes none do.
Three separate questions follow: are the cluster-robust p-values reliable with 48 uneven clusters (wild bootstrap); does the result survive a design-based test that does not lean on a sampling model (permutation); and what is the family of tests being corrected (multiple testing).

## Part 1 — wild cluster bootstrap over states
- **Method.** Restricted (null-imposed) wild cluster bootstrap-t for the coefficient of interest, clusters = states (Cameron–Gelbach–Miller). State sizes are very uneven (some states contribute few rural counties), so use Webb six-point weights (MacKinnon–Webb) and report the number of draws (9,999).
- **Equation.** For `Y_it = β·D_it + X_it'γ + a_i + d_t + ε_it`, impose `β = 0`, form residuals `ε̃`, resample `ε*_it = w_s(i)·ε̃_it` with one weight per state, rebuild `Y*`, re-estimate, and compare the observed t-statistic with the bootstrap distribution.
- **Applies to:** TWFE for T1, T3, T4, E2 × 7 outcomes. CS keeps its own state-resampling bootstrap (already used).
- **Dependency (done 2026-10-22):** `wildboottest==0.3.2` and `statsmodels==0.14.5` are pinned in `requirements.txt`; `wildboottest` is **not** installed with pyfixest 0.60.0 (it is imported lazily), so it must be installed separately. Tested with absorbed county + year fixed effects: it runs. `bootstrap_type` accepts "11", "31", "13", "33" for clustered errors; weight types are `rademacher`, `mammen`, `webb` (the pyfixest docstring says `normal`, which raises an error here; `inference_tools.wild_cluster_test` rejects it up front).

## Part 2 — permutation (randomization) inference, registry B12
- **Question.** Is the observed TWFE E2 coefficient unusual relative to coefficients obtained when the timing of treatment is reassigned at random?
- **Design.** Keep the set of ever-treated counties and the cohort sizes; reassign each treated county's first-treated year by shuffling the observed cohorts across the ever-treated counties **within state** (preserves state-level timing shocks), keep never-treated counties fixed; recompute β each time; `p = share of draws with |β_perm| ≥ |β_obs|`. 2,000 draws, `default_rng(seed)`.
- **What it tests:** the sharp null of no effect for any county. Stronger than a zero-average null, so it is a complement to Part 1, not a substitute.
- **Scope.** TWFE on Poor MH Days and Frequent Mental Distress (primary), then the other outcomes. CS is too slow for 2,000 refits unless the bootstrap draws inside CS are cut to a handful; leave CS out of the first pass **[DECIDE]**.
- **Event-study version.** Compare the observed joint pre-trend statistic with the permuted distribution (the use named for A20 in the registry).

## Part 3 — multiple-testing family and correction (D-39)
**DECIDED 2026-10-22 (D-42):** the confirmatory family is the two mental-health outcomes across T1, T3, T4 and E2. Chosen after the review had shown this family's counts, and recorded as such in D-42.

| Family | Members | Correction | Purpose |
|---|---|---|---|
| F1 confirmatory | Poor MH Days and Frequent Mental Distress × {T1, T3, T4, E2} (8 tests) | Holm (family-wise error); Romano–Wolf step-down using the joint wild-bootstrap draws from Part 1 | the claim about mental health |
| F2 exploratory | all 28 reported A2 cells (adds deaths of despair and the four crime outcomes) | Benjamini–Hochberg (false discovery rate) | screening, labelled exploratory; crime outcomes fail pre-trends (D-23) |

Report unadjusted and adjusted p-values side by side; the family and the correction go in the table note. Romano–Wolf needs the bootstrap statistics for all members of a family drawn with the same state weights (Part 1 must keep the draws).

## Part 4 — pre-trend inference dependency
HonestDiD on the CS event study needs a bootstrap covariance matrix (open item in `2026-10-17_MODEL-RUN-STATE.md`). Part 1 and Part 2 do not need it, but the CS pre-period results (5 of 11 pre-period event times significant for Poor MH Days) need to be reconciled with the flat TWFE leads before either is cited.

## Implementation (nothing built yet)
- New helper module `inference_tools.py` (**created and tested 2026-10-22**, `tests/test_inference_tools.py`): `wild_cluster_pvalue`, `permutation_pvalue`, `holm`, `romano_wolf`, `bh_qvalues`. Each tested on a simulated panel with known truth and on a null panel (size near 5%).
- New script `script4z-inference.py` (or `script4zb-…` if the triangulation script takes `4z`) writing dated CSVs; exhibit scripts read them; no estimation in exhibit scripts.
- `PYTHONHASHSEED=0`, `default_rng(seed)` once, seeds recorded in the output file.

## Verification
(i) Size check on a null panel: rejection rate near 5% for both Part 1 and Part 2; (ii) the bootstrap p-value matches the cluster-robust p-value approximately on a balanced toy panel with many clusters; (iii) permutation test recovers a planted effect; (iv) `/audit-reproducibility` on any p-value that enters the paper; (v) each family and correction recorded in the decision register before the run.

## Risks
With 3 treated cohorts and 2 for Frequent Mental Distress, the permutation distribution of cohort assignments is limited; report how many distinct assignments exist. Few treated clusters concentrated in a handful of states make wild-bootstrap p-values conservative or unstable; show the weight scheme and a rerun with Rademacher weights as a check.

## Verified references (fresh-context check, 2026-10-22)

| Method | Reference |
|---|---|
| Wild cluster bootstrap-t | Cameron, Gelbach & Miller, "Bootstrap-Based Improvements for Inference with Clustered Errors", *Review of Economics and Statistics* 90(3):414–427, 2008, doi:10.1162/rest.90.3.414 |
| Uneven cluster sizes | MacKinnon & Webb, "Wild bootstrap inference for wildly different cluster sizes", *Journal of Applied Econometrics* 32(2):233–254, 2017, doi:10.1002/jae.2508 |
| Six-point (Webb) weights | Webb, "Reworking wild bootstrap-based inference for clustered errors", *Canadian Journal of Economics* 56(3):839–858, 2023, doi:10.1111/caje.12661 (earlier: Queen's Economics Department WP 1315) |
| Step-down family-wise error | Romano & Wolf, "Stepwise multiple testing as formalized data snooping", *Econometrica* 73(4):1237–1282, 2005, doi:10.1111/j.1468-0262.2005.00615.x |
| Holm; Benjamini–Hochberg | Holm, *Scandinavian Journal of Statistics* 6(2):65–70, 1979; Benjamini & Hochberg, *Journal of the Royal Statistical Society B* 57(1):289–300, 1995 |
| Sensitivity to parallel trends | Rambachan & Roth, *Review of Economic Studies* 90(5):2555–2591, 2023, doi:10.1093/restud/rdad018; R `HonestDiD` 0.2.8. Relative magnitudes use the conditional/least-favorable hybrid (C-LF) method; FLCI is for the smoothness restriction |
