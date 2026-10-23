# Cross-check of `did_cs.py` against independent Callaway-Sant'Anna implementations
**2026-10-23** · script `qa-scripts/qa_cs_crosscheck.py` (+ `qa_cs_crosscheck.R`, `qa_cs_crosscheck_py.py`); outputs in
`Data/output/tables/qa_cs_crosscheck/` (cells, summary, and the extracts used). Nothing in the pipeline was changed.

## Question
Does `did_cs.att_gt_table` (outcome-regression Callaway-Sant'Anna, not-yet-treated comparison, base year g-1) agree with
R `did` 2.5.1, `csdid` 0.4.2 and `moderndid` 0.3.1? All four were run on one identical balanced extract per outcome (counties
with the outcome and the 8 pre-committed controls in every window year; first-expansion cohorts), with the same settings.

## Result
| Outcome (window; counties; cohorts) | Post-treatment cells | Overall ATT | Pre-treatment cells |
|---|---|---|---|
| Poor MH Days (2010-2023; 2,171; 2012, 2017, 2022) | 18 of 18 agree, largest gap 0 to 3e-10 | 0.041667 in all four, difference 0.0 | 12 compared, largest gap 0.0116 days |
| Frequent Mental Distress (2016-2023; 2,251; 2017, 2022) | 9 of 9 agree, largest gap 3e-10 | 58.458812 in all four, difference 0.0 | 5 compared, largest gap 16.2 (one cell 7.7) |

The three packages agree with each other to the digit in every cell. **Every post-treatment estimate and the overall ATT of `did_cs.py` is
reproduced exactly (1e-6 tolerance, replication protocol).**

## The one difference: covariate timing in pre-treatment cells
17 pre-treatment (placebo) cells differ, all with event time before the base year. Reproduced by hand on four cells: `did_cs.py` measures the
covariates at the base year g-1; the three packages measure them at the earlier of the two years in the pair, which for a pre-treatment cell
is year t. Neither is a bug; they are two conventions. It changes only the pre-period placebo estimates, not any post-treatment estimate.

## Consequence and what is not decided
The pre-period results (CS shows 5 of 11 significant pre-period event times for Poor MH Days, while TWFE leads look flat) depend in part on this
convention. **Not decided:** which convention the paper's CS event study should use. If the packages' convention is adopted, the pre-period
placebo cells and their significance would be re-estimated; that is a researcher choice (estimand-adjacent) and would get a D-number. Standard errors
were not compared (`did_cs.py` bootstraps over states; the packages use analytic influence functions).

## Resolution (2026-10-23): D-44 adopted, re-estimated, verified
The packages' convention was adopted (D-44; commit `5d0745b`). After the change `did_cs.py` agrees with R `did`, `csdid` and `moderndid`
in all 44 compared cells (largest gap 0.0). `script4a`, `script4u` and `script4x` were re-run in full (`PYTHONHASHSEED=0`); the exhibit
scripts `script4n`, `4p`, `4q` were re-run afterwards.

**Unchanged, verified against the 2026-10-22 outputs:** every post-treatment ATT(g,t) and the overall ATT and event-time estimates at e >= 0 in
`script4a` (A6) and `script4u` (largest difference 0.0); all 56 CS rows and all 224 TWFE rows of `script4x` (0.0); the self-checks of `script4u`
against `script4a` are identical to the previous run (1e-14 or smaller).

**Changed:** pre-treatment cells in the covariate-adjusted arm (268 of 536 att_gt cells and 112 of 224 aggregated rows in `script4a`; 427 of 428
pre-treatment rows in `script4u`). Pre-treatment event times significant at 5%, covariate-adjusted CS, before -> after:

| Outcome (`script4a`, A6) | Pre-treatment event times | Significant before | Significant after |
|---|---|---|---|
| Poor MH Days | 11 | 5 | 6 |
| Frequent Mental Distress | 5 | 0 | 0 |
| Deaths of Despair | 16 | 2 | 3 |
| Aggravated Assault | 20 | 6 | 5 |
| Assault, all severities | 20 | 4 | 6 |
| Violence index (partial) | 20 | 5 | 3 |
| NIBRS curated total | 20 | 5 | 5 |

`script4u`, specification "controls": first expansion 27 -> 28 of 112; first entry (clean controls) 9 -> 12 of 102.

**Reading.** The convention did not cause the pre-period problem: Poor MH Days stays at about half of its pre-treatment event times significant (5 of 11
before, 6 of 11 after). These are single-cell tests at 5%, many far leads rest on one cohort, and no joint pre-trend test for CS has been run; the TWFE leads
test (p = 0.43 for Poor MH Days) and the CS pre-period cells still disagree, and that disagreement is open (needs a CS joint pre-test and HonestDiD on CS).
