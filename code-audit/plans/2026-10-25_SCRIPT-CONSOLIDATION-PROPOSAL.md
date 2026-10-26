# Script consolidation proposal (after stage 0 and stage 1) — 2026-10-25
Status: PROPOSAL, nothing executed. Written while the D-53 re-runs were in progress; no script that was running was touched.

## Where we are
Stage 0 has 6 scripts (`script0a` to `0f`), stage 1 has 4 (`script1a` to `1d`). Stage 4 still has **21 Python scripts + 2 R scripts**, plus 7 shared modules, 6 FSIS sub-scripts, 6 QA one-offs and 6 tooling scripts. Most of the remaining sprawl is stage 4.

## Stage 4 today: purpose and a suggested home
| Script | Purpose (one line) | Suggestion |
|---|---|---|
| `script4a-design-estimates.py` (812) | TWFE grid (A2), dairy x FSIS (A4), event study (A5), Callaway-Sant'Anna (A6) | Keep as the core of `script4a-estimates.py`; absorb 4u, 4r, 4x as sections |
| `script4u-did-by-treatment.py` | TWFE, event study, CS for first expansion and first entry (it must reproduce 4a's first-expansion numbers exactly) | Section `designs` of 4a (same estimators, different design loop; removes the duplicated self-check) |
| `script4r-fsis-treatment.py` | Slaughterhouses as the treatment | Section `fsis` of 4a |
| `script4x-species-forest.py` | Per-species TWFE and CS, forest plots | Section `species` of 4a; its plots go to the figures script |
| `script4j-control-diagnostics.py` | Control roster, covariate pre-trends, VIF | `script4b-diagnostics.py` section `controls` |
| `script4t-pretrend-leads.py` | Outcome leads for each treatment | `4b` section `leads` |
| `script4d-balance-panel.py` | Balance and detectable-effect tables | `4b` section `balance` |
| `script4y-control-set-selection.py` | Add-one and drop-one control sensitivity | `4b` section `control_sets` |
| `script4w-honestdid-export.py` + `4w-honestdid.R` | Inputs and R run for Rambachan-Roth sensitivity | `4b` section `honestdid` (exports); the R file stays as a helper in `r/` |
| `script4z-inference.py` | Wild bootstrap, Holm, Romano-Wolf (D-42, D-43) | `script4c-inference.py` |
| `script4ze-permutation-inference.py` | Within-state permutation test (D-47) | `4c` section `permutation` |
| `script4za-estimator-triangulation.py` | Six estimators on E2 and E3 (D-45) | `script4d-triangulation.py` section `estimators` |
| `script4zc-treatment-by-model.py` | Every treatment against every model, incl. dCDH for T1, T3, T4 (D-46) | `4d` section `treatment_by_model` |
| `script4za-dcdh.R` | R helper called by `estimators.dcdh_estimates` | Move to `r/` |
| `script4zb`, `4zd`, `4zf` | Forest plots (triangulation, treatment-by-model) and permutation figure | One `script4e-figures.py`, plus the figures now inside 4x and 4y |
| `script4n`, `4p`, `4q` | Regression tables, full appendix, Beamer frames (no estimation) | One `script4f-tables.py` with three sections |
| `script4b-covariate-sequence.py`, `4c-covariate-stability.py` | Superseded by 4y; kept as the 09-23 record | Move to `z-archive/` (headers already say superseded) |
| `script4_treatment.py` | Treatment, sample, outcome, control definitions | Keep; the single source |

Result: stage 4 goes from 21 Python scripts to **6** (`4a` estimates, `4b` diagnostics, `4c` inference, `4d` triangulation, `4e` figures, `4f` tables) plus `script4_treatment.py`. Each keeps `--only <section>` so a partial run stays possible (the long jobs, such as dCDH at about 30 minutes, are sections).

## Shared modules (7)
| Module | Suggestion |
|---|---|
| `did_tools.py` (89 lines), `did_cs.py` (164), `estimators.py` (366) | Merge into one `estimators.py` (about 620 lines): helpers, Callaway-Sant'Anna, the six triangulation estimators |
| `designs.py` (135) | Keep (designs, county groups, outcome windows, `usable_sample`); depends on `script4_treatment` |
| `inference_tools.py`, `figstyle.py`, `packages.py` | Keep |
| `functions.py` (551) | Keep for now; its "used in" comments are stale. Later: split loaders, geojson and FIPS helpers |

## Smaller clean-ups (no risk)
- `fsis-scripts/script0i` and `0j` (the "targeted" HUD fill) are the non-default `--hud-strategy targeted` path of `script0e-fsis-clean-all.py`; the panel in use was built from the bulk fill (`script0k`). Archive 0i and 0j and drop that option, or keep them if you still use it (your call).
- `qa-scripts/`: the three Callaway-Sant'Anna cross-check files (`qa_cs_crosscheck*.py/.R`) move into `qa-scripts/cs-crosscheck/`; `diagnostic_post_clean.py`, `pull_2022_cattle_totals.py`, `qa_2017_dta_vs_api.py` are one-offs whose findings are in `code-audit/`: archive.
- `scripts/` (6 tooling scripts) is infrastructure (hooks, quality gate, issue filing); no change.

## Order (each step must leave every output unchanged)
1. Archive superseded and one-off scripts (4b, 4c, QA one-offs, optionally 0i/0j): moves only.
2. Merge the three estimator modules into `estimators.py`; update imports; `tests/test_triangulation.py` and `tests/test_inference_tools.py` pass.
3. Merge into `script4a-estimates.py` (4u, 4r, 4x); then `4b`, `4c`, `4d`; then the exhibit scripts `4e`, `4f`.
4. Update README run order, `CLAUDE.md`, `MODEL-RUN-STATE`, register entry D-54.

## How equivalence is proved
The D-53 re-runs of 2026-10-25 are the baseline. For each merged script I run the old script and the new section in a scratch home folder on the same panel and compare every CSV (counts exact, values at rtol 1e-6, atol 1e-8 via `tests/compare_outputs.py`); figures by inventory (every old file stem has a new one) and `figure_check`. Estimation scripts run with `PYTHONHASHSEED=0`. A merge that changes any number is reverted, not patched.

## Needs your decision
1. Is **6 scripts for stage 4** the right target, or do you want the exhibit scripts (4e, 4f) merged too?
2. Archive `fsis-scripts/script0i` and `0j`?
3. Keep the names `script4a`..`script4f`, or rename them for their role (for example `script4-estimates`)? Renaming touches every reference.
