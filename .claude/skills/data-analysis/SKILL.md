---
name: data-analysis
description: End-to-end Python analysis workflow for the county-year panel — pre-flight, exploration, estimation with pyfixest, publication-ready tables and figures, review. Use when the user says "analyze this", "run a regression on X", "explore this CSV", "add a specification", "full analysis workflow", or points at a panel file and asks for empirical results. Produces stage-lettered scripts (script{N}{letter}-description.py) and dated outputs under Data/output/. Estimation choices that touch an estimand, sample, control set or inference return to the user.
argument-hint: "[panel path or description of the analysis goal]"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Edit", "Bash", "Agent", "Monitor"]
---

# Data Analysis Workflow (Python)

**Input:** `$ARGUMENTS` — a panel file path or an analysis goal (for example "event study of deaths of despair on first expansion, state-clustered").

## Constraints

- Follow `.claude/rules/python-code-conventions.md`, `inference-robustness.md`, `figure-and-table-standards.md`, `results-reporting.md`.
- Treatments, sample, outcomes and controls come from `script4_treatment.py`; estimators from `estimators.py` / `designs.py`. Do not re-define them. If something is missing from the shared module, propose the change to the user (shared-module change).
- New scripts follow the stage-letter convention and the short header; outputs go to `Data/output/{tables,figs}/` via `packages.py`, dated.
- Run with `PYTHONHASHSEED=0`. Seeds via `default_rng` once at the top.
- Run the `py-reviewer` agent on the generated script before presenting results.
- Results first, interpretation only when asked.

## Phase 0: Pre-flight report (before any code)

```markdown
## Pre-Flight Report
**Panel:** <path, run date>  rows / counties / years / key variables (from head(), dtypes)
**Missing data:** % missing for each variable used; any D-34 zero-rule effect
**Conventions read:** python-code-conventions, inference-robustness, decision register entries that bear on this task (D-ids)
**Task in one sentence:** ...
**Specification in plain language:** treatment, outcome, sample window, fixed effects, clustering, controls, estimator, and the estimand
**Equation:** <one line, as in METHODS.md>
**Plan:** 3-5 bullets
**Needs your decision:** any estimand / sample / control-set / inference choice
```

If an input cannot be read, or a decision is open, stop and ask.

## Phase 1: Setup
Header, imports, directories from `packages.py`, `rng` if stochastic, load via `latest_file_glob()` and `normalize_panel_key()`, print counts.

## Phase 2: Exploration
Summary statistics and missingness, distributions of key variables, pre-treatment means by treatment group, outcome trends over time, balance. Save diagnostic figures to the script's `figs_dir` subfolder.

## Phase 3: Estimation
- Panel estimation with `pyfixest` (`feols` with county and year fixed effects); clustered by state unless the user chose otherwise (document why).
- Start from the pre-committed specification; progressive controls follow `script4y`'s pattern, not ad hoc.
- Confirm each new term changes the fit.
- Report standardized and raw effects; leads/lags for event studies; Callaway-Sant'Anna via `estimators.py`.
- Each specification run is appended to the ledger (below), kept or not.

### Specification ledger
Every specification estimated, including dropped and failed ones, gets one row appended to `quality_reports/spec-ledger.md` (local, append-only; a correction is a new row). Columns: date, commit, script:line, outcome, specification, sample, status (kept / dropped / failed), why, estimate. `/spec-register` maintains and audits it against the registry A-IDs and decision D-IDs.

## Phase 4: Exhibits
Tables: results CSV then LaTeX via the shared writer (booktabs, clustered-SE note, no stars by default). Figures: per `figure-and-table-standards.md`, PDF and PNG. Run `/figure-check` before citing.

## Phase 5: Review and save
1. Outputs dated and saved; counts printed.
2. `py-reviewer` on the new script; fix Critical/High items that are mechanical; return anything touching specification to the user.
3. Report: what was run, on which panel, with which seed, what was not run.

## Important
- **Reproduce, do not guess.** If the user specifies a regression, run exactly that.
- **Show your work.** Print summary statistics and counts before regressions.
- **Check for problems:** collinearity (dropped terms in pyfixest output), outliers, perfect prediction, few clusters.
- No hardcoded values: sample restrictions and windows are named constants in the shared module.

## Long-running fits
For fits over a couple of minutes (`script4u` about 30 minutes, `script4x` about 25), launch with Bash `run_in_background: true`, log to a file under `Data/output/`, and use the Monitor tool on `tail -f` of that log filtered for milestones and `Error|Traceback`. Do not poll with sleep loops.
