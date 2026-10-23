---
name: robustness-suite
description: Plan and run the standard robustness battery for a headline estimate (TWFE and Callaway-Sant'Anna on the reported set) and assemble one summary exhibit — alternative clustering and wild-cluster bootstrap, leave-one-state-out, drop-influential-counties, control-set sweep, analysis-window sensitivity, alternative comparison group, placebo outcomes or leads, and multiple-outcome adjustment. Use when the user says "robustness checks", "run the robustness suite", "stress-test this estimate", "what would a referee ask", or before an estimate is called headline. The battery is proposed in a plan the user approves; checks that weaken the result are reported with equal prominence.
argument-hint: "[estimate id, e.g. A2 / T1 x deaths_of_despair] [--plan-only]"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Edit", "Bash", "Agent", "Monitor"]
---

# Robustness Suite

For one headline estimate at a time. Standards: `.claude/rules/inference-robustness.md`, `results-reporting.md`, `replication-protocol.md`.

## Phase 0: Pre-flight and plan (always; stop here with `--plan-only`)

Echo: the estimate (registry A-id, treatment, outcome, estimator, fixed effects, clustering, window, control set), the panel file and run date, and the open decision/blocker IDs that bear on it (A21 window, A22 pre-trends). Then propose the battery below as a checklist with a cost estimate per item (long runs: `script4u` about 30 minutes, `script4x` about 25). **The user approves the list before anything runs**; any item that changes an estimand, sample or control set is labelled "researcher decision".

## The battery (default menu; the approved plan fixes the actual list)

| # | Check | What it answers |
|---|-------|-----------------|
| 1 | Alternative clustering: county, state (reported), state-by-year; wild-cluster bootstrap p-value for state clusters | Is inference driven by the clustering choice or few clusters? |
| 2 | Leave-one-state-out (all states) | Is the estimate driven by one state? |
| 3 | Drop the top influential counties (by leverage / leave-one-county-out) | Outlier sensitivity |
| 4 | Control-set sweep (`script4y`: add-one-in, take-one-out) | Sensitivity to controls; documents, does not choose |
| 5 | Window sensitivity (full panel vs 2010-2015 vs 2010-2020) | The standing A21 question |
| 6 | Comparison group: never-treated vs not-yet-treated (Callaway-Sant'Anna) | Control-group choice (A22) |
| 7 | Pre-trend evidence: leads, Rambachan-Roth sensitivity (`script4w`) | Parallel-trends credibility |
| 8 | Placebo outcome or placebo treatment timing | Spurious-correlation check |
| 9 | Multiple outcomes: declared family, Holm / Benjamini-Hochberg, Romano-Wolf if feasible | Familywise error across the 7 outcomes |
| 10 | Alternative treatment definitions in the reported set (T1, T3, T4, E2) | Definition sensitivity |

## Phase 1: Run

Reuse existing scripts (`script4a/4t/4u/4w/4x/4y`) wherever they already compute the check; write a new stage script only for what is missing, following the stage-letter convention, `PYTHONHASHSEED=0`, shared modules for definitions. Each specification run is appended to the spec ledger (`/spec-register`). Long runs go in the background with a log.

## Phase 2: Summarise

One summary table (check, estimate, SE or CI, N, change relative to the headline, verdict "stable / weakens / reverses") and one figure (specification-curve style) per `figure-and-table-standards.md`. State the family and the correction. **Report the checks where the result weakens, first.** No interpretation until asked; flags separate.

## Phase 3: Record

Decision-register entry for anything chosen after seeing results (labelled as such); output files dated; run state note updated. Anything that fails a tolerance or contradicts the headline is raised to the user, not smoothed over.
