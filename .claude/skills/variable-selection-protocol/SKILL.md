---
name: variable-selection-protocol
description: Pre-specify and review the Phase 3 machine-learning variable-selection analysis before any model is fit — candidate set, role of each variable (confounder, mediator, collider, outcome-adjacent), cross-validation scheme respecting the county-year panel structure, selection method, stability criteria, and the post-selection inference plan. Use when the user says "variable selection", "lasso", "ridge", "ML framework", "which controls", "double selection", "phase 3", or before any selection model is run. Produces a dated protocol document to approve first; run results are compared to it afterwards.
argument-hint: "[outcome or 'all'] [--review <protocol path>]"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Bash", "Agent"]
---

# Variable-Selection Protocol (Phase 3)

Selection is only credible if specified before results are seen. This skill writes the protocol; it does not fit models. Companion rules: `inference-robustness.md` (post-selection inference), `decision-records.md`, `python-code-conventions.md`. Existing material: B-series in `code-audit/plans/2026-09-27_MODEL-NAMING.md`; pilots in `first-stage-testing/` (not used for results); variable roles and pre-trends in `script4j`, `script4k`; the control set decision D-16 / D-31.

## Protocol document (saved to `code-audit/plans/YYYY-MM-DD_variable-selection-protocol.md`, committed)

1. **Question and target.** What selection is for: choosing controls for the reported specification, discovering heterogeneity, or prediction. Selection for a causal coefficient uses the double-selection / double machine learning route below; prediction uses a held-out score.
2. **Candidate set.** Variable list from the variable register with causal role (pre-treatment confounder, mediator, collider, outcome-adjacent) from `script4j`. Post-treatment variables and colliders are excluded by rule, not by the algorithm. Coverage requirement for inclusion (share non-missing in the window).
3. **Estimand.** The coefficient whose inference matters (for example the TWFE effect of first expansion on each outcome), unchanged by selection.
4. **Method.** Primary and alternatives written in advance, each with its equation: lasso / elastic net for the outcome and for treatment (post-double-selection), cross-fitted double machine learning, ridge for stability comparison. Penalty chosen by cross-validation or plug-in rule, fixed in advance.
5. **Cross-validation scheme.** Folds grouped by state (clusters never split), time-respecting where prediction is the goal, fixed seed and fold assignment recorded. Fixed effects handled by demeaning within folds.
6. **Stability criteria.** Selection frequency across folds and seeds; a variable is "stable" at a pre-stated frequency threshold. Report the full path, not only the winners.
7. **Post-selection inference.** Estimate the coefficient with the selected controls on a sample not used for selection, or with cross-fitting / double selection; clustered-by-state SEs; never ordinary SEs on the selection sample.
8. **Comparison to the pre-committed set.** How the ML-selected set is compared with the 8-variable control set (overlap, change in the estimate); the comparison documents sensitivity and does not replace the pre-committed set unless the user decides so in a new D-record.
9. **Outputs and exhibits.** Table of selection frequencies, coefficient comparison, figure per `figure-and-table-standards.md`; scripts named in the stage-letter convention (a new Stage 3 letter series).
10. **Stopping rule and what will be reported if results are null or unstable.**

## `--review <protocol>`

Read-only audit of a protocol or of a finished run against it: every deviation (variable added, method changed, threshold moved) listed with its D-record or flagged as unregistered; confirm folds respect state clusters; confirm inference was not done on the selection sample.

## Decisions that are the user's

Candidate set, causal roles, method and threshold choices, and any change after seeing results. The skill drafts; the user approves and the protocol is committed before the first fit.
