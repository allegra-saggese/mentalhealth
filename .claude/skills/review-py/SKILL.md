---
name: review-py
description: Read-only Python code review for the analysis pipeline. Checks reproducibility (hash seed, RNG, paths, secrets), data-handling hygiene (merges, FIPS, silent fills), estimation correctness against the definitions in script4_treatment.py and METHODS.md, exhibit standards, and legibility for an external reviewer; produces a report without editing. Use when the user says "review this script", "code review", "audit the analysis code", "check script4x", or when a script feeds a paper table or figure. NOT for running the code; pair with `/audit-reproducibility` for numeric verification.
argument-hint: "[filename | stage (e.g. 4) | 'all']"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Agent"]
disallowed-tools: ["Edit", "MultiEdit"]
---

# Review Python Scripts

## Steps

1. **Identify scripts to review.**
   - A specific filename: that file only.
   - A stage number (`0`, `1`, `2`, `4`): all `script{N}*.py` plus the shared modules it imports.
   - `all`: every `script*.py`, `estimators.py`, `designs.py`, `functions.py`, `packages.py`. Exclude `z-archive/`, `first-stage-testing/`.
2. **For each script launch the `py-reviewer` agent** (fresh context) to follow its protocol and read `.claude/rules/python-code-conventions.md`, `figure-and-table-standards.md`, `inference-robustness.md`, and `METHODS.md`. It returns its report as its final response (ending with the findings `json` block); this skill saves it to `quality_reports/reviews/<script>_py_review.md` (local).
3. **Summarise:** issues per script by severity, top three critical issues, and any finding that touches an estimand, sample, control set or inference (those go to the user, never auto-fixed).
4. **Leave source unchanged.** Only reports are produced; fixes follow user review.

## Troubleshooting

- A reviewer cites a rule that does not exist: the finding is dropped by `scripts/validate-findings.py`; re-dispatch once.
- A script that only defines constants (`script4_treatment.py`) is reviewed for definitions and docstrings, not for runtime behaviour.
