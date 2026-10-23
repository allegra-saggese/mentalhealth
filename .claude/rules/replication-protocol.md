---
paths:
  - "script*.py"
  - "tests/**/*.py"
  - "quality_reports/passports/**"
---

# Replication Protocol

**Core principle:** every number in the paper traces to a script, a panel file and a run date; a reviewer who runs the scripts with the pinned environment gets the same numbers.

## Two kinds of replication

1. **Replicating a published result** (for example a benchmark estimate from a cited paper): reproduce it to the dot before extending. Record targets (table, column, value, SE), translate line by line, match specification exactly (covariates, sample, clustering, SE type), and do not "improve" during replication. Only then extend.
2. **Replicating our own pipeline** (rerun, refactor, new machine): outputs must match a saved baseline at the tolerances below. Use `tests/compare_outputs.py <baseline_root> <live_root>`.

## Tolerance thresholds

| Quantity | Tolerance | Rationale |
|----------|-----------|-----------|
| Counts (N, counties, county-years) | exact | no reason for any difference |
| Point estimates (same machine, `PYTHONHASHSEED=0`) | rtol 1e-6, atol 1e-8 | numerical noise only |
| Standard errors, same seed | rtol 1e-6, atol 1e-8 | deterministic |
| Standard errors across hash seeds | 1e-4 | known pyfixest hash-seed effect (about 2e-5) |
| Published values vs ours | within the rounding of the published digits | display rounding |
| p-values | same side of the pre-stated threshold | exact p may differ slightly |
| PNG exhibits | identical pixels at fixed seed and fonts | same renderer |

## If there is a mismatch

Do not proceed to extensions. Isolate the step that introduces it (sample, SE computation, defaults, variable construction, hash seed, panel version) and document the investigation even if unresolved. Use [`/diagnose`](../skills/diagnose/SKILL.md) to bisect the pipeline and [`/differential-audit`](../skills/differential-audit/SKILL.md) to compare two implementations.

The mismatch does not presume the code is right. The on-disk output is a challenger, not an oracle: one of {paper, code} must change, and the task is to find which. A named, concrete alternative specification that explains the gap marks the claim EXPLAINED; a vague note never downgrades a FAIL.

## Before citing any output

1. Read `code-audit/plans/*MODEL-RUN-STATE.md`: which script ran on which panel, when.
2. Confirm the output post-dates the current panel (the panel was rebuilt 2026-09-23; older outputs are not citable).
3. Run `PYTHONHASHSEED=0`; note the package versions from `requirements.txt`.

## Replication report (local, per check)

`quality_reports/` (git-ignored):

```markdown
# Replication Report: <target or pipeline stage>
**Date / commit:**  **Panel file and run date:**  **Seed / env:**
| Target | Reference | Ours | Diff | Status |
## Discrepancies  (target, investigation, resolution)
```

## Enforcement

[`/audit-reproducibility`](../skills/audit-reproducibility/SKILL.md) parses numeric claims from the manuscript, finds the matching values under `Data/output/`, and compares at the tolerances above. Run it before submission and before building the replication package ([`/replication-package`](../skills/replication-package/SKILL.md)); aim for zero FAILs.

## Claims provenance: `passport.yaml`

A per-paper YAML file (`quality_reports/passports/<paper-slug>.yaml`, from `templates/passport-template.yaml`) records, for each numeric claim in the manuscript, the script invocation and output file that produced it and every artifact that displays it (`appears_in`). Status values: PASS, FAIL, EXPLAINED (requires a named alternative specification), STALE (an input changed after verification), UNVERIFIED. Rules:

- Numeric claims only; prose and citations go through `/verify-claims`.
- Do not auto-populate the mapping without showing the user the inferred script and output; do not promote UNVERIFIED to PASS without running the audit.
- Two displays of one number are copies, not measurements: compare at the coarser display precision; a display that cannot be found is a FAIL, never a skip. Never fix a disagreement by typing one value over the other; re-derive both from the output.
- Adapt for Python: `source_file` is a `script*.py`, `output_file` a dated CSV/JSON under `Data/output/`, `output_field` a column or row key.
- The deposit version is exported into the replication package; the working passport stays local.

## Auto memory is an un-logged input

Native auto memory (`~/.claude/projects/.../memory/`) is machine-local and can shape a session invisibly. For a session that produces a reported number, note in the analysis log that it was active; for a final numbers-freeze or replication build, run with `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`. Anything that belongs to the project is promoted into the committed `MEMORY.md` via `/promote-memory`.
