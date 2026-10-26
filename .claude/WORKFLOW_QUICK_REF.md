# Workflow Quick Reference

**Model:** Contractor (you direct, Claude orchestrates). Onboarding mode on: extra check-ins until you say otherwise.

## The Loop

```
Your instruction
  -> [PLAN] if >3 files, shared module, estimation or deliverable substance -> you approve
  -> [EXECUTE] implement, run, verify
  -> [REPORT] what changed, what was run, what is ready (results before interpretation)
  -> repeat
```

## I ask you when

- Design fork: "Option A (fast) vs B (robust)?"
- A choice touches an estimand, sample, control set, inference or interpretive wording
- A shared module (`script4_treatment.py`, `estimators.py`, `designs.py`, `functions.py`, `packages.py`) must change
- Anything destructive or outward-facing (delete, push, PR, email)
- A replication check just misses tolerance

## I just execute when

- The fix is an obvious bug or pattern application
- Verification (tolerance checks, tests, compile, `py_compile`)
- Documentation (plans, logs, decision records, `[LEARN]` entries)
- Exhibits per `figure-and-table-standards.md`

## Onboarding check-ins (until you say stop)

1. At plan approval. 2. After each stage runs. 3. Before any re-estimation. Each with one short paragraph on what the workflow just did and why.

## Non-negotiables

- Paths only via `packages.py`; no hardcoded user paths elsewhere
- `PYTHONHASHSEED=0`; `default_rng(seed)` once at the top of stochastic code
- Reported set: county + year FE, SE clustered by state, T1/T3/T4/E2, TWFE + Callaway-Sant'Anna
- Tolerances: rtol 1e-6, atol 1e-8 (SEs 1e-4 across hash seeds)
- Publication-ready exhibits only; standards in `figure-and-table-standards.md`

## Quality gates

| Score | Action |
|-------|--------|
| >= 80 | Ready to commit |
| >= 90 | Ready to cite or open a PR |
| < 80 | Fix blocking issues |

## Exploration mode

Experimental work goes in `explorations/` (60/100 bar, no plan needed, 2-minute research-value check). See `.claude/rules/exploration-fast-track.md`. Anything that reaches the paper leaves exploration first.
