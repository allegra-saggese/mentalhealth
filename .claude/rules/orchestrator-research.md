---
paths:
  - "script*.py"
  - "explorations/**"
  - "qa-scripts/**"
---

# Research Project Orchestrator (simplified)

For analysis scripts, estimation and exhibits, use this loop instead of the multi-agent runtime.

```
Analysis task (user-requested, plan approved)
  Step 1: IMPLEMENT  execute the plan steps
  Step 2: VERIFY     run the code, check outputs (verification-protocol.md)
                     stochastic code: seeded; estimation: self-checks print zero difference
                     exhibits: file exists, opened, standards met
                     failure -> fix -> re-verify (at most 2 retries, then report)
  Step 3: SCORE      quality-gates.md rubric (python)
  Done if >= 80; otherwise fix blocking issues, re-verify, re-score
```

One agent, no review fan-out. The fan-out runtime in `orchestrator-protocol.md` is reached only through a skill that invokes it (`/review-paper --peer`, `/seven-pass-review`, `/deep-audit`).

## Checklist

- [ ] Script runs without error, `PYTHONHASHSEED=0`
- [ ] Imports at top; no hardcoded absolute paths; no keys in code
- [ ] `default_rng(seed)` once at top if stochastic
- [ ] Output files at the expected paths, dated
- [ ] Tolerance checks pass (rtol 1e-6, atol 1e-8; SEs 1e-4 across hash seeds)
- [ ] Anything touching an estimand, sample, control set or inference was sent back to the user, not decided
- [ ] Quality score >= 80
