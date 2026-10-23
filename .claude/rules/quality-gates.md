---
paths:
  - "script*.py"
  - "*.py"
  - "tests/**/*.py"
---

# Quality Review and Scoring (Python)

> **Framing:** thresholds are advisory at the harness level. `/commit` runs `scripts/quality_score.py` and halts on failure until the issue is fixed or explicitly overridden. The git pre-commit hook (`.githooks/pre-commit`, installed once with `./scripts/install-hooks.sh`) applies the hard checks to every commit.

## Thresholds

- **80 = commit**: good enough to save.
- **90 = cite / PR**: ready to quote in the paper or open a PR.
- **95 = excellence**: aspirational.

## Rubric (`.py` files)

`scripts/quality_score.py` is the single source for deductions; this table mirrors it. Rows caught by **script** are deducted automatically; rows caught by **reviewer** are a checklist for `/review-py`, `/deep-audit` or a human.

| Severity | Issue | Deduction | Caught by |
|----------|-------|-----------|-----------|
| Critical | Syntax error (`py_compile` fails) | -100 (auto-fail) | script |
| Critical | Secret or API-key literal | -100 (auto-fail) | script |
| Critical | Hardcoded user path (`/Users/...`, `~/Dropbox`) outside `packages.py` | -20 | script |
| Critical | Domain bug (wrong sample, wrong FE, wrong clustering, silent no-op term) | -30 | reviewer |
| Critical | Estimator re-implemented in an exhibit script instead of the shared module | -20 | reviewer |
| Major | Stochastic call (`random`, `np.random`, bootstrap, `train_test_split`, `KFold(shuffle=True)`) with no seed | -10 | script |
| Major | Figure saved without `dpi=` or without a vector copy | -5 | script |
| Major | Output written without a date stamp | -3 | script |
| Major | No count/uniqueness check after a merge | -5 | reviewer |
| Major | Missing the short standard header | -3 | script |
| Minor | `from x import *` in new code (other than `packages`) | -2 | script |
| Minor | Bare `except:` | -2 | script |
| Minor | Line over 120 characters | -1 | script |

## Enforcement

- **Score < 80:** `/commit` halts, lists blocking issues. Override only with an explicit signal ("commit anyway" plus a reason); the override is logged in the commit body.
- **Score < 90:** commit allowed, warning with recommendations.
- **Hard checks in the hook** (abort regardless of score): syntax error, secret literal, staged data file, hardcoded user path outside `packages.py`. Bypass with `SKIP_QUALITY_GATE=1` (score only) or `git commit --no-verify` (everything); give the reason in the commit message.

## Quality reports

Generated only at merge time, from `templates/quality-report.md`, to `quality_reports/merges/` (local).

## Tolerances

Defined in [`replication-protocol.md`](replication-protocol.md): counts exact; estimates and SEs rtol 1e-6 / atol 1e-8 at a fixed hash seed; SEs 1e-4 across hash seeds.
