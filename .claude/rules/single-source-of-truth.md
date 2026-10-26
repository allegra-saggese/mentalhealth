---
paths:
  - "script*.py"
  - "METHODS.md"
  - "README.md"
---

# Single Source of Truth

**One definition per thing; everything else is derived.**

```
raw data (Dropbox, never edited)
  -> script0* (clean)  -> script0f (panel)  -> script4_treatment.py (treatments, sample, outcomes, controls)
  -> estimation scripts (script4a, 4r, 4u, 4x, ...)  -> result CSVs in Data/output/
  -> exhibit scripts (4n, 4p, 4q, 2b, 4v, ...)  -> tables and figures  -> manuscript
```

## Where each thing is defined

| Thing | Authoritative file |
|-------|--------------------|
| Treatments, sample, outcomes, control set, `REPORT_*` | `script4_treatment.py` |
| Estimator equations and definitions | `METHODS.md` |
| Estimator code (TWFE, event study, Callaway-Sant'Anna) | `estimators.py`, `designs.py` |
| Paths and directories | `packages.py` |
| FIPS, loaders, per-capita transform | `functions.py` |
| Why each control/decision is what it is | `code-audit/plans/CONTROL-DECISION-REGISTER.md` |
| Which script ran on which panel | `code-audit/plans/*MODEL-RUN-STATE.md` |
| Figure and table style | `figstyle.py` and `figure-and-table-standards.md` |

## Rules

- **Never edit a derived artifact by hand** (a result CSV, a table fragment, a figure). Change the source, rerun, regenerate.
- **Exhibit scripts do not estimate.** They read the newest result file; if a number is missing, the estimation script gets the change.
- **Change a definition in one place** and run [`/blast-radius`](../skills/blast-radius/SKILL.md) over every consumer before and after (every `script4*` imports from `script4_treatment.py`).
- **A duplicated copy is a defect to schedule, not to extend** (for example estimation logic repeated between stage-4 scripts, scheduled for the script consolidation).
- When documentation and code disagree, the code that produced the reported number wins; fix the documentation and note it in the decision register.
