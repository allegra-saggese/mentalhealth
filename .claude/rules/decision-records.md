---
paths:
  - "code-audit/plans/**"
  - "script4_treatment.py"
---

# Decision Records

Every choice that changes what is estimated or reported gets a numbered decision (D-xx) in `code-audit/plans/CONTROL-DECISION-REGISTER.md` (or a dated note linked from it). The register is what a referee or coauthor reads to see why the specification is what it is.

## What needs a decision record

- Inclusion or exclusion of a control, outcome, treatment definition, sample restriction, or analysis window.
- A change to the zero rule, the forward-fill rule, the rural filter, or any variable construction.
- A choice of estimator, fixed effects, clustering level, or control group (never-treated vs not-yet-treated).
- Narrowing what is reported (the `REPORT_*` set): the specification keeps being estimated; the record explains why it is not shown.

## Format (one entry, kept short)

`D-xx | date | decision | alternatives considered | reason | evidence (file, number) | pre-committed or after seeing results | who decided`

Use `templates/decision-record.md` for anything longer than a paragraph.

## Rules

- **Pre-commit before looking.** If a choice was made after seeing results, the entry says so. Do not back-date.
- **Never delete or rewrite an entry.** Supersede it with a new D-number that links back; both stay.
- **Cite the ID** in the script that implements the decision, in exhibit notes where relevant, and in the paper's data appendix.
- **Open blockers** (for example A21, A22) live in the same folder and are listed in `CLAUDE.md` until closed.
- A choice that touches an estimand, sample, control set or inference is the user's decision. Draft the record; do not finalise it unprompted.
