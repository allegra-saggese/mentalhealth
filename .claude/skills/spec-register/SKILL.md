---
name: spec-register
description: Maintain and audit the append-only specification ledger — one row for every specification estimated (kept, dropped or failed) — and reconcile it with the analysis registry (A-ids), the decision register (D-ids) and the REPORT_* set in script4_treatment.py. Use when the user says "log this specification", "spec register", "what have we run", "which specifications are unregistered", "what else did we try", or before a result is quoted or a robustness exhibit is built. Flags specifications run without a registry id, reported results without a decision, and decisions made after seeing results that are not labelled as such.
argument-hint: "[add | audit | show <A-id or D-id>]"
allowed-tools: ["Read", "Grep", "Glob", "Write", "Bash"]
---

# Specification Register

The record a referee's "what else did you try?" deserves. It keeps the search visible; it never advises which specification to run.

**Files (local, git-ignored):** `quality_reports/spec-ledger.md`. **Committed sources it is reconciled against:** `code-audit/plans/2026-09-17_analysis-registry.md` (A-ids), `CONTROL-DECISION-REGISTER.md` (D-ids), `script4_treatment.py` (`REPORT_*`).

## `add` — append a row

One row per specification, as it is run, with: date, commit (short SHA, `-dirty` if uncommitted), script:line, outcome, specification (formula, fixed effects, clustering, estimator), sample and window, panel file and date, A-id, D-ids that apply, status (`kept` / `dropped` / `failed`), why (required unless kept), estimate (optional), pre-committed or after seeing results. Append only; never edit or delete a past row. A correction is a new row that cites the earlier one. Escape `|` inside a formula as `\|`.

## `audit` — reconcile

1. Every row has an A-id (or is flagged "unregistered"); every A-id marked built has at least one row.
2. Every specification shown in an exhibit (`REPORT_*` set and the robustness exhibits) has a `kept` row run on the current panel; list exhibits whose latest row predates the panel rebuild (2026-09-23).
3. Every `dropped` or `failed` row has a reason; the narrowing of the reported set (T2, A1, unconditional CS, LP-DiD) is recorded as reporting-level, not as deletion.
4. Any row whose choice was made **after seeing results** is labelled; list them for the paper's robustness section.
5. Count specifications per outcome (the multiple-testing family size) and print it next to the correction used.
6. Report: unregistered rows, stale exhibits, unlabelled post-hoc choices, family sizes. Read-only; fixes are the user's.

## `show <id>`

Print every ledger row and decision entry for an A-id or D-id, with the scripts and output files involved.
