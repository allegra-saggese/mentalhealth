# Executor Contract: [Task Title]

**Date:** [YYYY-MM-DD]
**Dispatched to:** [agent / model tier] — see [`model-routing.md`](../.claude/rules/model-routing.md)
**Status:** DRAFT | DISPATCHED | RETURNED | ACCEPTED

> Give an executor the **goal and the acceptance bar — never the implementation**
> ([`research-agent-laws.md`](../.claude/references/research-agent-laws.md) law 7). A prescribed
> mechanism is a hypothesis wearing the clothes of an instruction.

---

## Goal

[One or two sentences describing the state of the world when this is done. An outcome, not a
procedure: *"every exported function has a regression test that fails when its output changes"*,
not *"add test files under `tests/`"*.]

---

## Acceptance bar

[What evidence settles it: the command that is run, the artifact it writes, and the value that
counts as a pass. If a human has to look and judge, say who and against what.]

- [ ] [Evidence 1 — command → artifact → passing condition]
- [ ] [Evidence 2 — command → artifact → passing condition]

**Not evidence:** the executor's summary of its own work; a count read off a terminal rather than
computed (law 18); a gate that would pass rather than one that was run.

---

## Owned paths

Exact paths this executor may create or modify. Anything not listed belongs to someone else — in
a parallel wave, to another executor running right now.

| Path | Create / Edit | Note |
|------|---------------|------|
| `[path/to/file]` | edit | [why this file is in scope] |
| `[path/to/new-file]` | create | [what it is for] |

---

## Gates this work must pass

```bash
[python3 scripts/quality_score.py <files>]          # or the standing gate for the surface being touched
```

**Done** means the gate was **run** and is green, and the work is committed and pushed **where
the executor was authorized to commit** — reported in that order and not before (law 19). A gate
left for a pre-commit hook to discover is a gate that was not run. Committing is itself gated: if
this repo requires explicit sign-off to commit ([`/commit`](../.claude/skills/commit/SKILL.md)),
a delegated executor's *Done* is **gate-green and handed back with evidence**, and the commit is
a separate authorized step — never something the executor does on its own to satisfy this line.

---

## Output contract

Return exactly:

1. **Structured result** — [`files_changed`, `evidence`, `open_issues`, … name the fields].
   Every number in it cites the command that produced it.
2. **Banked artifact** — the raw output (log, table, findings JSON) written to
   `[quality_reports/…]`, so the record outlives the context that made it.

---

## Mechanisms offered (refusable hypotheses)

These are guesses about *how*, not instructions. An executor that measures a better route takes
it and says so in the return.

- **Hypothesis A:** [mechanism] — [why it might be right, and what would falsify it]
- **Hypothesis B:** [alternative]

**Refusal is a valid return.** *"The bar was already met — here is the measurement"* ends the
task successfully and is worth more than a change that was not needed.

---

## What NOT to touch

- [Paths owned by other executors in this wave]
- [Surfaces whose change would need its own review — anything that moves a reported number]
- [Blessed baselines and pins: never re-bless in the commit that moves them (law 3)]

---

## Return checklist (executor fills in)

- [ ] Acceptance-bar evidence produced, with the deriving command stated
- [ ] Gate run and green; committed and pushed **only where authorized** — otherwise handed back gate-green with evidence (see *Done*, above)
- [ ] Only owned paths touched
- [ ] Refused or replaced mechanisms named, with the measurement that justified it

---

## Filled example (synthetic)

**Goal.** The daily ingest produces one row per (site, day) for every site in the registry, and a
consumer can distinguish a day with no observations from a day that was never ingested.

**Acceptance bar.**

- [ ] `python3 pipeline/audit_coverage.py --since 2020-01-01` exits 0 and writes
      `outputs/coverage_audit.csv`; the audit's final line reports `missing_days: 0`.
- [ ] The same command on the pre-change commit reports a non-zero `missing_days` — the
      detector is shown to fire before it is trusted to pass.

**Owned paths.** `pipeline/ingest.py` (edit), `pipeline/audit_coverage.py` (create),
`tests/test_coverage.py` (create).

**Gates.** `pytest tests/ -q` and the repo's standing checker suite, both green before reporting.

**Output contract.** Structured return with `files_changed`, `evidence` (the two command lines and
their final summary lines, verbatim), `open_issues`; `outputs/coverage_audit.csv` banked in the
run directory.

**Mechanisms (refusable).** *Hypothesis A:* a left join against a generated calendar table is the
cheapest way to make absence explicit. *Hypothesis B:* an ingest-time sentinel row. If the loader
already emits an explicit no-observation marker, neither is needed — say so and stop.

**What NOT to touch.** The published schema for downstream consumers, the retention policy, and
any file under `outputs/` other than the new audit CSV.

---

## Cross-references

- [`research-agent-laws.md`](../.claude/references/research-agent-laws.md) — laws 7, 8, 18, 19
- [`orchestrator-protocol.md`](../.claude/rules/orchestrator-protocol.md) — the fan-out runtime this contract is dispatched into
- [`screening-rubric.md`](screening-rubric.md) — the screening counterpart, when the executor's job is to triage rather than to build
- [`requirements-spec.md`](requirements-spec.md) — for the upstream question of *what is required* before anything is dispatched
