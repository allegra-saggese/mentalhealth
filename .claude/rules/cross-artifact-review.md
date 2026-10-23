---
description: Paper and code cross-artifact review. When /review-paper runs, auto-invoke /review-py on the referenced scripts and /audit-reproducibility on the pair; surface the findings alongside the paper review.
paths: ["paper/**/*.tex", "manuscript/**/*.tex", "master_supporting_docs/**/*.tex", "*.tex"]
---

# Cross-Artifact Review Protocol

A paper is not an island. Its claims depend on the code that produced them; reviewing the paper without the code reviews half the artifact.

## The dependency graph

```
manuscript.tex    --cites-->  Table 2 (booktabs fragment from script4n)
Table 2           --from-->   Data/output/tables/<date>_twfe_results.csv
results CSV       --by-->     script4a-design-estimates.py
script4a          --uses-->   script4_treatment.py (definitions) + <date>_panel.csv
panel             --by-->     script1b-generate-panel.py  <--  script0a..0g  <--  Data/raw/
```

A bug in `script1b` or in a definition in `script4_treatment.py` invalidates every table downstream. A manuscript review that never opens the code misses this class of error.

## When to apply

When `/review-paper` runs on a manuscript that references analysis outputs. Detection is pattern-based: `\input{...}` of generated table fragments, `%% source: script4x-...py` comments, numeric claims (coefficients, N, p-values) together with the pipeline scripts in the repo, or table labels matching result filenames. A paper with none of these triggers nothing; `--no-cross-artifact` skips it.

## The protocol

1. **Identify referenced scripts** from `\input` paths, source comments, and table/figure labels. Follow the chain upstream through the shared modules to the panel builder.
2. **Auto-invoke `/review-py`** on each identified script, each in a fresh-context subagent. Save to `quality_reports/cross_artifact_<paper>/review_py_<script>.md` (local).
3. **Auto-invoke `/audit-reproducibility <manuscript> Data/output/`** once. Save to `quality_reports/cross_artifact_<paper>/reproducibility.md`.
4. **Surface findings** in the paper review under "Cross-Artifact Findings": scripts reviewed, reproducibility PASS/FAIL (k of m claims within tolerance), code issues by severity, and a table of paper claim, code location, issue.
5. **Escalate:** any tolerance FAIL, or any code CRITICAL affecting a paper claim, becomes CRITICAL in the paper review; unrelated code CRITICALs become separate action items.

In `--peer` mode this runs before the editor's desk review so the editor sees reproducibility evidence first.

## What this rule does not require

- Running the pipeline (`/audit-reproducibility` reads existing outputs).
- Git archaeology; we review the current state.
- Judging code style beyond what bears on whether the results are defensible.

## Cross-references

[`/review-paper`](../skills/review-paper/SKILL.md) · [`/review-py`](../skills/review-py/SKILL.md) (adapted from `review-r`) · [`/audit-reproducibility`](../skills/audit-reproducibility/SKILL.md) · [`replication-protocol.md`](replication-protocol.md)
