# Agent Fleet Manifest

The 18 specialist subagents, what each is for, the model tier it runs at ([`model-routing.md`](../rules/model-routing.md)), and which skill dispatches it. This makes the fleet legible — when a fan-out skill picks a lens, this is the roster it picks from. Reviewers are **read-only** (they report `FINDING`s per [`orchestration-schemas.md`](orchestration-schemas.md)); only the fixer edits files.

> **Keep this in sync** with `.claude/agents/*.md` frontmatter (`model:` / `effort:`) and with `model-routing.md`. The surface-sync gate counts agents; it does not yet diff this table row-for-row, so update it when you add or retier an agent. (It is a `references/` file, so it is not itself counted as an agent.)

## High-judgment tier — Opus tier

A wrong "looks fine" from one of these is expensive (a desk-reject, a hallucinated citation, a biased estimator shipped). Never demote to save cost ([`model-routing.md`](../rules/model-routing.md) anti-pattern). The four gate agents pin `effort: high`, one level above the Opus default; the rest run at `medium` (the Opus default — see `model-routing.md` § effort).

| Agent | Effort | Role | Read/Write | Disposition-aware | Dispatched by |
|---|---|---|---|---|---|
| `editor` | high | Desk review, referee selection, editorial synthesis | read-only | selects referee dispositions | `/review-paper --peer` / `--variance` |
| `domain-referee` | high | Substance referee (contribution, positioning, external validity) | read-only | yes (6-way taxonomy) | `/review-paper --peer` |
| `methods-referee` | high | Methodology referee (paper-type-aware identification/inference) | read-only | yes (6-way taxonomy) | `/review-paper --peer` |
| `claim-verifier` | high | Fresh-context CoVe verifier (citations, numbers, novelty) | read-only | no | `/verify-claims`, post-flight in `/lit-review` · `/research-ideation` · `/respond-to-referees` · `/review-paper --peer`; hallucination gate |
| `domain-reviewer` | medium | Field-specific substance review of a manuscript section (5 lenses; **template — customize**) | read-only | no | `/seven-pass-review` (methods lens) |
| `verifier` | medium | End-to-end script-run and output verification gate | read-only | no | `/commit` |

## Review / critique tier — Sonnet tier, effort: high

| Agent | Role | Read/Write | Dispatched by |
|---|---|---|---|
| `py-reviewer` | Python code quality, reproducibility, data-handling hygiene | read-only | `/review-py`, `/data-analysis` |
| `proofreader` | Grammar, typos, overflow, terminology | read-only | `/seven-pass-review` (prose lens) |
| `humanize-auditor` | AI-voice tell detection (10 categories) | read-only | `/humanize` |

## Mechanical / voting tier — Haiku tier (the current Haiku takes no effort setting; the `effort: low` pin applies to a successor that does)

| Agent | Role | Read/Write | Dispatched by |
|---|---|---|---|
| `promote-memory-council` | Five-critic vote on `[LEARN]` promotion (generality/staleness/redundancy/evidence/format) | read-only | `/promote-memory` |

## The referee disposition taxonomy

Only the referees (`domain-referee`, `methods-referee`) and the `editor` that assigns them are disposition-aware. The 6-way taxonomy: **STRUCTURAL · CREDIBILITY · MEASUREMENT · POLICY · THEORY · SKEPTIC**. `--peer` samples 2; `--variance N` samples N (with replacement, ≥1 SKEPTIC when N≥3); `--stress` forces SKEPTIC×2. Dispositions are fixed in the `RUN_CONFIG` before launch (referees are forked and cannot be re-prompted mid-run).

## Cross-references

- [`.claude/rules/model-routing.md`](../rules/model-routing.md) — the tiering rationale + the do-not-demote anti-pattern.
- [`.claude/rules/orchestrator-protocol.md`](../rules/orchestrator-protocol.md) — how the fleet is fanned out, reduced, and judged.
- [`.claude/references/orchestration-schemas.md`](orchestration-schemas.md) — the `FINDING`/`SCORECARD` shape every reviewer returns.
