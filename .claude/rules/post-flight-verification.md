---
paths:
  - ".claude/skills/lit-review/SKILL.md"
  - ".claude/skills/research-ideation/SKILL.md"
  - ".claude/skills/respond-to-referees/SKILL.md"
  - ".claude/skills/review-paper/SKILL.md"
  - ".claude/skills/interview-me/SKILL.md"
alwaysApply: false
---

# Post-Flight Verification (anti-hallucination)

Symmetric partner to **Pre-Flight Reports** (skill-level `## Phase 0`, e.g. `/data-analysis`; the fan-out form is the `RUN_CONFIG` echo in `.claude/rules/orchestrator-protocol.md` § "RUN_CONFIG: collect interactivity *before* launch"). Where Pre-Flight proves inputs were read *before* work, Post-Flight proves the output's factual claims hold *after* drafting — before the skill returns to the user.

**Adapted from:** Dhuliawala et al. 2023, "Chain-of-Verification Reduces Hallucination in Large Language Models" ([arXiv:2309.11495](https://arxiv.org/abs/2309.11495)). The **independence trick** — answer verification questions in a context that does not contain the original draft — is architecturally enforced here by running `claim-verifier` as its own `Agent` call, which starts in a fresh context. It literally cannot self-confirm; it has never seen the draft. (A *conversation* fork — `/fork`, a fork-mode subtask — inherits the conversation and would defeat this; never use one for verification.)

## When this rule applies

Any skill whose output contains **factual claims that can be independently verified** against a source:

| Skill | High-risk claim types |
|-------|----------------------|
| `/lit-review` | Citations (paper exists, authors, year, venue); paraphrased claims ("Smith 2019 finds X") |
| `/research-ideation` | "Has anyone tested this?" negative-literature claims; dataset structure claims; estimator feasibility claims |
| `/respond-to-referees` | "We added X on page Y" assertions about actual revisions |
| `/review-paper` (`--peer` mode) | Novelty-probe claims from WebSearch ("this paper's contribution is novel" / "similar to Jones 2020") |
| `/interview-me` | Papers referenced in the research spec (if any cited) |

Does **not** apply to mechanical skills (`/commit`, table writers, `py_compile`) — they produce compiled output verified by external processes (the compiler), not factual text.

## The 4-step CoVe protocol

### Step 1 — Draft (the skill's normal output generation)

Produce the response as usual. Do not short-circuit the Post-Flight check even for "obviously correct" drafts.

### Step 2 — Extract claims

From the draft, identify every assertion of the form:

- **Citation claims:** "Author (Year) shows X."
- **Existence claims:** "A dataset called X contains Y fields."
- **Numerical facts:** "N = 10,000" / "coefficient = 0.42" / "p < 0.01"
- **Named entities:** researcher names, paper titles, venues, package names
- **Negative literature claims:** "No prior work studies X."

Skip:
- **Opinions** ("this is a promising direction") — not verifiable
- **Suggestions** ("the user could try a log specification") — forward-looking
- **Definitions Claude introduces itself** ("let τ denote the treatment effect")

### Step 3 — Generate verification questions

For each extracted claim, write one specific, answerable question whose answer can confirm or refute the claim from the source material alone. Good questions name the source explicitly. Bad questions are open-ended.

| Bad question | Good question |
|-------------|--------------|
| "Is Author (2021) about calibration?" | "In Author (2021), what is the exact estimator name given in Section 4?" |
| "Does the estimator need a regularity condition?" | "Does Author (2021) Section 4 Assumption 2 require a monotone response, or merely a continuous one?" |

### Step 4 — Answer in fresh context, then reconcile

Spawn `claim-verifier` via the `Agent` tool with `subagent_type=claim-verifier` — a named subagent starts in a **fresh context**. Do **not** use a conversation fork (`/fork`, a fork-mode subtask): a fork inherits the conversation, draft included, which defeats the independence CoVe depends on. Hand it: claims, verification questions, source material pointers. **Do not include the draft.**

Receive back a verification report. Three outcomes:

- **PASS** (no claim contradicted, no retrieval failure): return the draft; any claim whose source was genuinely inaccessible (LOW-WARN, `cannot-verify`) carries an inline uncertainty flag so the user knows to double-check it.
- **PARTIAL** (a transient retrieval failure — MED-WARN — and no contradiction): return the draft with those claims flagged for a re-check.
- **FAIL** (at least one claim contradicts the source, or a citation looks fabricated — HIGH-WARN): **regenerate the affected section** using the verifier's evidence. If regeneration still fails after 2 attempts, return the best draft with discrepancies surfaced as a warning block — do not silently ship a known-wrong claim.

## Output contract

Every skill that applies this rule must include a structured Post-Flight block in its response (can be collapsed by the user; visible on demand):

```markdown
## Post-Flight Verification

**Claims extracted:** N
**Verified independently:** N (fresh-context `claim-verifier` agent)
**Outcome:** PASS | PARTIAL | FAIL → regenerated

### Verified

| ID | Claim | Evidence |
|----|-------|----------|
| C1 | [claim] | [source + loc] |

### Unverifiable (user review recommended)

- **C4** — [reason, e.g., paywalled source]

### Discrepancies (regenerated)

- **C3** — original draft: "N = 10,000"; source shows: "N = 1,000". Corrected in final response.
```

## Fail-closed semantics

- If the verifier agent errors out, returns malformed output, or times out, **do not silently ship the draft.** Surface a block like:
  > "Post-Flight verification failed (verifier error: …). Draft has not been independently checked. Treat the output as provisional."
- This mirrors Pre-Flight's fail-closed contract. Hallucination discipline is most valuable precisely when things are going sideways — that's when silent failures are most expensive.

## Opt-out

`--no-verify` flag skips Post-Flight in `/lit-review`, `/research-ideation`, `/respond-to-referees` and `/interview-me`. Useful for speed-critical iterations or when the user is actively reading the source material themselves. Document the opt-out in each skill's argument hints. Exception: `/review-paper` has no `--no-verify`; its opt-out is `--no-novelty-check`, which skips the novelty probe entirely. If the probe runs, Post-Flight is mandatory.

## Cross-references

- `.claude/agents/claim-verifier.md` — the fresh-context verifier.
- `.claude/rules/review-fencing.md` — the environment side of the same discipline: a fresh context does not fence the checkout the reviewer stands in.
- `.claude/skills/verify-claims/SKILL.md` — user-facing wrapper for ad-hoc verification of any text.
- `.claude/rules/orchestrator-protocol.md` — Pre-Flight (input side): the `RUN_CONFIG` Pre-Flight Report and the skills' `## Phase 0` Pre-Flight Reports.
- `.claude/rules/cross-artifact-review.md` — pattern-based; Post-Flight is draft-based.
- `.claude/rules/summary-parity.md` — rule against enumerative summaries drifting from their bodies; Post-Flight is the factual equivalent for draft content.
