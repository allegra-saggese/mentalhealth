# Orchestration Schemas (the review runtime's data contracts)

The skills that fan out to reviewer subagents (`/seven-pass-review`, `/deep-audit`, `/review-paper --adversarial` and `--peer`) used to describe their findings as free-form markdown the synthesizer re-parsed by eye. This file is the **shared structured contract** they reduce over instead — so a synthesizer counts typed objects, a gate predicate is a deterministic check, and the same severity vocabulary means the same thing in every skill.

Reviewers emit a JSON **array** conforming to [`finding-schema.json`](finding-schema.json), written to a `.json` file beside the prose report and validated with `scripts/validate-findings.py` (exit 0 required) **before** the synthesizer reduces. That validator is the runtime; this file is the readable contract around it. See [`.claude/rules/orchestrator-protocol.md`](../rules/orchestrator-protocol.md) for how the fan-out → reduce → judge → loop-until-dry runtime uses them.

---

## 1. `FINDING` — one issue a reviewer raises

Every reviewer subagent that feeds a reducing skill (lens, referee, critic, auditor) emits findings as a **JSON array**
conforming to [`finding-schema.json`](finding-schema.json) — the machine-checked contract.
(`claim-verifier` and `humanize-auditor` keep their own per-claim and detect-only report formats — no reducing skill consumes them.) This sketch is a readable rendering of that schema, not a second contract (it was, once, and
the two drifted — Codex caught a producer ordering the old shape while the validator required
the new one):

```json
[
  {
    "id": "<sha1 of 'file:line:locus'>",
    "file": "main.tex",
    "line": 214,
    "locus": "Table 2, col 3",
    "lens": "methods",
    "severity": "blocker",
    "rule": "conditional PT must be stated when covariates enter the estimand",
    "claim": "Identification rests on conditional PT but the text claims unconditional.",
    "evidence": "p.11 states the assumption unconditionally vs Eq.(4) conditions on X_i.",
    "failing_case": "covariate-dependent treatment timing: Eq.(4) fails while the text's claim stands",
    "suggested_fix": "State the conditional PT assumption explicitly, or drop the covariates.",
    "mechanical": false,
    "confidence": "high"
  }
]
```

- **`id` is computed, never invented**: `python3 scripts/validate-findings.py --id FILE LINE LOCUS`.
  Deterministic and lens-independent, so the same defect found by two lenses dedups to one.
- **Reports are bare arrays** — no `findings:` wrapper.
- **Validate before reducing**: `python3 scripts/validate-findings.py <report>.json` (exit 0
  required). A reviewer whose report does not validate has not reviewed.
- **Who writes the file.** Reviewer agents are read-only — they cannot write the `.json` or run
  the id script. Each reviewer **ends its final response with one fenced `json` block**: the
  array, every required schema field except `id` (`verdict` stays unset — the refute-biased
  verification pass sets it; see §7), with a one-line `Scorecard: N/10` just above the block. The dispatching skill extracts that block, runs
  `python3 scripts/validate-findings.py --fill-ids block.json > <report>.json.tmp && mv <report>.json.tmp <report>.json`
  (it computes every `id` and validates; only a valid array is kept — a failed run must not leave
  an empty `<report>.json` behind), and saves the prose report beside it. A reviewer with nothing
  to report returns `[]` — an explicit empty array, not an omitted block. Each `file:line:locus`
  appears once per array: two issues at one spot are merged or given distinct loci, because
  equal coordinates mean equal ids and a duplicate id fails validation.
- **Quotes are checked against the source.** After `--fill-ids`, the skill runs
  `python3 scripts/validate-findings.py --check-quotes <report>.json`: every double-quoted span of
  12+ characters in `evidence` must occur in the finding's `file`, or in another file the evidence
  names by path (a parity finding quotes the Beamer source while citing the Quarto file). It is
  compared with LaTeX markup, dashes, whitespace and case normalised — except that a quote
  containing a backslash must match the source as written; backticked commands and outputs are not
  checked; a PDF is read through `pdftotext`, and a file that cannot be read as text leaves its
  quotes unchecked rather than failed. A
  miss goes back to its reviewer once, with the checker's output; a finding whose quote still does
  not match is dropped before judging and listed as `quote not in source`. An invented quotation
  is the review error that does the most damage, and it is the one a string match can catch.
- The old prose fields map as §7's table records: `location`→`file`+`line`+`locus`,
  `finding`→`claim`, `recommendation`→`suggested_fix`, `CRITICAL`→`blocker`. The concept
  behind `change_my_mind` lives on inside `failing_case` — state the concrete configuration
  or missing hypothesis, which is also exactly what would reverse the finding.

**Severity is the single cross-skill vocabulary.** Map every skill's local words onto it:

| Local term (agent or skill) | JSON `severity` | Gate reading |
|---|---|---|
| FATAL / desk-reject-worthy / hard-gate failure / Critical / CRITICAL / HIGH-WARN (fabricated citation) | `blocker` | CRITICAL |
| Major Concern / "blocks submission" / Visual-Regression / High / MAJOR / ADDRESSABLE (weighty) | `major` (`blocker` if it blocks the artifact) | MAJOR |
| Minor Concern / Medium / MED / Low / MINOR / ADDRESSABLE (light) / MED-WARN / LOW-WARN (retrieval problems, not defects) | `minor` | MINOR |
| TASTE / polish / style with no rule behind it | `nit` | — (never gates) |

- The old `change_my_mind` field is **gone** — its content lives in `failing_case`, which is required on every finding: the concrete configuration that breaks the claim is also exactly what would reverse the finding.
- `confidence` is for the judge/verifier, not the author — a `low`-confidence CRITICAL is a prime candidate for the hallucination gate (§4).

## 2. `SCORECARD` — a reviewer's aggregate

The dispatching skill builds one scorecard per reviewer — counts from the reviewer's validated array, `score` from its `Scorecard: N/10` line — and the synthesizer stacks them:

```yaml
scorecard:
  lens: methods
  critical: 1         # count of `blocker` findings
  major: 3
  minor: 5
  score: 6            # 0–10, the reviewer's holistic read of its own lens (referees who score 0–100 report composite ÷ 10)
  verdict: REVISE-MAJOR   # SUBMIT | REVISE-MINOR | REVISE-MAJOR | REJECT  (artifact-level lenses)
```

For parity/gate skills (`qa-quarto`), the lens verdict is the hard-gate roll-up: `APPROVED` iff every hard gate passes (zero CRITICAL), else `BLOCKED`.

## 3. Gate predicates (how `reduce` decides)

The synthesizer's verdict is a **deterministic function of the typed findings**, not a re-judgment:

| Predicate | Rule |
|---|---|
| **PASS / APPROVED** | `sum(CRITICAL) == 0` across all lenses (and, for gate skills, every hard gate true) |
| **REVISE** | `sum(CRITICAL) == 0` and `sum(MAJOR) > 0` |
| **BLOCK / FAIL** | `sum(CRITICAL) > 0` |
| **converged (loop-until-dry)** | **2 consecutive** rounds each produce **0 new** blocker/major findings (deduped by the deterministic `id`), **and** every lens returned a valid findings array in both; a round with a missing lens (no valid array after one re-dispatch) is not dry and does not count ([`orchestrator-protocol.md`](../rules/orchestrator-protocol.md) §4) |

"New" is measured against the running set of already-seen findings: within a round by **`id`** (`sha1(file:line:locus)` — exact, lens-independent), and **across rounds by `file` + `locus`**, because a fixer's edit shifts line numbers and so gives an unfixed defect below it a fresh `id`. A finding whose `file` + `locus` was already seen is not new — so a critic re-flagging an unfixed issue does not count as progress, and a fixer silently re-introducing one does not hide.

## 4. Post-judge hallucination gate (the synthesizer cannot invent CRITICALs)

A synthesizer/editor/judge reduces lens findings — it must not **introduce** a blocking claim no lens raised. (The audit found `editor.md` could desk-reject on a reason neither referee gave.) The gate:

1. After the judge produces its verdict, diff its CRITICAL/desk-reject reasons against the union of lens `findings`.
2. Any CRITICAL the judge introduced that is **not traceable** to a lens finding is a **candidate hallucination**.
3. Re-verify each candidate in a **fresh context** — spawn `claim-verifier` (a named `Agent` call, never a conversation fork, so it never sees the draft) with the claim + the artifact location it cites, per [`post-flight-verification.md`](../rules/post-flight-verification.md).
   - Verifier confirms (grounded in a quote/location) → keep the CRITICAL; annotate `[JUDGE-ADDED, verified]`.
   - Verifier cannot ground it → **drop it to a flagged note**, tag `[JUDGE-HALLUCINATED]`, and **recompute the verdict** under §3 without it.
4. A judge may always *downgrade* or *de-duplicate* lens findings freely; it may only *introduce* a blocking finding that survives the gate.

This is cheap (it runs only on judge-introduced CRITICALs, usually 0–2) and it is the single most important guard for trusting an autonomous review near a credibility-sensitive artifact.

## 5. `RUN_CONFIG` — the pre-run input contract

A fan-out runtime collects every interactive choice **before** launch, so no subagent stalls waiting on the user mid-run (subagents cannot prompt). See `orchestrator-protocol.md` → "RUN_CONFIG".

```yaml
run_config:
  artifact: path/to/manuscript.tex        # what is being reviewed
  mode: peer                              # default | adversarial | peer | seven-pass | excellence | audit
  journal: QJE                            # --peer: resolved against journal-profiles.md (else null)
  dispositions: [SKEPTIC, MEASUREMENT]    # --peer/--variance: sampled before launch (else null)
  n_referees: 3                           # --variance N (else null)
  peeves: { critical: 2, constructive: 1 }# referee peeve budget (stress doubles critical)
  fresh_context: true                     # re-audit rounds run in a fresh Agent context (never a fork)
  max_rounds: 5                           # loop-until-dry FALLBACK cap (not the primary stop)
  cross_artifact: true                    # auto-invoke /review-py + /audit-reproducibility
  novelty_check: true                     # editor WebSearch probe (Post-Flight-verified)
  spend_cap_tokens: 500000                # warn-and-ask ceiling, not a context limit
```

Gather it, echo it back as the **Pre-Flight Report**, and only then spawn the fleet. Any unresolved required field (e.g. an unknown `journal`) halts before launch — never mid-run.

---

## Cross-references

- [`.claude/rules/orchestrator-protocol.md`](../rules/orchestrator-protocol.md) — the runtime that consumes these schemas.
- [`.claude/references/agent-fleet.md`](agent-fleet.md) — which agent fills which lens, at which model tier.
- [`.claude/rules/post-flight-verification.md`](../rules/post-flight-verification.md) — the fresh-context verifier mechanism the §4 gate reuses.
- [`.claude/rules/summary-parity.md`](../rules/summary-parity.md) — the two-strikes rule the loop reuses for repeatedly-flagged findings.


---

## 7. The FINDING contract is machine-validated (v2.5)

The schema above stopped being prose. It is now
[`finding-schema.json`](finding-schema.json), enforced by
[`scripts/validate-findings.py`](../../scripts/validate-findings.py) with an exit code.

### Reconciling this with the prose schema above (Codex review, PR #140)

§1's readable sketch and this JSON schema are **two serializations of one contract**, and they were
inconsistent when §7 was added. The JSON schema is now **authoritative**; §1 remains as the
human-readable sketch. The mapping:

| §1 prose field | Schema field | Note |
|---|---|---|
| `location` | `file` + `line` + `locus` | split, so dedup can be exact |
| `finding` | `claim` | one sentence, unchanged in meaning |
| `recommendation` | `suggested_fix` | renamed only |
| `severity: CRITICAL \| MAJOR \| MINOR` | `severity: blocker \| major \| minor \| nit` | **`CRITICAL` maps to `blocker`**; `nit` is new |
| — | `rule`, `evidence`, `failing_case`, `mechanical`, `confidence` | new required fields |
| `findings:` wrapper | **bare array** | reports are arrays |

The gate predicate is unchanged in meaning: **`blocker` > 0 → BLOCK**, `major` > 0 → REVISE,
else PASS. Where older text says `CRITICAL`, read `blocker`.

**Reports are ARRAYS of finding objects** — not `{"findings": [...]}` wrappers.

```bash
echo '[]' | python3 scripts/validate-findings.py            # smoke-test the harness first
python3 scripts/validate-findings.py review-report.json     # validate a real report
python3 scripts/validate-findings.py --id paper.tex 42 thm:main
```

### Deterministic finding ids — how dedup stops being fuzzy

```
id = sha1("<file>:<line>:<locus>")
```

The same defect gets the same id from every lens and in every session **while the file is
unchanged** — so dedup across lenses in one round is exact. Between rounds a fixer edits files,
which shifts line numbers and therefore ids; match across rounds on `file` + `locus` (the named
thing — a theorem label, slide title, function) instead (§3). That keeps `loop-until-dry` honest
(a round is dry when every lens reviewed and it produces no finding whose `file` + `locus` is new) and the two-strikes
rule checkable — the same `file` + `locus` surviving rounds N and N+2 escalates to the human
rather than being patched a third time. The validator recomputes every id and rejects any that
does not match its own coordinates, so ids cannot be invented.

### Every finding must cite a rule

`rule` is required. **A finding that cites no documented rule or standard is an opinion**, and
opinions do not gate a commit. This is the single cheapest defence against the measured
tendency of a reviewer told to find gaps to find them in sound work.

### `failing_case` is required

A concrete configuration under which the claim breaks, or the exact missing hypothesis.
*"This could be clearer"* does not validate.

### The `mechanical` whitelist

`mechanical: true` is permitted **only** for fixes that cannot change a result:

- a typo, a broken cross-reference, a malformed citation key
- a formatting or overflow fix
- adding a missing label or docstring
- a rename with no external callers

**Never mechanical:** anything touching an **estimand**, an **assumption**, a
**specification**, an **inference procedure**, a **sample definition**, or **reporting
language**. Those return to the researcher. This is the same boundary as the standing
escalation rule.

### Two-stage, refute-biased

The reviewer pass proposes; a **separate verifier pass, biased to refute**, decides. Only
`verdict: "confirmed"` findings ship. A finding the verifier cannot ground is dropped, not
downgraded to a warning.

### Smoke-test the harness before spending review effort

Run `echo '[]' | python3 scripts/validate-findings.py` once per environment. A review run that
measures everything and then cannot write a valid report has wasted the whole pass.

### Per-lens evidence burdens

A generic "provide evidence" is too weak. Each lens owes a specific proof:

| Lens | The burden |
|---|---|
| `identification` | name the assumption and a design under which it fails |
| `estimation` | name the estimator and the exact step that diverges from its definition |
| `inference` | name the variance channel — clustering level, bootstrap, degrees of freedom |
| `numeric-claim` | the manuscript value, the output value, and the tolerance |
| `reproducibility` | the command run and its output |
| `citation` | the quoted passage that does or does not support the claim |
| `measurement` | the construct, the proxy, and the gap between them |
| `parity` | the two artifacts and the specific element that differs |
| `code-quality` | file:line plus the input that triggers the defect |

### "Does NOT count" — suppress known false alarms

Apply before verification:

- A **documented, defensible alternative** is `EXPLAINED`, not a defect. A different-but-named
  choice (a clustering small-sample correction; a bandwidth-selection rule; an MC seed) is
  recorded, never "fixed".
- A **deliberate simplification for teaching** is not a substantive error.
- **Prose taste** is not a finding unless it changes meaning.
- A **stale claim already flagged** by `claim-reconcile` is one finding, not one per surface.
- Anything on the **HELD list** is recorded, not acted on.
