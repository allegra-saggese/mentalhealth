# Skill evals — the gap between *verified* and *measured*

The mechanical checks (`scripts/check-links.py`, `tests/test_precommit.sh`) prove the repo is **internally consistent and
currently true**. They cannot prove that a skill's instructions actually produce good output.
That is a different claim and it needs a different instrument.

**Two things to measure, separately.** Seeing a skill trigger tells you Claude *found* it, not
that it *did what you intended*:

| Question | Instrument |
|---|---|
| Does Claude invoke this skill on the prompts it should — and not on the ones it shouldn't? | description-trigger evals |
| When it does run, is the output what you wanted? | output-quality evals |

## The method: a baseline comparison

Collect a few realistic prompts. Run each in a **fresh session** with the skill available, and
again with it disabled. Compare. A fresh session matters because leftover context from
authoring the skill masks gaps in the written instructions — you will believe the skill says
something it only implied.

If the with-skill run is not better, the skill is costing context for nothing.

## Running them

**The shipped harness is primary.** From a normal shell (NOT inside a Claude Code session —
nested headless calls hang), at the repo root:

```bash
./scripts/run-skill-eval.sh <skill> .claude/skills/<skill>/evals/cases [--replicates N]
```

It runs the behavioral manipulation check first (each arm must *prove* skill access by
retrieving a marker phrase — set per skill in `evals/marker.txt`), then grades each case in
both arms, N=3 replicates by default, with a variance gate on **both** arms and every headless
call sandboxed (no Write/Edit/Bash).

The `skill-creator` plugin is an optional alternative that automates a richer loop:

```
/plugin marketplace add anthropics/claude-plugins-official
/plugin install skill-creator@claude-plugins-official
```

Then ask it to evaluate a skill. It stores cases in `evals/evals.json`, spawns a **subagent per
case** so each starts clean, records tokens and duration, grades assertions into
`grading.json`, and aggregates with-skill vs without-skill into `benchmark.json` — so you can
weigh the pass-rate improvement against the token and time cost.

It also does the two things hardest to do by hand: a **blind A/B between two versions** of a
skill, so you can confirm an edit is an improvement before committing it, and
**description tuning** — generating should-trigger and should-not-trigger prompts, measuring
the hit rate, and proposing description edits when the skill fires on the wrong requests.

## Which skills to evaluate first

Rank by *cost of being wrong*, not by how often they run:

1. **`/review-paper --peer`** — informs submission decisions.
2. **`claim-verifier`** — HIGH-WARN fails the verification closed.
3. **`/audit-reproducibility`** — gates the replication package.
4. **`/challenge`** — its output becomes a robustness claim in a paper.

## The relationship to `/vaccinate`

They are complementary and neither substitutes for the other:

- **`/vaccinate`** asks *can this checker detect a defect I planted?* — a **recall** question
  about a checker.
- **Evals** ask *does this skill produce better work than not having it?* — a **quality**
  question about an instruction set.

A skill can pass evals and still be a useless reviewer: it produces well-formed, plausible
findings that miss real defects. Only `/vaccinate` catches that. And a checker can vaccinate
cleanly while its skill wrapper triggers on the wrong prompts. Run both.

## The harness is itself a check, and must be qualified

Everything in this repo that clears work has to prove it can detect a failure. **That includes
the eval harness.** Three defects were found in its first version by running it, not by reading
it:

| Defect | Symptom | Fix |
|---|---|---|
| **N=1 replicate** | the same case scored 3/3 in one run and 1/3 ten minutes later — opposite conclusions from sampling noise | default `--replicates 3`, and a **variance gate** that refuses to report when per-case sd > 0.5 |
| **`nottrigger` written backwards** | asserted a substring must be *present*, so it scored 1/1 whatever happened and measured nothing | `nottrigger-*` cases now **invert**: the assertion must be **absent** |
| **single-substring grading** | a correct answer phrased differently scored as a miss | assertions list **alternatives** (`a \| b \| c`); any one counts |

**Assertions are literal substrings on purpose.** A model grading a model reintroduces exactly
the failure being measured. Alternatives make that tractable without importing a judge.

> **Do not record a ledger row from a high-variance run.** The harness exits 3 and says so.
> A number that changes between runs is not a measurement, and writing it down as one is worse
> than having no number — it looks like evidence.

## Cost

Each case costs `2 x replicates` headless runs (with-skill and without-skill), at roughly
60–90 s each. Nine cases at N=3 is ~54 runs, well over an hour. Budget accordingly, and **do
not mechanically generate cases to raise coverage** — 180 cases that measure nothing are worse
than nine that measure something, because the ledger then carries green rows that mean nothing.

## Recording results

Eval results go in the same ledger as qualification runs
(`quality_reports/qualification/LEDGER.md`), with the skill name, the case count, pass rate
with and without, and the token delta. **An eval with no recorded baseline is an anecdote.**

> **Status (2026-08-22).** The harness is qualified (negative control delta 0; positive
> control delta +2; behavioral manipulation check), and the five priority skills have
> recorded results in `quality_reports/qualification/LEDGER.md` — including one composite the
> harness withheld for a noisy baseline arm, and two near-zero deltas recorded as-is. The
> measured benefit concentrates in distinctive doctrine; recall-style cases the baseline
> already answers measure ~0, so the next iteration is procedure-sensitive cases, not more
> recall cases.
