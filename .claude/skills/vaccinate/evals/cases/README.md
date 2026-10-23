# Eval cases

One file per case. The prompt is everything between `## Prompt` and `## Assert`; assertions
are the `- ` lines from `## Assert` **to the next heading** — so keep notes either above
`## Prompt`, or under their own heading after the Assert block.

```markdown
## Prompt
The exact user message to send, verbatim. Write what a real user would type,
not a well-specified instruction — the point is to test the skill's triggering
and its instructions, not your prompt-writing.

## Assert
- a literal substring that must appear in a correct answer
- another one
```

**Assertions are literal substrings, matched case-insensitively.** That is crude on
purpose: a grader that is itself a model introduces the failure mode being measured. If a
case cannot be graded by substring, it probably is not a case — it is a judgement, and
judgements belong to you.

## Writing cases that measure something

- **Realistic prompts.** "review this" beats "using the seven-pass protocol, review this",
  because the second tests your memory of the skill rather than the skill.
- **Assertions on substance, not shape.** Assert that a specific defect is named, not that
  the answer contains "## Findings".
- **Include should-NOT-trigger cases.** A skill that fires on everything is as broken as
  one that never fires. Name them `nottrigger-*.md` and assert the skill's signature output
  is *absent*.
- **Two or three replicates.** A single run does not separate a real improvement from
  sampling noise.

## Priority order

Rank by **cost of being wrong**, not by frequency of use:
`/review-paper --peer` → `claim-verifier` → `/audit-reproducibility` → `/challenge`.
