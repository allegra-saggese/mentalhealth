# Provenance

<!--
  Copy to quality_reports/PROVENANCE.md and fill in. This is the answer to "the numbers match — match WHAT?"
  Protocol: .claude/references/provenance-and-ground-truth.md
-->

**Status:** [draft | frozen for release X.Y.Z]
**Last verified:** YYYY-MM-DD

## Reference roles

Name every source of truth, its ROLE, and the exact commit. A version string is not enough —
versions get re-tagged, commits do not.

| Reference | Role | Observed ref |
| --- | --- | --- |
| `owner/canonical-package` | primary statistical and API oracle | HEAD `<sha>`; version X.Y.Z |
| `owner/second-implementation` | deeper-test source | HEAD `<sha>` |
| `owner/legacy` | legacy-behaviour source | HEAD `<sha>` |
| `owner/published-paper` | empirical acceptance suite | HEAD `<sha>` |
| `owner/engineering-ref` | engineering reference (style/architecture only) | HEAD `<sha>` |

**Precedence when references disagree — declare this BEFORE measuring:**

> `<primary>` governs. Where `<secondary>` raises and `<primary>` warns, `<primary>` governs.

## Licence notes

Record each reference's licence, including honest ambiguity.

- `<ref>` reports `License: <X>` in its metadata.
- *(Example of ambiguity worth recording: classifiers say Apache while `setup.py` says MIT —
  treat MIT as binding until clarified.)*

## Clean-room boundary

- Reference code may be **read** to understand public behaviour, tests, architecture, and style.
- **Do not copy implementation code** unless an entry below records: source path, commit,
  licence, copied/adapted lines, justification, and a licence-compatible decision.
- **Generated numerical outputs, manifests, behavioural descriptions, and public docs may be
  used as test oracles.**
- Engineering-reference *patterns* may be adopted at the design level; code snippets are
  subject to the same rules.

> Licence compatibility is a gate, not a footnote. Importing copyleft material into a
> permissively-licensed repo is a violation regardless of how useful it is.

## Artifact ledger

| Artifact class | Source allowed | Copy policy |
| --- | --- | --- |
| Contract documents | original synthesis from observed behaviour | owned by this repo |
| Reference locks | commit/version metadata from public repos | owned by this repo |
| Fixture inputs | generated here | owned; generator logged |
| Reference expected outputs | generated from the pinned oracle | allowed as a test oracle |
| Implementation code | new, written here | no copying unless an entry below records it |

## Tolerance registry

| Comparison | Tolerance | Why |
| --- | --- | --- |
| `<estimate>` vs `<oracle>` | `1e-8` relative | analytic identity; anything larger is a defect |
| `<bootstrap SE>` vs `<oracle>` | `5e-2` relative | bootstrap-SE sampling noise is ≈ SE/√(2(B−1)) ≈ 2.2% at B = 999; doubled for headroom. (A `1e-3` tolerance is defensible only with matched seeds/RNG streams on both sides.) |

Fix tolerances **before** comparing. A tolerance loosened after a failed comparison converts
evidence into decoration; if it must be loosened, record it here as an approved divergence.

> **R note — the `all.equal` scale trap.** `all.equal` silently degrades to an ABSOLUTE
> comparison when the target's magnitude is below `tolerance`, so small-SE cells can pass under
> unbounded relative error. Pass `scale = max(mean(abs(target)), .Machine$double.eps)`.

## Divergence ledger

Not every difference is a bug. See `inst/spec/divergence-kinds.csv` (or the table below).

| id | kind | summary |
| --- | --- | --- |
| `F001-DIV001` | `behavioral-default` | this port defaults to X; the reference defaults to Y — recorded, not fixed |
| `F002-DIV001` | `language-surface` | construct has no analogue in the target language |
| `F003-DIV001` | `upstream-conflict` | references disagree; `<primary>` governs |

Kinds: `behavioral-default` · `language-surface` · `api-surface` · `internal-api` ·
`test-placement` · `upstream-conflict`.

## Future ledger entries

Append here whenever the clean-room boundary is crossed with permission.
