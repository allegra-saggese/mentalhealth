# Defect Library — what to seed

Seeding a *realistic* defect is the hard part. A typo is not a test. These are drawn from
defect classes that have actually survived review in empirical work.

Each entry: what to change, and **what a competent checker should say**.

---

## Empirical analysis code (R / Stata / Python)

| # | Seed | Expected detection |
|---|---|---|
| A1 | Flip a sign in a treatment-effect assembly | estimate changes sign; a reconciliation or fixture check fails |
| A2 | Off-by-one in a lag window (`t-1` as the first lag vs `t`) | the fitted series shifts one period; the first-lag coefficient moves |
| A3 | Cluster at the wrong level (unit instead of treatment-assignment) | SEs shrink materially; clustering assertion fails |
| A4 | Drop rows silently in a merge (`inner` where `left` was intended) | N falls; a post-merge count assertion fails |
| A5 | Remove `set.seed()` / `set seed` | results stop reproducing bit-for-bit |
| A6 | Reorder factor levels so the reference category changes | coefficients re-based; contrast labels no longer match text |
| A7 | Ignore survey/sampling weights in one specification | point estimate diverges from the weighted table |
| A8 | Widen a comparison tolerance by 10× | a previously-failing cell now passes — **the check should flag the tolerance change itself** |
| A9 | Replace a computed constant with a hardcoded literal | value no longer tracks its input; lineage check fails |
| A10 | Return the input unchanged from a transformation function | downstream identical to upstream; substantiveness check fails |

## Manuscripts

| # | Seed | Expected detection |
|---|---|---|
| M1 | Change a coefficient in the text so it no longer matches its table | numeric-provenance check fails on that claim |
| M2 | Cite a real paper for a claim it does not make | citation-fidelity / CoVe flags unsupported attribution |
| M3 | Invent a plausible-looking reference | fabricated-citation check fails |
| M4 | Upgrade hedged language to definitive ("suggests" → "demonstrates") | reporting-language escalation flagged |
| M5 | State a pointwise result as uniform | overclaim: headline exceeds what is proved |
| M6 | Change N in the abstract but not in the table | cross-artifact consistency fails |
| M7 | Drop a required assumption from a theorem statement while keeping the proof | proof gap: conclusion no longer follows |
| M8 | Swap an assumption's conditional form for its unconditional form in a measurement-model writeup | domain review flags assumption mismatch |

## Replication packages

| # | Seed | Expected detection |
|---|---|---|
| R1 | Remove a package from the environment lockfile | cold-run install fails |
| R2 | Point a script at an absolute local path | portability check fails |
| R3 | Delete one output a table depends on | wiring check fails |
| R4 | Leave a raw restricted-data file in the deposit | disclosure screen flags it |
| R5 | Break the master script's completion sentinel | run appears to succeed while incomplete |

## Slides / teaching

| # | Seed | Expected detection |
|---|---|---|
| S1 | Overflow a frame past the text block | visual audit flags overflow |
| S2 | Use inconsistent notation for the same object across two slides | notation-consistency check fails |
| S3 | Mis-state an assumption in a definition box | domain review flags it |

## Infrastructure / the template itself

| # | Seed | Expected detection |
|---|---|---|
| T1 | Name a superseded model as current | model-currency gate fails |
| T2 | Claim a skill count that disagrees with disk | surface-sync fails |
| T3 | Invoke a tool in a skill body without declaring it in `allowed-tools` | skill-integrity parity check fails |
| T4 | Add an unfalsifiable superlative ("the only workflow that…") | product-claim check fails |
| T5 | Add a link to a file that does not exist | link check fails |

---

## Choosing N

- **Smoke qualification:** 3 seeds + 1 control. Enough to catch a dead checker.
- **Standard:** 6–10 seeds across ≥3 classes + 1 control, 2 replicates each.
- **Before a submission-grade claim:** every class the check claims to cover, ≥2 replicates,
  plus a clean control per class.

Report what you did **not** seed. A qualification that covers three classes licenses claims
about three classes.
