# Model naming — reconciled
**2026-09-27** · authoritative. Supersedes the A-numbering in `2026-09-17_analysis-registry.md`.

---

## THE RULE

| Prefix | Meaning |
|---|---|
| **A** | **Implemented in the 4-series scripts.** Runs today. Must be re-run on the 2026-09-23 panel. |
| **B** | **Specified only.** No implementation in the 4-series. Not scheduled. |

The old scheme mixed the two, and its numbering gaps (A4/A5 built only in the legacy
`script4-model-test.py`; A18 appended later as a "candidate") made the run set impossible to
read off. A is now a contiguous list of things that exist.

---

## A-SERIES — built, to re-run on the fixed panel

All six are estimated over **T1–T4 × 7 outcomes**, county-clustered SEs headline.

| New | Old | Model | Where | Question |
|---|---|---|---|---|
| **A1** | A1 | Pooled OLS, **state + year FE** | `script4a` ✅ **RUN 09-28** | Benchmark. Between-county contrast. |
| **A2** | A2 | TWFE, **county + year FE** | `script4a` ✅ **RUN 09-28** | **Headline.** Within-county contrast. |
| **A3** | A3 | **Other-animal conditioning** (horse race) | `script4l` ✅ **RUN 09-28** | Is the dairy coefficient dairy, or "any large CAFO"? **Answered: no. See D-19.** |
| **A4** | A6 | Dairy × FSIS interaction | `script4a` ⚠️ **RUN 09-28 — not identified for 5/7 outcomes, D-22** | Concentrated where a slaughterhouse is also present? |
| **A5** | A7 | Event study, ±8/8 around first entry | `script4a` ⚠️ **RUN 09-28 — 3 crime outcomes fail pre-trends, D-23** | Does the outcome move at entry, and were pre-trends flat? |
| **A6** | A18 | Callaway–Sant'Anna staggered DiD | `script4a` ✅ **RUN 09-28 — headline does not survive, D-21** | TWFE-robust ATT(g,t) under staggered adoption. |

**Old A4 (size-threshold) and old A5 (functional-form) are retired, not renamed.** Both were
built only in the legacy `script4-model-test.py`, and both are now subsumed: running T1–T4
through A1/A2 *is* the functional-form sensitivity test, and the 500+ size definition is
fixed by D-treatment decisions rather than tested.

### Supporting runs (not models — diagnostics)

| ID | Script | Output |
|---|---|---|
| **S1** | `script4b-covariate-sequence.py` | 840 nested-covariate regressions. **CURRENT** (09-23). |
| **S2** | `script4c-covariate-stability.py` | 56-path stability synthesis. **CURRENT** (09-27). |
| **S3** | `script4j-control-diagnostics.py` | Control roster, covariate pre-trends, headline VIF. Stale. |
| **S4** | `script4d-power-balance.py` | Power, balance, coverage, MDE. Stale. |

---

## A3 — EXPANDED to host T6

**Decided 2026-09-27.** T6 (log(1+large dairy) conditional on other CAFO activity) is not a
separate model; it is the **treatment that A3 estimates**. A3 becomes a *set* of
specifications rather than one regression.

### The conditioning pool

`cafo_*_large` counts, 2026-09-23 panel:

| Type | Coverage | >0 in | mean | within SD share |
|---|---|---|---|---|
| dairy (treatment) | 84.6% | 18.6% | 1.15 | 0.133 |
| **cattle** | 91.3% | 70.0% | 9.80 | 0.242 |
| **hogs** | 89.8% | 30.1% | 3.98 | 0.173 |
| **chickens** | 90.5% | 6.5% | 0.13 | 0.338 |
| ~~beef~~ | 91.3% | 38.8% | 2.01 | 0.266 |

**`beef` is EXCLUDED from the pool.** It is a strict subset of `cattle`: `beef > 0 &
cattle == 0` occurs in **0.0%** of rows, while `cattle > 0 & beef == 0` occurs in 31.2%.
Entering both double-counts the same operations and induces avoidable collinearity. `cattle`
is kept as the broader measure. *(If the substantive interest is specifically beef feedlots
rather than all cattle, this reverses — flag for review.)*

### The grid

Three conditioning types → **2³ = 8 subsets**, from none (= A2) to all three:

```
{}  {cattle}  {hogs}  {chickens}
{cattle,hogs}  {cattle,chickens}  {hogs,chickens}  {cattle,hogs,chickens}
```

× 4 treatments × 7 outcomes × 2 FE specs = **448 regressions**.

The empty set reproduces A2 exactly and is the reference against which the dairy coefficient
is read. Covariates held at the **pre-committed step-8 set** (D-16) throughout, so the only
thing varying across the 8 is which other animals are conditioned on.

**Functional form must match the treatment.** T2/T4 condition on counts (and log(1+count)
respectively); T1 conditions on presence; T3 on per-10k. Mixing a count treatment with a
binary control changes what the coefficient means. The old A3 used binary `any_large_*`
against a binary treatment, which was internally consistent but does not generalise to T2–T4.

### Interpretation caveat, to state in the paper
Other-animal CAFO counts are **not clearly pre-determined** with respect to dairy. If large
dairy expansion displaces or attracts other livestock operations, these are post-treatment
and conditioning on them is a bad control. A3 is therefore a **robustness probe on what the
dairy coefficient contains**, not a better-identified specification than A2.

---

## B-SERIES — specified, not built, not scheduled

| New | Old | Model |
|---|---|---|
| B1 | A8 | Ridge, pooled (raw) |
| B2 | A9 | Ridge, within-transformed |
| B3 | A10 | **Control specification curve** (all subsets — the thing S1 is *not*) |
| B4 | A11 | Covariate importance ranking, Ridge + RF |
| B5 | A12 | Random Forest variable importance, 2010–2020 |
| B6 | A13 | Double Lasso / DoubleML |
| B7 | A14 | TWFE re-run, 2010–2020 window + interactions |
| B8 | A15 | Causal vs ML comparison table |
| B9 | A16 | Vibration of effects |
| B10 | A17 | Pass/fail matrix |
| B11 | A19 | Heterogeneous treatment effects |
| B12 | A20 | Permutation / placebo inference |
| B13 | A21 | Window justification — **flagged blocking for the deck** |
| B14 | A22 | Control-group documentation for the event study — **flagged blocking for the deck** |

B13 and B14 carry a blocking flag from 2026-09-17. They are not model runs — they are
write-ups the event study needs before it can be presented. Deferred, not resolved.
