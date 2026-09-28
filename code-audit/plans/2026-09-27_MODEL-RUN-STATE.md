# Model run state — what has been run, on which panel
**2026-09-27** · supersedes the run-state sections of `2026-09-18_ANALYSIS-STATUS.md`

---

## THE CONTROLLING FACT

The panel was **rebuilt 2026-09-23 17:35** (`Data/merged/2026-09-23_panel.csv`) to fix the
CHR release-year vs data-year misalignment: 10 control variables were replaced with
correctly-dated annual sources (SAIPE, BLS LAUS, SAHIE, Census PEP). See
`2026-09-21_CHR-YEAR-MISALIGNMENT.md`.

**Any output dated 2026-09-18 was produced on the pre-fix panel and is stale.**

| Script | Output folder | Last run | Panel | State |
|---|---|---|---|---|
| `script4b-covariate-sequence.py` | `tables/script4b/` | 2026-09-23 22:11 | 09-23 | **CURRENT** |
| `script4c-covariate-stability.py` | `tables/script4c/` | 2026-09-27 | 09-23 | **CURRENT** |
| `script4a-twfe-eventstudy.py` | `tables/script4a/` | 2026-09-18 10:14 | 09-17 | **STALE** |
| `script4d-power-balance.py` | `tables/script4a/` | 2026-09-18 11:05 | 09-17 | **STALE** |
| `script4f-core-models.py` | `tables/script4f/` | 2026-09-18 10:08 | 09-17 | **STALE** |
| `script4i-cs-covariates.py` | `tables/script4f/` | 2026-09-18 12:45 | 09-17 | **STALE** |
| `script4j-control-diagnostics.py` | `tables/script4a/` | 2026-09-18 12:42 | 09-17 | **STALE** |
| `script4e` / `4h` / `4g` (LaTeX) | `tables/script4a/latex/` | 2026-09-18 | 09-17 | **STALE** |
| `script4k-variable-register.py` | `output/codebooks/` | 2026-09-22 16:40 | 09-22 | **STALE** ¹ |

¹ Ran before the rebuild, and its `source` / `data_years_TRUE` / `year_lag` fields are now
wrong for the 10 replaced variables regardless of date.

**Nothing in the current LaTeX/beamer deck reflects the corrected panel.**

---

## WHAT IS ACTUALLY ESTIMATED RIGHT NOW

Only the covariate-sequence grid. On the corrected panel we have:

**840 regressions** = 4 treatments × 7 outcomes × 2 FE specs × 15 covariate steps.
This is run **S1**; the stability synthesis is **S2**. Model IDs follow
`2026-09-27_MODEL-NAMING.md` (A = built, B = specified only).

- **Treatments** T1 presence (`tr_e1_lg_bin`), T2 count (`tr_i3_lg_count`),
  T3 per 10k (`tr_i4_lg_p10k`), T4 log(1+count) (`tr_c3_log_lg`).
  T2 and T4 carry `log_pop` as a specification regressor, not a covariate.
  **T6 (conditional on small-CAFO count) is DEFINED but NOT RUN** — its control set is
  still undecided.
- **Outcomes** Poor MH Days, Frequent Mental Distress, Deaths of Despair, Aggravated
  Assault, Assault (all severities), Violence index (partial), NIBRS curated total.
- **FE specs** A1 pooled (state + year), A2 within (county + year).
- **SEs** state-clustered, county-clustered, heteroskedastic — all three recorded.
  County-clustered is the headline.

Zero failures. Max VIF **6.25** across all 840; **2.63** under county FE. Collinearity is
not a binding problem anywhere in the design.

### Key results — see D-13 to D-16 in `CONTROL-DECISION-REGISTER.md`
- 29 of 56 paths (52%) are fragile; 2 sign flips; 15 significance changes.
- A1 pooled is markedly less stable than A2 within (median 27.4% vs 14.6%).
- **Most movement happens after step 8**, where the sample starts falling (median 13.8%
  through step 8 vs 22.8% after). 11 of 32 large movers are stable until then.
- Deaths of Despair is the outlier (median 109%, both sign flips) and is demoted to
  exploratory.
- **Pre-committed spec = step 8** (`COVARIATE_ORDER[:8]`, county+year FE, county-clustered).

---

## NOT BUILT

| | Status |
|---|---|
| Covariate **co-movement map** (step 3 of the agreed workflow) | not built |
| Covariate **pre-trends** on the rebuilt panel | 4j exists, not re-run |
| **T6** control set | undefined |
| **F-tests** between final specs (state FE vs county FE; step 8 vs step 14) | not built |
| **D-15** sample-vs-covariate decomposition | not built |
| Callaway–Sant'Anna on the corrected panel | 4f/4i exist, not re-run |

---

## RE-RUN ORDER (dependency-correct)

1. `script4j` — control diagnostics + covariate pre-trends *(inputs to every choice below)*
2. `script4a` — A1/A2/A3/A6/A7/A18 grid + VIF audit
3. `script4f` → `script4i` — core models + Callaway–Sant'Anna
4. `script4d` — power / balance / coverage *(needs 4a's N and switcher counts)*
5. `script4k` — variable register *(after the source fields are corrected)*
6. `script4e` → `script4h` → `script4g` — LaTeX tables, result tables, slides *(last)*

`script4b` and `script4c` are already current and do not need re-running unless
`COVARIATE_ORDER` changes.

---

## OPEN ITEMS CARRIED FORWARD

- `access_to_healthy_foods_per100k` sits at position 11 in `COVARIATE_ORDER` despite D-05
  deciding to drop it. **Contradiction — must be resolved before any re-run.**
- `%_rural` removal (D-03) still marked "decision pending" in the register though agreed
  in conversation.
- Advisor feedback received but not yet incorporated.
