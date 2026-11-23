# Model run state — what has been run, on which panel
**2026-10-17** · supersedes `2026-09-27_MODEL-RUN-STATE.md`. Decisions referenced as D-nn are in
`CONTROL-DECISION-REGISTER.md` (D-31 to D-38 are new).

---

## THE CONTROLLING FACT

The code changed after the last estimation run. **Outputs dated 2026-10-12 were produced before
the missing-as-zero rule (D-34) and the reporting layer (D-32), so every number in them predates
the current `load_panel`.** Current panel on disk: `Data/merged/2026-10-12_panel.csv`
(65,828 rows, 2,879 rural counties, 2000–2023). The panel file itself is unchanged by D-34;
the rule is applied when the panel is loaded.

| Script | Last output found | State against current code |
|---|---|---|
| `script4a-design-estimates.py` | 2026-10-12 (run log) | **STALE** — predates D-34 |
| `script4r-fsis-treatment.py` | 2026-10-12 | **STALE** — predates D-34 |
| `script4b-covariate-sequence.py` | 2026-09-23 | Robustness; run on the 09-23 panel and the earlier step-8 membership (D-31). Not re-run. |
| `script4c-covariate-stability.py` | 2026-09-27 | Same as 4b |
| `script4s-species-overlap.py` | 2026-10-17 | **CURRENT** (new; descriptive; census years 2002–2022) |
| `qa-scripts/pull_2022_cattle_totals.py` | 2026-10-16 | **CURRENT** (D-35) |
| `script4n`, `4p`, `4q` | none found in the configured `LATEX_OUT` (`latex-outputs-run2`) on 10-17 | Rewritten 10-12 and edited 10-16 (reported vs robustness split). Not verified as run; treat as **NOT RUN**. |
| `script4j`, `4d-balance-panel`, `4k` | output dates not checked in this note | Verify before citing |
| `script4e`, `4g`, `4h`, `4d-power-balance` | 2026-09-18 | **LEGACY / STALE** — still read the retired `A1_A2` and `script4f` filenames; archive rather than patch |

---

## WHAT CHANGED SINCE 09-27 (one line each; detail in the register)

- D-31 Run 2 (10-12): A3 dropped; step-8 membership revised; A4/A5/A6 additions; LP-DiD, Conley, `script4r`.
- D-32 Team feedback 10-15: T1, T3, T4, E2 / county FE / state SE / TWFE + CS-adjusted are the
  **reported** set (`REPORT_*`); everything is still estimated; cut specs go to robustness exhibits.
- D-33 CS reported arm = covariate-adjusted (0 fallback cells; same signs as unconditional).
- D-34 Blank CAFO count → 0 from 2002 on, 2000–01 left missing (17,804 dairy values recoded).
  Headline A2 effect: 0 sign flips, 2 of 28 significance changes.
- D-35 2022 NASS totals pulled: bins = totals in every county; 2022 verified.
- D-36 Species set = dairy, beef, hogs, chickens; `script4s` overlap descriptives.
- D-37 API key removed from code (history still holds it — revoke).
- D-38 Caches / editor files untracked; `.gitignore` extended.

---

## RE-RUN ORDER (after D-34)

1. `script4j` — control diagnostics *(inputs to the control choices)*
2. `script4a` — A2, A4, A5, A6 (+ A1 / A7 / Conley, computed but not reported)
3. `script4r` — slaughterhouse treatment
4. `script4d-balance-panel`
5. `script4n` → `script4p` → `script4q` — tables, appendix, slides *(last; no estimation)*

`script4b` / `4c` need re-running only if `COVARIATE_ORDER` changes. If they are re-run, they now
still compute A1, T2 and all three SEs (D-32).

---

## NOT BUILT

| | Status |
|---|---|
| **Per-species runs** (dairy, beef, hogs, chickens) of A2 and CS-adjusted | not built — `load_panel(animal=...)` exists; needs a loop and the E2 cohort per species |
| **Forest plots**: T1, T3, T4, E2 (TWFE) + CS-adjusted × 7 outcomes, one plot per species | not built (team request 10-15) |
| Joint multi-species model | undecided — see D-36 for the evidence |
| Named sensitivity: step-8 + uninsured + obesity (D-29) | not run |
| CS cohort-definition sensitivity (D-30) | not tested |
| B-series (B1–B14), incl. B13 window justification and B14 event-study control group | specified only; B13/B14 still flagged blocking for the deck |

---

## OPEN ITEMS

**Data / pipeline**
- `script0b` does not read the 2022 totals file (D-35); a re-run labels 2022 "dark". Append the file
  and add 2022 to `VERIFIED_WAVES`.
- Move the missing-as-zero logic into `script1b`, species-aware, at the next panel rebuild (D-34).
  `script1b` line ~359 still says the 2022 file "has not been added".
- Carried from 09-27, status **unverified**: `access_to_healthy_foods_per100k` position in
  `COVARIATE_ORDER` vs D-05; `%_rural` removal (D-03) marked pending.

**Security**
- **Revoke and replace the NASS API key** (D-37).

**Release readiness (reviewer-style audit, 2026-10-17)**
- No `environment.yml` / `requirements.txt`; README package list is wrong (omits `pyfixest`,
  `doubleml`, `scipy`, `requests`; lists `linearmodels`, `differences`, which no script imports).
- No LICENSE; no data-availability statement (which sources are public, which need manual steps:
  CDC WONDER, FBI bulk files, FSIS FOIA).
- README still names the retired `script4l`; omits `script4r`, `script4s` and the QA scripts.
- Two `script4d-*` files; letters f, i, l, m, o skipped; legacy files at root
  (`script4-model-test.py`, `script3-ridge.py`, `script4e/g/h`, `script4d-power-balance`).
- No single run-order driver; "latest file" lookups make results depend on what is on disk.
- Hard-coded result numbers inside slide and appendix prose (e.g. switcher counts) go stale on re-run.
- 16 files use `from packages import *`; `functions.py` "used in" comments name deleted scripts.
- Internal notes (`code-audit/`) live in the same repo as the public code; decide what ships.

---

## Addendum 2026-10-25: new panel file, stage-4 inputs unchanged
`Data/merged/2026-10-25_panel*.csv` (written by `script0f-generate-panel.py`, which replaced `script1b-generate-panel.py`) is now the newest panel. It differs from `2026-10-12_panel.csv` only in seven generic CAFO columns
(`small_cafo`, `medium_cafo`, `large_cafo`, `bin_sum_cafo`, `total_heads_cafo`, `any_large_cafo`, `any_medium_or_large_cafo`; D-51). `script4_treatment.load_panel()` returns 184 columns that are identical on the old and the new panel
outside those seven, and no stage-4 script reads them (`git grep`), so **no estimation output needs re-running because of this change**; the table above still describes which script ran on which treatment values.
Not changed, still open: forward-fill carries an earlier census wave's species counts into a later wave that has no row (T1, T3, T4 differ in 119 county-years across 21 counties if census-year blanks are set to 0 first; E2 and E3 do not change).
Stage 1 (`script1a` to `script1d`) was run on the new panel on 2026-10-25; its outputs are in `Data/output/{tables,figs,maps}/script1*/`.

---

## Addendum 2026-10-25 (later): D-53 panel, re-run state
**Correction to the addendum above:** its statement that no estimation output needs re-running holds for D-51 only. D-53 (census-year blanks are 0 before the forward-fill; `script0f` rebuilt the panel the same day, the D-51 files are in `Data/merged/_archive/2026-10-25_D-51/`) changes T1, T3, T4, `tr_c*`, `cond_log_*` and the species size classes, so everything that reads them was re-run on `Data/merged/2026-10-25_panel.csv` and **those 2026-10-25 outputs are the current ones**; earlier outputs of the same scripts are superseded.
| Script | Run on 2026-10-25 | Result vs previous |
|---|---|---|
| `script4a` (A1/A2, A4, A5, A6, A7, VIF) | yes | A2 T1, T2, T3, T4 rows change; E2 rows, A6, A7, HEADLINE identical |
| `script4j`, `4d`, `4t`, `4y` | yes | leads and control-set tables change |
| `script4x` (species) | yes | hogs, chickens, dairy, beef rows change (193 of 280) |
| `script4z` (inference) | yes | T1, T3, T4 p-values move; E2 estimate and wild p identical, Romano-Wolf p moves slightly |
| `script4zc`, `4zd` | yes | TWFE and dCDH T1, T3, T4 rows change (42 of 140); E2/E3 rows identical |
| `script4n`, `4p`, `4q` | yes | tables, appendix and frames rebuilt from the new files |
| `script1b`, `1c`, `1d` | yes | stage-1 outputs rebuilt on the new panel |
| `script4u`, `4w`, `4za`, `4zb`, `4ze`, `4zf`, `4r` | **no** | read none of the changed columns (first-expansion/first-entry dairy cohorts, outcomes and controls identical); their earlier outputs stand |
Before/after numbers: `2026-10-25_D53-BEFORE-AFTER.md`. Not changed and still open: 68 county-years (dairy) stay blank because the county was not rural in the census year (D-53).

**Addendum 2026-10-26 (`script4a --only fsis`, i.e. the old `script4r`):** re-run on the 2026-10-25 panel, `PYTHONHASHSEED=0`, about 20 seconds. The R1/R2 TWFE results (14 rows) equal the 2026-10-22 values to 1e-12 (coefficient, state-clustered standard error, p) with identical N, counties and switchers; the earlier outputs stood, as the table above says, and the new dated files (`2026-10-26_4a-fsis_*`) are the ones the slaughterhouse forest now reads. The R3 event study and R4 LP-DiD files were rewritten by the same run and were not compared.

**Addendum 2026-10-29 (`script4g-spillovers.py`, new, D-58):** all five sections run on the 2026-10-25 panel with `PYTHONHASHSEED=0` (`4g-moran`, `4g-donut` 300 bootstrap draws, `4g-conley`, `4g-division-fe`, `4g-durbin` 499 bootstrap draws on the confirmatory family); files dated 2026-10-29. `W_cz` not run (no crosswalk). No existing result file was changed. The 15 counties without an outline are excluded from every `4g` result.

**Addendum 2026-10-30 (`script4g-spillovers.py --cafo/--scope`, D-59):** twelve runs on the 2026-10-25 panel, `PYTHONHASHSEED=0`, finished in about 1 hour of wall time (four workers, 10:35 to 11:36): dairy/large (moran and durbin only, now with `W_cz` from `commuting-zones-2020.csv`), and beef, cattle, hogs, chickens, aggregate x large, all, plus dairy/all, all five sections. Files `2026-10-30_4g-<section>_<stem>__<cafo>_<scope>.csv` (dairy/large: the usual names) and `..._failures__...csv`. Eight donut cells failed with an IndexError (first entry, four crime outcomes, cattle/all and aggregate/all, 28 or fewer treated counties) and are listed in those failures files; every other cell ran. Quick test outputs were moved to `Data/output/_archive/2026-10-30_quick-test-outputs/`. Nothing is in the deck. `script4b --only outcome-smoothing` ran the same day (`4b-outcome-smoothing`).

**Addendum 2026-10-30 (D-60):** `script4b --only leads` for aggregate/large and cattle/large; `script4g --only cs-pretrends` for aggregate/large, cattle/large and dairy/large; `script4g --cafo fsis --only moran donut conley division-fe durbin` (suffix `__fsis_slaughter`); `script4g --only interaction` for the six types (large). All on the 2026-10-25 panel, `PYTHONHASHSEED=0`, no failed cells; files dated 2026-10-30 (`4b-leads_*__<cafo>_<scope>`, `4g-cs-pretrends_*`, `4g-interaction_*`, `4g-*__fsis_slaughter`). The FSIS runs use 2017-2023, so the Poor MH Days and Frequent Mental Distress samples span the 2021 CDC method change and the carried-forward 2019 year (see 2026-10-30_OUTCOME-SMOOTHING.md). Nothing is in the deck.

**Addendum 2026-11-22 (D-61):** `script4h-gp.py --only export`, `R/nnrcm_sim_check.R`, `R/nnrcm_fit.R` (seven outcomes, `NNRCM_LIB` = scratch build), `script4h-gp.py --only summary`. 2026-10-25 panel, `PYTHONHASHSEED=0`, seeds 20261122; files `4h-gp_sim_check`, `4h-gp_fits__<outcome>`, `4h-gp_fits_with_range`, `4h-gp_range_summary` dated 2026-11-22. 91 of 825 fits did not converge (listed in the fits files, `error` column). Nothing is in the deck; no estimate changed.

**Addendum 2026-11-23 (D-62 to D-65, plan `2026-11-23_headline-designs-and-fsis`):** all on the 2026-10-25 panel and panel_fsis, `PYTHONHASHSEED=0`, files dated 2026-11-23. `script4i` (reads existing results, estimates nothing; `4i-headline_evidence`); `script4j` (full-window rows reproduce A2 0.104, Callaway-Sant'Anna 0.037 and 0.026; `4j-split_split_results`); `script4k` (`4k-fsis-large_*`: entry 97 counties, exit 42, exit not estimable by Callaway-Sant'Anna); `script4l` (`4l-fsis-cz_*`: 35 entering zones); `script4m` (`4m-fsis-clean_*`: 404 of 1,456 FSIS-panel counties have a closure; 12 entering zones after dropping closure zones; many 91 counties vs none 1,296-1,423). Results read against the pre-stated rules: the closure-clean county entry design fails pre-trends (4 of 5 pre-period estimates significant), many-vs-none fails the lead test for every outcome that has leads, so neither is read as a finding. HonestDiD (2026-10-22) and the earlier 4d estimator files (2026-10-24) were not re-run (their inputs did not change, see the 2026-10-25 addendum). Nothing is in the deck.

