# October QA
**Opened 2026-10-17.** Issues found while checking the code, what was done about each, and the
open questions we have to settle together. Decisions already made are in
`CONTROL-DECISION-REGISTER.md` (D-31 to D-38); what is stale or unbuilt is in
`2026-10-17_MODEL-RUN-STATE.md`.

Status key: **FIXED** (done and committed or staged) · **OPEN** (needs a fix) · **ASK** (needs a decision).

---

## 1. Re-run after the zero rule (D-34)
**Status: DONE 2026-10-17, all seven scripts exit 0** (`4j`, `4a`, `4r`, `4d-balance`, `4n`, `4p`, `4q`).
`4a` took 7 minutes (CS bootstrap, one arm). Old outputs were *moved* (not deleted) to
`output/_archive/2026-10-17_before-zero-rule/` first (83 files; README inside), so a script that
failed to write would have failed loudly instead of reading 10-12 numbers.

**Before/after, headline A2 (T1, T3, T4, E2 × 7 outcomes; county + year FE, step-8, state SE):**
- 0 of 28 sign changes; median |change in coefficient| 3.7%; samples +3.2% to +5.5% (e.g. Poor MH
  Days 34,090 → 35,918 rows).
- 2 of 28 significance changes, both E2: Poor MH Days p<0.01 → p<0.05; Violence index p<0.05 → p<0.10.
- Largest relative moves: T3 on Deaths of Despair +24% (insignificant before and after); E2 on
  Violence index -9.8%, NIBRS total -8.4%.

**Callaway-Sant'Anna (covariate-adjusted overall ATT):** every ATT within 6% of the old value; z-values
essentially unchanged (Poor MH Days 1.50 → 1.41; Frequent Mental Distress 0.82; Deaths of Despair 0.25;
Aggravated Assault 1.00 → 1.01; Assault all severities 1.96; Violence index 1.99 → 1.98; NIBRS total 2.20).

**TWFE-vs-CS verdicts:** 6 of 7 unchanged. One change: **Violence index "survives CS" → "CS-only"**,
because the TWFE p-value moved across 0.05 (see issue 23). Pattern as before: TWFE significant on 5 of 7
outcomes and CS significant on 2 of 7 (Violence index, NIBRS total); Poor MH Days, Frequent Mental
Distress, Aggravated Assault and Assault-all-severities are "TWFE-only".

**Other results worth knowing:** A4 still identified for Poor MH Days and Frequent Mental Distress only
(five outcomes suppressed: no wave transition inside 2017-2023). VIF audit: 105 specs, max 5.27 on
`log_pop`. `4r` (slaughterhouse entry) ran; LP-DiD pre-trend tests showed 0 significant leads for the
outcomes printed. Outputs: `output/tables/script4a`, `script4r`, `output/latex-outputs-run2/`
(`BEAMER_FRAMES.tex`, `BEAMER_FRAMES_robustness.tex`, `_BODY_regressions.tex`, `_BODY_robustness.tex`,
`FULL_RESULTS_APPENDIX.pdf`).

---

## 2. Bugs and issues found

### Real bugs
| # | Issue | Where | Status |
|---|---|---|---|
| 1 | **`log_per10k()` crashes with `NameError: name 'np' is not defined`.** `functions.py` never imports numpy. Confirmed by calling the function. It is used by the stage-2 figure scripts and by the first-stage ridge pilot. | `functions.py:505-506` | **OPEN** — one-line fix (`import numpy as np`); not applied because `functions.py` is shared. Also verify whether the stage-2 scripts really ran end to end after this was written (the 06-09 notes say they did). |
| 2 | **Hard-coded NASS API key** in four files, committed and pushed (`6bdb09f`, `c1fa465`). | `script0b`, `qa_2017_dta_vs_api`, `z-archive/script0p`, the 2022 pull script | **FIXED** in the working files (environment only); key revoked by you. The old key remains in history. |
| 3 | **Species overlap script filled 2000–01 with zeros**, contradicting the rule (D-34); recode count was inflated (28,774 reported, correct figure 6,142). | `script4s-species-overlap.py` | **FIXED** (committed `56de826`); census-year tables unchanged. |
| 4 | **Stale CS-unconditional / A1 / T2 / county-SE paths in `4b`/`4c`.** They still compute the cut specs and `4c` still tests significance on county-clustered SEs. By design (D-32) but inconsistent with the reporting layer. | `script4b`, `script4c` | **ASK** — see Q8. |
| 5 | **Zero-fill is implemented in two places** with different logic: `script1b` (species-blind, only when *all* species are blank) and `load_panel` (per species). The panel file on disk still has the blanks. | `script1b:373-383`, `script4_treatment.load_panel` | **OPEN** — move into `script1b` at the next panel rebuild (D-34). |
| 6 | **`script0b` does not read the new 2022 totals file**, so a re-run labels 2022 "dark" again; `VERIFIED_WAVES` still lists 2002–2017. | `script0b`, `script4_treatment` | **OPEN** (D-35). |

### Code quality (a reviewer would notice)
| # | Issue | Status |
|---|---|---|
| 7 | 40 unused imports and 8 f-strings with no placeholders (pyflakes), across the scripts. Cosmetic. | OPEN |
| 8 | 16 files use `from packages import *` / `from functions import *`; static checkers cannot detect undefined names (37 warnings). This is how bug 1 stayed hidden. | OPEN |
| 9 | `functions.py` "used in:" comments name scripts that no longer exist (`script0b-ag-raw*.py`, `script2h-*`). | OPEN |
| 10 | `script1b` comment (line ~359) still says the 2022 file "has not been added". | OPEN |
| 11 | Result numbers hard-coded in slide and appendix prose (e.g. "89 switchers", "17,975") go stale on every re-run. One was made computed (VIF audit); the rest remain. | OPEN |
| 12 | No tests; one `assert` across the main scripts. | OPEN |
| 13 | Checked and clean: no invalid escape sequences, no leftover `breakpoint`/`pdb`/`TODO`/`DEBUG` markers. | — |

### Reproducibility / release (from the reviewer-style audit)
| # | Issue | Status |
|---|---|---|
| 14 | No `environment.yml`/`requirements.txt`; the README package list is wrong (omits `pyfixest`, `doubleml`, `scipy`, `requests`; lists `linearmodels`, `differences`, which nothing imports). | OPEN |
| 15 | No LICENSE; no data-availability statement. | OPEN |
| 16 | README still names the retired `script4l`; omits `script4r`, `script4s` and the QA scripts. (Rows for the moved first-stage scripts were updated today.) | OPEN |
| 17 | Two `script4d-*` names; letters f, i, l, m, o skipped. One of the two is now in `first-stage-testing/`. | PARTLY FIXED |
| 18 | No single run-order driver; "latest file" lookups make results depend on what is on disk (mitigated this week by the archive step). | OPEN |
| 19 | Internal notes (`code-audit/`) share a repo with the public code. | ASK — Q10 |
| 20 | Tracked caches / editor files. | **FIXED** (`9556da1`) |
| 21 | **`script4p` depends on two fragments no script produces.** `TABLE_covariates_EXCLUDED_FOR_CAUSE.tex` and `..._NO_PRETREND_TEST.tex` exist only in the old folder `slides/latex-outputs-2809` (dated 09-28). `4p` aborted ("missing fragments") until I **copied** them into `latex-outputs-run2`. They are static tables of the D-29 exclusions; content not re-verified. | **OPEN** — generate them from `COVARIATE_EXCLUDED` / the register, or commit them as source files. |
| 22 | **Conley SEs were never computed in run 2**: `4a` skipped them because `Data/raw/gazetteer/2020_Gaz_counties_national.txt` is absent. The Conley table in `4n` therefore does not exist. (Not reported anyway, D-32.) | **OPEN** — download the file or drop the code path. |
| 23 | **Borderline significance.** CS for Assault, all severities has z = 1.96 and is classed not significant by the third decimal; Violence index z = 1.98 and its TWFE p sits near 0.05. Verdict labels on these two outcomes can flip on small data changes. | **ASK** — report z/p rather than a verdict label? |
| 24 | **pyfixest 0.60 silently ignores the varying-slope syntax `fips[year]`.** A model "with county trends" returned exactly the baseline coefficient (+2.113 vs +2.113 on a test panel where explicit trend dummies give +0.307). Any county-trend robustness built on it would have been a false pass. | **FIXED** in `script4u` (explicit residualization, checked against pyfixest with no trend: coefficients to 1e-9, SEs within 0.15%). |
| 25 | **CS code exists twice**: `script4a` and the new `did_cs.py` (copied unchanged; `script4u` asserts they agree to 1e-15). | **OPEN** — switch `4a` to import `did_cs`. |
| 26 | **No Python HonestDiD worth citing.** The PyPI package needs torch (~2 GB) + cvxpy, is a 0.1.1 beta by one author with a placeholder email. | **ASK** — use the R reference packages `HonestDiD` and `did` from CRAN (not installed yet). |
| 27 | **Sample construction is duplicated** across `script4u`, `script4t`, `script4v` and `script4w-honestdid-export.py` (cohort definitions, the entry sample, the never-large set). `script4a` and `did_cs.py` duplicate the CS code too. | **OPEN** — one shared "designs" module; check outputs unchanged after. |
| 28 | **"Never large dairy" is 2,103 counties in `script4u` but 2,120 in `script4v`.** 17 counties have no row in any census year, so `script4u` (which builds the set from census-year rows) leaves them out and `script4v` (which treats missing as never-large) includes them. | **OPEN** — decide which is right; make both use the same definition. |
| 29 | **HonestDiD is installed** (R `HonestDiD` 0.2.8, from CRAN) and run on the TWFE event studies only. A CS version needs the bootstrap covariance of the event-time ATTs, which `did_cs.py` does not save. | **OPEN** |
| 30 | Treatment-timing measurement: effects appear at event times 2 and later, treatment dates are census waves 5 years apart. Event-time paths therefore mix true timing and measurement error. Not a bug; a limit on every event-study exhibit. | NOTE |
| 31 | **HonestDiD run printed "50 or more warnings" that were not captured**, and the relative-magnitudes grid (Mbar = 0, 0.5, ..., 2) is too coarse to give a breakdown value finer than 0.5. In 23 of 28 cases zero is already in the original interval, so the breakdown is 0 by construction. | **OPEN** — re-run with warnings logged and a grid of 0.1; consider also running on CS. |
| 32 | **Results depend on `PYTHONHASHSEED`.** Same code, same data, same machine: unseeded runs fall into two classes (seeds 0, 4, 6 vs 3, 5, 7 in a test), differing by about 2e-5 relative in `pyfixest` SEs (coefficients about 1e-8, p-values up to 6e-4). Each seed reproduces itself exactly. The mechanism is presumably set ordering inside the formula/demeaning code; not traced. Found while proving the 10-22 refactor changed nothing: the 10-17 baseline vs the refactored run differed only because they landed on different seeds. | **FIXED for runs**: README and `environment.yml` set `PYTHONHASHSEED=0`; the canonical outputs were regenerated with it. **OPEN**: find the cause; check any knife-edge p-value (QA #23) under both classes. |

---

## 3. First-stage scripts moved (2026-10-17)
Moved with `git mv` to `first-stage-testing/` and renamed `firststage_*`; nothing edited; the folder
has its own README. Not part of the pipeline and not used for any reported result.

`script3-ridge`, `script4-model-test` (1,538-line first model test), `script4d-power-balance`,
`script4e-latex-tables`, `script4g-slides`, `script4h-result-tables`.

Not moved, listed under Q9: `script2e-hispanic-chr-explorer` (exploratory but part of stage 2).
Comments in other files still use the old script names.

---

## 4. Open questions

**Q1. Forest-plot exposure: "large" alone, or all sizes too?** You asked for the count of small +
medium + large together, not only large. Large remains the headline measure; the question is how
to define T1/T3/T4/E2 on the all-sizes count. Presence of *any* dairy (T1 on all sizes) is true in
about 91% of rural census-year county-years, against about 21% for large dairy, so it has far less
variation to work with. E2 ("first positive change") on the all-sizes count would mostly reflect
small farms, and total dairy operations tend to *fall* (70–75% of positive large-herd changes
happen while the county's total falls). Proposal:
run both definitions side by side, large first, and label the all-sizes set as a separate panel of
the same plot.

**Q2. Forest-plot axes.** Each estimate in its own units on a free axis, or standardized by the
outcome's within-county SD? T1 (0/1), T3 (per 10k), T4 (log) and CS (ATT) are in different units.

**Q3. Species and power.** Dairy, beef, hogs, chickens (cattle excluded: it contains dairy and
beef). Chickens have large operations in about 6.5% of county-years; expect CS to be underpowered
there. Joint multi-species model, yes or no? (D-36 gives the overlap evidence.)

**Q4. E2 for species other than dairy.** The "positive change while the total falls" pattern was
measured for dairy only. Check hogs and beef before reading E2 as expansion.

**Q5. T3 is kept at the team's request despite D-27** (per-capita variation partly reflects
population drift). Keep it in every exhibit with a caveat?

**Q6. CS cohort definition** (D-30): `4f` used a different cohort per treatment; `4a` uses
`add_year` only. Do we test sensitivity?

**Q7. Named sensitivity** (D-29): step-8 plus uninsured plus obesity. Run it before a referee asks?

**Q8. `4b`/`4c`.** You suggested combining them. Also: update them to the reported set (no A1, no
T2, state SEs), or keep them as the record of how step 8 was chosen? The step-8 membership changed
in run 2 after they last ran (D-31).

**Q9. `script2e` (Hispanic explorer) and the rest of stage 2:** keep as descriptive pipeline, or
move to first-stage-testing?

**Q10. Public release.** What ships? The proposal: tag a release that excludes `code-audit/` and
`z-archive/`, and add LICENSE (which?), `environment.yml` and a data-availability statement
(CDC WONDER, FBI bulk files and the FSIS FOIA file are manual downloads).

**Q11. 2000–01.** Leave blank (current), or add the 1997 census so those years can be filled?
That is a `script0b` change and a new pull.

**Q12. Where the zero rule lives.** Keep in `load_panel`, or move into `script1b` and rebuild the
panel now rather than at the next planned rebuild?
