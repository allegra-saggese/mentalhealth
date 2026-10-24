# Control variable decision register
**Living document — every include/exclude decision, with its justification.**
Started 2026-09-21. Update whenever a control enters or leaves a specification.

Purpose: each inclusion and exclusion must be defensible to a referee. This file is
the record. Cite the decision ID in the paper's data appendix.

---

## CURRENTLY INCLUDED — `CONTROL_PRETREAT` (18 as of 2026-09-27)

| # | Variable | v-code | Why included | Risk flags |
|---|---|---|---|---|
| 1 | `%_65_and_older` | v053 | Age structure predicts both agricultural land use and mental-health reporting. Predetermined. | pre-trend flat |
| 2 | `%_asian` | v081 | Predetermined demographic composition. | near-zero in non-metro; adds little |
| 3 | `%_below_18_years_of_age` | v052 | Age structure. Predetermined. | pre-trend flat |
| 4 | `%_female` | v057 | Predetermined. | pre-trend flat |
| 5 | `%_hispanic` | v056 | Strong confounder for agricultural county composition. | labelled mediator on labour-recruitment grounds, but pre-trend is FLAT (0.006 SD) — the mediation concern is not supported by the data |
| 6 | `%_native_hawaiian/other_pacific_islander` | v080 | Predetermined. | negligible in non-metro; adds little |
| 7 | `%_not_proficient_in_english_per100k` | v059 | Tracks immigrant labour composition. | labelled mediator; pre-trend FLAT |
| 9 | `access_to_healthy_foods_per100k` | v030→v083 | Food environment. | **MEASURE BREAK** v030→v083 at release 2013 — two different measures spliced. See D-05 |
| 10 | `adult_obesity_per100k` | v011 | Health stock, slow-moving. | pre-trend flat |
| 11 | `adult_smoking_per100k` | v009/v095 | Behavioural health stock. | labelled COLLIDER; FAILS pre-trend (2 sig); also a one-year code blip in 2011. See D-04 |
| 12 | `children_in_poverty_per100k` | v024 | Local economic conditions. | mediator; pre-trend flat; lag −2 confirmed |
| 13 | `children_in_single-parent_households_per100k` | v082 | Family structure, slow-moving. | FAILS pre-trend (2 sig) despite confounder label |
| 14 | `driving_alone_to_work_per100k` | v067 | Commuting pattern, proxies labour-market geography. | labelled confounder but 4 SIGNIFICANT POST coefficients — empirically a mediator |
| 15 | `median_household_income` | v063 | Income level drives both CAFO siting and mental health. | BOTH confounder and mediator: 2 sig pre AND 3 sig post, rising to +0.13 SD by t+6. Lag −2 confirmed |
| 16 | `some_college_per100k` | v069 | Education stock, predetermined. | 1 sig post |
| 17 | `teen_births_per100k` | v014 | Social conditions. | labelled COLLIDER; 4 sig pre AND 4 sig post. See D-04 |
| 18 | `unemployment_per100k` | v023 | Local labour market. | mediator; pre-trend flat; lag −2 confirmed |
| 19 | `uninsured_adults_per100k` | v003 | Healthcare access. | labelled mediator but signature is confounder (2 pre, 1 post) |

---

## EXCLUDED — with reasons

### D-01 · Excluded as plausible MEDIATORS (7) — `CONTROL_MEDIATORS`
`poor_physical_health_days`, `premature_death`, `preventable_hospital_stays`,
`low_birthweight_per100k`, `diabetes_prevalence_per100k`,
`physical_inactivity_per100k`, `poor_or_fair_health_per100k`

**Why:** all are health outcomes a CAFO could plausibly move. Conditioning on a mediator
blocks part of the causal path and biases toward zero.
**Retained in** `CONTROL_COMPREHENSIVE` (26) so the two can be compared as a robustness arm.
**Caveat on this reasoning:** the treatment explains <1% of within-county variation in the
full control set (treatment VIF 1.010, R² 0.0096), so the mediation channel is empirically
weak. This exclusion is precautionary, not evidence-driven.

### D-02 · Excluded on COVERAGE (<92% on outcome rows) (6)
`income_inequality` (73.0%), `social_associations` (66.1%),
`%_non-hispanic_african_american` (63.0%), `air_pollution_particulate_matter` (78.0%),
`mental_health_providers` (79.1%), `primary_care_physicians` (89.1%)

**Why:** listwise deletion. The legacy 25-variable set cut the Poor MH Days sample from
~34,000 rows / 284 switching counties to ~10,500 / 96 — two-thirds of the identifying
variation — for covariates county FE largely absorbs anyway.

### D-03 · REMOVED — `%_rural`  (decided 2026-09-27)
**Why:** 93.7% of county-year values are identical to the prior year (decennial census,
carried forward; median 2 distinct values per county across 13 years). Within-county SD is
7.5% of raw SD, so county FE absorbs ~92.5% of it and it contributes almost nothing to beta.
**Additional:** the sample is filtered on NCHS codes 3–6 (`non_large_metro`), which INCLUDES
530 metropolitan counties (326 medium metro, 204 small metro = 18.6%). `%_rural` is a
different concept — population share in census-defined rural blocks — so it is not a
consistency check on the filter either.
**Status:** agreed in principle 2026-09-21, not yet removed.

### D-04 · CANDIDATES FOR REMOVAL — the two COLLIDERS (decision pending)
`adult_smoking_per100k`, `teen_births_per100k`
**Why:** both are plausibly caused by treatment AND by the outcome; conditioning on a
collider induces spurious association. Both also FAIL the covariate pre-trend test.
**Counter-argument that must be recorded:** `adult_smoking` has a measure-code break
(v009→v095→v009 around 2011), which would produce a spurious pre-trend on its own. So its
failed pre-trend is not independent evidence. Decide on the causal argument, not the test.

### D-05 · DROP `access_to_healthy_foods` — decided 2026-09-21
Three independent reasons:
1. **Measure break**: v030 (releases 2010–2012) → v083 (2013+) are different CHR measures
   merged into one column. Pre-2013 and post-2013 values are not the same quantity.
2. **No annual source exists.** Confirmed against the USDA Food Access Research Atlas
   download page — published roughly every 5 years only. The 2025 CHR release uses 2019 data
   (a 6-year lag).
3. **Its within-county variation is almost certainly artificial.** It has the HIGHEST
   within-SD share of any control (0.898) despite coming from an irregularly-published
   source. A variable updated every 5 years cannot legitimately be the most
   annually-variable control in the set — that movement is the v030/v083 splice.

---

## STANDING DATA-INTEGRITY ISSUE AFFECTING ALL OF THE ABOVE
CHR variables are dated by RELEASE year, not data year. Confirmed lags: −2 for
`median_household_income`, `children_in_poverty`, `unemployment`; −3 for
`poor_mental_health_days`. See `2026-09-21_CHR-YEAR-MISALIGNMENT.md`. Until the panel is
re-dated, every control is misaligned with treatment, and the outcome is misaligned with
the controls by one further year.

---

## OUTCOME DECISIONS

### D-06 · DROP `violent_crime` (CHR v043) as an outcome — agreed 2026-09-21
**Why:** CHR pools it across multiple years — 46.7% of county-year values are identical to
the prior year — so it cannot support annual event-study timing. Also release-year dated,
and discontinued after the 2022 release.
**Consequence:** the UCR violent-crime definition (homicide + rape + robbery + aggravated
assault) cannot be reconstructed from the panel, because ROBBERY is not present in the
NIBRS block. Check the raw UCR files before accepting this limitation.

### D-07 · RENAME / re-scope the NIBRS outcomes — DONE 2026-09-21
- `crime_assault` renamed to **`assault_total_per100k`**, labelled "Assault, all severities".
  It is 79% simple assault, which UCR excludes from violent crime, so it must never be
  presented as a violent-crime measure.
- `aggravated_assault_per100k` promoted to its own outcome **`crime_agg_assault`**
  ("Aggravated Assault") — the largest genuine UCR violent offence available, annually
  dated, no pooling.
- New **`violence_index_partial`** ("Violence index (partial)") = aggravated assault +
  rape + intimidation. Explicitly NON-UCR. Mean 85.6 per 100k.

### D-08 · CORRECTION — what `total_incidents_per100k` actually is
I previously described it as "all NIBRS incidents including property crime". **That was
wrong.** `script0d` sums a CURATED list of 12 offence types chosen with the research team:
aggravated assault, simple assault, intimidation, rape, statutory rape, incest, fondling,
sexual assault with an object, kidnapping/abduction, DUI, and two human-trafficking codes.
Property and drug offences are NOT included. It is dominated by assault — simple assault
alone is 68% of it; simple + aggravated is 86%. Renamed in the outcome list to
"NIBRS curated total" so the label does not overstate its scope. It remains close to a
rescaling of `assault_total_per100k` and adds little independent information.

### D-09 · MAJOR CAVEAT — the NIBRS data are ARRESTS, not offences
The raw files are `nibrs_arrestee_segment_YYYY.csv`. Every NIBRS variable in the panel is
an **arrest** count, not an offence known to police. Arrest rates confound crime incidence
with policing intensity, staffing and clearance. Published crime rates are offence-based
and are therefore **not comparable** to these. This must be stated wherever the crime
outcomes are reported. Not previously flagged anywhere.

### D-10 · ROBBERY is available but not extracted (decision pending)
Robbery IS present in the raw `nibrs_arrestee_segment` files already in the repo
(29,310 arrests in 2021 alone; 64 distinct offence codes are available). It is absent from
the panel only because `script0d`'s `crime_cols` list does not select it. Adding it would
complete three of the four UCR violent components (aggravated assault, rape, robbery —
homicide would still be missing from NIBRS). **Requires re-running stage 0**, which
regenerates the panel. Not done.


---

# OUTCOME SET — FINAL (decided 2026-09-21)

**Seven outcomes, all run separately as Y_it. No composites dropped.**

| # | Label | Variable | Coverage | Years |
|---|---|---|---|---|
| 1 | Poor MH Days | `poor_mental_health_days` | 54.7% | 2010–2023 |
| 2 | Frequent Mental Distress | `frequent_mental_distress_per100k` | 32.9% | 2016–2023 |
| 3 | Deaths of Despair | `crude_rate_from_census_pop` | 19.8% | 2000–2020 |
| 4 | Aggravated Assault | `crime_agg_assault` | 38.1% | 2000–2021 |
| 5 | Assault, all severities | `assault_total_per100k` | 38.1% | 2000–2021 |
| 6 | Violence index (partial) | `violence_index_partial` | 38.1% | 2000–2021 |
| 7 | NIBRS curated total | `total_incidents_per100k` | 38.1% | 2000–2021 |

### D-11 · KEEP all four crime outcomes — decided 2026-09-21
I recommended dropping the composites (`assault_total`, `violence_index_partial`,
`total_incidents`) on redundancy grounds and replacing them with simple assault and
intimidation as separate components. **Team decision: keep all four and run them
separately.** Rationale: reporting every measure is more transparent than pre-selecting,
and the reader can see for themselves which move together.

**Redundancy must be disclosed when reporting, not used to drop:**
- `assault_total_per100k` vs `total_incidents_per100k`: **r = 0.989** (raw). These are
  close to the same variable; do not present them as independent corroboration.
- `violence_index_partial` vs `crime_agg_assault`: **r = 0.916** (raw). The index is 67%
  aggravated assault by construction.
- Within-county (what the FE estimator uses), the underlying components are genuinely
  distinct: aggravated vs simple assault r = 0.372, aggravated vs intimidation r = 0.264,
  simple vs intimidation r = 0.317.

**Not added** (were offered as replacements, not taken): `simple_assault_per100k` and
`intimidation_per100k` as standalone outcomes.
**Excluded as too sparse**: `rape_per100k` (44.6% non-zero), `driving_under_the_influence`
(32.0% non-zero).

### Standing caveats on all four crime outcomes
- **D-09**: these are ARREST counts (`nibrs_arrestee_segment`), not offences known to
  police. Confounds crime with policing intensity. Not comparable to published UCR rates.
- NIBRS coverage rises from 21% of counties (2000) to 70% (2021); **76.4% of counties
  switch in and out of reporting** and only 43 report every year. No county-year ever
  reports zero, so "missing" and "no crime" are indistinguishable.
- Despite that, 45.3% of treated counties are observable both pre- and post-treatment in
  the crime sample, against 49.7% for Poor MH Days — comparable, not disqualifying.
- Crime data ends 2021, so the 2022 entry cohort contributes no post-period.


---

# SOURCE TIERING — decided 2026-09-21

Following the CHR 2025 "Select and additional measures, data sources and years" PDF.
Every control is now assigned to a tier that determines how it may be used and reported.

### TIER A · Re-source at true annual data year (9 variables) — TO DO
The underlying source IS annual AND IS published for every county regardless of population.
CHR's lag here is an avoidable error.

| Variable | Correct source | CHR lag (2025 release) |
|---|---|---|
| `median_household_income` | SAIPE | −2 |
| `children_in_poverty_per100k` | SAIPE | −2 |
| `unemployment_per100k` | BLS LAUS | −2 |
| `uninsured_adults_per100k` | SAHIE | −3 |
| `%_65_and_older` | Census PEP | −2 |
| `%_female` | Census PEP | −2 |
| `%_hispanic` | Census PEP | −2 |
| `%_asian` | Census PEP | −2 |
| `%_below_18_years_of_age` | Census PEP | −2 |
| `%_native_hawaiian/other_pacific_islander` | Census PEP | −2 |

(That is 10 rows; the 6 demographics come from one PEP pull.)

### TIER B · Keep as ACS 5-year estimates (4 variables) — KEEP, with restrictions
`some_college_per100k`, `driving_alone_to_work_per100k`,
`%_not_proficient_in_english_per100k`, `children_in_single-parent_households_per100k`,
**`teen_births_per100k`** (CHR pooled 7-year — added to this tier 2026-09-21)

**No annual alternative exists.** ACS 1-year estimates are published only for areas with
population ≥ 65,000, and **80.4% of our counties fall below that** (median county
population 21,568). The 5-year estimate is the correct and only data.

**RESTRICTIONS — these must be observed:**
- Report them explicitly as 5-year rolling estimates, not annual values.
- **Do NOT test or interpret pre-trends on these variables.** Consecutive releases share
  four of five years of underlying data, so the series is mechanically smoothed and cannot
  show a differential trend regardless of the truth.
- **This invalidates my earlier covariate pre-trend result for `driving_alone_to_work`**
  (4 significant post-treatment coefficients, read as a mediator signature). That reading
  is unsafe for a 5-year rolling estimate and is withdrawn.
- Their within-SD share (0.31–0.49) reflects the rolling window sliding forward, not real
  annual change.

### TIER C · Drop
- `%_rural` — decennial only, within-SD share 0.069, and the sample is NCHS 3–6
  (non-large-metro), not rural, so it is not a consistency check either. See D-03.
- `access_to_healthy_foods_per100k` — see D-05.

### D-12 · `teen_births_per100k` — no better source exists
CHR pools NCHS Natality over 7 years (2017–2023 in the 2025 release). Alternatives assessed:
- **CDC WONDER Natality, annual**: county teen births fall below CDC's <10 suppression
  threshold in an estimated **48.6%** of our counties, and below the <20 "unreliable" flag
  in **70.1%**. For scale, Deaths of Despair — which already lives under this rule — has
  only 22.6% coverage. Annual sourcing would cost roughly half the sample.
- **Restricted-use NCHS natality files**: full counts, but require a data use agreement.
  Out of scope.

**DECIDED 2026-09-21: KEEP the CHR pooled measure.** It moves to TIER B — usable as a
control, but it **cannot be assessed for pre-trends**, for the same reason as the four ACS
5-year measures: the series is pooled/smoothed and cannot show a differential trend
regardless of the truth.

**NCHS annual alternative assessed and rejected** (data.cdc.gov `3h58-x6cd`, hierarchical
Bayesian space-time model, county-level, 2003–2020, no suppression). Three reasons:
1. **It ends in 2020.** Our outcomes run to 2023 (mental health) and 2021 (crime).
   Swapping costs 6,289 rows and 78 switching counties on Poor MH Days, and 6,640 rows /
   135 switchers on Frequent Mental Distress. Coverage within 2003–2020 is better
   (99.7% vs 57.4%) but the endpoint more than cancels it.
2. **It is MORE smoothed than the measure it would replace.** AR(1) of within-county
   deviations: NCHS +0.984 vs CHR pooled +0.953. Bayesian borrowing across years makes a
   nominally annual series behave more smoothly than a 7-year pooled average.
3. **Spatial borrowing is a specific hazard for this design.** The model borrows strength
   from NEIGHBOURING counties. Dairy CAFO counties cluster geographically, so treated and
   control counties would borrow from each other, attenuating the very contrast being
   estimated.

`teen_births` remains independently flagged as a COLLIDER under D-04. That question is
separate from sourcing and is still open.

---

## COVARIATE-PATH STABILITY (script4b → script4c)

### D-13 · The covariate sequence is a SAMPLE experiment after step 8, not a confounding experiment
**Established 2026-09-27.** Source: `Data/output/tables/script4c/2026-09-27_path_stability.csv`
(56 paths = 4 treatments × 7 outcomes × 2 FE specs, built from the 840 regressions in
`script4b/2026-09-23_covariate_sequence.csv`). Headline vcov = county-clustered.

`COVARIATE_ORDER` puts the eight full-coverage covariates first, so **through step 8 the
sample is intact**. Step 9 adds SAHIE (2008 start) and steps 12–14 add the ACS 5-year
block; by step 14 the panel has lost a **median 35.7% of rows (max 42.0%)**.

Decomposing where β actually moves:

| | median \|move\| |
|---|---|
| step 0 → step 8 (sample intact) | **13.8%** |
| step 8 → step 14 (sample falls) | **22.8%** |

Of the 32 paths that move more than 25% in total, **11 are stable through step 8 and only
break afterwards**. For those, the movement is a statement about *which counties remain*,
not about confounding.

**Consequence:** step-8 and step-14 estimates answer different questions and must be
reported as two columns, never as one path. A covariate entering after step 8 cannot be
credited with "controlling for" anything until the same regression is re-run on the
step-14 subsample with the step-8 covariate set (not yet built — see D-15).

### D-14 · Deaths of Despair is not usable as a headline outcome
**Established 2026-09-27.** Most fragile outcome by a wide margin: median \|total change\|
**109.0%**, both of the study's two sign flips, 75% of its paths fragile.

| Treatment | β step 0 | β step 8 | β step 14 | move to step 8 | total move |
|---|---|---|---|---|---|
| T1 | −0.343 | −0.360 | −0.867 | **−5.1%** | −153% |
| T4 | −0.292 | −0.333 | −1.112 | **−14.0%** | −280% |
| T2 | +0.018 | +0.009 | −0.034 | −52.7% | −287% (sign flip) |
| T3 | +0.416 | +0.094 | −1.325 | −77.4% | −418% (sign flip) |

For T1 and T4 the coefficient is **essentially flat through step 8** and then triples or
quadruples once the sample drops. Both "gain significance" over the path. That is the
signature of a subsample effect, not of confounder adjustment.

This compounds a known limitation: Deaths of Despair has only **22.6% county coverage**
(CDC suppression at counts <10, see D-12), so the step-14 subsample is a small, selected,
and systematically larger-county group.

**DECIDED: Deaths of Despair is reported as exploratory only.** It does not carry a
headline claim and any result for it is shown with the full step-0 / step-8 / step-14
path attached, not as a point estimate. Assault (all severities) is the most stable
outcome (median 18.3%, 12.5% fragile) and Poor MH Days remains the headline (D-16).

### D-15 · OPEN — sample-vs-covariate decomposition not yet built
The step-8/step-14 comparison confounds two changes at once. The clean test re-runs the
**step-8 covariate set on the step-14 subsample**: any remaining gap between that and the
step-8 full-sample estimate is pure sample composition. Not yet run.

### D-16 · Pre-committed specification = step 8
**DECIDED 2026-09-27.** The reported specification is the **eight full-coverage covariates**
(`COVARIATE_ORDER[:8]`), county + year FE, county-clustered SEs. Reasons:
1. The sample is intact (N ≈ 65,578, no truncation of the 2008-pre window).
2. All eight are annually measured and can be tested for pre-trends. The step 9–14 block
   cannot be (ACS 5-year rolling, SAHIE start date) — see D-12 and the TIER B note.
3. It is fixed in advance of seeing the stability table, which is what keeps the reported
   p-values valid. Step 14 is reported alongside as a robustness column, not as a choice.

**This is a pre-commitment, not a selection.** script4c must never be used to pick the
reported step; it exists to disclose how much the answer moves.

### D-03 CLOSED · `%_rural` removed 2026-09-27
Removed from `CONTROL_COMPREHENSIVE` in `script4_treatment.py`. It had already been absent
from `COVARIATE_ORDER` and documented in `COVARIATE_EXCLUDED`, so the two definitions of the
control pool disagreed. They now agree.

`CONTROL_COMPREHENSIVE` 26 → **25**; `CONTROL_PRETREAT` (derived) 19 → **18**.

No re-run of script4b/4c is required — `%_rural` was never in `COVARIATE_ORDER`, so the 840
regressions are unaffected. The stale scripts that DO use `CONTROL_PRETREAT` (4a, 4f, 4i, 4j)
will pick up the change when re-run.

---

## MODEL A3 — dairy conditional on other large CAFO presence

### D-17 · T6 retired; the conditioning is a MODEL, not a treatment
**Decided 2026-09-28.** T6 ("log(1+large dairy) given the baseline number of SMALL dairy
CAFOs") is removed from `script4_treatment.py`. Two separate decisions closed it:
1. The conditioning we want is on **other CAFO TYPES**, not small dairy. That is a
   conditioning set applied to T1–T4, not a fifth treatment. It is now model **A3**.
2. **Small-dairy conditioning is dropped entirely** — the consolidation question is not
   being pursued.

`CORE_TREATMENTS` is now exactly T1–T4, with no gaps and no unrun members.

### D-18 · A3 conditions on PRESENCE, DISAGGREGATED, and excludes beef
**Decided 2026-09-28.** Pool = `any_lg_cattle`, `any_lg_hogs`, `any_lg_chickens`.

- **Presence, not counts.** A binary indicator means the same thing against a binary (T1),
  count (T2), per-capita (T3) or logged (T4) treatment. A count control against a binary
  treatment would change what the coefficient reads as.
- **Disaggregated, never a pooled "any other large CAFO."** On the 09-23 panel, **0 counties**
  have a large dairy and no other large CAFO (14 county-years, 4 counties, all of which have
  other CAFOs in other years). A pooled indicator has no counterfactual cell and only 39 of
  759 dairy counties ever switch it. By type it is thin but estimable — switchers among
  dairy-ever counties: hogs 119, chickens 79, cattle 58.
- **`cafo_beef_large` excluded.** Strict subset of `cafo_cattle_large`: `beef>0 & cattle==0`
  in **0.0%** of rows, `cattle>0 & beef==0` in 31.2%. Entering both double-counts the same
  operations. *Reverses if the substantive interest is beef feedlots specifically.*

### D-19 · A3 RESULT — the within estimate is not "any CAFO"
**Run 2026-09-28**, `script4l-a3-horserace.py`, 448 regressions (4 × 7 × 2 × 8 subsets),
zero failures, covariates held at the D-16 step-8 set. Output:
`Data/output/tables/script4l/2026-09-28_A3_horserace.csv`.

**The prediction was registered in the script before the run**: within-county correlation
between dairy presence and the three indicators is +0.084 / −0.001 / −0.024, so A2 should
barely move; levels correlations are +0.277 / +0.117 / +0.176, so A1 should move.

| | median \|move\| vs unconditioned | max |
|---|---|---|
| **A2 within** | **3.03%** | 13.63% |
| **A1 pooled** | 18.02% | 454.50% |

**A2 confirmed the prediction.** Conditioning on all three other animal types moves the
dairy coefficient by a median 3% and never more than 14%. This is the substantive finding:
**the within-county dairy estimate is not picking up "this county has a large CAFO."**

Three checks that the A2 null is real and not an artifact:
- **Not a sample effect.** N 22,137 → 21,968; switchers unchanged at 286.
- **Not dead controls.** The conditioning coefficients are themselves non-trivial
  (median β: cattle −0.159, hogs −1.626, chickens −1.806).
- **Not collinearity.** Max VIF 3.35 across all 448.

**The A1 movement is not a real reversal.** All 12 A1 specs moving >25% are
**insignificant after conditioning** (p = 0.20–0.73), and the two "sign flips" (T1 and T4
on Poor MH Days) are near-zero coefficients crossing zero with p = 0.40 and p = 0.67. The
454% and 411% figures are small-denominator artifacts (β goes −0.0045 → +0.0140), not
evidence of a dairy effect being explained away. **Percentage change is the wrong statistic
when the reference β is indistinguishable from zero**, and these should be reported as
levels with CIs, not as percentages.

Of the single types, **cattle** moves β most under A2 (median \|move\| 2.13%), hogs least
(0.44%) — consistent with cattle being the only one with meaningful within-county
co-movement (+0.084).

### D-20 · A3 is a probe, not a better-identified model
Other-animal presence is **not clearly pre-determined** with respect to dairy. If large
dairy expansion displaces or attracts other livestock, these indicators are post-treatment
and conditioning on them is a bad control. A3 answers *what does the dairy coefficient
contain*; A2 remains the headline specification.

---

## PART A RE-RUN ON THE CORRECTED PANEL (script4a, 2026-09-28)

`script4a-twfe-eventstudy.py` was archived to `z-archive/` and rebuilt as
`script4a-design-estimates.py`. Changes: treatment grid is now CORE T1–T4 (not the legacy 12);
7 outcomes (not 6); headline control set is the **D-16 step-8 pre-committed set**, with
`CONTROL_PRETREAT` (18) reported alongside as a robustness arm; model IDs follow
`2026-09-27_MODEL-NAMING.md` (old A6→A4, A7→A5, A18→A6).

### D-21 · The headline result does NOT survive Callaway–Sant'Anna
**Poor MH Days, T1, county+year FE, step-8 controls:**

| Estimator | β | SE (state) | significant |
|---|---|---|---|
| **A2 TWFE** | **+0.0795** | 0.0360 | **yes** (p = 0.032) |
| **A6 Callaway–Sant'Anna** | **+0.0416** | 0.0263 | **no** (t = 1.58) |

The CS estimate is **roughly half** the TWFE estimate and not distinguishable from zero.
Verdict recorded in the output as *"TWFE-only (likely forbidden comparisons)"*.

Under staggered adoption with heterogeneous effects, TWFE uses already-treated counties as
controls for later-treated ones, and those comparisons can carry negative weights. A6 removes
them by construction. **The gap is the expected signature of that problem.**

This is the single most important result of the re-run and must not be buried: the headline
number is an artifact of the estimator at least in part. It does not mean the effect is zero —
CS is less precise (3 cohorts, 39 cells, unconditional parallel trends) — but the TWFE result
cannot be reported on its own.

Across all 7 outcomes: 1 TWFE-only, 1 CS-only, **5 null in both**.

### D-22 · A4 (dairy × FSIS) is NOT IDENTIFIED for five of seven outcomes
Within-county switchers in the 2017–2023 estimating sample:

| Outcome | N | dairy switchers | interaction switchers |
|---|---|---|---|
| Poor MH Days | 17,989 | 89 | 105 |
| Frequent Mental Distress | 17,989 | 89 | 105 |
| Deaths of Despair | 2,946 | **0** | 7 |
| Aggravated Assault | 7,279 | **0** | 19 |
| Assault, all severities | 7,279 | **0** | 19 |
| Violence index (partial) | 7,279 | **0** | 19 |
| NIBRS curated total | 7,279 | **0** | 19 |

**Zero dairy switchers** means the dairy main effect is fully absorbed by county FE; the
interaction is then identified off 7–19 counties. The coefficients printed for those five
outcomes (e.g. Aggravated Assault β = +8.87, SE = 9.78) are **not interpretable** and must not
be reported as nulls — a null requires power the design does not have here.

**DECIDED: A4 is reported for the two mental-health outcomes only**, flagged exploratory, and
the crime rows are suppressed with this reason stated. The crime outcomes end in 2021, leaving
5 years against an FSIS window that starts in 2017.

### D-23 · Three of four crime outcomes FAIL the event-study pre-trend test
A5, significant pre-period coefficients (p < 0.05, state-clustered):

| Outcome | sig. pre-period coefs |
|---|---|
| Poor MH Days | 0 — flat |
| Frequent Mental Distress | 0 — flat |
| Deaths of Despair | 0 — flat |
| **Aggravated Assault** | **2 — FAILS** |
| **Assault, all severities** | **3 — FAILS** |
| Violence index (partial) | 0 — flat |
| **NIBRS curated total** | **3 — FAILS** |

Counties that later gain a large dairy were already on a different crime trajectory
beforehand. **Those three outcomes cannot be read causally from the event study.** The mental
health outcomes pass.

Note this cuts the other way from D-14: the mental-health outcomes have clean pre-trends but
fail the estimator test (D-21); the crime outcomes fail the pre-trend test.

### D-24 · Collinearity is not a problem anywhere in Part A
119 specs audited. Max VIF **6.54** in levels (`children_in_poverty_per100k`), **2.39**
within-transformed. **Zero specs above 10, zero regressors absorbed by the FE.** The
within-transformed figure is the relevant one for A2. This closes the collinearity question
for the A-series: every instability found so far is about sample, estimator or pre-trends,
never about collinearity.

---

## CORRECTIONS TO THE 2026-09-28 PART A RUN (same day, before any reporting)

### D-25 · SUPERSEDES D-21 · the TWFE-vs-CS table was comparing two different treatments
The first version of the comparison put **A2 on T1** (`tr_e1_lg_bin`, contemporaneous
presence) against **A6 on the cohort definition** (`add_year` = first wave with a positive
change in the large-dairy count, which is binary and **absorbing**). Those are not the same
treatment:

| | T1 presence | CS cohort (`tr_e2_add_absorb`) |
|---|---|---|
| counties ever treated | 759 | 572 |
| switchers | 365 | 572 |
| disagreement | — | 10.6% of rows |

A gap between the two could not be attributed to the estimator, because the treatment moved
too. **The TWFE side is now estimated on `tr_e2_add_absorb`**, so the only thing that changes
between the columns is the estimator. `E2` is added to the grid for this purpose only.

### D-26 · CORRECTED RESULT — CS attenuates the estimate on 5 of 7 outcomes

| Outcome | TWFE β | CS ATT | CS/TWFE | verdict |
|---|---|---|---|---|
| Poor MH Days | 0.1043* | 0.0416 | 0.40 | TWFE-only |
| **Frequent Mental Distress** | 163.15* | **142.93*** | 0.88 | **survives CS** |
| Deaths of Despair | −0.255 | +0.027 | — | null in both |
| Aggravated Assault | 7.54* | 3.05 | 0.40 | TWFE-only |
| Assault, all severities | 35.05* | 14.02 | 0.40 | TWFE-only |
| Violence index (partial) | 8.71* | 6.56 | 0.75 | TWFE-only |
| NIBRS curated total | 36.39* | 17.09 | 0.47 | TWFE-only |

\* significant at 5%, state-clustered.

**Five of seven outcomes are significant under TWFE and lose significance under CS**, with the
CS estimate landing at roughly **40–50% of the TWFE estimate** in four of them. That ratio is
consistent across unrelated outcomes, which is the signature of a systematic estimator
problem (forbidden comparisons) rather than seven separate substantive findings.

**Only Frequent Mental Distress survives** — and its CS estimate rests on just 2 cohorts and
14 cells (the outcome starts in 2016), so it is the least well-powered of the seven. A lone
survivor on the thinnest data is not a result to lead with.

This replaces the earlier reading ("5 null in both"), which was an artifact of the treatment
mismatch in D-25.

### D-27 · T3 (per-capita) manufactures false identifying variation
Found while fixing A4. In the 2017–2021 window the **raw large-dairy count changes in 0
counties** — there is no ag-census wave transition inside it. Yet:

| Treatment | "switchers" 2017–2021 |
|---|---|
| T1 presence | 0 |
| T2 count | 0 |
| T4 log(1+count) | 0 |
| **T3 per 10k** | **603** |

T3 is count ÷ population. Population moves every year in 2,705 counties, so **T3 varies
within county even when no CAFO ever opens or closes.** A coefficient identified off that
variation is a population effect wearing a treatment's name.

**Consequence — NARROWER THAN FIRST WRITTEN (corrected same day).** Measured over the full
panel the contamination is negligible:

| | |
|---|---|
| corr(within-county T3, within-county 1/pop) | **+0.003** |
| counties whose CAFO count never changes | 2,048 of 2,733 (75%) |
| ...of those, counties where T3 still varies | **74** |
| share of all within-county T3 variance from those counties | **1.0%** |

The reason is that most constant-count counties have count = 0, so T3 = 0 and does not drift.
Only counties with a positive, unchanging count drift, and there are 74.

So this is **NOT a panel-wide problem**. It bites in exactly one situation: a window
containing **no ag-census wave transition**, where the raw count cannot change at all and so
100% of T3's apparent variation is population drift. That is precisely the A4 2017–2021 case.
Any short-window or sub-period analysis must therefore check the raw count; the full-panel
A1/A2/S1 results do not need revisiting on this ground.

**Fixed in A4:** the identification test now runs on `cafo_dairy_large` (the numerator), which
catches this for every functional form at once. A4 went from 13 "reportable" specs to **8** —
the five T3 crime/despair rows it had wrongly admitted are now correctly suppressed.

**Not an issue for** the full-panel A1/A2 grid, script4b's 840 regressions, or the 56
stability paths — all span multiple wave transitions, and the 1.0% figure above bounds the
contamination there.

### D-28 · A4 now runs all four treatments
Previously hard-coded to T1, which made it the only Part A model not estimated over the
decided treatment set. Now T1–T4 × 7 outcomes = 28 specs, of which **8 are identified** (four
treatments × the two mental-health outcomes). Deaths of Despair and all four crime outcomes
are suppressed: their year ranges end in 2020/2021, so no wave transition falls inside the
2017–2023 FSIS window.

---

### D-29 · ONE control set. `CONTROL_PRETREAT` dropped as a robustness arm — 2026-09-29
The step-8 set is now the only control set estimated. Reason: **the ten extra members of
`CONTROL_PRETREAT` are each excluded for a stated substantive reason.** Running that set as a
"robustness check" would present known colliders as a legitimate alternative and imply the
exclusions were arbitrary.

**The coverage argument does NOT carry, and should not be used.** On the headline sample
(Poor MH Days + treatment non-missing, 34,199 rows) the joint listwise cost is:

| set | rows kept |
|---|---|
| step-8 | 34,098 (**99.7%**) |
| pretreat-18 | 29,046 (84.9%) |

That is a **15% loss, not the 40% quoted earlier** — the 40% was a median across all seven
outcomes, dominated by the crime outcomes' own coverage. The individual coverage figures
previously cited (53.4% / 57.6% / 65.9%) are **full-panel 2000–2023 figures** and are
misleading for outcomes that begin in 2010 or later; on the MH sample these variables run
93–100%. Two of them cost zero rows.

#### Why each of the ten is out

**Excluded for cause (5):**

| Variable | Reason | Ref |
|---|---|---|
| `adult_smoking_per100k` | **COLLIDER** — responds to distress and to local economic shocks, so caused by both outcome and treatment. Also fails pre-trend (2 sig). | D-04 |
| `teen_births_per100k` | **COLLIDER** — same reasoning. Pooled over 7 years, so untestable anyway. | D-04, D-12 |
| `access_to_healthy_foods_per100k` | **MEASURE BREAK** — v030→v083 spliced at the 2013 release; two different measures in one series. | D-05 |
| `driving_alone_to_work_per100k` | No consistent series across the window (2011–2023 only). | — |
| `%_native_hawaiian/other_pacific_islander` | Negligible in non-metro counties; contributes no usable variation. | — |

**Excluded because pre-trends cannot be tested (5):**

| Variable | Why untestable | Row cost on MH sample |
|---|---|---|
| `some_college_per100k` | ACS 5-year rolling | 2,336 |
| `children_in_single-parent_households_per100k` | ACS 5-year rolling | 2,336 |
| `%_not_proficient_in_english_per100k` | ACS 5-year rolling | 2,336 |
| `adult_obesity_per100k` | BRFSS model-based, 3-year lag — smoothed | **0** |
| `uninsured_adults_per100k` | SAHIE, begins 2008 | **0** |

A rolling or model-smoothed series cannot show a differential pre-trend *regardless of the
truth*, so including it buys adjustment that cannot be validated. The ACS block additionally
shares one window — whichever enters first pays the whole 2,336-row cost and the other two
then look free.

#### Honest weak point
**`uninsured_adults_per100k` has the weakest exclusion case.** SAHIE is genuinely annual, it
begins in 2008 (before the 2010 outcome start, so testable within the window), and it costs
zero rows. Its exclusion rests on a mediator argument — CAFO employment plausibly changes
insurance coverage — which is an argument, not a demonstration. `adult_obesity_per100k` is
similar: the pre-trend objection is sound, but it is also plausibly a mediator, and the two
justifications should not be blurred.

**If challenged on the control set, these two are where the challenge will land.** Worth
running as a named sensitivity (step-8 + uninsured + obesity) rather than leaving it to a
referee. Not yet run.

### D-30 · `script4f` and `script4i` retired — 2026-09-29
**`script4i-cs-covariates.py` → MERGED into `script4a` A6.** The covariate-adjusted
Callaway–Sant'Anna arm now runs beside the unconditional one, on the same cells and the same
bootstrap draws, so the two differ only in the conditioning. The headline TWFE-vs-CS table now
uses the **adjusted** arm, which conditions on the same step-8 set as the TWFE side — the
comparison therefore isolates the estimator instead of confounding it with the control set.
Cells too thin to fit the adjustment fall back to unconditional and are counted in
`n_fallback_cells` (currently **0** everywhere).

**`script4f-core-models.py` → ARCHIVED, nothing salvaged.** It was already **broken**: it
unpacks `CORE_TREATMENTS` as 3-tuples, which raises `ValueError` against the current
str→str dict. Everything it did is replicated: TWFE → A2, event study → A5,
Callaway–Sant'Anna → A6. Its three control sets (19/26/9) are retired under D-29 and its
6-outcome, 3-treatment grid predates D-11 and D-17.

**One thing was lost and is recorded here rather than silently dropped:** 4f defined a
*different cohort per treatment* (`coh_T1` = first year of any presence, `coh_T2` = `add_year`,
`coh_T3` = `entry_year`). `script4a` uses `add_year` only. Whether the CS result is sensitive
to the cohort definition is a legitimate robustness question and is **not currently tested**.

**Downstream:** `script4g-slides.py` and `script4h-result-tables.py` still read
`tables_dir/script4f`. Both are stale and must be repointed when rebuilt.


---

### D-31 · Run 2 (2026-10-12): changes made on the Sept 30 feedback
Recorded retroactively; these changes were made in code on 10-12 without an entry here.
- **A3 (other-animal horse race) dropped.** `script4l-a3-horserace.py` retired to `z-archive/`.
- **Step-8 control set revised** in `script4_treatment`: `%_female` removed, the Asian / AIAN /
  NHPI shares combined into one, `%_black` added. The *pre-commitment logic of D-16 is unchanged*
  (the set is fixed before any coefficient is read), but the membership differs from the set
  that D-16 originally named, so the stability results in `script4b`/`4c` (run 09-23/09-27) were
  computed on the earlier membership.
- **A4** now saves SEs for every term (dairy, FSIS, interaction).
- **A5** also run (a) with 2000 log population × year and (b) with log population as the outcome
  (depopulation placebo).
- **A6** event-time ATTs carry bootstrap SEs (aggregated inside each draw).
- **Added:** A7 LP-DiD (`estimators.lpdid`), Conley spatial SEs on the A2 grid, and
  `script4r-fsis-treatment.py` (slaughterhouses as a treatment in their own right).

### D-32 · Team feedback (2026-10-15): what is REPORTED vs what is ESTIMATED
The team asked for **T1, T3, T4 and E2** (T2 removed), **county FE only**, **state-clustered SEs
only**, **TWFE and Callaway–Sant'Anna (covariate-adjusted) only**, and **no LP-DiD**.

**Decision: these are choices about reporting, not about estimation.** The estimation scripts
still compute everything (T2, A1 state-FE, county and heteroskedastic SEs, CS-unconditional,
LP-DiD, Conley) so no work is lost and the full set stays available as robustness. Implemented
as `REPORT_TREATMENTS / REPORT_FE / REPORT_SE / REPORT_CS_ARM` in `script4_treatment`, read by:
- `script4n` — main tables show the reported set; the rest are written as separate tables
  collected in `_BODY_robustness.tex` (A2 with T2, A1, A4 with T2, both CS arms, LP-DiD, Conley).
- `script4q` — main frames show the reported set; A1 frames go to `BEAMER_FRAMES_robustness.tex`.
- `script4a` — figure F1 shows the reported treatments only.

Treatment IDs are **not renumbered** (T3 and T4 keep their names, so D-27 and earlier output still
refer to the same variable). **Naming:** the team's "E2 (T5)" is `tr_e2_add_absorb`; it is called
**E2** everywhere, because T5 was previously the dropped log-per-capita treatment (2026-09-23).
`script4b`/`4c` are deliberately left as they were (robustness; the step-8 pre-commitment stands).
**T3 is kept at the team's request despite D-27** (per-capita variation partly reflects
population-denominator drift); it should carry that caveat wherever it appears.

### D-33 · CS-unconditional vs CS-adjusted: report the adjusted arm
Answer to the team's question. **Unconditional:** ATT(g,t) is a raw difference of long
differences; parallel trends assumed with no covariates. **Adjusted:** outcome regression on the
step-8 set measured at base period g−1, fitted on clean controls only; parallel trends assumed
*conditional on X*, the same assumption TWFE makes, so the TWFE-vs-CS comparison isolates the
estimator. On the 10-12 run: **0 fallback cells** and the **same sign on all 7 outcomes**;
magnitudes differ (overall ATT, unconditional → adjusted: Poor MH Days 0.042 → 0.039; Frequent
Mental Distress 143 → 51; Deaths of Despair 0.027 → 0.057; Aggravated Assault 3.05 → 2.56;
Assault, all severities 14.0 → 20.6; Violence index 6.6 → 7.3; NIBRS total 17.1 → 26.6).

### D-34 · Missing CAFO count = zero from 2002 on; 2000–01 stay missing — 2026-10-16
**Rule** (in `script4_treatment.load_panel`, no panel rebuild): a blank `cafo_{species}_*` count in
a census-covered year is set to 0. **Why blanks arise:** the panel is every rural county-year
(65,828 rows, not subset to dairy). `script1b` zero-fills a census-year county only when it has no
CAFO row for *any* species (lines 373–383), so a county with a hog row but no dairy row keeps blank
dairy columns. The ag census is exhaustive, so no dairy row means no dairy operations.

**Evidence that these are zeros, not suppression** (dairy, all five census waves):
- The raw NASS operations pulls contain **no "(D)" values at all** (601,720 rows); suppression of
  operations is by omission. "(D)" appears only in head counts.
- A head-count row (number or "(D)") exists for **100%** of counties with milk-cow operations in
  every wave, including 2022 (2,362 of 2,362). The converse holds for the blanks: **none of the
  947 blank census cells has any NASS milk-cow record** (no total, no head-count row).
- 2002–2017: in all 10,110 dairy rows the size bins sum **exactly** to NASS's county total.
- 2022: see D-35.

**2000–01 are not filled.** No census is taken in them in this pipeline; a blank is an unmeasured
year. Filling them would code **158 counties** (266 county-years in the Deaths of Despair sample)
that had a large dairy in 2002 as untreated, then switching on in 2002 — spurious timing.

**Effect on the data:** treatment non-missing rows 55,719 → 60,170; E2-treated counties
572 → 576 (cohorts 2007/2012/2017/2022: 284/142/101/45 → 288/141/102/45); estimating samples
+3.3% to +5.5%; 17,804 `cafo_dairy_*` values recoded. **Effect on the headline A2 estimates**
(T1, T3, T4, E2 × 7 outcomes, county + year FE, step-8, state SE): **0 sign changes, 2 of 28
significance changes** (E2 on Poor MH Days p<0.01 → p<0.05; E2 on Violence index p<0.05 →
p<0.10); T1/T3/T4 coefficients move typically 1–6%, E2 by 1–10%. CS not re-estimated.
**Open:** the logic belongs in `script1b` (species-aware) and should move there at the next panel
rebuild; until then `load_panel` is the single place it is applied.

### D-35 · 2022 verification: NASS totals were never pulled, not unavailable — 2026-10-16
The 2022 raw cache (dated 2026-05-06) predates `script0b`'s total-operations request, and
`script0b` never re-pulls a cached year, so 2022 had bins but no cattle TOTAL rows and was labelled
"dark". Re-pulled with `qa-scripts/pull_2022_cattle_totals.py` (file
`2026-10-16_usda_2022_api_cafo_ops_totals_raw.csv`, 16,432 rows). **Result, 2022:** milk cows
2,362 of 2,362 counties bins = total (national 36,024 on both sides); beef cows 3,053 of 3,053
(622,162); incl. calves 3,060 of 3,060 (732,123); **0 suppressed, 0 bins above total, 0 total-only,
0 bins-only.** A head-count cross-check (bins vs numeric 2022 head counts, 1,512 counties) finds
0 over-counts and 0 under-counts, as in every earlier wave. **2022 is therefore verified and
needs no exception in D-34.** *Not yet done:* `script0b` does not read the new totals file, so a
re-run would label 2022 "dark" again, and `VERIFIED_WAVES` in `script4_treatment` still lists
2002–2017 only.

### D-36 · Species set for overlap and per-species runs — 2026-10-16
**Dairy, beef, hogs, chickens.** `cattle` is excluded as a species: the USDA cattle inventory
contains the dairy and beef herds (dairy without cattle: 0–2 county-years per wave; beef without
cattle: 0), so counting it makes nearly every dairy county look multi-species. `script4s` writes
the descriptives. **Results (census waves 2002–2022, 2002 on, blank = 0):** 27–29% of rural
counties have large CAFOs of 2+ species; of large-dairy counties, 43–48% also have hogs, 41–46%
beef, 14–21% chickens; 633 of 759 counties that ever have a large dairy share a wave with another
species; 884 counties have 2+ species in at least two waves. Dairy alone is the single most common
dairy-containing cell (≈110–150 counties per wave); the 3- and 4-species cells are thin.
**Caveat for E2 by species:** 70–75% of positive changes in large-dairy counts occur while the
county's total dairy operations fall (consolidation); check hogs and beef before interpreting E2.
**Caveat on power:** chickens have large operations in ≈6.5% of county-years; expect CS to be
underpowered there. Joint modelling of species has not been decided.

### D-37 · Credentials removed from code — 2026-10-16
A NASS API key was hard-coded in `script0b` (line 56), `qa-scripts/qa_2017_dta_vs_api.py`,
`z-archive/script0p-usda-2022-api-only.py` and, briefly, `qa-scripts/pull_2022_cattle_totals.py`.
It is in git history (commit `6bdb09f`) and in the pushed commit `c1fa465`. **Removed from the
working files; all four now read `USDA_NASS_API_KEY` from the environment only.** Removing it
from the files does not remove it from history: **the key must be revoked and replaced at NASS.**

### D-38 · Repository hygiene — 2026-10-17
`__pycache__/` (three Python versions), `.spyproject/` and `.DS_Store` files (71 files) were
tracked and are now untracked; Python and editor rules added to `.gitignore`. Remaining packaging
gaps for a public release are listed in `2026-10-17_MODEL-RUN-STATE.md`.

### D-39 · OPEN (drafted 2026-10-22, awaiting decision) · Multiple-testing family and correction
**Finding** (`2026-10-22_ANALYSIS-ASSESSMENT.md`): of the 28 reported A2 cells (T1, T3, T4, E2 × 7
outcomes) 13 are significant at 5% unadjusted, 3 survive Holm or Benjamini–Hochberg; for E2 across the
7 outcomes none survive. **Not decided:** which family is confirmatory. Options to choose from before any
new result is read: (a) Poor MH Days only, with E2 as the single primary test (D-16 already names it the
headline); (b) the two mental-health outcomes × the reported treatments; (c) all 28 cells. Correction
candidates: Holm (family-wise error), Romano–Wolf step-down using the joint bootstrap draws, Benjamini–Hochberg
for exploratory families. Report unadjusted and adjusted values. Plan: `2026-10-22_PLAN_inference-upgrade.md`.

### D-40 · OPEN (drafted 2026-10-22, awaiting decision) · Timing resolution of treatment
Treatment can change only at the five census waves, so the true change falls in the five years before the
wave at which it is observed (METHODS.md §2). Event-time paths therefore mix real timing with measurement
error, and annual leads and lags overstate the timing precision. **Not decided:** whether to add a design on
a census-wave panel (outcomes averaged in the windows around 2002/07/12/17/22) as a robustness arm, and
whether annual event-study figures carry a note on this. No result depends on the choice yet.

### D-41 · 2026-10-22 · A22 (event-study control group) is answered by the CS arm; write it down
`did_cs.py` uses never-treated and not-yet-treated counties as comparisons at `max(t, g)` and never an
already-treated county (METHODS.md §2). The implicit comparison group of the TWFE event
study is not stated anywhere (this is A22). A22 should be closed by stating this in the paper's design section and by reporting the CS
event study as the primary event-study exhibit; the TWFE event study stays as a robustness exhibit
(it is still estimated, not deleted). A21 (full-panel vs 2010–2020 window) remains open.

### D-42 · DECIDED 2026-10-22 · Confirmatory family for multiple testing (closes D-39)
**Decision (the owner's):** the confirmatory family is the **two mental-health outcomes (Poor MH Days, Frequent
Mental Distress) across the four reported treatments T1, T3, T4 and E2: 8 tests.** Correction: Holm (family-wise
error) and Romano–Wolf step-down on the joint wild-bootstrap draws; unadjusted and adjusted values reported side
by side. All 28 reported A2 cells form an **exploratory** family, corrected with Benjamini–Hochberg and labelled
exploratory. The crime outcomes and deaths of despair are not part of the confirmatory claim (D-14, D-23).
**Made after seeing results, and recorded as such:** when the choice was made the 2026-10-22 review had already
shown this family's unadjusted count (6 of 8 cells significant at 5%) and its Holm and Benjamini–Hochberg count
(1 of 8). The family was not fixed blind; any adjusted value reported later for it carries that label.
Implementation: `2026-10-22_PLAN_inference-upgrade.md`, Part 3.

### D-43 · DECIDED 2026-10-22 · Romano–Wolf is applied within each outcome; Holm across the family (refines D-42)
**Decision (the owner's):** Romano–Wolf step-down needs every test to use the same resampling weights in each bootstrap
draw. The four treatments of one outcome share one sample and the same 48 states, so one seed gives them identical state
weights (check: the bootstrap t-statistics correlate at 0.42 or more across treatments). The two outcomes have different
samples, so no valid joint draws exist across outcomes. Therefore: **Romano–Wolf within each outcome (4 tests each)**, and
**Holm over all 8 wild-bootstrap p-values** for the cross-outcome family-wise correction. A single Romano–Wolf over all 8 is
not reported. Implemented in `script4z-inference.py` (9,999 draws, Webb weights, seed 20261022); the script first reproduces
script4a's A2 coefficient and p-value for each cell and stops if they differ. Exploratory family F2 (28 cells):
Benjamini–Hochberg on the analytic state-clustered p-values (a wild bootstrap of all 28 was not run).

### D-44 · DECIDED 2026-10-23 · Callaway–Sant'Anna covariates are measured at the earlier of year t and the base year g−1
**Decision (the owner's):** adopt the convention of R `did`, `csdid` and `moderndid`. In the covariate-adjusted arm the covariates for
cell (g,t) are measured in year `min(t, g−1)`: the base year g−1 for every post-treatment cell (unchanged), year t for a pre-treatment cell
before the base year (changed; previously g−1). **Alternative not taken:** keep covariates at g−1 for every cell.
**Reason:** `did_cs.py` then reproduces all three independent implementations in every cell, not only the post-treatment ones.
**Evidence:** `2026-10-23_CS-CROSSCHECK.md`. Before the change 17 of 44 pre-treatment cells differed (largest gap 0.0116 days for
Poor MH Days; 16.2 for Frequent Mental Distress); after it every cell agrees (largest gap 0.0) for both mental-health outcomes.
**Pre-committed or after seeing results:** after seeing the size of the pre-period differences, before seeing the re-estimated
pre-period results. **Effect:** pre-treatment cells and their significance change in `script4a` (A6, both CS arms), `script4u` (CS arms) and
`script4x` (CS rows); post-treatment cells and the overall ATT are unchanged by construction and were checked against the
2026-10-22 outputs. **Code:** `did_cs.att_gt_table` and `base_period_covariates`; `METHODS.md` section 2 updated.

### D-45 · DECIDED 2026-10-23 · Rule for calling the mental-health result estimator-robust (fixed before the triangulation is run)
**Decision (the owner's):** in `script4za-estimator-triangulation.py` an estimator **agrees** with Callaway–Sant'Anna (covariate-adjusted, same sample) if its
overall post-treatment ATT has the same sign and lies inside CS's 95% interval. The Poor MH Days result (first expansion) is called **estimator-robust** if at
least **4 of the 5** other estimators agree (LP-DiD, Sun–Abraham, imputation, de Chaisemartin–D'Haultfœuille, stacked DiD). **Alternatives not taken:** all 5 must
agree (too fragile with 2–3 cohorts); a magnitude test (ratio 0.5 to 2), which a wide CS interval makes almost impossible to fail. All results are reported whatever the verdict.
**Pre-committed or after seeing results:** before any of the five estimators has been run on the real panel; the CS result for Poor MH Days (0.0372, not significant) was
already known when the rule was set. Plan: `2026-10-22_PLAN_estimator-triangulation.md`.

### D-46 · DECIDED 2026-10-24 (scaling by the owner; T3/T4 details drafted for the owner's confirmation) · Presence and dose treatments under de Chaisemartin–D'Haultfœuille (dCDH)
**Decision:** treatments T1 (presence 0/1), T3 (per 10,000 residents) and T4 (log(1 + count)) are run through dCDH (R `DIDmultiplegtDYN`, 9 effects, no placebos,
state-clustered) **not normalized**: the estimate is the effect of the treatment change counties actually experienced, not per unit of treatment. The average first
change (T1 +0.42, T3 +0.17, T4 +0.27) is stored beside each estimate (`note` column) and printed on the figure. **Alternative not taken:** `normalized`, which divides
by cumulative treatment over the horizon (per unit of treatment-period) and so shrinks with the horizon; chosen against because it is not comparable to the TWFE coefficient except at event time 0 (pinned in `tests/test_triangulation.py`).
**Drafted details (made after a Poor MH Days trial run, not pre-committed):** T3 uses the package's default matching on the period-one value (`continuous=1` rejected: standard error ten times larger, 0.420 vs 0.040);
T4 runs **without** time-varying controls for every outcome because with them 6 of 7 outcomes fail inside the package ("all observations are fixed-effects singletons"), T1 and T3 keep the 8 controls; T2 (raw count) stays out (D-32).
**Caveats:** T3 variation is partly population drift (D-27); T3 rests on about 665 switchers; the units of T1, T3, T4, E2, E3 differ, so line-to-line magnitudes are not comparable.
**Evidence:** `script4zc-treatment-by-model.py` (140 rows, 0 failures; TWFE rows re-fitted and equal to the script4a A2 grid to 1e-6); figures `script4zd-treatment-by-model-figures.py`.
**Code:** `estimators.dcdh_estimates(normalized=...)`, `script4za-dcdh.R`.

### D-47 · DECIDED 2026-10-24 · Permutation inference for the cohort designs: within-state shuffle of first-treated years
**Decision (plan approved by the owner):** for first expansion (E2, primary) and first entry (E3, secondary), all 7 outcomes, the TWFE coefficient (county + year FE, the 8 controls) is
compared with its distribution when each state's first-treated years are shuffled among that state's ever-treated counties (never-treated counties fixed, dummy rebuilt as year >= cohort).
2,000 draws, seed 20261023 (one stream per cell), p = (1 + #{|null| >= |observed|}) / (1 + 2,000); the clustered t-statistic gives a secondary p-value. Samples are `designs.usable_sample` (moved out of `script4za`; its outputs were re-run and are identical).
**Scope limits fixed in advance:** T1, T3, T4 are not cohort-based and are not permuted; the test is of the sharp null, conditional on each state's number of treated counties and its set of cohorts; only states with two or more different cohorts can be shuffled (29 of 40 for E2 Poor MH Days);
no new multiple-testing family: p-values are reported unadjusted beside the wild-bootstrap p-values of `script4z` (D-42, D-43). **Alternatives not taken:** shuffling across all counties (ignores state shocks); permuting event-study pre-trends (placebo question set aside by the owner).
**Pre-committed or after seeing results:** design fixed before any permuted result was read; the wild-bootstrap results for E2 were already known. **Code and tests:** `script4ze-permutation-inference.py`, `script4zf-permutation-figure.py`, `tests/test_inference_tools.py` (size 0.10 over 40 null panels, power 1.00 over 15, reproducible).
