# Control variable decision register
**Living document — every include/exclude decision, with its justification.**
Started 2026-09-21. Update whenever a control enters or leaves a specification.

Purpose: each inclusion and exclusion must be defensible to a referee. This file is
the record. Cite the decision ID in the paper's data appendix.

---

## CURRENTLY INCLUDED — `CONTROL_PRETREAT` (19)

| # | Variable | v-code | Why included | Risk flags |
|---|---|---|---|---|
| 1 | `%_65_and_older` | v053 | Age structure predicts both agricultural land use and mental-health reporting. Predetermined. | pre-trend flat |
| 2 | `%_asian` | v081 | Predetermined demographic composition. | near-zero in non-metro; adds little |
| 3 | `%_below_18_years_of_age` | v052 | Age structure. Predetermined. | pre-trend flat |
| 4 | `%_female` | v057 | Predetermined. | pre-trend flat |
| 5 | `%_hispanic` | v056 | Strong confounder for agricultural county composition. | labelled mediator on labour-recruitment grounds, but pre-trend is FLAT (0.006 SD) — the mediation concern is not supported by the data |
| 6 | `%_native_hawaiian/other_pacific_islander` | v080 | Predetermined. | negligible in non-metro; adds little |
| 7 | `%_not_proficient_in_english_per100k` | v059 | Tracks immigrant labour composition. | labelled mediator; pre-trend FLAT |
| 8 | `%_rural` | v058 | Within-sample rurality gradient. | **FLAGGED FOR REMOVAL** — see D-03 |
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

### D-03 · FLAGGED FOR REMOVAL — `%_rural`  (decision pending)
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
