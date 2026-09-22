# BLOCKING: CHR variables are dated by RELEASE year, not data year
**2026-09-21** · affects every CHR-sourced variable in the panel — outcomes and controls

---

## The defect

`script0c-health-raw.py:186` assigns `year` from the **filename**:
`analytic_data2020.csv` → `year = 2020`.

The CSV's own column confirms what that is: it is literally named **`Release Year`**.
CHR names it that because it is *not* the year the data describes.

## Confirmed lags — from CHR's own release code

Source: `github.com/countyhealthrankings/county_health_measure_calculations`,
`calculations/` folder, files titled **R2025** (the 2025 release).

| Panel variable | v-code | CHR source file | Data year | Lag |
|---|---|---|---|---|
| `median_household_income` | v063 | `est23all.xls` (Census SAIPE) | 2023 | **−2** |
| `children_in_poverty` | v024 | `est23all.xls` (Census SAIPE) | 2023 | **−2** |
| `unemployment_per100k` | v023 | `laucnty23.xlsx` (BLS LAUS) | 2023 | **−2** |
| `poor_mental_health_days` | v042 | `Estimates4CHRR_2022_Final.xlsx` (PLACES/BRFSS) | 2022 | **−3** |

**The lags differ by measure.** The main outcome is lagged 3; the economic controls 2.

## Independently confirmed empirically

Two national benchmarks, two variables, both point to the same lag for the controls:

- **median household income** vs national SAIPE — YoY growth correlation by lag:
  0 → +0.258, 1 → +0.004, **2 → +0.634**, 3 → +0.260
- **unemployment** vs national BLS, using the 2020 COVID spike as a marker —
  correlation by lag: 0 → +0.455, 1 → +0.757, **2 → +0.968**, 3 → +0.696.
  The panel's 2022 unemployment (6.66%) *is* the 2020 national spike (8.1%).

## What a panel row actually contains

The row labelled `year = 2020` holds:
- income, unemployment, child poverty → **2018**
- poor mental health days → **2017**
- CAFO counts → **2017 ag census**, forward-filled (up to 5 years stale)

Three reference periods in one row. This misaligns outcome-to-control AND
outcome-to-treatment.

## Two further data-integrity problems found in the same scan

### 1. Measure definition breaks — variables spliced from different measures
| Variable | Codes | Switch |
|---|---|---|
| `access_to_healthy_foods_per100k` | v030 → v083 | at release 2013 |
| `adult_smoking_per100k` | v009 → v095 → v009 | 2011 only, then reverts |

v030 and v083 are **different measures**, merged into one panel column. The 2010–2012
values and the 2013+ values are not the same quantity. Note `adult_smoking` also
FAILED the covariate pre-trend test — a definitional break would produce exactly that.

### 2. Measures that do not update annually
Share of county-year values *identical* to the county's prior year:

| Variable | % identical | Reading |
|---|---|---|
| `%_rural` | **93.7%** | effectively static (decennial, carried forward) |
| `access_to_healthy_foods` | 62.8% | multi-year vintage |
| **`violent_crime`** (outcome) | **46.7%** | **pooled over multiple years** |
| `poor_mental_health_days` (outcome) | 18.6% | partial reuse |
| `adult_smoking` | 16.8% | partial reuse |
| the other 14 controls | 0–9% | annual |

`violent_crime` has nearly half its year-over-year observations literally identical,
which mechanically blurs any event-study timing estimated on it.

## Coverage notes from the same scan
- `frequent_mental_distress` (v145) exists only from release **2016** — hence 2 cohorts in CS.
- `violent_crime` (v043) is **absent from the 2023 release** — measure discontinued.
- The demographic block (v052–v081) starts at release **2011**, which is what imposes the
  2011 floor on every estimating sample — not `median_household_income`, which costs 0–1 rows.

## Impact
Every estimate currently in the deck is affected. The event study is affected most:
t = 0 is not t = 0.

## Fix
1. Pull each measure's documented data years from CHR's `calculations/` notebooks
   (organised by v-code) for **every** release year we use, not just R2025 — the lag
   need not be constant across releases.
2. Re-date the panel to data-year, per measure.
3. Decide how to handle measures whose reference period spans several years
   (`violent_crime`, `%_rural`, `access_to_healthy_foods`) — they cannot be given a
   single year.
4. Split `access_to_healthy_foods` at the v030/v083 break, or drop pre-2013.
5. Re-run everything.

## Status
NOT a refinement. This is a correctness issue and outranks the open SE-choice and
collider questions. For the current presentation it belongs in limitations.
