# Large-Scale Livestock, Slaughterhouse Work, and Mental Health in the US

## Purpose
This project combines US health, crime, population, and agriculture data into one large, unbalanced county-year panel covering roughly 2000–2023 (source-dependent). The goal is causal analysis between large-scale livestock operations (farms, slaughter and processing facilities) and mental health outcomes among US agriculture workers, using a repeated cross-section at the county (FIPS) level.

---

## Requirements

- **Python 3.13** (results produced on 3.13.5; macOS). Install the pinned packages with either
  `pip install -r requirements.txt` or `conda env create -f environment.yml` (the conda file installs Python and then the same pinned `requirements.txt`).
  Key packages: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, **`pyfixest`** (all estimation), `statsmodels` and `wildboottest` (multiple-testing corrections and wild cluster bootstrap), `requests`, `openpyxl`, `pyarrow`; optional `plotly` + `kaleido` for maps.
- **R 4.x** only for `script4w-honestdid.R` (Rambachan–Roth sensitivity analysis): `install.packages("HonestDiD")` (see `r-requirements.txt`).
- **LaTeX (`pdflatex`)** only for compiling the table and slide fragments written by `script4n`, `script4p` and `script4q`; without it the `.tex` files are still written.
- Shared helpers: `packages.py` (imports and directory definitions) and `functions.py` (FIPS normalization, CSV fallback readers, USDA NASS API wrappers, `latest_file_glob()`, `normalize_panel_key()`, `log_per10k()`, `load_county_geojson()`).
- **Credentials are read from the environment only, never from code:** `USDA_NASS_API_KEY` (`script0b`, `qa-scripts/pull_2022_cattle_totals.py`) and `HUD_API_TOKEN` (`script0e` / `fsis-scripts/`). Both are free to request from the agencies.
- `script0c` (Part 2) downloads SAIPE, BLS LAUS, SAHIE and Census PEP directly (no key); the BLS host `download.bls.gov` must be reachable.

### Hardcoded paths that must change per user
| Script | Variable | What it points to |
|--------|----------|-------------------|
| `packages.py` | `db_base = os.path.expanduser("~/Dropbox/Mental")` | Dropbox root — all other paths derive from this |

**Fix `db_base` in `packages.py` first.** All directory paths in all other scripts are derived from `db_data = db_base/Data` and will resolve correctly once the root is set.

---

## Directory Layout

### Repository (code only; data lives on Dropbox)

```
mentalhealth/
├── script0a … script0f-*.py   # Stage 0 — data in: ingestion, cleaning, and the panel build (script0f)
├── script1a … script1d-*.py   # Stage 1 — QA, tables, figures, maps (descriptive, before analysis)
├── first-stage-testing/       # first-phase pilots and test versions, kept for provenance, NOT run (see its README)
├── script4*.py                # Stage 4 — design-based estimates, tables, slides
├── functions.py, packages.py  # shared helpers; db_base path lives in packages.py
├── did_cs.py, estimators.py   # shared estimator code (Callaway–Sant'Anna; legacy LP-DiD / Conley, not reported)
├── requirements.txt, environment.yml, r-requirements.txt   # pinned environment
├── fsis-scripts/              # FSIS sub-scripts, called by script0e
├── qa-scripts/                # one-off diagnostics (not part of the run order)
├── code-audit/                # internal documentation: decision logs and model status
│   └── plans/                 #   dated notes; see "Internal documentation" below
├── test-dfs/                  # small QA extracts
└── z-archive/                 # superseded scripts, kept for reference, not run
```

### Dropbox data tree (`db_base/Data`)

```
Data/
├── raw/
│   ├── cdc/          # CDC WONDER deaths-of-despair annual CSVs
│   │                 #   pattern: cty-level-deathsofdespair-YYYY.csv
│   ├── crime/
│   │   └── total-v1/ # FBI UCR annual incident CSVs (~7 GB combined in memory)
│   ├── fips/         # FIPS crosswalk .txt files (foruse_*.txt)
│   ├── mental/       # County Health Rankings annual CSVs
│   ├── nchs/         # NCHSurb-rural-codes.csv (single file)
│   ├── population/   # US Census population CSVs (4 vintage files)
│   ├── usda/         # USDA NASS fallback .dta files (used if API fails)
│   └── fsis/         # USDA FSIS FOIA inspection CSVs
│
├── clean/            # Panel-ready inputs only (~8 files per run)
│   └── diagnostic/   # Intermediates + QA CSVs from all script0* runs
│
├── merged/           # Panel outputs from script0f
│
└── output/
    ├── figs/         # Figures from script1* and script4* (one subfolder per script)
    ├── maps/         # HTML + PNG maps from script1d
    └── tables/       # CSV outputs from script1* and script4* (one subfolder per script)
```

---

## Script Execution Order

### Stage 0 — Data in: ingestion, cleaning, panel build
Each script reads from `Data/raw/` and writes panel-ready CSVs to `Data/clean/`. Intermediates and QA files go to `Data/clean/diagnostic/`.

| Script | Purpose | Key output (in `Data/clean/`) |
|--------|---------|-------------------------------|
| `script0a-identifiers.py` | County identifiers and population: FIPS crosswalk, Census population (1990–2024; a measurement, kept here because it arrives with the county key) and the NCHS urban-rural class expanded to annual | `*_fips_full.csv`, `*_population_full.csv`, `*-rural-key.csv` |
| `script0b-usda-raw.py` | USDA NASS CAFO operations by size — **census years only** (2002, 2007, 2012, 2017, 2022) via API | `*_cafo_ops_by_size_compact.csv` |
| `script0c-health-controls.py` | County Health Rankings MH survey, CDC deaths of despair (Part 1) and annual, correctly-dated county controls from SAIPE, BLS LAUS, SAHIE and Census PEP (Part 2), which replace 10 CHR controls whose release-year dating misaligned them with the data year (see `code-audit/plans/2026-09-21_CHR-YEAR-MISALIGNMENT.md`). Flags `--skip-download`, `--only health\|controls` | `*_mentalhealthrank_full.csv`, `*_cdc_county_year_deathsofdespair.csv`, `*_annual_controls_county_year.csv` |
| `script0d-crime-raw.py` | FBI UCR violent crime, collapsed to FIPS-year | `*_crime_fips_level_final.csv` |
| `script0e-fsis-clean-all.py` | FSIS slaughterhouse pipeline orchestrator (runs `fsis-scripts/` sub-scripts via `subprocess`; requires `HUD_API_TOKEN`) | `*_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip.csv` |
| `script0f-generate-panel.py` | **Master panel builder** (the only script that writes the panel) — builds the rural county-year panel from the newest clean file of each source, forward-fills CAFO census data to all years, applies `non_large_metro == 1` filter, derives crude rate columns, CHR count imputation, rate standardization, column cleanup | `Data/merged/*_panel*.csv` (in `Data/merged/`, not `clean/`) |

**`fsis-scripts/` sub-scripts** (run in this order by `script0e`; each fills blanks only and never overwrites an existing `fips_code`):
`script0e-fsis-slaughterhouses` (raw FOIA → establishment-year panel) → `0h` (size/type panel) → `0i` (HUD ZIP→county API fill) → `0j` (year-aware refill) → `0k` (HUD bulk ZIP-year fill) → `0l-fsis-manual-and-deterministic-fill` (one pass, in this order: manual ZIP→FIPS, deterministic completion [ZIP, CT, city+state, establishment history], manual county-label→FIPS; a missing manual template skips its step). It writes the interim and county files once; the earlier three-script chain overwrote one file name three times (D-48).

### Stage 1 — QA and descriptive work (before analysis; none of these writes the panel)

| Script | Purpose | Output location |
|--------|---------|-----------------|
| `script1a-panel-qa.py` | Checks that can fail: key integrity and source duplication, variable inventory and coverage, CDC crude-rate sense check, correlation tables and FSIS column QA, missingness, county coverage check against the rural key, CAFO crosscheck, QA memo. `--only <section>` | `Data/output/tables/script1a/`, `Data/output/figs/script1a/` |
| `script1b-tables.py` | Numeric tables: summary statistics, CAFO concentration (HHI, Gini, entry/exit), CAFO and FSIS size-bin summaries, species overlap, the master variable register (xlsx + csv codebook) | `Data/output/tables/script1b/`, `Data/output/codebooks/` |
| `script1c-figures.py` | All non-map graphics: binned scatters, correlations, distributions, CAFO composition and top counties, threshold and presence, ridgelines, trend sweeps, state trends and dots, species overlap, raw trends and event-time paths. Figures use `figstyle.py` (183 mm, 600 dpi PNG + vector PDF). Each outcome is drawn over its own window (`designs.OUTCOME_WINDOWS`: Poor MH Days 2010–2023, Frequent Mental Distress 2016–2023, deaths of despair and crime 2002–2020; descriptive stage only) | `Data/output/figs/script1c/`, `Data/output/tables/script1c/` |
| `script1d-maps.py` | All maps (Plotly, HTML + PNG): FSIS density, outcome values, first differences, CAFO by animal and size, and one family per treatment (entry per wave, first-expansion cohort, large operations, species, the Poor MH Days estimation sample, sample coverage, FSIS entry, change between census waves) | `Data/output/maps/script1d/` |

The Hispanic-share and health explorer (`script2e-hispanic-chr-explorer.py`) is archived in `z-archive/`. Old stage-2 outputs stay in their original folders under `Data/output/`; D-49 records the consolidation.

### Stage 3 — Pilot (moved)
The ridge/LASSO pilot (`script3-ridge.py`) and the first single-file model test (`script4-model-test.py`) now live in `first-stage-testing/` as `firststage_*.py`. They were the first phase of development, are not part of the pipeline, and are not used for any reported result. ML variable selection remains specified-only (B-series in `code-audit/plans/2026-09-27_MODEL-NAMING.md`).

### Stage 4 — Part A: design-based estimates

**What is reported.** Per the team's 2026-10-15 feedback the reported set is treatments **T1** (large-dairy presence), **T3** (per 10,000 residents), **T4** (log(1 + count)) and **E2** (first expansion: 0 until the first census wave at which the large-dairy count rises, 1 from then on), **county + year fixed effects**, **standard errors clustered by state**, and the estimators **TWFE** and **Callaway–Sant'Anna covariate-adjusted** (CS). These choices live in `REPORT_*` in `script4_treatment.py` and only decide what the exhibit scripts show: the estimation scripts still compute the other specifications (T2 count, pooled state-FE A1, unconditional CS, LP-DiD), which appear only in the robustness exhibits. The **control set** is the 8 full-coverage controls `COVARIATE_ORDER[:8]`, pre-committed before any result was read (decision D-16, revised D-31). Every `script4*` file takes its panel, treatments and controls from `script4_treatment.py`, so each variable is defined in exactly one place. Outcomes: 7 (3 mental-health / mortality, 4 crime).

| Script | Role |
|--------|------|
| `script4_treatment.py` | **Shared module** — treatment, sample, outcome and control definitions; the missing-count-is-zero rule (from 2002) is applied in `load_panel` (importable; not run) |
| `script4a-design-estimates.py` | A2 TWFE (headline), A4 dairy × FSIS, A5 event study, A6 Callaway–Sant'Anna; also A1 pooled state-FE and A7 LP-DiD (computed, not reported) |
| `script4b-…`, `script4c-…` | Earlier run of adding controls in a fixed order and its summary; **superseded by `script4y`**, kept unchanged as the record of the 09-23 run |
| `script4d-balance-panel.py` | Balance and detectable-effect tables |
| `script4j-control-diagnostics.py` | Coverage, causal role and pre-trends for each control |
| `script4r-fsis-treatment.py` | Slaughterhouses as the treatment (R0–R3; R4 LP-DiD computed, not reported) |
| `script4t-pretrend-leads.py` | Outcome pre-trends (leads) for T1, T3, T4, E2 and first entry |
| `script4u-did-by-treatment.py` | TWFE, event study and CS for first expansion and **first entry** (0 → 1+ large dairies, clean controls) with the pre-trend adjustments: county trends, state × year FE, baseline characteristics × time polynomial, past outcomes in CS |
| `script4w-honestdid-export.py`, `script4w-honestdid.R` | Rambachan–Roth sensitivity analysis on the TWFE event studies (Python writes inputs, R runs `HonestDiD`) |
| `script4x-species-forest.py` | Per-species TWFE and CS estimates (large and all-sizes counts) and the forest plots |
| `script4y-control-set-selection.py` | **How the control set was chosen:** cumulative add-one-in, each control added alone, take-one-out of the final 8, take-one-out of all 14, and one selection table. Documents the pre-committed set; does not choose it |
| `script4n-regression-tables.py`, `script4p-full-appendix.py` | LaTeX tables and combined appendix (no estimation): reported exhibits plus a separate robustness set |
| `script4q-beamer-frames.py` | Beamer drop-in frames from the result CSVs (reported set; robustness frames in a second file) |
| `did_cs.py` | Shared Callaway–Sant'Anna code used by `4u` and `4x`; `4a` still holds its own copy (merge pending) |

**Run state.** The panel was rebuilt on 2026-09-23 to fix the CHR release-year misalignment. Outputs dated before that predate the fix and should not be cited; `code-audit/plans/2026-09-27_MODEL-RUN-STATE.md` records which script was last run on which panel — check it before citing any output. `qa-scripts/` holds one-off diagnostics (population-column comparison, 2017 .dta vs API check).

---

## Running the reported results

After the panel exists in `Data/merged/` (stages 0–1), the reported results are produced in this order. Each script writes dated files to `Data/output/`; the exhibit scripts read the newest file, so run in order and archive old outputs first (see `code-audit/plans/2026-10-17_MODEL-RUN-STATE.md`).

**Set `PYTHONHASHSEED=0` before running.** `pyfixest`'s results depend on Python's per-process string-hash seed: the same script run twice can differ by about 2e-5 in the standard errors (coefficients by about 1e-8) depending on the seed. Nothing material changes, but unseeded runs are not byte-for-byte reproducible. `environment.yml` sets it automatically for the conda environment; otherwise export it.

```bash
export PYTHONHASHSEED=0
python3 script4j-control-diagnostics.py      # control pre-trends
python3 script4a-design-estimates.py         # A2, A4, A5, A6 (and unreported A1, A7)        ~7 min
python3 script4r-fsis-treatment.py           # slaughterhouse treatment
python3 script4d-balance-panel.py            # balance tables
python3 script4t-pretrend-leads.py           # outcome leads, all treatments
python3 script4u-did-by-treatment.py         # first entry + pre-trend adjustments          ~30 min
python3 script4w-honestdid-export.py && Rscript script4w-honestdid.R   # sensitivity analysis
python3 script4x-species-forest.py           # per-species results and forest plots         ~25 min
python3 script4y-control-set-selection.py    # control-set selection exhibit
python3 script4n-regression-tables.py && python3 script4p-full-appendix.py && python3 script4q-beamer-frames.py
```

`script4u` and `script4x` check themselves: for the first-expansion / dairy-large case they must reproduce `script4a`'s TWFE and CS estimates exactly (they assert or print the difference).

---

## Data availability and access

No raw data is stored in this repository (`*.csv` and `*.xlsx` are git-ignored); the panel is rebuilt from the sources below by stages 0–1. All are US federal public data or public-use extracts; check each source's terms of use. The one non-downloadable input is the FSIS file, which was obtained by FOIA request.

| Source | Script | Access |
|---|---|---|
| US Census population estimates and FIPS crosswalk | `script0a` | Public; four vintage CSVs plus FIPS files from the Census Bureau, placed in `raw/population` and `raw/fips` |
| USDA NASS Census of Agriculture, CAFO operations by size | `script0b` | Public, via the NASS Quick Stats API (free key; `USDA_NASS_API_KEY`); `.dta` fallbacks in `raw/usda` |
| County Health Rankings | `script0c` | Public annual CSVs from County Health Rankings & Roadmaps, manual download to `raw/mental` |
| CDC WONDER deaths of despair (county-year) | `script0c` | Public-use extracts by manual query (annual CSVs); WONDER's data-use restrictions apply, and counts below 10 are suppressed by CDC |
| FBI UCR / NIBRS annual incident files (arrestee segment) | `script0d` | Public bulk files, manual download to `raw/crime/total-v1` |
| NCHS urban-rural classification | `script0a` | Public NCHS file, manual download to `raw/nchs` |
| USDA FSIS establishment data | `script0e`, `fsis-scripts/` | **FOIA response (April 2026), not redistributable**; ZIP-to-county fill uses the HUD USPS crosswalk API (`HUD_API_TOKEN`) |
| SAIPE, BLS LAUS, SAHIE, Census PEP | `script0c` | Public; downloaded automatically |

Replicating without the FSIS file is possible for everything except the slaughterhouse analyses (`script4r` and the FSIS interaction A4).

---

## Key Design Decisions & Assumptions

### Rural Filter (primary sample restriction)
`script0f` retains only county-years where `non_large_metro == 1` per the NCHS 6-level classification (`script0a`). Large central and large fringe metro counties are dropped. All downstream analysis is on this restricted sample (~2,900 counties, ~65,800 county-years in the 2026-09-23 panel).

### FIPS Normalization
All sources are normalized to a 5-digit zero-padded string FIPS via `normalize_panel_key()` in `functions.py`. DC is normalized to `11001`. Non-US state codes and territories are dropped.

### CAFO Data Design
The USDA Agricultural Census runs every five years (2002, 2007, 2012, 2017, 2022). `script0b` outputs **census years only** — no inter-census duplication in the compact file. `script0f` then forward-fills each census year's values to subsequent years (2002 → 2003–2006, 2007 → 2008–2011, etc.), treating the most recent census as the best available observation until the next one supersedes it.

**CAFO zero-fill**: Counties absent from the census in a census year are confirmed to have zero operations (the agricultural census is exhaustive). Those county-years are set to 0 before forward-filling, so the confirmed zero propagates correctly to inter-census years. Only census years `{2002, 2007, 2012, 2017, 2022}` are zero-filled — non-census-year absences are left as NaN.

**Missing counts (D-34, 2026-10-16).** `script0f` zero-fills only a county with no CAFO row for *any* species, so a county with, say, a hog row but no dairy row keeps blank dairy columns. Because the census is exhaustive (checked against NASS for dairy in all five waves), `script4_treatment.load_panel` sets a missing `cafo_{species}_*` count to zero from 2002 on. 2000–01 stay missing: no census was taken in them, so a blank is an unmeasured year, not an absent operation.

**5 commodity types**: cattle (incl calves), beef (cows beef), dairy (cows milk), hogs (all classes), chickens (layers). Beef and dairy are subsets of cattle — aggregate totals use only cattle + hogs + chickens to avoid double-counting; since D-51 the panel's generic size columns (`small_cafo`, `medium_cafo`, `large_cafo`, `bin_sum_cafo`, `total_heads_cafo`, `any_large_cafo`, `any_medium_or_large_cafo`) follow this rule too (before D-51 they summed all five rows; `any_*` were counts of 0 to 5, now 0/1). Beef and dairy get separate `_total` columns for disaggregated analysis.

**CAFO values are operation counts, not animal head counts.** Standard transformation for scatter plots and regressions: **log(cafo_total_ops / population × 10,000 + 1)** via `log_per10k()` in `functions.py`.

### FSIS Panel Separation
FSIS slaughterhouse data is only available from 2017 onward. To avoid truncating the full time-series, the main `panel.csv` **drops all FSIS columns**. A separate `panel_fsis.csv` covers 2017+ with FSIS columns included. Use `panel.csv` for analyses not requiring FSIS; use `panel_fsis.csv` for slaughterhouse analyses.

### CDC Mortality Suppression
CDC suppresses crude rates when deaths < 10 per cell. This causes ~85–93% missingness in `crude_rate_*` for rural counties — **this is expected, not a bug**. Raw death counts are always retained. The derived `cdc_in_query` flag distinguishes counties absent from the CDC WONDER download from those explicitly returning Deaths = 0.

### CHR Variable Handling
- Rate-only CHR variables get back-calculated `*_count_imputed` columns using census population
- Variables with num/denom pairs are ratio-checked; flags archived to `output/tables/*_ratio_flags_qa.csv` then dropped from the panel
- All CHR variables converted to per-100,000 scale where applicable; redundant raw_value columns dropped

### Analytical Panel Window
Based on coverage audits (`script1a`):
- `poor_mental_health_days` (CHR): usable from 2010 onward (~90% fill)
- `frequent_mental_distress` (CHR): 2016+ only — not suitable for pre-2016 panels
- FSIS establishments: 2017+ only (use `panel_fsis.csv`)
- **Recommended window: 2010–2015** (adequate MH coverage, CAFO census year 2012, pre-FSIS)

---

## Panel Output Files (`Data/merged/`)

| File | Description |
|------|-------------|
| `YYYY-MM-DD_panel.csv` | Full rural panel, all years, **no FSIS columns** — primary analysis file (149 cols, ~65.8k rows, 2000–2023, ~2,900 counties) |
| `YYYY-MM-DD_panel_05_10.csv` | Year slice 2005–2010 |
| `YYYY-MM-DD_panel_10_20.csv` | Year slice 2010–2020 |
| `YYYY-MM-DD_panel_census_years.csv` | USDA census years only: 2002, 2007, 2012, 2017, 2022 |
| `YYYY-MM-DD_panel_fsis.csv` | 2017+ with FSIS columns (more columns; FSIS fields added) |

### How to load the latest panel
```python
from functions import latest_file_glob, normalize_panel_key
import pandas as pd, os

merged_dir = os.path.expanduser("~/Dropbox/Mental/Data/merged")
path = latest_file_glob(merged_dir, "*_panel.csv")
df = pd.read_csv(path, low_memory=False)
df = normalize_panel_key(df, dropna=True)
```

---

## Internal documentation

`code-audit/plans/` holds dated decision logs and status notes — not reviewer-facing. Start with `CONTROL-DECISION-REGISTER.md` (every control inclusion and exclusion, D-numbered), `2026-09-27_MODEL-NAMING.md` (A/B model IDs) and `2026-09-27_MODEL-RUN-STATE.md` (what was run on which panel). `code-audit/code-audit/` has the data explainer and an HTML code review.
