# Large-Scale Livestock, Slaughterhouse Work, and Mental Health in the US

## Purpose
This project combines US health, crime, population, and agriculture data into one large, unbalanced county-year panel covering roughly 2000–2023 (source-dependent). The goal is causal analysis between large-scale livestock operations (farms, slaughter and processing facilities) and mental health outcomes among US agriculture workers, using a repeated cross-section at the county (FIPS) level.

---

## Requirements

- Python 3.11+ (most recent runs on 3.13)
- Core packages (set in `packages.py`): `pandas`, `numpy`, `glob`, `re`, `matplotlib`, `seaborn`
- ML / estimation packages (`sklearn`, `statsmodels`, `linearmodels`, `differences`): imported directly by `script3-ridge.py` and the `script4*` estimation scripts; check each script's imports
- Optional: `plotly` + `kaleido` (used in `script2f-maps.py` and other figure scripts for choropleth maps; scripts degrade gracefully if unavailable)
- Helper utilities in `functions.py`: FIPS normalization, CSV fallback readers, USDA NASS API wrappers, `latest_file_glob()`, `latest_files_by_descriptor()`, `normalize_panel_key()`, `to_numeric_series()`, `log_per10k()`, `load_county_geojson()`
- **USDA NASS API key** required for `script0b` — set env var `USDA_NASS_API_KEY`
- **HUD API token** required for `script0f` FSIS pipeline — set env var `HUD_API_TOKEN`
- `script0g` downloads SAIPE, BLS LAUS, SAHIE and Census PEP directly (no key); the BLS host `download.bls.gov` must be reachable

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
├── script0a … script0g-*.py   # Stage 0 — raw ingestion
├── script1a, script1b-*.py    # Stage 1 — QA & panel build
├── script2a … script2f-*.py   # Stage 2 — QA, descriptive figures, maps
├── script3-ridge.py           # Stage 3 — ridge/LASSO pilot (stub)
├── script4*.py                # Stage 4 — design-based estimates, tables, slides
├── functions.py, packages.py  # shared helpers; db_base path lives in packages.py
├── fsis-scripts/              # FSIS sub-scripts, called by script0f
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
├── merged/           # Panel outputs from script1b
│
└── output/
    ├── figs/         # All PNGs from script2* (one subfolder per script)
    └── tables/       # All CSV outputs from script2* + QA tables
```

---

## Script Execution Order

### Stage 0 — Raw Ingestion & Source Cleaning
Each script reads from `Data/raw/` and writes panel-ready CSVs to `Data/clean/`. Intermediates and QA files go to `Data/clean/diagnostic/`.

| Script | Purpose | Key output (in `Data/clean/`) |
|--------|---------|-------------------------------|
| `script0a-pop-fips-raw-merge.py` | US Census population (1990–2024) + FIPS crosswalk panel | `*_population_full.csv`, `*_fips_full.csv` |
| `script0b-usda-raw.py` | USDA NASS CAFO operations by size — **census years only** (2002, 2007, 2012, 2017, 2022) via API | `*_cafo_ops_by_size_compact.csv` |
| `script0c-health-raw.py` | County Health Rankings MH survey + CDC deaths-of-despair | `*_mentalhealthrank_full.csv`, `*_cdc_county_year_deathsofdespair.csv` |
| `script0d-crime-raw.py` | FBI UCR violent crime, collapsed to FIPS-year | `*_crime_fips_level_final.csv` |
| `script0e-nchs-urban.py` | NCHS urban-rural classification expanded to annual panel | `*-rural-key.csv` |
| `script0f-fsis-clean-all.py` | FSIS slaughterhouse pipeline orchestrator (runs `fsis-scripts/` sub-scripts via `subprocess`; requires `HUD_API_TOKEN`) | `*_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip.csv` |
| `script0g-annual-controls.py` | Annual, correctly-dated county controls from SAIPE, BLS LAUS, SAHIE and Census PEP. Replaces 10 CHR controls whose release-year dating misaligned them with the data year (see `code-audit/plans/2026-09-21_CHR-YEAR-MISALIGNMENT.md`) | `*_annual_controls_county_year.csv` |

**`fsis-scripts/` sub-scripts** (run in this order by `script0f`; each fills blanks only and never overwrites an existing `fips_code`):
`script0e-fsis-slaughterhouses` (raw FOIA → establishment-year panel) → `0h` (size/type panel) → `0i` (HUD ZIP→county API fill) → `0j` (year-aware refill) → `0k` (HUD bulk ZIP-year fill) → `0l` (manual ZIP→FIPS) → `0m` (deterministic completion: ZIP, CT, city+state, establishment history) → `0n` (manual county-label→FIPS).

### Stage 1 — QA & Panel Build

| Script | Purpose | Output location |
|--------|---------|-----------------|
| `script1a-var-completeness.py` | Panel coverage QA: integrity checks, mental health variable inventory, outcome distributions, CDC suppression audit, CAFO per-capita transformation analysis, coverage heatmap | `Data/output/tables/panel-qa/` |
| `script1b-generate-panel.py` | **Master panel builder** — builds rural county-year panel, forward-fills CAFO census data to all years, applies `non_large_metro == 1` filter, derives crude rate columns, CHR count imputation, rate standardization, column cleanup | `Data/merged/*_panel*.csv` |

### Stage 2 — QA, Visualization, Maps

| Script | Purpose | Output location |
|--------|---------|-----------------|
| `script2a-panel-qa.py` | Panel QA, missingness, composition and summary statistics: CDC crude-rate sense check, correlation tables, FSIS column QA | `Data/merged/`, `Data/output/tables/` |
| `script2b-final-visuals.py` | Core analytical figures: CAFO / FSIS exposure vs outcomes (binned scatter, time series, heatmap) | `Data/output/figs/core-visuals/` |
| `script2c-cafo-composition.py` | CAFO geographic concentration and size-composition | `Data/output/figs/cafo-composition/` |
| `script2d-threshold-presence.py` | CAFO threshold / presence vs outcomes (non-linear by design) | `Data/output/figs/threshold-presence/` |
| `script2e-hispanic-chr-explorer.py` | Hispanic population as confounder and heterogeneity dimension | `Data/output/figs/hispanic-chr/` |
| `script2f-maps.py` | Plotly choropleths: FSIS counts, mental-outcome coverage, ag first-differences | `Data/output/figs/{fsis-choropleth,mental-outcome-coverage-maps,ag-first-diff-maps}/` |

### Stage 3 — Pilot

| Script | Purpose | Note |
|--------|---------|------|
| `script3-ridge.py` | LASSO/Ridge variable selection, panel OLS with county × year FE, DiD event study around census years | Pilot — not for interpretation |
| `script4-model-test.py` | Independent replication of an external dairy-CAFO memo's finding | Legacy; superseded by the Part A scripts below |

### Stage 4 — Part A: design-based estimates

All models run over treatments **T1–T4** (large-CAFO presence, count, per 10k, log(1+count)) × 7 outcomes (3 mental-health / mortality, 4 crime), county-clustered SEs as the headline. Every `script4*` file takes its panel, treatments and controls from the single shared module `script4_treatment.py` so each variable is defined in exactly one place. Model IDs follow `code-audit/plans/2026-09-27_MODEL-NAMING.md` (A = implemented, B = specified only).

| Script | Role |
|--------|------|
| `script4_treatment.py` | **Shared module** — treatment, sample and control definitions (importable; not run) |
| `script4a-design-estimates.py` | A1 pooled OLS (state+year FE), A2 TWFE (county+year FE, headline), A4 dairy × FSIS, A5 event study, A6 Callaway–Sant'Anna |
| `script4l-a3-horserace.py` | A3 — dairy conditional on other large CAFO types |
| `script4b-covariate-sequence.py` | Covariate-sequence grid: 840 regressions adding controls one at a time |
| `script4c-covariate-stability.py` | Synthesis of 4b — how far each coefficient moves along the sequence (no new estimation) |
| `script4d-balance-panel.py` | Balance table on the corrected panel |
| `script4d-power-balance.py` | Sample, power and coverage workbook for Part A (older; pre-rebuild) |
| `script4j-control-diagnostics.py` | Coverage, causal role and pre-trends for each control |
| `script4k-variable-register.py` | Master variable register, one row per panel column |
| `script4e-latex-tables.py`, `script4h-result-tables.py`, `script4n-regression-tables.py`, `script4p-full-appendix.py` | LaTeX tables and the combined appendix (no estimation) |
| `script4g-slides.py`, `script4q-beamer-frames.py` | Beamer deck and drop-in frames, generated from result CSVs |

**Run state.** The panel was rebuilt on 2026-09-23 to fix the CHR release-year misalignment. Outputs dated before that predate the fix and should not be cited; `code-audit/plans/2026-09-27_MODEL-RUN-STATE.md` records which script was last run on which panel — check it before citing any output. `qa-scripts/` holds one-off diagnostics (population-column comparison, 2017 .dta vs API check).

---

## Key Design Decisions & Assumptions

### Rural Filter (primary sample restriction)
`script1b` retains only county-years where `non_large_metro == 1` per the NCHS 6-level classification (`script0e`). Large central and large fringe metro counties are dropped. All downstream analysis is on this restricted sample (~2,900 counties, ~65,800 county-years in the 2026-09-23 panel).

### FIPS Normalization
All sources are normalized to a 5-digit zero-padded string FIPS via `normalize_panel_key()` in `functions.py`. DC is normalized to `11001`. Non-US state codes and territories are dropped.

### CAFO Data Design
The USDA Agricultural Census runs every five years (2002, 2007, 2012, 2017, 2022). `script0b` outputs **census years only** — no inter-census duplication in the compact file. `script1b` then forward-fills each census year's values to subsequent years (2002 → 2003–2006, 2007 → 2008–2011, etc.), treating the most recent census as the best available observation until the next one supersedes it.

**CAFO zero-fill**: Counties absent from the census in a census year are confirmed to have zero operations (the agricultural census is exhaustive). Those county-years are set to 0 before forward-filling, so the confirmed zero propagates correctly to inter-census years. Only census years `{2002, 2007, 2012, 2017, 2022}` are zero-filled — non-census-year absences are left as NaN.

**5 commodity types**: cattle (incl calves), beef (cows beef), dairy (cows milk), hogs (all classes), chickens (layers). Beef and dairy are subsets of cattle — aggregate totals use only cattle + hogs + chickens to avoid double-counting. Beef and dairy get separate `_total` columns for disaggregated analysis.

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
