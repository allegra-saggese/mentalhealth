# Large-Scale Livestock, Slaughterhouse Work, and Mental Health in the US

## Purpose
This project combines US health, crime, population, and agriculture data into one large, unbalanced county-year panel covering roughly 1999–2022 (source-dependent). The goal is to identify causal relationships between large-scale livestock operations (particularly slaughterhouse work) and mental health outcomes among US agriculture workers, using a repeated cross-section at the county (FIPS) level.

---

## Requirements

- Python 3.11
- Core packages (set in `packages.py`): `pandas`, `numpy`, `glob`, `re`, `matplotlib`, `seaborn`
- ML packages (`sklearn`, `statsmodels`): only required for `script3-ridge.py`; imported there directly
- Optional: `plotly` + `kaleido` (used in `script2b`, `script2c`, `script2h` for choropleth maps; scripts degrade gracefully if unavailable)
- Helper utilities in `functions.py`: FIPS normalization, CSV fallback readers, USDA NASS API wrappers, `latest_file_glob()`, `normalize_panel_key()`, `to_numeric_series()`, `log_per10k()`, `load_county_geojson()`
- **USDA NASS API key** required for `script0b` — set env var `USDA_NASS_API_KEY`
- **HUD API token** required for `script0f` FSIS pipeline — set env var `HUD_API_TOKEN`

### Hardcoded paths that must change per user
| Script | Variable | What it points to |
|--------|----------|-------------------|
| `packages.py` | `db_base = os.path.expanduser("~/Dropbox/Mental")` | Dropbox root — all other paths derive from this |
| `script0a`, `script0b`, `script0d` | `repo = "/Users/allegrasaggese/Documents/GitHub/mentalhealth"` | GitHub repo root for sys.path |
| `script0b` | `donor_path = ".../2026-02-23_ag_annual_df.csv"` | Local backfill CSV for 2012/2017 USDA data |
| `script0d` | `interim_override = "/Users/allegrasaggese/Dropbox/..."` | Local Dropbox copy for crime QA outputs |
**Fix `db_base` in `packages.py` first.** All directory paths in all other scripts are derived from `db_data = db_base/Data` and will resolve correctly once the root is set.

---

## Raw Data Layout (local Dropbox, not in repo)

```
Data/
└── raw/
    ├── cdc/          # CDC WONDER deaths-of-despair annual CSVs
    │                 #   pattern: cty-level-deathsofdespair-YYYY.csv
    ├── crime/
    │   └── total-v1/ # FBI UCR annual incident CSVs (~7 GB combined in memory)
    ├── fips/         # FIPS crosswalk .txt files (foruse_*.txt)
    │                 #   pipe-delimited (2020), comma-delimited (2010),
    │                 #   whitespace/fixed-width (2000)
    ├── mental/       # County Health Rankings annual CSVs
    │                 #   year inferred from filename (must contain 4-digit year)
    ├── nchs/         # NCHSurb-rural-codes.csv (single file)
    ├── population/   # US Census population CSVs (4 vintage files):
    │                 #   cc-est2024-agesex-all, cc-est00int-tot,
    │                 #   cc-est2010-2020, cc-est-1990-2000
    ├── usda/         # USDA NASS fallback .dta files (used if API fails)
    └── fsis/         # USDA FSIS FOIA inspection CSVs
                      #   (processed via fsis-scripts/ subfolder)

Data/
├── clean/            # Output of script0* — one dated CSV per source
├── merged/           # Output of script1c — full panel + year-slice exports
│   └── figs/
│       ├── panel-sumstats-by-farms/   # QA tables, correlation CSVs, QA memo
│       │   ├── maps/                  # Plotly choropleth HTMLs
│       │   └── plots/                 # PNG scatter and distribution plots
│       ├── core-visuals/              # Final analytical figures (script2f)
│       ├── binned_scatter/            # Binned scatter PNGs (script2d)
│       └── state_trends_large_cafo/   # State-level trend plots (script2d)
└── FOIA-USDA-request/
    └── qa-fsis/      # FSIS QA templates (manual ZIP/county fill spreadsheets)
```

---

## Script Execution Order

### Stage 0 — Raw Ingestion & Source Cleaning
Each script reads from `Data/raw/` and writes a dated CSV to `Data/clean/`.

| Script | Purpose | Key output file (in `Data/clean/`) |
|--------|---------|-----------------------------------|
| `script0a-pop-fips-raw-merge.py` | US Census population (1990–2024) + FIPS crosswalk panel **and** USDA NASS CAFO operations by size (API + `.dta` fallback) | `*_population_full.csv`, `*_cafo_ops_by_size_compact.csv` |
| `script0c-health-raw.py` | County Health Rankings MH survey + CDC deaths-of-despair | `*_mentalhealthrank_full.csv`, `*_cdc_county_year_deathsofdespair.csv`, `*_mh_mortality_fips_yr.csv` |
| `script0d-crime-raw.py` | FBI UCR violent crime, collapsed to FIPS-year | `*_crime_fips_level_final.csv` |
| `script0e-nchs-urban.py` | NCHS urban-rural classification expanded to annual panel | `*-rural-key.csv` |
| `script0f-fsis-clean-all.py` | FSIS slaughterhouse pipeline orchestrator (runs `fsis-scripts/` sub-scripts) | `*_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip.csv` |

### Stage 1 — QA, Merge, Derived Variables

| Script | Purpose | Output location |
|--------|---------|-----------------|
| `script1a-QA-indiv.py` | Population column QA across all merged files | `allegra-dropbox-copy/interim-data/` |
| `script1b-audit-mental-vars.py` | Mental health variable coverage audit, outcome distribution plots, CAFO per-capita transformation analysis | `Data/merged/figs/panel-sumstats-by-farms/` |
| `script1c-merge-dataclean.py` | **Master merge** — builds rural county-year panel, applies `non_large_metro == 1` filter, derives crude rate columns | `Data/merged/*_full_merged.csv` + year-slice exports |

### Stage 2 — QA Checks & Visualization

| Script | Purpose | Output location |
|--------|---------|-----------------|
| `script2a-qa.py` | **QA pipeline (3 sections):** (1) CDC crude-rate sense check + re-export of merged panel with diagnostic columns; (2) pairwise correlation tables (Pearson + Spearman), FSIS 12-column QA, QA memo; (3) FSIS size-bin counts vs. poor mental health days scatter (2017 cross-section) | `Data/merged/figs/panel-sumstats-by-farms/` |
| `script2b-summary-stats-vis.py` | **Combined stats + visualization (2 sections):** (A) broad missingness/trend/violin/binned-scatter/state-level batch; (B) targeted CAFO/FSIS/MH sumstats, coverage check, county and state maps, chicken scatter facets, CAFO unit cross-check | `Data/merged/figs/` and `figs/panel-sumstats-by-farms/` |
| `script2c-final-visuals.py` | **Core analytical figures**: binned scatter (A1/A2/A3), county time-series (B), outcome cross-correlation heatmap (C), violin/box plots (D/E), choropleth maps (F1/F2) | `Data/merged/figs/core-visuals/` |
| `script2d-cafo-composition.py` | CAFO geographic concentration: top-20 county rankings, stacked size-composition bars | `Data/merged/figs/cafo-composition/` |
| `script2e-threshold-presence.py` | CAFO threshold/presence analysis: binary presence vs. outcomes, dose-response tiers, small vs. large CAFO comparison | `Data/merged/figs/threshold-presence/` |
| `script2f-hispanic-chr-explorer.py` | Hispanic population as confounder: CHR correlation heatmap, outcome profiles by %Hispanic quartile, partial scatter residualized on %Hispanic | `Data/merged/figs/hispanic-chr/` |
| `script2g-choropleth-maps.py` | **Choropleth map pipeline (3 sections):** (1) FSIS establishment counts by type (2017–2023); (2) mental health outcome values by year; (3) agricultural first-difference maps for identifying variation | `figs/fsis-choropleth/`, `figs/mental-outcome-coverage-maps/`, `figs/ag-first-diff-maps/` |

### Stage 3 — Analysis

| Script | Purpose | Note |
|--------|---------|------|
| `script3-ridge.py` | Ridge regression pilot on county-year panel | In progress; uses a stale 2025 data vintage — results not for interpretation |

---

## Key Design Decisions & Assumptions

### Rural Filter (primary sample restriction)
`script1c` retains only county-years where `non_large_metro == 1` per the NCHS 6-level classification (`script0e`). Counties coded 1 (Large central metro) or 2 (Large fringe metro) are dropped. All downstream analysis is on this restricted sample.

### FIPS Normalization
All sources are normalized to a 5-digit zero-padded string FIPS via `normalize_panel_key()` in `functions.py`. DC is normalized to `11001`. Non-US state codes and territories are dropped.

### CAFO Size Classification (script0b)
Size thresholds are applied per animal type using USDA NASS inventory bin codes. Bins are integers assigned to inventory-size categories (e.g. "1 to 24 head" = bin 1). Cutoffs:

| Animal | Medium >= bin | Large >= bin |
|--------|--------------|-------------|
| Broiler chickens | 3 | 5 |
| Layer chickens | 7 | 9 |
| Cattle / Hogs | 6 | 7 |

**CAFO values are operation counts, not animal head counts.** Confirmed by `script2c` cross-check against the pre-merged compact file.

### CDC Mortality Suppression (script0c, script1c)
CDC suppresses crude rates when deaths < 10 per cell. This causes ~85–93% missingness in `crude_rate_*` for rural counties — **this is expected, not a bug**. Raw death counts are always retained. The derived `cdc_in_query` flag distinguishes counties absent from the CDC WONDER download (unknown) from those explicitly returning Deaths = 0 (ambiguous zero — rate set to NaN).

### CHR Variable Selection (script0c)
County Health Rankings variables are included if: present in >=11 annual files (majority rule) OR present in >=8 files with >=80% average fill. Duplicate column names across years are resolved by keeping the most complete version.

### Analytical Panel Window
Based on coverage audits (`script1b`, `script2a`):
- `poor_mental_health_days` (CHR): usable from 2010 onward (~90% fill)
- `frequent_mental_distress` (CHR): 2016+ only — not suitable for pre-2016 panels
- FSIS establishments: **2017 cross-section only**
- **Recommended window: 2010–2015** (adequate MH coverage, CAFO census year 2012, pre-FSIS)

### CAFO Per-Capita Transformation
Raw CAFO operation counts are heavily right-skewed and mechanically correlated with county size. Standard transformation for all scatter plots and regressions: **log(cafo_total_ops / population x 10,000 + 1)**. Implemented via `log_per10k()` in `functions.py` (shared across all viz scripts).

---

## Final Output Files

### Merged panel (`Data/merged/`)
| File | Description |
|------|-------------|
| `YYYY-MM-DD_full_merged.csv` | Full rural panel (all years) — **primary analysis file** |
| `YYYY-MM-DD_full_merged_2005_2010.csv` | Year slice 2005–2010 |
| `YYYY-MM-DD_full_merged_2010_2020.csv` | Year slice 2010–2020 |
| `YYYY-MM-DD_full_merged_census_years.csv` | Census years only: 2002, 2005, 2007, 2012 |

All files are date-stamped. Use `latest_file_glob(merged_dir, "*_full_merged.csv")` from `functions.py` to auto-select the most recent version.

### Core analytical figures (`Data/merged/figs/core-visuals/`)
Produced by `script2d-final-visuals.py`:
| File | Description |
|------|-------------|
| `*_A1_cafo_total_vs_outcomes_*.png` | Binned scatter: total CAFO (log per-capita) vs. 6 outcomes, pooled 2010–2015 |
| `*_A2_cafo_by_type_vs_outcomes_*.png` | Binned scatter: by animal type (cattle/hogs/chickens) vs. 3 core outcomes |
| `*_A3_fsis_vs_outcomes_2017.png` | Binned scatter: FSIS establishments vs. outcomes, 2017 cross-section only |
| `*_B_county_time_series_by_state.png` | Dual-axis county trends: CAFO exposure vs. poor mental health days, 1 county/state |
| `*_C_outcome_crosscorrelation.png` | Spearman correlation heatmap across all 6 outcome variables |

### How to load the latest panel
```python
from functions import latest_file_glob, normalize_panel_key
import pandas as pd, os

merged_dir = os.path.expanduser("~/Dropbox/Mental/Data/merged")
path = latest_file_glob(merged_dir, "*_full_merged.csv")
df = pd.read_csv(path, low_memory=False)
df = normalize_panel_key(df, dropna=True)
```
