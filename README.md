# Large-Scale Livestock, Slaughterhouse Work, and Mental Health in the US

## Purpose
This project combines US health, crime, population, and agriculture data into one large, unbalanced county-year panel covering roughly 2000–2023 (source-dependent). The goal is causal analysis between large-scale livestock operations (farms, slaughter and processing facilities) and mental health outcomes among US agriculture workers, using a repeated cross-section at the county (FIPS) level.

---

## Requirements

- **Python 3.13** (results produced on 3.13.5; macOS). Install the pinned packages with either
  `pip install -r requirements.txt` or `conda env create -f environment.yml` (the conda file installs Python and then the same pinned `requirements.txt`).
  Key packages: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, **`pyfixest`** (all estimation), `statsmodels` and `wildboottest` (multiple-testing corrections and wild cluster bootstrap), `requests`, `openpyxl`, `pyarrow`; optional `plotly` + `kaleido` for maps.
- **R 4.x** only for `R/honestdid.R` (Rambachan–Roth sensitivity analysis): `install.packages("HonestDiD")` (see `r-requirements.txt`).
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
├── estimators.py              # shared estimator code: DiD helpers, Callaway–Sant'Anna, LP-DiD, triangulation estimators, Conley
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
└── output/          # one folder, split only by kind of file; the producing script and section are in the file name
    ├── tables/       # every CSV / TXT / MD result, flat
    ├── figures/      # every PDF / PNG figure, flat
    ├── maps/         # HTML + PNG maps (script1d), flat
    ├── latex/        # .tex fragments, REGRESSION_TABLES, FULL_RESULTS_APPENDIX, BEAMER_FRAMES; slides/ holds the deck's
    │                 #   fragments and slides/assets/ the undated links to the newest slide figures (repo slides/generated -> here)
    ├── codebooks/    # <date>_VARIABLE_REGISTER.{xlsx,csv} from script1b --only register (every dated version kept)
    ├── _archive/     # superseded dated copies, by cleanup date, with MANIFEST.csv files (original -> new path)
    ├── _legacy/      # outputs of retired scripts (stage 2, pre-October stage 4, Sept-18 slide build); MANIFEST.csv
    └── _baseline_pre_refactor_2026-10-22/   # frozen 10-22 refactor baseline, old folder layout; do not move
```

**File names.** Every output is `YYYY-MM-DD_<tag>_<stem>.<ext>`, e.g. `tables/2026-10-25_4a-cohorts_static_twfe.csv` or
`figures/2026-10-26_4f-slides_forest_species_dairy__main.pdf`. The tag names the script and section that wrote the file:

| Tag | Written by | | Tag | Written by |
|---|---|---|---|---|
| `0f` | `script0f-generate-panel.py` (ratio flags QA) | | `4b-honestdid` | `R/honestdid.R` |
| `1a` `1b` `1c` `1d` | `script1a`-`script1d` (all sections) | | `4b-honestdid-inputs` | `script4b --only honestdid` (undated) |
| `4a-designs` `4a-cohorts` `4a-species` `4a-fsis` | `script4a-estimates.py --only <section>` | | `4c-wild` `4c-permutation` | `script4c-inference.py` |
| `4b-controls` `4b-leads` `4b-pretrends` `4b-balance` `4b-control-set` | `script4b-diagnostics.py` | | `4d-estimators` `4d-treatment-by-model` | `script4d-triangulation.py` |
| `4f-triangulation` `4f-treatment-by-model` `4f-permutation` `4f-pretrends` `4f-slides` | `script4f-figures.py` | | `qa-cs-crosscheck` | `qa-scripts/qa_cs_crosscheck.py` |
| `4g-moran` `4g-donut` `4g-conley` `4g-division-fe` `4g-durbin` | `script4g-spillovers.py` | | `4b-species-counts` | `script4b --only species-counts` |

`--quick` runs add `-quick` to the tag (`4a-cohorts-quick`), so smoke-run files never mix with full runs. `4b-covariate-sequence` is the 09-23
output of the archived `z-archive/script4b-covariate-sequence.py`. The rule lives in `packages.py` only: scripts build paths with
`out_file(folder, tag, name)` and find inputs with `out_latest(folder, tag, pattern)` / `out_glob(...)`; the files in `latex/` keep fixed
names because LaTeX `\input`s them. The 2026-10-26 move from per-script folders to this layout is recorded file by file in
`_archive/2026-10-26_layout-migration_MANIFEST.csv`.

**Archiving convention.** Nothing in `Data/output/` is deleted. When a newer dated copy of an output exists, the older copy moves to
`_archive/<YYYY-MM-DD>_cleanup/` (first done 2026-10-26: 524 files); `out_latest()` always finds the newest copy, which stays in place.
Outputs of a retired script move to `_legacy/`. `*-quick` files can be moved to `_archive/` after a smoke run.

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
`script0e-fsis-slaughterhouses` (sections `extract`: raw FOIA → establishment-year panel, and `size-panel`: size/type interim panel) → `script0i-fsis-hud-fill` (`--strategy bulk`, the default: HUD bulk ZIP-year fill; or `targeted`: per-ZIP lookups, then the year-aware refill) → `0l-fsis-manual-and-deterministic-fill` (one pass, in this order: manual ZIP→FIPS, deterministic completion [ZIP, CT, city+state, establishment history], manual county-label→FIPS; a missing manual template skips its step). It writes the interim and county files once; the earlier three-script chain overwrote one file name three times (D-48).

### Stage 1 — QA and descriptive work (before analysis; none of these writes the panel)

| Script | Purpose | Output location |
|--------|---------|-----------------|
| `script1a-panel-qa.py` | Checks that can fail: key integrity and source duplication, variable inventory and coverage, CDC crude-rate sense check, correlation tables and FSIS column QA, missingness, county coverage check against the rural key, CAFO crosscheck, QA memo. `--only <section>` | `Data/output/tables/`, `Data/output/figures/` (tag `1a`) |
| `script1b-tables.py` | Numeric tables: summary statistics, CAFO concentration (HHI, Gini, entry/exit), CAFO and FSIS size-bin summaries, species overlap, the master variable register (xlsx + csv codebook: role, control-set and reported-treatment flags, decision ids, coverage, source) | `Data/output/tables/` (tag `1b`), `Data/output/codebooks/` |
| `script1c-figures.py` | All non-map graphics: binned scatters, correlations, distributions, CAFO composition and top counties, threshold and presence, ridgelines, trend sweeps, state trends and dots, species overlap, raw trends and event-time paths, and the fill-rate bar chart of the 24 candidate controls (`control-coverage`, with a `.txt` table). Figures use `figstyle.py` (183 mm, 600 dpi PNG + vector PDF). Each outcome is drawn over its own window (`designs.OUTCOME_WINDOWS`: Poor MH Days 2010–2023, Frequent Mental Distress 2016–2023, deaths of despair and crime 2002–2020; descriptive stage only) | `Data/output/figures/`, `Data/output/tables/` (tag `1c`) |
| `script1d-maps.py` | All maps (Plotly, HTML + PNG): FSIS density, outcome values, first differences, CAFO by animal and size, and one family per treatment (entry per wave, first-expansion cohort, large operations, species, the Poor MH Days estimation sample, sample coverage, FSIS entry, change between census waves) | `Data/output/maps/` (tag `1d`) |

The Hispanic-share and health explorer (`script2e-hispanic-chr-explorer.py`) is archived in `z-archive/`. Old stage-2 outputs are in `Data/output/_legacy/` (moved 2026-10-26); D-49 records the consolidation.

### Stage 3 — Pilot (moved)
The ridge/LASSO pilot (`script3-ridge.py`) and the first single-file model test (`script4-model-test.py`) now live in `first-stage-testing/` as `firststage_*.py`. They were the first phase of development, are not part of the pipeline, and are not used for any reported result. ML variable selection remains specified-only (B-series in `code-audit/plans/2026-09-27_MODEL-NAMING.md`).

### Stage 4 — Part A: design-based estimates

**What is reported.** Per the team's 2026-10-15 feedback the reported set is treatments **T1** (large-dairy presence), **T3** (per 10,000 residents), **T4** (log(1 + count)) and **E2** (first expansion: 0 until the first census wave at which the large-dairy count rises, 1 from then on), **county + year fixed effects**, **standard errors clustered by state**, and the estimators **TWFE** and **Callaway–Sant'Anna covariate-adjusted** (CS). These choices live in `REPORT_*` in `script4_treatment.py` and only decide what the exhibit scripts show: the estimation scripts still compute the other specifications (T2 count, pooled state-FE A1, unconditional CS), which appear only in the robustness exhibits. **LP-DiD is switched off** (meeting of 2026-10-26, D-57): `INCLUDE_LPDID = False` in `estimators.py`; set it to `True` to run and draw it again. The **control set** is the 8 full-coverage controls `COVARIATE_ORDER[:8]`, pre-committed before any result was read (decision D-16, revised D-31). Every `script4*` file takes its panel, treatments and controls from `script4_treatment.py`, so each variable is defined in exactly one place. Outcomes: 7 (3 mental-health / mortality, 4 crime).

| Script | Role |
|--------|------|
| `script4_treatment.py` | **Shared module** — treatment, sample, outcome and control definitions; the missing-count-is-zero rule (from 2002) is applied in `load_panel` (importable; not run) |
| `script4a-estimates.py` | **All design-based estimates**, sections `designs` (A2 TWFE headline, A4 dairy × FSIS, A5 event study, A6 Callaway–Sant'Anna; A1 pooled state-FE computed, not reported; A7 LP-DiD off, D-57), `cohorts` (first expansion and **first entry** with the pre-trend adjustments: county trends, state × year FE, baseline characteristics × time polynomial), `species` (per-species TWFE and CS, large and all-sizes counts, forest plots) and `fsis` (slaughterhouses as the treatment; R4 LP-DiD off, D-57). `cohorts` and `species` check themselves against `designs` |
| `script4b-diagnostics.py` | Sections `controls` (coverage, causal role and pre-trends of each control; also the lag-level pre-trend coefficients, the correlation among the 8 selected controls and plain-text `.txt` versions of the VIF and control tables), `leads` (outcome pre-trends for T1, T3, T4, E2 and first entry), `pretrends` (raw pre-period paths and straight-line fits for outcomes and controls), `balance` (balance and detectable-effect tables), `control-set` (**how the control set was chosen**: add-one-in, each alone, take-one-out; documents the pre-committed set, does not choose it) and `honestdid` (writes the inputs of the Rambachan–Roth sensitivity analysis; then `Rscript R/honestdid.R`) |
| `script4c-inference.py` | Sections `wild` (wild cluster bootstrap, Holm, Romano–Wolf within outcome, Benjamini–Hochberg; D-42, D-43) and `permutation` (within-state cohort permutation test; D-47; flags `--design`, `--quick`) |
| `script4d-triangulation.py` | Sections `estimators` (is the Callaway–Sant'Anna result specific to it? Sun–Abraham, imputation, dCDH, stacked DiD, and LP-DiD only when `INCLUDE_LPDID` is True; D-45 as changed by D-57: 3 of 4 agree; needs R `DIDmultiplegtDYN`) and `treatment-by-model` (every treatment against every model; D-46) |
| `script4e-tables.py` | Sections `tables` (LaTeX regression tables, no estimation), `appendix` (combined appendix; needs `balance` and `tables` first), `frames` (Beamer drop-in frames) and `slides` (the table fragments of `slides/deck.tex`, plus undated links to the slide figures, in `latex/slides/`) |
| `script4g-spillovers.py` | **Spatial spillover checks** (D-58), sections `moran` (Moran's I of the A2 residuals and the raw outcome), `donut` (TWFE and Callaway–Sant'Anna after dropping never-treated counties within 50 and 100 km of a treated county), `conley` (spatial standard errors, 100/250/500 km), `division-fe` (Census division × year fixed effects) and `durbin` (spatial Durbin panel model: direct, spillover and total effect; weights W_dist 100 km, W_cz when a commuting-zone crosswalk is in `Data/raw/cz/`). Needs the cached county outlines `Data/clean/geo/*_us_counties.geojson`; uses `spatial.py`; tests in `tests/test_spatial.py` |
| `script4f-figures.py` | Sections `slides` and `slides-other` (all 15 forest plots, event studies, leads, raw paths, control pre-trends and coverage and the permutation figure redrawn at slide size, 145 x 70 mm, 8 pt; main panels are the three mental-health and mortality outcomes, backup panels the crime outcomes; `Data/output/figures/` (tag `4f-slides`)), and sections `triangulation`, `treatment-by-model` and `permutation`: forest and null-distribution figures read from the result files (the species forest plots are drawn in `script4a-estimates.py --only species`); and the pre-trend figures with dotted straight-line fits, `pretrends-leads` (five treatment formats), `pretrends-controls` (the 8 selected controls on an x-y axis), `pretrends-raw` and `pretrends-controls-raw` (raw paths, treated against never-treated counties), written to `Data/output/figures/` (tag `4f-pretrends`) |
| `z-archive/script4b-…`, `z-archive/script4c-…` | Earlier run of adding controls in a fixed order and its summary; superseded by the `control-set` section, kept unchanged as the record of the 09-23 run |
| `estimators.py` | Shared estimator code (DiD helpers, Callaway–Sant'Anna, LP-DiD, stacked, imputation, saturated, dCDH); formerly `did_tools.py` and `did_cs.py` |

**Run state.** The panel was rebuilt on 2026-09-23 to fix the CHR release-year misalignment. Outputs dated before that predate the fix and should not be cited; `code-audit/plans/2026-09-27_MODEL-RUN-STATE.md` records which script was last run on which panel — check it before citing any output. `qa-scripts/` holds one-off diagnostics (population-column comparison, 2017 .dta vs API check).

---

## Running the reported results

After the panel exists in `Data/merged/` (stages 0–1), the reported results are produced in this order. Each script writes dated files to `Data/output/`; the exhibit scripts read the newest file, so run in order and archive old outputs first (convention under the Dropbox data tree above; run state in `code-audit/plans/2026-10-17_MODEL-RUN-STATE.md`).

**Set `PYTHONHASHSEED=0` before running.** `pyfixest`'s results depend on Python's per-process string-hash seed: the same script run twice can differ by about 2e-5 in the standard errors (coefficients by about 1e-8) depending on the seed. Nothing material changes, but unseeded runs are not byte-for-byte reproducible. `environment.yml` sets it automatically for the conda environment; otherwise export it.

```bash
export PYTHONHASHSEED=0
python3 script4b-diagnostics.py --only controls                      # control roster and pre-trends (other sections read them)
python3 script4a-estimates.py                                        # designs (~7 min), cohorts (~30 min), species (~25 min), fsis
python3 script4b-diagnostics.py --only leads pretrends balance control-set honestdid && Rscript R/honestdid.R   # diagnostics, sensitivity analysis
python3 script4c-inference.py                                        # wild bootstrap family (~15 min), then permutation tests (~1 h)
python3 script4d-triangulation.py                                    # estimator triangulation (~20 min), treatment by model (~30 min)
python3 script4e-tables.py                                           # tables, then appendix, then Beamer frames
python3 script4f-figures.py                                          # triangulation, treatment-by-model, permutation and pre-trend figures
python3 script4f-figures.py --only slides slides-other && python3 script4e-tables.py --only slides   # slide figures and fragments
(cd slides && latexmk -xelatex deck.tex)                             # the deck (Gotham theme needs XeLaTeX; `slides/generated` is a link to the Dropbox folder)
```

Each script takes `--only <section> ...` and keeps the flags of the script it replaced (`--quick`, `--no-cs`, `--plot-only`, `--design`, `--outcome`). Outputs carry the tag of the script and section that wrote them (table above); the old-script-to-section map, with the folder each section used before 2026-10-26, is in `code-audit/plans/2026-10-26_SCRIPT-NAME-MAP.md`.

The `cohorts` and `species` sections check themselves: for the first-expansion / dairy-large case they must reproduce the `designs` TWFE and CS estimates exactly (they assert or print the difference).

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

**Missing counts (D-34, 2026-10-16).** `script0f` zero-fills a county with no CAFO row for *any* species, and (D-53, 2026-10-25) sets any other blank species count in a census year to 0 before the forward-fill; before D-53 such a blank was filled with the previous wave's count, so a county whose last large dairy operation had closed kept it in the panel (119 county-years in 21 counties for T1, T3, T4). Because the census is exhaustive (checked against NASS for dairy in all five waves), `script4_treatment.load_panel` sets a missing `cafo_{species}_*` count to zero from 2002 on. 2000–01 stay missing: no census was taken in them, so a blank is an unmeasured year, not an absent operation.

**5 commodity types**: cattle (incl calves), beef (cows beef), dairy (cows milk), hogs (all classes), chickens (layers). Beef and dairy are subsets of cattle — aggregate totals use only cattle + hogs + chickens to avoid double-counting; since D-51 the panel's generic size columns (`small_cafo`, `medium_cafo`, `large_cafo`, `bin_sum_cafo`, `total_heads_cafo`, `any_large_cafo`, `any_medium_or_large_cafo`) follow this rule too (before D-51 they summed all five rows; `any_*` were counts of 0 to 5, now 0/1). Beef and dairy get separate `_total` columns for disaggregated analysis.

**CAFO values are operation counts, not animal head counts.** Standard transformation for scatter plots and regressions: **log(cafo_total_ops / population × 10,000 + 1)** via `log_per10k()` in `functions.py`.

### FSIS Panel Separation
FSIS slaughterhouse data is only available from 2017 onward. To avoid truncating the full time-series, the main `panel.csv` **drops all FSIS columns**. A separate `panel_fsis.csv` covers 2017+ with FSIS columns included. Use `panel.csv` for analyses not requiring FSIS; use `panel_fsis.csv` for slaughterhouse analyses.

### CDC Mortality Suppression
CDC suppresses crude rates when deaths < 10 per cell. This causes ~85–93% missingness in `crude_rate_*` for rural counties — **this is expected, not a bug**. Raw death counts are always retained. The derived `cdc_in_query` flag distinguishes counties absent from the CDC WONDER download from those explicitly returning Deaths = 0.

### CHR Variable Handling
- Rate-only CHR variables get back-calculated `*_count_imputed` columns using census population
- Variables with num/denom pairs are ratio-checked; flags archived to `output/tables/*_0f_ratio_flags_qa.csv` then dropped from the panel
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
