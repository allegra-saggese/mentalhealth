# Health Data Audit (2026-02-25)

## Scope
This document explains the health data audit outputs produced from the 2026-02-24 health rebuild and the follow-up column drop/coverage analyses.

This audit covers:
- Mental health (County Health Rankings) pipeline outputs.
- CDC mortality disaggregated outputs.
- MH + mortality combined panel outputs.
- Variable-selection drop-off behavior.
- Rural-filter coverage impacts.
- Plan for converting rates to percentages.
- Plan for obtaining canonical county-year mortality totals.

## Current Health Datasets
### 1) Mental health cleaned panel
- File: `2026-02-24_mentalhealthrank_full.csv`
- Location: `/Users/allegrasaggese/Dropbox/Mental/Data/clean`
- Unit of observation: `fips-year`
- Coverage: years `2010-2023` (14 years)
- Rows/cols: `44,709 x 311`
- Distinct FIPS: `3,200`
- Fill profile (column-level): mean ~`59.77%`, median ~`70.81%`
- Source: CHR yearly county files in `/Users/allegrasaggese/Dropbox/Mental/Data/raw/mental`

### 2) Mortality disaggregated panel
- File: `2026-02-24_mortality_sex_race_disagg.csv`
- Location: `/Users/allegrasaggese/Dropbox/Mental/Data/clean`
- Unit of observation: `fips-year-race-sex`
- Coverage: years `2002-2016` (15 years), `51` states including DC
- Rows/cols: `100,581 x 11`
- Distinct county-years: `40,113`
- Distinct FIPS: `2,921`
- Source: CDC tranche files in `/Users/allegrasaggese/Dropbox/Mental/Data/raw/cdc`
- Race set currently present: `White`, `Black or African American`, `Asian or Pacific Islander`, `American Indian or Alaska Native`
- Sex set currently present: `Female`, `Male`

### 3) MH + mortality combined panel
- File: `2026-02-24_mh_mortality_fips_yr.csv`
- Location: `/Users/allegrasaggese/Dropbox/Mental/Data/clean`
- Unit of observation: `fips-year` (outer union)
- Coverage: years `2002-2023` (22 years)
- Rows/cols: `66,014 x 335`
- Distinct FIPS: `3,200`
- Fill profile (column-level): mean ~`40.37%`, median ~`47.96%`

## QA CSV Dictionary
### A) Core QA files (`qa-health` folder)
#### `2026-02-24_qa_health_mh_file_inventory.csv`
- Columns: `file`, `year`, `n_rows`, `n_cols`, `n_dup_cols_after_clean`
- Purpose: per-input-file sanity checks for CHR ingestion and column deduping.

#### `2026-02-24_qa_health_mh_column_presence.csv`
- Columns: `column`, `dfs_with_col`, `avg_fill_pct_when_present`
- Purpose: MH column presence/fill profile used to drive keep/drop filtering.

#### `2026-02-24_qa_health_stage_fill.csv`
- Columns: `stage`, `column`, `n_rows`, `n_non_null`, `fill_pct`
- Purpose: missingness by stage and by variable (where fill drops enter).

#### `2026-02-24_qa_health_stage_keys.csv`
- Columns: `stage`, `keys`, `n_rows`, `n_unique_keys`, `n_duplicate_rows`, `note`
- Purpose: key integrity checks by stage.

#### `2026-02-24_qa_health_overlap_by_year.csv`
- Columns: `year`, `rows`, `mh_rows`, `mort_rows`, `overlap_rows`
- Purpose: yearly overlap diagnostics between MH and mortality in the union panel.

### B) Supplemental audit files (workspace)
Location:
- `/Users/allegrasaggese/Documents/GitHub/mentalhealth/interim-dfs-copy`

#### `2026-02-24_mh_columns_kept_vs_dropped.csv`
- Columns: `column`, `dfs_with_col`, `avg_fill_pct_when_present`, `selected_by_rule`, `present_in_final_mh`, `dropped`, `drop_reason`
- Purpose: full ledger of MH variable selection outcomes.

#### `2026-02-24_mh_columns_dropped_only.csv`
- Same schema as above.
- Purpose: dropped-only subset for quick review.

#### `2026-02-24_mh_columns_drop_summary.csv`
- Columns: `metric`, `value`
- Purpose: compact summary counts of kept vs dropped columns and drop reasons.

#### `2026-02-24_merge_columns_summary.csv`
- Columns: `descriptor`, `status`, `source_file`, `source_nonkey_cols`, `expected_merged_cols`, `present_in_merged`, `dropped_from_merged`
- Purpose: dataset-level merge inclusion/drop summary in the current big merged output.

#### `2026-02-24_merge_columns_detail.csv`
- Columns: `descriptor`, `column`, `in_merged`
- Purpose: column-level keep/drop indicator by descriptor in current merged file.

#### `2026-02-24_mh_fill_delta_after_rural_filter.csv`
- Columns: `column`, `fill_all_pct`, `fill_rural_pct`, `delta_rural_minus_all_pp`
- Purpose: variable-level coverage change after rural filter (`non_large_metro == 1`).

## Key Findings
### MH filtering outcomes
- Total cleaned MH columns evaluated: `1,129`
- Kept in final MH output: `310`
- Dropped: `819`
- Main drop mode: low presence across years (`dfs_with_col < 8`)

Interpretation:
- Many dropped variables are high quality when present but only appear in a small number of years. The current rule prioritizes longitudinal consistency over episodic high-quality variables.

### Rural filter impacts on MH coverage
- MH rows before filter: `44,709`
- MH rows after filter: `37,908` (`84.79%` retained)
- Largest coverage losses are concentrated in mortality/violence-related indicators, for example:
- `infant_mortality_*` and `homicides_*` fields decline by about `4-5.3` percentage points in fill.

### Merge inclusion status (current merged output)
Current merged file reviewed:
- `/Users/allegrasaggese/Dropbox/Mental/Data/merged/2026-02-24_full_merged.csv`

Observed status:
- `mentalhealthrank_full`: included
- `population_full`: included
- `cafo_ops_by_size_compact`: included
- `crime_fips_level_final`: partially included (one `n_rows` collision issue)
- `mh_mortality_fips_yr`: excluded in current merged output (all expected columns absent)

## Rates to Percentages Plan
### Goal
Standardize mortality metrics to percentages where this improves interpretability while retaining count/rate fields for diagnostics.

### Conversion formulas
Given subgroup deaths `D`, subgroup population `P`, county-year total deaths `D_total`, county-year total population `P_total`:
- Crude incidence per 100k: `rate_100k = D / P * 100000`
- Percent of subgroup population dying: `pct_subgroup_deaths_of_subgroup_pop = D / P * 100 = rate_100k / 1000`
- Percent of county-year deaths contributed by subgroup: `pct_of_county_deaths = D / D_total * 100`
- Subgroup population share within county-year: `pct_subgroup_pop_of_county_pop = P / P_total * 100`

### Recommended output columns
Add to county-year mortality panel (and disaggregated where applicable):
- `mortality_pct_of_subgroup_pop`
- `mortality_pct_of_county_deaths`
- `mortality_pct_subgroup_of_county_pop`
- keep existing `deaths`, `population`, and `crude_rate_per_100k` columns.

### Guardrails
- Do not compute percentage fields when denominator is missing or zero.
- Keep missing as `NA`; do not coerce to `0`.
- Add denominator provenance fields so each percentage can be audited.

## County-Year Totals Mortality Data Plan
### Why this is needed
Current disaggregated mortality is not complete enough to act as canonical county-year totals.

Empirical completeness signal from current disaggregated file:
- Expected full subgroup combos per county-year: `8` (`4 races x 2 sexes`)
- County-years with all 8 combos: ~`0.54%`
- County-years with only 1-2 combos: ~`68.89%`

Implication:
- Summing disaggregated rows will understate totals in many county-years due to suppression/partial coverage.

### Required additional dataset
Acquire a non-disaggregated county-year mortality totals dataset (all-cause) with at least:
- `fips`
- `year`
- `deaths_total`
- `population_total`
- `crude_rate_total_per_100k` (optional if recomputable)

Prefer source directly from CDC county-year totals (same concept/timeframe as disaggregated pull).

### Integration design
- Keep disaggregated dataset for subgroup analysis.
- Use canonical non-disaggregated totals for county-year aggregate mortality outcomes.
- Merge disaggregated-to-total only for derived shares (subgroup share of county deaths/pop).
- Add QA checks:
- compare `sum(disaggregated deaths)` vs `deaths_total` by county-year
- compute completeness ratio and flag low-completeness county-years

## Immediate Next Steps
1. Move or copy supplemental CSVs from workspace into `qa-health` if you want all audit outputs in one folder.
2. Add percentage-field generation in the mortality pipeline with strict denominator checks.
3. Add a canonical county-year total mortality pull and link it in the health build.
4. Add a completeness score column to disaggregated outputs before any aggregation use.
