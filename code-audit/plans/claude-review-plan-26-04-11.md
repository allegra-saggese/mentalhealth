script1* — What Each Script Does

  ---
  script1a-QA-indiv.py — Population Column QA

  Purpose: Scans every *_full_merged*.csv in Data/merged/ and audits all population-related columns across those files.

  Inputs:
  - Data/merged/*_full_merged*.csv (all merged files, not just the latest)

  Outputs → allegra-dropbox-copy/interim-data/:
  - {today}_qa_population_columns_summary.csv — per-column fill rate, zero rate, min/median/max
  - {today}_qa_population_columns_pairwise.csv — correlation + mean abs/pct diff between each pop column and a chosen reference column
  - {today}_qa_population_columns_catalog.csv — flat list of (file, column) pairs

  Logic: Regex-matches column names containing "pop", "population", "popestimate", "population_estimate". Picks a reference column by
  priority (population_population_full > population_full_* > alphabetical first). Computes pairwise diagnostics vs. that reference.

  ---
  script1b-audit-mental-vars.py — Mental Health Coverage Audit

  Purpose: Full audit of panel integrity and mental health variable coverage in the merged panel.

  Inputs:
  - Latest *_full_merged.csv from Data/merged/

  Outputs → Data/merged/figs/panel-sumstats-by-farms/:
  - qa_panel_integrity_mental_focus.csv — row count, county count, year range, duplicate count
  - qa_panel_source_duplication_counts.csv — how many county-years have n_rows > 1 per source
  - qa_mental_focus_variable_inventory.csv — fill rates + year coverage for all MH-related columns
  - qa_mental_outcome_core_overall.csv — fill + distribution stats for 4 core outcomes
  - qa_mental_outcome_core_by_year.csv — fill rates for core outcomes broken out by year
  - qa_mental_outcome_counties_any.csv — how many counties ever have data per variable
  - qa_mental_crosssource_duplicate_check.csv — correlation + exact-match % between same variable from two sources
  - qa_mental_coverage_memo.md — plain-text summary

  Logic: Loads the merged panel, deduplicates on (fips, year), filters to MH-related columns (suffix _mentalhealthrank_full + any of 8
  mental keywords), then runs a sequence of coverage checks. Checks 2 cross-source pairs to see if the same variable from different raw
  sources agrees.

  ---
  script1c-merge-dataclean.py — Build the Main Analysis Panel

  Purpose: This is the core merge script. It takes clean files from Data/clean/, applies the rural filter, aggregates to county-year, and
  joins everything into a single analysis panel.

  Inputs (latest file per descriptor from Data/clean/):
  - cafo_ops_by_size_compact — BASE dataset (CAFO operations, cattle/hogs/chickens × small/medium/large)
  - cdc_county_year_deathsofdespair — CDC deaths-of-despair county-year mortality
  - crime_fips_level_final — crime data
  - fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip — FSIS slaughterhouse data
  - mentalhealthrank_full — County Health Rankings mental health
  - population_full — census population
  - rural-key file (any descriptor containing "rural-key") — provides non_large_metro flag

  Outputs → Data/merged/:
  - {today}_full_merged.csv — main panel
  - {today}_full_merged_2005_2010.csv
  - {today}_full_merged_2010_2020.csv
  - {today}_full_merged_census_years.csv (years: 2002, 2005, 2007, 2012)

  Logic: Filters everything to rural counties (non_large_metro == 1). Collapses each source to one row per (fips, year): numeric columns are
   summed (with min_count=1 to preserve NaN for all-missing groups), text columns take first observed value. All columns get suffixed with
  their source descriptor. Everything is left-joined onto the set of rural (fips, year) keys. Also builds a separate CAFO animal×size pivot
  block (cafo_cattle_small, cafo_hogs_medium, etc.).

  ---
  script1d-cdc-cruderate-sensecheck.py — CDC Crude Rate Validation

  Purpose: Verifies that CDC's reported crude death rates are consistent with what you'd calculate from their raw deaths data plus census
  population.

  Inputs:
  - Latest *_full_merged.csv from Data/merged/

  Outputs:
  - Overwrites (re-exports) the full merged panel + 3 slices with QA columns appended, all with today's date
  - QA CSVs → Data/merged/figs/panel-sumstats-by-farms/:
    - {today}_qa_cdc_cruderate_sensecheck_overall.csv
    - {today}_qa_cdc_cruderate_sensecheck_by_year.csv
    - {today}_qa_cdc_cruderate_sensecheck_outliers_top1000.csv

  Logic: Recalculates crude rate as deaths / census_pop × 100,000 and as deaths / cdc_pop × 100,000. Computes differences from the
  CDC-reported rate, absolute differences, % differences, and z-scores (both global-SD and within-year-SD). Appends all diagnostic columns
  to the merged panel and re-exports it alongside QA summaries.

  ---
  ---
  Reviewer Assessment — Bugs, Errors, and Priority

  ---
  Bug (High Priority)

  script1b, line 71–88: Panel integrity check always reports 0 duplicates.

  The script deduplicates on (fips, year) at line 71 and then computes n_duplicate_fips_year_rows at line 88 — on the already-deduped frame.
   This metric will be 0 by construction and is therefore meaningless. The integrity check needs to run on the raw frame before
  deduplication.

  ---
  Design Errors (Medium Priority — could affect results or reproducibility)

  script1c: CAFO is the base dataset, so counties with no CAFO data are dropped entirely.

  The panel is built by filtering allowed_keys to rural counties, then left-joining CAFO as the base. Any rural county-year that doesn't
  appear in the CAFO data is excluded. This is a major selection decision — you're implicitly conditioning on CAFO presence — and it is not
  commented anywhere in the code. An academic reviewer would flag this immediately.

  script1c, _reduce_to_county_year: Numeric aggregation uses sum for all numeric columns.

  Summing makes sense for count variables (CAFO operations, n_establishments). It is wrong for rate/level variables (crude death rates,
  mental health scores). If any source contributes more than one row per county-year, rates will be doubled. The n_rows tracking gives you
  an after-the-fact way to detect this, but the aggregation choice itself is inappropriate for mixed variable types.

  script1d: Re-implements latest_file, to_num, and norm_key locally instead of using functions.py.

  Every other script in the 1* series imports from functions.py. script1d rolls its own versions of the same three helpers. If the key
  normalization logic in functions.py is updated (e.g., FIPS handling), script1d will silently diverge. This is a maintenance risk and an
  inconsistency that would concern a replicability reviewer.

  script1d: Hardcoded absolute paths.

  MERGED_DIR and OUT_QA_DIR are hardcoded to /Users/allegrasaggese/Dropbox/Mental/... instead of using db_data and db_me from packages.py.
  This breaks portability for any other collaborator or machine.

  script1d: File overwrite ordering with script1c is undocumented and fragile.

  Both script1c and script1d output {today}_full_merged.csv. If run on the same day, the second one overwrites the first. The intended
  pipeline order (1c → 1d, so 1d's version is the final one with QA columns) is nowhere documented. If someone runs 1c after 1d, the QA
  columns are silently lost.

  ---
  Documentation/Clarity Gaps (Lower Priority)

  script1c: Rural filter (non_large_metro == 1) is undocumented.
  It should have a comment explaining what this variable means and why this is the project's sample restriction.

  script1d: Global vs. within-year z-scores mixed in by-year summary.
  The by_year QA table reports share_abs_z_gt_* thresholds using the global SD (not year-specific), so those z-scores are cross-year, not
  within-year. The column z_diff_from_zero_sd_within_year exists but is not used in the summarize() function. The by-year table would be
  more interpretable if it used the within-year SD.

  script1c: 80% parse-rate threshold for numeric classification is arbitrary and undocumented.
  The cutoff that determines whether a column gets treated as numeric or text has no justification in comments. Columns with suppression
  codes like (d) could easily fall below or above this threshold in ways that are not obvious.

  script1b: today_str is redefined locally when it is already imported from functions.py.
  Minor, but a reviewer would notice the redundancy.