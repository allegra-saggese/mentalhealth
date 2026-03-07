# FSIS Slaughterhouse Build Plan (Establishment-Year -> County-Year)

## Objective
Build a defensible county-year slaughterhouse presence dataset where:
- a county-year is flagged as `1` if at least one slaughterhouse establishment is present in any monthly file that year, and
- location is attached from MPI directory data, with backward fallback when geography is missing in a given year.

Primary first output is an `establishment_id x year` dataset that keeps all establishments and adds operation categorization.

---

## Inputs
- Raw folder: `/Users/allegrasaggese/Dropbox/Mental/Data/raw/fsis`
- File families:
  - Establishment Demographic files (monthly/periodic; mixed legacy and modern schemas)
  - MPI Directory files (contain location fields such as county and FIPS)

---

## Required Inventory Table
Create a full inventory table before merges:
- Output file: `YYYY-MM-DD_fsis_file_inventory.csv`
- Suggested output dir: `/Users/allegrasaggese/Dropbox/Mental/Data/clean` (or QA subfolder)

Inventory columns:
- `file_path`
- `file_name`
- `family` (`demographic` / `mpi` / `other`)
- `file_ext`
- `year_inferred`
- `month_inferred`
- `snapshot_date_inferred`
- `header_row_used`
- `sheet_name_used`
- `n_rows_read`
- `n_cols_read`
- `key_style` (`legacy_30ish`, `modern_253col`, etc.)
- `read_status` (`ok` / `error`)
- `error_msg`

This table is a hard requirement and is saved even if downstream steps fail.

---

## Dataset Build Steps

### 1) Discover + classify files
- Recursively list all files in `raw/fsis`.
- Exclude temporary files (`~$*`) and `.DS_Store`.
- Classify each file into `demographic`, `mpi`, `other` using filename/path patterns.
- Parse year/month/snapshot date from filename and parent folder text.

### 2) Parse demographic files with robust header detection
- Legacy files (30-ish columns): detect header row by searching first rows for tokens like `EstNumber`, `EstID`, `EstablishmentNumber`, `EstablishmentID`.
- Modern files (253 columns): usually header row 0 and snake_case names.
- Standardize key columns to canonical names:
  - `establishment_id`
  - `establishment_number`
  - `establishment_name`
  - slaughter-related fields (`slaughter`, `meat_slaughter`, `poultry_slaughter`, species-level slaughter flags, volume categories)

### 3) Define row-level operation signals and categories
- For each establishment-row in a monthly demographic file, create:
  - `is_slaughterhouse_row`
  - `is_processing_row`
  - `operation_category_row`
- Default rule:
  - `1` if any slaughter indicator is positive/present (`slaughter == yes`, `meat_slaughter == yes`, `poultry_slaughter == yes`, species slaughter flag present, or non-null slaughter volume category).
  - `0` otherwise.
- Processing is also detected from processing indicator fields, activities text (contains `process*`), and processing volume category.
- Row categories:
  - `both_slaughter_and_processing`
  - `slaughter_only`
  - `processing_only`
  - `other_or_unclear` (has operation context but not cleanly classifiable by string flags)
  - `neither_signal`

### 4) Collapse monthly demographic records to establishment-year
- Group by `establishment_id` + `year` (fallback key: `establishment_number` when ID missing).
- Annual presence rule:
  - `is_slaughterhouse_year = 1` if any monthly record in that year has `is_slaughterhouse_row = 1`.
- Also keep annual processing + category fields:
  - `processing_present_year`
  - `operation_category_year`
- Keep diagnostics:
  - `n_files_seen_in_year`
  - `first_snapshot_date`
  - `last_snapshot_date`
  - `source_file_count`

Primary outputs from this stage:
- `YYYY-MM-DD_fsis_establishment_year_all.csv`
- `YYYY-MM-DD_fsis_establishment_year_slaughterhouse.csv`

### 5) Parse MPI files for geography
- Read MPI directory files across months/years.
- Canonical location fields:
  - `county`
  - `fips_code`
  - `state`
  - optionally `city`, `zip`, `latitude`, `longitude`
- Keep `establishment_id`, `establishment_number`, and snapshot date/year.

### 6) Build establishment-year geography table
- Within each `establishment_id x year`, pick the best same-year location:
  - prefer non-null `fips_code`;
  - if multiple values exist, use most frequent (`mode`) and flag conflict.
- Mark `geo_source = "same_year"` when found.

### 7) Backward fallback geography rule (your requirement)
- If an establishment-year has no geography in that same year, use the most recent prior year where that `establishment_id` has non-null geography.
- Do not use future years for fallback.
- Track:
  - `geo_source = "prior_year_fallback"`
  - `geo_fallback_from_year`

### 8) Additional "from wherever possible" geography fallback
- After same-year and prior-year fallback, fill remaining geography gaps from undated/reference location files (for example, directory files without a reliable year token).
- This step fills only still-missing fields and is tracked as:
  - `geo_source = "undated_reference"`

### 9) Merge slaughterhouse establishment-year with geography
- Merge annual slaughterhouse panel to annual geography panel by `establishment_id` + `year`.
- Fallback to `establishment_number` only when `establishment_id` is unavailable.
- Keep only establishments with `is_slaughterhouse_year = 1` for county presence outputs.
- Preserve unmatched rows in a QA file.

### 10) Build county-year presence panel
- Group establishment-year rows by `fips_code` + `year`.
- Output:
  - `slaughterhouse_present = 1` if `n_establishments > 0`
  - `n_establishments`
  - counts by subtype/category (meat/poultry/both/slaughter-only)

Output:
- `YYYY-MM-DD_fsis_county_year_slaughterhouse_presence.csv`

---

## QA Outputs (Required)
- `YYYY-MM-DD_fsis_file_inventory.csv`
- `YYYY-MM-DD_fsis_establishment_year_unmatched_geo.csv`
- `YYYY-MM-DD_fsis_geo_conflicts.csv` (same establishment-year, multiple FIPS/county)
- `YYYY-MM-DD_fsis_summary_metrics.csv` with:
  - n files by family/year
  - n establishment-years
  - pct with same-year geo
  - pct with fallback geo
  - pct still missing geo
  - n county-years flagged

---

## Key Choices and Assumptions
- Primary key is `establishment_id`; `establishment_number` is fallback only.
- Year comes from file metadata (filename/path date parsing), not spreadsheet content.
- Annual slaughterhouse presence uses "any-month-in-year" logic.
- All establishment-year rows are kept in the `..._establishment_year_all.csv` output; slaughterhouse subsets are derived views, not destructive filters.
- Geography fallback is backward-only (prior appearance of same establishment ID).
- If same-year/prior-year geography is still missing, undated reference files are used as last fallback.
- If geography still missing after backward fallback, establishment-year stays in QA unmatched file and is excluded from county-year panel.
- Size and volume classifiers are retained in the establishment-year output (`size_classifier_mode`, `processing_volume_category_mode`, `slaughter_volume_category_mode`).
- No deletion of raw files; all transforms are reproducible and documented by inventory + QA artifacts.

---

## Implementation Note
Planned implementation target is `script0e-fsis-slaughterhouses.py`, with outputs written to the same clean/QA conventions used in the existing project scripts.
