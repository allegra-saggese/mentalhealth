# Original header of `script1b-generate-panel.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

Builds one county-year merged panel from the latest version of each clean file.

Panel frame: ALL counties where rural-key non_large_metro == 1.
  The rural-key (NCHS urban-rural classification) is the backbone of the panel.
  Every other dataset — including CAFO — is LEFT JOINED onto this frame, so
  rural counties with no data for a given source are always present as NaN rows
  (or 0 for CAFO counts, see zero-fill section below).

Key design decisions:

- CAFO (USDA Agricultural Census) — FORWARD FILL:
  The USDA Agricultural Census is conducted every five years (2002, 2007, 2012,
  2017, 2022). The clean compact (script0b) contains only those five census years.
  To produce a year-on-year panel, census values are forward-filled to all
  intervening years: 2002 values carry through 2003–2006, 2007 values through
  2008–2011, and so on. This is the intended imputation strategy — each census
  is treated as the best available observation until the next one supersedes it.

  Two additional CAFO adjustments applied after forward-fill:
    (a) Zero-fill: rural counties absent from the USDA Census in a census year
        are confirmed zero-operation counties (the census is exhaustive). Those
        county-years are filled with 0 before forward-filling, so the zero
        propagates correctly to inter-census years.
    (b) Non-census years for truly absent counties remain NaN until a census
        year establishes a value, at which point forward-fill takes over.

- CDC deaths-of-despair: three-state missingness flag (cdc_in_query,
  deaths_is_zero, crude_rate_from_census_pop). See inline comments.

- CHR (County Health Rankings): CHR rate-only variables get imputed count
  columns (*_count_imputed). Num/denom variables get a *_ratio_flag QA column.

- FSIS (slaughterhouses): available 2017–present only. Exported separately as
  panel_fsis.csv (2017+); main panel.csv drops FSIS columns entirely so the
  full time-series is not truncated.

Outputs (all to Data/merged/):
  panel.csv               — full panel, all years, no FSIS columns
  panel_05_10.csv         — 2005–2010 slice
  panel_10_20.csv         — 2010–2020 slice
  panel_census_years.csv  — USDA census years only (2002, 2007, 2012, 2017, 2022)
  panel_fsis.csv          — 2017+ with FSIS columns

````
