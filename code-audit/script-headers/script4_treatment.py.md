# Original header of `script4_treatment.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4_treatment.py  --  SHARED TREATMENT / SAMPLE MODULE for all script4* analyses.

Importable (underscore name) because stage-numbered scripts use dashes and cannot be
imported. Every script4* file loads its panel and treatment variables from here, so
that a treatment variable is defined in EXACTLY ONE place.

Why this module exists
----------------------
In the previous single-file script4-model-test.py, treatment variables were built
inline in four different blocks, some guarded by `if col not in df.columns`. That let
two different missingness rules coexist for the same underlying variable:
  - Blocks 1/2 used  `cafo_dairy_large.fillna(0) > 0`   (NaN -> untreated)
  - Blocks 1b-1e used `(x > 0).where(x.notna())`         (NaN -> NaN)
CAFO is 100% missing in 2000-2001 (first ag census is 2002), so the first rule silently
codes 158 counties that demonstrably HAD a large dairy CAFO in 2002 as untreated in
2000-01 -- 7.1% of the Deaths of Despair estimating sample.

RULE ENFORCED HERE: treatment is NaN wherever the underlying CAFO count is NaN.
UPDATED 2026-10-16: a blank CAFO count from the first census year (2002) onward is
first set to ZERO in load_panel (no census row = no operations; verified against
NASS, see the block in load_panel). Only 2000-01 stay NaN, because they were never
measured. The fillna(0) that caused the 158-county problem was applied to ALL years
including 2000-01; this one is not.

Size definition
---------------
`cafo_dairy_large` = dairy operations with 500+ milk cows (USDA NASS top inventory
bin, code 7; see script0b-usda-raw.py:663). medium = 200-499, small = <200.
NOTE: EPA's regulatory "Large CAFO" threshold for dairy is 700+ mature cows. USDA
publishes no 700 cutpoint, so 500+ is the nearest available bin. This is NOT the
regulatory definition and must be stated as such in the paper.

Treatment timing
----------------
CAFO counts come from the ag census (2002, 2007, 2012, 2017, 2022) and are
forward-filled to inter-census years by script1b. ALL within-county treatment
variation therefore occurs at those 5 dates only. Event "timing" carries up to 5
years of measurement error: a county recorded as changing in 2012 actually changed
somewhere in 2008-2012.

Exits
-----
Per team decision 2026-09-18, treatment is ABSORBING: once a county has a positive
change event it stays treated. Exits and contractions never switch treatment back
off. Exit counties are NOT dropped -- dropping them would select on a treatment path
that may correlate with the outcome.

````
