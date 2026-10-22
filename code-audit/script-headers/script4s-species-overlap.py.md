# Original header of `script4s-species-overlap.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4s-species-overlap.py  --  HOW MANY COUNTIES HAVE LARGE CAFOs OF MORE THAN ONE SPECIES?
(added 2026-10-15, team feedback)

QUESTION
--------
Dairy is the treatment in Part A. If the same counties also host large hog,
cattle or chicken operations, a dairy-only model attributes their effects to
dairy. This script counts how often that happens, by year, so the team can decide
whether to model species jointly. DESCRIPTIVE ONLY: no regressions.

DEFINITIONS
-----------
Species      dairy, beef, hogs, chickens    (the four modelled)
             cattle is NOT a species here: USDA's cattle inventory contains the
             dairy and beef herds, so cattle overlaps dairy and beef by
             construction. Counting it would make nearly every dairy county look
             multi-species. It appears only in the appendix nesting check.
Present      cafo_{species}_large > 0 in the county-year. "Large" is the USDA NASS
             top size bin for that species (500+ milk cows for dairy).
Missing      from 2002 on, a missing large-CAFO count is treated as ZERO (no large
             operation), not as a dropped observation: every county-year from 2002 stays
             in every table. 2000-01 are EXCLUDED from all tables, because no census
             was taken in them and a blank there is an unmeasured year, not a zero
             (same rule as script4_treatment.load_panel). The number of values recoded
             this way, by species and year, is written to <date>_species_nan_as_zero.csv.

CENSUS YEARS VS ALL YEARS
-------------------------
CAFO counts come from the ag census (2002, 2007, 2012, 2017, 2022) and are
forward-filled to the years between (script1b). Years between census waves are
therefore copies of the last wave, not new observations. The pair and
combination tables use census years only so every row is independent evidence.
The by-year count table covers every panel year, flagged `census_year`, because
the regression sample is county-years.

OUTPUTS  ->  Data/output/tables/script4s/   and   Data/output/figs/script4s/
    <date>_species_count_by_year.csv       counties with 0,1,2,3,4 species, every year
    <date>_species_combination_census.csv  exact species combinations, census years
    <date>_species_pair_census.csv         pairwise co-occurrence and P(B | A), census years
    <date>_species_county_summary.csv      county-level: ever / persistently multi-species
    <date>_species_nan_as_zero.csv         how many missing counts were set to zero, by species and year
    <date>_species_cattle_check.csv        appendix: cattle contains dairy and beef
    <date>_F1_species_count_census.png     stacked bars, census years

````
