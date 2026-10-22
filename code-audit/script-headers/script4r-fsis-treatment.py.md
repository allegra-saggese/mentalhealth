# Original header of `script4r-fsis-treatment.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4r-fsis-treatment.py  --  SLAUGHTERHOUSES AS THE TREATMENT (added 2026-10-12)

Sept 30 feedback: test slaughterhouses as a treatment in their own right, not
only as the FSIS term interacted with dairy (A4).

TREATMENTS
----------
  F1  Count      number of FSIS establishments with slaughter activity in the
                 county-year (n_slaughterhouse_present_establishments_fsis).
                 A county absent from the FSIS directory has 0 plants, so NaN -> 0
                 INSIDE the FSIS window (2017-2023); outside it, NaN.
  F2  Entry      absorbing: 1 from the first year the county's slaughter count
                 RISES (an additional plant, whether or not it already had one).
                 Cohort g = that year. Never-rising counties are controls.

IDENTIFICATION LIMITS
---------------------
* FSIS starts in 2017, so an increase is first observable in 2018. Entries
  before 2017 are invisible (left-censoring): a county coded "never" may have
  gained a plant in 2015. Pre-2017 outcome years are used as pre-periods only.
* Crime ends 2021 and Deaths of Despair 2020, so only the 2018-2021 / 2018-2020
  cohorts identify those outcomes. Mental health runs to 2023.
* The 2020 cohort is large (small-plant entry during COVID supply disruptions);
  read cohort-specific estimates with that in mind.

MODELS
------
  R0  Event counts by year: how many counties first gain a plant (power check)
  R1  TWFE, F1 count, 2017-2023, county + year FE, step-8 controls
  R2  TWFE, F2 entry, county + year FE, step-8 controls, sample 2012-2023
  R3  Event study around entry, leads -5..-2, lags 0..5 (ref -1)
  R4  LP-DiD on F2 cohorts (clean controls), horizons -5..5
  MDE for R1/R2 at 80% power: 2.8 x SE, in within-county SDs of the outcome.

All SEs clustered by state. Outputs -> Data/output/tables/script4r/

````
