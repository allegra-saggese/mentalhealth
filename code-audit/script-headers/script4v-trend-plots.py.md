# Original header of `script4v-trend-plots.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4v-trend-plots.py  --  WHAT DO THE OUTCOME TRENDS LOOK LIKE?  (added 2026-10-17)

Descriptive pictures and simple time regressions, no controls and no fixed effects, so the raw
paths are visible before any model is applied. Nothing here is an estimate of a treatment effect.

GROUPS (county level, from the zero-filled CAFO counts in load_panel)
    never large dairy        no large dairy operation in any census wave                      (comparison)
    enters                   first goes from 0 to 1+ large dairies at some wave (first entry)
    expands only             a positive change in an existing large-dairy count, never enters
    large dairy throughout   has a large dairy at some wave, never enters or expands

FIGURES  (-> Data/output/figs/script4v/)
    F1  outcome means by calendar year, one line per group, 7 panels (own y-axis each)
    F2  event-time path: mean change in the outcome since the year before treatment (g-1), treated
        counties vs comparison counties, for each of the two treatment definitions
    F3  pre-treatment slope difference by cohort: treated minus comparison, in outcome SDs per year

TABLES  (-> Data/output/tables/script4v/): the numbers behind every figure.

SIMPLE TIME REGRESSION (F3 / pre_window_slopes.csv)
    For cohort g, window = the 5 years before treatment (g-5 ... g-1), counties with >= 3 years of data:
        Y_it = a_i + b * year_t + c * (treated_i x year_t) + e_it       county FE, SE clustered by state
    c = how much faster (or slower) treated counties' outcomes were already moving per year.
    Outcomes that start late (poor mental-health days from 2010, frequent mental distress from 2016) have
    no usable window for early cohorts; those cells are simply absent.

COLOURS: categorical slots 1-4 of the reference palette (validated; the aqua/yellow contrast warning is
met with a legend, different line styles and the CSV tables).

````
