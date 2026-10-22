# Original header of `script2b-final-visuals.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script2b-final-visuals.py

Core descriptive visualizations — CAFO / FSIS exposure vs. outcomes.

Purpose:
    Produces analytically useful figures rather than QA diagnostics.
    All X-axis treatment variables are expressed as log(ops per 10k population + 1)
    to remove the mechanical correlation between raw counts and county size.
    Outcome variables from County Health Rankings are already population-standardized
    (rates or percentages) and are used as-is.

Figures produced:
    A1 — Binned scatter: CAFO total ops (per capita, log) vs. all 6 outcomes.
         Pooled 2010–2015. 20 equal-count bins, mean ± 95% CI, OLS fit.
    A2 — Binned scatter: CAFO by animal type (cattle / hogs / chickens)
         vs. 3 core outcomes. Same period.
    A3 — Binned scatter: FSIS establishments (per capita, log) vs. all outcomes.
         2017 cross-section only (FSIS data not available before 2017).
    B  — County time series: 1 representative county per state, 2010–2015.
         CAFO exposure on left axis, Poor Mental Health Days on right axis.
    C  — Outcome cross-correlation heatmap (Spearman, 2010–2015).
         Shows which outcomes co-move before committing to a primary Y variable.
    D  — Violin plots: outcome distributions by CAFO intensity quartile.
         Pooled 2010–2015. Inner lines show Q1/median/Q3 within each violin.
    E  — Box-and-whisker (median, no mean): same CAFO quartile grouping.
         Overlaid with jitter of raw county observations. Median line in red.
    F1 — Choropleth map: log(CAFO ops per 10k pop + 1), rural US counties, 2012.
    F2 — Choropleth map: Poor Mental Health Days, rural US counties, 2012.
         (F maps saved as both interactive HTML and static PNG)

Outputs to: Dropbox/Mental/Data/output/figs/core-visuals/ (PNGs/HTML)
            Dropbox/Mental/Data/output/tables/core-visuals/ (CSV)

Dependencies: packages.py, functions.py (shared project utilities)

````
