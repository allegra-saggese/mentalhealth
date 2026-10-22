# Original header of `script4p-full-appendix.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4p-full-appendix.py  --  ONE document: every regression and balance table.

Combines the fragments written by
    script4d-balance-panel.py     -> _BODY_balance.tex
    script4n-regression-tables.py -> _BODY_regressions.tex
plus the two covariate-exclusion tables, into a single compilable appendix.

NO estimation and NO table construction happens here. The tables are built once,
in the scripts that own them, and this file only orders and frames them. Editing
a table means editing 4d or 4n and re-running; nothing here needs to change.

RUN ORDER
---------
    python3 script4j-control-diagnostics.py   # control pre-trends, VIF
    python3 script4a-design-estimates.py      # A1, A2, A4, A5, A6, A7
    python3 script4r-fsis-treatment.py        # slaughterhouse treatment
    python3 script4d-balance-panel.py         # balance fragment
    python3 script4n-regression-tables.py     # regression fragment
    python3 script4p-full-appendix.py         # this file

Output -> LATEX_OUT/FULL_RESULTS_APPENDIX.tex / .pdf

````
