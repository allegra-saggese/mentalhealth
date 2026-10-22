# Original header of `script4n-regression-tables.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4n-regression-tables.py  --  regression tables in standard paper format.

REWRITTEN 2026-10-12 (run 2). Reads the latest CSVs from script4a (A1, A2, A4,
A5, A6, A7, VIF, Conley) and script4r (slaughterhouse R0-R4). NO estimation.

FORMAT (applies to every table)
-------------------------------
    columns  = outcomes (1)-(7), grouped Mental health | Mortality | Arrests
    panels   = treatments (Panel A, B, ...), one coefficient row each,
               standard error in parentheses directly below
    bottom   = Observations, Counties, State clusters, Controls, FE rows
    stars    = *** p<0.01, ** p<0.05, * p<0.10   (state-clustered p-values)
Notes go in the document, not inside the tabular, so each fragment can be
\input into a paper, a slide or the appendix unchanged.

A3 was dropped 2026-10-12 and is no longer read.

REPORTED vs ROBUSTNESS (2026-10-15, team feedback). script4_treatment.REPORT_*
sets what appears in the MAIN exhibits: treatments T1, T3, T4, E2; county + year
FE (A2); state-clustered SEs; Callaway-Sant'Anna covariate-adjusted. Nothing is
dropped: the other specifications (T2, A1 state FE, CS-unconditional, LP-DiD,
Conley, the all-treatment grid) are still written, as separate tables collected
in _BODY_robustness.tex and the second half of REGRESSION_TABLES.tex.

Output -> LATEX_OUT/tables/*.tex, LATEX_OUT/REGRESSION_TABLES.tex (+ .pdf if
pdflatex is available), LATEX_OUT/_BODY_regressions.tex for script4p.

````
