# Original header of `script4d-balance-panel.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4d-balance-panel.py  --  rebuilds BALANCEpanel.pdf on the corrected panel.

REPLACES the 2026-09-18 version of script4d-power-balance.py, which produced
BALANCEpanel.pdf against the pre-CHR-fix panel, the legacy 12-treatment grid,
6 outcomes and the retired CORE_9 control set.

WHAT CHANGED
------------
    panel       2026-09-23 (10 controls re-sourced to correct data years)
    treatments  T1-T4 + E2   (was 12 legacy definitions; T5 dropped, T6 retired)
    outcomes    7            (was 6; four crime outcomes settled under D-11)
    controls    step-8 pre-committed set, D-16/D-29  (was CORE_9)

WHAT EACH COLUMN MEANS
----------------------
    N                 estimating-sample rows after listwise deletion
    Treated/Control   county-year observations with treatment >0 / ==0. For the
                      continuous treatments (T2, T3, T4) this split is
                      DESCRIPTIVE ONLY -- the estimator uses the full continuous
                      variation, not a dichotomy.
    Counties          total, and those ever treated
    Switch./Trans.    counties whose treatment CHANGES, and the number of changes.
                      Under county FE only switchers identify beta. This is the
                      honest denominator on every result in the deck.
    SE                cluster-robust at the state level, from the actual A2
                      regression (county + year FE, step-8 controls)
    MDE               minimum detectable effect at 80% power, two-sided a=0.05:
                      2.8016 x SE, expressed in WITHIN-COUNTY standard deviations
                      of the outcome. Within-county, not raw, because that is the
                      variation a county-FE model can actually use.

Output -> LATEX_OUT (Data/output/latex-outputs-run2/)
              BALANCEpanel.tex / .pdf        (one table per treatment + outcomes)
              <date>_balance_panel.csv       (same numbers, machine readable)

````
