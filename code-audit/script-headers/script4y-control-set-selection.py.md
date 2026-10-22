# Original header of `script4y-control-set-selection.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4y-control-set-selection.py  --  HOW THE CONTROL SET WAS CHOSEN: ADD ONE IN, TAKE ONE OUT  (added 2026-10-17)

Replaces script4b (adding controls along a fixed order) and script4c (summarising that run), and adds the
take-one-out checks. 4b and 4c are kept, unchanged, as the record of the earlier 09-23 run.

THE CONTROL SET. 14 candidate controls (COVARIATE_ORDER); the 8 with full coverage and testable pre-trends are the
final control set (COVARIATE_ORDER[:8]). That choice was PRE-COMMITTED (decision D-16, revised in D-31): it was fixed
before any coefficient from this script was read, on grounds of coverage, causal role (confounder, mediator, collider)
and whether the series can be tested for pre-trends. THIS SCRIPT DOES NOT SELECT THE CONTROL SET. It documents how much
each control matters for the treatment coefficient. Choosing a specification because of what these paths show and then
quoting conventional standard errors would be post-selection inference.

FOUR FAMILIES of regressions, for each reported treatment (T1, T3, T4, E2) x outcome (7), all with county FE +
year FE and SE clustered by state:
    A1  cumulative add-one-in      controls enter one at a time in COVARIATE_ORDER (step 0 = none ... 14 = all).
                                   Order-dependent; the sample shrinks after step 8 (partial-coverage controls).
    A2  each added alone           the reference has no controls; each of the 14 is added by itself.
                                   Order-free.
    B1  take-one-out of the 8      reference = the final 8; drop each of them in turn.
    B2  take-one-out of all 14     reference = all 14; drop each in turn.
SAMPLE. "own" = complete cases for that regression's variables. "common14" = complete cases for ALL 14 controls, so
differences between regressions are due to the control and not to a changing sample (A2 and B2 only; for A1 and B1
the first 8 controls leave the sample intact).
Every regression stores beta, SE, p, N, switchers (counties whose treatment changes) and the largest within-transformed
VIF. Change is measured against that family's reference: 100 * (beta - beta_ref) / |beta_ref|.

OUTPUTS -> Data/output/tables/script4y/
    <date>_regressions.csv             every regression
    <date>_path_summary.csv            A1: per treatment x outcome, movement from step 0 to 8 to 14
    <date>_covariate_summary.csv       per covariate: median |change| across the 28 treatment x outcome cells, by family
    <date>_selection_table.csv         the exhibit: one row per candidate control with coverage, causal role,
                                       pre-trend verdict (script4j), reason for exclusion, and the movement above
Usage:  python3 script4y-control-set-selection.py            full
        python3 script4y-control-set-selection.py --quick    one treatment, two outcomes

````
