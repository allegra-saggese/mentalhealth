# Analysis assessment — what has been tried, and how it fares against a causal bar
**2026-10-22** · read-only review; nothing was re-run. Every number below is read from a dated
2026-10-22 result table under `Data/output/tables/` (file named in the last column group).
Decisions referenced as D-nn are in `CONTROL-DECISION-REGISTER.md`; A-numbers are in
`2026-09-27_MODEL-NAMING.md`. A positive coefficient means a worse outcome (more poor mental-health
days, more arrests, more deaths), which is the direction the hypothesis predicts.

## The bar used
An estimate counts as causal-grade only if all four hold:
1. **Estimator**: valid when treatment starts at different times and effects differ across counties.
2. **Pre-trends**: before treatment, treated and comparison counties moved together.
3. **Adjustments**: the result survives county-specific trends, state-by-year effects, and conditioning on past outcomes.
4. **Many tests**: the result survives having tested 7 outcomes and several treatments.

## What was run, and the result

| Approach (script) | What it identifies | Result | Bar (1 / 2 / 3 / 4) | Supports hypothesis? |
|---|---|---|---|---|
| A1 pooled, state + year FE (`script4a`) | Differences between counties | Crime and deaths of despair significantly **negative** (NIBRS total −19.4, p = 0.045); Poor MH Days null | Not causal | No; the sign flips against A2, which signals selection |
| A2 TWFE, county + year FE; T1, T3, T4, E2 (`script4a`) | Change within a county over time | Poor MH Days positive and significant for all four treatments (E2 0.104, p = 0.019). Frequent Mental Distress significant for E2 and T4. Assault family significant. Deaths of despair null. | ✗ / mixed / ✗ / ✗ | Yes on paper |
| TWFE event study, leads test (`script4u`, `script4t`) | Pre-trends | Flat for Poor MH Days (p = 0.43), Frequent Mental Distress (0.15), deaths of despair (0.25). Fails for aggravated assault (0.001), assault (0.000), NIBRS total (0.000), violence index (0.019). | Pass for mental health, fail for crime | Mental health usable; crime cannot be read causally |
| A6 Callaway–Sant'Anna (CS), covariate-adjusted — the reported arm (`script4a`) | Average effect on treated counties, robust to staggered timing | Poor MH Days 0.037 (SE 0.026), about 36% of TWFE, not significant. Frequent Mental Distress 51 (SE 62), not significant. Assault 20.5, violence index 7.2, NIBRS total 26.4 significant. | ✓ / mixed / ✗ / ✗ | Mental health: not supported. Crime: significant but fails pre-trends |
| CS pre-period estimates (`script4a`) | Pre-trends under CS | 5 of 11 pre-period event times significant for Poor MH Days | ✗ | Disagrees with the flat TWFE leads |
| Pre-trend adjustments on E2 (`script4u`) | Robustness | Poor MH Days: county trends 0.035 (p = 0.17); state×year FE 0.095 (p = 0.027); baseline characteristics × time 0.097 (p = 0.023). Frequent Mental Distress −10 with county trends. Outcomes significant out of 7: baseline 5, county trends 0, state×year 2. | ✗ | Mixed |
| CS with past outcomes (`script4u`) | Same, conditioning on prior outcomes | Poor MH Days −0.008; several outcomes change sign | ✗ | No |
| First entry, clean controls (`script4u`) | Entry from zero to one or more large dairies | Baseline: 2 of 7 significant, all 7 positive; 0 of 7 with baseline characteristics × time; CS not significant | ✗ | Direction consistent, not significant |
| T1/T3/T4 leads (`script4t`) | Pre-trends for the continuous treatments | T4 fails for Poor MH Days (p = 0.016); T3 borderline for Frequent Mental Distress (0.071) | ✗ for T4 | Weakens T4 |
| HonestDiD (`script4w`) | Sensitivity to parallel-trends violations | Poor MH Days: baseline set already includes zero for the first post-period; where testable, breaks at 0.5× the largest pre-period violation. Only 12 of 56 target sets exclude zero at baseline. | ✗ | Fragile |
| Species runs (`script4x`) | Is the signal dairy-specific? | Large dairy carries it (TWFE E2 significant on 5 of 7). Beef, hogs, chickens essentially null. | Specificity helps | Dairy-specific in TWFE; thin in CS |
| A4 dairy × slaughterhouse (`script4a`) | Interaction | Identified for the 2 mental-health outcomes only; Poor MH Days T1 p = 0.046, T3 p = 0.029; 89+ switchers, 2017 on | Underpowered | Exploratory |
| R-series, slaughterhouse as treatment (`script4r`) | Slaughterhouse effect | Null; Frequent Mental Distress with F2 entry −184 (wrong sign) | n/a | No |
| Control-set sensitivity (`script4y`) | Dependence on the controls | Poor MH Days E2 moves 2.4% to the pre-committed 8 and 9.5% to all 14; no sign flips | ✓ | Stable |
| Multiple testing (computed in this review, not yet in the pipeline) | Inference | 13 of 28 reported A2 cells significant at 5%; 3 survive Holm or Benjamini–Hochberg. E2 across 7 outcomes: 0 survive. | ✗ | Weakens every headline |

## Verdict
- Poor MH Days is the best-supported outcome: TWFE pre-trends flat, dairy-specific, stable to the control set. It fails on CS significance, the county-trend adjustment, HonestDiD and multiple testing. **Suggestive, not causal-grade.**
- The crime outcomes fail pre-trends and are arrests, not offences. They cannot support the hypothesis.
- Frequent Mental Distress has 2 cohorts and starts in 2016; it flips sign under county trends.
- Blockers still open: A21 (full-panel vs 2010–2020 window) and A22 (event-study control group — CS now uses never-treated and not-yet-treated counties, which largely answers A22; the answer should be written down).

## Not reviewed here
LP-DiD (`script4a` A7, `script4r` R4), Conley spatial SEs, `script4b`/`4c` (run on the earlier control membership), `script4j`, `script4d`, `script4s`, `script4v` F1/F2 figures.

## Gaps and follow-ups
Drafted as D-39 to D-41 in the decision register (D-39 was closed by D-42, which fixes the confirmatory family); the two pre-specified plans are
`2026-10-22_PLAN_estimator-triangulation.md` and `2026-10-22_PLAN_inference-upgrade.md`.
Further candidates, not yet planned: dose-response DiD for T3/T4; a census-wave collapsed panel;
spillover and border-pair designs; a worker-exposure triple difference (needs employment-share data);
Phase 3 ML selection under `/variable-selection-protocol`.
