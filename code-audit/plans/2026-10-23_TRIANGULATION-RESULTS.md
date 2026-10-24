# Estimator triangulation: results (2026-10-23)
`script4za-estimator-triangulation.py`, rule D-45 fixed before the run (an estimator agrees with Callaway-Sant'Anna if its overall ATT has the same sign
and lies inside CS's 95% interval; the Poor MH Days first-expansion result is estimator-robust if at least 4 of the 5 other estimators agree).
One complete-case sample per design and outcome for all estimators except imputation, which alone drops counties with a single untreated
observation (pyfixest limitation). 14 of 14 cells ran, 0 estimator failures. Outputs: `Data/output/tables/script4za/2026-10-23_*.csv`.
Standard errors are each estimator's own, state-clustered (CS: state bootstrap, 300 draws). "Overall" is each estimator's own post-treatment
aggregation, so windows and weights differ slightly (LP-DiD and dCDH stop at event time 8).

**Pre-specified cell (D-45), first expansion, Poor MH Days: 4 of 5 agree, so the rule returns estimator-robust.** The estimator that disagrees
is stacked DiD (0.0925, SE 0.0424), just above CS's interval (-0.017 to 0.092 rounded). First entry, Poor MH Days: 4 of 5 (descriptive).

### first expansion

Overall post-treatment ATT (SE). `*` = agrees with CS (same sign, inside CS's 95% interval).

| Outcome | CS | TWFE | LP-DiD | Sun-Abraham | Imputation | dCDH | Stacked | Agree |
|---|---|---|---|---|---|---|---|---|
| Poor MH Days | 0.03718 (0.02783) | 0.09627 (0.04316) | 0.03087 (0.02891)* | 0.03779 (0.03666)* | 0.07268 (0.0432)* | 0.03409 (0.0292)* | 0.09251 (0.04244) | 4/5 |
| Frequent Mental Distress | 50.71 (62.51) | 130.3 (77.29) | 54.9 (73.04)* | 144.5 (85.49)* | 155.8 (119.7)* | 111.8 (86.36)* | 133.4 (79.07)* | 5/5 |
| Deaths of Despair | 0.04967 (0.2138) | -0.2281 (0.2931) | 0.1418 (0.2049)* | 0.4403 (0.3618)* | -0.2136 (0.3448) | 0.2435 (0.3099)* | -0.1856 (0.2936) | 3/5 |
| Aggravated Assault | 2.729 (2.828) | 6.719 (2.617) | 0.07146 (2.525)* | 2.264 (3.634)* | 8.204 (2.774)* | 1.077 (3.156)* | 5.547 (2.342)* | 5/5 |
| Assault, all severities | 21.06 (10.74) | 32.12 (13.84) | 10.62 (9.883)* | 10.47 (15)* | 35.73 (16.68)* | 8.042 (12.03)* | 26.4 (12.51)* | 5/5 |
| Violence index (partial) | 7.457 (3.956) | 7.198 (3.52) | 3.086 (3.368)* | 5.184 (5.218)* | 9.527 (4.042)* | 3.234 (4.259)* | 5.024 (3.039)* | 5/5 |
| NIBRS curated total | 27.01 (12.39) | 32.36 (15.45) | 14 (10.88)* | 12.69 (16.91)* | 36.7 (18.96)* | 10.16 (12.88)* | 25.9 (13.78)* | 5/5 |

### first entry (clean controls)

Overall post-treatment ATT (SE). `*` = agrees with CS (same sign, inside CS's 95% interval).

| Outcome | CS | TWFE | LP-DiD | Sun-Abraham | Imputation | dCDH | Stacked | Agree |
|---|---|---|---|---|---|---|---|---|
| Poor MH Days | 0.02621 (0.03564) | 0.105 (0.05923) | 0.01998 (0.04189)* | 0.02455 (0.05086)* | 0.07399 (0.06309)* | 0.02478 (0.04259)* | 0.1049 (0.05918) | 4/5 |
| Frequent Mental Distress | 55.36 (60.56) | 131.9 (100.3) | 55.38 (78.75)* | 154.9 (94.3)* | 121.8 (169.8)* | 134.3 (96.75)* | 135.5 (102.2)* | 5/5 |
| Deaths of Despair | 0.3307 (0.5918) | 0.004645 (0.3937) | 0.2383 (0.3513)* | 1.251 (0.442)* | 0.06071 (0.4835)* | 0.8229 (0.4107)* | -0.06271 (0.3987) | 4/5 |
| Aggravated Assault | -1.808 (4.313) | 3.908 (4.875) | -2.362 (4.125)* | -1.369 (4.729)* | 4.704 (4.682) | -0.5632 (4.496)* | 3.592 (4.04) | 3/5 |
| Assault, all severities | 19.57 (13.43) | 34.38 (14.08) | 13.7 (12.87)* | 15.35 (14.27)* | 41.36 (14.52)* | 15.11 (12.82)* | 28.66 (12.74)* | 5/5 |
| Violence index (partial) | 3.663 (5.206) | 6.26 (5.595) | 0.2307 (5.046)* | 2.422 (5.787)* | 8.56 (5.33)* | 1.334 (5.543)* | 4.981 (4.739)* | 5/5 |
| NIBRS curated total | 25.18 (15.61) | 35.75 (15.75) | 15.61 (14.73)* | 17.84 (16.39)* | 44.6 (15.97)* | 16.04 (14.65)* | 29.42 (14.19)* | 5/5 |

### Pre-treatment placebo cells significant at 5% (event times -8 to -2), mental-health outcomes

| Design | Outcome | CS | LP-DiD | Sun-Abraham | Imputation | dCDH | Stacked |
|---|---|---|---|---|---|---|---|
| first expansion | Poor MH Days | 6/7 | 1/7 | 0/7 | 0/7 | 0/6 | 0/7 |
| first expansion | Frequent Mental Distress | 0/5 | 0/5 | 0/5 | 0/5 | 0/2 | 0/5 |
| first entry (clean controls) | Poor MH Days | 4/7 | 0/7 | 0/7 | 0/7 | 0/6 | 0/7 |
| first entry (clean controls) | Frequent Mental Distress | 0/5 | 0/5 | 0/5 | 0/5 | 0/2 | 0/5 |

## Flags (read before citing the verdict)
- The rule measures agreement with CS, and CS's own interval is wide and includes zero for Poor MH Days (0.0372, SE 0.0278). "Robust" here means the
  estimators are consistent with a small, imprecise effect; it is not evidence of a significant one.
- The estimators fall into two groups across almost every outcome: LP-DiD, Sun-Abraham, dCDH and CS give smaller estimates, imputation and stacked DiD
  give larger ones, close to TWFE (Poor MH Days: 0.027 to 0.037 vs 0.073 to 0.093). Imputation and stacked DiD are the two with nominal significance for
  Poor MH Days (stacked p about 0.03, imputation p about 0.09). Why the groups differ (weighting, comparison group) has not been investigated.
- Frequent Mental Distress: imputation runs on cohort 2022 only (the 2017 cohort has one pre-period and is dropped), so its sample differs from the others.
- The pre-period placebo cells are not a comparison on equal terms: CS uses one base year for all pre-period cells (universal base period), the others do not.
  CS shows 6 of 7 (first expansion) and 4 of 7 (first entry) significant placebo cells for Poor MH Days; the five other estimators show 0 or 1.
- The crime outcomes fail the TWFE pre-trend test (D-23), so their agreement counts say little about causality.
- Adjusted p-values are not applied here: this is a robustness check, not a new test family (D-42 covers the confirmatory family).
