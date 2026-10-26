# D-53 before and after (2026-10-25)

Census-year blank CAFO counts are 0 before the forward-fill (D-53). Results only; no interpretation. "Before" = the newest earlier output of the same script (dates in each header); "after" = the 2026-10-25 re-run on `Data/merged/2026-10-25_panel.csv`. All fifteen re-runs exited 0 (`script1b`, `1c`, `1d`, `4j`, `4a`, `4d`, `4t`, `4y`, `4x`, `4z`, `4zc`, `4zd`, `4n`, `4p`, `4q`). E2 and E3 for dairy are unchanged by construction (their input columns are identical); scripts `4u`, `4w`, `4za`, `4ze`, `4r` read none of the changed columns and were not re-run.

## script4a A2 (TWFE, county + year FE, 8 controls, state-clustered SE); before = 2026-10-23

| Treatment | Outcome | Estimate before | after | SE before | after | p before | after | N |
|---|---|---|---|---|---|---|---|---|
| T1 | Aggravated Assault | 2.0424 | 1.9347 | 4.4815 | 4.4899 | 0.6509 | 0.6687 | 23,357 |
| T3 | Aggravated Assault | 0.9072 | 0.8920 | 0.8943 | 0.8921 | 0.3160 | 0.3230 | 23,357 |
| T4 | Aggravated Assault | 4.5263 | 4.4493 | 4.3690 | 4.3794 | 0.3060 | 0.3153 | 23,357 |
| T1 | Assault, all severities | 18.0377 | 18.1085 | 13.4449 | 13.5001 | 0.1868 | 0.1868 | 23,357 |
| T3 | Assault, all severities | 10.1148 | 10.1242 | 2.7846 | 2.7842 | 0.0007 | 0.0007 | 23,357 |
| T4 | Assault, all severities | 36.8923 | 36.9545 | 15.7716 | 15.8214 | 0.0240 | 0.0242 | 23,357 |
| T1 | Deaths of Despair | -0.3320 | -0.3527 | 0.3818 | 0.3696 | 0.3890 | 0.3448 | 11,850 |
| T3 | Deaths of Despair | 0.1589 | 0.1549 | 0.6151 | 0.6130 | 0.7973 | 0.8016 | 11,850 |
| T4 | Deaths of Despair | -0.3144 | -0.3258 | 0.3811 | 0.3761 | 0.4135 | 0.3907 | 11,850 |
| T1 | Frequent Mental Distress | 103.1145 | 92.0253 | 72.2227 | 72.4101 | 0.1600 | 0.2100 | 21,554 |
| T3 | Frequent Mental Distress | 62.8669 | 62.0931 | 37.9471 | 37.8483 | 0.1042 | 0.1076 | 21,554 |
| T4 | Frequent Mental Distress | 180.0392 | 172.3585 | 88.3019 | 87.5114 | 0.0471 | 0.0548 | 21,554 |
| T1 | NIBRS curated total | 19.3407 | 19.5812 | 14.8068 | 14.8572 | 0.1984 | 0.1945 | 23,357 |
| T3 | NIBRS curated total | 11.4208 | 11.4545 | 3.3944 | 3.3967 | 0.0016 | 0.0016 | 23,357 |
| T4 | NIBRS curated total | 39.1880 | 39.3742 | 17.6793 | 17.7219 | 0.0320 | 0.0316 | 23,357 |
| T1 | Poor MH Days | 0.0806 | 0.0785 | 0.0365 | 0.0355 | 0.0320 | 0.0318 | 35,918 |
| T3 | Poor MH Days | 0.0415 | 0.0417 | 0.0186 | 0.0187 | 0.0301 | 0.0308 | 35,918 |
| T4 | Poor MH Days | 0.1461 | 0.1451 | 0.0383 | 0.0385 | 0.0004 | 0.0005 | 35,918 |
| T1 | Violence index (partial) | 3.3000 | 3.2679 | 5.7536 | 5.7766 | 0.5693 | 0.5745 | 23,357 |
| T3 | Violence index (partial) | 1.2458 | 1.2411 | 1.2380 | 1.2396 | 0.3199 | 0.3223 | 23,357 |
| T4 | Violence index (partial) | 5.8080 | 5.7859 | 5.8374 | 5.8552 | 0.3253 | 0.3286 | 23,357 |

E2 rows: estimate, SE, p identical before and after (14 of 14 A1/A2 rows).

## script4z confirmatory family (9,999 wild-bootstrap draws, seed 20261022); before = 2026-10-22

| Outcome | Treatment | p analytic before | after | p wild before | after | Romano-Wolf within outcome before | after | Holm over 8 before | after |
|---|---|---|---|---|---|---|---|---|---|
| Poor MH Days | T1 | 0.0320 | 0.0318 | 0.0446 | 0.0446 | 0.0791 | 0.0802 | 0.2676 | 0.2676 |
| Poor MH Days | T3 | 0.0301 | 0.0308 | 0.0478 | 0.0467 | 0.0791 | 0.0802 | 0.2676 | 0.2676 |
| Poor MH Days | T4 | 0.0004 | 0.0005 | 0.0028 | 0.0028 | 0.0046 | 0.0047 | 0.0224 | 0.0224 |
| Poor MH Days | E2 | 0.0190 | 0.0190 | 0.0229 | 0.0229 | 0.0671 | 0.0696 | 0.1603 | 0.1603 |
| Frequent Mental Distress | T1 | 0.1600 | 0.2100 | 0.1643 | 0.2181 | 0.2336 | 0.2361 | 0.3258 | 0.3288 |
| Frequent Mental Distress | T3 | 0.1042 | 0.1076 | 0.1629 | 0.1644 | 0.2336 | 0.2361 | 0.3258 | 0.3288 |
| Frequent Mental Distress | T4 | 0.0471 | 0.0548 | 0.0561 | 0.0646 | 0.1872 | 0.1852 | 0.2676 | 0.2676 |
| Frequent Mental Distress | E2 | 0.0425 | 0.0425 | 0.0623 | 0.0623 | 0.1872 | 0.1852 | 0.2676 | 0.2676 |

E2 estimate and analytic and wild p-values are identical; its Romano-Wolf p moves slightly because the step-down uses the four treatments' joint draws.

## script4zc treatment by model, T1/T3/T4 (dCDH is the effect of the treatment change, not per unit); before = 2026-10-24

| Model | Treatment | Outcome | Estimate before | after | SE before | after | p before | after |
|---|---|---|---|---|---|---|---|---|
| TWFE | T1 | Aggravated Assault | 2.0424 | 1.9347 | 4.4815 | 4.4899 | 0.6486 | 0.6665 |
| TWFE | T3 | Aggravated Assault | 0.9072 | 0.8920 | 0.8943 | 0.8921 | 0.3104 | 0.3174 |
| TWFE | T4 | Aggravated Assault | 4.5263 | 4.4493 | 4.3690 | 4.3794 | 0.3002 | 0.3096 |
| dCDH | T1 | Aggravated Assault | -2.9044 | -2.0794 | 4.0271 | 3.9930 | 0.4708 | 0.6025 |
| dCDH | T3 | Aggravated Assault | -2.0519 | -2.0573 | 5.6642 | 5.6631 | 0.7172 | 0.7164 |
| dCDH | T4 | Aggravated Assault | -1.6336 | -1.1671 | 3.4587 | 3.3073 | 0.6367 | 0.7242 |
| TWFE | T1 | Assault, all severities | 18.0377 | 18.1085 | 13.4449 | 13.5001 | 0.1797 | 0.1798 |
| TWFE | T3 | Assault, all severities | 10.1148 | 10.1242 | 2.7846 | 2.7842 | 0.0003 | 0.0003 |
| TWFE | T4 | Assault, all severities | 36.8923 | 36.9545 | 15.7716 | 15.8214 | 0.0193 | 0.0195 |
| dCDH | T1 | Assault, all severities | 10.0438 | 10.3520 | 11.8735 | 11.6655 | 0.3976 | 0.3749 |
| dCDH | T3 | Assault, all severities | 16.1044 | 16.0226 | 17.0406 | 17.0202 | 0.3446 | 0.3465 |
| dCDH | T4 | Assault, all severities | 19.1507 | 19.2433 | 14.0483 | 13.9498 | 0.1728 | 0.1678 |
| TWFE | T1 | Deaths of Despair | -0.3320 | -0.3527 | 0.3818 | 0.3696 | 0.3845 | 0.3399 |
| TWFE | T3 | Deaths of Despair | 0.1589 | 0.1549 | 0.6151 | 0.6130 | 0.7962 | 0.8005 |
| TWFE | T4 | Deaths of Despair | -0.3144 | -0.3258 | 0.3811 | 0.3761 | 0.4094 | 0.3863 |
| dCDH | T1 | Deaths of Despair | -0.2984 | -0.3499 | 0.4441 | 0.4396 | 0.5016 | 0.4260 |
| dCDH | T3 | Deaths of Despair | 5.1597 | 5.1900 | 3.6359 | 3.6572 | 0.1559 | 0.1559 |
| dCDH | T4 | Deaths of Despair | -0.0072 | -0.0840 | 0.4734 | 0.4615 | 0.9879 | 0.8555 |
| TWFE | T1 | Frequent Mental Distress | 103.1145 | 92.0253 | 72.2227 | 72.4101 | 0.1534 | 0.2038 |
| TWFE | T3 | Frequent Mental Distress | 62.8669 | 62.0931 | 37.9471 | 37.8483 | 0.0976 | 0.1009 |
| TWFE | T4 | Frequent Mental Distress | 180.0392 | 172.3585 | 88.3019 | 87.5114 | 0.0415 | 0.0489 |
| dCDH | T1 | Frequent Mental Distress | 96.2032 | 84.2357 | 93.8820 | 97.8479 | 0.3055 | 0.3893 |
| dCDH | T3 | Frequent Mental Distress | 123.0839 | 123.3241 | 105.8267 | 106.0268 | 0.2448 | 0.2448 |
| dCDH | T4 | Frequent Mental Distress | 158.8065 | 153.2950 | 109.2527 | 113.0316 | 0.1461 | 0.1750 |
| TWFE | T1 | NIBRS curated total | 19.3407 | 19.5812 | 14.8068 | 14.8572 | 0.1915 | 0.1875 |
| TWFE | T3 | NIBRS curated total | 11.4208 | 11.4545 | 3.3944 | 3.3967 | 0.0008 | 0.0007 |
| TWFE | T4 | NIBRS curated total | 39.1880 | 39.3742 | 17.6793 | 17.7219 | 0.0266 | 0.0263 |
| dCDH | T1 | NIBRS curated total | 10.2206 | 10.5841 | 13.5585 | 13.2317 | 0.4510 | 0.4238 |
| dCDH | T3 | NIBRS curated total | 17.2365 | 17.1488 | 18.7807 | 18.7609 | 0.3587 | 0.3607 |
| dCDH | T4 | NIBRS curated total | 24.4953 | 24.4829 | 16.2906 | 16.1494 | 0.1327 | 0.1295 |
| TWFE | T1 | Poor MH Days | 0.0806 | 0.0785 | 0.0365 | 0.0355 | 0.0271 | 0.0269 |
| TWFE | T3 | Poor MH Days | 0.0415 | 0.0417 | 0.0186 | 0.0187 | 0.0253 | 0.0260 |
| TWFE | T4 | Poor MH Days | 0.1461 | 0.1451 | 0.0383 | 0.0385 | 0.0001 | 0.0002 |
| dCDH | T1 | Poor MH Days | 0.0630 | 0.0553 | 0.0360 | 0.0372 | 0.0799 | 0.1375 |
| dCDH | T3 | Poor MH Days | 0.0337 | 0.0342 | 0.0565 | 0.0570 | 0.5503 | 0.5491 |
| dCDH | T4 | Poor MH Days | 0.1311 | 0.1272 | 0.0415 | 0.0447 | 0.0016 | 0.0044 |
| TWFE | T1 | Violence index (partial) | 3.3000 | 3.2679 | 5.7536 | 5.7766 | 0.5663 | 0.5716 |
| TWFE | T3 | Violence index (partial) | 1.2458 | 1.2411 | 1.2380 | 1.2396 | 0.3143 | 0.3167 |
| TWFE | T4 | Violence index (partial) | 5.8080 | 5.7859 | 5.8374 | 5.8552 | 0.3198 | 0.3231 |
| dCDH | T1 | Violence index (partial) | -2.2249 | -1.3168 | 5.6499 | 5.6090 | 0.6937 | 0.8144 |
| dCDH | T3 | Violence index (partial) | -0.0965 | -0.1060 | 6.7906 | 6.7890 | 0.9887 | 0.9875 |
| dCDH | T4 | Violence index (partial) | 2.6099 | 3.0045 | 5.3606 | 5.2104 | 0.6264 | 0.5642 |

The 98 rows for E2 and E3 under TWFE, Callaway-Sant'Anna, LP-DiD, Sun-Abraham, imputation, dCDH (cohort designs) and stacked DiD are unchanged.

## script4x per-species results; before = 2026-10-23

| Species | Scope | Rows with a changed estimate | Rows |
|---|---|---|---|
| beef | all sizes | 27 | 35 |
| beef | large | 0 | 35 |
| chickens | all sizes | 35 | 35 |
| chickens | large | 6 | 35 |
| dairy | all sizes | 35 | 35 |
| dairy | large | 21 | 35 |
| hogs | all sizes | 35 | 35 |
| hogs | large | 25 | 35 |

Full tables: `Data/output/tables/script4x/2026-10-25_species_results.csv` against the previous file. Also re-run with changed values (rows differing from the previous file): `script4t` leads (80 of 174 coefficient rows, 21 of 35 joint tests), `script4y` control-set selection (regressions 1,764 of 2,352 rows), `script4a` A4 dairy x FSIS (28 of 28 rows), A5 event study (64 of 236 rows) and the VIF audit (84 of 105 rows). Identical: `script4a` A6 Callaway-Sant'Anna (both files), A7 LP-DiD, HEADLINE_twfe_vs_cs, `script4x` outcome_within_sd, control roster and control pre-trends. `script4d` balance tables were re-run (LaTeX and CSV in the LaTeX output folder) and not compared here.

