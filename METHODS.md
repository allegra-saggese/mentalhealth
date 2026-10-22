# Methods

The estimators, treatment definitions and tests used in the stage-4 scripts, written once. Script headers point here
instead of repeating it; the original long headers are kept verbatim in `code-audit/script-headers/`. Decisions and their history are in `code-audit/plans/CONTROL-DECISION-REGISTER.md` (D-numbers).

**Notation.** `i` = county, `t` = year, `Y_it` = an outcome, `X_it` = the control set, `a_i` = county fixed effect,
`d_t` = year fixed effect, `g_i` = the county's first treated year (always a census wave: 2007, 2012, 2017 or 2022),
`D_it` = 1 from `g_i` on and 0 before (0 for never-treated counties). All standard errors are clustered by state.

---

## 1. Data and treatment definitions

**Panel.** Rural county-years, 2000–2023 (`Data/merged/*_panel.csv`, built by `script1b`).

**CAFO counts.** Operation counts from the USDA ag census (2002, 2007, 2012, 2017, 2022), carried forward to the years
between waves. Sizes are small, medium and large (the top size bin; for dairy, 500+ milk cows). *All sizes* means the sum of the
three. A missing count is set to zero from 2002 on and left missing for 2000–01 (D-34): the census is exhaustive, so a county
with no row had no operations, and 2000–01 were never measured.

**Size caveat.** "Large" dairy is the USDA top inventory bin, 500 or more milk cows. It is **not** EPA's regulatory Large CAFO threshold (700+ mature cows); USDA publishes no 700 cut point, so 500+ is the nearest available bin, and this must be stated in the paper.

**Absorbing treatment.** For E2 and E3 treatment never switches back off: exits and contractions do not change `D_it`, and exit counties are not dropped (dropping them would select on a treatment path that may correlate with the outcome).

**Species.** Dairy, beef, hogs, chickens. Cattle is not a species here: the USDA cattle inventory contains the dairy and beef herds (D-36).

**Treatments** (for a count `C_it` of large or all-size operations):

| ID | Definition |
|---|---|
| T1 | presence: `1[C_it > 0]` |
| T3 | `C_it / population_it × 10,000` |
| T4 | `log(1 + C_it)`; `log population` also enters the regression |
| E2 *first expansion* | `g_i` = first census wave at which `C` rises relative to the county's previous observed wave; `D_it = 1[t ≥ g_i]` |
| E3 *first entry* | as E2 but the first wave at which `C` goes from 0 to 1 or more (`tr_e3_entry_absorb`) |

**Entry sample (clean controls).** Entry counties plus counties that never have a large dairy in any census wave. Counties that only
expand an existing large dairy, or have one throughout, are left out because they are not valid "untreated" comparisons for an entry design (D-32).

**County groups used in the descriptive figures:** never large dairy; enters; expands only; large dairy throughout.

**Outcomes (7):** poor mental-health days, frequent mental distress, deaths of despair, aggravated assault, assault (all severities),
violence index (partial), NIBRS curated total. The NIBRS series are arrests, not offences known to police.

---

## 2. Estimators

**Static TWFE.**

    Y_it = β · D_it + X_it'γ + a_i + d_t + ε_it

**Event study (TWFE).**

    Y_it = Σ_{e ≠ −1} θ_e · 1[ t − g_i = e ] + X_it'γ + a_i + d_t + ε_it

`e = t − g_i`, clipped to [−8, 8]; reference `e = −1`; never-treated counties carry no event-time dummy. The `θ_e` with `e < −1` are the pre-trend
test ("leads"). Joint test: Wald χ² that all leads are zero, using the state-clustered covariance.

**Callaway–Sant'Anna (CS).** For cohort `g` (counties first treated in `g`) and year `t`:

    ATT(g,t) = E[ Y_t − Y_{g−1} | cohort g ]  −  E[ Y_t − Y_{g−1} | comparison counties ]

Comparison counties are never treated or not yet treated at `max(t, g)`; a county that is already treated is never a comparison.

*Covariate-adjusted* (the reported arm): fit `m_{g,t}(x) = E[ Y_t − Y_{g−1} | X_{g−1} = x, comparison ]` by linear regression on the comparison
counties, with `X` measured in the base year `g−1`, then

    ATT_adj(g,t) = mean over cohort-g counties of [ (Y_t − Y_{g−1}) − m̂_{g,t}(X_{g−1}) ]

Cells too thin to fit the adjustment fall back to the unconditional estimate and are counted (`n_fallback_cells`; 0 in all runs so far).
The *unconditional* arm skips the adjustment (kept for robustness).

*Aggregation:* event-time `ATT(e)` = mean of `ATT(g, g+e)` weighted by the number of treated counties; the overall ATT is the same weighting over all
post-treatment cells (`t ≥ g`). *Inference:* bootstrap resampling whole states with replacement, 300 draws, fixed seed (20260928).
Implemented in `did_cs.py`. CS does not remove pre-trends; it changes the comparison group and weighting (it never uses already-treated counties as controls).

**Treatment timing.** Treatment can only change at census waves five years apart, so the true change happened somewhere in the five years before the
observed wave. Event-time paths therefore mix real timing with measurement error.

---

## 3. Pre-trend tests and adjustments

**Outcome leads for continuous treatments (T1, T3, T4)** (`script4t`):

    Y_it = b0 · D_it + Σ_{k=1..4} b_k · D_{i,t+k} + [log pop for T4] + X_it'γ + a_i + d_t + ε_it

`D_{i,t+k}` is the county's own treatment `k` years ahead; the joint test is `b_1 = … = b_4 = 0`. The last four panel years drop out.

**Pre-treatment slope test** (`script4v`): for cohort `g`, over the five years before treatment, counties with ≥ 3 years of data:

    Y_it = a_i + b · year_t + c · (treated_i × year_t) + ε_it

`c` = how much faster or slower treated counties' outcomes were already moving per year.

**Adjustments** (each a second specification beside the baseline, `script4u`):

| Name | Specification |
|---|---|
| county trends | adds `λ_i · t`: a county-specific linear time trend. Implemented by residualizing on county intercepts, county trends and year effects (alternating projections); `pyfixest`'s varying-slope syntax is silently ignored (QA #24) |
| state × year FE | replaces `d_t` with state-by-year effects: treated counties are compared only with comparison counties in the same state and year |
| baseline × time polynomial | adds 2002 values of log population, median household income, % Hispanic, % 65+, unemployment and the 2002 number of dairy operations (all sizes and large), each interacted with `(year−2002)/10` and its square |
| past outcomes (CS) | adds the outcome's own levels in `g−1` and `g−2` to `X_{g−1}` |

County trends can absorb a treatment effect that builds gradually (Wolfers 2006; Meer & West 2016), so they are a robustness check, not the headline specification.

**Sensitivity analysis** (`script4w`): Rambachan & Roth (2023) via R `HonestDiD`, applied to the TWFE event study. Relative-magnitudes restriction (post-treatment violations
no larger than `M̄` times the largest pre-treatment violation) and smoothness restriction, for the first post-period effect and the average of the first five.

---

## 4. Control set

The control set is the eight full-coverage controls `COVARIATE_ORDER[:8]`: median household income, unemployment, % 65+, % Hispanic, % under 18,
children in poverty, % Asian/AIAN/NHPI, % Black. It was **pre-committed** (D-16, revised D-31) on coverage, causal role and whether the series can be
tested for pre-trends. `script4y` does not select it; it documents how far the treatment coefficient moves under four families of regressions:

| Family | What is run |
|---|---|
| A1 | controls added cumulatively in `COVARIATE_ORDER` (0 to 14) |
| A2 | each of the 14 added alone to a no-control model |
| B1 | the final 8, with each dropped in turn |
| B2 | all 14, with each dropped in turn |

"Common sample" = complete cases for all 14 controls, so differences are due to the control and not to a changing sample.

---

## 5. Species analysis (`script4x`)

For each species and scope (large; all sizes): TWFE for T1, T3, T4 and E2 and CS-adjusted on the E2 cohorts, for all seven outcomes. The forest plots divide
each coefficient by the outcome's within-county SD, so rows share one scale; treatment units still differ by row.

---

## 6. Validation built into the scripts

- `script4u` and `script4x` reproduce `script4a`'s TWFE and CS results exactly for first expansion / dairy-large (differences about 1e-15).
- The county-trend routine equals `pyfixest` when trends are switched off (coefficients to 1e-9).
- `tests/compare_outputs.py` compares two output trees file by file; it is how a refactor is shown to change no result.

---

## 7. Descriptive definitions

**Causal role of a control** (`script4j`; judgements about the causal graph, stated so a reader can disagree explicitly, not statistical findings):

| Role | Meaning | Effect of conditioning on it |
|---|---|---|
| confounder | plausibly causes both CAFO siting and the outcome, and is not itself caused by CAFO presence | removes bias |
| mediator | plausibly caused by CAFO presence and in turn affects the outcome | blocks part of the causal path; biases the estimate toward zero |
| collider | plausibly caused by both the treatment and the outcome | induces a spurious association |

**Covariate pre-trend test** (`script4j`): each control is the dependent variable in the event-study specification (county + year FE). The number of significant
pre-period coefficients and their joint test show whether treated and comparison counties were already diverging on that control before treatment.

**Balance table columns** (`script4d`): N = estimating-sample rows after listwise deletion. Treated / control = county-years with the treatment above / at zero
(descriptive only for continuous treatments). Switching counties = counties whose treatment changes; under county FE only these identify the coefficient.
SE = state-clustered, from the A2 regression. **MDE** = minimum detectable effect at 80% power, two-sided 5%: `2.8016 × SE`, expressed in within-county SDs of the outcome
(within-county because that is the variation a county-FE model can use).
