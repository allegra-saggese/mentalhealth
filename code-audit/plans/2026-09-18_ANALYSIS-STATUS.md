# Analysis status — Part A
**2026-09-18** · rural US counties, county-year panel 2000–2023 · 2,879 counties, 65,828 rows

---

# 1. TREATMENT VARIABLES — all 12 built

All are built once in `script4_treatment.py` and persisted. **One missingness rule: NaN stays NaN, no `fillna(0)` anywhere.**

**Size definition throughout**: `cafo_dairy_large` = dairy operations with **500+ milk cows** (USDA NASS top inventory bin, code 7). Medium = 200–499, small = <200. *This is not EPA's regulatory Large CAFO threshold of 700+ mature cows — USDA publishes no 700 cutpoint.*

**Δ means**: change between **consecutive ag census waves** for the same county — a 5-year difference. Only four wave-pairs exist: 2002→2007, 2007→2012, 2012→2017, 2017→2022. All within-county treatment variation occurs at these four dates; other years are forward-filled.

### Group E — extensive margin (was the county treated?)
| ID | Column | Definition |
|---|---|---|
| **E1** | `tr_e1_lg_bin` | 1 if county has ≥1 large dairy operation that year |
| **E2** | `tr_e2_add_absorb` | 1 from the county's first wave with Δlarge > 0 onward. **Absorbing** — exits never switch it off |
| **E3** | `tr_e3_entry_absorb` | 1 from the county's first 0 → positive wave onward. Absorbing |

### Group I — intensive margin (how much?)
| ID | Column | Definition |
|---|---|---|
| **I1** | `tr_i1_add_cum` | cumulative sum of positive Δlarge across waves (dose). `log_pop` as separate control |
| **I2** | `tr_i2_add_nevents` | count of positive-change waves so far. `log_pop` control |
| **I3** | `tr_i3_lg_count` | raw count of large operations. `log_pop` control |
| **I4** | `tr_i4_lg_p10k` | large operations per 10,000 residents |

### Group C — composition (is it concentration?)
| ID | Column | Conditioning | Definition |
|---|---|---|---|
| **C1** | `tr_c1_lg_share` | — | large ÷ total dairy operations |
| **C2** | `tr_c2_lg_share_scr` | — | C1, restricted to counties with ≥10 total operations |
| **C3** | `tr_c3_log_lg` | **`cond_log_sm`** | **log(1+large) entered JOINTLY with log(1+small)** — the conditional model |
| **C4** | `tr_c3_log_lg` | `cond_smbase_x_t` | log(1+large), conditional on baseline small-farm count × time trend |
| **C5** | `tr_c5_hhi` | — | Herfindahl index across the three size bins |

### Dropped
- **M1 / M2** (build vs. consolidate binary split) — removed 2026-09-18. Partitioned counties into two arbitrary bins and returned near-identical coefficients; the C3 conditional model answers the same question properly.
- **M3** — never ran, byte-identical to E2.

### The three designated CORE treatments
`T1 = E1` · `T2 = C3` · `T3 = E3`. **Only these three are run through Callaway–Sant'Anna.**
*(Naming duplication — T1/T2/T3 are the same columns as E1/C3/E3. Worth collapsing.)*

---

# 2. OUTCOMES — all 6

| Outcome | Column | Years | N | Complete | Mean (SD) |
|---|---|---|---|---|---|
| Poor MH Days | `poor_mental_health_days` | 2010–2023 | 36,035 | 54.7% | 3.96 (0.98) |
| Frequent Mental Distress | `frequent_mental_distress_per100k` | **2016–2023** | 21,627 | 32.9% | 13,544 (2,793) |
| Deaths of Despair | `crude_rate_from_census_pop` | 2000–2020 | 13,034 | 19.8% | 20.36 (9.79) |
| Violent Crime (CHR) | `violent_crime` | 2010–2022 | 30,135 | 45.8% | 254.0 (196.5) |
| Assault (Agg+Simple) | `crime_assault` | 2000–2021 | 25,061 | 38.1% | 267.6 (242.5) |
| Total Incidents (NIBRS) | `total_incidents_per100k` | 2000–2021 | 25,061 | 38.1% | 309.9 (271.4) |

**Coverage is the binding constraint on cohorts.** Frequent Mental Distress starts 2016, so only the 2017 and 2022 cohorts are observable — that is why CS returns 2 cohorts / 14 cells for it. Deaths of Despair is truncated after 2020 (source coverage → 0). CDC **suppresses** despair counts below 10 deaths, so those are missing, not zero.

---

# 3. CONTROLS

**Default: `CONTROL_PRETREAT`, 19 variables.** Derived under a stated rule from the candidate pool: 162 panel columns → 107 candidates (excluding identifiers, CHR numerator/denominator/CI companions, treatments, outcomes, population denominators) → 105 numeric → 26 at ≥92% coverage → **19** after removing 7 plausible mediators.

- `CONTROL_COMPREHENSIVE` (26) = all high-coverage candidates, including mediators
- `CONTROL_CORE9` (9) = legacy set, **inherited from `script3-ridge.py`, never derived**. Kept only for continuity.

**No usable non-numeric controls exist** — checked. Only `pct_of_total_deaths_raw_despair` (a string, 29.6% coverage, derived from an outcome) and `state_abbrev` (the state identifier).

---

# 4. MODEL SPECIFICATIONS — explicit

Let $i$ = county, $t$ = year, $D$ = treatment, $X$ = the 19 controls.

### A1 — Pooled cross-section
$$Y_{it} = \beta D_{it} + X_{it}\gamma + \alpha_{s(i)} + \delta_t + \varepsilon_{it}$$
**State + year FE.** Identifies off *between-county* variation. **Not causal** — selection into CAFO presence is not plausibly captured by $X$ + state + year. Run as the benchmark that motivates county FE.

### A2 — Two-way FE (within county)
$$Y_{it} = \beta D_{it} + X_{it}\gamma + \alpha_i + \delta_t + \varepsilon_{it}$$
**County + year FE.** Only counties whose treatment *changes* identify $\beta$. Assumes parallel trends conditional on $X$. **Vulnerable to staggered-adoption bias** — this is the spec A18 tests.

### A2-C3 — The conditional model (core, T2)
$$Y_{it} = \beta_L \log(1+L_{it}) + \beta_S \log(1+S_{it}) + X_{it}\gamma + \alpha_i + \delta_t + \varepsilon_{it}$$
$L$ = large operations, $S$ = small operations, **entered jointly**. $\beta_L$ = association with large operations *holding small-farm count fixed*; $\beta_S$ = the reverse. The ratio model C1 is the restricted case forcing $\beta_L = -\beta_S$.

### A3 — Horse race
A2 plus `any_large_hogs`, `any_large_beef`, `any_large_chickens` entered jointly. Tests whether "dairy" is really "any large CAFO."

### A7 — TWFE event study
$$Y_{it} = \sum_{e=-8,\,e\neq-1}^{8} \theta_e \mathbf{1}[t - g_i = e] + X_{it}\gamma + \alpha_i + \delta_t + \varepsilon_{it}$$
$g_i$ = county's cohort year. Omitted period $e=-1$. Never-treated pooled into the base. Pre-period $\theta_e$ are the parallel-trends check.

### A18 — Callaway–Sant'Anna (FRONT AND CENTRE)
$$ATT(g,t) = \mathbb{E}\!\left[Y_t - Y_{g-1} \mid G=g\right] - \mathbb{E}\!\left[Y_t - Y_{g-1} \mid \text{control}\right]$$
Control group = **never-treated + not-yet-treated at $t$** (cohort > $t$). No already-treated unit is ever used as a control, which is exactly what TWFE gets wrong. Aggregated to event time and to an overall ATT by cohort size. Inference: **300-rep cluster bootstrap over states**.

**Caveat, state it explicitly:** A18 runs **without covariates** — $(g,t)$ cells are too thin for a doubly-robust version — so it assumes **unconditional** parallel trends, a *stronger* assumption than A2 makes.

### A6 — Dairy × FSIS interaction — **DO NOT PRESENT**
2017–2023 only, ~46–50% coverage. **VIF 34.3** on `%_rural`. Too thin.

---

# 5. STANDARD ERRORS — the open tweak

Every estimate reports three. Poor MH Days, core treatments, PRETREAT controls:

| | β | SE state | p | SE county | p | SE hetero | p |
|---|---|---|---|---|---|---|---|
| T1 | +0.0470 | 0.0265 | .082 | 0.0296 | .113 | 0.0196 | **.017** |
| T2 | +0.1209 | 0.0322 | .0005 | 0.0281 | .0000 | 0.0182 | .0000 |
| T3 | +0.0794 | 0.0392 | **.049** | 0.0395 | .045 | 0.0248 | .001 |

**County clustering is not uniformly more conservative than state** — for T1 it is *larger* (0.0296 vs 0.0265), for T2 *smaller*. Heteroskedastic-only SEs are always smallest and would flip T1 to significant. **T1 and T3 sit right at the 0.05 boundary and their verdict depends on the SE choice.** This is the tweak to settle before presenting.

Notes: 42 of 49 states carry entry events, so state clustering has adequate support (≥42 clusters). Estimation uses `pyfixest` — the previous `within_transform()` did one-shot demeaning, exact only for balanced panels, and this panel is unbalanced (76% of counties have the full span), which biased every point estimate by ~6%.

---

# 6. RESULTS

## 6.1 Pooled (A1) vs Within (A2) — Poor MH Days

**Every pooled estimate is null and near zero.** E1 −0.011 (p=.48), C3 −0.003 (p=.81), I3 +0.0008 (p=.14). Cross-sectional variation carries no signal; all results below come from within-county variation.

## 6.2 A2 within, all 12 treatments — Poor MH Days

| ID | β | SE | p | N | Switchers |
|---|---|---|---|---|---|
| **I3** raw count | +0.0242 | 0.0032 | **.0000** | 29,060 | 572 |
| **I1** cumulative | +0.0299 | 0.0058 | **.0000** | 29,060 | 447 |
| **C3** conditional | +0.1209 | 0.0322 | **.0005** | 29,082 | 572 |
| **C4** baseline×trend | +0.1206 | 0.0326 | **.0006** | 29,082 | 572 |
| **E2** any change | +0.0921 | 0.0299 | **.0034** | 29,082 | 266 |
| **I2** n events | +0.0625 | 0.0231 | **.0094** | 29,060 | 447 |
| **I4** per-10k | +0.0443 | 0.0218 | **.048** | 29,060 | 696 |
| **E3** entry | +0.0794 | 0.0392 | **.049** | 29,082 | 148 |
| E1 presence | +0.0470 | 0.0265 | .082 | 29,082 | 230 |
| C5 HHI | −0.0794 | 0.0655 | .231 | 28,476 | 1,068 |
| C1 share | +0.0555 | 0.0528 | .299 | 28,476 | 674 |
| C2 share screened | +0.1261 | 0.2489 | .615 | 11,154 | 479 |

**Read the betas:** all positive except C5. **8 of 12 significant.** Magnitudes are in days per month against a base of 3.96 days and a within-county SD of 0.69 — so C3's +0.121 is **0.18 within-SD**, and I3's +0.024 is per additional operation. The count-based measures (I1, I3) have the tightest SEs because they use the most variation. **C5 (HHI) is the only negative and the only one with >1,000 switchers** — concentration per se does not predict the outcome.

## 6.3 The conditional model — what is actually driving it

| Outcome | log(LARGE) | log(SMALL) |
|---|---|---|
| Poor MH Days | **+0.121** (0.032)* | +0.001 (0.016) |
| Deaths of Despair | **−0.815** (0.347)* | −0.020 (0.231) |
| Assault | **+33.6** (15.8)* | −5.63 (3.97) |
| Total Incidents | **+37.6** (17.7)* | −5.61 (4.81) |
| Violent Crime | +18.2 (9.70) | −4.34 (3.97) |
| Frequent Mental Distress | +33.3 (67.8) | +0.17 (33.6) |

**`log(small)` is null in all six.** Holding small-farm counts fixed, large operations still move the outcome. The mechanism is not small farms disappearing. VIF confirms both are separately identified (1.034 and 1.033).

## 6.4 Callaway–Sant'Anna — the headline

| Treatment | Outcome | ATT | SE | p | **Pre-period mean** | Cohorts | Cells |
|---|---|---|---|---|---|---|---|
| T1 | Poor MH Days | +0.019 | 0.049 | .699 | −0.085 | 3 | 39 |
| T1 | **Freq. Distress** | **+193.2** | 81.6 | **.018** | −34.1 | 2 | 14 |
| T1 | Deaths of Despair | −0.038 | 0.243 | .877 | −0.258 | 4 | 65 |
| T1 | Violent Crime | +11.9 | 9.04 | .190 | +3.20 | 3 | 36 |
| T1 | Assault | +8.45 | 14.2 | .552 | −16.4 | 4 | 94 |
| T1 | Total Incidents | +11.2 | 14.0 | .424 | −16.6 | 4 | 94 |
| T2 | Poor MH Days | +0.042 | 0.026 | .114 | −0.043 | 3 | 39 |
| T2 | **Freq. Distress** | +142.9 | 68.6 | **.037** | **+133.1** | 2 | 14 |
| T2 | Deaths of Despair | +0.027 | 0.208 | .896 | −0.141 | 3 | 60 |
| T3 | **Freq. Distress** | +162.1 | 71.1 | **.023** | +58.4 | 2 | 14 |
| T3 | Assault | +18.2 | 13.0 | .162 | −16.6 | 3 | 80 |
| T3 | Total Incidents | +21.4 | 13.6 | .117 | −18.4 | 3 | 80 |

**Across all 18 treatment × outcome cells: 8 TWFE-only, 7 null in both, 3 CS-only, 0 significant under both.**

The 3 CS-significant results are all Frequent Mental Distress — the **worst-covered outcome** (2016–2023, 2 cohorts, 14 cells). For **T2 the pre-period mean is +133.1 against a post ATT of +142.9 — 93% of the "effect" is present before treatment.** T3's pre-period is +58.4. Only T1 has a clean pre-period.

**Do not present Frequent Mental Distress as a finding.**

## 6.5 Control-set sensitivity — Poor MH Days

| Treatment | 9 (legacy) | 19 (default) | 26 (full) |
|---|---|---|---|
| T1 | 0.0785* | 0.0470 | 0.0363 |
| T2 | 0.1403* | 0.1209* | 0.0741* |
| T3 | 0.1159* | 0.0794* | 0.0591 |

T1 and T3 lose significance under the fullest set. **Only T2 survives all three — and it halves.**

---

# 7. DIAGNOSTICS

- **VIF**: 72 within-county specs, **max 3.09**, none above 5. Pooled specs run 5–6.8 on raw levels — exactly the cross-sectional collinearity county FE removes. A6 alone is flagged at 34.3.
- **Power**: MDE at 80% power ≈ **0.12–0.17 within-county SD** for the core treatments. Moderate effects detectable; small ones not.
- **Panel balance**: unbalanced. 1–13 obs per county, **76.2% have the full span**, 350 counties have internal gaps.
- **Identifying variation**: 938 positive-change events, 572 counties, 4 cohorts (284/142/101/45). Timing carries **up to 5 years of measurement error** from census forward-fill.

---

# 8. WHAT TO PRESENT

**Safe:** the conditional-model result (large operations carry it, small-farm counts do not); the TWFE→CS collapse as a *methodological* finding; the diagnostics; the power/MDE framing of the nulls.

**Do not claim:** any causal effect on mental health; Frequent Mental Distress; Deaths of Despair (TWFE-only and *negative*); A6.

**Slide order:** Setting → Identifying variation → Model specs → Conditional model → TWFE vs CS → Control sensitivity → Diagnostics → Informative null.

---

# 9. OPEN ITEMS

### 0. BLOCKING — CHR variables are dated by RELEASE year, not data year
See `code-audit/plans/2026-09-21_CHR-YEAR-MISALIGNMENT.md`. Confirmed against CHR's
own release code: `median_household_income`, `children_in_poverty` and `unemployment`
are lagged **2 years**; `poor_mental_health_days` is lagged **3**. The row labelled
`year = 2020` holds 2018 income, 2017 mental-health days, and forward-filled 2017
CAFO counts. Also found: `access_to_healthy_foods` splices two different measures
(v030 → v083 at 2013), and `violent_crime` has 46.7% of year-over-year values
identical (pooled vintage). **Outranks everything below.** Every estimate in the
deck is affected; for the current presentation this belongs in limitations.

### Remaining
1. **SE choice for T1/T3** — verdict flips on clustering level. Settle before presenting.
2. **CS runs without covariates** — unconditional parallel trends. Main caveat.
3. CS only ran on the 3 core treatments; the other 9 have TWFE only.
4. T1/T2/T3 duplicate E1/C3/E3 — collapse the naming.
5. CDC suppression of despair counts is a selection problem; no selection model built.
6. 3 controls remain excluded with no stated basis (`adult_smoking`, `teen_births`, `children_in_single-parent_households`).

---

# 10. FILES

**Results** — `Dropbox/Mental/Data/output/`
| | |
|---|---|
| Core | `tables/script4f/` — `CORE_HEADLINE.csv`, `CORE_twfe_by_controlset.csv`, `CORE_callaway_santanna.csv` |
| Full grid (12 treatments) | `tables/script4a/2026-09-18_A1_A2_treatment_grid.csv` |
| Event study, CS cells, VIF | `tables/script4a/` |
| Power / balance / coverage | `tables/script4a/2026-09-18_power_and_coverage.xlsx` |
| LaTeX tables | `tables/script4a/latex/` |
| Figures | `figs/script4a/` F1–F4 |
| Slides | `slides/core_results.tex` / `.pdf` |

**Code** — `script4_treatment.py` · `script4a-twfe-eventstudy.py` · `script4d-power-balance.py` · `script4e-latex-tables.py` · `script4f-core-models.py` · `script4g-slides.py` — all uncommitted.
