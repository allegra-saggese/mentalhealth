#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4-model-test.py

Purpose:
    (a) Replicate, independently, the dairy-CAFO finding described in the
        external memo `GalinaAnalysis/2026-08-25-claude-cafo-patterns/cafo_memo.html`
        (Dropbox — NOT part of this repo, NOT modified or read programmatically
        by this script; only its written conclusions are being checked against).
        Three specs, in the same progression as that memo:
          1. Cross-section (pooled, state + year FE)         -> should be null/favorable
          2. Within-county (county + year FE)                -> sign should flip negative
          3. Event study around first large-dairy-CAFO entry -> dynamic, growing effect
    (b) Ridge-regression cross-check: does an atheoretical, regularized regression
        over the full control/treatment pool pick out the same dairy-CAFO
        relationship, independent of the TWFE causal design? Run twice —
        pooled (raw) and within-transformed (county+year demeaned) — since the
        memo's own point is that pooled and within give different answers.

    (e) Control specification curve + covariate importance ranking: instead of
        picking one control vector X_i by hand, (i) run the within (county+year
        FE) dairy regression with log_pop always included plus every possible
        subset of a 9-variable core control set (512 combinations per outcome),
        to see how much the dairy coefficient moves depending on which controls
        are in X_i; (ii) rank covariates (including dairy itself) by Ridge and
        Random Forest importance on a wider ~20-variable set, so dairy's own
        predictive weight can be compared directly against standard controls.

    Callaway-Sant'Anna staggered DiD and interacted/heterogeneous treatment
    definitions are a separate, still-under-discussion follow-on (see QA/plans/).

Sample: rural counties only (non_large_metro == 1 via `rural` col), 2000-2023.

Figures -> Dropbox/Mental/Data/output/figs/script4/
  Y1_dairy_levels_vs_within.png       Cross-section vs within-county coefficient comparison
  Y1b_dairy_isolated_vs_conditional.png  Dairy alone vs. dairy + beef/hogs/chickens jointly
  Y2_dairy_event_study.png            Event-study around first large-dairy-CAFO entry
  Y3_ridge_pooled_vs_within.png       Ridge coefficients, pooled vs within-transformed
  Y4_spec_curve.png                   Dairy beta across all 512 control combinations, w/ 95% CI
  Y5_covariate_ranking.png            Ridge + Random Forest importance, dairy included

Tables -> Dropbox/Mental/Data/output/tables/script4/
  Block1_dairy_levels_vs_within.csv
  Block1b_dairy_isolated_vs_conditional.csv
  Block2_dairy_event_study.csv
  Block2b_dairy_event_study_joint_tests.csv
  Block3_ridge_pooled_vs_within.csv
  Block4_spec_curve.csv
  Block5_covariate_ranking.csv
"""

from packages import *
from functions import *
import itertools
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from doubleml import DoubleMLData, DoubleMLPLR

# ── Directories ──────────────────────────────────────────────────────────────
merged_dir    = os.path.join(db_data, "merged")
out_dir       = os.path.join(figs_dir, "script4")
tables_s4_dir = os.path.join(tables_dir, "script4")
for _d in (out_dir, tables_s4_dir):
    os.makedirs(_d, exist_ok=True)
today_str = date.today().strftime("%Y-%m-%d")

POP_COL = "population"

# ── Load panel ───────────────────────────────────────────────────────────────
df_raw = pd.read_csv(latest_file_glob(merged_dir, "*_panel.csv"), low_memory=False)
df_raw = df_raw[df_raw["rural"] == 1].copy()
df_raw["state_fips"] = df_raw["fips"].astype("string").str[:2]
print(f"Rural panel: {len(df_raw):,} rows | {df_raw['fips'].nunique():,} counties | "
      f"years {int(df_raw['year'].min())}-{int(df_raw['year'].max())}")

# The main "*_panel.csv" does not carry FSIS establishment counts (confirmed
# 100% NaN for n_unique_establishments_fsis etc.) -- FSIS lives in a separate
# file, "*_panel_fsis.csv", merged in here on (fips, year). FSIS coverage is
# 2017-2023 only, ~46-50% of rural counties per year (used below for the
# CAFO x FSIS interaction, Block 1e).
try:
    _fsis_path = latest_file_glob(merged_dir, "*_panel_fsis.csv")
    _fsis = pd.read_csv(_fsis_path, low_memory=False)[
        ["fips", "year", "n_unique_establishments_fsis"]
    ]
    df_raw = df_raw.merge(_fsis, on=["fips", "year"], how="left")
    print(f"FSIS merged from {os.path.basename(_fsis_path)}: "
          f"{df_raw['n_unique_establishments_fsis'].notna().sum():,} non-null county-years "
          f"({int(_fsis['year'].min())}-{int(_fsis['year'].max())})")
except FileNotFoundError:
    df_raw["n_unique_establishments_fsis"] = np.nan
    print("FSIS panel file not found -- CAFO x FSIS interaction (Block 1e) will be skipped.")

# =============================================================================
# TREATMENT VARIABLE DICTIONARY -- every dairy-CAFO variable built below,
# what it measures, and which comparisons it feeds. Built out incrementally
# per team discussion; documented here so the different model versions in
# this script don't have to be reverse-engineered from the code.
#
#   any_large_{animal}      binary: >=1 LARGE {animal} CAFO in this county-year.
#                            Built for hogs/beef/dairy/chickens -- feeds the
#                            isolated-vs-conditional "horse race" (Block 1b).
#   any_medlarge_dairy       binary: >=1 MEDIUM-OR-LARGE dairy CAFO.
#   any_dairy                binary: >=1 dairy CAFO of ANY size (small/medium/large).
#     -> any_large_dairy / any_medlarge_dairy / any_dairy feed the size-threshold
#        comparison (Block 1c): does requiring "large" specifically matter?
#   cafo_dairy_large          raw COUNT of large dairy CAFOs (not divided by
#                              population). Used with log_pop as a SEPARATE
#                              control, rather than folded into the treatment
#                              itself -- see Block 1d.
#   cafo_dairy_large_log_ct   log1p(raw count) -- same idea, logged.
#   cafo_dairy_large_raw      raw count DIVIDED BY population, per 10k residents,
#                              NOT logged (Part b ridge + Block 1d).
#   cafo_dairy_large_log      log1p(per-10k rate) -- the transform used
#                              throughout Part (a)/(b) prior to this review.
#     -> These 4 encode two separate choices that get conflated if not kept
#        distinct: (i) per-capita RATE (divide by population, bakes in a
#        1/population functional form) vs. RAW COUNT + a separate log_pop
#        control (lets the data estimate the population relationship
#        freely); (ii) logged vs. unlogged. Block 1d runs all 4 (+ binary
#        presence) side by side so this is tested, not assumed.
#   any_fsis                  binary: >=1 FSIS-registered establishment
#                              (slaughter/processing) in this county-year,
#                              2017-2023 only. IMPORTANT: the FSIS source file
#                              lists ONLY county-years with >=1 establishment
#                              (confirmed: min non-null value = 1, never 0) --
#                              a county absent from that file is not "unknown,"
#                              it has zero establishments. So absence is
#                              filled to 0 within the confirmed 2017-2023
#                              coverage window, not left as NaN -- treating it
#                              as NaN would make any_fsis constant (=1)
#                              wherever non-missing, which is what happened
#                              on the first pass (any_fsis got a ~0 coefficient
#                              and the interaction term collapsed onto
#                              any_large_dairy alone -- caught and fixed here).
#                              This is a *different* situation from the CDC
#                              despair data, where absence genuinely is
#                              ambiguous (suppression) -- FSIS is a federal
#                              regulatory registry, not a survey.
#   any_large_dairy_x_fsis    interaction: any_large_dairy * any_fsis --
#                              "has both a large dairy CAFO AND FSIS-registered
#                              processing/slaughter capacity" (Block 1e).
# =============================================================================

# Presence indicators for all 4 CAFO animal types -- needed for the isolated
# vs. conditional ("horse race") TWFE comparison below. .where(notna()) keeps
# a missing large-op count as missing rather than coding it as "absent",
# matching the external memo's twfe.py construction.
for _animal in ["hogs", "beef", "dairy", "chickens"]:
    _col = f"cafo_{_animal}_large"
    df_raw[f"any_large_{_animal}"] = (df_raw[_col] > 0).astype(float).where(df_raw[_col].notna())

# Dairy at three size thresholds -- large only (used everywhere above), medium
# + large combined, and any size at all (cafo_dairy_total, small+medium+large).
_dairy_medlarge = df_raw["cafo_dairy_medium"].fillna(0) + df_raw["cafo_dairy_large"].fillna(0)
_dairy_medlarge_na = df_raw["cafo_dairy_medium"].isna() & df_raw["cafo_dairy_large"].isna()
df_raw["any_medlarge_dairy"] = (_dairy_medlarge > 0).astype(float).where(~_dairy_medlarge_na)
df_raw["any_dairy"] = (df_raw["cafo_dairy_total"] > 0).astype(float).where(df_raw["cafo_dairy_total"].notna())

DAIRY_THRESHOLDS = {
    "Large only":       "any_large_dairy",
    "Medium + large":   "any_medlarge_dairy",
    "Any size":         "any_dairy",
}

# Raw count (not divided by population) and its log, for the functional-form
# comparison in Block 1d -- these get log_pop as a SEPARATE control instead
# of being pre-divided by population.
df_raw["cafo_dairy_large_log_ct"] = np.log1p(df_raw["cafo_dairy_large"])
df_raw["log_pop"] = np.log(df_raw[POP_COL].where(df_raw[POP_COL] > 0))

# FSIS presence + CAFO x FSIS interaction (Block 1e). Absence within the
# confirmed 2017-2023 coverage window is filled to 0 (true zero establishments,
# not missing -- see dictionary note above); outside that window, NaN.
_fsis_covered_years = df_raw["year"].between(2017, 2023)
df_raw["any_fsis"] = np.where(
    _fsis_covered_years,
    (df_raw["n_unique_establishments_fsis"].fillna(0) > 0).astype(float),
    np.nan,
)
df_raw["any_large_dairy_x_fsis"] = df_raw["any_large_dairy"] * df_raw["any_fsis"]

# ── Outcome construction fixes (found reviewing Galina's cafo_analysis_files/) ─
# Despair: crude_rate_despair is CDC's own pre-computed crude rate, which CDC
# suppresses/flags as unreliable at low death counts. crude_rate_from_census_pop
# (built in script1b-generate-panel.py, deaths / census population * 100k) was
# built for exactly this reason -- QA sense-check shows corr=0.9999 against
# CDC's own rate wherever both exist, and it recovers ~2x the usable sample
# (27% vs 13% coverage in recent years) by not being suppressed. Both variants
# hit a hard cliff to 0% coverage after 2020 (source data itself, not
# suppression) -- truncated explicitly below rather than relying on it being
# implicitly NaN.
DESPAIR_COL = "crude_rate_from_census_pop"
df_raw.loc[df_raw["year"] > 2020, DESPAIR_COL] = np.nan

# Assault: combine aggravated + simple assault (NIBRS), NaN-propagated the same
# way as total_incidents_per100k's own coverage -- matches how the external
# memo's own pipeline builds this outcome, rather than aggravated assault alone.
df_raw["crime_assault"] = (
    df_raw["aggravated_assault_per100k"].fillna(0) + df_raw["simple_assault_per100k"].fillna(0)
)
df_raw.loc[df_raw["total_incidents_per100k"].isna(), "crime_assault"] = np.nan

# ── Outcomes (hard MH + despair + crime, per team discussion) ────────────────
OUTCOMES = {
    "Poor MH Days":            "poor_mental_health_days",
    "Frequent Mental Distress":"frequent_mental_distress_per100k",
    "Deaths of Despair":       DESPAIR_COL,
    "Violent Crime (CHR)":     "violent_crime",
    "Assault (Agg+Simple)":    "crime_assault",
    "Total Incidents (NIBRS)": "total_incidents_per100k",
}

# Control pool — same demographic/health controls used elsewhere in the repo
# (script3-ridge.py CONTROL_COLS), reused here for consistency across scripts.
CONTROL_COLS = [
    "adult_obesity_per100k", "adult_smoking_per100k", "unemployment_per100k",
    "children_in_poverty_per100k", "uninsured_adults_per100k",
    "median_household_income", "income_inequality",
    "%_hispanic", "%_non-hispanic_african_american", "%_65_and_older", "%_female",
    "mental_health_providers_per100k", "primary_care_physicians_per100k",
    "food_insecurity_per100k", "physical_inactivity_per100k",
    "poor_physical_health_days", "teen_births_per100k", "low_birthweight_per100k",
    "premature_death", "preventable_hospital_stays", "some_college_per100k",
    "children_in_single-parent_households_per100k", "diabetes_prevalence_per100k",
    "air_pollution_-_particulate_matter", "social_associations_per100k",
]

# =============================================================================
# Helpers — generic two-way FE machinery (mirrors script3-ridge.py's approach,
# parameterized for cluster variable + confidence level so we can match the
# external memo's spec exactly for the replication, rather than this repo's
# usual default of county-clustered / 95% CI).
# =============================================================================

def within_transform(df, y_col, x_cols, entity="fips", time="year"):
    """Two-way within estimator: demean by entity then by time period."""
    keep = [y_col] + x_cols + [entity, time]
    out = df[keep].dropna().copy()
    for col in [y_col] + x_cols:
        grand_mean  = out[col].mean()
        entity_mean = out.groupby(entity)[col].transform("mean")
        time_mean   = out.groupby(time)[col].transform("mean")
        out[col]    = out[col] - entity_mean - time_mean + grand_mean
    return out


def run_fe_ols(df, treatment_col, outcome_col, control_cols, entity="fips", time="year",
               cluster_col="state_fips", z=1.96, label=""):
    """
    Two-way FE (within transformation) OLS with clustered SEs.
    z=1.96 -> 95% CI (this repo's convention); pass z=1.645 for 90% CI (memo's spec).
    """
    x_cols = [treatment_col] + [c for c in control_cols if c != treatment_col]
    keep_extra = [cluster_col] if cluster_col not in (entity, time) else []
    dm = within_transform(df[[*{outcome_col, *x_cols, entity, time, *keep_extra}]],
                           outcome_col, x_cols, entity=entity, time=time)
    if cluster_col not in dm.columns and cluster_col in df.columns:
        dm = dm.join(df[cluster_col])
    if len(dm) < 200:
        print(f"    [{label}] too few obs ({len(dm)}), skip")
        return None
    Y = dm[outcome_col]
    X = sm.add_constant(dm[x_cols], has_constant="add")
    groups = dm[cluster_col] if cluster_col in dm.columns else dm[entity]
    try:
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception as e:
        print(f"    [{label}] OLS failed: {e}")
        return None
    beta, se = res.params.get(treatment_col, np.nan), res.bse.get(treatment_col, np.nan)
    return {"beta": beta, "se": se, "ci_lo": beta - z*se, "ci_hi": beta + z*se,
            "pval": res.pvalues.get(treatment_col, np.nan), "N": int(res.nobs), "r2": res.rsquared}


def run_fe_ols_multi(df, treatment_cols, outcome_col, control_cols, entity="fips", time="year",
                      cluster_col="state_fips", z=1.96, label=""):
    """
    Two-way FE OLS with MULTIPLE treatment columns entered simultaneously
    ("horse race" -- each animal type's coefficient is conditional on the
    others also being in the model), vs. run_fe_ols's single-treatment
    (isolated) spec. Matches the external memo's twfe.py, which enters
    any_large_hogs/beef/dairy/chickens together in one regression. Returns
    a dict keyed by treatment column, one result per co-estimated coefficient.
    """
    x_cols = list(treatment_cols) + [c for c in control_cols if c not in treatment_cols]
    keep_extra = [cluster_col] if cluster_col not in (entity, time) else []
    dm = within_transform(df[[*{outcome_col, *x_cols, entity, time, *keep_extra}]],
                           outcome_col, x_cols, entity=entity, time=time)
    if cluster_col not in dm.columns and cluster_col in df.columns:
        dm = dm.join(df[cluster_col])
    if len(dm) < 200:
        print(f"    [{label}] too few obs ({len(dm)}), skip")
        return {}
    Y = dm[outcome_col]
    X = sm.add_constant(dm[x_cols], has_constant="add")
    groups = dm[cluster_col] if cluster_col in dm.columns else dm[entity]
    try:
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception as e:
        print(f"    [{label}] OLS failed: {e}")
        return {}
    out = {}
    for tcol in treatment_cols:
        beta, se = res.params.get(tcol, np.nan), res.bse.get(tcol, np.nan)
        out[tcol] = {"beta": beta, "se": se, "ci_lo": beta - z*se, "ci_hi": beta + z*se,
                     "pval": res.pvalues.get(tcol, np.nan), "N": int(res.nobs), "r2": res.rsquared}
    return out


def run_pooled_ols(df, treatment_col, outcome_col, control_cols, fe_cols,
                    cluster_col="state_fips", z=1.96, label=""):
    """
    Pooled OLS with dummy FE (e.g. state + year) rather than county FE —
    used for the cross-sectional / "levels" replication (memo Section 2).
    """
    x_cols = [treatment_col] + [c for c in control_cols if c != treatment_col]
    cols_needed = list(set([outcome_col, *x_cols, cluster_col, *fe_cols]))
    sub = df[cols_needed].dropna(subset=[outcome_col, treatment_col]).copy()
    if len(sub) < 200:
        print(f"    [{label}] too few obs, skip")
        return None
    X = sub[x_cols].copy()
    for fe in fe_cols:
        # .astype("string") -> get_dummies returns nullable "boolean" dtype, which
        # statsmodels/numpy cannot cast to a numeric array alongside float64 columns
        # ("Pandas data cast to numpy dtype of object"). Force plain float dummies.
        dummies = pd.get_dummies(sub[fe].astype("string"), prefix=fe, drop_first=True).astype(float)
        X = pd.concat([X, dummies], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    keep_mask = X.notna().all(axis=1) & sub[outcome_col].notna()
    X, y, groups = X[keep_mask], sub.loc[keep_mask, outcome_col], sub.loc[keep_mask, cluster_col]
    X = sm.add_constant(X, has_constant="add")
    try:
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception as e:
        print(f"    [{label}] pooled OLS failed: {e}")
        return None
    beta, se = res.params.get(treatment_col, np.nan), res.bse.get(treatment_col, np.nan)
    return {"beta": beta, "se": se, "ci_lo": beta - z*se, "ci_hi": beta + z*se,
            "pval": res.pvalues.get(treatment_col, np.nan), "N": int(res.nobs), "r2": res.rsquared}


def build_entry_cohort(df, size_cols, census_years, entity="fips", time="year"):
    """
    Generic "first large-CAFO entry" cohort builder for a given animal type.
    size_cols: list of column(s) whose sum defines "large ops of this type"
               (kept as a list so callers can sum multiple size/animal columns
               if ever extended beyond dairy).
    Returns (df_cens, entry_events) where entry_events has one row per county
    that goes from 0 -> >0 large ops between consecutive Census waves,
    excluding counties already treated at the first available wave
    (always-treated, per memo's convention).
    """
    d = df.copy()
    d["_large_total"] = d[size_cols].sum(axis=1, min_count=1)
    d_cens = d[d[time].isin(census_years)].sort_values([entity, time]).copy()
    d_cens["_large_lag"] = d_cens.groupby(entity)["_large_total"].shift(1)

    entry = (
        (d_cens["_large_total"] > 0) &
        (d_cens["_large_lag"] == 0) &
        d_cens["_large_lag"].notna()
    )
    entry_events = d_cens[entry][[entity, time]].rename(columns={time: "event_year"})
    return d, entry_events


def run_event_study(df_full, events_df, outcome_col, control_cols=None,
                     entity="fips", time="year", cluster_col="state_fips",
                     n_leads=10, n_lags=9, z=1.645, label=""):
    """
    TWFE event-study regression. Omitted category: t_rel = -1.
    Control group: never-treated + not-yet-treated counties (pooled).
    z=1.645 -> 90% CI, matching the external memo's reported spec.
    """
    control_cols = control_cols or []
    treated_fips = set(events_df[entity].unique())
    first_event = events_df.groupby(entity)["event_year"].min().reset_index()

    keep_cols = list(set([outcome_col, entity, time, cluster_col, *control_cols]))
    d = df_full[keep_cols].merge(first_event, on=entity, how="left")
    never_fips = set(df_full[entity].unique()) - treated_fips

    d["t_rel"] = np.where(d["event_year"].notna(), d[time] - d["event_year"], np.nan)
    d_treated = d[d[entity].isin(treated_fips) & d["t_rel"].between(-n_leads, n_lags)].copy()
    d_never = d[d[entity].isin(never_fips)].copy()
    d_never["t_rel"] = np.nan
    d_pool = pd.concat([d_treated, d_never], ignore_index=True)

    if d_pool[outcome_col].notna().sum() < 200:
        print(f"    [{label}] too few obs, skip")
        return pd.DataFrame(), {}

    rel_times = [t for t in range(-n_leads, n_lags + 1) if t != -1]
    for t in rel_times:
        d_pool[f"d_t{t:+d}"] = (d_pool["t_rel"] == t).astype(float)
    dummy_cols = [f"d_t{t:+d}" for t in rel_times]
    x_cols = dummy_cols + control_cols

    dm = within_transform(d_pool, outcome_col, x_cols, entity=entity, time=time)
    if cluster_col not in dm.columns:
        dm = dm.join(d_pool[cluster_col])
    if len(dm) < 200:
        return pd.DataFrame(), {}

    Y = dm[outcome_col]
    X = sm.add_constant(dm[x_cols], has_constant="add")

    # With few entry cohorts (here: 4 non-2002 Census waves) and outcomes that
    # only start partway through the panel (e.g. CHR mental-health vars from
    # 2010), extreme leads/lags can be populated by only ONE cohort in ANY
    # given calendar year -- making that event-time dummy perfectly collinear
    # with the year FE it's demeaned against. Plain OLS does not detect this;
    # it silently returns a minimum-norm solution that (mis)assigns the same
    # combined effect across the collinear dummies (visible as identical
    # beta/SE across adjacent relative-time points). pyfixest (used in the
    # external memo) drops these automatically -- we do the same explicitly,
    # dropping the least-supported dummy one at a time until full rank.
    dropped_t = []
    active_dummies = list(dummy_cols)
    while True:
        cols_now = active_dummies + control_cols
        X_now = sm.add_constant(dm[cols_now], has_constant="add")
        rank = np.linalg.matrix_rank(X_now.values)
        if rank >= X_now.shape[1] or not active_dummies:
            break
        support = dm[active_dummies].abs().sum().sort_values()
        weakest = support.index[0]
        dropped_t.append(int(weakest.replace("d_t", "")))
        active_dummies.remove(weakest)
    if dropped_t:
        print(f"    [{label}] dropped {len(dropped_t)} unidentified (collinear) "
              f"event-time dummy(ies): t_rel={sorted(dropped_t)}")

    x_cols_final = active_dummies + control_cols
    X = sm.add_constant(dm[x_cols_final], has_constant="add")
    try:
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": dm[cluster_col]})
    except Exception as e:
        print(f"    [{label}] event-study OLS failed: {e}")
        return pd.DataFrame(), {}

    rows = []
    for t in rel_times:
        col = f"d_t{t:+d}"
        if col in res.params.index:
            beta, se = res.params[col], res.bse[col]
            rows.append({"t_rel": t, "beta": beta, "se": se,
                         "ci_lo": beta - z*se, "ci_hi": beta + z*se,
                         "pval": res.pvalues[col], "outcome": label})
        else:
            rows.append({"t_rel": t, "beta": np.nan, "se": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan,
                         "pval": np.nan, "outcome": label})
    rows.append({"t_rel": -1, "beta": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                 "pval": np.nan, "outcome": label})
    es_df = pd.DataFrame(rows).sort_values("t_rel")

    # --- Joint significance (Wald/F) tests, using the fitted cluster-robust
    # covariance -- NOT eyeballing individual noisy coefficients one at a
    # time. Pre-period test = formal parallel-pre-trends check (H0: all
    # pre-treatment event-time coefficients are jointly zero). Post-period
    # test = whether the treatment effect is jointly distinguishable from
    # zero across the post window, taken together.
    pre_terms  = [c for c in active_dummies if int(c.replace("d_t", "")) < -1]
    post_terms = [c for c in active_dummies if int(c.replace("d_t", "")) >= 0]

    def _joint_test(terms):
        if len(terms) < 1:
            return {"F": np.nan, "df_num": 0, "df_denom": np.nan, "pval": np.nan, "n_terms": 0}
        restriction = ", ".join(f"{t} = 0" for t in terms)
        try:
            wt = res.f_test(restriction)
            return {"F": float(wt.fvalue), "df_num": int(wt.df_num),
                    "df_denom": float(wt.df_denom), "pval": float(wt.pvalue),
                    "n_terms": len(terms)}
        except Exception as e:
            print(f"    [{label}] joint test failed on {terms}: {e}")
            return {"F": np.nan, "df_num": len(terms), "df_denom": np.nan, "pval": np.nan, "n_terms": len(terms)}

    joint = {
        "outcome": label,
        "pre":  _joint_test(pre_terms),
        "post": _joint_test(post_terms),
    }
    print(f"    [{label}] joint pre-trend test:  F({joint['pre']['df_num']},{joint['pre']['df_denom']:.0f})="
          f"{joint['pre']['F']:.2f}  p={joint['pre']['pval']:.3f}  "
          f"({'REJECT flat pre-trend' if joint['pre']['pval'] < 0.05 else 'cannot reject flat pre-trend'})")
    print(f"    [{label}] joint post-effect test: F({joint['post']['df_num']},{joint['post']['df_denom']:.0f})="
          f"{joint['post']['F']:.2f}  p={joint['post']['pval']:.3f}  "
          f"({'jointly significant post effect' if joint['post']['pval'] < 0.05 else 'not jointly significant'})")

    return es_df, joint


# =============================================================================
# PART (a): Replicate the dairy-CAFO finding
# =============================================================================
print("\n" + "="*78)
print("PART (a): Dairy CAFO replication — levels, within, event study")
print("="*78)

CENSUS_YEARS = [2002, 2007, 2012, 2017, 2022]
DAIRY_LARGE_COL = "cafo_dairy_large"

df_dairy, dairy_entry = build_entry_cohort(
    df_raw, size_cols=[DAIRY_LARGE_COL], census_years=CENSUS_YEARS,
)
df_dairy["dairy_large_present"] = (df_dairy[DAIRY_LARGE_COL].fillna(0) > 0).astype(float)
print(f"Dairy entry events: {len(dairy_entry):,} county-census-year obs "
      f"({dairy_entry['fips'].nunique():,} unique counties)")

# --- Block 1: levels vs within, binary presence -----------------------------
levels_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    ctrl = [c for c in CONTROL_COLS if c in df_dairy.columns]

    pooled = run_pooled_ols(
        df_dairy, "dairy_large_present", outcome_col, ctrl,
        fe_cols=["state_fips", "year"], z=1.96, label=f"pooled|{outcome_key}",
    )
    within = run_fe_ols(
        df_dairy, "dairy_large_present", outcome_col, ctrl,
        cluster_col="state_fips", z=1.96, label=f"within|{outcome_key}",
    )
    for spec_name, res in [("Pooled (state+year FE)", pooled), ("Within (county+year FE)", within)]:
        if res is None:
            continue
        levels_rows.append({"outcome": outcome_key, "spec": spec_name, **res})
        sig = "*" if res["pval"] < 0.05 else " "
        print(f"  {spec_name:26s} | {outcome_key:26s} beta={res['beta']:+.4f}  "
              f"p={res['pval']:.3f}{sig}  N={res['N']:,}")

levels_df = pd.DataFrame(levels_rows)
levels_csv = os.path.join(tables_s4_dir, f"{today_str}_Block1_dairy_levels_vs_within.csv")
levels_df.to_csv(levels_csv, index=False)
print("Saved:", levels_csv)

# Figure Y1: levels vs within comparison
fig, ax = plt.subplots(figsize=(9, 6))
spec_colors = {"Pooled (state+year FE)": "#4393c3", "Within (county+year FE)": "#d6604d"}
y_labels, y_pos = [], []
for i, outcome_key in enumerate(OUTCOMES.keys()):
    sub = levels_df[levels_df["outcome"] == outcome_key]
    for j, spec in enumerate(spec_colors):
        row = sub[sub["spec"] == spec]
        if row.empty:
            continue
        row = row.iloc[0]
        yy = i * 3 + j * 0.9
        ax.errorbar(row["beta"], yy, xerr=[[row["beta"]-row["ci_lo"]], [row["ci_hi"]-row["beta"]]],
                    fmt="o", color=spec_colors[spec], ms=6, capsize=3, elinewidth=1.2,
                    markeredgecolor="white")
        if row["pval"] < 0.05:
            ax.text(row["ci_hi"] + 0.01, yy, "*", va="center", fontsize=10, color=spec_colors[spec])
    y_labels.append(outcome_key)
    y_pos.append(i * 3 + 0.45)
ax.axvline(0, color="black", lw=0.8, ls=":")
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel("beta (large dairy CAFO presence)", fontsize=9)
legend_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=k)
                  for k, c in spec_colors.items()]
ax.legend(handles=legend_handles, fontsize=8, loc="best")
ax.set_title(
    "Dairy CAFO presence: pooled (cross-sectional) vs within-county coefficients\n"
    "Replication check against external memo Sections 2-3 | * = p < 0.05",
    fontsize=10,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_Y1_dairy_levels_vs_within.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)

# --- Block 1b: isolated vs conditional ("horse race") dairy coefficient -----
# Isolated: dairy alone (within, county+year FE) -- same as Block 1's within row.
# Conditional: dairy's coefficient with hogs/beef/chickens presence ALSO in
# the regression -- matches the external memo's twfe.py, which enters all
# four animal types simultaneously. Controls held identical across both specs
# so the only thing that changes is whether the other 3 CAFO types are
# partialled out -- isolates "does conditioning on other animal types change
# the dairy answer" from "how many demographic controls are included."
print("\nIsolated vs conditional (horse race) dairy coefficient...")
ANIMAL_PRESENCE_COLS = ["any_large_dairy", "any_large_beef", "any_large_hogs", "any_large_chickens"]
horserace_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    ctrl = [c for c in CONTROL_COLS if c in df_raw.columns]

    isolated = run_fe_ols_multi(
        df_raw, ["any_large_dairy"], outcome_col, ctrl,
        cluster_col="state_fips", z=1.96, label=f"isolated|{outcome_key}",
    )
    conditional = run_fe_ols_multi(
        df_raw, ANIMAL_PRESENCE_COLS, outcome_col, ctrl,
        cluster_col="state_fips", z=1.96, label=f"conditional|{outcome_key}",
    )
    for spec_name, res_dict in [("Isolated (dairy alone)", isolated),
                                 ("Conditional (+ beef/hogs/chickens)", conditional)]:
        res = res_dict.get("any_large_dairy")
        if res is None:
            continue
        horserace_rows.append({"outcome": outcome_key, "spec": spec_name, **res})
        sig = "*" if res["pval"] < 0.05 else " "
        print(f"  {spec_name:36s} | {outcome_key:24s} beta={res['beta']:+.4f}  "
              f"p={res['pval']:.3f}{sig}  N={res['N']:,}")

horserace_df = pd.DataFrame(horserace_rows)
hr_csv = os.path.join(tables_s4_dir, f"{today_str}_Block1b_dairy_isolated_vs_conditional.csv")
horserace_df.to_csv(hr_csv, index=False)
print("Saved:", hr_csv)

# Figure Y1b: isolated vs conditional comparison
fig, ax = plt.subplots(figsize=(9, 6))
hr_colors = {"Isolated (dairy alone)": "#1b7837", "Conditional (+ beef/hogs/chickens)": "#762a83"}
y_labels, y_pos = [], []
for i, outcome_key in enumerate(OUTCOMES.keys()):
    sub = horserace_df[horserace_df["outcome"] == outcome_key]
    for j, spec in enumerate(hr_colors):
        row = sub[sub["spec"] == spec]
        if row.empty:
            continue
        row = row.iloc[0]
        yy = i * 3 + j * 0.9
        ax.errorbar(row["beta"], yy, xerr=[[row["beta"]-row["ci_lo"]], [row["ci_hi"]-row["beta"]]],
                    fmt="o", color=hr_colors[spec], ms=6, capsize=3, elinewidth=1.2,
                    markeredgecolor="white")
        if row["pval"] < 0.05:
            ax.text(row["ci_hi"] + 0.01, yy, "*", va="center", fontsize=10, color=hr_colors[spec])
    y_labels.append(outcome_key)
    y_pos.append(i * 3 + 0.45)
ax.axvline(0, color="black", lw=0.8, ls=":")
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel("beta (large dairy CAFO presence), within county+year FE", fontsize=9)
legend_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=k)
                  for k, c in hr_colors.items()]
ax.legend(handles=legend_handles, fontsize=8, loc="best")
ax.set_title(
    "Dairy coefficient: isolated vs. conditional on other CAFO animal types\n"
    "Same controls both specs | * = p < 0.05 | matches memo's twfe.py horse-race design",
    fontsize=10,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_Y1b_dairy_isolated_vs_conditional.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)

# --- Block 1c: dairy size threshold comparison (large / medium+large / any) -
# Same within (county+year FE) spec, same controls, only the treatment
# definition changes: does requiring LARGE specifically matter, or would
# medium+large or any dairy CAFO at all give a similar answer?
print("\nDairy size threshold comparison (large vs medium+large vs any size)...")
threshold_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    ctrl = [c for c in CONTROL_COLS if c in df_raw.columns]
    for thresh_label, thresh_col in DAIRY_THRESHOLDS.items():
        res = run_fe_ols(df_raw, thresh_col, outcome_col, ctrl,
                          cluster_col="state_fips", z=1.96, label=f"{thresh_label}|{outcome_key}")
        if res is None:
            continue
        threshold_rows.append({"outcome": outcome_key, "threshold": thresh_label, **res})
        sig = "*" if res["pval"] < 0.05 else " "
        print(f"  {thresh_label:16s} | {outcome_key:26s} beta={res['beta']:+.4f}  "
              f"p={res['pval']:.3f}{sig}  N={res['N']:,}")

threshold_df = pd.DataFrame(threshold_rows)
th_csv = os.path.join(tables_s4_dir, f"{today_str}_Block1c_dairy_size_threshold.csv")
threshold_df.to_csv(th_csv, index=False)
print("Saved:", th_csv)

# Figure Y1c: threshold comparison
fig, ax = plt.subplots(figsize=(9, 6))
th_colors = {"Large only": "#762a83", "Medium + large": "#b35806", "Any size": "#4393c3"}
y_labels, y_pos = [], []
for i, outcome_key in enumerate(OUTCOMES.keys()):
    sub = threshold_df[threshold_df["outcome"] == outcome_key]
    for j, thresh in enumerate(th_colors):
        row = sub[sub["threshold"] == thresh]
        if row.empty:
            continue
        row = row.iloc[0]
        yy = i * 4 + j * 0.9
        ax.errorbar(row["beta"], yy, xerr=[[row["beta"]-row["ci_lo"]], [row["ci_hi"]-row["beta"]]],
                    fmt="o", color=th_colors[thresh], ms=6, capsize=3, elinewidth=1.2,
                    markeredgecolor="white")
        if row["pval"] < 0.05:
            ax.text(row["ci_hi"] + 0.01, yy, "*", va="center", fontsize=10, color=th_colors[thresh])
    y_labels.append(outcome_key)
    y_pos.append(i * 4 + 0.9)
ax.axvline(0, color="black", lw=0.8, ls=":")
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel("beta (dairy CAFO presence), within county+year FE", fontsize=9)
legend_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=k)
                  for k, c in th_colors.items()]
ax.legend(handles=legend_handles, fontsize=8, loc="best")
ax.set_title(
    "Dairy CAFO presence, by size threshold: large only vs medium+large vs any size\n"
    "Same controls, same within-county+year FE spec | * = p < 0.05",
    fontsize=10,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_Y1c_dairy_size_threshold.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)

# --- Block 1d: functional form -- per-capita rate vs. raw count + log_pop --
# large dairy only, 5 functional forms of the SAME underlying variable, all
# with the same demographic controls (log_pop included explicitly in every
# row here, including the per-10k versions, so the comparison is apples to
# apples): binary presence, raw count, log(raw count), per-10k rate
# (unlogged), log(per-10k rate). Tests whether (i) it matters how many CAFOs
# exist (binary vs. count) and (ii) dividing by population upfront vs.
# controlling for population separately changes the answer.
print("\nFunctional form comparison (large dairy): presence vs count vs per-capita, log vs raw...")
FUNCTIONAL_FORMS = {
    "Binary presence":         "any_large_dairy",
    "Raw count":               "cafo_dairy_large",
    "Log(raw count)":          "cafo_dairy_large_log_ct",
    "Per-10k rate (unlogged)": "cafo_dairy_large_raw" if "cafo_dairy_large_raw" in df_raw.columns else None,
    "Log(per-10k rate)":       "cafo_dairy_large_log" if "cafo_dairy_large_log" in df_raw.columns else None,
}
# Per-10k rate columns are built later, in Part (b) -- compute them here too
# so this comparison doesn't depend on run order.
_pop_safe = df_raw[POP_COL].replace(0, np.nan)
if "cafo_dairy_large_raw" not in df_raw.columns:
    df_raw["cafo_dairy_large_raw"] = (df_raw["cafo_dairy_large"] / _pop_safe) * 10_000
    FUNCTIONAL_FORMS["Per-10k rate (unlogged)"] = "cafo_dairy_large_raw"
if "cafo_dairy_large_log" not in df_raw.columns:
    df_raw["cafo_dairy_large_log"] = np.log1p(df_raw["cafo_dairy_large_raw"])
    FUNCTIONAL_FORMS["Log(per-10k rate)"] = "cafo_dairy_large_log"

form_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    ctrl_base = [c for c in CONTROL_COLS if c in df_raw.columns]
    for form_label, form_col in FUNCTIONAL_FORMS.items():
        # log_pop included as a control in every row -- for the per-10k
        # (already population-normalized) versions this is a genuine
        # over-specification check: does the per-capita rate still "work"
        # once population is ALSO separately controlled for?
        ctrl = ["log_pop"] + [c for c in ctrl_base if c != "log_pop"]
        res = run_fe_ols(df_raw, form_col, outcome_col, ctrl,
                          cluster_col="state_fips", z=1.96, label=f"{form_label}|{outcome_key}")
        if res is None:
            continue
        form_rows.append({"outcome": outcome_key, "form": form_label, **res})
        sig = "*" if res["pval"] < 0.05 else " "
        print(f"  {form_label:26s} | {outcome_key:26s} beta={res['beta']:+.5f}  "
              f"p={res['pval']:.3f}{sig}  N={res['N']:,}")

form_df = pd.DataFrame(form_rows)
form_csv = os.path.join(tables_s4_dir, f"{today_str}_Block1d_dairy_functional_form.csv")
form_df.to_csv(form_csv, index=False)
print("Saved:", form_csv)

# --- Block 1e: CAFO x FSIS interaction (exploratory -- thin sample) ---------
# FSIS only exists 2017-2023 at ~46-50% county coverage, so this is
# necessarily underpowered relative to the rest of Part (a) -- flagged
# explicitly rather than presented on equal footing with the fuller-sample
# results above. Single regression per outcome: dairy alone, FSIS alone, and
# their interaction, together -- not three separate regressions -- so the
# interaction term is read as "the ADDITIONAL association with having both,
# beyond dairy alone and FSIS alone."
print("\nCAFO x FSIS interaction (2017-2023 only, ~46-50% county coverage -- exploratory)...")
interaction_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    ctrl = [c for c in CONTROL_COLS if c in df_raw.columns]
    res_dict = run_fe_ols_multi(
        df_raw, ["any_large_dairy", "any_fsis", "any_large_dairy_x_fsis"], outcome_col, ctrl,
        cluster_col="state_fips", z=1.96, label=f"dairy_x_fsis|{outcome_key}",
    )
    for term in ["any_large_dairy", "any_fsis", "any_large_dairy_x_fsis"]:
        res = res_dict.get(term)
        if res is None:
            continue
        interaction_rows.append({"outcome": outcome_key, "term": term, **res})
        sig = "*" if res["pval"] < 0.05 else " "
        print(f"  {term:26s} | {outcome_key:26s} beta={res['beta']:+.4f}  p={res['pval']:.3f}{sig}  N={res['N']:,}")

interaction_df = pd.DataFrame(interaction_rows)
inter_csv = os.path.join(tables_s4_dir, f"{today_str}_Block1e_dairy_fsis_interaction.csv")
interaction_df.to_csv(inter_csv, index=False)
print("Saved:", inter_csv)

# --- Block 2: event study around dairy entry --------------------------------
print("\nEvent study (state-clustered SEs, 90% CI, to match memo spec)...")
print("Joint (Wald/F) tests use the same cluster-robust covariance as the point")
print("estimates -- this is the correct way to assess pre-trends/post-effects,")
print("not eyeballing individually noisy per-lag coefficients one at a time.\n")
es_rows = []
joint_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    ctrl = [c for c in CONTROL_COLS if c in df_dairy.columns]
    es, joint = run_event_study(
        df_dairy, dairy_entry, outcome_col, control_cols=ctrl,
        cluster_col="state_fips", n_leads=10, n_lags=9, z=1.645, label=outcome_key,
    )
    if not es.empty:
        es_rows.append(es)
    if joint:
        for period in ("pre", "post"):
            joint_rows.append({"outcome": outcome_key, "period": period, **joint[period]})

es_df = pd.concat(es_rows, ignore_index=True) if es_rows else pd.DataFrame()
es_csv = os.path.join(tables_s4_dir, f"{today_str}_Block2_dairy_event_study.csv")
es_df.to_csv(es_csv, index=False)

joint_df = pd.DataFrame(joint_rows)
joint_csv = os.path.join(tables_s4_dir, f"{today_str}_Block2b_dairy_event_study_joint_tests.csv")
joint_df.to_csv(joint_csv, index=False)
print("Saved:", joint_csv)
print("Saved:", es_csv)

# Figure Y2: event study grid, one panel per outcome
n_out = len(OUTCOMES)
n_cols_es = 3
n_rows_es = int(np.ceil(n_out / n_cols_es))
fig, axes = plt.subplots(n_rows_es, n_cols_es, figsize=(n_cols_es*5, n_rows_es*4))
axes = axes.flatten()
for i, outcome_key in enumerate(OUTCOMES.keys()):
    ax = axes[i]
    sub = es_df[es_df["outcome"] == outcome_key].sort_values("t_rel") if not es_df.empty else pd.DataFrame()
    if sub.empty:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        ax.set_title(outcome_key, fontsize=9)
        continue
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(-0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax.errorbar(sub["t_rel"], sub["beta"], yerr=[sub["beta"]-sub["ci_lo"], sub["ci_hi"]-sub["beta"]],
                fmt="o", color="#762a83", ms=5, capsize=3, elinewidth=1)
    # Formal joint (Wald/F) tests, not an eyeballed max|beta| heuristic --
    # correctly assesses the pre-trend/post-effect block using the same
    # cluster-robust covariance as the point estimates.
    jt = joint_df[joint_df["outcome"] == outcome_key] if not joint_df.empty else pd.DataFrame()
    if not jt.empty:
        pre_p  = jt.loc[jt["period"] == "pre",  "pval"].squeeze()
        post_p = jt.loc[jt["period"] == "post", "pval"].squeeze()
        txt = (f"joint pre-trend: p={pre_p:.3f}\n"
               f"joint post-effect: p={post_p:.3f}")
        ax.text(0.04, 0.94, txt, transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8))
    ax.set_title(outcome_key, fontsize=9, fontweight="bold")
    ax.set_xlabel("years relative to first large dairy CAFO", fontsize=7)
    ax.tick_params(labelsize=7)
for j in range(n_out, len(axes)):
    axes[j].set_visible(False)
fig.suptitle(
    "Event study: first large dairy CAFO entry (Census wave) -> outcomes\n"
    "Rural US counties | county + year FE, state-clustered SEs, 90% CI | omitted: t=-1\n"
    "Replication check against external memo Section 4",
    fontsize=10, y=1.02,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_Y2_dairy_event_study.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)


# =============================================================================
# PART (b): Ridge regression cross-check (pooled vs within-transformed)
# =============================================================================
print("\n" + "="*78)
print("PART (b): Ridge cross-check — does regularized regression agree?")
print("="*78)

# Build the CAFO/FSIS treatment pool (log-per-10k, same transform used
# throughout the repo) alongside the demographic control pool. Dairy also
# gets a raw (unlogged) per-10k version at all 3 size thresholds, so the
# logged vs. raw choice can be compared directly rather than assumed.
def _log_per10k(series, pop):
    x = pd.to_numeric(series, errors="coerce")
    p = pd.to_numeric(pop, errors="coerce").replace(0, np.nan)
    return np.log1p((x / p) * 10_000)

def _raw_per10k(series, pop):
    x = pd.to_numeric(series, errors="coerce")
    p = pd.to_numeric(pop, errors="coerce").replace(0, np.nan)
    return (x / p) * 10_000

CAFO_RAW_COLS = {
    "cafo_dairy_log":         ["cafo_dairy_small", "cafo_dairy_medium", "cafo_dairy_large"],
    "cafo_dairy_medlarge_log":["cafo_dairy_medium", "cafo_dairy_large"],
    "cafo_dairy_large_log":   ["cafo_dairy_large"],
    "cafo_beef_log":     ["cafo_beef_small", "cafo_beef_medium", "cafo_beef_large"],
    "cafo_beef_large_log": ["cafo_beef_large"],
    "cafo_hogs_log":     ["cafo_hogs_small", "cafo_hogs_medium", "cafo_hogs_large"],
    "cafo_hogs_large_log": ["cafo_hogs_large"],
    "cafo_chickens_log": ["cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large"],
    "cafo_chickens_large_log": ["cafo_chickens_large"],
}
# Same 3 dairy thresholds, unlogged (raw count per 10k population).
DAIRY_RAW_UNLOGGED_COLS = {
    "cafo_dairy_raw":          ["cafo_dairy_small", "cafo_dairy_medium", "cafo_dairy_large"],
    "cafo_dairy_medlarge_raw": ["cafo_dairy_medium", "cafo_dairy_large"],
    "cafo_dairy_large_raw":    ["cafo_dairy_large"],
}
FSIS_RAW_COLS = {
    "fsis_total_log":     "n_unique_establishments_fsis",
    "fsis_slaughter_log": "n_slaughterhouse_present_establishments_fsis",
}

df_ridge = df_raw.copy()
for new_col, src in CAFO_RAW_COLS.items():
    cols = [c for c in src if c in df_ridge.columns]
    total = df_ridge[cols].sum(axis=1, min_count=1) if cols else np.nan
    df_ridge[new_col] = _log_per10k(total, df_ridge[POP_COL])
for new_col, src in DAIRY_RAW_UNLOGGED_COLS.items():
    cols = [c for c in src if c in df_ridge.columns]
    total = df_ridge[cols].sum(axis=1, min_count=1) if cols else np.nan
    df_ridge[new_col] = _raw_per10k(total, df_ridge[POP_COL])
for new_col, src_col in FSIS_RAW_COLS.items():
    df_ridge[new_col] = _log_per10k(df_ridge[src_col], df_ridge[POP_COL]) if src_col in df_ridge.columns else np.nan

TREATMENT_PREDICTORS = (list(CAFO_RAW_COLS.keys()) + list(DAIRY_RAW_UNLOGGED_COLS.keys())
                        + list(FSIS_RAW_COLS.keys()))
CONTROL_PREDICTORS   = [c for c in CONTROL_COLS if c in df_ridge.columns]
ALL_PREDICTORS       = TREATMENT_PREDICTORS + CONTROL_PREDICTORS

def run_ridge(X_df, y, alphas=np.logspace(-3, 3, 25)):
    """
    Standardized RidgeCV. Drops any predictor column that is entirely NaN
    over this outcome's sample *before* imputing (SimpleImputer silently
    drops such columns from its output, which would otherwise desync
    ridge.coef_ from the column index).
    """
    all_nan = X_df.columns[X_df.isna().all()].tolist()
    if all_nan:
        print(f"    (dropping {len(all_nan)} all-NaN predictor(s) for this outcome: {all_nan})")
    use_cols = [c for c in X_df.columns if c not in all_nan]
    X_use = X_df[use_cols]

    imp = SimpleImputer(strategy="mean")
    scl = StandardScaler()
    X_imp = scl.fit_transform(imp.fit_transform(X_use))
    y_arr = y.values
    ridge = RidgeCV(alphas=alphas, cv=5)
    ridge.fit(X_imp, y_arr)
    coefs = pd.Series(ridge.coef_, index=use_cols).reindex(X_df.columns)  # NaN for dropped cols
    return coefs, ridge.alpha_, ridge.score(X_imp, y_arr)

ridge_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    if outcome_col not in df_ridge.columns:
        continue
    mask = df_ridge[outcome_col].notna()
    sub = df_ridge.loc[mask].copy()
    if mask.sum() < 200:
        print(f"  [{outcome_key}] too few obs, skip")
        continue

    # Drop predictors with zero coverage in this outcome's sample up front —
    # within_transform's dropna() otherwise zeroes out every row the moment
    # any single column (e.g. FSIS, not merged into this panel build) is 100% NaN.
    valid_predictors = [c for c in ALL_PREDICTORS if sub[c].notna().any()]
    dropped = sorted(set(ALL_PREDICTORS) - set(valid_predictors))
    if dropped:
        print(f"    (no coverage at all for: {dropped} — excluded from both specs)")

    # --- Pooled (raw levels) ---
    X_pooled = sub[valid_predictors]
    coefs_pooled, alpha_pooled, r2_pooled = run_ridge(X_pooled, sub[outcome_col])
    for pred, coef in coefs_pooled.reindex(ALL_PREDICTORS).items():
        ridge_rows.append({"outcome": outcome_key, "predictor": pred, "spec": "Pooled",
                            "std_coef": coef, "alpha": alpha_pooled, "r2": r2_pooled})

    # --- Within-transformed (county + year demeaned) ---
    dm = within_transform(sub, outcome_col, valid_predictors)
    if len(dm) >= 200:
        coefs_within, alpha_within, r2_within = run_ridge(dm[valid_predictors], dm[outcome_col])
        for pred, coef in coefs_within.reindex(ALL_PREDICTORS).items():
            ridge_rows.append({"outcome": outcome_key, "predictor": pred, "spec": "Within",
                                "std_coef": coef, "alpha": alpha_within, "r2": r2_within})
    else:
        print(f"  [{outcome_key}] within-transformed N too small ({len(dm)}), pooled only")

    print(f"  {outcome_key}:")
    _dairy_cols = ["cafo_dairy_large_log", "cafo_dairy_large_raw",
                   "cafo_dairy_medlarge_log", "cafo_dairy_medlarge_raw",
                   "cafo_dairy_log", "cafo_dairy_raw"]
    for _col in _dairy_cols:
        _p = coefs_pooled.get(_col, np.nan)
        _w = coefs_within.get(_col, np.nan) if len(dm) >= 200 else np.nan
        print(f"    {_col:26s} std-coef  pooled={_p:+.4f}  within={_w:+.4f}")

ridge_df = pd.DataFrame(ridge_rows)
ridge_csv = os.path.join(tables_s4_dir, f"{today_str}_Block3_ridge_pooled_vs_within.csv")
ridge_df.to_csv(ridge_csv, index=False)
print("Saved:", ridge_csv)

# Figure Y3: ridge coefficients, dairy vars highlighted, pooled vs within
if not ridge_df.empty:
    n_out = ridge_df["outcome"].nunique()
    fig, axes = plt.subplots(1, n_out, figsize=(5.5*n_out, 8), sharey=False)
    if n_out == 1:
        axes = [axes]
    for ax, outcome_key in zip(axes, OUTCOMES.keys()):
        sub = ridge_df[ridge_df["outcome"] == outcome_key]
        if sub.empty:
            continue
        pooled = sub[sub["spec"] == "Pooled"].set_index("predictor")["std_coef"]
        within = sub[sub["spec"] == "Within"].set_index("predictor")["std_coef"]
        order = pooled.abs().sort_values(ascending=True).index
        y_pos = np.arange(len(order))
        colors = ["#762a83" if "dairy" in p else "#999999" for p in order]
        ax.barh(y_pos - 0.18, pooled.reindex(order), height=0.34, color=colors, alpha=0.6, label="Pooled")
        if not within.empty:
            ax.barh(y_pos + 0.18, within.reindex(order), height=0.34, color=colors, alpha=1.0, label="Within")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(order, fontsize=6)
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(outcome_key, fontsize=9, fontweight="bold")
        ax.set_xlabel("standardized ridge coefficient", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Ridge cross-check: standardized coefficients across CAFO/FSIS + demographic controls\n"
        "Purple bars = dairy CAFO predictors | Pooled (raw) vs Within (county+year demeaned)",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f"{today_str}_Y3_ridge_pooled_vs_within.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)

# =============================================================================
# PART (e): Control specification curve + covariate importance ranking
#
# Instead of asserting one control vector X_i by hand, test how much the
# dairy coefficient depends on which controls are in X_i (specification
# curve), and separately rank covariates -- WITH dairy itself included -- by
# predictive importance (Ridge, Random Forest), so dairy's own weight can be
# compared directly against standard demographic/health controls.
# =============================================================================
print("\n" + "="*78)
print("PART (e): Control specification curve + covariate importance ranking")
print("="*78)

# --- Block 5a: specification curve -------------------------------------------
# log_pop always in X_i; every possible subset of this 9-variable core set
# added on top (2^9 = 512 combinations per outcome). All 9 chosen for
# >=92% coverage in the 2010-2023 window (see coverage table reviewed with
# the team) so the combinations aren't themselves driven by missingness.
CORE_9 = [
    "adult_obesity_per100k", "uninsured_adults_per100k", "unemployment_per100k",
    "children_in_poverty_per100k", "%_female", "%_65_and_older", "%_hispanic",
    "median_household_income", "some_college_per100k",
]
CORE_9 = [c for c in CORE_9 if c in df_raw.columns]

spec_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    if outcome_col not in df_raw.columns:
        continue
    for r in range(0, len(CORE_9) + 1):
        for combo in itertools.combinations(CORE_9, r):
            controls = ["log_pop"] + list(combo)
            res = run_fe_ols(df_raw, "any_large_dairy", outcome_col, controls,
                              cluster_col="state_fips", z=1.96, label=f"spec|{outcome_key}")
            if res is None:
                continue
            spec_rows.append({
                "outcome": outcome_key, "n_controls": len(controls),
                "controls": "+".join(controls), **res,
            })
    n_done = sum(1 for r in spec_rows if r["outcome"] == outcome_key)
    print(f"  {outcome_key:26s}: {n_done} / {2**len(CORE_9)} specs estimated")

spec_df = pd.DataFrame(spec_rows)
spec_csv = os.path.join(tables_s4_dir, f"{today_str}_Block4_spec_curve.csv")
spec_df.to_csv(spec_csv, index=False)
print("Saved:", spec_csv)

print(f"\n{'outcome':28s} {'beta min':>9s} {'median':>9s} {'beta max':>9s} {'share sig (p<.05)':>18s}")
for outcome_key in OUTCOMES:
    sub = spec_df[spec_df["outcome"] == outcome_key]
    if sub.empty:
        continue
    print(f"{outcome_key:28s} {sub['beta'].min():9.3f} {sub['beta'].median():9.3f} "
          f"{sub['beta'].max():9.3f} {(sub['pval'] < 0.05).mean():18.2f}")

# Figure Y4: ordered dot + 95% CI plot per outcome -- shows the RANGE of beta
# across all 512 control combinations AND, using each spec's own SE, which
# specs are significant vs not. More informative than a plain box/whisker
# since the box/whisker would show only the point-estimate distribution and
# drop the uncertainty (SE) on each individual spec.
n_out = len(OUTCOMES)
n_cols_sc = 3
n_rows_sc = int(np.ceil(n_out / n_cols_sc))
fig, axes = plt.subplots(n_rows_sc, n_cols_sc, figsize=(n_cols_sc*5.5, n_rows_sc*4.2))
axes = axes.flatten()
for i, outcome_key in enumerate(OUTCOMES.keys()):
    ax = axes[i]
    sub = spec_df[spec_df["outcome"] == outcome_key].sort_values("beta").reset_index(drop=True)
    if sub.empty:
        ax.set_visible(False)
        continue
    x = np.arange(len(sub))
    sig = sub["pval"] < 0.05
    colors = np.where(sig, "#1b7837", "#b2b2b2")
    ax.errorbar(x[~sig], sub.loc[~sig, "beta"], yerr=1.96*sub.loc[~sig, "se"],
                fmt="o", ms=1.5, color="#b2b2b2", elinewidth=0.4, alpha=0.6, label="not sig. (p>=.05)")
    ax.errorbar(x[sig], sub.loc[sig, "beta"], yerr=1.96*sub.loc[sig, "se"],
                fmt="o", ms=1.5, color="#1b7837", elinewidth=0.4, alpha=0.6, label="sig. (p<.05)")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_title(outcome_key, fontsize=9, fontweight="bold")
    ax.set_xlabel("specifications, sorted by beta", fontsize=7)
    ax.set_ylabel("dairy beta, 95% CI", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best", markerscale=3)
fig.suptitle(
    f"Specification curve: dairy coefficient across all {2**len(CORE_9)} control combinations\n"
    "log_pop always included; every subset of 9 core controls (adult obesity, uninsured, "
    "unemployment, poverty, %female, %65+, %hispanic, income, some college) added on top",
    fontsize=10, y=1.02,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_Y4_spec_curve.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)

# --- Block 5b: Ridge + Random Forest covariate ranking, dairy included ------
# Same wide ~20-variable candidate set as before, but "any_large_dairy" is
# now IN the candidate set too (not run separately) -- so its own importance
# can be read directly off the same ranking as the standard controls.
WIDE_20 = [
    "log_pop", "any_large_dairy", "adult_obesity_per100k", "uninsured_adults_per100k",
    "unemployment_per100k", "children_in_poverty_per100k", "access_to_healthy_foods_per100k",
    "premature_death", "preventable_hospital_stays", "poor_physical_health_days",
    "low_birthweight_per100k", "teen_births_per100k", "%_female", "%_65_and_older",
    "%_hispanic", "some_college_per100k", "diabetes_prevalence_per100k",
    "median_household_income", "children_in_single-parent_households_per100k",
    "adult_smoking_per100k", "primary_care_physicians_per100k", "high_school_graduation",
]
WIDE_20 = [c for c in WIDE_20 if c in df_raw.columns]

rank_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    if outcome_col not in df_raw.columns:
        continue
    mask = df_raw[outcome_col].notna()
    sub = df_raw.loc[mask]
    if mask.sum() < 200:
        print(f"  [{outcome_key}] too few obs, skip")
        continue

    X_wide = sub[WIDE_20]
    y_wide = sub[outcome_col]
    imp = SimpleImputer(strategy="mean")
    scl = StandardScaler()
    X_imp = scl.fit_transform(imp.fit_transform(X_wide))

    ridge = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5).fit(X_imp, y_wide)
    ridge_coefs = pd.Series(np.abs(ridge.coef_), index=WIDE_20)

    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(imp.transform(X_wide), y_wide)
    rf_importance = pd.Series(rf.feature_importances_, index=WIDE_20)

    dairy_rf_rank = int((rf_importance.rank(ascending=False))["any_large_dairy"])
    print(f"  {outcome_key:26s}: dairy RF importance={rf_importance['any_large_dairy']:.4f} "
          f"(rank {dairy_rf_rank}/{len(WIDE_20)})")

    for var in WIDE_20:
        rank_rows.append({
            "outcome": outcome_key, "variable": var,
            "ridge_abs_std_coef": ridge_coefs[var], "rf_importance": rf_importance[var],
            "is_dairy": var == "any_large_dairy",
        })

rank_df = pd.DataFrame(rank_rows)
rank_csv = os.path.join(tables_s4_dir, f"{today_str}_Block5_covariate_ranking.csv")
rank_df.to_csv(rank_csv, index=False)
print("Saved:", rank_csv)

# Figure Y5: ranking bar chart per outcome, dairy highlighted
fig, axes = plt.subplots(n_rows_sc, n_cols_sc, figsize=(n_cols_sc*5.5, n_rows_sc*4.5))
axes = axes.flatten()
for i, outcome_key in enumerate(OUTCOMES.keys()):
    ax = axes[i]
    sub = rank_df[rank_df["outcome"] == outcome_key].sort_values("rf_importance", ascending=True)
    if sub.empty:
        ax.set_visible(False)
        continue
    colors = np.where(sub["is_dairy"], "#762a83", "#4393c3")
    ax.barh(sub["variable"], sub["rf_importance"], color=colors)
    ax.set_title(outcome_key, fontsize=9, fontweight="bold")
    ax.set_xlabel("Random Forest importance", fontsize=7)
    ax.tick_params(labelsize=6)
fig.suptitle(
    "Covariate importance ranking (Random Forest) -- dairy CAFO presence included\n"
    "Purple bar = any_large_dairy | Blue bars = standard demographic/health controls",
    fontsize=10, y=1.02,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_Y5_covariate_ranking.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)

# =============================================================================
# PART (f): 2010-2020 window -- interactions, ML model selection (Random
# Forest + Double Lasso), then re-run the causal (TWFE) estimate.
# Reviewed against Econ 224 (Leung) lecture notes: LEC-2 section 3
# (Double/Debiased ML: cross-fitted Lasso nuisance models + Neyman-orthogonal
# partially-linear estimator -- the "double lasso" here) and LEC-3 section 2
# (Random Forests: bagged, decorrelated trees; m=sqrt(d) covariates per
# split; explicitly a PREDICTION tool, not a causal one -- "random forests
# are best suited for pure prediction and often considered black boxes,"
# while lasso/CART are the ones suited to understanding X-Y relationships).
#
# MODEL SPECIFICATION DICTIONARY -- one sentence each, for team sharing:
#
#   TWFE (within, county+year FE):
#     Estimates the association between dairy CAFO exposure and the outcome
#     using only within-county changes over time, after removing every
#     time-invariant county characteristic and every national year-to-year
#     shock.
#
#   Random Forest (variable importance):
#     A black-box, non-linear predictive model that ranks which covariates
#     -- including dairy CAFO exposure -- best predict the outcome, without
#     producing an interpretable causal coefficient; used here for model
#     selection, not causal inference.
#
#   Double Lasso / Double ML (partially linear, cross-fitted):
#     Uses two separate cross-fitted Lasso regressions to remove the
#     predictable part of both the outcome and the dairy-CAFO variable using
#     all covariates, then estimates the association from what is left
#     over -- a machine-learning way of controlling for many covariates
#     without hand-picking which ones belong in X_i.
#
#   Per-capita CAFO spec:
#     Measures dairy CAFO exposure as operations per 10,000 residents, so a
#     bigger county with proportionally more CAFOs counts as the same
#     "exposure" as a smaller county with fewer.
#
#   Raw/net count CAFO spec:
#     Measures dairy CAFO exposure as the raw number of operations in the
#     county, with population entered as its own separate control instead
#     of being divided out of the treatment variable.
# =============================================================================
print("\n" + "="*78)
print("PART (f): 2010-2020 window -- interactions, RF + Double Lasso, re-run TWFE")
print("="*78)

df_ml = df_raw[df_raw["year"].between(2010, 2020)].copy()
print(f"2010-2020 window: {len(df_ml):,} rows | {df_ml['fips'].nunique():,} counties")

# --- Interaction terms -------------------------------------------------------
# dairy x income inequality: memo flagged inequality as a candidate mechanism,
#   not just a confounder -- worth seeing if the dairy association is
#   concentrated where inequality is high.
# dairy x %hispanic: tests the labor-composition channel directly.
# any_large_dairy_x_fsis: already built (Block 1e) -- reused here.
df_ml["dairy_x_incineq"]  = df_ml["any_large_dairy"] * df_ml["income_inequality"]
df_ml["dairy_x_hispanic"] = df_ml["any_large_dairy"] * df_ml["%_hispanic"]

INTERACTION_TERMS = ["dairy_x_incineq", "dairy_x_hispanic", "any_large_dairy_x_fsis"]
ML_CONTROLS = [c for c in WIDE_20 if c not in ("any_large_dairy",)]
DAIRY_SPECS = {
    "Per-capita rate": "cafo_dairy_large_raw",
    "Raw count":       "cafo_dairy_large",
}

# --- Block 6a: Random Forest variable importance, both dairy specs ---------
# Tuned per LEC-3 convention: num.trees~500, m=sqrt(d) features per split
# (max_features="sqrt"), small min leaf size, deep/overgrown trees (bagging
# handles the variance, so no max_depth restriction) -- rather than the
# arbitrary max_depth=6 used in Part (e).
print("\nRandom Forest variable importance (2010-2020 window)...")
rf_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    if outcome_col not in df_ml.columns:
        continue
    mask = df_ml[outcome_col].notna()
    sub = df_ml.loc[mask]
    if mask.sum() < 200:
        continue
    for spec_label, spec_col in DAIRY_SPECS.items():
        preds = [spec_col] + ML_CONTROLS + INTERACTION_TERMS
        preds = [p for p in preds if p in sub.columns]
        X = sub[preds]
        y = sub[outcome_col]
        imp = SimpleImputer(strategy="mean")
        rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf.fit(imp.fit_transform(X), y)
        importance = pd.Series(rf.feature_importances_, index=preds)
        rank = int(importance.rank(ascending=False)[spec_col])
        print(f"  {outcome_key:26s} | {spec_label:16s} importance={importance[spec_col]:.4f} "
              f"(rank {rank}/{len(preds)})")
        for var in preds:
            rf_rows.append({"outcome": outcome_key, "dairy_spec": spec_label,
                             "variable": var, "rf_importance": importance[var],
                             "is_dairy": var == spec_col})

rf_df = pd.DataFrame(rf_rows)
rf_csv = os.path.join(tables_s4_dir, f"{today_str}_Block6a_randomforest_2010_2020.csv")
rf_df.to_csv(rf_csv, index=False)
print("Saved:", rf_csv)

# --- Block 6b: Double Lasso / Double ML (partially linear), both dairy specs
# Cross-fitted LassoCV for both nuisance functions E[Y|X] and E[D|X]
# (n_folds=5), Neyman-orthogonal partially-linear estimator -- matches LEC-2
# section 3.3-3.4 exactly (DoubleMLPLR in the `doubleml` package, the same
# library/method demonstrated in the course notes with lasso and random
# forest learners).
print("\nDouble Lasso / Double ML (2010-2020 window, partially linear)...")
dml_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    if outcome_col not in df_ml.columns:
        continue
    for spec_label, spec_col in DAIRY_SPECS.items():
        cols_needed = [outcome_col, spec_col] + ML_CONTROLS
        sub = df_ml[cols_needed].dropna()
        if len(sub) < 500:
            print(f"  {outcome_key:26s} | {spec_label:16s} too few obs ({len(sub)}), skip")
            continue
        try:
            dml_data = DoubleMLData(sub, y_col=outcome_col, d_cols=spec_col, x_cols=ML_CONTROLS)
            ml_l = LassoCV(cv=5, max_iter=5000)
            ml_m = LassoCV(cv=5, max_iter=5000)
            model = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=5)
            model.fit()
            s = model.summary.loc[spec_col]
            beta, se, pval = s["coef"], s["std err"], s["P>|t|"]
            dml_rows.append({"outcome": outcome_key, "dairy_spec": spec_label,
                              "beta": beta, "se": se, "pval": pval, "N": len(sub)})
            sig = "*" if pval < 0.05 else " "
            print(f"  {outcome_key:26s} | {spec_label:16s} beta={beta:+.5f}  "
                  f"se={se:.5f}  p={pval:.3f}{sig}  N={len(sub):,}")
        except Exception as e:
            print(f"  {outcome_key:26s} | {spec_label:16s} FAILED: {repr(e)[:150]}")

dml_df = pd.DataFrame(dml_rows)
dml_csv = os.path.join(tables_s4_dir, f"{today_str}_Block6b_doublelasso_2010_2020.csv")
dml_df.to_csv(dml_csv, index=False)
print("Saved:", dml_csv)

# --- Block 6c: re-run the causal (TWFE) estimate, same window, both specs --
# Same within (county+year FE) design as Part (a), restricted to 2010-2020
# and using the ML_CONTROLS set the RF/Double Lasso step just ran over --
# directly comparable to Block 6b's Double Lasso numbers.
print("\nTWFE re-run (2010-2020 window), both dairy specs + interactions...")
twfe_rows = []
for outcome_key, outcome_col in OUTCOMES.items():
    for spec_label, spec_col in DAIRY_SPECS.items():
        res = run_fe_ols(df_ml, spec_col, outcome_col, ML_CONTROLS,
                          cluster_col="state_fips", z=1.96, label=f"{spec_label}|{outcome_key}")
        if res is None:
            continue
        twfe_rows.append({"outcome": outcome_key, "term": spec_label, **res})
        sig = "*" if res["pval"] < 0.05 else " "
        print(f"  {outcome_key:26s} | {spec_label:16s} beta={res['beta']:+.5f}  "
              f"p={res['pval']:.3f}{sig}  N={res['N']:,}")
    # interaction terms, dairy presence + each interaction jointly
    for inter_col in INTERACTION_TERMS:
        res_dict = run_fe_ols_multi(df_ml, ["any_large_dairy", inter_col], outcome_col, ML_CONTROLS,
                                     cluster_col="state_fips", z=1.96, label=f"{inter_col}|{outcome_key}")
        res = res_dict.get(inter_col)
        if res is None:
            continue
        twfe_rows.append({"outcome": outcome_key, "term": inter_col, **res})
        sig = "*" if res["pval"] < 0.05 else " "
        print(f"  {outcome_key:26s} | {inter_col:24s} beta={res['beta']:+.5f}  "
              f"p={res['pval']:.3f}{sig}  N={res['N']:,}")

twfe_df = pd.DataFrame(twfe_rows)
twfe_csv = os.path.join(tables_s4_dir, f"{today_str}_Block6c_twfe_2010_2020.csv")
twfe_df.to_csv(twfe_csv, index=False)
print("Saved:", twfe_csv)

print(f"\nAll script4 outputs saved to:\n  figs:   {out_dir}\n  tables: {tables_s4_dir}")
