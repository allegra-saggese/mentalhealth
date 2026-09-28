#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4a-design-estimates.py  --  PART A: the design-based estimates.

REBUILT 2026-09-28 on the corrected panel. The previous file
(z-archive/script4a-twfe-eventstudy_PRE-REBUILD_2026-09-18.py) ran on the
pre-CHR-fix panel and used the old A-numbering. Two things changed:

  1. PANEL. Ten control variables were re-sourced to their correct data years
     (SAIPE / BLS LAUS / SAHIE / Census PEP). See 2026-09-21_CHR-YEAR-MISALIGNMENT.md.
  2. NAMING. A = built and runnable, B = specified only. See 2026-09-27_MODEL-NAMING.md.

MODELS IN THIS FILE
-------------------
    A1   Pooled OLS                 state + year FE          benchmark
    A2   TWFE within-county         county + year FE         HEADLINE
    A4   Dairy x FSIS interaction   2017-2023, thin sample   exploratory
    A5   Event study                leads/lags around entry
    A6   Callaway-Sant'Anna         staggered DiD, ATT(g,t)

    A3 (other-animal horse race) lives in script4l-a3-horserace.py -- it is a
    448-regression grid and does not belong inside this file. Run 2026-09-28.

WHAT CHANGED SUBSTANTIVELY FROM THE 09-18 VERSION
--------------------------------------------------
* TREATMENTS. A1/A2 now run over CORE_TREATMENTS (T1-T4) -- the decided
  treatment set -- not the legacy 12-definition TREATMENTS dict. T5 was dropped
  and T6 retired (D-17), so T1-T4 is the complete set with no unrun members.
  The legacy grid is still available via --legacy-grid for continuity checks.
* OUTCOMES. 7, not 6 (the four crime outcomes were settled 2026-09-21, D-11).
* CONTROLS. ONE control set: the D-16 PRE-COMMITTED STEP-8 SET
  (COVARIATE_ORDER[:8]). All eight are annually measured at 99.7-99.9% coverage,
  the estimating sample stays intact (99.7% of outcome rows), and every one can
  be tested for pre-trends.
  CONTROL_PRETREAT (18) was REMOVED as a robustness arm on 2026-09-29 (D-29).
  Its ten extra members are each excluded for a stated substantive reason --
  two colliders (adult_smoking, teen_births), a spliced measure break
  (access_to_healthy_foods), an unusable series (driving_alone_to_work), a
  variable with no usable variation in non-metro counties
  (%_native_hawaiian/other_pacific_islander), and five that cannot be tested for
  pre-trends because they are rolling or model-smoothed. Running that set as a
  "robustness check" would imply the exclusions were arbitrary. They are not.
  THE STEP-8 SET IS PRE-COMMITTED. It was fixed in D-16 before the stability
  table in script4c was read. Choosing it afterwards would have been
  post-selection inference and would invalidate every p-value below.

STANDARD ERRORS
---------------
Three variance estimators side by side on every estimate:
    se_state   CRV1 on state_fips   <- HEADLINE. Treatment assignment is
                                       spatially correlated within state (dairy
                                       regions), and it is the most conservative
                                       of the three here.
    se_county  CRV1 on fips
    se_hetero  heteroskedasticity-robust
Reporting all three is the answer to the SE question raised in review; the
headline choice is not an understatement of uncertainty.

ESTIMATION uses pyfixest.feols, never the legacy within_transform() helper --
that applied a SINGLE demeaning pass, exact only for balanced panels. This panel
is unbalanced (77.3% have the full span) and one-shot demeaning gave
beta = +0.08466 where correct alternating projections give +0.07970: a 6.2%
error in the POINT ESTIMATE, not just the SE.

IDENTIFYING VARIATION is reported inline (n_switchers, n_transitions). Under
county FE only counties whose treatment CHANGES contribute to beta, so a row
with 30 switchers is not the same evidence as one with 300 even at equal N.

Outputs -> Data/output/tables/script4a/  and  figs/script4a/
"""
import warnings; warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
from scipy.stats import norm as _norm

from script4_treatment import (
    load_panel, OUTCOMES, TREATMENTS, CORE_TREATMENTS, CORE_TREATMENT_LABELS,
    SPEC_EXTRA_REGRESSORS, COVARIATE_ORDER,
    db_data, figs_dir, tables_dir, os, date,
)
from functions import latest_file_glob

OUT_FIGS   = os.path.join(figs_dir,   "script4a")
OUT_TABLES = os.path.join(tables_dir, "script4a")
for _d in (OUT_FIGS, OUT_TABLES):
    os.makedirs(_d, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")

# D-16 / D-29: the step-8 set is the ONLY control set run. CONTROL_PRETREAT was
# dropped as a robustness arm on 2026-09-29 -- not on coverage grounds (on the
# headline sample it costs only 15% of rows) but because each of its ten extra
# members is excluded FOR CAUSE: two colliders, one measure break, one unusable
# series, one with no variation, and five that cannot be tested for pre-trends.
# Reporting a set that contains known colliders as a "robustness check" would
# imply the exclusions were arbitrary. They are not. See D-29.
CONTROLS_HEADLINE = COVARIATE_ORDER[:8]
CONTROL_SETS = {"step8 (pre-committed, D-16)": CONTROLS_HEADLINE}

USE_LEGACY_GRID = "--legacy-grid" in sys.argv
GRID_TREATMENTS = (TREATMENTS if USE_LEGACY_GRID
                   else {k: (v, SPEC_EXTRA_REGRESSORS.get(k, []), CORE_TREATMENT_LABELS[k])
                         for k, v in CORE_TREATMENTS.items()})
# E2 (tr_e2_add_absorb) is NOT a core treatment, but the A2-vs-A6 comparison
# below requires it: it is the binary absorbing treatment that CS's cohort
# definition implies. Added to the grid so the two estimators can be compared on
# the SAME treatment. It is reported only in that comparison table.
if not USE_LEGACY_GRID:
    GRID_TREATMENTS["E2"] = ("tr_e2_add_absorb", [],
                             "First positive change in large dairy ops (absorbing)")

N_BOOT = 300                              # cluster-bootstrap reps for A6
RNG    = np.random.default_rng(20260928)


# =============================================================================
# Helpers
# =============================================================================
def _clean(name):
    # "/" must be stripped: %_native_hawaiian/other_pacific_islander otherwise
    # splits into two tokens and pyfixest's formula parser fails.
    return (name.replace("%", "pct").replace("-", "_").replace(" ", "_")
                .replace("(", "").replace(")", "").replace("/", "_")
                .replace("+", "p").replace(",", "_"))


def prep(df, cols):
    """Subset to `cols`, drop incomplete rows, return (frame, {orig: clean})."""
    cols = list(dict.fromkeys(cols))
    sub = df[cols].dropna().copy()
    return sub.rename(columns={c: _clean(c) for c in cols}), {c: _clean(c) for c in cols}


def variation(sub, tcol):
    """Counties whose treatment changes, and how many changes. This is what
    actually identifies a within-county coefficient."""
    n_units = sub["fips"].nunique()
    nun = sub.groupby("fips")[tcol].nunique()
    sw = nun[nun > 1].index
    if len(sw) == 0:
        return dict(n_counties=n_units, n_states=sub["state_fips"].nunique(),
                    n_switchers=0, n_transitions=0)
    e = sub[sub["fips"].isin(sw)].sort_values(["fips", "year"]).copy()
    e["_d"] = e.groupby("fips")[tcol].diff()
    return dict(n_counties=n_units, n_states=sub["state_fips"].nunique(),
                n_switchers=int(len(sw)), n_transitions=int((e["_d"].abs() > 0).sum()))


def rhs_vif(sub, rhs_cols, within=True):
    """VIF over the FULL right-hand side of THIS spec, not just the control block.

    Computed on WITHIN-TRANSFORMED data when `within`, because that is the
    variation a county+year FE regression actually uses. Raw-level VIF is the
    wrong diagnostic: collinearity among these controls is almost entirely
    cross-sectional and county FE removes it (children_in_poverty 9.10 raw ->
    1.13 within).

    n_zero_var counts regressors with NO within-county variation -- absorbed by
    the FE, contributing nothing. This is how the old C4 spec was caught
    conditioning on a county-constant variable.
    """
    W = sub[rhs_cols].astype(float).copy()
    if within:
        for c in rhs_cols:
            W[c] = (W[c] - sub.groupby("fips")[c].transform("mean")
                        - sub.groupby("year")[c].transform("mean") + sub[c].mean())
    zero_var = [c for c in rhs_cols if W[c].std() <= 1e-10]
    keep = [c for c in rhs_cols if c not in zero_var]
    if len(keep) < 2:
        return {"vif_max": np.nan, "vif_max_var": "", "vif_treat": np.nan,
                "n_zero_var": len(zero_var)}
    try:
        R = np.corrcoef(W[keep].values, rowvar=False)
        v = pd.Series(np.diag(np.linalg.pinv(R)), index=keep)
    except Exception:
        return {"vif_max": np.nan, "vif_max_var": "", "vif_treat": np.nan,
                "n_zero_var": len(zero_var)}
    return {"vif_max": float(v.max()), "vif_max_var": str(v.idxmax()),
            "vif_treat": float(v.get(rhs_cols[0], np.nan)), "n_zero_var": len(zero_var)}


def fit(sub, y, tcol, extra, controls_clean, fe, label):
    """One regression, three variance estimators. Returns a flat dict or None."""
    fml = f"{y} ~ " + " + ".join([tcol] + list(extra) + controls_clean) + f" | {fe}"
    if len(sub) < 200:
        return None
    out = {}
    try:
        for key, vc in [("state", {"CRV1": "state_fips"}),
                        ("county", {"CRV1": "fips"}),
                        ("hetero", "hetero")]:
            m = pf.feols(fml, data=sub, vcov=vc)
            t = m.tidy()
            if tcol not in t.index:
                return None
            r = t.loc[tcol]
            out[f"se_{key}"] = float(r["Std. Error"])
            out[f"p_{key}"]  = float(r["Pr(>|t|)"])
            if key == "state":
                out["beta"] = float(r["Estimate"])
                out["N"] = int(m._N)
                out["r2_within"] = float(getattr(m, "_r2_within", np.nan))
                for x in extra:
                    if x in t.index:
                        out[f"cond_{x}_beta"] = float(t.loc[x, "Estimate"])
                        out[f"cond_{x}_p"]    = float(t.loc[x, "Pr(>|t|)"])
    except Exception as e:
        print(f"      [{label}] failed: {type(e).__name__}: {e}")
        return None
    out["ci_lo"] = out["beta"] - 1.96 * out["se_state"]
    out["ci_hi"] = out["beta"] + 1.96 * out["se_state"]
    return out


# =============================================================================
print("=" * 78)
print("script4a -- PART A design-based estimates  (A1, A2, A4, A5, A6)")
print("=" * 78)

df = load_panel()
print(f"panel: {os.path.basename(df.attrs['panel_path'])}  {df.shape[0]:,} rows, "
      f"{df.fips.nunique():,} counties, {int(df.year.min())}-{int(df.year.max())}")
print(f"treatment grid: {'LEGACY 12' if USE_LEGACY_GRID else 'CORE T1-T4'}  "
      f"({len(GRID_TREATMENTS)} definitions) x {len(OUTCOMES)} outcomes")
print(f"controls: step-8 pre-committed ({len(CONTROLS_HEADLINE)}), D-16 -- single set, D-29")
print(f"cohorts: {df[df.cohort.notna()].groupby('cohort').fips.nunique().to_dict()}")

VIF_AUDIT = []


# =============================================================================
# A1 + A2 : the treatment grid, both FE structures, both control sets
# =============================================================================
print("\n" + "=" * 78)
print("A1 (pooled state+year) / A2 (within county+year)")
print("=" * 78)

FE_SPECS = {"A1": ("state_fips + year", False), "A2": ("fips + year", True)}

grid_rows = []
for tid, spec in GRID_TREATMENTS.items():
    tcol, extra, tlabel = spec
    for okey, ocol in OUTCOMES.items():
        for cs_label, controls in CONTROL_SETS.items():
            need = ["fips", "year", "state_fips", ocol, tcol] + list(extra) + controls
            sub, mp = prep(df, need)
            if len(sub) < 200:
                continue
            cc = [mp[c] for c in controls]
            var = variation(sub, mp[tcol])
            for reg, (fe, within) in FE_SPECS.items():
                r = fit(sub, mp[ocol], mp[tcol], [mp[x] for x in extra], cc, fe,
                        f"{reg}|{tid}|{okey}|{cs_label}")
                if not r:
                    continue
                vif = rhs_vif(sub, [mp[tcol]] + [mp[x] for x in extra] + cc, within=within)
                grid_rows.append({"registry": reg, "treatment_id": tid, "treatment": tlabel,
                                  "outcome": okey, "control_set": cs_label,
                                  **r, **var, **vif})
                VIF_AUDIT.append({"model": reg, "treatment_id": tid, "outcome": okey,
                                  "control_set": cs_label, "within": within, **vif})
    print(f"  {tid} done")

grid = pd.DataFrame(grid_rows)
gpath = os.path.join(OUT_TABLES, f"{TODAY}_A1_A2_treatment_grid.csv")
grid.to_csv(gpath, index=False)
print(f"\nSaved: {gpath}   ({len(grid):,} rows)")

hl = grid[(grid.registry == "A2") & (grid.control_set.str.startswith("step8"))
          & (grid.outcome == "Poor MH Days")]
if len(hl):
    print("\n--- A2 within, step-8 controls, Poor MH Days ---")
    print(hl[["treatment_id", "beta", "se_state", "p_state", "N",
              "n_switchers", "vif_max"]].to_string(index=False))


# =============================================================================
# A4 : dairy x FSIS interaction   (was A6)
# =============================================================================
# Is the dairy association concentrated where a slaughterhouse is also present?
# ONE regression per treatment x outcome, containing the dairy term, the FSIS
# term and their interaction -- not three separate regressions, which would not
# identify the interaction. This is a TWFE interaction model, NOT an event study
# (that is A5); it has no leads or lags.
#
# RUNS ACROSS ALL FOUR TREATMENTS (fixed 2026-09-28). The previous version
# hard-coded tr_e1_lg_bin (T1) alone, which made A4 the only model in Part A not
# estimated over the decided treatment set. There is no reason for the
# restriction: T2/T3/T4 x FSIS-presence are all well defined (count x presence,
# per-capita x presence, log-count x presence).
#
# SAMPLE: 2017-2023 only, because that is the FSIS coverage window.
#
# THE IDENTIFICATION PROBLEM THIS WINDOW CREATES (D-22). Treatment is measured at
# ag-census waves and forward-filled, so it moves ONLY in 2007, 2012, 2017 and
# 2022 (163 / 123 / 101 / 89 counties respectively). Restricting to 2017-2023
# leaves exactly ONE wave transition inside the window: 2022.
#   - mental health outcomes run to 2023  -> 2022 is inside  -> 89 switchers
#   - crime outcomes end in 2021          -> NO transition   -> 0 switchers
#   - Deaths of Despair ends in 2020      -> NO transition   -> 0 switchers
# With zero switchers, county FE absorbs the dairy term completely and the
# interaction is identified only off counties whose FSIS status changes -- 19 at
# best. Those coefficients are NOT INTERPRETABLE and are written out with
# identified=False rather than reported as nulls: a null claims power this design
# does not have. Only the two mental-health outcomes are reportable.
print("\n" + "=" * 78)
print("A4: dairy x FSIS interaction (2017-2023 only, exploratory)")
print("=" * 78)

fsis_rows = []
try:
    _fp = latest_file_glob(os.path.join(db_data, "merged"), "*_panel_fsis.csv")
    _f = pd.read_csv(_fp, low_memory=False)[["fips", "year", "n_unique_establishments_fsis"]]
    _f["fips"] = _f["fips"].astype(str).str.zfill(5)
    dff = df.merge(_f, on=["fips", "year"], how="left")
    cov = dff["year"].between(2017, 2023)
    # Outside 2017-2023 the FSIS panel simply does not cover: NaN, never 0.
    dff["any_fsis"] = np.where(cov, (dff["n_unique_establishments_fsis"].fillna(0) > 0)
                               .astype(float), np.nan)
    for tid, tcol in CORE_TREATMENTS.items():
        extra_spec = SPEC_EXTRA_REGRESSORS.get(tid, [])
        ix = f"{tcol}_x_fsis"
        dff[ix] = dff[tcol] * dff["any_fsis"]
        for okey, ocol in OUTCOMES.items():
            need = (["fips", "year", "state_fips", ocol, tcol, "any_fsis", ix,
                     "cafo_dairy_large"] + list(extra_spec) + CONTROLS_HEADLINE)
            sub, mp = prep(dff, need)
            if len(sub) < 200:
                continue
            cc  = [mp[c] for c in CONTROLS_HEADLINE]
            var = variation(sub, mp[tcol])
            # IDENTIFICATION TEST -- on the RAW CAFO COUNT, not on the treatment
            # variable. T3 is count/population, so its within-county variance is
            # non-zero whenever POPULATION moves, even when no CAFO ever opens or
            # closes. In 2017-2021 the raw count changes in 0 counties but T3
            # "switches" in 603 -- that variation is denominator drift, not
            # exposure, and a coefficient identified off it is not a dairy effect.
            # Testing the numerator catches this for every functional form at once.
            raw_sw = int((sub.groupby("fips")[mp["cafo_dairy_large"]].nunique() > 1).sum())
            identified = raw_sw > 0
            r = fit(sub, mp[ocol], mp[ix],
                    [mp[tcol], mp["any_fsis"]] + [mp[x] for x in extra_spec],
                    cc, "fips + year", f"A4|{tid}|{okey}")
            if r:
                vif = rhs_vif(sub, [mp[ix], mp[tcol], mp["any_fsis"]]
                              + [mp[x] for x in extra_spec] + cc)
                fsis_rows.append({"registry": "A4", "treatment_id": tid,
                                  "treatment": CORE_TREATMENT_LABELS[tid],
                                  "outcome": okey, "term": f"{tid} x FSIS",
                                  "identified": identified,
                                  "n_raw_count_switchers": raw_sw,
                                  "not_identified_reason": ("" if identified else
                                      "raw large-dairy count never changes inside this "
                                      "outcome's year range -- no ag-census wave "
                                      "transition. Any apparent variation in T3 is "
                                      "population-denominator drift, not exposure."),
                                  **r, **var, **vif})
                VIF_AUDIT.append({"model": "A4", "treatment_id": tid, "outcome": okey,
                                  "control_set": "step8", "within": True, **vif})
    print(f"  FSIS merged from {os.path.basename(_fp)}")
except Exception as e:
    print(f"  FSIS panel unavailable ({type(e).__name__}: {e}) -- A4 skipped")

fs = pd.DataFrame(fsis_rows)
if len(fs):
    fs.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_A4_fsis_interaction.csv"), index=False)
    ok = fs[fs.identified]
    print(f"\n  REPORTABLE ({len(ok)} of {len(fs)} specs -- the rest have 0 switchers):")
    if len(ok):
        print(ok[["treatment_id", "outcome", "beta", "se_state", "p_state", "N",
                  "n_switchers", "n_raw_count_switchers"]].to_string(index=False))
    bad = sorted(fs.loc[~fs.identified, "outcome"].unique())
    print(f"\n  NOT IDENTIFIED, suppressed: {', '.join(bad)}")


# =============================================================================
# A5 : TWFE event study around the first positive change   (was A7)
# =============================================================================
# Leads/lags relative to the county's first positive change in large-dairy ops,
# omitted category t_rel = -1. Control group = never-treated pooled with
# not-yet-treated (the standard TWFE event study). 95% CI throughout.
#
# READ THE PRE-TREND COEFFICIENTS BEFORE THE POST ONES. Treatment timing carries
# up to 5 years of measurement error (ag census every 5 years, forward-filled),
# so leads close to zero are partly contaminated by already-treated periods.
#
# This estimator inherits the forbidden-comparison problem that A6 exists to fix.
# Read the two together, never A5 alone.
print("\n" + "=" * 78)
print("A5: TWFE event study around first positive change")
print("=" * 78)

LEADS, LAGS = 8, 8
CLEAN_HEAD = [_clean(c) for c in CONTROLS_HEADLINE]
es_rows = []
for okey, ocol in OUTCOMES.items():
    need = ["fips", "year", "state_fips", ocol, "t_rel"] + CONTROLS_HEADLINE
    sub = df[need + ["cohort"]].copy()
    sub["ev"] = sub["t_rel"].clip(-LEADS, LAGS)
    # never-treated (t_rel NaN) are pooled into the omitted category so pyfixest
    # KEEPS them as the comparison group instead of dropping them.
    sub.loc[sub["t_rel"].isna(), "ev"] = -1
    sub = sub.drop(columns=["t_rel", "cohort"]).dropna()
    sub = sub.rename(columns={c: _clean(c) for c in sub.columns})
    if len(sub) < 500:
        continue
    # VIF for A5 is computed on the COVARIATE BLOCK ONLY. The event-time
    # indicators are a mutually exclusive partition of the sample, so they are
    # collinear with each other BY CONSTRUCTION and their VIF is not a
    # diagnostic -- it would flag a design feature as a problem. What can
    # meaningfully be checked is whether the controls are collinear with each
    # other in the event-study sample, which differs from the A2 sample.
    vif5 = rhs_vif(sub, CLEAN_HEAD, within=True)
    VIF_AUDIT.append({"model": "A5", "treatment_id": "E2", "outcome": okey,
                      "control_set": "step8", "within": True,
                      "note": "covariate block only; event-time dummies excluded "
                              "(collinear by construction)", **vif5})

    fml = f"{_clean(ocol)} ~ i(ev, ref=-1) + " + " + ".join(CLEAN_HEAD) + " | fips + year"
    try:
        m = pf.feols(fml, data=sub, vcov={"CRV1": "state_fips"})
        t = m.tidy()
    except Exception as e:
        print(f"  [{okey}] event study failed: {type(e).__name__}: {e}")
        continue
    for idx, r in t.iterrows():
        if not str(idx).startswith("ev::"):
            continue
        try:
            e = int(float(str(idx).split("::")[1]))
        except (ValueError, IndexError):
            continue
        es_rows.append({"registry": "A5", "outcome": okey, "event_time": e,
                        "beta": float(r["Estimate"]), "se_state": float(r["Std. Error"]),
                        "p_state": float(r["Pr(>|t|)"]),
                        "ci_lo": float(r["Estimate"]) - 1.96 * float(r["Std. Error"]),
                        "ci_hi": float(r["Estimate"]) + 1.96 * float(r["Std. Error"])})
    pre = [x for x in es_rows if x["outcome"] == okey and x["event_time"] < -1]
    nsig = sum(1 for x in pre if x["p_state"] < 0.05)
    print(f"  {okey:26s} pre-period coefs={len(pre):2d}  significant={nsig}  "
          f"{'PRE-TREND CONCERN' if nsig else 'pre-trend flat'}")

es = pd.DataFrame(es_rows)
es.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_A5_event_study.csv"), index=False)


# =============================================================================
# A6 : CALLAWAY-SANT'ANNA STAGGERED DiD   (was A18)
# =============================================================================
# WHY THIS EXISTS. A2/A5 are two-way FE estimators. Under staggered adoption with
# heterogeneous effects, TWFE is a variance-weighted average of 2x2 DiDs in which
# ALREADY-TREATED units serve as controls for later-treated ones. Those
# "forbidden comparisons" can carry NEGATIVE weights, so the TWFE coefficient
# need not lie inside the range of the true county-level effects. CS removes them
# by construction.
#
# ESTIMATOR, written out rather than called from a package (no `differences` or
# `csdid` installed, and an explicit implementation is reviewable):
#
#   For cohort g (first positive change) and calendar year t, base period g-1:
#       ATT(g,t) = E[Y_t - Y_{g-1} | G = g]  -  E[Y_t - Y_{g-1} | control]
#   Control = NOT-YET-TREATED at t (cohort > t) plus NEVER-TREATED. No unit that
#   is already treated at t is ever used as a control.
#
#   ATT(e) for event time e = t - g is the cohort-size-weighted mean of
#   ATT(g, g+e). Overall ATT is the cohort-size-weighted mean of post ATT(g,t).
#
# INFERENCE: nonparametric cluster bootstrap resampling STATES with replacement
# (300 reps), matching the state-clustered SEs used elsewhere in this file.
#
# NO COVARIATES -- THE MAIN CAVEAT, STATED PLAINLY. The (g,t) cells are thin and
# a doubly-robust version would need a propensity model estimated inside each
# cell. Parallel trends is therefore assumed UNCONDITIONALLY here, which is a
# STRONGER assumption than A2 makes. The covariate-adjusted variant is in
# script4i.
print("\n" + "=" * 78)
print("A6: Callaway-Sant'Anna staggered DiD (not-yet-treated + never-treated controls)")
print("=" * 78)

COHORTS = sorted(df.loc[df["cohort"].notna(), "cohort"].unique())
print(f"  cohorts: {[int(c) for c in COHORTS]}")
print(f"  control group: not-yet-treated (cohort > t) + never-treated")
print(f"  inference: {N_BOOT}-rep cluster bootstrap over states")
print(f"  covariates: NONE -- unconditional parallel trends. See docstring.\n")


def att_gt_table(wide, cohort_of, cohorts, years, keep_fips=None):
    """All ATT(g,t) for one outcome. `wide` is fips x year."""
    out = []
    idx = wide.index if keep_fips is None else wide.index.intersection(keep_fips)
    coh = cohort_of.reindex(idx)
    for g in cohorts:
        base = g - 1
        if base not in wide.columns:
            continue
        treated = idx[(coh == g).values]
        if len(treated) < 10:
            continue
        for t in years:
            if t == base or t not in wide.columns:
                continue
            ctrl = idx[((coh.isna()) | (coh > max(t, g))).values]
            d_all = wide[t] - wide[base]
            dt, dc = d_all.reindex(treated).dropna(), d_all.reindex(ctrl).dropna()
            if len(dt) < 10 or len(dc) < 10:
                continue
            out.append({"cohort": int(g), "year": int(t), "event_time": int(t - g),
                        "att": float(dt.mean() - dc.mean()),
                        "n_treated": int(len(dt)), "n_control": int(len(dc))})
    return out


cs_rows, csagg_rows = [], []
for okey, ocol in OUTCOMES.items():
    d = df[["fips", "year", "state_fips", "cohort", ocol]].dropna(subset=[ocol])
    if d.empty:
        continue
    wide      = d.pivot_table(index="fips", columns="year", values=ocol, aggfunc="first")
    cohort_of = d.groupby("fips")["cohort"].first()
    state_of  = d.groupby("fips")["state_fips"].first()
    years     = sorted(d["year"].unique())
    pt = att_gt_table(wide, cohort_of, COHORTS, years)
    if not pt:
        print(f"  {okey:26s} no estimable (g,t) cells")
        continue

    states   = state_of.dropna().unique()
    by_state = {s: state_of.index[state_of == s] for s in states}
    keys = [(r["cohort"], r["year"]) for r in pt]
    boot, boot_overall = {k: [] for k in keys}, []
    for _b in range(N_BOOT):
        draw   = RNG.choice(states, size=len(states), replace=True)
        fips_b = pd.Index(np.concatenate([by_state[s].values for s in draw]))
        pb = att_gt_table(wide, cohort_of, COHORTS, years,
                          keep_fips=pd.Index(pd.unique(fips_b)))
        m = {(r["cohort"], r["year"]): r["att"] for r in pb}
        for k in keys:
            if k in m:
                boot[k].append(m[k])
        post = [r for r in pb if r["event_time"] >= 0]
        if post:
            w = np.array([r["n_treated"] for r in post], float)
            boot_overall.append(float(np.average([r["att"] for r in post], weights=w)))

    for r in pt:
        b  = boot[(r["cohort"], r["year"])]
        se = float(np.std(b, ddof=1)) if len(b) > 10 else np.nan
        cs_rows.append({"registry": "A6", "outcome": okey, **r, "se_state": se,
                        "ci_lo": r["att"] - 1.96 * se, "ci_hi": r["att"] + 1.96 * se,
                        "p_state": float(2 * (1 - _norm.cdf(abs(r["att"] / se))))
                        if se and se > 0 else np.nan})

    pdf = pd.DataFrame(pt)
    for e, grp in pdf.groupby("event_time"):
        w = grp["n_treated"].astype(float)
        csagg_rows.append({"registry": "A6", "outcome": okey, "agg_level": "event_time",
                           "event_time": int(e),
                           "att": float(np.average(grp["att"], weights=w)),
                           "n_cohorts": int(grp["cohort"].nunique()),
                           "n_treated": int(grp["n_treated"].sum())})
    post = pdf[pdf.event_time >= 0]
    if len(post):
        overall = float(np.average(post["att"], weights=post["n_treated"].astype(float)))
        se_o = float(np.std(boot_overall, ddof=1)) if len(boot_overall) > 10 else np.nan
        csagg_rows.append({"registry": "A6", "outcome": okey, "agg_level": "overall_ATT",
                           "event_time": np.nan, "att": overall, "se_state": se_o,
                           "ci_lo": overall - 1.96 * se_o, "ci_hi": overall + 1.96 * se_o,
                           "n_cohorts": int(post["cohort"].nunique()),
                           "n_treated": int(post["n_treated"].sum())})
        star = "*" if (se_o and abs(overall) > 1.96 * se_o) else " "
        print(f"  {okey:26s} overall ATT={overall:+9.4f}  se={se_o:7.4f}{star}  "
              f"cells={len(pdf):3d}  cohorts={post['cohort'].nunique()}")

cs   = pd.DataFrame(cs_rows)
csag = pd.DataFrame(csagg_rows)
cs.to_csv(os.path.join(OUT_TABLES,   f"{TODAY}_A6_cs_att_gt.csv"), index=False)
csag.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_A6_cs_aggregated.csv"), index=False)


# =============================================================================
# HEADLINE COMPARISON: A2 (TWFE) vs A6 (Callaway-Sant'Anna)
# =============================================================================
# The comparison is only meaningful if the TREATMENT DEFINITION is held fixed and
# the ONLY thing that changes is the estimator. A gap is then the
# forbidden-comparison problem showing up.
#
# WHICH TREATMENT CS ACTUALLY USES (corrected 2026-09-28). CS is defined on
# cohorts, and `cohort = add_year` = the first ag-census wave at which the county
# records a POSITIVE change in its large-dairy count. That treatment is BINARY
# and ABSORBING -- once treated, always treated. It is `tr_e2_add_absorb`, NOT
# T1.
#
# T1 (tr_e1_lg_bin) is contemporaneous PRESENCE and can switch back off when a
# county loses its last large dairy. The two disagree on 10.6% of rows
# (T1=1/absorb=0 in 5,462 rows; T1=0/absorb=1 in 430) and cover different county
# sets (759 vs 572 ever-treated; 365 vs 572 switchers).
#
# Comparing A2(T1) against A6(cohort) would therefore change the estimator AND
# the treatment at once, and any gap could not be attributed to either. The TWFE
# side of this table is estimated on tr_e2_add_absorb for that reason. T1 results
# live in the A1/A2 grid above and are not comparable to CS.
print("\n" + "=" * 78)
print("HEADLINE: A2 (TWFE) vs A6 (Callaway-Sant'Anna)")
print("  treatment held fixed at tr_e2_add_absorb (the CS cohort definition)")
print("=" * 78)

CMP_TID = "E2"
cmp_rows = []
for okey in OUTCOMES:
    a2 = grid[(grid.registry == "A2") & (grid.treatment_id == CMP_TID)
              & (grid.outcome == okey) & (grid.control_set.str.startswith("step8"))]
    a6 = (csag[(csag["agg_level"] == "overall_ATT") & (csag.outcome == okey)]
          if len(csag) else pd.DataFrame())
    if a2.empty or a6.empty:
        continue
    a2, a6 = a2.iloc[0], a6.iloc[0]
    sig2 = a2.p_state < 0.05
    sig6 = bool(a6.se_state and abs(a6.att) > 1.96 * a6.se_state)
    cmp_rows.append({
        "outcome": okey,
        "TWFE_beta": a2.beta, "TWFE_se": a2.se_state, "TWFE_sig": sig2,
        "CS_ATT": a6.att, "CS_se": a6.se_state, "CS_sig": sig6,
        "ratio_CS_over_TWFE": a6.att / a2.beta if a2.beta else np.nan,
        "verdict": ("survives CS" if (sig2 and sig6) else
                    "TWFE-only (likely forbidden comparisons)" if sig2 else
                    "CS-only" if sig6 else "null in both"),
        "n_switchers_TWFE": a2.n_switchers, "n_treated_CS": a6.n_treated,
    })
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_HEADLINE_twfe_vs_cs.csv"), index=False)
if len(cmp):
    print(cmp[["outcome", "TWFE_beta", "TWFE_se", "TWFE_sig",
               "CS_ATT", "CS_se", "CS_sig", "verdict"]].to_string(index=False))


# =============================================================================
# VIF AUDIT
# =============================================================================
# VIF IS COMPUTED PER MODEL, NOT ONCE FOR THE COVARIATE SET. Two reasons, both
# demonstrated on this panel:
#   1. THE FE STRUCTURE CHANGES IT. Collinearity among these controls is almost
#      entirely CROSS-SECTIONAL, and county FE removes it. On the headline
#      sample median_household_income is VIF 2.98 in raw levels but 1.16
#      within-transformed; children_in_poverty 2.86 -> 1.13; the block max falls
#      2.98 -> 1.45. Quoting a raw-level VIF for a county-FE model overstates
#      collinearity by roughly a factor of two.
#   2. THE RIGHT-HAND SIDE CHANGES IT. The treatment and any conditioning terms
#      are part of the design matrix. Adding T2 plus log_pop moves the block max
#      1.45 -> 1.57. A3's conditioning sets and A4's interaction move it further.
# So each row below is the VIF of the ACTUAL design matrix that produced the
# corresponding coefficient, within-transformed wherever the model uses county FE.
va = pd.DataFrame(VIF_AUDIT)
va.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_VIF_audit.csv"), index=False)
print("\n" + "=" * 78)
print("VIF AUDIT -- per model, on the actual design matrix")
print("=" * 78)
if len(va):
    summ = (va.groupby("model")
              .agg(specs=("vif_max", "size"), transform=("within", "first"),
                   vif_max=("vif_max", "max"), vif_median=("vif_max", "median"),
                   over10=("vif_max", lambda s: int((s > 10).sum())),
                   absorbed=("n_zero_var", "sum"))
              .reset_index())
    summ["transform"] = np.where(summ["transform"], "within", "raw levels")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n  TOTAL specs audited: {len(va):,}   max VIF anywhere: {va.vif_max.max():.2f}"
          f"  on {va.loc[va.vif_max.idxmax(),'vif_max_var']}")
    print(f"  specs above 10 (the usual rule of thumb): {int((va.vif_max > 10).sum())}")
    print(f"  regressors absorbed by the FE (zero within-variance): {int(va.n_zero_var.sum())}")
    print("\n  A6 has no VIF: Callaway-Sant'Anna estimates cell means, not a regression.")
    print("  A3 VIF lives with its own output in script4l.")


# =============================================================================
# FIGURES
# =============================================================================
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

outs = [o for o in OUTCOMES if o in set(grid.outcome)]

# --- F1: A2 within coefficients, step-8 controls, all treatments -------------
if len(outs):
    n = len(outs); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, okey in zip(axes.ravel(), outs):
        v = grid[(grid.registry == "A2") & (grid.outcome == okey)
                 & (grid.control_set.str.startswith("step8"))].sort_values("treatment_id")
        if v.empty:
            ax.axis("off"); continue
        y = np.arange(len(v))
        ax.errorbar(v.beta, y, xerr=1.96 * v.se_state, fmt="o", ms=4, lw=1, capsize=2)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(v.treatment_id, fontsize=8)
        ax.set_title(okey, fontsize=9)
    for ax in axes.ravel()[len(outs):]:
        ax.axis("off")
    fig.suptitle("A2 within-county, step-8 pre-committed controls (95% CI, state-clustered)",
                 fontsize=10)
    fig.tight_layout()
    f1 = os.path.join(OUT_FIGS, f"{TODAY}_F1_A2_treatment_grid.png")
    fig.savefig(f1, bbox_inches="tight"); plt.close(fig)
    print("\nSaved:", f1)

# --- F2: A5 event studies ----------------------------------------------------
if len(es):
    eo = [o for o in OUTCOMES if o in set(es.outcome)]
    n = len(eo); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, okey in zip(axes.ravel(), eo):
        v = es[es.outcome == okey].sort_values("event_time")
        ax.errorbar(v.event_time, v.beta, yerr=1.96 * v.se_state, fmt="o-", ms=3, lw=1, capsize=2)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.axvline(-1, color="red", lw=0.8, ls=":")
        ax.set_title(okey, fontsize=9); ax.set_xlabel("event time", fontsize=8)
    for ax in axes.ravel()[len(eo):]:
        ax.axis("off")
    fig.suptitle("A5 TWFE event study (95% CI, state-clustered; ref = -1)", fontsize=10)
    fig.tight_layout()
    f2 = os.path.join(OUT_FIGS, f"{TODAY}_F2_A5_event_study.png")
    fig.savefig(f2, bbox_inches="tight"); plt.close(fig)
    print("Saved:", f2)

# --- F3: A6 event-time ATT ---------------------------------------------------
if len(csag):
    ev = csag[csag["agg_level"] == "event_time"]
    eo = [o for o in OUTCOMES if o in set(ev.outcome)]
    if eo:
        n = len(eo); ncol = 4; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
        for ax, okey in zip(axes.ravel(), eo):
            v = ev[ev.outcome == okey].sort_values("event_time")
            ax.plot(v.event_time, v.att, "o-", ms=3, lw=1)
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.axvline(0, color="red", lw=0.8, ls=":")
            ax.set_title(okey, fontsize=9); ax.set_xlabel("event time", fontsize=8)
        for ax in axes.ravel()[len(eo):]:
            ax.axis("off")
        fig.suptitle("A6 Callaway-Sant'Anna ATT by event time (cohort-size weighted)",
                     fontsize=10)
        fig.tight_layout()
        f3 = os.path.join(OUT_FIGS, f"{TODAY}_F3_A6_cs_event_time.png")
        fig.savefig(f3, bbox_inches="tight"); plt.close(fig)
        print("Saved:", f3)

print("\n" + "=" * 78)
print("script4a COMPLETE")
print(f"  tables -> {OUT_TABLES}")
print(f"  figs   -> {OUT_FIGS}")
print("=" * 78)
