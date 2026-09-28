#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4a-twfe-eventstudy.py  --  PART A: design-based estimates.

Registry IDs covered (see code-audit/plans/2026-09-17_analysis-registry.md):
    A1   Pooled cross-section OLS      (state + year FE)
    A2   TWFE within-county            (county + year FE)
    A3   Horse race                    (dairy conditional on beef/hogs/chickens)
    A6   Dairy x FSIS interaction      (2017-2023, thin sample)
    A7   TWFE event study              (leads/lags around first positive change)
    A18  Callaway-Sant'Anna staggered DiD   <- promoted into Part A, 2026-09-18
(A5, functional form, is subsumed by treatments I3/I4/C3 in the grid below.)
(A4, size-threshold sensitivity, was DROPPED 2026-09-18: the whole point of the
 Group E/I/M/C treatment family is that small and large operations are not
 expected to act alike, so re-running one model across pooled size thresholds
 asks a question the treatment design already answers more directly.)

A1, A2 and A18 are run across the FULL treatment grid defined in
script4_treatment.TREATMENTS -- 14 definitions spanning presence, entry, dose,
build-vs-consolidation, and composition -- crossed with all 6 outcomes.

STANDARD ERRORS
---------------
Every estimate reports THREE variance estimators side by side, because SE choice
was the specific concern raised in review:
    se_state   CRV1 clustered on state_fips   <- HEADLINE. Most conservative here.
    se_county  CRV1 clustered on fips
    se_hetero  heteroskedasticity-robust, no clustering
On the reference spec (Poor MH Days, CORE_9, county+year FE) these were
0.0372 / 0.0331 / 0.0207 -- state clustering is the most conservative of the three,
so reporting it is not an understatement of uncertainty. Treatment assignment is
spatially correlated within state (dairy regions), which is the substantive reason
to cluster there rather than at county.

Estimation uses `pyfixest.feols`, NOT the old script4-model-test.py helpers.
Reason: `within_transform()` in that file applied a SINGLE demeaning pass
(x - mean_i - mean_t + grand_mean). That is exact only for a balanced panel. This
panel is unbalanced (1-13 observations per county; only 77.3% have the full span),
and one-shot demeaning gave beta = +0.08466 where correct alternating-projection
demeaning gives +0.07970 -- a 6.2% error in the POINT ESTIMATE, not just the SE.
statsmodels additionally did not count absorbed FE in its residual degrees of
freedom. pyfixest handles both.

IDENTIFYING VARIATION
---------------------
Every row reports n_switchers and n_transitions alongside beta. With county FE,
only counties whose treatment CHANGES contribute to beta; on the reference spec
that was 93 of 2,231 counties under the legacy 25-control vector. Reporting it
inline makes the credibility of each row visible instead of a judgement call.

Outputs -> Dropbox/Mental/Data/output/tables/script4a/  and  figs/script4a/
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib.pyplot as plt
from scipy.stats import norm as _norm

from script4_treatment import (
    load_panel, OUTCOMES, TREATMENTS, CENSUS_YEARS,
    CONTROL_CORE9, CONTROL_FULL25, CONTROL_PRETREAT, db_data, figs_dir, tables_dir, os, date,
)
from functions import latest_file_glob

OUT_FIGS   = os.path.join(figs_dir,   "script4a")
OUT_TABLES = os.path.join(tables_dir, "script4a")
for _d in (OUT_FIGS, OUT_TABLES):
    os.makedirs(_d, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")

CONTROLS = CONTROL_PRETREAT       # derived pool minus mediators (19). See module
                                  # docstring: CORE_9 was inherited, not derived;
                                  # FULL25 destroys 2/3 of the switchers.
N_BOOT   = 300                    # bootstrap reps for A18 cluster-bootstrap SEs
RNG      = np.random.default_rng(20260918)


# =============================================================================
# Name sanitising -- pyfixest formulas cannot contain %, -, or spaces
# =============================================================================
def _clean(name):
    # NOTE: "/" must be stripped too -- %_native_hawaiian/other_pacific_islander
    # otherwise splits into two tokens and pyfixest's formula parser fails.
    return (name.replace("%", "pct").replace("-", "_").replace(" ", "_")
                .replace("(", "").replace(")", "").replace("/", "_")
                .replace("+", "p").replace(",", "_"))

def prep(df, cols):
    """Subset to `cols`, drop incomplete rows, return (frame, {orig: clean})."""
    cols = list(dict.fromkeys(cols))
    sub = df[cols].dropna().copy()
    mapping = {c: _clean(c) for c in cols}
    return sub.rename(columns=mapping), mapping


def variation(sub, tcol):
    """Counties whose treatment changes, and how many changes there are.
    This is what actually identifies a within-county coefficient."""
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
    """
    VIF over the FULL right-hand side of THIS spec -- treatment, any conditioning
    terms, and all controls -- not just the control block.

    Computed on the WITHIN-TRANSFORMED data when `within`, because that is the
    variation a county+year FE regression actually uses. Raw-level VIF is the wrong
    diagnostic here: collinearity among these controls is almost entirely
    cross-sectional and county FE removes it (children_in_poverty: 9.10 raw -> 1.13
    within).

    Also returns n_zero_var: regressors with NO within-county variation. Those are
    absorbed by the fixed effects and contribute nothing -- this is how the original
    C4 spec was caught silently conditioning on a county-constant variable.
    """
    W = sub[rhs_cols].astype(float).copy()
    if within:
        for c in rhs_cols:
            W[c] = (W[c] - sub.groupby("fips")[c].transform("mean")
                        - sub.groupby("year")[c].transform("mean") + sub[c].mean())
    zero_var = [c for c in rhs_cols if W[c].std() <= 1e-10]
    keep = [c for c in rhs_cols if c not in zero_var]
    if len(keep) < 2:
        return {"vif_max": np.nan, "vif_max_var": "", "vif_treat": np.nan, "n_zero_var": len(zero_var)}
    try:
        R = np.corrcoef(W[keep].values, rowvar=False)
        v = pd.Series(np.diag(np.linalg.pinv(R)), index=keep)
    except Exception:
        return {"vif_max": np.nan, "vif_max_var": "", "vif_treat": np.nan, "n_zero_var": len(zero_var)}
    return {"vif_max": float(v.max()), "vif_max_var": str(v.idxmax()),
            "vif_treat": float(v.get(rhs_cols[0], np.nan)), "n_zero_var": len(zero_var)}


def fit(sub, y, tcol, extra, fe, label):
    """
    One regression, three variance estimators. `fe` is the pyfixest FE string,
    e.g. "fips + year" (within) or "state_fips + year" (pooled).
    Returns a flat dict, or None if the cell is too thin to estimate.
    """
    rhs = " + ".join([tcol] + list(extra) + CLEAN_CONTROLS)
    fml = f"{y} ~ {rhs} | {fe}"
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
                # co-estimated conditioning coefficient (e.g. log(small) in C3/C4)
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
print("script4a -- PART A: design-based estimates (A1, A2, A3, A4, A6, A7, A18)")
print("=" * 78)

df = load_panel()
EV = df.attrs["events"]
print(f"panel: {os.path.basename(df.attrs['panel_path'])}  {df.shape[0]:,} rows, "
      f"{df['fips'].nunique():,} rural counties, {int(df.year.min())}-{int(df.year.max())}")
print(f"controls: PRETREAT ({len(CONTROLS)})  {CONTROLS}")
print(f"wave events -- entry {EV.is_entry.sum()} | expansion {EV.is_expansion.sum()} | "
      f"build {EV.is_build.sum()} | consolidate {EV.is_consolidate.sum()} | exit {EV.is_exit.sum()} (ignored, absorbing)")
print(f"cohorts: {df[df.cohort.notna()].groupby('cohort').fips.nunique().to_dict()}")

CLEAN_CONTROLS = [_clean(c) for c in CONTROLS]

# =============================================================================
# A1 + A2 : the treatment grid, pooled and within
# =============================================================================
print("\n" + "=" * 78)
print("A1 (pooled: state+year FE) and A2 (within: county+year FE) x 14 treatments x 6 outcomes")
print("=" * 78)

grid_rows = []
for tid, (tcol, cond, tlabel) in TREATMENTS.items():
    for okey, ocol in OUTCOMES.items():
        need = ["fips", "year", "state_fips", ocol, tcol] + list(cond) + CONTROLS
        sub, mp = prep(df, need)
        if len(sub) < 200:
            continue
        ct, cy = mp[tcol], mp[ocol]
        ce = [mp[c] for c in cond]
        var = variation(sub, ct)
        for spec, fe in [("A1 pooled (state+year FE)", "state_fips + year"),
                         ("A2 within (county+year FE)", "fips + year")]:
            r = fit(sub, cy, ct, ce, fe, f"{tid}|{okey}|{spec}")
            if r is None:
                continue
            vif = rhs_vif(sub, [ct] + ce + CLEAN_CONTROLS, within=fe.startswith("fips"))
            grid_rows.append({"registry": spec.split()[0], "spec": spec,
                              "treatment_id": tid, "treatment": tlabel,
                              "treatment_col": tcol,
                              "conditioning": "+".join(cond) if cond else "",
                              "outcome": okey, **r, **var, **vif})
    print(f"  {tid:3s} {tlabel:38s} done")

grid = pd.DataFrame(grid_rows)
gpath = os.path.join(OUT_TABLES, f"{TODAY}_A1_A2_treatment_grid.csv")
grid.to_csv(gpath, index=False)
print(f"\nSaved: {gpath}   ({len(grid):,} rows)")

print("\n--- A2 (within) results, all treatments, Poor MH Days ---")
v = grid[(grid.registry == "A2") & (grid.outcome == "Poor MH Days")]
for _, r in v.iterrows():
    star = "*" if r.p_state < 0.05 else " "
    print(f"  {r.treatment_id:3s} {r.treatment:36s} b={r.beta:+9.4f}  "
          f"se_st={r.se_state:7.4f}{star} se_cty={r.se_county:7.4f} se_het={r.se_hetero:7.4f}  "
          f"N={r.N:6,}  switchers={r.n_switchers:4d}")


# =============================================================================
# A3 : horse race -- is the dairy coefficient really dairy, or "any large CAFO"?
# =============================================================================
# Isolated = dairy alone. Conditional = dairy entered jointly with beef/hogs/
# chickens presence. Controls held identical across the two so the ONLY thing that
# changes is whether the other three animal types are partialled out.
print("\n" + "=" * 78)
print("A3: horse race -- dairy isolated vs conditional on beef/hogs/chickens")
print("=" * 78)

for _a in ["hogs", "beef", "chickens"]:
    _c = f"cafo_{_a}_large"
    df[f"any_large_{_a}"] = (df[_c] > 0).astype(float).where(df[_c].notna())
ANIMALS = ["any_large_hogs", "any_large_beef", "any_large_chickens"]

hr_rows = []
for okey, ocol in OUTCOMES.items():
    need = ["fips", "year", "state_fips", ocol, "tr_e1_lg_bin"] + ANIMALS + CONTROLS
    sub, mp = prep(df, need)
    if len(sub) < 200:
        continue
    ct, cy = mp["tr_e1_lg_bin"], mp[ocol]
    var = variation(sub, ct)
    for lab, extra in [("Isolated (dairy alone)", []),
                       ("Conditional (all 4 animals)", [mp[a] for a in ANIMALS])]:
        r = fit(sub, cy, ct, extra, "fips + year", f"A3|{okey}|{lab}")
        if r:
            vif = rhs_vif(sub, [ct] + list(extra) + CLEAN_CONTROLS)
            hr_rows.append({"registry": "A3", "outcome": okey, "model": lab, **r, **var, **vif})
hr = pd.DataFrame(hr_rows)
hr.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_A3_horse_race.csv"), index=False)
print(hr[["outcome", "model", "beta", "se_state", "p_state", "N"]].to_string(index=False))


# =============================================================================
# A6 : dairy x FSIS interaction -- exploratory, thin by construction
# =============================================================================
# FSIS establishment counts exist only 2017-2023 at ~46-50% county coverage, so
# this is underpowered relative to everything above and is flagged as exploratory
# rather than presented on equal footing.
print("\n" + "=" * 78)
print("A6: dairy x FSIS interaction (2017-2023 only, exploratory)")
print("=" * 78)

fsis_rows = []
try:
    _fp = latest_file_glob(os.path.join(db_data, "merged"), "*_panel_fsis.csv")
    _f = pd.read_csv(_fp, low_memory=False)[["fips", "year", "n_unique_establishments_fsis"]]
    _f["fips"] = _f["fips"].astype(str).str.zfill(5)
    dff = df.merge(_f, on=["fips", "year"], how="left")
    cov = dff["year"].between(2017, 2023)
    dff["any_fsis"] = np.where(cov, (dff["n_unique_establishments_fsis"].fillna(0) > 0).astype(float), np.nan)
    dff["dairy_x_fsis"] = dff["tr_e1_lg_bin"] * dff["any_fsis"]
    for okey, ocol in OUTCOMES.items():
        need = ["fips", "year", "state_fips", ocol, "tr_e1_lg_bin", "any_fsis", "dairy_x_fsis"] + CONTROLS
        sub, mp = prep(dff, need)
        if len(sub) < 200:
            continue
        var = variation(sub, mp["tr_e1_lg_bin"])
        r = fit(sub, mp[ocol], mp["dairy_x_fsis"],
                [mp["tr_e1_lg_bin"], mp["any_fsis"]], "fips + year", f"A6|{okey}")
        if r:
            vif = rhs_vif(sub, [mp["dairy_x_fsis"], mp["tr_e1_lg_bin"], mp["any_fsis"]] + CLEAN_CONTROLS)
            fsis_rows.append({"registry": "A6", "outcome": okey, "term": "dairy x FSIS", **r, **var, **vif})
    print(f"  FSIS merged from {os.path.basename(_fp)}")
except Exception as e:
    print(f"  FSIS panel unavailable ({type(e).__name__}) -- A6 skipped")
fs = pd.DataFrame(fsis_rows)
if len(fs):
    fs.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_A6_fsis_interaction.csv"), index=False)
    print(fs[["outcome", "beta", "se_state", "p_state", "N"]].to_string(index=False))


# =============================================================================
# A7 : TWFE event study around the first positive change
# =============================================================================
# Leads/lags relative to the county's first positive change in large-dairy ops,
# omitted category t_rel = -1. Control group = never-treated plus not-yet-treated,
# pooled (the standard TWFE event study). 95% CI throughout -- the old script used
# 90% here and 95% everywhere else to match an external memo; that asymmetry is
# dropped, since it made rows non-comparable within the same figure.
#
# READ THE PRE-TREND COEFFICIENTS BEFORE THE POST ONES. Treatment timing carries
# up to 5 years of measurement error (census forward-fill), so leads close to zero
# are partly contaminated by already-treated periods.
print("\n" + "=" * 78)
print("A7: TWFE event study around first positive change")
print("=" * 78)

LEADS, LAGS = 8, 8
es_rows = []
for okey, ocol in OUTCOMES.items():
    need = ["fips", "year", "state_fips", ocol, "t_rel"] + CONTROLS
    sub = df[need + ["cohort"]].copy()
    # never-treated: t_rel is NaN. Bin them to a sentinel far outside the window
    # so pyfixest keeps them as the comparison group rather than dropping them.
    sub["ev"] = sub["t_rel"].clip(-LEADS, LAGS)
    sub.loc[sub["t_rel"].isna(), "ev"] = -1        # never-treated pooled into base
    sub = sub.drop(columns=["t_rel", "cohort"]).dropna()
    sub = sub.rename(columns={c: _clean(c) for c in sub.columns})
    if len(sub) < 500:
        continue
    fml = f"{_clean(ocol)} ~ i(ev, ref=-1) + " + " + ".join(CLEAN_CONTROLS) + " | fips + year"
    try:
        m = pf.feols(fml, data=sub, vcov={"CRV1": "state_fips"})
        t = m.tidy()
    except Exception as e:
        print(f"  [{okey}] event study failed: {e}")
        continue
    for idx, r in t.iterrows():
        if not str(idx).startswith("ev::"):
            continue
        try:
            e = int(float(str(idx).split("::")[1]))
        except (ValueError, IndexError):
            continue
        es_rows.append({"registry": "A7", "outcome": okey, "event_time": e,
                        "beta": float(r["Estimate"]), "se_state": float(r["Std. Error"]),
                        "p_state": float(r["Pr(>|t|)"]),
                        "ci_lo": float(r["Estimate"]) - 1.96 * float(r["Std. Error"]),
                        "ci_hi": float(r["Estimate"]) + 1.96 * float(r["Std. Error"])})
    pre = [x for x in es_rows if x["outcome"] == okey and x["event_time"] < -1]
    npre_sig = sum(1 for x in pre if x["p_state"] < 0.05)
    print(f"  {okey:26s} pre-period coefs={len(pre):2d}  significant={npre_sig}  "
          f"{'PRE-TREND CONCERN' if npre_sig else 'pre-trend flat'}")

es = pd.DataFrame(es_rows)
es.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_A7_event_study.csv"), index=False)


# =============================================================================
# A18 : CALLAWAY-SANT'ANNA STAGGERED DiD   (promoted into Part A, 2026-09-18)
# =============================================================================
# WHY THIS EXISTS. A2/A7 are two-way FE estimators. Under staggered adoption with
# heterogeneous effects, TWFE is a variance-weighted average of 2x2 DiDs in which
# ALREADY-TREATED units serve as controls for later-treated ones. Those
# "forbidden comparisons" can carry negative weights, so the TWFE coefficient need
# not lie inside the range of the true county-level effects. Callaway-Sant'Anna
# removes them by construction.
#
# ESTIMATOR, written out rather than called from a package (no `differences` or
# `csdid` installed, and an explicit implementation is reviewable):
#
#   For cohort g (first positive change) and calendar year t, with base period g-1:
#       ATT(g,t) = E[ Y_t - Y_{g-1} | G = g ]  -  E[ Y_t - Y_{g-1} | control ]
#   Control group = NOT-YET-TREATED at t (cohort > t) plus NEVER-TREATED.
#   Only clean comparisons enter: no unit that is already treated at t is ever
#   used as a control.
#
#   Aggregation: ATT(e) for event time e = t - g is the cohort-size-weighted mean
#   of ATT(g, g+e) over cohorts observed at that horizon. Overall ATT is the
#   cohort-size-weighted mean of post-treatment ATT(g,t).
#
# INFERENCE: nonparametric cluster bootstrap resampling STATES with replacement
# (300 reps), matching the state-clustered SEs used everywhere else in this script.
#
# NO COVARIATES. The unconditional version is reported because the (g,t) cells are
# thin -- the smallest cohort has 45 counties -- and a doubly-robust version would
# need a propensity model estimated inside each cell. Parallel trends is therefore
# assumed UNCONDITIONALLY here, which is a stronger assumption than A2 makes.
# Stated plainly rather than buried: this is the main caveat on A18.
print("\n" + "=" * 78)
print("A18: Callaway-Sant'Anna staggered DiD (not-yet-treated + never-treated controls)")
print("=" * 78)

COHORTS = sorted(df.loc[df["cohort"].notna(), "cohort"].unique())
print(f"  cohorts: {[int(c) for c in COHORTS]}")
print(f"  control group: not-yet-treated (cohort > t) + never-treated")
print(f"  inference: {N_BOOT}-rep cluster bootstrap over states\n")


def att_gt_table(wide, cohort_of, state_of, cohorts, years, keep_fips=None):
    """All ATT(g,t) for one outcome. `wide` is fips x year. Returns list of dicts."""
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
            # clean controls only: never-treated, or not yet treated AT t
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
    wide = d.pivot_table(index="fips", columns="year", values=ocol, aggfunc="first")
    cohort_of = d.groupby("fips")["cohort"].first()
    state_of  = d.groupby("fips")["state_fips"].first()
    years = sorted(d["year"].unique())
    pt = att_gt_table(wide, cohort_of, state_of, COHORTS, years)
    if not pt:
        print(f"  {okey:26s} no estimable (g,t) cells")
        continue

    # ---- cluster bootstrap over states -------------------------------------
    states = state_of.dropna().unique()
    by_state = {s: state_of.index[state_of == s] for s in states}
    keys = [(r["cohort"], r["year"]) for r in pt]
    boot = {k: [] for k in keys}
    boot_overall = []
    for _b in range(N_BOOT):
        draw = RNG.choice(states, size=len(states), replace=True)
        fips_b = pd.Index(np.concatenate([by_state[s].values for s in draw]))
        pb = att_gt_table(wide, cohort_of, state_of, COHORTS, years,
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
        b = boot[(r["cohort"], r["year"])]
        se = float(np.std(b, ddof=1)) if len(b) > 10 else np.nan
        cs_rows.append({"registry": "A18", "outcome": okey, **r, "se_state": se,
                        "ci_lo": r["att"] - 1.96 * se, "ci_hi": r["att"] + 1.96 * se,
                        "p_state": float(2 * (1 - _norm.cdf(abs(r["att"] / se))))
                        if se and se > 0 else np.nan})

    # event-time aggregation, cohort-size weighted
    pdf = pd.DataFrame(pt)
    for e, grp in pdf.groupby("event_time"):
        w = grp["n_treated"].astype(float)
        csagg_rows.append({"registry": "A18", "outcome": okey, "agg_level": "event_time",
                           "event_time": int(e),
                           "att": float(np.average(grp["att"], weights=w)),
                           "n_cohorts": int(grp["cohort"].nunique()),
                           "n_treated": int(grp["n_treated"].sum())})
    post = pdf[pdf.event_time >= 0]
    if len(post):
        overall = float(np.average(post["att"], weights=post["n_treated"].astype(float)))
        se_o = float(np.std(boot_overall, ddof=1)) if len(boot_overall) > 10 else np.nan
        csagg_rows.append({"registry": "A18", "outcome": okey, "agg_level": "overall_ATT",
                           "event_time": np.nan, "att": overall, "se_state": se_o,
                           "ci_lo": overall - 1.96 * se_o, "ci_hi": overall + 1.96 * se_o,
                           "n_cohorts": int(post["cohort"].nunique()),
                           "n_treated": int(post["n_treated"].sum())})
        star = "*" if (se_o and abs(overall) > 1.96 * se_o) else " "
        print(f"  {okey:26s} overall ATT={overall:+9.4f}  se={se_o:7.4f}{star}  "
              f"cells={len(pdf):3d}  cohorts={post['cohort'].nunique()}")

cs   = pd.DataFrame(cs_rows)
csag = pd.DataFrame(csagg_rows)
cs.to_csv(os.path.join(OUT_TABLES,   f"{TODAY}_A18_cs_att_gt.csv"), index=False)
csag.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_A18_cs_aggregated.csv"), index=False)


# =============================================================================
# HEADLINE COMPARISON: A2 (TWFE) vs A18 (Callaway-Sant'Anna), same treatment
# =============================================================================
# Both use E2 (any positive change, absorbing) so the treatment DEFINITION is
# held fixed and the only thing that changes is the estimator. A gap between them
# is the forbidden-comparison problem showing up: TWFE uses already-treated
# counties as controls for later-treated ones; CS does not.
print("\n" + "=" * 78)
print("HEADLINE: TWFE (A2) vs Callaway-Sant'Anna (A18) -- same treatment, same sample frame")
print("=" * 78)

cmp_rows = []
for okey in OUTCOMES:
    a2 = grid[(grid.registry == "A2") & (grid.treatment_id == "E2") & (grid.outcome == okey)]
    a18 = csag[(csag["agg_level"] == "overall_ATT") & (csag.outcome == okey)] if len(csag) else pd.DataFrame()
    if a2.empty or a18.empty:
        continue
    a2, a18 = a2.iloc[0], a18.iloc[0]
    sig2  = a2.p_state < 0.05
    sig18 = bool(a18.se_state and abs(a18.att) > 1.96 * a18.se_state)
    cmp_rows.append({
        "outcome": okey,
        "TWFE_beta": a2.beta, "TWFE_se": a2.se_state, "TWFE_sig": sig2,
        "CS_ATT": a18.att, "CS_se": a18.se_state, "CS_sig": sig18,
        "ratio_CS_over_TWFE": a18.att / a2.beta if a2.beta else np.nan,
        "verdict": ("survives CS" if (sig2 and sig18) else
                    "TWFE-only (likely forbidden comparisons)" if sig2 else
                    "CS-only" if sig18 else "null in both"),
        "n_switchers_TWFE": a2.n_switchers, "n_treated_CS": a18.n_treated,
    })
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_HEADLINE_twfe_vs_cs.csv"), index=False)
print(cmp[["outcome", "TWFE_beta", "TWFE_se", "TWFE_sig",
           "CS_ATT", "CS_se", "CS_sig", "verdict"]].to_string(index=False))


# =============================================================================
# FIGURES
# =============================================================================
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# --- F1: treatment grid, A2 within coefficients, all 14 treatments -----------
outs = [o for o in OUTCOMES if o in set(grid.outcome)]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, okey in zip(axes.ravel(), outs):
    v = grid[(grid.registry == "A2") & (grid.outcome == okey)].copy()
    v = v.sort_values("treatment_id")
    y = np.arange(len(v))
    colors = ["#2166ac" if p < 0.05 else "#bdbdbd" for p in v.p_state]
    ax.errorbar(v.beta, y, xerr=1.96 * v.se_state, fmt="o", ms=4, elinewidth=1.2,
                capsize=2, ecolor="#888888", linestyle="none")
    ax.scatter(v.beta, y, c=colors, s=34, zorder=3, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.treatment_id} {r.treatment[:26]}" for r in v.itertuples()], fontsize=6.5)
    ax.set_title(okey, fontsize=9, fontweight="bold")
    ax.set_xlabel("beta (units differ by row)", fontsize=7)
for j in range(len(outs), len(axes.ravel())):
    axes.ravel()[j].set_visible(False)
fig.suptitle("A2 within-county (county+year FE) across all 14 treatment definitions\n"
             "State-clustered 95% CI | blue = p<0.05 | x-axis units are NOT comparable across rows",
             fontsize=11)
plt.tight_layout()
f1 = os.path.join(OUT_FIGS, f"{TODAY}_F1_treatment_grid_within.png")
fig.savefig(f1, bbox_inches="tight"); plt.close(fig)
print("\nSaved:", f1)

# --- F2: event studies, A7 --------------------------------------------------
if len(es):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ax, okey in zip(axes.ravel(), outs):
        v = es[es.outcome == okey].sort_values("event_time")
        if v.empty:
            ax.set_visible(False); continue
        pre, post = v[v.event_time < 0], v[v.event_time >= 0]
        for part, col in [(pre, "#888888"), (post, "#b2182b")]:
            ax.errorbar(part.event_time, part.beta, yerr=1.96 * part.se_state,
                        fmt="o", ms=4, color=col, elinewidth=1.1, capsize=2)
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.axvline(-0.5, color="#2166ac", lw=1.0, ls="--")
        ax.set_title(okey, fontsize=9, fontweight="bold")
        ax.set_xlabel("years since first positive change", fontsize=7)
    for j in range(len(outs), len(axes.ravel())):
        axes.ravel()[j].set_visible(False)
    fig.suptitle("A7 TWFE event study | grey = pre-period, red = post | omitted t = -1\n"
                 "Timing carries up to 5 years of error (census forward-fill): read leads with that in mind",
                 fontsize=11)
    plt.tight_layout()
    f2 = os.path.join(OUT_FIGS, f"{TODAY}_F2_event_study.png")
    fig.savefig(f2, bbox_inches="tight"); plt.close(fig)
    print("Saved:", f2)

# --- F3: A18 event-time ATT -------------------------------------------------
if len(csag):
    ev_agg = csag[csag["agg_level"] == "event_time"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for ax, okey in zip(axes.ravel(), outs):
        v = ev_agg[ev_agg.outcome == okey].sort_values("event_time")
        if v.empty:
            ax.set_visible(False); continue
        c = ["#888888" if e < 0 else "#b2182b" for e in v.event_time]
        ax.scatter(v.event_time, v.att, c=c, s=30, zorder=3)
        ax.plot(v.event_time, v.att, color="#cccccc", lw=0.8, zorder=1)
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.axvline(-0.5, color="#2166ac", lw=1.0, ls="--")
        ax.set_title(okey, fontsize=9, fontweight="bold")
        ax.set_xlabel("event time", fontsize=7)
    for j in range(len(outs), len(axes.ravel())):
        axes.ravel()[j].set_visible(False)
    fig.suptitle("A18 Callaway-Sant'Anna: cohort-size-weighted ATT(e) by event time\n"
                 "Clean comparisons only -- never-treated + not-yet-treated controls",
                 fontsize=11)
    plt.tight_layout()
    f3 = os.path.join(OUT_FIGS, f"{TODAY}_F3_cs_event_time.png")
    fig.savefig(f3, bbox_inches="tight"); plt.close(fig)
    print("Saved:", f3)

# --- F4: headline TWFE vs CS ------------------------------------------------
if len(cmp):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(cmp))
    ax.errorbar(cmp.TWFE_beta / cmp.TWFE_se, y - 0.15, fmt="s", ms=7,
                color="#2166ac", label="TWFE (A2)", linestyle="none")
    ax.errorbar(cmp.CS_ATT / cmp.CS_se, y + 0.15, fmt="o", ms=7,
                color="#b2182b", label="Callaway-Sant'Anna (A18)", linestyle="none")
    for c in (-1.96, 1.96):
        ax.axvline(c, color="#999999", lw=0.9, ls="--")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(cmp.outcome, fontsize=9)
    ax.set_xlabel("t-statistic (state-clustered) | dashed lines = +/-1.96", fontsize=9)
    ax.set_title("Does the TWFE result survive a heterogeneity-robust estimator?\n"
                 "Same treatment (E2, absorbing positive change), same outcomes", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    f4 = os.path.join(OUT_FIGS, f"{TODAY}_F4_twfe_vs_cs.png")
    fig.savefig(f4, bbox_inches="tight"); plt.close(fig)
    print("Saved:", f4)

print("\n" + "=" * 78)
print("script4a COMPLETE")
print(f"  tables -> {OUT_TABLES}")
print(f"  figs   -> {OUT_FIGS}")
print("=" * 78)


# =============================================================================
# VIF AUDIT -- one row per estimated spec, so collinearity is a reported
# diagnostic rather than a one-off check on a single regression.
# =============================================================================
# Flags:
#   vif_max > 5     conventional attention threshold
#   n_zero_var > 0  a regressor with NO within-county variation was included and
#                   silently absorbed by the fixed effects. This is a SPEC ERROR,
#                   not a collinearity problem, and matters more than VIF: the
#                   original C4 conditioned on a county-constant baseline, so the
#                   regression quietly reduced to the unconditional one.
print("\n" + "=" * 78)
print("VIF AUDIT across every estimated spec")
print("=" * 78)

audit = pd.concat([
    grid.assign(block=grid.registry + " " + grid.treatment_id),
    hr.assign(block="A3"),
    (fs.assign(block="A6") if len(fs) else pd.DataFrame()),
], ignore_index=True)

if "vif_max" in audit.columns:
    a = audit.dropna(subset=["vif_max"])
    print(f"  specs audited: {len(a):,}")
    print(f"  vif_max  > 10 : {(a.vif_max > 10).sum()}")
    print(f"  vif_max  >  5 : {(a.vif_max > 5).sum()}")
    print(f"  vif_max  >  3 : {(a.vif_max > 3).sum()}")
    print(f"  overall max   : {a.vif_max.max():.2f}  ({a.loc[a.vif_max.idxmax(), 'vif_max_var']})")
    print(f"  specs with an ABSORBED (zero within-variation) regressor: {(a.n_zero_var > 0).sum()}")
    bad = a[(a.vif_max > 5) | (a.n_zero_var > 0)]
    if len(bad):
        print("\n  FLAGGED:")
        print(bad[["block", "outcome", "vif_max", "vif_max_var", "n_zero_var"]].to_string(index=False))
    else:
        print("\n  No spec exceeds VIF 5 and no spec contains an absorbed regressor.")
    audit[["block", "registry", "outcome", "beta", "se_state", "p_state", "N",
           "n_switchers", "vif_max", "vif_max_var", "vif_treat", "n_zero_var"]].to_csv(
        os.path.join(OUT_TABLES, f"{TODAY}_VIF_audit.csv"), index=False)
    print(f"\nSaved: {os.path.join(OUT_TABLES, f'{TODAY}_VIF_audit.csv')}")
