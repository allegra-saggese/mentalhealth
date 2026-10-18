#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4u-did-by-treatment.py  --  DiD ESTIMATES FOR TWO "FIRST TREATED YEAR" DEFINITIONS (added 2026-10-17)

Runs the same three estimators for two definitions of when a county becomes treated, and adds the
pre-trend adjustments the team asked about. Written as a new script (script4a untouched) with the CS
code in did_cs.py; a built-in check reproduces script4a's results for the first definition.

TREATMENT DEFINITIONS   (g = a county's first treated year; always a census wave: 2007/12/17/22)
    first expansion   tr_e2_add_absorb  g = first wave the county's number of large dairies goes UP.
                      Sample: every rural county (same as script4a).
    first entry       tr_e3_entry_absorb g = first wave the count goes from 0 to 1+.
                      Sample: entry counties + counties that NEVER have a large dairy in any census
                      wave (clean comparison group). Counties that only expand an existing large dairy
                      or have one throughout are LEFT OUT, because they are not valid "untreated"
                      comparisons for an entry design (decision Q-A, 2026-10-17).

ESTIMATORS    (all: 8 controls from COVARIATE_ORDER[:8], county + year fixed effects, SE clustered by state)
    static TWFE      Y_it = b D_it + X_it'c + a_i + d_t + e_it
    event study      Y_it = sum_{e != -1} theta_e 1[t - g_i = e] + X_it'c + a_i + d_t + e_it,
                     e = year - g clipped to [-8, 8], reference e = -1, never-treated in the reference.
                     Leads (e < -1) are the pre-trend test; joint Wald chi-square that all leads are 0.
    Callaway-Sant'Anna, covariate-adjusted (see did_cs.py), bootstrap over states.

PRE-TREND ADJUSTMENTS    (each is a second specification next to the baseline)
    county trends    adds a county-specific linear time trend l_i * t to the static TWFE and the event study.
                     NOT done with pyfixest's  fips[year]  syntax: in pyfixest 0.60 that term is silently
                     ignored (tested 2026-10-17: identical coefficient to the model without it). Instead the
                     outcome and every regressor are residualized on county intercepts + county linear trends +
                     year effects by alternating projections (Frisch-Waugh), then OLS with state-clustered
                     SEs (fe_ols below). fe_ols with no trend is checked to equal pyfixest.
    past outcomes    adds the outcome's own levels in years g-1 and g-2 to the CS covariates, so comparison
                     counties are matched on their pre-treatment outcome path.
    state x year FE  replaces the year effects with state-by-year effects: treated counties are compared only
                     with comparison counties in the SAME STATE and year, which removes regional drift.
                     (checked: every treated county sits in a state that also has comparison counties.)
    baseline x time  adds baseline (2002) county characteristics x (year-2002) and x (year-2002)^2:
                     log population, median household income, % Hispanic, % 65+, unemployment, and the 2002
                     number of dairy operations (all sizes, and large). Lets different kinds of counties drift
                     differently using only pre-treatment information, without a free slope per county.
    (the HonestDiD sensitivity analysis is a separate step, not in this script.)

OUTPUTS -> Data/output/tables/script4u/
    <date>_static_twfe.csv           design, spec, outcome, beta, se, p, N, treated counties
    <date>_event_study.csv           design, spec, outcome, event_time, beta, se, p
    <date>_event_study_leads_joint.csv   joint Wald test of the leads
    <date>_cs_att_gt.csv, <date>_cs_aggregated.csv   design, spec, outcome, ATT(g,t) / event-time / overall
    <date>_validation_vs_script4a.csv   max abs difference from script4a for the first-expansion baseline

Usage:  python3 script4u-did-by-treatment.py            full run (300 bootstrap draws; about 30 minutes)
        python3 script4u-did-by-treatment.py --no-cs    TWFE and event studies only (minutes); the CS files are left as they were
        python3 script4u-did-by-treatment.py --quick    20 draws, two outcomes (for testing)
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import sys

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import chi2

from did_cs import run_cs
from script4_treatment import (
    load_panel, OUTCOMES, COVARIATE_ORDER, CENSUS_YEARS, tables_dir, os, date,
)

QUICK = "--quick" in sys.argv
NO_CS = "--no-cs" in sys.argv
N_BOOT = 20 if QUICK else 300
OUT = os.path.join(tables_dir, "script4u" + ("_quick" if QUICK else "")); os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
CONTROLS = COVARIATE_ORDER[:8]
CLIP = 8
OUTCOME_LIST = dict(list(OUTCOMES.items())[:2]) if QUICK else OUTCOMES


def clean(name):
    return (name.replace("%", "pct").replace("-", "_").replace(" ", "_").replace("(", "")
                .replace(")", "").replace("/", "_").replace("+", "p").replace(",", "_"))


def residualize(frame, cols, trend, tol=1e-11, max_iter=500):
    """Remove county intercepts, (optionally) county linear trends and year effects from `cols`
    by alternating projections. Returns a residualized copy of frame[cols]."""
    V = frame[cols].to_numpy(dtype=float).copy()
    g = pd.Series(frame["fips"].to_numpy())
    yr = pd.Series(frame["year"].to_numpy(dtype=float))
    t = yr - yr.groupby(g).transform("mean")           # centred within county
    tt = (t ** 2).groupby(g).transform("sum").replace(0, np.nan)
    for _ in range(max_iter):
        prev = V.copy()
        for j in range(V.shape[1]):
            v = pd.Series(V[:, j])
            v = v - v.groupby(g).transform("mean")                         # county intercept
            if trend:
                slope = ((t * v).groupby(g).transform("sum") / tt).fillna(0.0)
                v = v - slope * t                                           # county linear trend
            v = v - v.groupby(yr).transform("mean")                         # year effect
            V[:, j] = v.to_numpy()
        if np.max(np.abs(V - prev)) < tol:
            break
    return pd.DataFrame(V, columns=cols, index=frame.index)


def fe_ols(frame, y, xcols, trend):
    """OLS of y on xcols after removing county FE, year FE and (if trend) county linear trends.
    SE clustered on state_fips with pyfixest's small-sample convention G/(G-1)*(N-1)/(N-K)."""
    R = residualize(frame, [y] + xcols, trend)
    X, yv = R[xcols].to_numpy(), R[y].to_numpy()
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ yv
    e = yv - X @ b
    cl = frame["state_fips"].to_numpy()
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cl):
        idx = cl == c
        sc = X[idx].T @ e[idx]
        meat += np.outer(sc, sc)
    G, N, K = len(np.unique(cl)), len(yv), X.shape[1]
    Vc = XtXi @ meat @ XtXi * (G / (G - 1)) * ((N - 1) / (N - K))
    se = np.sqrt(np.diag(Vc))
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(np.abs(b / se), G - 1))
    return pd.DataFrame({"beta": b, "se": se, "p": p}, index=xcols), Vc, N


def joint_wald(model, names):
    cn = list(model._coefnames)
    idx = [cn.index(n) for n in names if n in cn]
    if not idx:
        return np.nan, 0, np.nan
    b = np.asarray(model.coef())[idx]
    V = np.asarray(model._vcov)[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.solve(V, b))
    return stat, len(idx), float(1 - chi2.cdf(stat, len(idx)))


df = load_panel().sort_values(["fips", "year"]).reset_index(drop=True)

# ---- baseline characteristics x time polynomial (columns added to df) ----
BASE_YEAR = 2002
BL_VARS = ["log_pop", "median_household_income", "%_hispanic", "%_65_and_older", "unemployment_per100k",
           "cafo_dairy_total", "cafo_dairy_large"]
_bl = df[df["year"] == BASE_YEAR].set_index("fips")[BL_VARS]
_bl = (_bl - _bl.mean()) / _bl.std()                         # standardised so the interactions are well scaled
_t1 = (df["year"] - BASE_YEAR) / 10.0
POLY_COLS = []
for k, v in enumerate(BL_VARS):
    b = df["fips"].map(_bl[v])
    df[f"bl{k}_t1"], df[f"bl{k}_t2"] = b * _t1, b * _t1 ** 2
    POLY_COLS += [f"bl{k}_t1", f"bl{k}_t2"]
PF_SPECS = [("baseline", "fips + year", []),
            ("state x year FE", "fips + state_fips^year", []),
            ("baseline chars x time polynomial", "fips + year", POLY_COLS)]

# ---- sample for the entry design: entry counties + counties that never have a large dairy ----
waves = df[df.year.isin(CENSUS_YEARS)].pivot(index="fips", columns="year", values="cafo_dairy_large")
never_large = set(waves.index[~(waves.fillna(0) > 0).any(axis=1)])
entry_counties = set(df.loc[df["entry_year"].notna(), "fips"].unique())
df_entry = df[df["fips"].isin(never_large | entry_counties)].copy()
print(f"entry design: {len(entry_counties)} entry counties + {len(never_large)} never-large counties "
      f"= {df_entry.fips.nunique()} of {df.fips.nunique()} counties")

DESIGNS = {
    "first expansion": dict(data=df, cohort="add_year", treat="tr_e2_add_absorb"),
    "first entry (clean controls)": dict(data=df_entry, cohort="entry_year", treat="tr_e3_entry_absorb"),
}
for d in DESIGNS.values():
    d["data"] = d["data"].copy()
    d["data"]["t_rel"] = d["data"]["year"] - d["data"][d["cohort"]]

static_rows, es_rows, joint_rows, gt_all, agg_all, fe_checks = [], [], [], [], [], []

for dname, D in DESIGNS.items():
    data, treat, cohort = D["data"], D["treat"], D["cohort"]
    print(f"\n##### {dname}: {data[cohort].notna().groupby(data.fips).first().sum() if False else data.loc[data[cohort].notna(),'fips'].nunique()} treated counties")
    for okey, ocol in OUTCOME_LIST.items():
        # ----------------------------- static TWFE -----------------------------
        for spec, fe, extra in PF_SPECS:
            need = list(dict.fromkeys(["fips", "year", "state_fips", ocol, treat] + CONTROLS + extra))
            sub = data[need].dropna().rename(columns={c: clean(c) for c in need})
            nsw = int((sub.groupby("fips")[clean(treat)].nunique() > 1).sum())
            xc = [clean(treat)] + [clean(c) for c in CONTROLS] + [clean(c) for c in extra]
            f = f"{clean(ocol)} ~ " + " + ".join(xc) + f" | {fe}"
            try:
                m = pf.feols(f, data=sub, vcov={"CRV1": "state_fips"}); t = m.tidy().loc[clean(treat)]
                static_rows.append({"design": dname, "spec": spec, "outcome": okey, "beta": float(t["Estimate"]),
                                    "se_state": float(t["Std. Error"]), "p_state": float(t["Pr(>|t|)"]),
                                    "N": int(m._N), "n_switchers": nsw})
                if spec == "baseline":
                    # internal check: fe_ols without trends must reproduce pyfixest
                    chk, _, _ = fe_ols(sub.reset_index(drop=True), clean(ocol), xc, trend=False)
                    fe_checks.append({"design": dname, "outcome": okey,
                                      "beta_diff": abs(chk.loc[clean(treat), "beta"] - float(t["Estimate"])),
                                      "se_diff": abs(chk.loc[clean(treat), "se"] - float(t["Std. Error"])) / float(t["Std. Error"])})
                    res, _, N_ = fe_ols(sub.reset_index(drop=True), clean(ocol), xc, trend=True)
                    r = res.loc[clean(treat)]
                    static_rows.append({"design": dname, "spec": "county trends", "outcome": okey, "beta": float(r.beta),
                                        "se_state": float(r.se), "p_state": float(r.p), "N": int(N_), "n_switchers": nsw})
            except Exception as e:
                print(f"  static [{dname}|{spec}|{okey}] failed: {type(e).__name__}: {e}")
        # ----------------------------- event study -----------------------------
        def build_es(extra):
            need = list(dict.fromkeys(["fips", "year", "state_fips", ocol, "t_rel"] + CONTROLS + extra))
            e_ = data[need].copy()
            e_["ev"] = e_["t_rel"].clip(-CLIP, CLIP)
            e_.loc[e_["t_rel"].isna(), "ev"] = -1
            return e_.drop(columns=["t_rel"]).dropna().rename(columns={c: clean(c) for c in e_.columns if c != "t_rel"})
        for spec, fe, extra in PF_SPECS:
            try:
                es = build_es(extra)
                f = (f"{clean(ocol)} ~ i(ev, ref=-1) + " + " + ".join(clean(c) for c in CONTROLS + extra) + f" | {fe}")
                m = pf.feols(f, data=es, vcov={"CRV1": "state_fips"}); tid = m.tidy()
                leads = []
                for idx, r in tid.iterrows():
                    if not str(idx).startswith("ev::"):
                        continue
                    e = int(float(str(idx).split("::")[1]))
                    es_rows.append({"design": dname, "spec": spec, "outcome": okey, "event_time": e,
                                    "beta": float(r["Estimate"]), "se_state": float(r["Std. Error"]),
                                    "p_state": float(r["Pr(>|t|)"]), "N": int(m._N)})
                    if e < -1:
                        leads.append(str(idx))
                stat, k, p = joint_wald(m, leads)
                joint_rows.append({"design": dname, "spec": spec, "outcome": okey, "n_leads": k,
                                   "wald_chi2": stat, "joint_p": p, "N": int(m._N)})
            except Exception as e:
                print(f"  event [{dname}|{spec}|{okey}] failed: {type(e).__name__}: {e}")
        # county trends: event-time dummies built by hand, residualized with county trends
        try:
            es = build_es([])
            evs = sorted(int(x) for x in es["ev"].unique() if x != -1)
            D_ = pd.DataFrame({f"ev_{e}": (es["ev"] == e).astype(float) for e in evs}, index=es.index)
            ee = pd.concat([es.reset_index(drop=True), D_.reset_index(drop=True)], axis=1)
            xc = list(D_.columns) + [clean(c) for c in CONTROLS]
            res, Vc, N_ = fe_ols(ee, clean(ocol), xc, trend=True)
            lead_idx = [i for i, e in enumerate(evs) if e < -1]
            bl = res.beta.to_numpy()[lead_idx]; Vl = Vc[np.ix_(lead_idx, lead_idx)]
            stat = float(bl @ np.linalg.solve(Vl, bl))
            for e in evs:
                r = res.loc[f"ev_{e}"]
                es_rows.append({"design": dname, "spec": "county trends", "outcome": okey, "event_time": e,
                                "beta": float(r.beta), "se_state": float(r.se), "p_state": float(r.p), "N": int(N_)})
            joint_rows.append({"design": dname, "spec": "county trends", "outcome": okey, "n_leads": len(lead_idx),
                               "wald_chi2": stat, "joint_p": float(1 - chi2.cdf(stat, len(lead_idx))), "N": int(N_)})
        except Exception as e:
            print(f"  event [{dname}|county trends|{okey}] failed: {type(e).__name__}: {e}")
        # ----------------------------- Callaway-Sant'Anna -----------------------------
        for spec, lags in ([] if NO_CS else [("controls", 0), ("controls + past outcomes", 2)]):
            gt, ag = run_cs(data, ocol, cohort, CONTROLS, n_boot=N_BOOT, outcome_lags=lags)
            if len(gt):
                gt.insert(0, "outcome", okey); gt.insert(0, "spec", spec); gt.insert(0, "design", dname)
                ag.insert(0, "outcome", okey); ag.insert(0, "spec", spec); ag.insert(0, "design", dname)
                gt_all.append(gt); agg_all.append(ag)
        print(f"  done: {okey}", flush=True)

static = pd.DataFrame(static_rows); es_df = pd.DataFrame(es_rows); joint = pd.DataFrame(joint_rows)
outputs = [("static_twfe", static), ("event_study", es_df), ("event_study_leads_joint", joint)]
if not NO_CS:
    gt_df = pd.concat(gt_all, ignore_index=True); agg_df = pd.concat(agg_all, ignore_index=True)
    outputs += [("cs_att_gt", gt_df), ("cs_aggregated", agg_df)]
for name, x in outputs:
    x.to_csv(os.path.join(OUT, f"{TODAY}_{name}.csv"), index=False)

# ---------------------------- validation against script4a (first expansion, baseline) ------------------
val = []
A4 = os.path.join(tables_dir, "script4a")
def newest(pat):
    f = sorted(glob.glob(os.path.join(A4, pat)))
    return pd.read_csv(f[-1]) if f else None
grid, a5, a6 = newest("*_A1_A2_treatment_grid.csv"), newest("*_A5_event_study.csv"), newest("*_A6_cs_aggregated.csv")
mine = "first expansion"
if grid is not None:
    g = grid[(grid.registry == "A2") & (grid.treatment_id == "E2")].set_index("outcome")
    s = static[(static.design == mine) & (static.spec == "baseline")].set_index("outcome")
    common = s.index.intersection(g.index)
    val.append({"check": "static TWFE beta", "max_abs_diff": float((s.loc[common, "beta"] - g.loc[common, "beta"]).abs().max()),
                "n": len(common)})
    val.append({"check": "static TWFE se", "max_abs_diff": float((s.loc[common, "se_state"] - g.loc[common, "se_state"]).abs().max()),
                "n": len(common)})
if a5 is not None:
    a = a5[a5.registry == "A5"][["outcome", "event_time", "beta"]]
    m_ = es_df[(es_df.design == mine) & (es_df.spec == "baseline")].merge(a, on=["outcome", "event_time"], suffixes=("", "_4a"))
    val.append({"check": "event study beta", "max_abs_diff": float((m_.beta - m_.beta_4a).abs().max()), "n": len(m_)})
if a6 is not None and not NO_CS:
    a = a6[(a6.arm == "covariate-adjusted") & (a6.agg_level == "overall_ATT")][["outcome", "att", "se_state"]]
    m_ = agg_df[(agg_df.design == mine) & (agg_df.spec == "controls") & (agg_df.agg_level == "overall_ATT")] \
        .merge(a, on="outcome", suffixes=("", "_4a"))
    val.append({"check": "CS overall ATT", "max_abs_diff": float((m_.att - m_.att_4a).abs().max()), "n": len(m_)})
    val.append({"check": "CS overall ATT bootstrap se", "max_abs_diff": float((m_.se_state - m_.se_state_4a).abs().max()), "n": len(m_)})
fc = pd.DataFrame(fe_checks)
val.append({"check": "fe_ols (no trend) vs pyfixest, static beta", "max_abs_diff": float(fc.beta_diff.max()), "n": len(fc)})
val.append({"check": "fe_ols (no trend) vs pyfixest, static se (RELATIVE diff)", "max_abs_diff": float(fc.se_diff.max()), "n": len(fc)})
val = pd.DataFrame(val)
val.to_csv(os.path.join(OUT, f"{TODAY}_validation_vs_script4a.csv"), index=False)


def st(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


print("\n" + "=" * 100 + "\nVALIDATION: first-expansion baseline vs script4a (should be ~0)\n" + "=" * 100)
print(val.to_string(index=False))
print("\n" + "=" * 100 + "\nSTATIC TWFE   beta (stars)   [county FE + year FE, SE clustered by state]\n" + "=" * 100)
for dname in DESIGNS:
    print(f"\n{dname}")
    for o in OUTCOME_LIST:
        r = static[(static.design == dname) & (static.outcome == o)]
        print(f"  {o[:24]:24s} " + " | ".join(f"{x.spec}: {x.beta:+.3g}{st(x.p_state)} (z{x.beta / x.se_state:+.1f})" for x in r.itertuples())
              + f" | N={int(r.N.iloc[0]):,} switchers={int(r.n_switchers.iloc[0])}")
print("\n" + "=" * 100 + "\nEVENT STUDY: joint test that all leads = 0 (p)\n" + "=" * 100)
print(joint.pivot_table(index=["design", "outcome"], columns="spec", values="joint_p").round(3).to_string())
print("\n" + "=" * 100 + "\nCS OVERALL ATT   att (stars) [bootstrap over states]\n" + "=" * 100)
ov = agg_df[agg_df.agg_level == "overall_ATT"] if not NO_CS else pd.DataFrame(columns=["design", "outcome"])
for dname in DESIGNS:
    print(f"\n{dname}")
    for o in OUTCOME_LIST:
        r = ov[(ov.design == dname) & (ov.outcome == o)]
        print(f"  {o[:24]:24s} " + " | ".join(f"{x.spec}: {x.att:+.3g}{st(x.p_state)} (z{x.att / x.se_state:+.1f}, fb={int(x.n_fallback_cells)})" for x in r.itertuples()))
print(f"\nSaved -> {OUT}")
