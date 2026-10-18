#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
did_cs.py  --  Callaway & Sant'Anna (2021) staggered DiD, covariate-adjusted, shared module.

Added 2026-10-17 for script4u. The two core functions (att_gt_table, base_period_covariates) are
copied from script4a-design-estimates.py without change to the estimator, so that
script4u reproduces script4a's E2 results exactly (it asserts this). script4a still holds its own
copy: merging the two is a planned clean-up, not done yet.

WHAT IS ESTIMATED
-----------------
Cohort g = the first census wave at which a county is treated. For each cohort g and calendar year t
    ATT(g,t) = E[ Y_t - Y_{g-1} | cohort g ]  -  E[ Y_t - Y_{g-1} | comparison counties ]
Comparison counties = never treated, or not yet treated by year max(t, g). Counties already treated
are never comparisons.
Covariate adjustment (outcome regression): fit  dY = X'b  on the comparison counties only, with X
measured in the BASE YEAR g-1 (so it is pre-treatment by construction), then
    ATT_adj(g,t) = mean over cohort-g counties of [ dY_i - X_i'b ].
Optional `outcome_lags`: also put the outcome's own levels in years g-1 ... g-lags into X, so
comparison counties are matched on their pre-treatment outcome path.
Aggregation: event time e = t - g ->  ATT(e) = weighted mean of ATT(g, g+e), weights = number of
treated counties; overall ATT = weighted mean over all post-treatment cells (t >= g).
Inference: nonparametric bootstrap resampling whole STATES with replacement (state clustering).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def att_gt_table(wide, cohort_of, cohorts, years, keep_fips=None,
                 Xbase=None, use_cov=False, min_n=10):
    """All ATT(g,t) for one outcome. `wide` is fips x year. Unchanged from script4a.

    use_cov=False  unconditional.   use_cov=True  outcome-regression adjusted: fit dY = X'b on the
    clean comparison group only, X measured at base year g-1, predict each treated county's
    counterfactual change, ATT = mean(dY treated) - mean(predicted). Cells too thin to fit fall back
    to the unconditional estimate and are flagged fallback=1.
    """
    out = []
    idx = wide.index if keep_fips is None else wide.index.intersection(keep_fips)
    coh = cohort_of.reindex(idx)
    for g in cohorts:
        base = g - 1
        if base not in wide.columns:
            continue
        treated = idx[(coh == g).values]
        if len(treated) < min_n:
            continue
        Xb = Xbase.get(base) if (use_cov and Xbase) else None
        for t in years:
            if t == base or t not in wide.columns:
                continue
            ctrl = idx[((coh.isna()) | (coh > max(t, g))).values]
            d_all = wide[t] - wide[base]
            dt, dc = d_all.reindex(treated).dropna(), d_all.reindex(ctrl).dropna()
            if len(dt) < min_n or len(dc) < min_n:
                continue
            rec = {"cohort": int(g), "year": int(t), "event_time": int(t - g)}

            if Xb is None:
                out.append({**rec, "att": float(dt.mean() - dc.mean()),
                            "n_treated": int(len(dt)), "n_control": int(len(dc)),
                            "fallback": 0})
                continue

            Xc = Xb.reindex(dc.index).dropna()
            Xt = Xb.reindex(dt.index).dropna()
            if len(Xc) < max(min_n, Xb.shape[1] + 5) or len(Xt) < min_n:
                out.append({**rec, "att": float(dt.mean() - dc.mean()),
                            "n_treated": int(len(dt)), "n_control": int(len(dc)),
                            "fallback": 1})
                continue
            yc = dc.reindex(Xc.index).values
            A = np.column_stack([np.ones(len(Xc)), Xc.values])
            try:
                b, *_ = np.linalg.lstsq(A, yc, rcond=None)
            except np.linalg.LinAlgError:
                continue
            pred = np.column_stack([np.ones(len(Xt)), Xt.values]) @ b
            out.append({**rec,
                        "att": float(dt.reindex(Xt.index).mean() - pred.mean()),
                        "n_treated": int(len(Xt)), "n_control": int(len(Xc)),
                        "fallback": 0})
    return out


def base_period_covariates(d, cohorts, xcols, wide=None, outcome_lags=0):
    """Covariates at each candidate base year g-1, standardised within that year.

    outcome_lags = 0 reproduces script4a. outcome_lags = k > 0 adds the outcome's own levels in
    years g-1, g-2, ..., g-k (from `wide`, fips x year) as extra columns.
    """
    Xbase = {}
    for g in cohorts:
        b = g - 1
        sub = d[d.year == b].set_index("fips")[xcols]
        if outcome_lags and wide is not None:
            for k in range(outcome_lags):
                if (b - k) in wide.columns:
                    sub = sub.join(wide[b - k].rename(f"_ylag{k}"), how="left")
        if len(sub):
            sd = sub.std().replace(0, np.nan)
            Xbase[b] = ((sub - sub.mean()) / sd).dropna(axis=1, how="all")
    return Xbase


def run_cs(df, ocol, cohort_col, xcols, n_boot=300, seed=20260928, outcome_lags=0):
    """Covariate-adjusted CS for one outcome, bootstrap over states.

    Mirrors the structure of script4a's A6 loop (same seed, same draws, same SE definition).
    Returns (att_gt DataFrame, aggregated DataFrame with event-time rows and one overall row).
    """
    cohorts = sorted(df.loc[df[cohort_col].notna(), cohort_col].unique())
    d = df[["fips", "year", "state_fips", cohort_col, ocol] + xcols].dropna(subset=[ocol])
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide = d.pivot_table(index="fips", columns="year", values=ocol, aggfunc="first")
    cohort_of = d.groupby("fips")[cohort_col].first()
    state_of = d.groupby("fips")["state_fips"].first()
    years = sorted(d["year"].unique())
    Xbase = base_period_covariates(d, cohorts, xcols, wide=wide, outcome_lags=outcome_lags)

    states = state_of.dropna().unique()
    by_state = {s: state_of.index[state_of == s] for s in states}
    pt = att_gt_table(wide, cohort_of, cohorts, years, Xbase=Xbase, use_cov=True)
    if not pt:
        return pd.DataFrame(), pd.DataFrame()

    keys = [(r["cohort"], r["year"]) for r in pt]
    boot, boot_overall, boot_e = {k: [] for k in keys}, [], {}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        draw = rng.choice(states, size=len(states), replace=True)
        fips_b = pd.Index(np.concatenate([by_state[s].values for s in draw]))
        pb = att_gt_table(wide, cohort_of, cohorts, years,
                          keep_fips=pd.Index(pd.unique(fips_b)), Xbase=Xbase, use_cov=True)
        m = {(r["cohort"], r["year"]): r["att"] for r in pb}
        for k in keys:
            if k in m:
                boot[k].append(m[k])
        pbd = pd.DataFrame(pb)
        if len(pbd):
            for e, g_ in pbd.groupby("event_time"):
                boot_e.setdefault(int(e), []).append(
                    float(np.average(g_["att"], weights=g_["n_treated"].astype(float))))
        post_b = [r for r in pb if r["event_time"] >= 0]
        if post_b:
            w = np.array([r["n_treated"] for r in post_b], float)
            boot_overall.append(float(np.average([r["att"] for r in post_b], weights=w)))

    def z_p(att, se):
        return (float(2 * (1 - norm.cdf(abs(att / se)))) if se and se > 0 else np.nan)

    gt_rows = []
    for r in pt:
        b = boot[(r["cohort"], r["year"])]
        se = float(np.std(b, ddof=1)) if len(b) > 10 else np.nan
        gt_rows.append({**r, "se_state": se, "p_state": z_p(r["att"], se)})

    pdf = pd.DataFrame(pt)
    agg = []
    for e, grp in pdf.groupby("event_time"):
        att_e = float(np.average(grp["att"], weights=grp["n_treated"].astype(float)))
        be = boot_e.get(int(e), [])
        se_e = float(np.std(be, ddof=1)) if len(be) > 10 else np.nan
        agg.append({"agg_level": "event_time", "event_time": int(e), "att": att_e, "se_state": se_e,
                    "p_state": z_p(att_e, se_e), "n_cohorts": int(grp["cohort"].nunique()),
                    "n_treated": int(grp["n_treated"].sum()), "n_fallback_cells": int(grp["fallback"].sum())})
    post = pdf[pdf.event_time >= 0]
    if len(post):
        overall = float(np.average(post["att"], weights=post["n_treated"].astype(float)))
        se_o = float(np.std(boot_overall, ddof=1)) if len(boot_overall) > 10 else np.nan
        agg.append({"agg_level": "overall_ATT", "event_time": np.nan, "att": overall, "se_state": se_o,
                    "p_state": z_p(overall, se_o), "n_cohorts": int(post["cohort"].nunique()),
                    "n_treated": int(post["n_treated"].sum()), "n_fallback_cells": int(post["fallback"].sum())})
    return pd.DataFrame(gt_rows), pd.DataFrame(agg)
