#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimators.py -- shared estimators added 2026-10-12 for run 2.

    lpdid()        Local-projections DiD (Dube, Girardi, Jorda & Taylor 2023)
    conley_se()    Spatial HAC (Conley 1999) standard error for one coefficient
                   of a fixed-effects regression
    load_centroids()  county lat/lon from the Census Gazetteer, if present

Imported by script4a (dairy, E2) and script4r (slaughterhouses).
"""
import os
import numpy as np
import pandas as pd
import pyfixest as pf


# =============================================================================
# LP-DiD
# =============================================================================
def lpdid(panel, y, cohort="cohort", unit="fips", time="year", cluster="state_fips",
          horizons=range(-8, 9), controls=None, min_treated=10):
    """
    For each horizon h, one stacked regression

        Y_{i,t+h} - Y_{i,t-1} = beta_h * 1[g_i = t] + delta_t + X_{i,t-1}'gamma + e

    on the sample of
        * NEWLY TREATED units: first-treated in year t (cohort == t)
        * CLEAN CONTROLS: not yet treated at max(t, t+h), or never treated
    stacked over every t that is some cohort's first-treatment year. delta_t is a
    fixed effect for the event year t, so each treated unit is compared only with
    clean controls observed over the same window. Already-treated units never
    enter as controls -- the TWFE problem CS also removes.

    h < -1 are the pre-trend test (long difference from t-1 back to t+h); h = -1
    is zero by construction and skipped. X (optional) is measured at t-1, before
    treatment. SEs clustered on `cluster`.

    `panel` is long: one row per unit-year with columns unit, time, y, cohort
    (NaN = never treated), cluster, and any controls. Returns one row per h.
    """
    controls = list(controls or [])
    wide = panel.pivot_table(index=unit, columns=time, values=y, aggfunc="first")
    coh = panel.groupby(unit)[cohort].first().reindex(wide.index)
    cl = panel.groupby(unit)[cluster].first().reindex(wide.index)
    Xp = (panel.set_index([unit, time])[controls] if controls else None)
    event_years = sorted(coh.dropna().unique())
    out = []
    for h in horizons:
        if h == -1:
            continue
        rows = []
        for t in event_years:
            base, tgt = t - 1, t + h
            if base not in wide.columns or tgt not in wide.columns:
                continue
            treated = coh.index[(coh == t).values]
            clean = coh.index[(coh.isna() | (coh > max(t, tgt))).values]
            dy = (wide[tgt] - wide[base])
            for grp, flag in ((treated, 1), (clean, 0)):
                d = dy.reindex(grp).dropna()
                if d.empty:
                    continue
                f = pd.DataFrame({unit: d.index, "t_event": int(t), "treat": flag,
                                  "dy": d.values, cluster: cl.reindex(d.index).values})
                if controls:
                    xb = Xp.reindex(pd.MultiIndex.from_arrays(
                        [d.index, np.repeat(base, len(d))]))
                    for c in controls:
                        f[c] = xb[c].values
                rows.append(f)
        if not rows:
            continue
        S = pd.concat(rows, ignore_index=True).dropna()
        n_tr = S.loc[S.treat == 1, unit].nunique()
        if n_tr < min_treated or S.treat.nunique() < 2:
            continue
        rhs = " + ".join(["treat"] + controls)
        try:
            m = pf.feols(f"dy ~ {rhs} | t_event", data=S, vcov={"CRV1": cluster})
            r = m.tidy().loc["treat"]
        except Exception as e:
            print(f"      LP-DiD h={h}: {type(e).__name__}: {e}")
            continue
        b, se = float(r["Estimate"]), float(r["Std. Error"])
        out.append({"h": int(h), "beta": b, "se": se, "p": float(r["Pr(>|t|)"]),
                    "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se,
                    "N": int(m._N), "n_treated": int(n_tr),
                    "n_control": int(S.loc[S.treat == 0, unit].nunique()),
                    "n_event_years": int(S.t_event.nunique())})
    return pd.DataFrame(out)


# =============================================================================
# Conley spatial HAC
# =============================================================================
GAZ_DIR = os.path.join("Data", "raw", "gazetteer")


def load_centroids(db_data):
    """County centroids (fips, lat, lon) from the Census Gazetteer national county
    file, placed by hand in Data/raw/gazetteer/ (the census.gov download is
    blocked from the analysis VM). Returns None if the file is absent."""
    d = os.path.join(db_data, "raw", "gazetteer")
    if not os.path.isdir(d):
        return None
    fs = [f for f in os.listdir(d) if f.lower().endswith(".txt") and "count" in f.lower()]
    if not fs:
        return None
    g = pd.read_csv(os.path.join(d, sorted(fs)[-1]), sep="\t", dtype={"GEOID": str})
    g.columns = [c.strip() for c in g.columns]
    return pd.DataFrame({"fips": g["GEOID"].str.zfill(5),
                         "lat": g["INTPTLAT"].astype(float),
                         "lon": g["INTPTLONG"].astype(float)})


def _haversine_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def conley_se(model, data, coef, lat, lon, unit="fips", time="year", cutoff_km=200.0):
    """
    Conley (1999) spatial HAC SE for `coef` in a pyfixest model with absorbed FE.

    Uses the FWL representation: the coefficient's influence function is
    x~_i * u_i, where x~ is the regressor residualised on all other regressors
    and the fixed effects. The meat sums x~_i u_i x~_j u_j over
        * pairs in the same year within `cutoff_km`, Bartlett-weighted by distance
        * all years of the same county (serial correlation, unit-cluster style)
    Same-county-same-year terms are counted once.
    """
    dm = pd.DataFrame(model._X, columns=model._coefnames)       # FE-demeaned design
    u = np.asarray(model._u_hat).ravel()
    j = model._coefnames.index(coef)
    others = [c for c in model._coefnames if c != coef]
    xk = dm[coef].values
    if others:
        Z = dm[others].values
        xk = xk - Z @ np.linalg.lstsq(Z, xk, rcond=None)[0]
    s = xk * u
    # `data` must be exactly the estimation sample (no NaNs, no dropped
    # singletons), row-aligned with lat/lon; the caller checks model._N.
    if len(data) != len(s):
        raise ValueError(f"row mismatch: data {len(data)} vs model {len(s)}")
    units = data[unit].values
    years = data[time].values
    la, lo = np.asarray(lat), np.asarray(lon)
    meat = 0.0
    # spatial, within year
    for yv in np.unique(years):
        k = np.where(years == yv)[0]
        D = _haversine_km(la[k], lo[k])
        W = np.clip(1 - D / cutoff_km, 0, None)
        meat += s[k] @ W @ s[k]
    # serial, within county, excluding the same-year diagonal already counted
    df_ = pd.DataFrame({"u": units, "s": s})
    g = df_.groupby("u")["s"]
    meat += float((g.sum() ** 2).sum() - (g.apply(lambda v: (v ** 2).sum())).sum())
    bread = float(xk @ xk)
    return float(np.sqrt(meat) / bread)


# =============================================================================
# Staggered-DiD triangulation (2026-10-23, D-45): stacked DiD and adapters that put every package's output in one table
# =============================================================================
# Plan: code-audit/plans/2026-10-22_PLAN_estimator-triangulation.md. Each adapter takes a long panel with columns
# unit, time, y, cohort (first treated year; NaN = never treated), cluster, controls, and returns TRI_COLUMNS rows:
# one 'overall' row (the estimator's own post-treatment ATT) and 'event' rows (event time e = year - cohort).
# Event-time estimates are shown for e in [-8, 8]; e = -1 is the reference.
import re
import subprocess
import tempfile
from pathlib import Path

from scipy.stats import norm

TRI_COLUMNS = ["estimator", "kind", "event_time", "est", "se", "p", "ci_lo", "ci_hi"]
DCDH_R_SCRIPT = Path(__file__).resolve().parent / "script4za-dcdh.R"


def _tri_rows(estimator, kind, event_time, est, se):
    """Common output table; p-value and 95% interval from the normal approximation."""
    est, se = np.atleast_1d(np.asarray(est, float)), np.atleast_1d(np.asarray(se, float))
    z = np.divide(est, se, out=np.full_like(est, np.nan), where=se > 0)
    return pd.DataFrame({"estimator": estimator, "kind": kind, "event_time": event_time, "est": est, "se": se,
                         "p": 2 * (1 - norm.cdf(np.abs(z))), "ci_lo": est - 1.96 * se, "ci_hi": est + 1.96 * se})


def stacked_did(panel, y, cohort="cohort", unit="fips", time="year", cluster="state_fips",
                controls=None, pre=8, post=8):
    """Stacked DiD (Cengiz et al. 2019) with corrective weights (Wing, Freedman & Hollingsworth 2024).

    One sub-experiment per cohort c: the treated cohort-c units plus clean controls (never treated, or first
    treated after c + post), years c-pre ... c+post. Sub-experiments are stacked with county-by-stack and
    year-by-stack fixed effects. Treated units have weight 1; controls in stack c have weight
    (treated units in c / all treated units) / (controls in c / all control units), so the control group is
    represented in proportion to the treated. Event version: treated x event-time dummies (reference -1);
    overall version: one treated x post dummy. Controls enter as regressors. SEs clustered on `cluster`.
    """
    controls = list(controls or [])
    stacks = []
    for c in sorted(panel[cohort].dropna().unique()):
        win = panel[(panel[time] >= c - pre) & (panel[time] <= c + post)]
        sub = win[(win[cohort] == c) | win[cohort].isna() | (win[cohort] > c + post)].dropna(subset=[y] + controls).copy()
        sub["treated"] = (sub[cohort] == c).astype(int)
        if sub.treated.nunique() < 2 or (c - 1) not in set(sub[time]):
            continue
        sub["stk"], sub["e"] = int(c), (sub[time] - c).astype(int)
        sub.attrs = {}                       # pd.concat compares .attrs, which can hold DataFrames (load_panel)
        stacks.append(sub)
    if not stacks:
        return pd.DataFrame(columns=TRI_COLUMNS)
    n_t = {s.stk.iloc[0]: s.loc[s.treated == 1, unit].nunique() for s in stacks}
    n_c = {s.stk.iloc[0]: s.loc[s.treated == 0, unit].nunique() for s in stacks}
    q = {c: (n_t[c] / sum(n_t.values())) / (n_c[c] / sum(n_c.values())) for c in n_t}
    S = pd.concat(stacks, ignore_index=True)
    S["w"] = np.where(S.treated == 1, 1.0, S.stk.map(q))
    S["uc"], S["tc"] = S[unit].astype(str) + "_" + S.stk.astype(str), S[time].astype(int).astype(str) + "_" + S.stk.astype(str)
    S["post_treated"] = ((S.treated == 1) & (S.e >= 0)).astype(int)
    rhs = "".join(f" + {c}" for c in controls)
    vc = {"CRV1": cluster}
    over = pf.feols(f"{y} ~ post_treated{rhs} | uc + tc", data=S, weights="w", vcov=vc).tidy().loc["post_treated"]
    ev = pf.feols(f"{y} ~ i(e, treated, ref=-1){rhs} | uc + tc", data=S, weights="w", vcov=vc).tidy()
    ev = ev[[bool(re.search(r"e::-?\d", str(i))) for i in ev.index]]
    e_t = [float(re.search(r"e::(-?\d+)", str(i)).group(1)) for i in ev.index]
    return pd.concat([_tri_rows("Stacked DiD", "overall", np.nan, over["Estimate"], over["Std. Error"]),
                      _tri_rows("Stacked DiD", "event", e_t, ev["Estimate"].values, ev["Std. Error"].values)], ignore_index=True)


def did2s_estimates(panel, y, treat, cohort="cohort", unit="fips", time="year", cluster="state_fips",
                    controls=None, clip=8):
    """Imputation (Gardner two-step; point estimates equal Borusyak-Jaravel-Spiess) via pyfixest.did2s.

    First stage on untreated observations only: y ~ controls | unit + time. Second stage: the residualised outcome
    on the treatment dummy (overall) or on event-time dummies (event, e clipped to [-clip, clip], reference -1).
    """
    controls = list(controls or [])
    d = panel.copy()
    if int(((d[treat] == 0).groupby(d[unit]).sum() < 2).sum()):
        raise ValueError("did2s needs at least 2 untreated observations per unit (pyfixest drops first-stage singletons "
                         "and its variance step then fails); drop those units first")
    d["rel_c"] = (d[time] - d[cohort]).clip(-clip, clip).fillna(np.inf)
    first = "~ " + (" + ".join(controls) if controls else "0") + f" | {unit} + {time}"
    over = pf.did2s(d, yname=y, first_stage=first, second_stage=f"~ {treat}", treatment=treat, cluster=cluster).tidy().loc[treat]
    ev = pf.did2s(d, yname=y, first_stage=first, second_stage="~ i(rel_c, ref=-1.0)", treatment=treat, cluster=cluster).tidy()
    ev = ev[[str(i).startswith("rel_c::") and not str(i).endswith("inf") for i in ev.index]]
    e_t = [float(str(i).split("::")[1]) for i in ev.index]
    return pd.concat([_tri_rows("Imputation (did2s)", "overall", np.nan, over["Estimate"], over["Std. Error"]),
                      _tri_rows("Imputation (did2s)", "event", e_t, ev["Estimate"].values, ev["Std. Error"].values)], ignore_index=True)


def saturated_estimates(panel, y, cohort="cohort", unit="fips", time="year", cluster="state_fips",
                        controls=None, max_e=8):
    """Sun-Abraham interaction-weighted estimator via pyfixest's saturated event study (beta in pyfixest 0.60).

    pyfixest fits the cohort x event-time coefficients. Its own aggregation fails when cohorts cover different event-time
    ranges (KeyError) and it has no overall ATT, so both are built here as weighted averages of the cohort x period
    coefficients: event time e = cohorts present at e, weights = their share of the observations at e; overall = all
    post-treatment cells, weights = share of treated observations. SEs use the full coefficient covariance. This reads
    pyfixest 0.60.0 internals (_beta_hat, _vcov, _coefnames, _data, _res_cohort_eventtime_dict);
    tests/test_triangulation.py checks it against a known truth.
    """
    from pyfixest.did.saturated_twfe import _compute_lincomb_stats
    controls = list(controls or [])
    d = panel.copy()
    d["g0"] = d[cohort].fillna(0)
    es = pf.event_study(d, yname=y, idname=unit, tname=time, gname="g0", cluster=cluster, estimator="saturated",
                        xfml=("~ " + " + ".join(controls)) if controls else None)
    mod = es.mod if hasattr(es, "mod") else es       # pyfixest returns the fitted model itself or a wrapper around it
    names, beta, vcov = mod._coefnames, mod._beta_hat, mod._vcov
    counts = mod._data[mod._data[mod._gname] > 0].groupby([mod._gname, "rel_time"]).size().to_dict()
    cohort_keys = {key: int(key.replace("cohort_dummy_", "")) for key in mod._res_cohort_eventtime_dict}
    periods = sorted({p for info in mod._res_cohort_eventtime_dict.values() for p in info["time"].tolist()})

    def cell_weights(period_filter):
        w = np.zeros(len(beta))
        for key, g_int in cohort_keys.items():
            for period in mod._res_cohort_eventtime_dict[key]["time"].tolist():
                if not period_filter(period) or not counts.get((g_int, period), 0):
                    continue
                for i in (i for i, n in enumerate(names) if re.search(rf"^(?:.+)::{period}:(?:.+)::{key}$", n)):
                    w[i] = counts[(g_int, period)]
        return w

    ev_e, ev_b, ev_se = [], [], []
    for period in periods:
        if -max_e <= period <= max_e:
            w = cell_weights(lambda p, e=period: p == e)
            if w.sum() > 0:
                s = _compute_lincomb_stats(R=w / w.sum(), coefs=beta, vcov=vcov)
                ev_e.append(float(period)); ev_b.append(s["Estimate"]); ev_se.append(s["Std. Error"])
    w = cell_weights(lambda p: p >= 0)
    over = _compute_lincomb_stats(R=w / w.sum(), coefs=beta, vcov=vcov)
    return pd.concat([_tri_rows("Sun-Abraham", "overall", np.nan, over["Estimate"], over["Std. Error"]),
                      _tri_rows("Sun-Abraham", "event", ev_e, ev_b, ev_se)], ignore_index=True)


def lpdid_pooled(panel, y, cohort="cohort", unit="fips", time="year", cluster="state_fips", controls=None, post=8):
    """Pooled LP-DiD ATT over the horizons h = 0..post that each cohort actually reaches.

    The stacked long differences Y_{i,t+h} - Y_{i,t-1} of lpdid() for h = 0..post go into ONE regression on the newly
    treated dummy with event-year x horizon fixed effects and the controls at t-1; each treated (unit, horizon) pair has
    equal weight and SEs are clustered on `cluster` (units repeat across horizons). Unlike pyfixest.lpdid(att=True) it does
    not drop cohorts that cannot reach the full window.
    """
    controls = list(controls or [])
    wide = panel.pivot_table(index=unit, columns=time, values=y, aggfunc="first")
    coh = panel.groupby(unit)[cohort].first().reindex(wide.index)
    cl = panel.groupby(unit)[cluster].first().reindex(wide.index)
    Xp = panel.set_index([unit, time])[controls] if controls else None
    rows = []
    for h in range(0, post + 1):
        for t in sorted(coh.dropna().unique()):
            base, tgt = t - 1, t + h
            if base not in wide.columns or tgt not in wide.columns:
                continue
            dy = wide[tgt] - wide[base]
            for grp, flag in ((coh.index[(coh == t).values], 1), (coh.index[(coh.isna() | (coh > max(t, tgt))).values], 0)):
                d = dy.reindex(grp).dropna()
                if d.empty:
                    continue
                f = pd.DataFrame({unit: d.index, "cell": f"{int(t)}_{h}", "treat": flag, "dy": d.values, cluster: cl.reindex(d.index).values})
                for c in controls:
                    f[c] = Xp.reindex(pd.MultiIndex.from_arrays([d.index, np.repeat(base, len(d))]))[c].values
                rows.append(f)
    S = pd.concat(rows, ignore_index=True).dropna()
    r = pf.feols("dy ~ treat" + "".join(f" + {c}" for c in controls) + " | cell", data=S, vcov={"CRV1": cluster}).tidy().loc["treat"]
    return float(r["Estimate"]), float(r["Std. Error"])


def lpdid_estimates(panel, y, cohort="cohort", unit="fips", time="year", cluster="state_fips", controls=None, pre=8, post=8):
    """LP-DiD (Dube et al.): event paths from lpdid() (controls at t-1); overall from lpdid_pooled() over h = 0..post."""
    controls = list(controls or [])
    r = lpdid(panel, y, cohort=cohort, unit=unit, time=time, cluster=cluster, horizons=range(-pre, post + 1), controls=controls)
    b, se = lpdid_pooled(panel, y, cohort, unit, time, cluster, controls, post)
    return pd.concat([_tri_rows("LP-DiD", "overall", np.nan, b, se),
                      _tri_rows("LP-DiD", "event", r["h"].astype(float).values, r["beta"].values, r["se"].values)], ignore_index=True)


def dcdh_estimates(panel, y, treat, unit="fips", time="year", cluster="state_fips", controls=None, effects=9, placebos=7):
    """de Chaisemartin-D'Haultfoeuille DID_l through the authors' R package DIDmultiplegtDYN (needs Rscript, polars, the package).

    Effect_l is event time l-1 (Effect_1 = the period treatment switches on); Placebo_l is event time -(l+1);
    overall = the package's average total effect over effects 1..`effects`. Controls via the package's `controls` option.
    """
    controls = list(controls or [])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        panel[[unit, time, y, treat, cluster] + controls].to_csv(tmp / "in.csv", index=False)
        subprocess.run(["Rscript", str(DCDH_R_SCRIPT), str(tmp), unit, time, y, treat, cluster, str(effects), str(placebos),
                        ",".join(controls)], check=True, capture_output=True, text=True)
        eff, plc, ate = (pd.read_csv(tmp / f"{n}.csv") for n in ("effects", "placebos", "ate"))
    ev = [_tri_rows("de Chaisemartin-D'Haultfoeuille", "event", np.arange(len(eff)) * 1.0, eff["Estimate"].values, eff["SE"].values)]
    if len(plc):
        ev.append(_tri_rows("de Chaisemartin-D'Haultfoeuille", "event", -(np.arange(len(plc)) + 2.0), plc["Estimate"].values, plc["SE"].values))
    return pd.concat([_tri_rows("de Chaisemartin-D'Haultfoeuille", "overall", np.nan, ate["Estimate"].iloc[0], ate["SE"].iloc[0])] + ev,
                     ignore_index=True)
