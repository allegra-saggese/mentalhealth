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
