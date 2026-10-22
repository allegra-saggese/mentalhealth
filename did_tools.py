#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
did_tools.py -- small helpers shared by the stage-4 DiD scripts (4t, 4u, 4w, 4x, 4y).

clean              turn a column name into a formula-safe name
stars              significance stars for printing
joint_wald         Wald chi-square that a set of model coefficients are all zero (clustered covariance)
residualize        remove county FE, year FE and optional county linear trends by alternating projections
fe_ols             OLS after residualize, SE clustered by state (used for the county-trend specification)
event_study_frame  the estimation sample for a TWFE event study (see METHODS.md section 2)
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2


def clean(name):
    return (name.replace("%", "pct").replace("-", "_").replace(" ", "_").replace("(", "")
                .replace(")", "").replace("/", "_").replace("+", "p").replace(",", "_"))

def stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def joint_wald(model, names):
    """Wald chi-square that the named coefficients are all zero (state-clustered covariance)."""
    cn = list(model._coefnames)
    idx = [cn.index(n) for n in names if n in cn]
    if not idx:
        return np.nan, 0, np.nan
    b = np.asarray(model.coef())[idx]
    V = np.asarray(model._vcov)[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.solve(V, b))
    return stat, len(idx), float(1 - chi2.cdf(stat, len(idx)))

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

def event_study_frame(data, ocol, controls, extra=(), clip=8):
    """Sample for the event study: event time e = t_rel clipped to [-clip, clip]; never-treated counties
    (t_rel missing) sit in the reference category e = -1; rows with any missing variable are dropped;
    columns are renamed with clean()."""
    need = list(dict.fromkeys(["fips", "year", "state_fips", ocol, "t_rel"] + list(controls) + list(extra)))
    e_ = data[need].copy()
    e_["ev"] = e_["t_rel"].clip(-clip, clip)
    e_.loc[e_["t_rel"].isna(), "ev"] = -1
    return e_.drop(columns=["t_rel"]).dropna().rename(columns={c: clean(c) for c in e_.columns if c != "t_rel"})

