#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
estimators.py -- every shared estimator and DiD helper in one module (sections below).

    LP-DiD, Conley             lpdid(), conley_se(), load_centroids()                          (added 2026-10-12 for run 2)
    Staggered-DiD triangulation stacked_did(), did2s_estimates(), saturated_estimates(), lpdid_pooled(), lpdid_estimates(),
                                dcdh_estimates() (R/dcdh.R)                                      (D-45, 2026-10-23)
    DiD helpers                 clean, stars, joint_wald, residualize, fe_ols, event_study_frame   (was did_tools.py)
    Callaway and Sant'Anna      att_gt_table(), base_period_covariates(), run_cs()                  (was did_cs.py; D-44)

Merged on 2026-10-25 (Stage C of the script consolidation); the code of did_tools.py and did_cs.py was moved unchanged.
"""
import os
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import chi2


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
DCDH_R_SCRIPT = Path(__file__).resolve().parent / "R" / "dcdh.R"


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


def dcdh_estimates(panel, y, treat, unit="fips", time="year", cluster="state_fips", controls=None, effects=9, placebos=7,
                   normalized=False):
    """de Chaisemartin-D'Haultfoeuille DID_l through the authors' R package DIDmultiplegtDYN (needs Rscript, polars, the package).

    Effect_l is event time l-1 (Effect_1 = the period treatment switches on); Placebo_l is event time -(l+1);
    overall = the package's average total effect over effects 1..`effects`. Controls via the package's `controls` option.
    normalized=True gives effects per one unit of treatment (for a dose or log-count treatment); the default effects are
    those of the whole treatment change, which equals one unit for a 0/1 treatment.
    """
    controls = list(controls or [])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        panel[[unit, time, y, treat, cluster] + controls].to_csv(tmp / "in.csv", index=False)
        subprocess.run(["Rscript", str(DCDH_R_SCRIPT), str(tmp), unit, time, y, treat, cluster, str(effects), str(placebos),
                        ",".join(controls), "1" if normalized else "0"], check=True, capture_output=True, text=True)
        eff, plc, ate = (pd.read_csv(tmp / f"{n}.csv") for n in ("effects", "placebos", "ate"))
    ev = [_tri_rows("de Chaisemartin-D'Haultfoeuille", "event", np.arange(len(eff)) * 1.0, eff["Estimate"].values, eff["SE"].values)]
    if len(plc):
        ev.append(_tri_rows("de Chaisemartin-D'Haultfoeuille", "event", -(np.arange(len(plc)) + 2.0), plc["Estimate"].values, plc["SE"].values))
    return pd.concat([_tri_rows("de Chaisemartin-D'Haultfoeuille", "overall", np.nan, ate["Estimate"].iloc[0], ate["SE"].iloc[0])] + ev,
                     ignore_index=True)


# =============================================================================
# DiD helpers (formerly did_tools.py)
# did_tools.py -- small helpers shared by the stage-4 DiD scripts (4t, 4u, 4w, 4x, 4y).
#
# clean              turn a column name into a formula-safe name
# stars              significance stars for printing
# joint_wald         Wald chi-square that a set of model coefficients are all zero (clustered covariance)
# residualize        remove county FE, year FE and optional county linear trends by alternating projections
# fe_ols             OLS after residualize, SE clustered by state (used for the county-trend specification)
# event_study_frame  the estimation sample for a TWFE event study (see METHODS.md section 2)
# =============================================================================


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


# =============================================================================
# Callaway and Sant'Anna (formerly did_cs.py)
# did_cs.py -- Callaway and Sant'Anna staggered DiD, covariate-adjusted, with a bootstrap over states.
#
# run_cs() estimates ATT(g,t) cells and the event-time and overall ATT; att_gt_table() and base_period_covariates() are the building blocks.
# script4a designs imports the building blocks; script4a cohorts and species call run_cs().
# Method:  METHODS.md section 2. Optional outcome_lags adds the outcome's own past levels to the covariates.
# =============================================================================


def att_gt_table(wide, cohort_of, cohorts, years, keep_fips=None,
                 Xbase=None, use_cov=False, min_n=10):
    """All ATT(g,t) for one outcome. `wide` is fips x year. Unchanged from script4a.

    use_cov=False  unconditional.   use_cov=True  outcome-regression adjusted: fit dY = X'b on the
    clean comparison group only, predict each treated county's counterfactual change,
    ATT = mean(dY treated) - mean(predicted). X is measured at the EARLIER of year t and the base year g-1
    (D-44): at g-1 for every post-treatment cell, at year t for a pre-treatment cell before the base year,
    as in R `did`, `csdid` and `moderndid`. Cells too thin to fit, or with no covariates for that year,
    fall back to the unconditional estimate and are flagged fallback=1.
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
        for t in years:
            if t == base or t not in wide.columns:
                continue
            Xb = Xbase.get(min(t, base)) if (use_cov and Xbase) else None
            ctrl = idx[((coh.isna()) | (coh > max(t, g))).values]
            d_all = wide[t] - wide[base]
            dt, dc = d_all.reindex(treated).dropna(), d_all.reindex(ctrl).dropna()
            if len(dt) < min_n or len(dc) < min_n:
                continue
            rec = {"cohort": int(g), "year": int(t), "event_time": int(t - g)}

            if Xb is None:
                out.append({**rec, "att": float(dt.mean() - dc.mean()),
                            "n_treated": int(len(dt)), "n_control": int(len(dc)),
                            "fallback": int(use_cov)})
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
    """Covariates for every year in `d`, standardised within that year, keyed by year.

    att_gt_table picks the year min(t, g-1) for each cell (D-44), so the base years g-1 and every earlier year are
    needed; `cohorts` is kept for the callers' signature. outcome_lags = 0 gives the covariates only.
    outcome_lags = k > 0 adds the outcome's own levels in years y, y-1, ..., y-k+1 (from `wide`, fips x year)
    as extra columns, where y is the covariate year.
    """
    Xbase = {}
    for b in sorted(d.year.unique()):
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
