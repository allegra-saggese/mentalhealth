#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4g-spillovers.py -- spatial spillover checks on the reported results, five sections (D-58; plan 2026-10-29).

    moran        Moran's I of the A2 residuals (first expansion E2) and of the raw outcome, county cross-section by year
    donut        TWFE and covariate-adjusted Callaway-Sant'Anna after dropping never-treated counties within 50 and 100 km of a treated county
    conley       Conley spatial standard errors (100, 250, 500 km) for the reported A2 cells
    division-fe  the A2 cells with Census division x year fixed effects in place of year fixed effects
    durbin       spatial Durbin panel model: own, spillover (indirect) and total effect of each treatment (spatial.py)

Reads:   panel via designs.load_designs() / script4_treatment.load_panel(); county outlines Data/clean/geo/*_us_counties.geojson (public plotly
         file, fetched once; the centroids come from it, spatial.county_centroids); optional county-to-commuting-zone crosswalk
         Data/raw/cz/*.csv or Data/clean/cz/*.csv (columns fips, cz): without it the commuting-zone weights W_cz are skipped and said to be.
Writes:  Data/output/tables/<date>_4g-<section>_<stem>.csv (naming: packages.py)
Run:     python3 script4g-spillovers.py [--only moran donut conley division-fe durbin] [--quick]    (--quick: one outcome, 20 bootstrap draws)
Method:  decision D-58 (the specifications were fixed in quality_reports/plans before any result was read); spatial.py for the weights and
         the Durbin model; METHODS.md for TWFE and Callaway-Sant'Anna. Nothing here changes a reported number: these are robustness checks.
Limits:  W covers rural panel counties only (urban neighbours are outside the panel); counties without an outline (15) are left out.
         The donut drops never-treated counties only; not-yet-treated comparison counties stay in the Callaway-Sant'Anna comparison group.
         The Durbin model is a TWFE-type model on a balanced panel (the outcome's descriptive window), not robust to staggered timing.
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import glob
import json
import os
from datetime import date

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import norm

import spatial as sp
from designs import load_designs, usable_sample, OUTCOME_WINDOWS
from estimators import clean, conley_se, run_cs
from packages import db_data, out_file, out_glob, tables_dir
from script4_treatment import OUTCOMES, COVARIATE_ORDER, CORE_TREATMENTS, SPEC_EXTRA_REGRESSORS, REPORT_TREATMENTS

TODAY = date.today().strftime("%Y-%m-%d")
CONTROLS = COVARIATE_ORDER[:8]
TREAT_COL = {**CORE_TREATMENTS, "E2": "tr_e2_add_absorb"}
SECTIONS = ["moran", "donut", "conley", "division-fe", "durbin"]
DIST_KM = 100                                  # W_dist: centroids within this distance (D-58)
DONUT_KM = (50, 100)                           # never-treated counties within these distances of a treated county are dropped (D-58)
CONLEY_KM = (100, 250, 500)                    # Bartlett cut-offs (D-58)
CONFIRMATORY = [(o, t) for o in ("Poor MH Days", "Frequent Mental Distress") for t in REPORT_TREATMENTS]   # D-42 family
CS_SEED, SEED = 20260928, 20261029

# Census divisions by state FIPS (the nine U.S. Census Bureau divisions)
DIVISION = {}
for _name, _states in {
        "New England": "09 23 25 33 44 50", "Middle Atlantic": "34 36 42", "East North Central": "17 18 26 39 55",
        "West North Central": "19 20 27 29 31 38 46", "South Atlantic": "10 11 12 13 24 37 45 51 54",
        "East South Central": "01 21 28 47", "West South Central": "05 22 40 48",
        "Mountain": "04 08 16 30 32 35 49 56", "Pacific": "02 06 15 41 53"}.items():
    DIVISION.update({s: _name for s in _states.split()})
assert len(DIVISION) == 51, "the nine divisions must cover the 50 states and DC"


# ------------------------------------------------------------------------------------------------------------- setup
def load_geometry(df):
    """Centroids (fips, lat, lon) of the panel counties from the cached county outlines, and the distance matrix among them."""
    files = sorted(glob.glob(os.path.join(db_data, "clean", "geo", "*_us_counties.geojson")))
    if not files:
        raise FileNotFoundError("no Data/clean/geo/*_us_counties.geojson: fetch the public plotly county outlines once (see this script's header)")
    cent = sp.county_centroids(json.load(open(files[-1])))
    fips = sorted(set(df["fips"]))
    have = [f for f in fips if f in cent]
    print(f"geometry: {os.path.basename(files[-1])}; {len(have):,} of {len(fips):,} panel counties have an outline "
          f"({len(fips) - len(have)} left out: {[f for f in fips if f not in cent][:6]}...)")
    geo = pd.DataFrame({"fips": have, "lat": [cent[f][0] for f in have], "lon": [cent[f][1] for f in have]})
    return geo, sp.haversine_km(geo.lat.values, geo.lon.values)


def load_cz():
    """County-to-commuting-zone crosswalk if the file has been placed (fips, cz); None otherwise."""
    for d in ("raw", "clean"):
        files = sorted(glob.glob(os.path.join(db_data, d, "cz", "*.csv")))
        if files:
            x = pd.read_csv(files[-1], dtype=str)
            return x.assign(fips=x["fips"].str.zfill(5))[["fips", "cz"]]
    return None


def a2_sample(df, geo, ocol, tcol, extra):
    """Complete-case A2 sample (outcome, treatment, extras, the control set), with centroids and Census division, columns renamed by clean()."""
    need = ["fips", "year", "state_fips", ocol, tcol] + list(extra) + CONTROLS
    sub = df[list(dict.fromkeys(need))].dropna().merge(geo, on="fips", how="inner")
    sub["division"] = sub["state_fips"].map(DIVISION)
    sub = sub.rename(columns={c: clean(c) for c in need if c not in ("fips", "year", "state_fips")}).reset_index(drop=True)
    return sub, {c: clean(c) for c in need}


def formula(mp, ocol, tcol, extra, fe="fips + year"):
    return f"{mp[ocol]} ~ " + " + ".join([mp[tcol]] + [mp[x] for x in extra] + [mp[c] for c in CONTROLS]) + f" | {fe}"


def fit_twfe(sub, fml, vcov="state", keep_singletons=False):
    """pyfixest TWFE. Default drops singleton fixed-effect rows exactly as script4a's A2 grid does (so the baseline reproduces it);
    keep_singletons=True keeps every row, which the Conley and Moran code need so that residuals and data stay row-aligned."""
    return pf.feols(fml, data=sub, vcov={"CRV1": "state_fips"} if vcov == "state" else vcov, fixef_rm="none" if keep_singletons else "singleton")


def coef_row(m, name):
    t = m.tidy().loc[name]
    return float(t["Estimate"]), float(t["Std. Error"]), float(t["Pr(>|t|)"])


def tags(section, quick):
    return f"4g-{section}" + ("-quick" if quick else "")


def baseline_grid():
    """The reported A2 cells from script4a (beta, se_state, N), for the self-checks."""
    f = out_glob(tables_dir, "4a-designs", "*_A1_A2_treatment_grid.csv")
    g = pd.read_csv(f[-1]) if f else None
    return g[g.registry == "A2"] if g is not None else None


# ------------------------------------------------------------------------------------------------------------- moran
def section_moran(df, geo, D, quick):
    rng = np.random.default_rng(SEED)
    rows = []
    idx = {f: i for i, f in enumerate(geo.fips)}
    for okey, ocol in list(OUTCOMES.items())[: 1 if quick else None]:
        tcol, extra = TREAT_COL["E2"], []
        sub, mp = a2_sample(df, geo, ocol, tcol, extra)
        m = fit_twfe(sub, formula(mp, ocol, tcol, extra), keep_singletons=True)
        if m._N != len(sub):
            print(f"  [{okey}] row mismatch, skipped"); continue
        sub["resid"] = np.asarray(m.resid()).ravel()
        for series, col in (("A2 residual (E2)", "resid"), ("raw outcome", mp[ocol])):
            for y, g in sub.groupby("year"):
                ii = np.array([idx[f] for f in g.fips])
                A = (D[np.ix_(ii, ii)] <= DIST_KM).astype(float)
                np.fill_diagonal(A, 0.0)
                keep = A.sum(axis=1) > 0
                if keep.sum() < 50:
                    continue
                W = sp.row_standardise(A[np.ix_(keep, keep)])
                r = sp.morans_i(g[col].values[keep], W, n_perm=19 if quick else 499, rng=rng)
                rows.append({"outcome": okey, "series": series, "year": int(y), "I": r["I"], "expected": r["expected"], "p_perm": r["p_perm"],
                             "counties": r["n"], "isolates_dropped": int((~keep).sum()), "w_km": DIST_KM})
        print(f"  moran {okey} done")
    out = pd.DataFrame(rows)
    out.to_csv(out_file(tables_dir, tags("moran", quick), f"{TODAY}_moran_by_year.csv"), index=False)
    summ = (out.groupby(["outcome", "series"]).agg(years=("year", "size"), mean_I=("I", "mean"), min_I=("I", "min"), max_I=("I", "max"),
                                                   share_p_below_05=("p_perm", lambda p: float((p < 0.05).mean()))).reset_index())
    summ.to_csv(out_file(tables_dir, tags("moran", quick), f"{TODAY}_moran_summary.csv"), index=False)
    print(summ.to_string(index=False))


# ------------------------------------------------------------------------------------------------------------- conley
def section_conley(df, geo, D, quick):
    rows = []
    grid = baseline_grid()
    for tid in REPORT_TREATMENTS:
        tcol, extra = TREAT_COL[tid], SPEC_EXTRA_REGRESSORS.get(tid, [])
        for okey, ocol in list(OUTCOMES.items())[: 1 if quick else None]:
            sub, mp = a2_sample(df, geo, ocol, tcol, extra)
            if len(sub) < 200:
                continue
            m = fit_twfe(sub, formula(mp, ocol, tcol, extra), vcov="hetero", keep_singletons=True)
            if m._N != len(sub):
                print(f"  [{tid}|{okey}] row mismatch, skipped"); continue
            ms = fit_twfe(sub, formula(mp, ocol, tcol, extra))
            b, se_state, p_state = coef_row(ms, mp[tcol])
            rec = {"treatment_id": tid, "outcome": okey, "beta": b, "se_state": se_state, "p_state": p_state, "N": int(m._N)}
            for km in CONLEY_KM:
                se = conley_se(m, sub, mp[tcol], sub["lat"].values, sub["lon"].values, cutoff_km=km)
                rec[f"se_conley_{km}km"], rec[f"p_conley_{km}km"] = se, float(2 * norm.sf(abs(b / se)))
            if grid is not None:
                g = grid[(grid.treatment_id == tid) & (grid.outcome == okey)]
                if len(g):
                    rec["check_vs_A2_beta"] = abs(b - float(g.beta.iloc[0])); rec["check_vs_A2_se_state"] = abs(se_state - float(g.se_state.iloc[0]))
            rows.append(rec)
        print(f"  conley {tid} done")
    out = pd.DataFrame(rows)
    out.to_csv(out_file(tables_dir, tags("conley", quick), f"{TODAY}_conley_results.csv"), index=False)
    print(out.round(4).to_string(index=False))
    # self-check: a cut-off of 0.001 km keeps only the same-county terms, so the Conley meat equals the county-clustered meat
    tcol, extra = TREAT_COL["E2"], []
    ocol = list(OUTCOMES.values())[0]
    sub, mp = a2_sample(df, geo, ocol, tcol, extra)
    m = fit_twfe(sub, formula(mp, ocol, tcol, extra), vcov="hetero", keep_singletons=True)
    mc = pf.feols(formula(mp, ocol, tcol, extra), data=sub, vcov={"CRV1": "fips"}, fixef_rm="none", ssc=pf.ssc(adj=False, cluster_adj=False))
    tiny = conley_se(m, sub, mp[tcol], sub["lat"].values, sub["lon"].values, cutoff_km=0.001)
    ref = float(mc.se()[mp[tcol]])
    print(f"  self-check: Conley at 0.001 km {tiny:.6g} vs county-clustered (no small-sample adjustment) {ref:.6g}  ->  {'ok' if abs(tiny - ref) < 1e-6 * ref else 'DIFFERS'}")


# ------------------------------------------------------------------------------------------------------------- division x year
def section_division(df, geo, D, quick):
    rows = []
    for tid in REPORT_TREATMENTS:
        tcol, extra = TREAT_COL[tid], SPEC_EXTRA_REGRESSORS.get(tid, [])
        for okey, ocol in list(OUTCOMES.items())[: 1 if quick else None]:
            sub, mp = a2_sample(df, geo, ocol, tcol, extra)
            if len(sub) < 200:
                continue
            sub["div_year"] = sub["division"].astype(str) + "_" + sub["year"].astype(str)
            base = fit_twfe(sub, formula(mp, ocol, tcol, extra))
            alt = fit_twfe(sub, formula(mp, ocol, tcol, extra, fe="fips + div_year"))
            b0, s0, p0 = coef_row(base, mp[tcol])
            b1, s1, p1 = coef_row(alt, mp[tcol])
            rows.append({"treatment_id": tid, "outcome": okey, "beta_baseline": b0, "se_baseline": s0, "p_baseline": p0,
                         "beta_division_year": b1, "se_division_year": s1, "p_division_year": p1, "N": int(alt._N),
                         "divisions": int(sub.division.nunique()), "division_year_cells": int(sub.div_year.nunique())})
        print(f"  division-fe {tid} done")
    out = pd.DataFrame(rows)
    out.to_csv(out_file(tables_dir, tags("division-fe", quick), f"{TODAY}_division_year_results.csv"), index=False)
    print(out.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------------------------- donut
def nearest_treated_km(fips_all, treated_fips, idx, D):
    """Distance in km from each county to the nearest county in treated_fips (itself excluded)."""
    ti = np.array([idx[f] for f in treated_fips])
    out = {}
    for f in fips_all:
        d = D[idx[f], ti]
        d = d[d > 0] if (d > 0).any() else d
        out[f] = float(d.min()) if len(d) else np.inf
    return pd.Series(out)


def section_donut(df, geo, D, quick):
    _df, DESIGNS, _p, _c = load_designs()
    idx = {f: i for i, f in enumerate(geo.fips)}
    geo_i = geo.assign(fips_i=geo.fips.astype(int))
    xcols = [f"x{i + 1}" for i in range(len(CONTROLS))]
    n_boot = 20 if quick else 300
    rows = []
    base4d = out_glob(tables_dir, "4d-estimators", "*_overall_estimates.csv")
    base4d = pd.read_csv(base4d[-1]) if base4d else None
    for dname, Dz in DESIGNS.items():
        tag = "E2" if dname == "first expansion" else "E3"
        for okey, ocol in list(OUTCOMES.items())[: 1 if quick else None]:
            d = usable_sample(Dz["data"], ocol, Dz["treat"], Dz["cohort"], CONTROLS)
            d["fips_s"] = d["fips"].astype(int).map(dict(zip(geo_i.fips_i, geo_i.fips)))
            d = d[d.fips_s.notna()].reset_index(drop=True)                       # counties with an outline
            treated = sorted(d.loc[d.cohort.notna(), "fips_s"].unique())
            allf = sorted(d.fips_s.unique())
            dist = nearest_treated_km(allf, treated, idx, D)
            never = d.groupby("fips_s").cohort.apply(lambda s: s.isna().all())
            for km in (0,) + DONUT_KM:
                drop = set(never.index[never & (dist.reindex(never.index) <= km)]) if km else set()
                s = d[~d.fips_s.isin(drop)].copy()
                s["state_fips"] = pd.factorize(s["state_fips"])[0]
                n_tr, n_ctl = s.loc[s.cohort.notna(), "fips"].nunique(), s.loc[s.cohort.isna(), "fips"].nunique()
                rec = {"design": tag, "outcome": okey, "radius_km": km, "counties": s.fips.nunique(), "treated_counties": n_tr,
                       "never_treated_kept": n_ctl, "never_treated_dropped": len(drop)}
                tw = pf.feols("y ~ D + " + " + ".join(xcols) + " | fips + year", data=s, vcov={"CRV1": "state_fips"}).tidy().loc["D"]
                rows.append({**rec, "estimator": "TWFE", "estimate": float(tw["Estimate"]), "se": float(tw["Std. Error"]), "p": float(tw["Pr(>|t|)"])})
                if n_tr >= 10:
                    _, agg = run_cs(s, "y", "cohort", xcols, n_boot=n_boot, seed=CS_SEED)
                    if len(agg):
                        ov = agg[agg.agg_level == "overall_ATT"].iloc[0]
                        rows.append({**rec, "estimator": "Callaway-Sant'Anna", "estimate": float(ov["att"]), "se": float(ov["se_state"]),
                                     "p": float(2 * norm.sf(abs(ov["att"] / ov["se_state"])))})
                if km == 0 and base4d is not None and not quick:
                    ref = base4d[(base4d.design.str.startswith("first " + ("expansion" if tag == "E2" else "entry"))) & (base4d.outcome == okey)
                                 & (base4d.estimator == "Callaway-Sant'Anna (benchmark)") & (base4d.kind == "overall")]
                    cs0 = [r for r in rows if r["design"] == tag and r["outcome"] == okey and r["radius_km"] == 0 and r["estimator"] != "TWFE"]
                    if len(ref) and cs0:
                        print(f"  check radius 0 vs 4d benchmark ({tag}, {okey}): {cs0[-1]['estimate']:.8f} vs {float(ref.est.iloc[0]):.8f}")
            print(f"  donut {tag} {okey}: {len(treated)} treated, never-treated dropped at 50/100 km: "
                  f"{[int((never & (dist.reindex(never.index) <= k)).sum()) for k in DONUT_KM]} of {int(never.sum())}")
    # reported A2 treatments (TWFE only): treated = ever positive on the variable
    for tid in ("T1", "T3", "T4"):
        tcol, extra = TREAT_COL[tid], SPEC_EXTRA_REGRESSORS.get(tid, [])
        for okey, ocol in list(OUTCOMES.items())[: 1 if quick else None]:
            sub, mp = a2_sample(df, geo, ocol, tcol, extra)
            ever = sub.groupby("fips")[mp[tcol]].max() > 0
            treated = sorted(ever.index[ever])
            allf = sorted(ever.index)
            dist = nearest_treated_km(allf, treated, idx, D)
            for km in (0,) + DONUT_KM:
                drop = set(ever.index[(~ever) & (dist.reindex(ever.index) <= km)]) if km else set()
                s = sub[~sub.fips.isin(drop)]
                b, se, p = coef_row(fit_twfe(s, formula(mp, ocol, tcol, extra)), mp[tcol])
                rows.append({"design": tid, "outcome": okey, "radius_km": km, "counties": s.fips.nunique(), "treated_counties": len(treated),
                             "never_treated_kept": s.fips.nunique() - len(treated), "never_treated_dropped": len(drop), "estimator": "TWFE",
                             "estimate": b, "se": se, "p": p})
        print(f"  donut {tid} done")
    out = pd.DataFrame(rows)
    out.to_csv(out_file(tables_dir, tags("donut", quick), f"{TODAY}_donut_results.csv"), index=False)
    print(out.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------------------------- durbin
def balanced_panel(df, geo, okey, ocol):
    """Counties observed with complete data in every year of the outcome's descriptive window (a balanced panel for the Durbin model)."""
    lo, hi = OUTCOME_WINDOWS[okey]
    need = ["fips", "year", ocol, "log_pop"] + [TREAT_COL[t] for t in REPORT_TREATMENTS] + CONTROLS
    s = df[df.year.between(lo, hi)][list(dict.fromkeys(need))].dropna()
    n_year = hi - lo + 1
    full = s.groupby("fips").year.nunique()
    s = s[s.fips.isin(full.index[full == n_year])].merge(geo[["fips"]], on="fips")
    return s, (lo, hi)


def section_durbin(df, geo, D, quick):
    rng = np.random.default_rng(SEED)
    cz = load_cz()
    idx = {f: i for i, f in enumerate(geo.fips)}
    rows = []
    n_boot = 19 if quick else 499
    state_of = df.groupby("fips").state_fips.first()
    for okey, ocol in list(OUTCOMES.items())[: 1 if quick else None]:
        s, (lo, hi) = balanced_panel(df, geo, okey, ocol)
        fips = np.array(sorted(s.fips.unique()))
        if len(fips) < 100:
            print(f"  [{okey}] only {len(fips)} balanced counties, skipped"); continue
        ii = np.array([idx[f] for f in fips])
        weights = {f"W_dist_{DIST_KM}km": (D[np.ix_(ii, ii)] <= DIST_KM).astype(float)}
        if cz is not None:
            weights["W_cz"] = sp.commuting_zone_adjacency(pd.Series(fips).map(dict(zip(cz.fips, cz.cz))).where(lambda x: x.notna(), None).values)
        for wname, A in weights.items():
            np.fill_diagonal(A, 0.0)
            keep = A.sum(axis=1) > 0
            f_k = fips[keep]
            W = sp.row_standardise(A[np.ix_(keep, keep)])
            ss = s[s.fips.isin(f_k)].sort_values(["fips", "year"])
            piv = lambda col: ss.pivot(index="fips", columns="year", values=col).loc[f_k].values
            eig = sp._eigs(W)
            Xs = [piv(c) for c in CONTROLS] + [piv("log_pop")]
            Y = piv(ocol)
            groups = state_of.reindex(f_k).values
            for tid in REPORT_TREATMENTS:
                Dm = piv(TREAT_COL[tid])
                ctrl = Xs if tid == "T4" else Xs[:-1]                      # log population only with T4, as in the A2 grid
                f = sp.sdm_fit(Y, Dm, ctrl, W, eig=eig)
                eff = sp.sdm_effects(f, rng=rng)
                rec = {"outcome": okey, "treatment_id": tid, "w": wname, "counties": len(f_k), "years": f"{lo}-{hi}", "isolates_dropped": int((~keep).sum()),
                       "rho": f["rho"], "se_rho": f["se_rho"], "beta": f["beta"], "se_beta": f["se_beta"], "theta": f["theta"], "se_theta": f["se_theta"],
                       "direct": eff["direct"], "se_direct": eff["se_direct"], "indirect": eff["indirect"], "se_indirect": eff["se_indirect"],
                       "total": eff["total"], "se_total": eff["se_total"], "loglik": f["loglik"], "boot_draws": 0}
                if (okey, tid) in CONFIRMATORY:
                    b = sp.sdm_wild_bootstrap(f, groups, n_boot=n_boot, rng=rng)
                    for k in ("rho", "beta", "theta", "direct", "indirect", "total"):
                        rec[f"boot_se_{k}"] = float(np.std(b[k], ddof=1))
                        rec[f"boot_p_{k}"] = float(min(1.0, 2 * min((b[k] <= 0).mean(), (b[k] >= 0).mean()) + 1.0 / (len(b[k]) + 1)))
                    rec["boot_draws"] = n_boot
                rows.append(rec)
                print(f"  durbin {okey} | {tid} | {wname}: N={len(f_k)}, rho={f['rho']:+.3f}, direct={eff['direct']:+.4f}, indirect={eff['indirect']:+.4f}")
    if cz is None:
        print("  W_cz not run: no county-to-commuting-zone crosswalk in Data/raw/cz or Data/clean/cz")
    out = pd.DataFrame(rows)
    out.to_csv(out_file(tables_dir, tags("durbin", quick), f"{TODAY}_durbin_results.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", nargs="+", choices=SECTIONS, help="run only these sections (default: all, in order)")
    ap.add_argument("--quick", action="store_true", help="first outcome only, few permutation and bootstrap draws (test run)")
    args = ap.parse_args()
    df, _designs, _p, _c = load_designs()
    geo, D = load_geometry(df)
    fns = {"moran": section_moran, "donut": section_donut, "conley": section_conley, "division-fe": section_division, "durbin": section_durbin}
    for s in [x for x in SECTIONS if args.only is None or x in args.only]:
        print("\n" + "#" * 100 + f"\n# section: {s}\n" + "#" * 100, flush=True)
        fns[s](df, geo, D, args.quick)


if __name__ == "__main__":
    main()
