#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4m-fsis-closure-clean.py -- plant closures kept out of the comparison, and many plants versus none (D-65; plan 2026-11-23).

    entry-clean  the D-63 county entry design and the D-64 zone designs (A, B) after dropping every county / zone with a plant closure
    many-vs-none counties with >= 2 large plants in 2017 and no closure vs counties with no FSIS establishment in 2017-2023: event study (ref 2016,
                 leads = pre-trend test) and the static Many x (year >= 2017) contrast, all seven outcomes
Closure: the county's count of FSIS establishments falls year on year in 2018-2023, or its large-plant indicator goes 1 -> 0 (zone: any panel county).
Reads:  panel via designs.load_designs(); Data/merged/*_panel_fsis.csv; the county-to-zone crosswalk; script4k / script4l / script4g helpers
Writes: Data/output/tables/<date>_4m-fsis-clean_{overall,event_time,manyvsnone,manyvsnone_event}.csv
Run:    PYTHONHASHSEED=0 python3 script4m-fsis-closure-clean.py [--quick]   (about 15 minutes)
Limits: see D-65; the many-vs-none contrast is descriptive and confounded with the 2016 CDC method change.
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import importlib.util
import os
from datetime import date

import numpy as np
import pandas as pd
import pyfixest as pf

from designs import load_designs, usable_sample, FSIS_WINDOW
from estimators import joint_wald, clean
from functions import latest_file_glob
from packages import db_data, out_file, tables_dir
from script4_treatment import OUTCOMES

TODAY = date.today().strftime("%Y-%m-%d")
HERE = os.path.dirname(os.path.abspath(__file__))
REF_YEAR, MANY_MIN_LARGE = 2016, 2


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def closure_and_baseline():
    """Per county: closure flag, any establishment in the window, number of large plants in 2017."""
    fp = latest_file_glob(os.path.join(db_data, "merged"), "*_panel_fsis.csv")
    f = pd.read_csv(fp, low_memory=False, usecols=["fips", "year", "n_unique_establishments_fsis", "n_size_bucket_4_fsis", "n_size_bucket_5_fsis"])
    f["fips"] = f["fips"].astype(str).str.zfill(5)
    f = f[f.year.between(*FSIS_WINDOW)]
    U = f.pivot_table(index="fips", columns="year", values="n_unique_establishments_fsis", aggfunc="first").fillna(0)
    Lg = (f.assign(L=f.n_size_bucket_4_fsis.fillna(0) + f.n_size_bucket_5_fsis.fillna(0)).pivot_table(index="fips", columns="year", values="L", aggfunc="sum").fillna(0) > 0).astype(int)
    out = pd.DataFrame(index=U.index)
    out["closure"] = (U.diff(axis=1).iloc[:, 1:] < 0).any(axis=1) | (Lg.diff(axis=1).iloc[:, 1:] < 0).any(axis=1)
    out["any_est"] = (U > 0).any(axis=1)
    out["large_2017"] = f[f.year == FSIS_WINDOW[0]].assign(L=lambda x: x.n_size_bucket_4_fsis.fillna(0) + x.n_size_bucket_5_fsis.fillna(0)).set_index("fips").L.reindex(U.index).fillna(0)
    return out


def many_vs_none(df, cb, n_quick):
    """Event study and static contrast of Many vs None, county + year effects, control set, state-clustered. Returns (summary rows, event rows)."""
    k = _load("script4k-fsis-large-plants")
    allc = pd.Series(df.fips.unique())
    many = set(cb.index[(cb.large_2017 >= MANY_MIN_LARGE) & ~cb.closure])
    none = set(allc[~allc.isin(cb.index[cb.any_est])])
    print(f"  many: {len(many)} counties; none: {len(none)} counties (of {len(allc):,}); the rest are left out")
    d = df[df.fips.isin(many | none)].copy()
    d["Many"] = d.fips.isin(many).astype(float)
    d["Post"] = (d.year >= FSIS_WINDOW[0]).astype(float)
    d["ManyPost"] = d.Many * d.Post
    rows, evrows = [], []
    for okey, ocol in list(OUTCOMES.items())[:n_quick]:
        u = d[["fips", "year", "state_fips", ocol, "Many", "ManyPost"] + k.CONTROLS].dropna()
        u = u.rename(columns={ocol: "y", **{c: f"x{i + 1}" for i, c in enumerate(k.CONTROLS)}})
        ys = sorted(set(u.year) - {REF_YEAR})
        for y in ys:
            u[f"m{y}"] = u.Many * (u.year == y)
        xs = " + ".join(k.XCOLS)
        me = pf.feols("y ~ " + " + ".join(f"m{y}" for y in ys) + " + " + xs + " | fips + year", data=u, vcov={"CRV1": "state_fips"})
        pre = [f"m{y}" for y in ys if y < REF_YEAR]
        stat, df_, p_pre = joint_wald(me, pre)
        ms = pf.feols("y ~ ManyPost + " + xs + " | fips + year", data=u, vcov={"CRV1": "state_fips"})
        t = ms.tidy().loc["ManyPost"]
        rows.append(dict(outcome=okey, estimate=float(t["Estimate"]), se=float(t["Std. Error"]), p=float(t["Pr(>|t|)"]), pre_years=len(pre),
                         pre_joint_p=p_pre, many_counties=int(u[u.Many == 1].fips.nunique()), none_counties=int(u[u.Many == 0].fips.nunique()), n_obs=len(u)))
        tt = me.tidy()
        for y in ys:
            r = tt.loc[f"m{y}"]
            evrows.append(dict(outcome=okey, year=y, estimate=float(r["Estimate"]), se=float(r["Std. Error"]), p=float(r["Pr(>|t|)"])))
        print(f"  many-vs-none {okey:26s} static {t['Estimate']:+9.3f} (p {t['Pr(>|t|)']:.3f}); leads {len(pre)}, joint p {p_pre:.3f}")
    return rows, evrows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_boot, n_out = (20, 1) if args.quick else (300, None)
    g, k, l = _load("script4g-spillovers"), _load("script4k-fsis-large-plants"), _load("script4l-fsis-cz-exposure")
    df, _designs, _p, _c = load_designs()
    cb = closure_and_baseline()
    print(f"  closures: {int(cb.closure.sum())} of {len(cb):,} counties in the FSIS panel")
    d = k.build_panel(df)
    cz = g.load_cz()
    d = d.merge(cz, on="fips", how="left")
    closure_fips = set(cb.index[cb.closure])
    versions = {}
    entry = k.designs_from(d)["entry"]
    versions["county entry"] = entry[~entry.fips.isin(closure_fips)].copy()
    dz = d[d.cz.notna()]
    bad = set(dz.loc[dz.fips.isin(closure_fips), "cz"])
    for name, z in l.zone_designs(dz[~dz.cz.isin(bad)]).items():
        versions[f"zone {name}"] = z
    rows, evs = [], []
    for vname, z0 in versions.items():
        print(f"  {vname}: {z0.fips.nunique():,} counties after dropping closures, {z0.loc[z0.cohort.notna(), 'fips'].nunique()} treated")
        for sens, z in (("baseline", z0), ("drop cohorts 2020-21", z0[~z0.cohort.isin(k.COVID_COHORTS)])):
            for okey, ocol in list(OUTCOMES.items())[:n_out]:
                r = k.estimate(z, ocol, n_boot)
                if r is None:
                    print(f"  {vname} {sens} {okey}: too few treated counties or no estimable cohort, skipped"); continue
                res, ev = r
                for x in res:
                    rows.append(dict(version=vname, sensitivity=sens, outcome=okey, **x))
                ev.insert(0, "outcome", okey); ev.insert(0, "sensitivity", sens); ev.insert(0, "version", vname)
                evs.append(ev)
                cs = res[1]
                print(f"  {vname:24s} {sens:22s} {okey:26s} CS {cs['estimate']:+9.3f} ({cs['se']:.3f}) p {cs['p']:.3f}  TWFE {res[0]['estimate']:+9.3f} "
                      f"(p {res[0]['p']:.3f})  treated {cs['treated_counties']:3d}  pre-sig {cs['pre_sig_5pct']}/{cs['pre_periods']}")
    t = pd.DataFrame(rows)
    t["p_holm_7_outcomes"] = np.nan
    for _, gp in t.groupby(["version", "sensitivity", "estimator"]):
        t.loc[gp.index, "p_holm_7_outcomes"] = k.holm(gp.p.values)
    t.to_csv(out_file(tables_dir, "4m-fsis-clean", f"{TODAY}_overall.csv"), index=False)
    pd.concat(evs, ignore_index=True).to_csv(out_file(tables_dir, "4m-fsis-clean", f"{TODAY}_event_time.csv"), index=False)
    mrows, mev = many_vs_none(df, cb, n_out)
    m = pd.DataFrame(mrows)
    m["p_holm_7_outcomes"] = k.holm(m.p.values)
    m.to_csv(out_file(tables_dir, "4m-fsis-clean", f"{TODAY}_manyvsnone.csv"), index=False)
    pd.DataFrame(mev).to_csv(out_file(tables_dir, "4m-fsis-clean", f"{TODAY}_manyvsnone_event.csv"), index=False)


if __name__ == "__main__":
    main()
