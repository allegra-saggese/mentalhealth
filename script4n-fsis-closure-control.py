#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4n-fsis-closure-control.py -- plant closure as its own control variable in the FSIS entry designs (D-66).

Samples: the D-63 county entry design and the D-64 zone designs A and B, with NO closure area dropped. ClosePost = 1 from the first closure year
(county design: the county's own; zone designs: any panel county in the zone). Estimators, each with and without ClosePost on the same sample:
TWFE (county + year effects, control set, state-clustered) and the imputation estimator (did2s; ClosePost enters the first stage; overall and
event time). All seven outcomes; Holm over 7 outcomes within estimator, version and spec. Callaway-Sant'Anna cannot take a time-varying control.
Reads:  panel via designs.load_designs(); Data/merged/*_panel_fsis.csv; the county-to-zone crosswalk; script4g / script4k / script4l helpers
Writes: Data/output/tables/<date>_4n-fsis-ctrl_{overall,event_time}.csv
Run:    PYTHONHASHSEED=0 python3 script4n-fsis-closure-control.py [--quick]   (about 10 minutes)
Limits: see D-66 (ClosePost is post-treatment if entry causes closures; short window; 12-35 entering zones).
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
from estimators import did2s_estimates, joint_wald
from functions import latest_file_glob
from packages import db_data, out_file, tables_dir
from script4_treatment import OUTCOMES

TODAY = date.today().strftime("%Y-%m-%d")
HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def closure_events():
    """County-year closure event: the FSIS establishment count falls over the previous year, or the large-plant indicator goes 1 -> 0 (2018-2023)."""
    fp = latest_file_glob(os.path.join(db_data, "merged"), "*_panel_fsis.csv")
    f = pd.read_csv(fp, low_memory=False, usecols=["fips", "year", "n_unique_establishments_fsis", "n_size_bucket_4_fsis", "n_size_bucket_5_fsis"])
    f["fips"] = f["fips"].astype(str).str.zfill(5)
    f = f[f.year.between(*FSIS_WINDOW)]
    U = f.pivot_table(index="fips", columns="year", values="n_unique_establishments_fsis", aggfunc="first").fillna(0)
    Lg = (f.assign(L=f.n_size_bucket_4_fsis.fillna(0) + f.n_size_bucket_5_fsis.fillna(0)).pivot_table(index="fips", columns="year", values="L", aggfunc="sum").fillna(0) > 0).astype(int)
    ev = ((U.diff(axis=1) < 0) | (Lg.diff(axis=1) < 0)).iloc[:, 1:].astype(int)
    return ev.stack().rename("close_event").reset_index()          # fips, year, close_event


def add_close_post(z, level):
    """ClosePost for the sample z: absorbing indicator of the unit's first closure (unit = county or commuting zone)."""
    key = "fips" if level == "county" else "cz"
    ev = z[["fips", "year"]].merge(CLOSE, on=["fips", "year"], how="left").fillna({"close_event": 0})
    if level == "zone":
        ev["cz"] = ev.fips.map(dict(zip(z.fips, z.cz)))
    g = ev.groupby([key, "year"]).close_event.max().groupby(level=0).cummax().rename("ClosePost").reset_index()
    out = z.merge(g, on=[key, "year"], how="left")
    out["ClosePost"] = out["ClosePost"].fillna(0.0)
    return out


def run_one(z, ocol, controls, xcols_label):
    """TWFE and imputation overall (+ imputation event time) on the usable sample with the given control list; returns rows and event frame."""
    d = usable_sample(z, ocol, "D", "cohort", controls)
    xcols = [f"x{i + 1}" for i in range(len(controls))]
    treated = d[d.cohort.notna()]
    if treated.fips.nunique() < 10:
        return None
    base = dict(spec=xcols_label, n_obs=len(d), counties=int(d.fips.nunique()), treated_counties=int(treated.fips.nunique()))
    m = pf.feols("y ~ D + " + " + ".join(xcols) + " | fips + year", data=d, vcov={"CRV1": "state_fips"})
    t = m.tidy().loc["D"]
    rows = [dict(**base, estimator="TWFE", estimate=float(t["Estimate"]), se=float(t["Std. Error"]), p=float(t["Pr(>|t|)"]))]
    n_close = None
    if "ClosePost" in [c for c in controls]:
        k = controls.index("ClosePost") + 1
        tc = m.tidy().loc[f"x{k}"]
        rows[0]["closure_coef"], rows[0]["closure_p"] = float(tc["Estimate"]), float(tc["Pr(>|t|)"])
    ev = None
    try:
        keep = d.groupby("fips").apply(lambda g: int((g.D == 0).sum()) >= 2)
        dd = d[d.fips.isin(keep.index[keep])]
        e = did2s_estimates(dd, "y", "D", controls=xcols, clip=6)
        ov = e[e.kind == "overall"].iloc[0]
        rows.append(dict(**base, estimator="Imputation (did2s)", estimate=float(ov.est), se=float(ov.se), p=float(ov.p)))
        ev = e[e.kind == "event"].copy()
    except Exception as exc:                                    # a failing imputation fit is reported, not skipped silently
        rows.append(dict(**base, estimator="Imputation (did2s)", estimate=np.nan, se=np.nan, p=np.nan, error=f"{type(exc).__name__}: {str(exc)[:120]}"))
    return rows, ev


def main():
    global CLOSE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_out = 1 if args.quick else None
    g, k, l = _load("script4g-spillovers"), _load("script4k-fsis-large-plants"), _load("script4l-fsis-cz-exposure")
    df, _designs, _p, _c = load_designs()
    CLOSE = closure_events()
    d = k.build_panel(df).merge(g.load_cz(), on="fips", how="left")
    versions = {"county entry": ("county", k.designs_from(d)["entry"])}
    for name, z in l.zone_designs(d[d.cz.notna()]).items():
        versions[f"zone {name}"] = ("zone", z)
    rows, evs = [], []
    for vname, (level, z0) in versions.items():
        z = add_close_post(z0, level)
        print(f"  {vname}: {z.fips.nunique():,} counties, {z.loc[z.cohort.notna(), 'fips'].nunique()} treated; units with a closure: "
              f"{int(z.groupby('fips' if level == 'county' else 'cz').ClosePost.max().sum())}")
        for okey, ocol in list(OUTCOMES.items())[:n_out]:
            for spec, controls in (("with ClosePost", k.CONTROLS + ["ClosePost"]), ("without ClosePost", k.CONTROLS)):
                r = run_one(z, ocol, controls, spec)
                if r is None:
                    print(f"  {vname} {okey} {spec}: too few treated counties, skipped"); continue
                res, ev = r
                for x in res:
                    rows.append(dict(version=vname, outcome=okey, **x))
                if ev is not None:
                    ev.insert(0, "spec", spec); ev.insert(0, "outcome", okey); ev.insert(0, "version", vname)
                    evs.append(ev)
                a = {x["estimator"]: x for x in res}
                print(f"  {vname:24s} {okey:26s} {spec:18s} TWFE {a['TWFE']['estimate']:+9.3f} (p {a['TWFE']['p']:.3f})  "
                      f"imputation {a['Imputation (did2s)']['estimate']:+9.3f} (p {a['Imputation (did2s)']['p']:.3f})")
    t = pd.DataFrame(rows)
    t["p_holm_7_outcomes"] = np.nan
    for _, gp in t.dropna(subset=["p"]).groupby(["version", "spec", "estimator"]):
        t.loc[gp.index, "p_holm_7_outcomes"] = k.holm(gp.p.values)
    t.to_csv(out_file(tables_dir, "4n-fsis-ctrl", f"{TODAY}_overall.csv"), index=False)
    if evs:
        e = pd.concat(evs, ignore_index=True)
        e.to_csv(out_file(tables_dir, "4n-fsis-ctrl", f"{TODAY}_event_time.csv"), index=False)


if __name__ == "__main__":
    main()
