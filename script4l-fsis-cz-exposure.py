#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4l-fsis-cz-exposure.py -- large-plant entry at the commuting-zone level (D-64; plan 2026-11-23, step 4).

Same plant rule, window, estimators and outcomes as script4k-fsis-large-plants.py (D-63); exposure is defined for the whole commuting zone.
    A  every panel county of an at-risk zone is treated from the zone's first large-plant year (hosts and neighbours)
    B  as A, but counties that ever host a large plant themselves are dropped (treated counties are neighbours of a host only)
TWFE also reports a zone-clustered p-value. Sensitivity: drop zone cohorts 2020-21.
Reads:  panel via designs.load_designs(); Data/merged/*_panel_fsis.csv; the county-to-zone crosswalk (script4g-spillovers.load_cz)
Writes: Data/output/tables/<date>_4l-fsis-cz_{overall,event_time}.csv
Run:    PYTHONHASHSEED=0 python3 script4l-fsis-cz-exposure.py [--quick]   (about 10 minutes)
Limits: see D-64 (rural panel only: plants in urban counties of a zone are invisible; short window; few zone clusters in version B).
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
from packages import out_file, tables_dir
from script4_treatment import OUTCOMES

TODAY = date.today().strftime("%Y-%m-%d")
HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def zone_designs(d):
    """{version: panel with cohort and absorbing D}; zones at risk have no large plant in any panel county in the first window year."""
    y0 = FSIS_WINDOW[0]
    w = d[d.year.between(*FSIS_WINDOW)]
    Z = w.groupby(["cz", "year"]).L.max().unstack()
    risk = Z.index[Z[y0] == 0]
    ev = Z.loc[risk].drop(columns=y0).eq(1)
    cohort = ev.idxmax(axis=1).where(ev.any(axis=1))
    host = set(w.loc[w.L == 1, "fips"])
    base = d[d.cz.isin(risk) & d.year.between(*FSIS_WINDOW)].copy()
    base["cohort"] = base.cz.map(cohort)
    base["D"] = (base.cohort.notna() & (base.year >= base.cohort)).astype(float)
    out = {"A all counties in zone": base, "B non-host counties only": base[~base.fips.isin(host)].copy()}
    print(f"  zones at risk {len(risk):,}; entering zones {int(cohort.notna().sum())}; cohorts {cohort.value_counts().sort_index().to_dict()}")
    for k, z in out.items():
        print(f"  {k}: {z.fips.nunique():,} counties, {z.loc[z.cohort.notna(), 'fips'].nunique():,} treated, "
              f"{z.loc[z.cohort.notna(), 'cz'].nunique()} treated zones, {z.loc[z.cohort.isna(), 'cz'].nunique()} never-treated zones")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_boot = 20 if args.quick else 300
    g, k = _load("script4g-spillovers"), _load("script4k-fsis-large-plants")
    df, _designs, _p, _c = load_designs()
    d = k.build_panel(df)
    cz = g.load_cz()
    if cz is None:
        raise SystemExit("no commuting-zone crosswalk found (Data/raw/cz or Data/clean/cz)")
    d = d.merge(cz, on="fips", how="left")
    miss = d.loc[d.cz.isna(), "fips"].nunique()
    print(f"  {miss} of {d.fips.nunique():,} panel counties have no commuting-zone match and are left out")
    d = d[d.cz.notna()]
    designs = zone_designs(d)
    rows, evs = [], []
    for vname, z0 in designs.items():
        for sens, z in (("baseline", z0), ("drop cohorts 2020-21", z0[~z0.cohort.isin(k.COVID_COHORTS)])):
            cz_of = dict(zip(z.fips.astype(int), z.cz))
            for okey, ocol in list(OUTCOMES.items())[: 1 if args.quick else None]:
                r = k.estimate(z, ocol, n_boot)
                if r is None:
                    print(f"  {vname} {sens} {okey}: too few treated counties or no estimable cohort, skipped"); continue
                res, ev = r
                u = usable_sample(z, ocol, "D", "cohort", k.CONTROLS)
                u["cz"] = u.fips.map(cz_of)
                m = pf.feols("y ~ D + " + " + ".join(k.XCOLS) + " | fips + year", data=u, vcov={"CRV1": "cz"})
                res[0]["p_zone_cluster"] = float(m.tidy().loc["D", "Pr(>|t|)"])
                res[0]["zone_clusters"] = int(u.cz.nunique())
                for x in res:
                    rows.append(dict(version=vname, sensitivity=sens, outcome=okey, **x))
                ev.insert(0, "outcome", okey); ev.insert(0, "sensitivity", sens); ev.insert(0, "version", vname)
                evs.append(ev)
                cs = res[1]
                print(f"  {vname[:1]} {sens:22s} {okey:26s} CS {cs['estimate']:+9.3f} ({cs['se']:.3f}) p {cs['p']:.3f}  TWFE {res[0]['estimate']:+9.3f} "
                      f"(p_state {res[0]['p']:.3f}, p_zone {res[0]['p_zone_cluster']:.3f})  treated {cs['treated_counties']:3d}  pre-sig {cs['pre_sig_5pct']}/{cs['pre_periods']}")
    t = pd.DataFrame(rows)
    t["p_holm_7_outcomes"] = np.nan
    for _, gp in t.groupby(["version", "sensitivity", "estimator"]):
        t.loc[gp.index, "p_holm_7_outcomes"] = k.holm(gp.p.values)
    t.to_csv(out_file(tables_dir, "4l-fsis-cz", f"{TODAY}_overall.csv"), index=False)
    pd.concat(evs, ignore_index=True).to_csv(out_file(tables_dir, "4l-fsis-cz", f"{TODAY}_event_time.csv"), index=False)


if __name__ == "__main__":
    main()
