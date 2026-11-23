#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4k-fsis-large-plants.py -- slaughterhouse (FSIS) large-plant entry and exit event studies (D-63; plan 2026-11-23, step 3).

A county has a large plant in a year when an FSIS establishment there has size code 4 or 5 (n_size_bucket_4/5_fsis > 0). 2017-2023 only.
    entry  counties with no large plant in 2017; treated from the first year one appears (absorbing)
    exit   counties with a large plant in 2017; treated from the first year without one (absorbing); the estimate is the effect of LOSING a plant
Each design x outcome: Callaway-Sant'Anna (covariate-adjusted, not-yet-treated comparison, state bootstrap 300 draws) and TWFE on the same
usable sample, plus event-time estimates with the count of significant pre-periods, and a sensitivity that drops cohorts 2020 and 2021 (COVID).
Reversal check (fixed in D-63): opposite signs and the exit 95% CI contains minus the entry estimate.
Reads:  panel via designs.load_designs(); Data/merged/*_panel_fsis.csv (latest); control set from script4_treatment
Writes: Data/output/tables/<date>_4k-fsis-large_{overall,event_time,reversal}.csv
Run:    PYTHONHASHSEED=0 python3 script4k-fsis-large-plants.py [--quick]   (about 10 minutes; --quick = first outcome, 20 draws)
Limits: see D-63 (window 2017-2023, COVID, 2021 method break, thin pre-periods; a large processing-only plant counts).
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import os
from datetime import date

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import norm

from designs import load_designs, usable_sample, FSIS_WINDOW
from estimators import run_cs
from functions import latest_file_glob
from packages import db_data, out_file, tables_dir
from script4_treatment import OUTCOMES, COVARIATE_ORDER

TODAY = date.today().strftime("%Y-%m-%d")
CONTROLS = COVARIATE_ORDER[:8]
XCOLS = [f"x{i + 1}" for i in range(len(CONTROLS))]
CS_SEED = 20260928
BUCKETS = ["n_size_bucket_4_fsis", "n_size_bucket_5_fsis"]
COVID_COHORTS = (2020, 2021)


def build_panel(df):
    """Panel with L = 1 if the county-year has a large FSIS plant (size code 4 or 5), 0 if not (0 also for counties absent from the FSIS panel), NaN outside 2017-2023."""
    fp = latest_file_glob(os.path.join(db_data, "merged"), "*_panel_fsis.csv")
    f = pd.read_csv(fp, low_memory=False, usecols=["fips", "year"] + BUCKETS)
    f["fips"] = f["fips"].astype(str).str.zfill(5)
    d = df.drop(columns=BUCKETS, errors="ignore").merge(f, on=["fips", "year"], how="left").sort_values(["fips", "year"]).reset_index(drop=True)
    inwin = d.year.between(*FSIS_WINDOW)
    d["L"] = np.where(inwin, (d[BUCKETS].fillna(0).sum(axis=1) > 0).astype(float), np.nan)
    return d


def designs_from(d):
    """Entry and exit designs as {name: DataFrame with cohort and absorbing treatment}. Events are first changes after the first window year."""
    w = d[d.year.between(*FSIS_WINDOW)]
    L = w.pivot(index="fips", columns="year", values="L")
    y0 = FSIS_WINDOW[0]
    out = {}
    for name, start, change in (("entry", 0, 1), ("exit", 1, 0)):
        keep = L.index[L[y0] == start]
        ev = L.loc[keep].drop(columns=y0).eq(change)
        cohort = ev.idxmax(axis=1).where(ev.any(axis=1))
        z = d[d.fips.isin(keep) & d.year.between(*FSIS_WINDOW)].copy()
        z["cohort"] = z.fips.map(cohort)
        z["D"] = (z.cohort.notna() & (z.year >= z.cohort)).astype(float)
        out[name] = z
        print(f"  {name}: {len(keep):,} counties at risk, {int(cohort.notna().sum())} events; cohorts {cohort.value_counts().sort_index().to_dict()}")
    return out


def estimate(z, ocol, n_boot):
    d = usable_sample(z, ocol, "D", "cohort", CONTROLS)
    treated = d[d.cohort.notna()]
    if treated.fips.nunique() < 10:
        return None
    m = pf.feols("y ~ D + " + " + ".join(XCOLS) + " | fips + year", data=d, vcov={"CRV1": "state_fips"})
    t = m.tidy().loc["D"]
    _, agg = run_cs(d, "y", "cohort", XCOLS, n_boot=n_boot, seed=CS_SEED)
    if agg.empty or "agg_level" not in agg:
        return None                                   # run_cs found no estimable cohort in this sample
    ov = agg[agg.agg_level == "overall_ATT"].iloc[0]
    ev = agg[(agg.agg_level == "event_time")].copy()
    ev["p"] = 2 * norm.sf((ev.att / ev.se_state).abs())
    pre = ev[ev.event_time <= -2]
    base = dict(n_obs=len(d), counties=int(d.fips.nunique()), treated_counties=int(treated.fips.nunique()),
                cohorts=",".join(str(int(c)) for c in sorted(treated.cohort.unique())))
    res = [dict(**base, estimator="TWFE", estimate=float(t["Estimate"]), se=float(t["Std. Error"]), p=float(t["Pr(>|t|)"])),
           dict(**base, estimator="Callaway-Sant'Anna", estimate=float(ov.att), se=float(ov.se_state),
                p=float(2 * norm.sf(abs(ov.att / ov.se_state))), pre_periods=len(pre), pre_sig_5pct=int((pre.p < 0.05).sum()))]
    return res, ev


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]))
    out = np.empty_like(adj)
    out[o] = adj
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_boot = 20 if args.quick else 300
    df, _designs, _p, _c = load_designs()
    d = build_panel(df)
    designs = designs_from(d)
    rows, evs = [], []
    for dname, z0 in designs.items():
        for sens, z in (("baseline", z0), ("drop cohorts 2020-21", z0[~z0.cohort.isin(COVID_COHORTS)])):
            for okey, ocol in list(OUTCOMES.items())[: 1 if args.quick else None]:
                r = estimate(z, ocol, n_boot)
                if r is None:
                    print(f"  {dname} {sens} {okey}: too few treated counties or no estimable cohort, skipped"); continue
                res, ev = r
                for x in res:
                    rows.append(dict(design=dname, sensitivity=sens, outcome=okey, **x))
                ev.insert(0, "outcome", okey); ev.insert(0, "sensitivity", sens); ev.insert(0, "design", dname)
                evs.append(ev)
                cs = res[1]
                print(f"  {dname:5s} {sens:22s} {okey:26s} CS {cs['estimate']:+9.3f} ({cs['se']:.3f}) p {cs['p']:.3f}  TWFE {res[0]['estimate']:+9.3f}  "
                      f"treated {cs['treated_counties']:3d}  pre-sig {cs['pre_sig_5pct']}/{cs['pre_periods']}")
    t = pd.DataFrame(rows)
    t["p_holm_7_outcomes"] = np.nan
    for k, g in t.groupby(["design", "sensitivity", "estimator"]):
        t.loc[g.index, "p_holm_7_outcomes"] = holm(g.p.values)
    t.to_csv(out_file(tables_dir, "4k-fsis-large", f"{TODAY}_overall.csv"), index=False)
    pd.concat(evs, ignore_index=True).to_csv(out_file(tables_dir, "4k-fsis-large", f"{TODAY}_event_time.csv"), index=False)
    # reversal check (rule fixed in D-63)
    rv = []
    for (sens, okey), g in t[t.estimator == "Callaway-Sant'Anna"].groupby(["sensitivity", "outcome"]):
        if set(g.design) == {"entry", "exit"}:
            a, b = g[g.design == "entry"].iloc[0], g[g.design == "exit"].iloc[0]
            ok = (bool(np.sign(a.estimate) != np.sign(b.estimate) and abs(b.estimate + a.estimate) <= 1.96 * b.se)
                  if np.isfinite(b.se) else "not assessable (exit standard error missing)")
            rv.append(dict(sensitivity=sens, outcome=okey, entry=a.estimate, entry_se=a.se, exit=b.estimate, exit_se=b.se, reversal_consistent=ok))
    pd.DataFrame(rv).to_csv(out_file(tables_dir, "4k-fsis-large", f"{TODAY}_reversal.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\nreversal check:"); print(pd.DataFrame(rv).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
