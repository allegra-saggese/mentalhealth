#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4j-measurement-split.py -- the headline design on both sides of the 2016 measurement break (D-62; plan 2026-11-23, step 2).

Large dairies -> Poor MH Days, first expansion (E2) and first entry (E3), estimated on three windows: 2010-2023 (the reported window, which must
reproduce the reported numbers), 2010-2015 and 2016-2023. Estimators: Callaway-Sant'Anna covariate-adjusted (control set of 8, not-yet-treated
comparison, state bootstrap, 300 draws) and static TWFE on the same sample (county + year effects, control set, state-clustered); for E2 also the A2
reported specification on the full panel. Cohorts after a window's last year are untreated inside it and stay as comparison counties (D-62).
Reads:  panel via designs.load_designs(); county outlines for the A2 sample merge (script4g-spillovers.load_geometry)
Writes: Data/output/tables/<date>_4j-split_split_results.csv
Run:    PYTHONHASHSEED=0 python3 script4j-measurement-split.py     (about 5 minutes)
Limits: 2010-2015 holds the 2012 census wave only; 2016-2023 the 2017 and 2022 waves, so the split also separates cohorts (D-62).
"""
import warnings; warnings.filterwarnings("ignore")
import importlib.util
import os
from datetime import date

import numpy as np
import pandas as pd
import pyfixest as pf

from designs import load_designs, cafo_designs, usable_sample
from estimators import run_cs
from packages import out_file, tables_dir
from script4_treatment import OUTCOMES

TODAY = date.today().strftime("%Y-%m-%d")
OUT = "Poor MH Days"
WINDOWS = {"full 2010-2023": (2010, 2023), "pre-break 2010-2015": (2010, 2015), "post-break 2016-2023": (2016, 2023)}
N_BOOT = 300


def _load_g():
    spec = importlib.util.spec_from_file_location("script4g", os.path.join(os.path.dirname(os.path.abspath(__file__)), "script4g-spillovers.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def window_sample(Dz, ocol, controls, lo, hi):
    """Design sample inside [lo, hi]: cohorts after hi become untreated (missing cohort), then the shared usable_sample rule (a pre-period is needed)."""
    data = Dz["data"][Dz["data"].year.between(lo, hi)].copy()
    data.loc[data[Dz["cohort"]] > hi, Dz["cohort"]] = np.nan
    return usable_sample(data, ocol, Dz["treat"], Dz["cohort"], controls)


def main():
    g = _load_g()
    ocol, controls, xcols = OUTCOMES[OUT], g.CONTROLS, [f"x{i + 1}" for i in range(len(g.CONTROLS))]
    df, designs, _p, _c = load_designs()
    df, _ = cafo_designs(df, "dairy", "large")
    geo, _D = g.load_geometry(df)
    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        for dname, Dz in designs.items():
            tag = "E2" if dname == "first expansion" else "E3"
            d = window_sample(Dz, ocol, controls, lo, hi)
            treated = d[d.cohort.notna()]
            info = dict(window=wname, design=tag, counties=int(d.fips.nunique()), treated_counties=int(treated.fips.nunique()),
                        cohorts=",".join(str(int(c)) for c in sorted(treated.cohort.unique())), n_obs=len(d))
            if info["treated_counties"] == 0:
                print(f"  {wname} {tag}: no treated counties, skipped"); continue
            m = pf.feols("y ~ D + " + " + ".join(xcols) + " | fips + year", data=d, vcov={"CRV1": "state_fips"})
            t = m.tidy().loc["D"]
            rows.append({**info, "estimator": "TWFE (design sample)", "estimate": float(t["Estimate"]), "se": float(t["Std. Error"]), "p": float(t["Pr(>|t|)"])})
            _, agg = run_cs(d, "y", "cohort", xcols, n_boot=N_BOOT, seed=g.CS_SEED)
            ov = agg[agg.agg_level == "overall_ATT"].iloc[0]
            from scipy.stats import norm
            rows.append({**info, "estimator": "Callaway-Sant'Anna", "estimate": float(ov.att), "se": float(ov.se_state), "p": float(2 * norm.sf(abs(ov.att / ov.se_state)))})
            print(f"  {wname} {tag}: TWFE {rows[-2]['estimate']:+.3f} ({rows[-2]['se']:.3f}), CS {ov.att:+.3f} ({ov.se_state:.3f}); treated {info['treated_counties']}, cohorts {info['cohorts']}")
        # A2 reported specification (E2, full panel sample)
        sub, mp = g.a2_sample(df[df.year.between(lo, hi)], geo, ocol, g.TREAT_COL["E2"], [])
        if sub[mp[g.TREAT_COL["E2"]]].nunique() > 1:
            m = g.fit_twfe(sub, g.formula(mp, ocol, g.TREAT_COL["E2"], []))
            b, se, p = g.coef_row(m, mp[g.TREAT_COL["E2"]])
            rows.append(dict(window=wname, design="E2", estimator="TWFE (A2 reported spec)", estimate=b, se=se, p=p, n_obs=int(m._N),
                             counties=int(sub.fips.nunique()), treated_counties=np.nan, cohorts=""))
    t = pd.DataFrame(rows)
    # replication of the reported numbers on the full window (D-62)
    f = t[t.window == "full 2010-2023"].set_index(["design", "estimator"])
    checks = {("E2", "TWFE (A2 reported spec)"): 0.104, ("E2", "Callaway-Sant'Anna"): 0.037, ("E3", "Callaway-Sant'Anna"): 0.026}
    for k, v in checks.items():
        print(f"  replication {k}: {f.loc[k, 'estimate']:.4f} vs reported {v}")
        assert abs(f.loc[k, "estimate"] - v) < 1e-3, f"full-window {k} does not reproduce the reported estimate"
    t.to_csv(out_file(tables_dir, "4j-split", f"{TODAY}_split_results.csv"), index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
