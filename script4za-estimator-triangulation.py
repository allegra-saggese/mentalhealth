"""script4za-estimator-triangulation.py -- is the Callaway-Sant'Anna result specific to Callaway-Sant'Anna?

Five more staggered-DiD estimators on the SAME sample as the CS benchmark: Sun-Abraham, imputation (did2s),
de Chaisemartin-D'Haultfoeuille (R package), stacked DiD, LP-DiD. Rule fixed before the run (D-45): an estimator
agrees with CS if its overall post-treatment ATT has the same sign and lies inside CS's 95% interval; the Poor MH Days
first-expansion result is called estimator-robust if at least 4 of the 5 agree. Everything is reported either way.
Plan:    code-audit/plans/2026-10-22_PLAN_estimator-triangulation.md.  Adapters and stacked DiD: estimators.py.
Reads:   designs.load_designs() (first expansion; first entry with clean controls), the 8 pre-committed controls.
Writes:  Data/output/tables/script4za/YYYY-MM-DD_{overall_estimates,event_paths,agreement,samples,failures}.csv
Run:     PYTHONHASHSEED=0 python3 script4za-estimator-triangulation.py [--outcome "Poor MH Days"] [--design expansion|entry] [--quick]
Sample:  all six estimators use one complete-case sample per (design, outcome); imputation alone drops counties with a single untreated row.
Needs:   Rscript with polars and DIDmultiplegtDYN (de Chaisemartin-D'Haultfoeuille). Runtime: about 20 minutes for 7 outcomes x 2 designs.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf

import estimators as est
from designs import load_designs, usable_sample
from estimators import run_cs
from script4_treatment import OUTCOMES, COVARIATE_ORDER, tables_dir, os, date

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
ARGV = sys.argv[1:]
ONLY_OUTCOMES = [ARGV[i + 1] for i, a in enumerate(ARGV) if a == "--outcome"]
ONLY_DESIGN = next((ARGV[i + 1] for i, a in enumerate(ARGV) if a == "--design"), None)
N_BOOT = 20 if QUICK else 300                   # CS state bootstrap, as in script4a/4u
CS_SEED = 20260928
CONTROLS = COVARIATE_ORDER[:8]
XCOLS = [f"x{i + 1}" for i in range(len(CONTROLS))]
CS = "Callaway-Sant'Anna (benchmark)"
FIVE = ["LP-DiD", "Sun-Abraham", "Imputation (did2s)", "de Chaisemartin-D'Haultfoeuille", "Stacked DiD"]
PRESPECIFIED = ("first expansion", "Poor MH Days")      # the one cell D-45 applies to
OUT = os.path.join(tables_dir, "script4za" + ("_quick" if QUICK else ""))
os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")


def cs_benchmark(d):
    """estimators.run_cs on the sample; overall ATT and event-time ATTs with the state-bootstrap SE."""
    _, agg = run_cs(d, "y", "cohort", XCOLS, n_boot=N_BOOT, seed=CS_SEED)
    ov = agg[agg.agg_level == "overall_ATT"].iloc[0]
    ev = agg[agg.agg_level == "event_time"]
    over = est._tri_rows(CS, "overall", np.nan, ov["att"], ov["se_state"])
    events = est._tri_rows(CS, "event", ev["event_time"].astype(float).values, ev["att"].values, ev["se_state"].values)
    return pd.concat([over, events], ignore_index=True), int(ov["n_cohorts"])


_df, DESIGNS, _poly, _counts = load_designs()
if ONLY_DESIGN:
    DESIGNS = {k: v for k, v in DESIGNS.items() if k.startswith("first " + ONLY_DESIGN)}
outcomes = {k: v for k, v in OUTCOMES.items() if not ONLY_OUTCOMES or k in ONLY_OUTCOMES}
if QUICK:
    outcomes = dict(list(outcomes.items())[:1])

overall_rows, event_rows, agree_rows, sample_rows, fail_rows = [], [], [], [], []
print("=" * 78)
print(f"script4za -- estimator triangulation   CS bootstrap draws {N_BOOT}   rule: D-45 (4 of 5, same sign, inside CS 95% CI)")
print("=" * 78)
for dname, D in DESIGNS.items():
    for okey, ocol in outcomes.items():
        d = usable_sample(D["data"], ocol, D["treat"], D["cohort"], CONTROLS)
        treated = d[d.cohort.notna()]
        cohorts = sorted(int(c) for c in treated.cohort.unique())
        sample_rows.append({"design": dname, "outcome": okey, "years": f"{d.year.min()}-{d.year.max()}", "counties": d.fips.nunique(),
                            "treated_counties": treated.fips.nunique(), "cohorts": str(cohorts), "rows": len(d)})
        print(f"\n{dname} | {okey}: {d.fips.nunique():,} counties, {treated.fips.nunique()} treated, cohorts {cohorts}, years {d.year.min()}-{d.year.max()}")
        if len(cohorts) == 0:
            continue
        kw = dict(y="y", cohort="cohort", unit="fips", time="year", cluster="state_fips", controls=XCOLS)
        # pyfixest.did2s cannot handle a county with a single untreated row (it drops first-stage singletons): imputation only
        d_imp = d[d.fips.map((d.D == 0).groupby(d.fips).sum()) >= 2]
        sample_rows[-1].update(imputation_counties=d_imp.fips.nunique(), imputation_treated=d_imp[d_imp.cohort.notna()].fips.nunique(),
                               imputation_cohorts=str(sorted(int(c) for c in d_imp.cohort.dropna().unique())))
        runs = {CS: lambda: cs_benchmark(d)[0],
                "TWFE (static, reference)": lambda: est._tri_rows("TWFE (static, reference)", "overall", np.nan,
                    *[float(v) for v in pf.feols("y ~ D + " + " + ".join(XCOLS) + " | fips + year", data=d, vcov={"CRV1": "state_fips"})
                      .tidy().loc["D"][["Estimate", "Std. Error"]]]),
                "LP-DiD": lambda: est.lpdid_estimates(d, **kw),
                "Sun-Abraham": lambda: est.saturated_estimates(d, **kw),
                "Imputation (did2s)": lambda: est.did2s_estimates(d_imp, treat="D", **kw),
                "de Chaisemartin-D'Haultfoeuille": lambda: est.dcdh_estimates(d, y="y", treat="D", unit="fips", time="year",
                                                                              cluster="state_fips", controls=XCOLS),
                "Stacked DiD": lambda: est.stacked_did(d, **kw)}
        tables = {}
        for name, fn in runs.items():
            try:
                tables[name] = fn().assign(design=dname, outcome=okey)
            except Exception as e:                               # a failing estimator is reported, never skipped silently
                fail_rows.append({"design": dname, "outcome": okey, "estimator": name, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                print(f"   FAILED {name}: {type(e).__name__}: {str(e)[:120]}")
        for name, t in tables.items():
            overall_rows.append(t[t.kind == "overall"])
            event_rows.append(t[t.kind == "event"])
        if CS not in tables:
            continue
        cs = tables[CS][tables[CS].kind == "overall"].iloc[0]
        n_agree = 0
        for name in FIVE:
            if name not in tables:
                continue
            o = tables[name][tables[name].kind == "overall"].iloc[0]
            same_sign, inside = bool(np.sign(o.est) == np.sign(cs.est)), bool(cs.ci_lo <= o.est <= cs.ci_hi)
            n_agree += same_sign and inside
            agree_rows.append({"design": dname, "outcome": okey, "estimator": name, "est": o.est, "se": o.se, "cs_est": cs.est,
                               "cs_ci_lo": cs.ci_lo, "cs_ci_hi": cs.ci_hi, "same_sign": same_sign, "inside_cs_ci": inside,
                               "agrees": same_sign and inside})
        tw = tables["TWFE (static, reference)"].iloc[0] if "TWFE (static, reference)" in tables else None
        verdict = ("ESTIMATOR-ROBUST" if n_agree >= 4 else "NOT robust") if (dname, okey) == PRESPECIFIED else "descriptive"
        print(f"   CS {cs.est:+.4g} (95% {cs.ci_lo:+.4g} to {cs.ci_hi:+.4g})" + (f"   TWFE {tw.est:+.4g}" if tw is not None else ""))
        for name in FIVE:
            r = next((x for x in agree_rows if x["design"] == dname and x["outcome"] == okey and x["estimator"] == name), None)
            if r:
                print(f"   {name:34s} {r['est']:+9.4g} (se {r['se']:.3g})   agrees with CS: {'yes' if r['agrees'] else 'no '}"
                      f"  (same sign {r['same_sign']}, inside CI {r['inside_cs_ci']})")
        print(f"   -> {n_agree} of {len(FIVE)} agree   [{verdict}]")

def save(rows, name):
    df_ = pd.concat(rows, ignore_index=True) if rows and isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)
    df_.to_csv(os.path.join(OUT, f"{TODAY}_{name}.csv"), index=False)
    return df_

save(overall_rows, "overall_estimates"); save(event_rows, "event_paths"); save(agree_rows, "agreement")
save(sample_rows, "samples"); save(fail_rows, "failures")
print(f"\nwrote {OUT}/{TODAY}_*.csv   (failures: {len(fail_rows)})")
