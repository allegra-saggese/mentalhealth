"""script4zc-treatment-by-model.py -- every treatment against every model: one tidy table for the forest plots (D-46).

Treatments: T1 (presence, E1), T3 (per 10,000 residents), T4 (log(1 + count)), E2 (first expansion), E3 (first entry).
Models: TWFE, Callaway-Sant'Anna, LP-DiD, Sun-Abraham, imputation, dCDH, stacked DiD. Only E2 and E3 have cohorts, so only they run
under the five cohort-based models; T1, T3, T4 run under TWFE and dCDH.
New estimation here: dCDH (R package DIDmultiplegtDYN) for T1, T3, T4, NOT normalized, i.e. the effect of the treatment changes
counties actually experienced; the average first-change size is stored beside each estimate so a per-unit reading is transparent.
Everything else is read: TWFE T1/T3/T4 from script4a's A2 grid (re-fitted here and checked to 1e-6), E2/E3 for all models from script4za.
Writes:  Data/output/tables/script4zc/YYYY-MM-DD_treatment_by_model.csv
Run:     PYTHONHASHSEED=0 python3 script4zc-treatment-by-model.py [--quick]     (about 30 minutes: dCDH, 7 outcomes x 3 treatments)
Needs:   Rscript with polars and DIDmultiplegtDYN.
"""
import sys
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import norm

import estimators as est
from functions import latest_file_glob
from script4_treatment import (
    load_panel, OUTCOMES, CORE_TREATMENTS, CORE_TREATMENT_LABELS, SPEC_EXTRA_REGRESSORS, COVARIATE_ORDER, tables_dir, os, date,
)

warnings.filterwarnings("ignore")
QUICK = "--quick" in sys.argv
CONTROLS = COVARIATE_ORDER[:8]
XCOLS = [f"x{i + 1}" for i in range(len(CONTROLS))]
DOSE = ["T1", "T3", "T4"]
# dCDH on T4 (log count; many distinct period-one values, so tiny matching groups) fails inside its covariate adjustment for 6 of 7 outcomes
# ("all observations are fixed-effects singletons"), so T4 runs WITHOUT time-varying controls under dCDH for every outcome (D-46).
DCDH_WITH_CONTROLS = {"T1": True, "T3": True, "T4": False}
EFFECTS = 9                                          # as in script4za; the overall is the package's average over effects 1..9
TOLERANCE = 1e-6
MODEL = {"Callaway-Sant'Anna (benchmark)": "Callaway-Sant'Anna", "TWFE (static, reference)": "TWFE", "LP-DiD": "LP-DiD",
         "Sun-Abraham": "Sun-Abraham", "Imputation (did2s)": "Imputation", "de Chaisemartin-D'Haultfoeuille": "dCDH",
         "Stacked DiD": "Stacked DiD"}
LABEL = {"E2": "First expansion (E2)", "E3": "First entry (E3)", "T1": "Presence, 0/1 (T1)",
         "T3": "Per 10,000 residents (T3)", "T4": "Log(1 + count) (T4)"}
OUT = os.path.join(tables_dir, "script4zc" + ("_quick" if QUICK else ""))
os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")


def row(model, tid, outcome, b, se, source, note=""):
    z = b / se if se else np.nan
    return {"model": model, "treatment_id": tid, "treatment": LABEL[tid], "outcome": outcome, "est": b, "se": se,
            "p": float(2 * (1 - norm.cdf(abs(z)))) if se else np.nan,
            "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se, "source": source, "note": note}


def first_change(d):
    """Average first change in the treatment over counties whose treatment changes at all (signed, absolute, count)."""
    d = d.sort_values(["fips", "year"])
    step = d.groupby("fips").treat.diff()
    first = step[step.fillna(0) != 0].groupby(d.fips[step.fillna(0) != 0]).first()
    return float(first.mean()), float(first.abs().mean()), int(first.size)


grid = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4a"), "*_A1_A2_treatment_grid.csv"))
a2 = grid[grid.registry == "A2"].set_index(["treatment_id", "outcome"])
tri = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4za"), "*_overall_estimates.csv"))
df = load_panel()
outcomes = dict(list(OUTCOMES.items())[:1]) if QUICK else OUTCOMES

rows, failures = [], []
print("=" * 78)
print(f"script4zc -- dCDH for T1, T3, T4 (not normalized, effects 1..{EFFECTS}); TWFE rows re-fitted and checked against the A2 grid")
print("=" * 78)
for okey, ocol in outcomes.items():
    for tid in DOSE:
        tcol, extras = CORE_TREATMENTS[tid], SPEC_EXTRA_REGRESSORS.get(tid, [])
        d = df[["fips", "year", "state_fips", ocol, tcol] + extras + CONTROLS].dropna().copy()
        d = d.rename(columns={ocol: "y", tcol: "treat", **dict(zip(CONTROLS, XCOLS))})
        d["fips"], d["state_fips"] = d.fips.astype(int), pd.factorize(d.state_fips)[0]
        d.attrs = {}
        # TWFE: the A2 grid's own regression, re-fitted here so the row is checked, not merely copied
        fit = pf.feols("y ~ treat" + "".join(f" + {c}" for c in extras + XCOLS) + " | fips + year", data=d,
                       vcov={"CRV1": "state_fips"}).tidy().loc["treat"]
        ref = a2.loc[(tid, okey)]
        assert np.isclose(fit["Estimate"], ref["beta"], rtol=TOLERANCE, atol=1e-8), f"TWFE beta differs from the A2 grid: {tid} {okey}"
        assert np.isclose(fit["Pr(>|t|)"], ref["p_state"], rtol=TOLERANCE, atol=1e-8), f"TWFE p differs from the A2 grid: {tid} {okey}"
        rows.append(row("TWFE", tid, okey, float(fit["Estimate"]), float(fit["Std. Error"]), "script4a A2 grid (re-fitted, matches)"))
        # dCDH: the effect of the treatment changes counties actually experienced
        mean_chg, mean_abs, n_chg = first_change(d)
        try:
            t = est.dcdh_estimates(d, y="y", treat="treat", unit="fips", time="year", cluster="state_fips",
                                   controls=(extras + XCOLS) if DCDH_WITH_CONTROLS[tid] else [], effects=EFFECTS, placebos=0,
                                   normalized=False)
            o = t[t.kind == "overall"].iloc[0]
            rows.append(row("dCDH", tid, okey, float(o.est), float(o.se), "script4zc (new dCDH run, not normalized" + ("" if DCDH_WITH_CONTROLS[tid] else "; no controls") + ")",
                            f"average first change {mean_chg:+.4g} (absolute {mean_abs:.4g}) over {n_chg} changing counties"))
            print(f"  {okey:26s} {tid}: TWFE {fit['Estimate']:+.4g} | dCDH {o.est:+.4g} (se {o.se:.3g}); "
                  f"average first change {mean_chg:+.3g}, {n_chg} changing counties", flush=True)
        except Exception as e:
            failures.append({"outcome": okey, "treatment_id": tid, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"  {okey} {tid}: dCDH FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)

# E2 (first expansion) and E3 (first entry) for every model: from script4za
for _, r in tri.iterrows():
    tid = "E2" if r.design == "first expansion" else "E3"
    rows.append(row(MODEL[r.estimator], tid, r.outcome, float(r.est), float(r.se), "script4za overall estimate"))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, f"{TODAY}_treatment_by_model.csv"), index=False)
pd.DataFrame(failures).to_csv(os.path.join(OUT, f"{TODAY}_failures.csv"), index=False)
print(f"\n{len(res)} rows ({len(failures)} dCDH failures) written to {OUT}/{TODAY}_treatment_by_model.csv")
