"""script4d-triangulation.py -- is the Callaway-Sant'Anna result specific to Callaway-Sant'Anna, and how do all treatments compare across models?

Sections (run in this order; `--only` selects one):
  estimators          = old script4za. Callaway-Sant'Anna (CS) benchmark + TWFE, LP-DiD, Sun-Abraham, imputation (did2s), de Chaisemartin-
                        D'Haultfoeuille (dCDH, R package), stacked DiD on ONE complete-case sample per (design, outcome). Rule fixed before the run
                        (D-45): an estimator agrees with CS if its overall post-treatment ATT has the same sign and lies inside CS's 95% interval;
                        the Poor MH Days first-expansion result is called estimator-robust if at least 4 of the 5 agree. Reported either way.
                        Writes Data/output/tables/script4za/YYYY-MM-DD_{overall_estimates,event_paths,agreement,samples,failures}.csv
  treatment-by-model  = old script4zc. Every treatment (T1, T3, T4, E2 first expansion, E3 first entry) against every model (D-46): one tidy table
                        for the forest plots. New estimation: dCDH for T1, T3, T4, NOT normalized. Everything else is read: TWFE T1/T3/T4 from
                        script4a's A2 grid (re-fitted and checked to 1e-6), E2/E3 for all models from the `estimators` section's overall estimates.
                        Writes Data/output/tables/script4zc/YYYY-MM-DD_{treatment_by_model,failures}.csv
Run:    PYTHONHASHSEED=0 python3 script4d-triangulation.py [--only estimators|treatment-by-model] [--quick] [--outcome "Poor MH Days"] [--design expansion|entry]
        `--outcome` and `--design` apply to `estimators` only (as in the old scripts); `--quick` writes to the *_quick folders.
Order:  `treatment-by-model` needs script4a (A1_A2 grid) and the `estimators` section already run (it reads the newest script4za overall estimates).
Plan:   code-audit/plans/2026-10-22_PLAN_estimator-triangulation.md.  Adapters and stacked DiD: estimators.py; sample builder: designs.usable_sample.
Needs:  Rscript with polars and DIDmultiplegtDYN. Runtime: about 20 minutes (`estimators`, 7 outcomes x 2 designs) + 30 minutes (`treatment-by-model`).
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import norm

import estimators as est
from designs import load_designs, usable_sample
from estimators import run_cs
from functions import latest_file_glob
from script4_treatment import (
    load_panel, OUTCOMES, CORE_TREATMENTS, CORE_TREATMENT_LABELS, SPEC_EXTRA_REGRESSORS, COVARIATE_ORDER, tables_dir, os, date,
)

warnings.filterwarnings("ignore")
SECTIONS = ["estimators", "treatment-by-model"]
ARGV = sys.argv[1:]
QUICK = "--quick" in ARGV
ONLY_OUTCOMES = [ARGV[i + 1] for i, a in enumerate(ARGV) if a == "--outcome"]
ONLY_DESIGN = next((ARGV[i + 1] for i, a in enumerate(ARGV) if a == "--design"), None)
CONTROLS = COVARIATE_ORDER[:8]
XCOLS = [f"x{i + 1}" for i in range(len(CONTROLS))]
TODAY = date.today().strftime("%Y-%m-%d")

# ---- section `estimators` (old script4za)
N_BOOT = 20 if QUICK else 300                   # CS state bootstrap, as in script4a/4u
CS_SEED = 20260928
CS = "Callaway-Sant'Anna (benchmark)"
FIVE = ["LP-DiD", "Sun-Abraham", "Imputation (did2s)", "de Chaisemartin-D'Haultfoeuille", "Stacked DiD"]
PRESPECIFIED = ("first expansion", "Poor MH Days")      # the one cell D-45 applies to

# ---- section `treatment-by-model` (old script4zc)
DOSE = ["T1", "T3", "T4"]
# dCDH on T4 (log count; many distinct period-one values, so tiny matching groups) fails inside its covariate adjustment for 6 of 7 outcomes
# ("all observations are fixed-effects singletons"), so T4 runs WITHOUT time-varying controls under dCDH for every outcome (D-46).
DCDH_WITH_CONTROLS = {"T1": True, "T3": True, "T4": False}
EFFECTS = 9                                          # as in the `estimators` section; the overall is the package's average over effects 1..9
TOLERANCE = 1e-6
MODEL = {"Callaway-Sant'Anna (benchmark)": "Callaway-Sant'Anna", "TWFE (static, reference)": "TWFE", "LP-DiD": "LP-DiD",
         "Sun-Abraham": "Sun-Abraham", "Imputation (did2s)": "Imputation", "de Chaisemartin-D'Haultfoeuille": "dCDH",
         "Stacked DiD": "Stacked DiD"}
LABEL = {"E2": "First expansion (E2)", "E3": "First entry (E3)", "T1": "Presence, 0/1 (T1)",
         "T3": "Per 10,000 residents (T3)", "T4": "Log(1 + count) (T4)"}


def output_dir(stem):
    """Output folder of a section (`script4za`, `script4zc`; `_quick` variants), created on demand."""
    out = os.path.join(tables_dir, stem + ("_quick" if QUICK else ""))
    os.makedirs(out, exist_ok=True)
    return out


# =============================================================================================================================================
# SECTION estimators  (old script4za-estimator-triangulation.py, code unchanged except that the module-level run is wrapped in a function)
# =============================================================================================================================================
def cs_benchmark(d):
    """estimators.run_cs on the sample; overall ATT and event-time ATTs with the state-bootstrap SE."""
    _, agg = run_cs(d, "y", "cohort", XCOLS, n_boot=N_BOOT, seed=CS_SEED)
    ov = agg[agg.agg_level == "overall_ATT"].iloc[0]
    ev = agg[agg.agg_level == "event_time"]
    over = est._tri_rows(CS, "overall", np.nan, ov["att"], ov["se_state"])
    events = est._tri_rows(CS, "event", ev["event_time"].astype(float).values, ev["att"].values, ev["se_state"].values)
    return pd.concat([over, events], ignore_index=True), int(ov["n_cohorts"])


def run_estimators():
    OUT = output_dir("script4za")
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


# =============================================================================================================================================
# SECTION treatment-by-model  (old script4zc-treatment-by-model.py, code unchanged except that the module-level run is wrapped in a function)
# =============================================================================================================================================
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


def run_treatment_by_model():
    OUT = output_dir("script4zc")
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



def main():
    parser = argparse.ArgumentParser(description="Estimator triangulation and treatment-by-model table (D-45, D-46).")
    parser.add_argument("--only", choices=SECTIONS, help="run one section (default: both, in order)")
    parser.add_argument("--quick", action="store_true", help="first outcome only, 20 bootstrap draws, *_quick output folders")
    parser.add_argument("--outcome", action="append", help="`estimators` only: restrict to this outcome (repeatable)")
    parser.add_argument("--design", choices=["expansion", "entry"], help="`estimators` only: restrict to one design")
    args = parser.parse_args()
    for section in ([args.only] if args.only else SECTIONS):
        {"estimators": run_estimators, "treatment-by-model": run_treatment_by_model}[section]()


if __name__ == "__main__":
    main()
