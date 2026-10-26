#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4u-did-by-treatment.py -- TWFE, event study and Callaway-Sant'Anna for first expansion and first entry, with pre-trend adjustments.

Designs: first expansion (all rural counties); first entry (entry counties plus never-large counties as controls; D-32).
Adjustments: county trends, state x year FE, baseline characteristics x time polynomial, past outcomes in CS.
Reads:   panel via designs.load_designs()
Writes:  Data/output/tables/script4u/<date>_{static_twfe, event_study, event_study_leads_joint, cs_att_gt, cs_aggregated, validation_vs_script4a}.csv
Run:     python3 script4u-did-by-treatment.py   (about 30 minutes; --no-cs skips CS; --quick is a test run)
Method:  METHODS.md sections 2-3. The first-expansion baseline is checked against script4a; the CS code is in estimators.py.
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import sys

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import chi2

from estimators import run_cs, clean, joint_wald, fe_ols, event_study_frame, stars as st
from designs import load_designs
from script4_treatment import OUTCOMES, COVARIATE_ORDER, tables_dir, os, date

QUICK = "--quick" in sys.argv
NO_CS = "--no-cs" in sys.argv
N_BOOT = 20 if QUICK else 300
OUT = os.path.join(tables_dir, "script4u" + ("_quick" if QUICK else "")); os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
CONTROLS = COVARIATE_ORDER[:8]
CLIP = 8
OUTCOME_LIST = dict(list(OUTCOMES.items())[:2]) if QUICK else OUTCOMES


df, DESIGNS, POLY_COLS, _counts = load_designs(with_polynomial=True)
PF_SPECS = [("baseline", "fips + year", []),
            ("state x year FE", "fips + state_fips^year", []),
            ("baseline chars x time polynomial", "fips + year", POLY_COLS)]

print(f"entry design: {_counts['n_entry']} entry counties + {_counts['n_never_large']} never-large counties "
      f"= {_counts['n_entry_sample']} of {_counts['n_all']} counties")

static_rows, es_rows, joint_rows, gt_all, agg_all, fe_checks = [], [], [], [], [], []

for dname, D in DESIGNS.items():
    data, treat, cohort = D["data"], D["treat"], D["cohort"]
    print(f"\n##### {dname}: {data[cohort].notna().groupby(data.fips).first().sum() if False else data.loc[data[cohort].notna(),'fips'].nunique()} treated counties")
    for okey, ocol in OUTCOME_LIST.items():
        # ----------------------------- static TWFE -----------------------------
        for spec, fe, extra in PF_SPECS:
            need = list(dict.fromkeys(["fips", "year", "state_fips", ocol, treat] + CONTROLS + extra))
            sub = data[need].dropna().rename(columns={c: clean(c) for c in need})
            nsw = int((sub.groupby("fips")[clean(treat)].nunique() > 1).sum())
            xc = [clean(treat)] + [clean(c) for c in CONTROLS] + [clean(c) for c in extra]
            f = f"{clean(ocol)} ~ " + " + ".join(xc) + f" | {fe}"
            try:
                m = pf.feols(f, data=sub, vcov={"CRV1": "state_fips"}); t = m.tidy().loc[clean(treat)]
                static_rows.append({"design": dname, "spec": spec, "outcome": okey, "beta": float(t["Estimate"]),
                                    "se_state": float(t["Std. Error"]), "p_state": float(t["Pr(>|t|)"]),
                                    "N": int(m._N), "n_switchers": nsw})
                if spec == "baseline":
                    # internal check: fe_ols without trends must reproduce pyfixest
                    chk, _, _ = fe_ols(sub.reset_index(drop=True), clean(ocol), xc, trend=False)
                    fe_checks.append({"design": dname, "outcome": okey,
                                      "beta_diff": abs(chk.loc[clean(treat), "beta"] - float(t["Estimate"])),
                                      "se_diff": abs(chk.loc[clean(treat), "se"] - float(t["Std. Error"])) / float(t["Std. Error"])})
                    res, _, N_ = fe_ols(sub.reset_index(drop=True), clean(ocol), xc, trend=True)
                    r = res.loc[clean(treat)]
                    static_rows.append({"design": dname, "spec": "county trends", "outcome": okey, "beta": float(r.beta),
                                        "se_state": float(r.se), "p_state": float(r.p), "N": int(N_), "n_switchers": nsw})
            except Exception as e:
                print(f"  static [{dname}|{spec}|{okey}] failed: {type(e).__name__}: {e}")
        # ----------------------------- event study -----------------------------
        for spec, fe, extra in PF_SPECS:
            try:
                es = event_study_frame(data, ocol, CONTROLS, extra, CLIP)
                f = (f"{clean(ocol)} ~ i(ev, ref=-1) + " + " + ".join(clean(c) for c in CONTROLS + extra) + f" | {fe}")
                m = pf.feols(f, data=es, vcov={"CRV1": "state_fips"}); tid = m.tidy()
                leads = []
                for idx, r in tid.iterrows():
                    if not str(idx).startswith("ev::"):
                        continue
                    e = int(float(str(idx).split("::")[1]))
                    es_rows.append({"design": dname, "spec": spec, "outcome": okey, "event_time": e,
                                    "beta": float(r["Estimate"]), "se_state": float(r["Std. Error"]),
                                    "p_state": float(r["Pr(>|t|)"]), "N": int(m._N)})
                    if e < -1:
                        leads.append(str(idx))
                stat, k, p = joint_wald(m, leads)
                joint_rows.append({"design": dname, "spec": spec, "outcome": okey, "n_leads": k,
                                   "wald_chi2": stat, "joint_p": p, "N": int(m._N)})
            except Exception as e:
                print(f"  event [{dname}|{spec}|{okey}] failed: {type(e).__name__}: {e}")
        # county trends: event-time dummies built by hand, residualized with county trends
        try:
            es = event_study_frame(data, ocol, CONTROLS, [], CLIP)
            evs = sorted(int(x) for x in es["ev"].unique() if x != -1)
            D_ = pd.DataFrame({f"ev_{e}": (es["ev"] == e).astype(float) for e in evs}, index=es.index)
            ee = pd.concat([es.reset_index(drop=True), D_.reset_index(drop=True)], axis=1)
            xc = list(D_.columns) + [clean(c) for c in CONTROLS]
            res, Vc, N_ = fe_ols(ee, clean(ocol), xc, trend=True)
            lead_idx = [i for i, e in enumerate(evs) if e < -1]
            bl = res.beta.to_numpy()[lead_idx]; Vl = Vc[np.ix_(lead_idx, lead_idx)]
            stat = float(bl @ np.linalg.solve(Vl, bl))
            for e in evs:
                r = res.loc[f"ev_{e}"]
                es_rows.append({"design": dname, "spec": "county trends", "outcome": okey, "event_time": e,
                                "beta": float(r.beta), "se_state": float(r.se), "p_state": float(r.p), "N": int(N_)})
            joint_rows.append({"design": dname, "spec": "county trends", "outcome": okey, "n_leads": len(lead_idx),
                               "wald_chi2": stat, "joint_p": float(1 - chi2.cdf(stat, len(lead_idx))), "N": int(N_)})
        except Exception as e:
            print(f"  event [{dname}|county trends|{okey}] failed: {type(e).__name__}: {e}")
        # ----------------------------- Callaway-Sant'Anna -----------------------------
        for spec, lags in ([] if NO_CS else [("controls", 0), ("controls + past outcomes", 2)]):
            gt, ag = run_cs(data, ocol, cohort, CONTROLS, n_boot=N_BOOT, outcome_lags=lags)
            if len(gt):
                gt.insert(0, "outcome", okey); gt.insert(0, "spec", spec); gt.insert(0, "design", dname)
                ag.insert(0, "outcome", okey); ag.insert(0, "spec", spec); ag.insert(0, "design", dname)
                gt_all.append(gt); agg_all.append(ag)
        print(f"  done: {okey}", flush=True)

static = pd.DataFrame(static_rows); es_df = pd.DataFrame(es_rows); joint = pd.DataFrame(joint_rows)
outputs = [("static_twfe", static), ("event_study", es_df), ("event_study_leads_joint", joint)]
if not NO_CS:
    gt_df = pd.concat(gt_all, ignore_index=True); agg_df = pd.concat(agg_all, ignore_index=True)
    outputs += [("cs_att_gt", gt_df), ("cs_aggregated", agg_df)]
for name, x in outputs:
    x.to_csv(os.path.join(OUT, f"{TODAY}_{name}.csv"), index=False)

# ---------------------------- validation against script4a (first expansion, baseline) ------------------
val = []
A4 = os.path.join(tables_dir, "script4a")
def newest(pat):
    f = sorted(glob.glob(os.path.join(A4, pat)))
    return pd.read_csv(f[-1]) if f else None
grid, a5, a6 = newest("*_A1_A2_treatment_grid.csv"), newest("*_A5_event_study.csv"), newest("*_A6_cs_aggregated.csv")
mine = "first expansion"
if grid is not None:
    g = grid[(grid.registry == "A2") & (grid.treatment_id == "E2")].set_index("outcome")
    s = static[(static.design == mine) & (static.spec == "baseline")].set_index("outcome")
    common = s.index.intersection(g.index)
    val.append({"check": "static TWFE beta", "max_abs_diff": float((s.loc[common, "beta"] - g.loc[common, "beta"]).abs().max()),
                "n": len(common)})
    val.append({"check": "static TWFE se", "max_abs_diff": float((s.loc[common, "se_state"] - g.loc[common, "se_state"]).abs().max()),
                "n": len(common)})
if a5 is not None:
    a = a5[a5.registry == "A5"][["outcome", "event_time", "beta"]]
    m_ = es_df[(es_df.design == mine) & (es_df.spec == "baseline")].merge(a, on=["outcome", "event_time"], suffixes=("", "_4a"))
    val.append({"check": "event study beta", "max_abs_diff": float((m_.beta - m_.beta_4a).abs().max()), "n": len(m_)})
if a6 is not None and not NO_CS:
    a = a6[(a6.arm == "covariate-adjusted") & (a6.agg_level == "overall_ATT")][["outcome", "att", "se_state"]]
    m_ = agg_df[(agg_df.design == mine) & (agg_df.spec == "controls") & (agg_df.agg_level == "overall_ATT")] \
        .merge(a, on="outcome", suffixes=("", "_4a"))
    val.append({"check": "CS overall ATT", "max_abs_diff": float((m_.att - m_.att_4a).abs().max()), "n": len(m_)})
    val.append({"check": "CS overall ATT bootstrap se", "max_abs_diff": float((m_.se_state - m_.se_state_4a).abs().max()), "n": len(m_)})
fc = pd.DataFrame(fe_checks)
val.append({"check": "fe_ols (no trend) vs pyfixest, static beta", "max_abs_diff": float(fc.beta_diff.max()), "n": len(fc)})
val.append({"check": "fe_ols (no trend) vs pyfixest, static se (RELATIVE diff)", "max_abs_diff": float(fc.se_diff.max()), "n": len(fc)})
val = pd.DataFrame(val)
val.to_csv(os.path.join(OUT, f"{TODAY}_validation_vs_script4a.csv"), index=False)


print("\n" + "=" * 100 + "\nVALIDATION: first-expansion baseline vs script4a (should be ~0)\n" + "=" * 100)
print(val.to_string(index=False))
print("\n" + "=" * 100 + "\nSTATIC TWFE   beta (stars)   [county FE + year FE, SE clustered by state]\n" + "=" * 100)
for dname in DESIGNS:
    print(f"\n{dname}")
    for o in OUTCOME_LIST:
        r = static[(static.design == dname) & (static.outcome == o)]
        print(f"  {o[:24]:24s} " + " | ".join(f"{x.spec}: {x.beta:+.3g}{st(x.p_state)} (z{x.beta / x.se_state:+.1f})" for x in r.itertuples())
              + f" | N={int(r.N.iloc[0]):,} switchers={int(r.n_switchers.iloc[0])}")
print("\n" + "=" * 100 + "\nEVENT STUDY: joint test that all leads = 0 (p)\n" + "=" * 100)
print(joint.pivot_table(index=["design", "outcome"], columns="spec", values="joint_p").round(3).to_string())
print("\n" + "=" * 100 + "\nCS OVERALL ATT   att (stars) [bootstrap over states]\n" + "=" * 100)
ov = agg_df[agg_df.agg_level == "overall_ATT"] if not NO_CS else pd.DataFrame(columns=["design", "outcome"])
for dname in DESIGNS:
    print(f"\n{dname}")
    for o in OUTCOME_LIST:
        r = ov[(ov.design == dname) & (ov.outcome == o)]
        print(f"  {o[:24]:24s} " + " | ".join(f"{x.spec}: {x.att:+.3g}{st(x.p_state)} (z{x.att / x.se_state:+.1f}, fb={int(x.n_fallback_cells)})" for x in r.itertuples()))
print(f"\nSaved -> {OUT}")
