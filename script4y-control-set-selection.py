#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4y-control-set-selection.py  --  HOW THE CONTROL SET WAS CHOSEN: ADD ONE IN, TAKE ONE OUT  (added 2026-10-17)

Replaces script4b (adding controls along a fixed order) and script4c (summarising that run), and adds the
take-one-out checks. 4b and 4c are kept, unchanged, as the record of the earlier 09-23 run.

THE CONTROL SET. 14 candidate controls (COVARIATE_ORDER); the 8 with full coverage and testable pre-trends are the
final control set (COVARIATE_ORDER[:8]). That choice was PRE-COMMITTED (decision D-16, revised in D-31): it was fixed
before any coefficient from this script was read, on grounds of coverage, causal role (confounder, mediator, collider)
and whether the series can be tested for pre-trends. THIS SCRIPT DOES NOT SELECT THE CONTROL SET. It documents how much
each control matters for the treatment coefficient. Choosing a specification because of what these paths show and then
quoting conventional standard errors would be post-selection inference.

FOUR FAMILIES of regressions, for each reported treatment (T1, T3, T4, E2) x outcome (7), all with county FE +
year FE and SE clustered by state:
    A1  cumulative add-one-in      controls enter one at a time in COVARIATE_ORDER (step 0 = none ... 14 = all).
                                   Order-dependent; the sample shrinks after step 8 (partial-coverage controls).
    A2  each added alone           the reference has no controls; each of the 14 is added by itself.
                                   Order-free.
    B1  take-one-out of the 8      reference = the final 8; drop each of them in turn.
    B2  take-one-out of all 14     reference = all 14; drop each in turn.
SAMPLE. "own" = complete cases for that regression's variables. "common14" = complete cases for ALL 14 controls, so
differences between regressions are due to the control and not to a changing sample (A2 and B2 only; for A1 and B1
the first 8 controls leave the sample intact).
Every regression stores beta, SE, p, N, switchers (counties whose treatment changes) and the largest within-transformed
VIF. Change is measured against that family's reference: 100 * (beta - beta_ref) / |beta_ref|.

OUTPUTS -> Data/output/tables/script4y/
    <date>_regressions.csv             every regression
    <date>_path_summary.csv            A1: per treatment x outcome, movement from step 0 to 8 to 14
    <date>_covariate_summary.csv       per covariate: median |change| across the 28 treatment x outcome cells, by family
    <date>_selection_table.csv         the exhibit: one row per candidate control with coverage, causal role,
                                       pre-trend verdict (script4j), reason for exclusion, and the movement above
Usage:  python3 script4y-control-set-selection.py            full
        python3 script4y-control-set-selection.py --quick    one treatment, two outcomes
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyfixest as pf

from script4_treatment import (load_panel, OUTCOMES, COVARIATE_ORDER, COVARIATE_EXCLUDED, CORE_TREATMENTS,
                               SPEC_EXTRA_REGRESSORS, tables_dir, os, date)

QUICK = "--quick" in sys.argv
OUT = os.path.join(tables_dir, "script4y" + ("_quick" if QUICK else "")); os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
ALL14 = COVARIATE_ORDER
FINAL8 = COVARIATE_ORDER[:8]
TREATS = {"T1": CORE_TREATMENTS["T1"], "T3": CORE_TREATMENTS["T3"], "T4": CORE_TREATMENTS["T4"], "E2": "tr_e2_add_absorb"}
if QUICK:
    TREATS = {"T1": TREATS["T1"]}
OUTS = dict(list(OUTCOMES.items())[:2]) if QUICK else OUTCOMES
EXTRA = {"T4": ["log_pop"]}                     # part of the specification, not a candidate control
DF = None


def clean(n):
    return (n.replace("%", "pct").replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")
             .replace("/", "_").replace("+", "p").replace(",", "_"))


def _init():
    global DF
    DF = load_panel().sort_values(["fips", "year"]).reset_index(drop=True)


def vif_max(frame, cols):
    """Largest VIF among `cols` after removing county and year means (the variation a FE model uses)."""
    if len(cols) < 2:
        return np.nan
    W = frame[cols].astype(float).copy()
    for c in cols:
        W[c] = W[c] - frame.groupby("fips")[c].transform("mean") - frame.groupby("year")[c].transform("mean") + W[c].mean()
    keep = [c for c in cols if W[c].std() > 1e-10]
    if len(keep) < 2:
        return np.nan
    try:
        return float(np.diag(np.linalg.pinv(np.corrcoef(W[keep].values, rowvar=False))).max())
    except Exception:
        return np.nan


def run_cell(args):
    tid, tcol, okey, ocol = args
    extra = EXTRA.get(tid, [])
    need = list(dict.fromkeys(["fips", "year", "state_fips", ocol, tcol] + extra + ALL14))
    data = DF[need]
    common = data.dropna().index                                     # complete cases for all 14 controls
    rows = []

    def fit(family, label, covs, sample, ref=False):
        cols = list(dict.fromkeys(["fips", "year", "state_fips", ocol, tcol] + extra + covs))
        sub = data.loc[common, cols] if sample == "common14" else data[cols].dropna()
        if len(sub) < 500 or sub[tcol].nunique() < 2:
            return
        s2 = sub.rename(columns={c: clean(c) for c in sub.columns})
        rhs = [clean(tcol)] + [clean(x) for x in extra] + [clean(c) for c in covs]
        try:
            m = pf.feols(f"{clean(ocol)} ~ " + " + ".join(rhs) + " | fips + year", data=s2, vcov={"CRV1": "state_fips"})
            t = m.tidy().loc[clean(tcol)]
        except Exception:
            return
        rows.append({"treatment": tid, "outcome": okey, "family": family, "sample": sample, "label": label,
                     "n_controls": len(covs), "is_reference": ref, "beta": float(t["Estimate"]),
                     "se": float(t["Std. Error"]), "p": float(t["Pr(>|t|)"]), "N": int(m._N),
                     "n_switchers": int((sub.groupby("fips")[tcol].nunique() > 1).sum()),
                     "vif_max": vif_max(sub, [tcol] + extra + covs)})

    for step in range(len(ALL14) + 1):                                              # A1
        fit("A1 cumulative", f"step {step}" + (f": +{ALL14[step - 1]}" if step else ""), ALL14[:step], "own", ref=(step == 0))
    for smp in ("own", "common14"):
        fit("A2 added alone", "none", [], smp, ref=True)                              # A2
        for c in ALL14:
            fit("A2 added alone", c, [c], smp)
    fit("B1 drop one of final 8", "all 8", FINAL8, "own", ref=True)                   # B1
    for c in FINAL8:
        fit("B1 drop one of final 8", c, [x for x in FINAL8 if x != c], "own")
    for smp in ("own", "common14"):
        fit("B2 drop one of all 14", "all 14", ALL14, smp, ref=True)                  # B2
        for c in ALL14:
            fit("B2 drop one of all 14", c, [x for x in ALL14 if x != c], smp)
    return rows


if __name__ == "__main__":
    tasks = [(tid, tcol, ok, oc) for tid, tcol in TREATS.items() for ok, oc in OUTS.items()]
    allrows = []
    with ProcessPoolExecutor(max_workers=min(len(tasks), 8), initializer=_init) as ex:
        for i, rows in enumerate(ex.map(run_cell, tasks), 1):
            allrows += rows
            print(f"  cell {i}/{len(tasks)} done ({tasks[i-1][0]} | {tasks[i-1][2]})", flush=True)
    R = pd.DataFrame(allrows)
    key = ["treatment", "outcome", "family", "sample"]
    ref = R[R.is_reference].set_index(key)[["beta", "p"]].rename(columns={"beta": "beta_ref", "p": "p_ref"})
    R = R.join(ref, on=key)
    R["pct_change_vs_ref"] = 100 * (R.beta - R.beta_ref) / R.beta_ref.abs()
    R["sign_flip"] = np.sign(R.beta) != np.sign(R.beta_ref)
    R["sig_flip"] = (R.p < .05) != (R.p_ref < .05)
    R.to_csv(os.path.join(OUT, f"{TODAY}_regressions.csv"), index=False)

    # ---------------- A1 path summary (the old script4c) ----------------
    a1 = R[R.family == "A1 cumulative"].copy()
    a1["step"] = a1.label.str.extract(r"step (\d+)")[0].astype(int)
    ps = []
    for (t, o), g in a1.groupby(["treatment", "outcome"]):
        g = g.sort_values("step").set_index("step")
        if 0 not in g.index:
            continue
        b0 = g.loc[0, "beta"]
        step_move = g.beta.diff().abs() / abs(b0) * 100
        ps.append({"treatment": t, "outcome": o, "beta_step0": b0, "beta_step8": g.beta.get(8, np.nan),
                   "beta_step14": g.beta.get(14, np.nan),
                   "change_0_to_8_pct": 100 * (g.beta.get(8, np.nan) - b0) / abs(b0),
                   "change_0_to_14_pct": 100 * (g.beta.get(14, np.nan) - b0) / abs(b0),
                   "sign_flips_along_path": int((np.sign(g.beta).diff().abs() > 0).sum()),
                   "sig_changes_along_path": int(((g.p < .05).astype(int).diff().abs() > 0).sum()),
                   "largest_single_step": int(step_move.idxmax()) if step_move.notna().any() else np.nan,
                   "largest_single_step_pct_of_step0": float(step_move.max()) if step_move.notna().any() else np.nan,
                   "N_step8": g.N.get(8, np.nan), "N_step14": g.N.get(14, np.nan)})
    pd.DataFrame(ps).to_csv(os.path.join(OUT, f"{TODAY}_path_summary.csv"), index=False)

    # ---------------- per-covariate movement ----------------
    alt = R[~R.is_reference & ~R.family.str.startswith("A1")].copy()
    cs = (alt.groupby(["family", "sample", "label"])
             .agg(median_abs_pct_change=("pct_change_vs_ref", lambda s: float(s.abs().median())),
                  max_abs_pct_change=("pct_change_vs_ref", lambda s: float(s.abs().max())),
                  sign_flips=("sign_flip", "sum"), sig_flips=("sig_flip", "sum"), cells=("beta", "size")).reset_index()
             .rename(columns={"label": "covariate"}))
    cs.to_csv(os.path.join(OUT, f"{TODAY}_covariate_summary.csv"), index=False)

    # ---------------- the selection table ----------------
    panel = load_panel()
    cov = pd.DataFrame({"covariate": ALL14, "order": range(1, 15),
                        "coverage_pct": [100 * panel[c].notna().mean() for c in ALL14],
                        "in_final_8": [c in FINAL8 for c in ALL14]})
    roster = sorted(glob.glob(os.path.join(tables_dir, "script4a", "*_control_pretrends.csv")))
    if roster:
        pt = pd.read_csv(roster[-1])[["variable", "causal_role", "n_pre_significant", "n_pre_coefs", "verdict"]]
        cov = cov.merge(pt.rename(columns={"variable": "covariate", "verdict": "pretrend_verdict"}), on="covariate", how="left")
    cov["excluded_because"] = cov.covariate.map(COVARIATE_EXCLUDED).fillna("")

    def pick(family, sample, col):
        x = cs[(cs.family == family) & (cs["sample"] == sample)].set_index("covariate")[col]
        return cov.covariate.map(x)
    cov["added_alone_median_abs_pct_change"] = pick("A2 added alone", "own", "median_abs_pct_change")
    cov["added_alone_common_sample"] = pick("A2 added alone", "common14", "median_abs_pct_change")
    cov["dropped_from_8_median_abs_pct_change"] = pick("B1 drop one of final 8", "own", "median_abs_pct_change")
    cov["dropped_from_14_median_abs_pct_change"] = pick("B2 drop one of all 14", "own", "median_abs_pct_change")
    cov["dropped_from_14_common_sample"] = pick("B2 drop one of all 14", "common14", "median_abs_pct_change")
    cov.to_csv(os.path.join(OUT, f"{TODAY}_selection_table.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nSELECTION TABLE  (median |% change in the treatment coefficient| across treatment x outcome cells)")
    show = cov[["order", "covariate", "coverage_pct", "in_final_8", "causal_role", "pretrend_verdict",
                "added_alone_median_abs_pct_change", "dropped_from_8_median_abs_pct_change",
                "dropped_from_14_median_abs_pct_change"]].round(1)
    show["covariate"] = show.covariate.str.slice(0, 34)
    print(show.to_string(index=False))
    p = pd.DataFrame(ps)
    print("\nA1 cumulative path: change in beta from step 0 to step 8 and to step 14 (median over outcomes, per treatment)")
    print(p.groupby("treatment")[["change_0_to_8_pct", "change_0_to_14_pct", "sign_flips_along_path", "sig_changes_along_path"]]
          .median().round(1).to_string())
    print(f"\n{len(R):,} regressions. Saved -> {OUT}")
