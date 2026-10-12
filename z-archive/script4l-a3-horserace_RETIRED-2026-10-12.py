#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4l-a3-horserace.py  --  MODEL A3. Dairy conditional on other large CAFO types.

THE QUESTION
------------
Is the dairy coefficient DAIRY, or is it "this county has a large CAFO"?
A3 re-runs each dairy treatment while conditioning on the PRESENCE of other
large CAFO types, and reads how far beta moves.

DESIGN
------
    treatment   T1-T4                    (4)
    outcome     7                        (7)
    FE          A1 pooled / A2 within    (2)
    subset      all 2^3 combinations of {cattle, hogs, chickens}   (8)
                                                         --------
                                            4 x 7 x 2 x 8 = 448 regressions

The EMPTY subset reproduces A2 (or A1) exactly and is the reference beta that
every other subset is compared against -- so the move attributable to the
conditioning is read within a path, not against a separately-estimated model.

Covariates are held FIXED at the pre-committed step-8 set (D-16) throughout, so
the only thing varying across the 8 subsets is which other animals are
conditioned on. Sample is intact at step 8 (no SAHIE/ACS truncation), which
matters here because the conditioning variables are themselves ~90% covered and
would otherwise compound two sample restrictions.

WHY PRESENCE, WHY DISAGGREGATED, WHY NO BEEF
--------------------------------------------
See A3_CONDITIONING / A3_EXCLUDED in script4_treatment.py. Briefly: a binary
indicator means the same thing against a binary, count, per-capita or logged
treatment; a pooled "any other large CAFO" has NO counterfactual (0 counties
have a large dairy and no other large CAFO); and beef is a strict subset of
cattle so entering both double-counts.

THE EXPECTED RESULT, STATED BEFORE THE RUN
-------------------------------------------
Within-county correlation between dairy presence and the three indicators is
+0.084 / -0.001 / -0.024. Under A2 the conditioning has almost nothing to
absorb, so beta should BARELY MOVE. That null is the expected result and is the
point: it is evidence the within estimate is not picking up "any CAFO".
Under A1 (between-county, levels correlation +0.277 / +0.117 / +0.176) beta
SHOULD move. Recording the prediction here so the run cannot be read post hoc.

CAVEAT
------
Other-animal presence is not clearly pre-determined. If dairy expansion
displaces or attracts other livestock these are post-treatment, and
conditioning on them is a bad control. A3 probes WHAT THE DAIRY COEFFICIENT
CONTAINS; it is not better identified than A2.

Output -> Data/output/tables/script4l/
"""
import warnings; warnings.filterwarnings("ignore")
from itertools import combinations
import numpy as np, pandas as pd, pyfixest as pf

from script4_treatment import (
    load_panel, OUTCOMES, CORE_TREATMENTS, CORE_TREATMENT_LABELS,
    SPEC_EXTRA_REGRESSORS, COVARIATE_ORDER, A3_CONDITIONING, A3_EXCLUDED,
    add_a3_conditioning, tables_dir, os, date,
)

OUT   = os.path.join(tables_dir, "script4l"); os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")

FE_SPECS = {"A1 pooled (state+year)": "state_fips + year",
            "A2 within (county+year)": "fips + year"}

# D-16: the pre-committed specification is the eight full-coverage covariates.
BASE_COVARIATES = COVARIATE_ORDER[:8]

COND = list(A3_CONDITIONING)                       # cattle, hogs, chickens
SUBSETS = [c for k in range(len(COND)+1) for c in combinations(COND, k)]   # 8

def cl(n):
    return (str(n).replace("%","pct").replace("-","_").replace("/","_").replace(" ","_")
                  .replace("(","").replace(")","").replace("+","p").replace(",","_"))

def vif_block(frame, cols, within):
    """VIF over the full RHS. Within-transformed for county-FE specs, since that
    is the variation the estimator uses; raw levels for pooled."""
    W = frame[cols].astype(float).copy()
    if within:
        for c in cols:
            W[c] = (W[c] - frame.groupby("fips")[c].transform("mean")
                        - frame.groupby("year")[c].transform("mean") + frame[c].mean())
    keep = [c for c in cols if W[c].std() > 1e-10]
    if len(keep) < 2:
        return np.nan, np.nan, len(cols)-len(keep)
    try:
        R = np.corrcoef(W[keep].values, rowvar=False)
        v = pd.Series(np.diag(np.linalg.pinv(R)), index=keep)
    except Exception:
        return np.nan, np.nan, len(cols)-len(keep)
    return float(v.max()), float(v.get(cols[0], np.nan)), len(cols)-len(keep)

def switchers(sub, col):
    return int((sub.groupby("fips")[col].nunique() > 1).sum())

# --------------------------------------------------------------------- data
df = load_panel(); df.attrs = {}
df = add_a3_conditioning(df)

print("="*78)
print("script4l -- MODEL A3: dairy conditional on other large CAFO presence")
print("="*78)
print(f"panel {len(df):,} rows")
print(f"conditioning pool : {', '.join(COND)}")
for k, v in A3_EXCLUDED.items():
    print(f"  excluded {k}: {v.splitlines()[0]}")
print(f"covariates held at : step-8 pre-committed set ({len(BASE_COVARIATES)} vars, D-16)")
print(f"subsets            : {len(SUBSETS)}  (2^{len(COND)}, empty set = plain A1/A2)")
print(f"total regressions  : {len(CORE_TREATMENTS)*len(OUTCOMES)*len(FE_SPECS)*len(SUBSETS):,}\n")

rows = []
for tid, tcol in CORE_TREATMENTS.items():
    extra = SPEC_EXTRA_REGRESSORS.get(tid, [])
    for okey, ocol in OUTCOMES.items():
        for fe_label, fe in FE_SPECS.items():
            within = fe.startswith("fips")
            ref_beta = None
            for subset in SUBSETS:
                rhs  = [tcol] + list(extra) + BASE_COVARIATES + list(subset)
                need = ["fips","year","state_fips",ocol] + rhs
                sub  = df[list(dict.fromkeys(need))].dropna()
                if len(sub) < 300:
                    continue
                ren = {c: cl(c) for c in sub.columns}
                s2  = sub.rename(columns=ren)
                vmax, vtreat, nzero = vif_block(sub, rhs, within)
                fml = f"{ren[ocol]} ~ " + " + ".join(ren[c] for c in rhs) + f" | {fe}"
                try:
                    out = {}
                    for key, vc in [("state", {"CRV1":"state_fips"}),
                                    ("county", {"CRV1":"fips"}),
                                    ("hetero", "hetero")]:
                        m = pf.feols(fml, data=s2, vcov=vc)
                        td = m.tidy()
                        if ren[tcol] not in td.index: raise KeyError(tcol)
                        r = td.loc[ren[tcol]]
                        out[f"se_{key}"] = float(r["Std. Error"])
                        out[f"p_{key}"]  = float(r["Pr(>|t|)"])
                        if key == "state":
                            out["beta"]      = float(r["Estimate"])
                            out["N"]         = int(m._N)
                            out["r2_within"] = float(getattr(m, "_r2_within", np.nan))
                    # coefficients ON the conditioning vars -- is the OTHER animal
                    # doing anything? A null on dairy means something different if
                    # the controls are themselves dead.
                    tdc = pf.feols(fml, data=s2, vcov={"CRV1":"fips"}).tidy()
                    for c in COND:
                        out[f"beta_{c}"] = (float(tdc.loc[ren[c],"Estimate"])
                                            if c in subset and ren[c] in tdc.index else np.nan)
                except Exception as e:
                    print(f"    {tid}|{okey}|{fe_label}|{subset} failed: {type(e).__name__}")
                    continue
                if ref_beta is None:
                    ref_beta = out["beta"]          # empty subset = the reference
                dpct = (100*(out["beta"]-ref_beta)/abs(ref_beta)
                        if ref_beta not in (None,0) and np.isfinite(ref_beta) else np.nan)
                rows.append({
                    "treatment_id": tid, "treatment": CORE_TREATMENT_LABELS[tid],
                    "outcome": okey, "fe_spec": fe_label,
                    "subset": "+".join(s.replace("any_lg_","") for s in subset) or "(none = A1/A2)",
                    "n_conditioning": len(subset),
                    **{f"has_{c.replace('any_lg_','')}": (c in subset) for c in COND},
                    **out,
                    "beta_reference":    ref_beta,
                    "d_beta_pct_vs_ref": dpct,
                    "vif_max": vmax, "vif_treatment": vtreat, "n_absorbed": nzero,
                    "n_counties":  sub["fips"].nunique(),
                    "n_switchers": switchers(sub, tcol),
                })
        print(f"  {tid} {okey:26s} done")

R = pd.DataFrame(rows)
p = os.path.join(OUT, f"{TODAY}_A3_horserace.csv")
R.to_csv(p, index=False)
expected = len(CORE_TREATMENTS)*len(OUTCOMES)*len(FE_SPECS)*len(SUBSETS)
print(f"\nSaved: {p}   ({len(R):,} rows)")
print(f"  failures: {expected-len(R):,}")

# ------------------------------------------------------------------ readout
full = R[R.n_conditioning == len(COND)]     # all three conditioned on
print("\nBETA MOVE WHEN ALL THREE ARE CONDITIONED ON (vs the same spec without)")
for fe_label in FE_SPECS:
    g = full[full.fe_spec == fe_label]
    print(f"  {fe_label:24s} median |move| {g.d_beta_pct_vs_ref.abs().median():6.2f}%"
          f"   max {g.d_beta_pct_vs_ref.abs().max():7.2f}%")
print("\n  Prediction was: A2 barely moves, A1 moves. Compare above.")

sign = full[np.sign(full.beta) != np.sign(full.beta_reference)]
print(f"\nsign flips vs reference: {len(sign)}")
print(f"max VIF across all {len(R)} regressions: {R.vif_max.max():.2f}")
