#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4b-covariate-sequence.py  --  A1/A2 with covariates added one at a time.

DESIGN
------
For every combination of
    treatment   T1-T4        (CORE_TREATMENTS: presence, count, per capita, log)
    outcome     7            (3 mental health / mortality + 4 crime)
    FE          2            A1 pooled state+year, A2 within county+year
    step        0..14        covariates added ONE AT A TIME in COVARIATE_ORDER
estimate the regression and record what changed.

  4 x 7 x 2 x 15 = 840 regressions.

Step 0 is the treatment with no covariates (plus log_pop for T2/T4, which is part
of the SPECIFICATION, not a covariate -- see SPEC_EXTRA_REGRESSORS).

WHAT IS RECORDED, AND WHY
-------------------------
  beta, se_state/county/hetero   the estimate under three variance estimators
  d_beta_pct                     % change in beta from the PREVIOUS step. This is
                                 the diagnostic for "does this covariate matter" --
                                 a control earns its place by MOVING the treatment
                                 coefficient, which is what confounding control means.
  vif_max, vif_treatment         collinearity. NOTE: VIF answers "is this redundant
                                 given the others", NOT "is this useful". A variable
                                 can have VIF 1.0 and be useless, or VIF 4 and be
                                 essential. Read it alongside d_beta_pct, not instead.
  partial_r2                     share of residual outcome variance the covariate
                                 block explains
  N, n_counties, n_switchers     what the step costs in sample and in identifying
                                 variation. Under county FE only switchers identify
                                 beta, so a step that halves switchers has halved
                                 the evidence even if N looks fine.

ORDER DEPENDENCE -- read this before interpreting the path
-----------------------------------------------------------
Forward stepwise results depend entirely on the order. The order is FIXED in
script4_treatment.COVARIATE_ORDER and grouped so the eight full-coverage
covariates enter first: through step 8 the sample is intact (N ~ 65,578) and a
change in beta is attributable to the covariate alone. From step 9 the sample
falls -- largest at step 9, where SAHIE's 2008 start truncates the window -- so
later steps confound the covariate with the sample restriction.

THIS IS A DIAGNOSTIC, NOT A SELECTION PROCEDURE. Choosing the final specification
because of what this path shows, then reporting conventional standard errors on
it, is post-selection inference and the p-values would be invalid. The reported
specification stays pre-committed; this run is an appendix exhibit.

Output -> Data/output/tables/script4b/
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyfixest as pf

from script4_treatment import (
    load_panel, OUTCOMES, CORE_TREATMENTS, CORE_TREATMENT_LABELS,
    SPEC_EXTRA_REGRESSORS, COVARIATE_ORDER, COVARIATE_EXCLUDED,
    tables_dir, os, date,
)

OUT = os.path.join(tables_dir, "script4b"); os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
FE_SPECS = {"A1 pooled (state+year)": "state_fips + year",
            "A2 within (county+year)": "fips + year"}

def cl(n):
    return (str(n).replace("%","pct").replace("-","_").replace("/","_").replace(" ","_")
                  .replace("(","").replace(")","").replace("+","p").replace(",","_"))

def vif_block(frame, cols, within):
    """VIF over the full right-hand side. Within-transformed for county-FE specs,
    since that is the variation the estimator uses; raw levels for pooled."""
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

def variation(sub, tcol):
    nun = sub.groupby("fips")[tcol].nunique()
    return int((nun > 1).sum())

df = load_panel(); df.attrs = {}
print("="*78)
print("script4b -- covariates added one at a time")
print("="*78)
print(f"panel {len(df):,} rows | {len(CORE_TREATMENTS)} treatments x {len(OUTCOMES)} outcomes "
      f"x {len(FE_SPECS)} FE x {len(COVARIATE_ORDER)+1} steps "
      f"= {len(CORE_TREATMENTS)*len(OUTCOMES)*len(FE_SPECS)*(len(COVARIATE_ORDER)+1):,} regressions")
print(f"excluded covariates: {', '.join(COVARIATE_EXCLUDED)}\n")

rows = []
for tid, tcol in CORE_TREATMENTS.items():
    extra = SPEC_EXTRA_REGRESSORS.get(tid, [])
    for okey, ocol in OUTCOMES.items():
        for fe_label, fe in FE_SPECS.items():
            within = fe.startswith("fips")
            prev_beta = None
            for step in range(len(COVARIATE_ORDER)+1):
                covs = COVARIATE_ORDER[:step]
                rhs  = [tcol] + list(extra) + covs
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
                        t = m.tidy()
                        if ren[tcol] not in t.index: raise KeyError(tcol)
                        r = t.loc[ren[tcol]]
                        out[f"se_{key}"] = float(r["Std. Error"])
                        out[f"p_{key}"]  = float(r["Pr(>|t|)"])
                        if key == "state":
                            out["beta"] = float(r["Estimate"])
                            out["N"] = int(m._N)
                            out["r2_within"] = float(getattr(m, "_r2_within", np.nan))
                except Exception as e:
                    print(f"    {tid}|{okey}|{fe_label}|step{step} failed: {type(e).__name__}")
                    continue
                dpct = (100*(out["beta"]-prev_beta)/abs(prev_beta)
                        if prev_beta not in (None, 0) and np.isfinite(prev_beta) else np.nan)
                rows.append({
                    "treatment_id": tid, "treatment": CORE_TREATMENT_LABELS[tid],
                    "outcome": okey, "fe_spec": fe_label,
                    "step": step,
                    "covariate_added": covs[-1] if covs else "(none - treatment only)",
                    "n_covariates": len(covs),
                    **out,
                    "d_beta_pct_from_prev": dpct,
                    "vif_max": vmax, "vif_treatment": vtreat, "n_absorbed": nzero,
                    "n_counties": sub["fips"].nunique(),
                    "n_switchers": variation(sub, tcol),
                })
                prev_beta = out["beta"]
        print(f"  {tid} {okey:26s} done")

R = pd.DataFrame(rows)
p = os.path.join(OUT, f"{TODAY}_covariate_sequence.csv")
R.to_csv(p, index=False)
print(f"\nSaved: {p}   ({len(R):,} rows)")
print(f"  failures: {len(CORE_TREATMENTS)*len(OUTCOMES)*len(FE_SPECS)*(len(COVARIATE_ORDER)+1)-len(R):,}")
print(f"  max VIF anywhere: {R.vif_max.max():.2f}   specs with an absorbed regressor: {(R.n_absorbed>0).sum()}")
