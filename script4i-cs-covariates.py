#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4i-cs-covariates.py  --  Callaway-Sant'Anna WITH covariates, vs without.

The A18 estimates reported so far are UNCONDITIONAL: ATT(g,t) is a raw difference
of long differences. That assumes parallel trends holds unconditionally, which is
a STRONGER assumption than the TWFE specification makes (TWFE conditions on X).
This script adds the outcome-regression (OR) adjustment so the two are comparable.

ESTIMATOR (outcome regression / "regression-adjusted" CS)
---------------------------------------------------------
For cohort g and period t, with base period g-1:

  1. Form the long difference for every unit:  dY_i = Y_i,t - Y_i,g-1
  2. Using ONLY the clean control group (never-treated + not-yet-treated at t),
     fit          dY_i = X_i' b + e_i
     where X_i are covariates measured at the BASE PERIOD g-1 -- pre-treatment by
     construction, so this cannot condition on a post-treatment variable.
  3. Predict the counterfactual change for each treated unit: dYhat_i = X_i' b
  4. ATT(g,t) = mean(dY_i | treated) - mean(dYhat_i | treated)

Step 2 is fitted on controls only, which is what makes this a valid adjustment
rather than a regression that soaks up the treatment effect.

Inference: 300-rep cluster bootstrap over states, redrawing the whole procedure
(including the control-group regression) inside each replication.

Aggregation: cohort-size-weighted, same as the unconditional version.

Output -> output/tables/script4f/
    *_CS_with_vs_without_covariates.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import norm as _norm
from script4_treatment import (load_panel, OUTCOMES, CORE_TREATMENTS,
                                CONTROL_PRETREAT, tables_dir, os, date)

OUT_T = os.path.join(tables_dir, "script4f"); os.makedirs(OUT_T, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
N_BOOT = 300
RNG = np.random.default_rng(20260918)

df = load_panel()
first_presence = df[df["tr_e1_lg_bin"]==1].groupby("fips")["year"].min().rename("coh_T1")
df = df.merge(first_presence, on="fips", how="left")
df["coh_T2"]=df["add_year"]; df["coh_T3"]=df["entry_year"]
COH = {"T1":"coh_T1","T2":"coh_T2","T3":"coh_T3"}
X_COLS = [c for c in CONTROL_PRETREAT if c in df.columns]
print(f"panel {df.shape[0]:,} rows | covariates: {len(X_COLS)}")


def att_gt(wide, Xbase, coh, cohorts, years, keep=None, use_cov=True, min_n=10):
    """All ATT(g,t). Xbase maps (fips, base_year) -> covariate row."""
    out=[]
    idx = wide.index if keep is None else wide.index.intersection(keep)
    c = coh.reindex(idx)
    for g in cohorts:
        base = g-1
        if base not in wide.columns: continue
        tr = idx[(c==g).values]
        if len(tr) < min_n: continue
        Xb = Xbase.get(base)                       # covariates AT THE BASE PERIOD
        for t in years:
            if t==base or t not in wide.columns: continue
            ctl = idx[((c.isna())|(c>max(t,g))).values]
            d = wide[t]-wide[base]
            dt = d.reindex(tr).dropna(); dc = d.reindex(ctl).dropna()
            if len(dt)<min_n or len(dc)<min_n: continue

            if not use_cov or Xb is None:
                out.append({"cohort":int(g),"year":int(t),"event_time":int(t-g),
                            "att":float(dt.mean()-dc.mean()),
                            "n_treated":len(dt),"n_control":len(dc)}); continue

            # --- outcome regression on the CONTROL group only ------------------
            Xc = Xb.reindex(dc.index).dropna()
            Xt = Xb.reindex(dt.index).dropna()
            if len(Xc) < max(min_n, len(X_COLS)+5) or len(Xt) < min_n:
                # too thin to fit the adjustment -> fall back to unconditional
                out.append({"cohort":int(g),"year":int(t),"event_time":int(t-g),
                            "att":float(dt.mean()-dc.mean()),
                            "n_treated":len(dt),"n_control":len(dc),"fallback":1}); continue
            yc = dc.reindex(Xc.index).values
            A  = np.column_stack([np.ones(len(Xc)), Xc.values])
            try:
                b,*_ = np.linalg.lstsq(A, yc, rcond=None)
            except np.linalg.LinAlgError:
                continue
            pred = np.column_stack([np.ones(len(Xt)), Xt.values]) @ b
            att  = float(dt.reindex(Xt.index).mean() - pred.mean())
            out.append({"cohort":int(g),"year":int(t),"event_time":int(t-g),"att":att,
                        "n_treated":len(Xt),"n_control":len(Xc),"fallback":0})
    return out


rows=[]
for tid,(tcol,_,tlab) in CORE_TREATMENTS.items():
    ccol=COH[tid]; cohorts=sorted(df.loc[df[ccol].notna(),ccol].unique())
    for okey,ocol in OUTCOMES.items():
        d=df[["fips","year","state_fips",ccol,ocol]+X_COLS].dropna(subset=[ocol])
        if d.empty: continue
        wide=d.pivot_table(index="fips",columns="year",values=ocol,aggfunc="first")
        coh=d.groupby("fips")[ccol].first(); st=d.groupby("fips")["state_fips"].first()
        years=sorted(d.year.unique())
        # covariates at each candidate base period, standardised for conditioning
        Xbase={}
        for g in cohorts:
            b=g-1
            sub=d[d.year==b].set_index("fips")[X_COLS]
            if len(sub)>0:
                sd=sub.std().replace(0,np.nan)
                Xbase[b]=((sub-sub.mean())/sd).dropna(axis=1, how="all")
        res={}
        for lab,uc in [("without",False),("with",True)]:
            pt=att_gt(wide,Xbase,coh,cohorts,years,use_cov=uc)
            if not pt: continue
            P=pd.DataFrame(pt); post=P[P.event_time>=0]; pre=P[P.event_time<0]
            if not len(post): continue
            ov=float(np.average(post.att,weights=post.n_treated.astype(float)))
            prem=float(np.average(pre.att,weights=pre.n_treated.astype(float))) if len(pre) else np.nan
            states=st.dropna().unique(); by={s:st.index[st==s] for s in states}
            bo=[]
            for _ in range(N_BOOT):
                dr=RNG.choice(states,size=len(states),replace=True)
                fb=pd.Index(pd.unique(np.concatenate([by[s].values for s in dr])))
                pb=att_gt(wide,Xbase,coh,cohorts,years,keep=fb,use_cov=uc)
                po=[r for r in pb if r["event_time"]>=0]
                if po:
                    w=np.array([r["n_treated"] for r in po],float)
                    bo.append(float(np.average([r["att"] for r in po],weights=w)))
            se=float(np.std(bo,ddof=1)) if len(bo)>10 else np.nan
            res[lab]=dict(att=ov,se=se,pre=prem,cells=len(P),cohorts=int(post.cohort.nunique()),
                          fb=int(P.get("fallback",pd.Series(dtype=float)).sum()) if "fallback" in P else 0)
        if "with" in res and "without" in res:
            a,b_=res["without"],res["with"]
            rows.append({"treatment_id":tid,"outcome":okey,
                "ATT_nocov":a["att"],"se_nocov":a["se"],
                "p_nocov":float(2*(1-_norm.cdf(abs(a["att"]/a["se"])))) if a["se"] else np.nan,
                "pre_nocov":a["pre"],
                "ATT_cov":b_["att"],"se_cov":b_["se"],
                "p_cov":float(2*(1-_norm.cdf(abs(b_["att"]/b_["se"])))) if b_["se"] else np.nan,
                "pre_cov":b_["pre"],"cells":b_["cells"],"cohorts":b_["cohorts"],
                "n_fallback_cells":b_["fb"]})
    print(f"  {tid} done")

R=pd.DataFrame(rows)
R.to_csv(os.path.join(OUT_T,f"{TODAY}_CS_with_vs_without_covariates.csv"),index=False)
print("\n=== Callaway-Sant'Anna: WITHOUT vs WITH covariates ===")
for _,r in R.iterrows():
    s1="*" if (pd.notna(r.p_nocov) and r.p_nocov<.05) else " "
    s2="*" if (pd.notna(r.p_cov) and r.p_cov<.05) else " "
    print(f"  {r.treatment_id} {r.outcome:26s} no-cov={r.ATT_nocov:+9.3f}({r.se_nocov:7.3f}){s1} "
          f"cov={r.ATT_cov:+9.3f}({r.se_cov:7.3f}){s2}  pre_cov={r.pre_cov:+8.3f}  "
          f"cells={int(r.cells)} fallback={int(r.n_fallback_cells)}")
print(f"\nSaved: {os.path.join(OUT_T,f'{TODAY}_CS_with_vs_without_covariates.csv')}")
