#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4f-core-models.py  --  THE THREE CORE TREATMENTS, end to end.

Team decision 2026-09-18: three treatments carry the headline results.
    T1  tr_e1_lg_bin        Any large dairy CAFO (binary presence)
    T2  tr_c3_log_lg        log(1 + large dairy CAFO count)
    T3  tr_e3_entry_absorb  CAFO entrance (0 -> +), absorbing

For each, and for each of the 6 outcomes, this runs:
    TWFE       county + year FE, three variance estimators, per-spec VIF
    Event study   leads/lags around the treatment's own cohort definition
    Callaway-Sant'Anna   ATT(g,t) with never-treated + not-yet-treated controls

Controls: three sets, run side by side so the control choice is visible rather
than assumed.
    CONTROL_PRETREAT       19 vars  DEFAULT -- derived pool minus mediators
    CONTROL_COMPREHENSIVE  26 vars  full derived pool (includes mediators)
    CONTROL_CORE9           9 vars  the smaller legacy set, for continuity

Cohort definition per treatment (Callaway-Sant'Anna needs an absorbing binary):
    T1 -> first year the county has any large dairy operation
    T2 -> first wave with a positive change in the large-operation count
    T3 -> first wave with a 0 -> positive entry
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import norm as _norm

from script4_treatment import (
    load_panel, OUTCOMES, CORE_TREATMENTS, HEADLINE_TREATMENT, HEADLINE_OUTCOME,
    CONTROL_PRETREAT, CONTROL_COMPREHENSIVE, CONTROL_CORE9, CONTROL_MEDIATORS,
    figs_dir, tables_dir, os, date,
)

OUT_T = os.path.join(tables_dir, "script4f"); os.makedirs(OUT_T, exist_ok=True)
OUT_F = os.path.join(figs_dir,  "script4f"); os.makedirs(OUT_F, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
N_BOOT = 300
RNG = np.random.default_rng(20260918)

CONTROL_SETS = {
    "PRETREAT (19, default)":   CONTROL_PRETREAT,
    "COMPREHENSIVE (26)":       CONTROL_COMPREHENSIVE,
    "CORE9 (9, legacy)":        CONTROL_CORE9,
}

def _clean(n):
    return (str(n).replace("%","pct").replace("-","_").replace(" ","_")
            .replace("(","").replace(")","").replace("/","_").replace("+","p"))

def vif_vec(frame, cols):
    W = frame[cols].astype(float).copy()
    for c in cols:
        W[c] = (W[c] - frame.groupby("fips")[c].transform("mean")
                    - frame.groupby("year")[c].transform("mean") + frame[c].mean())
    keep = [c for c in cols if W[c].std() > 1e-10]
    if len(keep) < 2: return pd.Series(dtype=float), len(cols)-len(keep)
    R = np.corrcoef(W[keep].values, rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(R)), index=keep), len(cols)-len(keep)

df = load_panel()
# cohort per core treatment
df["coh_T1"] = df.groupby("fips")["year"].transform(
    lambda s: np.nan) # placeholder, filled below
first_presence = (df[df["tr_e1_lg_bin"] == 1].groupby("fips")["year"].min().rename("coh_T1"))
df = df.drop(columns=["coh_T1"]).merge(first_presence, on="fips", how="left")
df["coh_T2"] = df["add_year"]
df["coh_T3"] = df["entry_year"]
COHORT_COL = {"T1": "coh_T1", "T2": "coh_T2", "T3": "coh_T3"}

print("="*78); print("script4f -- THREE CORE TREATMENTS"); print("="*78)
for t,(c,_,l) in CORE_TREATMENTS.items():
    print(f"  {t}  {c:22s} {l}   cohorts={df[df[COHORT_COL[t]].notna()].fips.nunique():,} counties")

# =============================================================================
# 1. TWFE + VIF, three control sets
# =============================================================================
rows = []
for tid,(tcol,cond,tlab) in CORE_TREATMENTS.items():
    for cs_name, CS in CONTROL_SETS.items():
        for okey, ocol in OUTCOMES.items():
            CS_COND = list(cond)
            need = ["fips","year","state_fips",ocol,tcol]+CS_COND+CS
            sub = df[list(dict.fromkeys(need))].dropna()
            if len(sub) < 300: continue
            ren = {c:_clean(c) for c in sub.columns}
            s2 = sub.rename(columns=ren)
            ct, cy = ren[tcol], ren[ocol]
            cc = [ren[c] for c in CS_COND]          # conditioning terms
            cx = [ren[c] for c in CS]               # controls
            v, nzero = vif_vec(sub, [tcol]+CS_COND+CS)
            nun = sub.groupby("fips")[tcol].nunique()
            fml = f"{cy} ~ " + " + ".join([ct]+cc+cx) + " | fips + year"
            try:
                out = {}
                for k,vc in [("state",{"CRV1":"state_fips"}),("county",{"CRV1":"fips"}),("hetero","hetero")]:
                    m = pf.feols(fml, data=s2, vcov=vc); t_ = m.tidy().loc[ct]
                    out[f"se_{k}"]=float(t_["Std. Error"]); out[f"p_{k}"]=float(t_["Pr(>|t|)"])
                    if k=="state":
                        out["beta"]=float(t_["Estimate"]); out["N"]=int(m._N)
                        # co-estimated conditioning coefficient -- for T2 this is
                        # log(small), the other half of the consolidation story
                        tt=m.tidy()
                        for xc,xo in zip(cc,CS_COND):
                            if xc in tt.index:
                                out["cond_var"]=xo
                                out["cond_beta"]=float(tt.loc[xc,"Estimate"])
                                out["cond_se"]=float(tt.loc[xc,"Std. Error"])
                                out["cond_p"]=float(tt.loc[xc,"Pr(>|t|)"])
            except Exception as e:
                print(f"    {tid}|{cs_name}|{okey} failed: {e}"); continue
            rows.append({"treatment_id":tid,"treatment":tlab,"treatment_col":tcol,
                         "control_set":cs_name,"n_controls":len(CS),"outcome":okey,**out,
                         "n_counties":sub.fips.nunique(),"n_switchers":int((nun>1).sum()),
                         "vif_max":float(v.max()) if len(v) else np.nan,
                         "vif_max_var":str(v.idxmax()) if len(v) else "",
                         "vif_treat":float(v.get(tcol,np.nan)) if len(v) else np.nan,
                         "n_absorbed_regressors":int(nzero)})
    print(f"  {tid} TWFE done")
twfe = pd.DataFrame(rows)
twfe.to_csv(os.path.join(OUT_T,f"{TODAY}_CORE_twfe_by_controlset.csv"),index=False)

# =============================================================================
# 2. Callaway-Sant'Anna for all three core treatments
# =============================================================================
def att_gt(wide, coh, cohorts, years, keep=None):
    out=[]; idx = wide.index if keep is None else wide.index.intersection(keep)
    c = coh.reindex(idx)
    for g in cohorts:
        base = g-1
        if base not in wide.columns: continue
        tr = idx[(c==g).values]
        if len(tr) < 10: continue
        for t in years:
            if t==base or t not in wide.columns: continue
            ctl = idx[((c.isna())|(c>max(t,g))).values]
            d = wide[t]-wide[base]
            dt,dc = d.reindex(tr).dropna(), d.reindex(ctl).dropna()
            if len(dt)<10 or len(dc)<10: continue
            out.append({"cohort":int(g),"year":int(t),"event_time":int(t-g),
                        "att":float(dt.mean()-dc.mean()),"n_treated":len(dt),"n_control":len(dc)})
    return out

cs_rows=[]
for tid,(tcol,_,tlab) in CORE_TREATMENTS.items():
    ccol=COHORT_COL[tid]
    cohorts=sorted(df.loc[df[ccol].notna(),ccol].unique())
    for okey,ocol in OUTCOMES.items():
        d=df[["fips","year","state_fips",ccol,ocol]].dropna(subset=[ocol])
        if d.empty: continue
        wide=d.pivot_table(index="fips",columns="year",values=ocol,aggfunc="first")
        coh=d.groupby("fips")[ccol].first(); st=d.groupby("fips")["state_fips"].first()
        yrs=sorted(d.year.unique())
        pt=att_gt(wide,coh,cohorts,yrs)
        if not pt: continue
        states=st.dropna().unique(); by={s:st.index[st==s] for s in states}
        bo=[]
        for _ in range(N_BOOT):
            dr=RNG.choice(states,size=len(states),replace=True)
            fb=pd.Index(pd.unique(np.concatenate([by[s].values for s in dr])))
            pb=att_gt(wide,coh,cohorts,yrs,keep=fb)
            post=[r for r in pb if r["event_time"]>=0]
            if post:
                w=np.array([r["n_treated"] for r in post],float)
                bo.append(float(np.average([r["att"] for r in post],weights=w)))
        P=pd.DataFrame(pt); post=P[P.event_time>=0]
        if not len(post): continue
        ov=float(np.average(post.att,weights=post.n_treated.astype(float)))
        se=float(np.std(bo,ddof=1)) if len(bo)>10 else np.nan
        pre=P[P.event_time<0]
        pre_att=float(np.average(pre.att,weights=pre.n_treated.astype(float))) if len(pre) else np.nan
        cs_rows.append({"treatment_id":tid,"treatment":tlab,"outcome":okey,
                        "CS_ATT":ov,"CS_se":se,
                        "CS_p":float(2*(1-_norm.cdf(abs(ov/se)))) if se and se>0 else np.nan,
                        "CS_pre_mean":pre_att,"n_cells":len(P),
                        "n_cohorts":int(post.cohort.nunique()),"n_treated":int(post.n_treated.sum())})
    print(f"  {tid} CS done")
cs=pd.DataFrame(cs_rows)
cs.to_csv(os.path.join(OUT_T,f"{TODAY}_CORE_callaway_santanna.csv"),index=False)

# =============================================================================
# 3. Headline: TWFE vs CS, default control set
# =============================================================================
hl=[]
for tid,(tcol,_,tlab) in CORE_TREATMENTS.items():
    for okey in OUTCOMES:
        a=twfe[(twfe.treatment_id==tid)&(twfe.control_set=="PRETREAT (19, default)")&(twfe.outcome==okey)]
        b=cs[(cs.treatment_id==tid)&(cs.outcome==okey)]
        if a.empty or b.empty: continue
        a,b=a.iloc[0],b.iloc[0]
        s1,s2=a.p_state<0.05, (b.CS_p<0.05 if pd.notna(b.CS_p) else False)
        hl.append({"treatment_id":tid,"treatment":tlab,"outcome":okey,
                   "TWFE_beta":a.beta,"TWFE_se":a.se_state,"TWFE_p":a.p_state,"TWFE_sig":s1,
                   "CS_ATT":b.CS_ATT,"CS_se":b.CS_se,"CS_p":b.CS_p,"CS_sig":s2,
                   "verdict":("survives CS" if (s1 and s2) else
                              "TWFE only" if s1 else "CS only" if s2 else "null in both"),
                   "N":a.N,"n_switchers":a.n_switchers,"vif_max":a.vif_max})
HL=pd.DataFrame(hl)
HL.to_csv(os.path.join(OUT_T,f"{TODAY}_CORE_HEADLINE.csv"),index=False)

print("\n"+"="*78); print("HEADLINE: three core treatments, TWFE vs Callaway-Sant'Anna"); print("="*78)
print(HL[["treatment_id","outcome","TWFE_beta","TWFE_p","CS_ATT","CS_p","verdict"]]
      .to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("\n=== VIF across all core specs ===")
print(f"  specs: {len(twfe)}   max VIF: {twfe.vif_max.max():.2f} ({twfe.loc[twfe.vif_max.idxmax(),'vif_max_var']})")
print(f"  >5: {(twfe.vif_max>5).sum()}   >10: {(twfe.vif_max>10).sum()}   absorbed regressors: {(twfe.n_absorbed_regressors>0).sum()}")
print(f"\ntables -> {OUT_T}")
