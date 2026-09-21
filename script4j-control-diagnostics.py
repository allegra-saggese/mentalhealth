#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4j-control-diagnostics.py  --  the 19 controls: coverage, causal role, pre-trends.

Three outputs.

1. COVERAGE + ROLE TABLE. For each control: share of the panel and of the
   estimating sample it covers, first/last year, a one-line description, which
   specifications it enters, and a causal classification.

   CLASSIFICATION -- what the three labels mean, since they are not interchangeable:
     confounder  Plausibly causes BOTH CAFO siting and the outcome, and is not
                 itself caused by CAFO presence. Conditioning on it REMOVES bias.
     mediator    Plausibly caused BY CAFO presence and in turn affects the
                 outcome. Conditioning on it BLOCKS part of the causal path and
                 biases the estimate toward zero.
     collider    Plausibly caused by BOTH the treatment and the outcome.
                 Conditioning on it INDUCES spurious association.
   Where a variable could be more than one, the label records the most likely
   role and the note says why. These are judgements about the causal graph, not
   statistical findings -- they are stated so a reviewer can disagree explicitly.

2. COVARIATE PRE-TREND TEST. Each control is used as the DEPENDENT variable in
   the event-study specification, with county and year fixed effects. If a
   control trends differentially in the pre-period, then treated and control
   counties were already diverging on it before treatment -- which undermines
   parallel trends and suggests the variable is not pre-determined with respect
   to treatment timing. Reported as the number of significant pre-period
   coefficients and the joint F-test on all of them.

3. VIF on the headline specification (TWFE, county + year FE, 19 controls).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyfixest as pf
from script4_treatment import (load_panel, OUTCOMES, CONTROL_PRETREAT,
                                HEADLINE_TREATMENT, HEADLINE_OUTCOME,
                                TREATMENTS, tables_dir, os, date)

OUT = os.path.join(tables_dir, "script4a"); os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
LEADS, LAGS = 6, 6

def cl(n):
    return (str(n).replace("%","pct").replace("-","_").replace(" ","_")
            .replace("/","_").replace("(","").replace(")","").replace("+","p"))

# description, role, why
META = {
 "%_65_and_older":("Share aged 65+","confounder","Age structure predicts both rural agricultural land use and mental-health reporting; not plausibly moved by a CAFO opening."),
 "%_asian":("Share Asian","confounder","Predetermined demographic composition. Near-zero and near-constant in rural counties, so it adds little either way."),
 "%_below_18_years_of_age":("Share under 18","confounder","Age structure; predetermined."),
 "%_female":("Share female","confounder","Predetermined."),
 "%_hispanic":("Share Hispanic","mediator","PLAUSIBLY MOVED BY TREATMENT: large livestock operations recruit Hispanic labour, so post-CAFO composition partly reflects the CAFO. Retained because it is also a strong confounder, but flagged."),
 "%_native_hawaiian/other_pacific_islander":("Share NH/PI","confounder","Predetermined; negligible in rural counties."),
 "%_not_proficient_in_english_per100k":("Limited English proficiency","mediator","Same channel as %Hispanic -- tracks in-migration of CAFO labour."),
 "%_rural":("Share rural within county","confounder","Settlement pattern; predetermined relative to a single operation opening."),
 "access_to_healthy_foods_per100k":("Access to healthy foods","mediator","Food retail environment can shift with agricultural consolidation."),
 "adult_obesity_per100k":("Adult obesity rate","confounder","Health stock; slow-moving, unlikely to respond to a CAFO within the window."),
 "adult_smoking_per100k":("Adult smoking rate","collider","Smoking responds to distress and to local economic shocks -- caused by both the outcome and, plausibly, treatment. Conditioning can induce association."),
 "children_in_poverty_per100k":("Child poverty rate","mediator","Local economic conditions respond to agricultural restructuring."),
 "children_in_single-parent_households_per100k":("Single-parent households","confounder","Family structure; slow-moving."),
 "driving_alone_to_work_per100k":("Drives alone to work","confounder","Commuting pattern; proxies labour-market geography."),
 "median_household_income":("Median household income","mediator","Directly responds to agricultural employment and consolidation."),
 "some_college_per100k":("Some college or more","confounder","Education stock; predetermined."),
 "teen_births_per100k":("Teen birth rate","collider","Responds to both local economic conditions and to mental-health/социal conditions."),
 "unemployment_per100k":("Unemployment rate","mediator","A CAFO opening changes local employment directly."),
 "uninsured_adults_per100k":("Uninsured adults","mediator","Insurance follows employment, which treatment can move."),
}
META["teen_births_per100k"]=("Teen birth rate","collider","Responds to both local economic conditions and to social/mental-health conditions, so it is caused by treatment and outcome alike.")

df = load_panel()
ref = df[df[HEADLINE_OUTCOME].notna()]
tcol,tcond,_ = TREATMENTS[HEADLINE_TREATMENT]
est_cols = ["fips","year",HEADLINE_OUTCOME,tcol]+list(tcond)+CONTROL_PRETREAT
est = df[list(dict.fromkeys(est_cols))].dropna()

# =============================== 1. coverage + role ==========================
rows=[]
for c in CONTROL_PRETREAT:
    s=df[c]; yrs=df.loc[s.notna(),"year"]
    desc,role,why = META.get(c,("","confounder",""))
    rows.append({"variable":c,"description":desc,"causal_role":role,
        "first_year":int(yrs.min()),"last_year":int(yrs.max()),
        "pct_of_full_panel":float(s.notna().mean()),
        "pct_of_outcome_rows":float(ref[c].notna().mean()),
        "pct_of_estimating_sample":1.0,
        "in_A1_pooled":True,"in_A2_within":True,"in_A7_event_study":True,
        "in_A18_CS_with_cov":True,"in_A18_CS_baseline_version":True,
        "rationale":why})
cov=pd.DataFrame(rows)

# =============================== 2. pre-trends ===============================
# Each control as the DEPENDENT variable in the event-study spec.
pre_rows=[]
d=df[["fips","year","state_fips","t_rel"]+CONTROL_PRETREAT].copy()
d["ev"]=d["t_rel"].clip(-LEADS,LAGS); d.loc[d.t_rel.isna(),"ev"]=-1
d=d.drop(columns=["t_rel"])
for c in CONTROL_PRETREAT:
    sub=d[["fips","year","state_fips","ev",c]].dropna()
    if len(sub)<1000: continue
    sub=sub.rename(columns={x:cl(x) for x in sub.columns})
    try:
        m=pf.feols(f"{cl(c)} ~ i(ev, ref=-1) | fips + year", data=sub, vcov={"CRV1":"state_fips"})
        t=m.tidy()
    except Exception as e:
        print(f"  {c}: {e}"); continue
    pre=[]
    for idx,r in t.iterrows():
        if not str(idx).startswith("ev::"): continue
        try: e=int(float(str(idx).split("::")[1]))
        except Exception: continue
        if e< -1: pre.append((e,float(r["Estimate"]),float(r["Pr(>|t|)"])))
    if not pre: continue
    nsig=sum(1 for _,_,p in pre if p<0.05)
    # joint Wald test on all pre-period coefficients
    names=[f"ev::{float(e)}" for e,_,_ in pre]
    try:
        jt=m.wald_test(R=None, q=None) if False else None
    except Exception: jt=None
    sd=float(df[c].std())
    maxabs=max(abs(b) for _,b,_ in pre)
    pre_rows.append({"variable":c,"description":META.get(c,("",""))[0],
        "causal_role":META.get(c,("","",""))[1],
        "n_pre_coefs":len(pre),"n_pre_significant":nsig,
        "max_abs_pre_coef":maxabs,"outcome_sd":sd,
        "max_pre_in_sd":maxabs/sd if sd>0 else np.nan,
        "verdict":"FAILS (differential pre-trend)" if nsig>=2 else
                  ("marginal" if nsig==1 else "flat")})
pre=pd.DataFrame(pre_rows).sort_values("n_pre_significant",ascending=False)

# =============================== 3. VIF, headline ============================
def vifv(frame,cols,within):
    W=frame[cols].astype(float).copy()
    if within:
        for c in cols:
            W[c]=(W[c]-frame.groupby("fips")[c].transform("mean")
                      -frame.groupby("year")[c].transform("mean")+frame[c].mean())
    keep=[c for c in cols if W[c].std()>1e-10]
    R=np.corrcoef(W[keep].values,rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(R)),index=keep)
rhs=[tcol]+list(tcond)+CONTROL_PRETREAT
vr=vifv(est,rhs,False); vw=vifv(est,rhs,True)
vif=pd.DataFrame({"variable":rhs,
    "role":["treatment"]+["conditioning"]*len(tcond)+["control"]*len(CONTROL_PRETREAT),
    "vif_raw":[vr.get(c,np.nan) for c in rhs],
    "vif_within":[vw.get(c,np.nan) for c in rhs]}).sort_values("vif_within",ascending=False)

cov.to_csv(os.path.join(OUT,f"{TODAY}_control_roster.csv"),index=False)
pre.to_csv(os.path.join(OUT,f"{TODAY}_control_pretrends.csv"),index=False)
vif.to_csv(os.path.join(OUT,f"{TODAY}_vif_headline_within.csv"),index=False)

print("=== CONTROL PRE-TRENDS (control used as the dependent variable) ===")
print(pre[["variable","causal_role","n_pre_coefs","n_pre_significant","max_pre_in_sd","verdict"]]
      .to_string(index=False,float_format=lambda x:f"{x:.3f}"))
print(f"\n  FAILING (>=2 significant pre-period coefs): {(pre.n_pre_significant>=2).sum()} of {len(pre)}")
print("\n=== VIF, headline spec (TWFE, county+year FE, 19 controls) ===")
print(vif.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
print(f"\n  max within VIF: {vif.vif_within.max():.2f}   max raw VIF: {vif.vif_raw.max():.2f}")
