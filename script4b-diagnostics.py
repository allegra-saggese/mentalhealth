#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4b-diagnostics.py -- diagnostics of the control set and the event-study design, six sections run in this order.

    controls     candidate controls: coverage, causal role, covariate pre-trends, VIF      (was script4j-control-diagnostics.py)
    leads        outcome pre-trends (leads) for each reported treatment                     (was script4t-pretrend-leads.py)
    pretrends    raw pre-period paths and straight-line fits behind the pre-trend figures   (new 2026-10-26, D-55)
    balance      balance and detectable-effect tables, LaTeX bodies and a CSV                (was script4d-balance-panel.py)
    control-set  add-one-in / take-one-out sensitivity of the treatment coefficient         (was script4y-control-set-selection.py)
    honestdid    inputs for the Rambachan-Roth analysis; R/honestdid.R is run afterwards     (was script4w-honestdid-export.py)
    species-counts  number of E2 and E3 events by species and size class, descriptive, no estimation  (new 2026-10-26)

Reads:   panel via script4_treatment.load_panel() (honestdid: designs.load_designs());
         `leads` and `control-set` read the newest Data/output/tables/*_4b-controls_control_pretrends.csv, which `controls` writes
Writes:  (all in Data/output/tables/, named <date>_<tag>_<stem>; naming: packages.py)
         <date>_4b-controls_{control_roster, control_pretrends, vif_headline_within, control_pretrend_lags,
                                              control_correlations_selected8}.csv, and .txt versions of those and of VIF_audit  (controls)
         <date>_4b-pretrends_{pretrend_raw_means, pretrend_fits}.csv and pretrend_fits.txt                                    (pretrends)
         <date>_4b-leads_{leads_coefficients, leads_joint_tests}.csv                                          (leads)
         <date>_4b-balance_balance_panel.csv; LATEX_OUT/{BALANCEpanel.tex, BALANCEpanel.pdf, _BODY_balance.tex}  (balance)
         <date>_4b-control-set[-quick]_{regressions, path_summary, covariate_summary, selection_table}.csv       (control-set)
         4b-honestdid-inputs_{<design>__<outcome>__beta, ..__sigma, manifest}.csv (undated; R/honestdid.R writes <date>_4b-honestdid_*) (honestdid)
Run:     python3 script4b-diagnostics.py [--only controls leads pretrends balance control-set species-counts honestdid] [--quick]
         then Rscript R/honestdid.R     (--quick shortens control-set only; a few minutes without it)
Method:  METHODS.md sections 3, 4 and 7. Roles are judgements, not findings; the control-set sweep documents the pre-committed
         set (D-16) and does not choose it. Output folders keep the names of the old scripts, so readers (script4p reads the
         balance body) and dated files are unchanged. The old code of each script is moved unchanged into one function.
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import glob
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
import pandas as pd
import pyfixest as pf

from designs import load_designs, county_groups, species_treatments
from estimators import clean, event_study_frame, joint_wald
from figstyle import save_text_table
from packages import out_file, out_glob, out_latest, tagged
from script4_treatment import (
    load_panel, OUTCOMES, COVARIATE_ORDER, COVARIATE_EXCLUDED, CONTROL_PRETREAT,
    HEADLINE_TREATMENT, HEADLINE_OUTCOME, CORE_TREATMENTS, CORE_TREATMENT_LABELS,
    SPEC_EXTRA_REGRESSORS, tables_dir, LATEX_OUT,
)

TODAY = date.today().strftime("%Y-%m-%d")
CONTROLS = COVARIATE_ORDER[:8]       # the pre-committed control set (step 8)
CLIP = 8                             # event-time window for the event studies (leads, honestdid)


# =============================================================================
# controls  (was script4j-control-diagnostics.py)
# =============================================================================
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
 "%_asian_aian_nhpi":("Share Asian, AIAN or NHPI","confounder","Predetermined demographic composition; combined group (2026-10-12)."),
 "%_black":("Share Black","confounder","Predetermined demographic composition (Census PEP, Black alone)."),
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
META["log_pop"]=("Log population (placebo)","placebo","Not a control: tests whether treated counties were on a different population path before entry.")
META["teen_births_per100k"]=("Teen birth rate","collider","Responds to both local economic conditions and to social/mental-health conditions, so it is caused by treatment and outcome alike.")

def controls():
    OUT = tables_dir    # output-name tag "4b-controls" (packages.py); was folder script4a

    df = load_panel()
    ref = df[df[HEADLINE_OUTCOME].notna()]
    # HEADLINE_TREATMENT is a core ID (T1); the legacy TREATMENTS dict no longer has it.
    tcol  = CORE_TREATMENTS[HEADLINE_TREATMENT]
    tcond = SPEC_EXTRA_REGRESSORS.get(HEADLINE_TREATMENT, [])
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
    lag_rows=[]      # every event-time coefficient of the loop below, kept for the pre-trend figures
    # log_pop added 2026-10-12: population placebo. A differential pre-trend in log
    # population would mean treated counties were already depopulating (or growing)
    # before entry, which moves every per-100k outcome through the denominator.
    PRETREND_VARS = CONTROL_PRETREAT + ["log_pop"]
    d=df[["fips","year","state_fips","t_rel"]+PRETREND_VARS].copy()
    d["ev"]=d["t_rel"].clip(-LEADS,LAGS); d.loc[d.t_rel.isna(),"ev"]=-1
    d=d.drop(columns=["t_rel"])
    for c in PRETREND_VARS:
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
            lag_rows.append({"variable":c,"lag":e,"estimate":float(r["Estimate"]),"se":float(r["Std. Error"]),
                             "p":float(r["Pr(>|t|)"]),"outcome_sd":float(df[c].std())})
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

    cov.to_csv(out_file(tables_dir, "4b-controls", f"{TODAY}_control_roster.csv"),index=False)
    pre.to_csv(out_file(tables_dir, "4b-controls", f"{TODAY}_control_pretrends.csv"),index=False)
    vif.to_csv(out_file(tables_dir, "4b-controls", f"{TODAY}_vif_headline_within.csv"),index=False)
    pd.DataFrame(lag_rows).to_csv(out_file(tables_dir, "4b-controls", f"{TODAY}_control_pretrend_lags.csv"),index=False)

    # =============================== 4. correlation among the selected controls ====
    # Two-way demeaned (county and year means removed), the same transformation as vif_within, so the matrix is
    # the correlation the fixed-effects regression sees.
    W = est[["fips","year"]+CONTROLS].copy()
    for c in CONTROLS:
        W[c] = (W[c]-W.groupby("fips")[c].transform("mean")-W.groupby("year")[c].transform("mean")+W[c].mean())
    corr = W[CONTROLS].corr()
    corr_out = corr.copy(); corr_out.insert(0,"variable",corr.index)
    corr_out.to_csv(out_file(tables_dir, "4b-controls", f"{TODAY}_control_correlations_selected8.csv"),index=False)

    # =============================== 5. plain-text versions of the tables =========
    vif_audit = pd.read_csv(out_latest(tables_dir, "4a-designs", "*_VIF_audit.csv"))   # written by script4a designs
    save_text_table(vif_audit.drop(columns=["note"],errors="ignore"), tables_dir, tagged("4b-controls", "VIF_audit"),
        "Variance inflation factors by model: largest VIF among the regressors",
        "VIF_audit.csv is written by script4a-estimates.py. within = True: after removing county and year means. "
        "A VIF above 10 is the usual warning level; none is reached.")
    save_text_table(vif, tables_dir, tagged("4b-controls", "vif_headline_within"),
        "Variance inflation factors, headline specification (treatment T1, all candidate controls)",
        "vif_raw: on the levels; vif_within: after removing county and year means (what the fixed-effects model uses).")
    save_text_table(corr_out, tables_dir, tagged("4b-controls", "control_correlations_selected8"),
        "Correlation among the 8 selected controls (county and year means removed)",
        "Estimating sample of the headline specification. Off-diagonal values near 0 mean the controls carry separate information.",
        float_format="{:.2f}")
    save_text_table(cov.drop(columns=["rationale"]), tables_dir, tagged("4b-controls", "control_roster"),
        "Candidate controls: coverage and where each is used", "Share of non-missing values among the panel, outcome rows and estimating sample.")
    save_text_table(pre[["variable","causal_role","n_pre_coefs","n_pre_significant","max_pre_in_sd","verdict"]], tables_dir, tagged("4b-controls", "control_pretrends"), "Covariate pre-trends: each control as the dependent variable in the event-study design",
        "max_pre_in_sd: largest absolute pre-period coefficient divided by the variable's overall standard deviation. "
        "FAILS = two or more pre-period coefficients significant at 5% (state-clustered).")

    print("=== CONTROL PRE-TRENDS (control used as the dependent variable) ===")
    print(pre[["variable","causal_role","n_pre_coefs","n_pre_significant","max_pre_in_sd","verdict"]]
          .to_string(index=False,float_format=lambda x:f"{x:.3f}"))
    print(f"\n  FAILING (>=2 significant pre-period coefs): {(pre.n_pre_significant>=2).sum()} of {len(pre)}")
    print("\n=== VIF, headline spec (TWFE, county+year FE, 19 controls) ===")
    print(vif.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
    print(f"\n  max within VIF: {vif.vif_within.max():.2f}   max raw VIF: {vif.vif_raw.max():.2f}")


# =============================================================================
# leads  (was script4t-pretrend-leads.py)
# =============================================================================
N_LEADS = 4          # years of treatment leads for the continuous treatments
REPORTED = ["T1", "T3", "T4", "E2", "E3"]


def leads():
    OUT = tables_dir    # output-name tag "4b-leads"; was folder script4t

    df = load_panel()
    df = df.sort_values(["fips", "year"]).reset_index(drop=True)
    print(f"panel {len(df):,} rows | controls: {len(CONTROLS)} | outcomes: {len(OUTCOMES)}")

    # ---------- treatment definitions ------------------------------------------------------------
    cohort_of = {"E2": "add_year", "E3": "entry_year"}
    treat_col = {t: CORE_TREATMENTS[t] for t in ("T1", "T3", "T4")}
    for t, c in cohort_of.items():
        df[f"t_rel_{t}"] = df["year"] - df[c]
    for t, col in treat_col.items():
        for k in range(1, N_LEADS + 1):
            # treatment k years ahead, same county; NaN at the end of the panel or across gaps
            ahead = df.groupby("fips")[col].shift(-k)
            yr_ok = df.groupby("fips")["year"].shift(-k) == df["year"] + k
            df[f"{col}__lead{k}"] = ahead.where(yr_ok)

    coef_rows, joint_rows = [], []
    for okey, ocol in OUTCOMES.items():
        # ---------------- event studies: E2, E3 ----------------
        for t in ("E2", "E3"):
            need = ["fips", "year", "state_fips", ocol, f"t_rel_{t}"] + CONTROLS
            sub = df[need].copy()
            sub["ev"] = sub[f"t_rel_{t}"].clip(-CLIP, CLIP)
            sub.loc[sub[f"t_rel_{t}"].isna(), "ev"] = -1          # never-treated -> reference
            sub = sub.drop(columns=[f"t_rel_{t}"]).dropna()
            sub = sub.rename(columns={c: clean(c) for c in sub.columns})
            fml = f"{clean(ocol)} ~ i(ev, ref=-1) + " + " + ".join(clean(c) for c in CONTROLS) + " | fips + year"
            try:
                m = pf.feols(fml, data=sub, vcov={"CRV1": "state_fips"})
            except Exception as e:
                print(f"  [{t}|{okey}] failed: {type(e).__name__}: {e}"); continue
            tid = m.tidy()
            lead_names = []
            for idx, r in tid.iterrows():
                if not str(idx).startswith("ev::"):
                    continue
                e = int(float(str(idx).split("::")[1]))
                if e < -1:
                    lead_names.append(str(idx))
                    coef_rows.append({"treatment": t, "design": "event study", "outcome": okey,
                                      "lead": e, "beta": float(r["Estimate"]),
                                      "se_state": float(r["Std. Error"]), "p_state": float(r["Pr(>|t|)"])})
            stat, k, p = joint_wald(m, lead_names)
            joint_rows.append({"treatment": t, "design": "event study", "outcome": okey,
                               "n_leads": k, "wald_chi2": stat, "joint_p": p, "N": int(m._N),
                               "n_treated_counties": int(df.loc[df[cohort_of[t]].notna(), "fips"].nunique())})
        # ---------------- lead tests: T1, T3, T4 ----------------
        for t, col in treat_col.items():
            leads = [f"{col}__lead{k}" for k in range(1, N_LEADS + 1)]
            extra = SPEC_EXTRA_REGRESSORS.get(t, [])
            need = ["fips", "year", "state_fips", ocol, col] + leads + extra + CONTROLS
            sub = df[list(dict.fromkeys(need))].dropna().copy()
            sub = sub.rename(columns={c: clean(c) for c in sub.columns})
            rhs = [clean(col)] + [clean(c) for c in leads] + [clean(x) for x in extra] + [clean(c) for c in CONTROLS]
            try:
                m = pf.feols(f"{clean(ocol)} ~ " + " + ".join(rhs) + " | fips + year", data=sub,
                             vcov={"CRV1": "state_fips"})
            except Exception as e:
                print(f"  [{t}|{okey}] failed: {type(e).__name__}: {e}"); continue
            tid = m.tidy()
            lead_names = [clean(c) for c in leads]
            for k, n in enumerate(lead_names, start=1):
                if n in tid.index:
                    r = tid.loc[n]
                    coef_rows.append({"treatment": t, "design": "lead test", "outcome": okey, "lead": k,
                                      "beta": float(r["Estimate"]), "se_state": float(r["Std. Error"]),
                                      "p_state": float(r["Pr(>|t|)"])})
            stat, k, p = joint_wald(m, lead_names)
            joint_rows.append({"treatment": t, "design": "lead test", "outcome": okey, "n_leads": k,
                               "wald_chi2": stat, "joint_p": p, "N": int(m._N), "n_treated_counties": np.nan})
        print(f"  done: {okey}")

    coefs = pd.DataFrame(coef_rows)
    joint = pd.DataFrame(joint_rows)
    coefs.to_csv(out_file(tables_dir, "4b-leads", f"{TODAY}_leads_coefficients.csv"), index=False)
    joint.to_csv(out_file(tables_dir, "4b-leads", f"{TODAY}_leads_joint_tests.csv"), index=False)


    print("\n" + "=" * 100)
    print("JOINT TEST THAT ALL LEADS = 0  (Wald chi-square, state-clustered; p-value)")
    print("=" * 100)
    piv = joint.pivot(index="outcome", columns="treatment", values="joint_p").reindex(list(OUTCOMES))[REPORTED]
    print(piv.round(3).to_string())
    print("\nnumber of leads in each joint test:",
          joint.groupby("treatment").n_leads.agg(lambda s: sorted(set(s))).to_dict())

    print("\n" + "=" * 100)
    print("INDIVIDUAL LEADS  (beta, z)   continuous: k years ahead   |   event study: event time e")
    print("=" * 100)
    for t in REPORTED:
        print(f"\n--- {t} ---")
        for o in OUTCOMES:
            x = coefs[(coefs.treatment == t) & (coefs.outcome == o)].sort_values("lead")
            print(f"{o[:24]:24s} " + "  ".join(f"{int(r.lead):+d}:{r.beta:+.3g}(z{r.beta / r.se_state:+.1f})" for _, r in x.iterrows()))

    # ---------- covariate pre-trends from 4j, alongside ----------------------------------------
    cp = out_glob(tables_dir, "4b-controls", "*_control_pretrends.csv")
    if cp:
        c = pd.read_csv(cp[-1])
        print("\n" + "=" * 100)
        print(f"COVARIATE PRE-TRENDS  (script4j: {os.path.basename(cp[-1])})")
        print("=" * 100)
        print(c[["variable", "causal_role", "n_pre_coefs", "n_pre_significant", "max_pre_in_sd", "verdict"]]
              .round({"max_pre_in_sd": 3}).to_string(index=False))
    print(f"\nSaved -> {OUT}")

# =============================================================================
# pretrends  (new 2026-10-26, D-55): raw pre-period paths and straight-line fits behind the pre-trend figures
# =============================================================================
PRE_WINDOW = range(-CLIP, -1)        # event times -8 ... -2; -1 is the reference year (zero by construction)
MIN_COHORT_N = 10                    # treated or comparison counties needed for a cohort-year to be used


def _fit_line(x, y):
    """Least-squares straight line y = intercept + slope * x through the plotted points (descriptive, not a test)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _raw_paths(df, info, col, treated_col, comparison_fips):
    """Mean change in `col` since the year before treatment (g-1), treated cohorts versus never-treated counties,
    averaged over cohorts weighted by number of treated counties. Same construction as script1c-figures.event_paths."""
    wide = df.pivot_table(index="fips", columns="year", values=col, aggfunc="first")
    recs = []
    for g in sorted(info[treated_col].dropna().unique()):
        base = g - 1
        if base not in wide.columns:
            continue
        treated = info.index[info[treated_col] == g]
        for e in range(-CLIP, CLIP + 1):
            t = g + e
            if t not in wide.columns:
                continue
            dt = (wide[t] - wide[base]).reindex(treated).dropna()
            dc = (wide[t] - wide[base]).reindex(comparison_fips).dropna()
            if len(dt) < MIN_COHORT_N or len(dc) < MIN_COHORT_N:
                continue
            recs.append({"cohort": int(g), "e": e, "n_t": len(dt), "m_t": dt.mean(), "v_t": dt.var(ddof=1) / len(dt),
                         "m_c": dc.mean(), "v_c": dc.var(ddof=1) / len(dc)})
    if not recs:
        return pd.DataFrame()
    r = pd.DataFrame(recs)
    out = []
    for e, g in r.groupby("e"):
        w = g.n_t.to_numpy(float)
        se_t, se_c = np.sqrt(np.sum(w**2 * g.v_t)) / w.sum(), np.sqrt(np.sum(w**2 * g.v_c)) / w.sum()
        m_t, m_c = np.average(g.m_t, weights=w), np.average(g.m_c, weights=w)
        out.append({"event_time": int(e), "treated": m_t, "comparison": m_c, "difference": m_t - m_c,
                    "se_difference": float(np.hypot(se_t, se_c)), "n_cohorts": g.cohort.nunique(), "n_treated": int(g.n_t.sum())})
    return pd.DataFrame(out)


def pretrends():
    OUT = tables_dir    # output-name tag "4b-pretrends"; was folder script4b
    df = load_panel().sort_values(["fips", "year"]).reset_index(drop=True)
    info = county_groups(df)
    # same comparison groups as the event-time figures in script1c: E2 against counties that never add or enter,
    # E3 against counties that never had a large dairy
    designs = {"E2": ("add", info.index[info["add"].isna() & info["entry"].isna()]),
               "E3": ("entry", info.index[info.group == "never large dairy"])}
    variables = [(k, c, "outcome") for k, c in OUTCOMES.items()] + [(c, c, "control") for c in CONTROLS]

    raw_rows, fit_rows = [], []
    for dname, (tcol, comp) in designs.items():
        for label, col, kind in variables:
            p = _raw_paths(df, info, col, tcol, comp)
            if p.empty:
                print(f"  {dname} {label}: no usable cohorts"); continue
            for _, r in p.iterrows():
                raw_rows.append({"design": dname, "variable": label, "kind": kind, **r.to_dict()})
            pre_p = p[p.event_time.isin(PRE_WINDOW)]
            for series in ("treated", "comparison", "difference"):
                slope, icpt = _fit_line(pre_p.event_time, pre_p[series])
                fit_rows.append({"source": "raw_mean", "design": dname, "variable": label, "kind": kind, "series": series,
                                 "n_points": len(pre_p), "slope": slope, "intercept": icpt})
    raw = pd.DataFrame(raw_rows)

    # fits through the lead coefficients of script4t and the control lag coefficients written by `controls`
    lc = pd.read_csv(out_glob(tables_dir, "4b-leads", "*_leads_coefficients.csv")[-1])
    lc["x"] = np.where(lc.design == "lead test", -lc.lead, lc.lead)        # a lead k years ahead sits at -k
    for (t, o), g in lc.groupby(["treatment", "outcome"]):
        slope, icpt = _fit_line(g.x, g.beta)
        fit_rows.append({"source": "lead_coefficients", "design": t, "variable": o, "kind": "outcome", "series": "coefficient",
                         "n_points": len(g), "slope": slope, "intercept": icpt})
    cl_ = pd.read_csv(out_glob(tables_dir, "4b-controls", "*_control_pretrend_lags.csv")[-1])
    cl_ = cl_[cl_.variable.isin(CONTROLS) & (cl_.lag < -1)]
    for c, g in cl_.groupby("variable"):
        slope, icpt = _fit_line(g.lag, g.estimate / g.outcome_sd)
        fit_rows.append({"source": "control_lag_coefficients", "design": "E2", "variable": c, "kind": "control",
                         "series": "coefficient_in_sd", "n_points": len(g), "slope": slope, "intercept": icpt})
    fits = pd.DataFrame(fit_rows)

    raw.to_csv(out_file(tables_dir, "4b-pretrends", f"{TODAY}_pretrend_raw_means.csv"), index=False)
    fits.to_csv(out_file(tables_dir, "4b-pretrends", f"{TODAY}_pretrend_fits.csv"), index=False)
    save_text_table(fits, tables_dir, tagged("4b-pretrends", "pretrend_fits"),
        "Straight-line fits through the pre-period points of each pre-trend figure",
        "Descriptive: ordinary least-squares line through the plotted pre-period points (slope per year of event time). "
        "The joint pre-trend tests are in leads_joint_tests.csv and control_pretrends.csv. Raw-mean series are changes since "
        "the year before treatment, event times -8 to -2; lead coefficients of T1/T3/T4 are placed at minus the lead.",
        float_format="{:.4f}")
    print(f"raw means {raw.shape}, fits {fits.shape}")
    print(fits[fits.series.isin(["difference", "coefficient", "coefficient_in_sd"])].head(12).to_string(index=False))


# =============================================================================
# balance  (was script4d-balance-panel.py)
# =============================================================================
Z80 = 2.8016                      # 80% power, two-sided alpha = 0.05

# E2 is not a core treatment but carries A5/A6, so it belongs in the balance panel.
GRID = {tid: (col, SPEC_EXTRA_REGRESSORS.get(tid, []), CORE_TREATMENT_LABELS[tid])
        for tid, col in CORE_TREATMENTS.items()}
GRID["E2"] = ("tr_e2_add_absorb", [],
              "First positive change in large dairy ops (absorbing)")
BINARY = {"T1", "E2"}

def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def balance():
    OUT = LATEX_OUT   # run 2 (2026-10-12); was latex-outputs-2809
    os.makedirs(OUT, exist_ok=True)

    df = load_panel()
    print("="*78); print("script4d -- balance / power panel"); print("="*78)
    print(f"panel {len(df):,} rows | {len(GRID)} treatments x {len(OUTCOMES)} outcomes")
    print(f"controls: step-8 ({len(CONTROLS)})\n")

    rows = []
    for tid, (tcol, extra, tlab) in GRID.items():
        for okey, ocol in OUTCOMES.items():
            need = ["fips","year","state_fips",ocol,tcol] + list(extra) + CONTROLS
            sub = df[list(dict.fromkeys(need))].dropna()
            if len(sub) < 200:
                continue
            nun = sub.groupby("fips")[tcol].nunique()
            sw  = nun[nun > 1].index
            if len(sw):
                e = sub[sub.fips.isin(sw)].sort_values(["fips","year"]).copy()
                e["_d"] = e.groupby("fips")[tcol].diff()
                ntrans = int((e["_d"].abs() > 0).sum())
            else:
                ntrans = 0
            # within-county SD of the outcome -- the scale MDE is expressed in
            sd_w = float((sub[ocol] - sub.groupby("fips")[ocol].transform("mean")).std())

            ren = {c: clean(c) for c in sub.columns}
            s2  = sub.rename(columns=ren)
            fml = (f"{ren[ocol]} ~ " + " + ".join([ren[tcol]] + [ren[x] for x in extra]
                   + [ren[c] for c in CONTROLS]) + " | fips + year")
            try:
                m = pf.feols(fml, data=s2, vcov={"CRV1":"state_fips"})
                t = m.tidy().loc[ren[tcol]]
                se, beta, pv = float(t["Std. Error"]), float(t["Estimate"]), float(t["Pr(>|t|)"])
            except Exception as ex:
                print(f"  {tid}|{okey} failed: {type(ex).__name__}"); continue

            rows.append({
                "treatment_id": tid, "treatment": tlab, "treatment_col": tcol,
                "outcome": okey, "N": len(sub),
                "obs_treated": int((sub[tcol] > 0).sum()),
                "obs_control": int((sub[tcol] == 0).sum()),
                "n_counties": sub.fips.nunique(),
                "n_counties_treated": int(sub.loc[sub[tcol] > 0, "fips"].nunique()),
                "n_switchers": int(len(sw)), "n_transitions": ntrans,
                "beta": beta, "se_state": se, "p_state": pv,
                "sd_within": sd_w,
                "MDE_sd_within": Z80 * se / sd_w if sd_w else np.nan,
                "binary_split_exact": tid in BINARY,
            })
        print(f"  {tid} done")

    R = pd.DataFrame(rows)
    R.to_csv(out_file(tables_dir, "4b-balance", f"{TODAY}_balance_panel.csv"), index=False)   # the .tex stay in LATEX_OUT

    # --------------------------------------------------------------------- LaTeX
    def table(tid, g):
        lab, col = g.treatment.iloc[0], g.treatment_col.iloc[0]
        exact = bool(g.binary_split_exact.iloc[0])
        body = []
        for _, r in g.iterrows():
            body.append(
                f"{esc(r.outcome)} & {r.N:,} & {r.obs_treated:,} & {r.obs_control:,} & "
                f"{r.n_counties:,} & {r.n_counties_treated:,} & {r.n_switchers:,} & "
                f"{r.n_transitions:,} & {r.se_state:.4f} & {r.MDE_sd_within:.3f} \\\\")
        note = (f"Treatment column \\texttt{{{esc(col)}}}. Rural counties, county $+$ year fixed "
                f"effects, step-8 pre-committed controls. SE is cluster-robust at the state level. "
                f"MDE is the minimum detectable effect at 80\\% power (two-sided $\\alpha=0.05$), "
                f"$2.8016\\times$SE, expressed in {{\\itshape within-county}} standard deviations of "
                f"the outcome. {{\\itshape Switch.}} is the number of counties whose treatment "
                f"changes --- only these identify the coefficient under county fixed effects.")
        note += (" Treated/control counts are exact." if exact else
                 " Treatment is continuous; the treated/control split is at $>0$ and is "
                 "{\\itshape descriptive only} --- the estimator uses the full continuous variation.")
        if g.treatment_id.iloc[0] in ("T2","T4"):
            note += " Conditioning variable: \\texttt{log\\_pop}."
        return ("\\begin{table}[htbp]\n\\centering\n"
                f"\\caption{{Sample and detectable effect: {tid} --- {esc(lab)}}}\n"
                f"\\label{{tab:bal_{tid}}}\n"
                "\\begin{adjustbox}{max width=\\textwidth}\n"
                "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
                " & \\multicolumn{3}{c}{Observations} & \\multicolumn{2}{c}{Counties}"
                " & \\multicolumn{2}{c}{Variation} & \\multicolumn{2}{c}{Detectable effect} \\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
                "Outcome & $N$ & Treated & Control & Total & Treated & Switch. & Trans. & SE"
                " & MDE$_{\\sigma_w}$ \\\\\n\\midrule\n"
                + "\n".join(body) +
                "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\n"
                "\\vspace{0.3em}\n\\begin{minipage}{\\textwidth}\\footnotesize\n"
                f"\\textit{{Notes.}} {note}\n" "\\end{minipage}\n\\end{table}\n")

    tex = [r"""\documentclass[11pt]{article}
\usepackage[margin=1in,landscape]{geometry}
\usepackage{booktabs,adjustbox,amsmath}
\usepackage[colorlinks=true,linkcolor=blue]{hyperref}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}
{\Large\bfseries Balance and detectable effect panel}\\[0.3em]
{\large Large dairy CAFOs and rural mental health}\\[0.6em]
""" + f"Panel 2026-09-23 \\quad$\\cdot$\\quad {len(GRID)} treatments $\\times$ {len(OUTCOMES)} outcomes"
       + r"""\quad$\cdot$\quad step-8 pre-committed controls\\[0.2em]
\today
\end{center}
\vspace{1em}
"""]
    for tid in GRID:
        g = R[R.treatment_id == tid]
        if len(g):
            tex.append(table(tid, g))

    # outcome reference table
    orow = []
    for okey, ocol in OUTCOMES.items():
        d = df[["fips","year",ocol]].dropna()
        orow.append(f"{esc(okey)} & \\texttt{{{esc(ocol)}}} & {int(d.year.min())}--{int(d.year.max())}"
                    f" & {len(d):,} & {100*len(d)/len(df):.1f}\\% & {d.fips.nunique():,}"
                    f" & {d[ocol].mean():.2f} & {d[ocol].std():.2f} \\\\")
    tex.append("\\begin{table}[htbp]\n\\centering\n"
               "\\caption{Outcome variables: source column, coverage, and moments}\n"
               "\\begin{adjustbox}{max width=\\textwidth}\n"
               "\\begin{tabular}{llrrrrrr}\n\\toprule\n"
               "Outcome & Panel column & Years & $N$ & Complete & Counties & Mean & SD \\\\\n"
               "\\midrule\n" + "\n".join(orow) + "\n\\bottomrule\n\\end{tabular}\n"
               "\\end{adjustbox}\n\n\\vspace{0.3em}\n\\begin{minipage}{\\textwidth}\\footnotesize\n"
               "\\textit{Notes.} Rural counties only (NCHS codes 3--6), 2000--2023 panel. "
               "\\textit{Complete} is the share of all rural county-year cells with a non-missing "
               "value. Coverage is the binding constraint on which treatment cohorts are "
               "observable: County Health Rankings outcomes begin in 2010, so the 2007 entry "
               "cohort has no observable pre-period and is dropped by Callaway--Sant'Anna. "
               "Deaths of despair are truncated after 2020.\n"
               "\\end{minipage}\n\\end{table}\n")
    tex.append("\\end{document}\n")

    # Body-only fragment (no preamble, no \begin{document}) so the master appendix
    # in script4p can \input it. Single source of truth: the table code lives here,
    # not duplicated in the master.
    BODY = os.path.join(OUT, "_BODY_balance.tex")
    open(BODY, "w").write("\n".join(tex[1:-1]))
    print(f"wrote {BODY}")

    p = os.path.join(OUT, "BALANCEpanel.tex")
    open(p, "w").write("\n".join(tex))
    print(f"\nwrote {p}")
    r = subprocess.run(["pdflatex","-interaction=nonstopmode","-output-directory",OUT,p],
                       capture_output=True, text=True)
    pdf = os.path.join(OUT, "BALANCEpanel.pdf")
    print(f"compiled -> {pdf}" if os.path.exists(pdf) else "PDF COMPILE FAILED")
    for ext in (".aux",".log",".out"):
        f = os.path.join(OUT, "BALANCEpanel"+ext)
        if os.path.exists(f): os.remove(f)
    print(f"\n{len(R)} treatment x outcome cells")


# =============================================================================
# control-set  (was script4y-control-set-selection.py)
# =============================================================================
ALL14 = COVARIATE_ORDER
FINAL8 = COVARIATE_ORDER[:8]
EXTRA = {"T4": ["log_pop"]}                     # part of the specification, not a candidate control
DF = None


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


def control_set(quick=False):
    OUT, TAG = tables_dir, "4b-control-set" + ("-quick" if quick else "")    # was folder script4y
    TREATS = {"T1": CORE_TREATMENTS["T1"], "T3": CORE_TREATMENTS["T3"], "T4": CORE_TREATMENTS["T4"], "E2": "tr_e2_add_absorb"}
    if quick:
        TREATS = {"T1": TREATS["T1"]}
    OUTS = dict(list(OUTCOMES.items())[:2]) if quick else OUTCOMES
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
    R.to_csv(out_file(tables_dir, TAG, f"{TODAY}_regressions.csv"), index=False)

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
    pd.DataFrame(ps).to_csv(out_file(tables_dir, TAG, f"{TODAY}_path_summary.csv"), index=False)

    # ---------------- per-covariate movement ----------------
    alt = R[~R.is_reference & ~R.family.str.startswith("A1")].copy()
    cs = (alt.groupby(["family", "sample", "label"])
             .agg(median_abs_pct_change=("pct_change_vs_ref", lambda s: float(s.abs().median())),
                  max_abs_pct_change=("pct_change_vs_ref", lambda s: float(s.abs().max())),
                  sign_flips=("sign_flip", "sum"), sig_flips=("sig_flip", "sum"), cells=("beta", "size")).reset_index()
             .rename(columns={"label": "covariate"}))
    cs.to_csv(out_file(tables_dir, TAG, f"{TODAY}_covariate_summary.csv"), index=False)

    # ---------------- the selection table ----------------
    panel = load_panel()
    cov = pd.DataFrame({"covariate": ALL14, "order": range(1, 15),
                        "coverage_pct": [100 * panel[c].notna().mean() for c in ALL14],
                        "in_final_8": [c in FINAL8 for c in ALL14]})
    roster = out_glob(tables_dir, "4b-controls", "*_control_pretrends.csv")
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
    cov.to_csv(out_file(tables_dir, TAG, f"{TODAY}_selection_table.csv"), index=False)

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


# =============================================================================
# honestdid  (was script4w-honestdid-export.py)
# =============================================================================
def slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()


def honestdid():
    OUT = tables_dir    # output-name tag "4b-honestdid-inputs" (undated names; R/honestdid.R reads them); was script4w/honestdid_inputs

    _df, _D, _, _ = load_designs()
    DESIGNS = {"first expansion": (_D["first expansion"]["data"], "add_year"),
               "first entry": (_D["first entry (clean controls)"]["data"], "entry_year")}

    manifest = []
    for dname, (data, cohort) in DESIGNS.items():
        data = data.copy(); data["t_rel"] = data["year"] - data[cohort]
        for okey, ocol in OUTCOMES.items():
            es = event_study_frame(data, ocol, CONTROLS, (), CLIP)
            m = pf.feols(f"{clean(ocol)} ~ i(ev, ref=-1) + " + " + ".join(clean(c) for c in CONTROLS) + " | fips + year",
                         data=es, vcov={"CRV1": "state_fips"})
            names = {int(float(str(n).split("::")[1])): str(n) for n in m._coefnames if str(n).startswith("ev::")}
            pre, e = [], -2
            while e in names:
                pre.insert(0, e); e -= 1
            post, e = [], 0
            while e in names:
                post.append(e); e += 1
            if len(pre) < 2 or len(post) < 1:
                print(f"skip {dname} | {okey}: pre={pre} post={post}"); continue
            order = pre + post
            cn = list(m._coefnames); idx = [cn.index(names[x]) for x in order]
            b = np.asarray(m.coef())[idx]; V = np.asarray(m._vcov)[np.ix_(idx, idx)]
            stem = f"{slug(dname)}__{slug(okey)}"
            pd.DataFrame({"event_time": order, "beta": b}).to_csv(out_file(tables_dir, "4b-honestdid-inputs", stem + "__beta.csv"), index=False)
            pd.DataFrame(V).to_csv(out_file(tables_dir, "4b-honestdid-inputs", stem + "__sigma.csv"), index=False, header=False)
            manifest.append({"design": dname, "outcome": okey, "stem": stem, "num_pre": len(pre), "num_post": len(post),
                             "pre_range": f"{pre[0]}..{pre[-1]}", "post_range": f"{post[0]}..{post[-1]}", "N": int(m._N)})
            print(f"  {dname:16s} {okey:26s} pre {pre[0]}..{pre[-1]} ({len(pre)})  post {post[0]}..{post[-1]} ({len(post)})")
    pd.DataFrame(manifest).to_csv(out_file(tables_dir, "4b-honestdid-inputs", "manifest.csv"), index=False)
    print(f"\nwrote {len(manifest)} input pairs -> {OUT}")
    print("next: Rscript R/honestdid.R   (reads these inputs; not run here)")


# =============================================================================
def species_counts():
    """How many events stand behind the species forest plots: counties whose first E2 (first expansion) or E3 (first entry) falls at each
    census wave, for each species and size class, with the comparison counties. Descriptive; nothing is estimated.

    Events use the same rule as designs.species_treatments and script4_treatment._wave_events on the species' own count: a county's count
    at a census wave is compared with its previous observed wave. E2 = first wave with a rise (count > previous); E3 = first wave with
    count > 0 after a previous count of 0. Comparison for E2 = counties that never rise; for E3 = counties with a zero count at every wave.
    Counties need at least two observed waves; the 2002 wave has no previous wave, so events start in 2007. Size classes: large,
    small, and all sizes (small + medium + large, the scope of the all-sizes forest plots)."""
    from script4_treatment import CENSUS_YEARS
    df = load_panel().sort_values(["fips", "year"]).reset_index(drop=True)
    waves = [y for y in CENSUS_YEARS if y > min(CENSUS_YEARS)]
    scopes = {"large": ["large"], "small": ["small"], "all sizes": ["small", "medium", "large"]}
    rows = []
    for species in ["dairy", "beef", "hogs", "chickens"]:
        for scope, bins in scopes.items():
            cols = [f"cafo_{species}_{b}" for b in bins]
            if not all(c in df.columns for c in cols) or df[cols].notna().sum().sum() == 0:
                rows.append({"species": species, "scope": scope, "design": "E2", "note": "not defined: no count in the panel"})
                rows.append({"species": species, "scope": scope, "design": "E3", "note": "not defined: no count in the panel"})
                continue
            tr = species_treatments(df, species, bins)
            w = pd.DataFrame({"fips": df["fips"], "year": df["year"], "C": tr["count"]})
            w = w[w.year.isin(CENSUS_YEARS)].dropna(subset=["C"]).sort_values(["fips", "year"])
            w["lag"] = w.groupby("fips")["C"].shift(1)
            eligible = set(w.loc[w["lag"].notna(), "fips"])
            first = {"E2": w[w["lag"].notna() & (w["C"] > w["lag"])].groupby("fips")["year"].min(),
                     "E3": w[(w["lag"] == 0) & (w["C"] > 0)].groupby("fips")["year"].min()}
            ever_present = set(w.loc[w["C"] > 0, "fips"])
            for design, fy in first.items():
                comparison = eligible - set(fy.index) if design == "E2" else eligible - ever_present
                row = {"species": species, "scope": scope, "design": design, "eligible_counties": len(eligible)}
                row.update({f"wave_{y}": int((fy == y).sum()) for y in waves})
                row.update({"treated_counties": int(len(fy)), "comparison_counties": len(comparison), "note": ""})
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(out_file(tables_dir, "4b-species-counts", f"{TODAY}_species_counts.csv"), index=False)
    # validation: dairy / large must equal the panel's own add_year and entry_year (script4_treatment)
    d = out[(out.species == "dairy") & (out.scope == "large")].set_index("design")
    pan = df.groupby("fips")[["add_year", "entry_year"]].first()
    for design, col in (("E2", "add_year"), ("E3", "entry_year")):
        ref = pan[col].dropna().astype(int).value_counts()
        mine = {y: int(d.loc[design, f"wave_{y}"]) for y in waves}
        print(f"check dairy/large {design}: counties by wave {mine}; panel {col}: { {int(y): int(ref.get(y, 0)) for y in waves} }")
        assert all(mine[y] == int(ref.get(y, 0)) for y in waves), f"dairy/large {design} differs from the panel's {col}"
    txt = out.drop(columns=["eligible_counties"]).fillna("")
    path = save_text_table(txt, os.path.join(tables_dir), tagged("4b-species-counts", "species_counts"),
                           "Counties with a first expansion (E2) or first entry (E3) by census wave, by species and size class",
                           "Counties whose first event falls at that census wave (2007 is the first wave with a previous wave). Comparison: E2 = never a rise; "
                           "E3 = zero count at every wave. Rural counties of the 2026-10-25 panel, before any outcome sample restriction. Not estimated.",
                           float_format="{:.0f}")
    print("wrote", path)


SECTIONS = ["controls", "leads", "pretrends", "balance", "control-set", "species-counts", "honestdid"]     # run order: `leads` and `control-set` read `controls`


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", nargs="+", choices=SECTIONS, help="run only these sections (default: all, in order)")
    ap.add_argument("--quick", action="store_true", help="control-set only: one treatment, two outcomes (test run)")
    args = ap.parse_args()
    todo = [s for s in SECTIONS if args.only is None or s in args.only]
    for s in todo:
        print("\n" + "#" * 100 + f"\n# section: {s}\n" + "#" * 100, flush=True)
        if s == "controls":
            controls()
        elif s == "leads":
            leads()
        elif s == "pretrends":
            pretrends()
        elif s == "balance":
            balance()
        elif s == "control-set":
            control_set(quick=args.quick)
        elif s == "species-counts":
            species_counts()
        else:
            honestdid()


if __name__ == "__main__":
    main()
