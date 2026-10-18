#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4t-pretrend-leads.py  --  OUTCOME PRE-TRENDS (LEADS) FOR EACH REPORTED TREATMENT (added 2026-10-17)

QUESTION
--------
Before treatment, do outcomes already move differently in counties that go on to be treated?
Answered for every reported treatment definition, next to the covariate pre-trends from
script4j. DESCRIPTIVE TESTS ONLY: no verdict labels are assigned here.

TWO DESIGNS, because the treatments are of two kinds
----------------------------------------------------
Absorbing event treatments (E2 first positive change; E3 first entry, 0 -> 1+ large dairy):
    EVENT STUDY. Y_it = sum_{e != -1} theta_e 1[t - g_i = e] + X_it'gamma + county FE + year FE.
    Event time e = year - cohort year, clipped to [-8, 8]; reference e = -1; never-treated
    counties sit in the reference category. Leads are e = -8 ... -2.

Non-absorbing / continuous treatments (T1 presence, T3 per 10k, T4 log count):
    LEAD TEST. Y_it = b0 D_it + sum_{k=1..4} b_k D_{i,t+k} + [log_pop for T4] + X_it'gamma
    + county FE + year FE. D_{i,t+k} is the county's own treatment k years AHEAD. If treatment
    does not predict the outcome before it changes, b_1 ... b_4 are zero. Because treatment moves
    only at census waves (2007, 2012, 2017, 2022) and is forward-filled, a lead differs from D_it
    only in the k years before a wave. The last four panel years drop out (no lead available).

Both use the pre-committed control set (COVARIATE_ORDER[:8]), state-clustered SEs.
JOINT TEST: Wald chi-square that all leads are zero, with the state-clustered covariance.
With about 49 clusters and several leads, the chi-square approximation is rough: read the
joint p alongside the individual leads.

OUTPUTS -> Data/output/tables/script4t/
    <date>_leads_coefficients.csv    every lead coefficient (treatment, outcome, lead, beta, se, p)
    <date>_leads_joint_tests.csv     one row per treatment x outcome: n leads, Wald chi2, p, N
    (the covariate pre-trends are read from script4j's <date>_control_pretrends.csv and printed alongside)
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import chi2, norm

from script4_treatment import (
    load_panel, OUTCOMES, COVARIATE_ORDER, SPEC_EXTRA_REGRESSORS,
    CORE_TREATMENTS, tables_dir, os, date,
)

OUT = os.path.join(tables_dir, "script4t"); os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
CONTROLS = COVARIATE_ORDER[:8]
CLIP = 8             # event-time window for the event studies
N_LEADS = 4          # years of treatment leads for the continuous treatments
REPORTED = ["T1", "T3", "T4", "E2", "E3"]


def clean(name):
    return (name.replace("%", "pct").replace("-", "_").replace(" ", "_").replace("(", "")
                .replace(")", "").replace("/", "_").replace("+", "p").replace(",", "_"))


def joint_wald(model, names):
    """Wald chi-square that the named coefficients are all zero (state-clustered covariance)."""
    cn = list(model._coefnames)
    idx = [cn.index(n) for n in names if n in cn]
    if not idx:
        return np.nan, 0, np.nan
    b = np.asarray(model.coef())[idx]
    V = np.asarray(model._vcov)[np.ix_(idx, idx)]
    stat = float(b @ np.linalg.solve(V, b))
    return stat, len(idx), float(1 - chi2.cdf(stat, len(idx)))


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
coefs.to_csv(os.path.join(OUT, f"{TODAY}_leads_coefficients.csv"), index=False)
joint.to_csv(os.path.join(OUT, f"{TODAY}_leads_joint_tests.csv"), index=False)


def st(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


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
cp = sorted(glob.glob(os.path.join(tables_dir, "script4a", "*_control_pretrends.csv")))
if cp:
    c = pd.read_csv(cp[-1])
    print("\n" + "=" * 100)
    print(f"COVARIATE PRE-TRENDS  (script4j: {os.path.basename(cp[-1])})")
    print("=" * 100)
    print(c[["variable", "causal_role", "n_pre_coefs", "n_pre_significant", "max_pre_in_sd", "verdict"]]
          .round({"max_pre_in_sd": 3}).to_string(index=False))
print(f"\nSaved -> {OUT}")
