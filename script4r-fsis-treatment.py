#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4r-fsis-treatment.py -- slaughterhouses (FSIS plants) as the treatment.

Treatments: F1 = number of FSIS plants (2017-2023); F2 = first year the count rises. Models R0-R3 (event counts, TWFE on F1 and F2, event study).
Reads:   panel via script4_treatment.load_panel(); Data/merged/*_panel_fsis.csv
Writes:  Data/output/tables/script4r/<date>_{R0_entry_events, R1_R2_twfe, R3_event_study, R4_lpdid}.csv
Run:     python3 script4r-fsis-treatment.py
Notes:   FSIS starts in 2017, so entries before 2017 are invisible; R4 (LP-DiD) is computed but not reported.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pyfixest as pf
from script4_treatment import load_panel, OUTCOMES, COVARIATE_ORDER, db_data, tables_dir, os, date
from functions import latest_file_glob
from estimators import lpdid

OUT = os.path.join(tables_dir, "script4r"); os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
CONTROLS = COVARIATE_ORDER[:8]
SLAUGHTER = "n_slaughterhouse_present_establishments_fsis"
FSIS_YEARS = (2017, 2023)


def cl(c):
    return (c.replace("%", "pct").replace("-", "_").replace(" ", "_").replace("(", "")
             .replace(")", "").replace("/", "_").replace("+", "p").replace(",", "_"))


df = load_panel()
fp = latest_file_glob(os.path.join(db_data, "merged"), "*_panel_fsis.csv")
f = pd.read_csv(fp, low_memory=False, usecols=["fips", "year", SLAUGHTER])
f["fips"] = f["fips"].astype(str).str.zfill(5)
df = df.merge(f, on=["fips", "year"], how="left")
inwin = df.year.between(*FSIS_YEARS)
df["F1_count"] = np.where(inwin, df[SLAUGHTER].fillna(0), np.nan)

# entry: first year the count rises, observed 2018+
w = df[inwin].sort_values(["fips", "year"])
w["d"] = w.groupby("fips")["F1_count"].diff()
first_up = w[w.d > 0].groupby("fips")["year"].min()
df["F_cohort"] = df["fips"].map(first_up)
df["F2_entry"] = np.where(df.F_cohort.notna() & (df.year >= df.F_cohort), 1.0, 0.0)
df["F_trel"] = df.year - df.F_cohort
print(f"FSIS merged from {os.path.basename(fp)}")

# ============================== R0 event counts ==============================
ev = w.assign(up=w.d > 0, down=w.d < 0)
r0 = pd.DataFrame({
    "counties_first_entry": first_up.value_counts().sort_index(),
    "counties_any_increase": ev[ev.up].groupby("year").fips.nunique(),
    "counties_any_decrease": ev[ev.down].groupby("year").fips.nunique(),
}).fillna(0).astype(int)
r0.index.name = "year"
r0.loc["total"] = r0.sum()
r0["note"] = ""
base17 = w[w.year == 2017]
print("\nR0  entry events by year")
print(r0.to_string())
print(f"  counties with >=1 slaughter plant in 2017: {(base17.F1_count > 0).sum()} of {len(base17)}")
# usable cohorts per outcome: cohort g needs outcome data at g-1 and g
use = []
for okey, ocol in OUTCOMES.items():
    yrs = set(df.loc[df[ocol].notna(), "year"])
    for g, n in first_up.value_counts().sort_index().items():
        use.append({"outcome": okey, "cohort": int(g), "counties": int(n),
                    "usable": (g - 1 in yrs) and (g in yrs)})
use = pd.DataFrame(use)
r0.to_csv(os.path.join(OUT, f"{TODAY}_R0_entry_events.csv"))
use.to_csv(os.path.join(OUT, f"{TODAY}_R0_cohorts_by_outcome.csv"), index=False)
print("\n  usable entry counties by outcome:")
print(use[use.usable].groupby("outcome").counties.sum().to_string())


# ============================== R1 / R2 TWFE =================================
def twfe(sub, y, t, label):
    fml = f"{y} ~ {t} + " + " + ".join(cl(c) for c in CONTROLS) + " | fips + year"
    m = pf.feols(fml, data=sub, vcov={"CRV1": "state_fips"})
    r = m.tidy().loc[t]
    sw = int((sub.groupby("fips")[t].nunique() > 1).sum())
    sdw = float((sub[y] - sub.groupby("fips")[y].transform("mean")).std())
    se = float(r["Std. Error"])
    return {"model": label, "beta": float(r["Estimate"]), "se_state": se,
            "p_state": float(r["Pr(>|t|)"]), "N": int(m._N),
            "n_counties": int(sub.fips.nunique()), "n_states": int(sub.state_fips.nunique()),
            "n_switchers": sw, "sd_within": sdw,
            "MDE_sd_within": 2.8016 * se / sdw if sdw > 0 else np.nan}

rows = []
for okey, ocol in OUTCOMES.items():
    need = ["fips", "year", "state_fips", ocol] + CONTROLS
    a = df[df.year.between(*FSIS_YEARS)][need + ["F1_count"]].dropna()
    a = a.rename(columns={c: cl(c) for c in a.columns})
    if len(a) > 200 and a.F1_count.std() > 0:
        try:
            rows.append({"outcome": okey, "treatment": "F1 count", **twfe(a, cl(ocol), "F1_count", "R1")})
        except Exception as e:
            print(f"  R1 {okey}: {e}")
    b = df[df.year.between(2012, 2023)][need + ["F2_entry"]].dropna()
    b = b.rename(columns={c: cl(c) for c in b.columns})
    if len(b) > 200 and b.F2_entry.std() > 0:
        try:
            rows.append({"outcome": okey, "treatment": "F2 entry", **twfe(b, cl(ocol), "F2_entry", "R2")})
        except Exception as e:
            print(f"  R2 {okey}: {e}")
tw = pd.DataFrame(rows)
tw.to_csv(os.path.join(OUT, f"{TODAY}_R1_R2_twfe.csv"), index=False)
print("\nR1/R2 TWFE")
print(tw[["model", "outcome", "beta", "se_state", "p_state", "N", "n_switchers",
          "MDE_sd_within"]].to_string(index=False))

# ============================== R3 event study ===============================
LEADS, LAGS = 5, 5
es_rows = []
for okey, ocol in OUTCOMES.items():
    need = ["fips", "year", "state_fips", ocol, "F_trel"] + CONTROLS
    s = df[df.year.between(2012, 2023)][need].copy()
    s["ev"] = s["F_trel"].clip(-LEADS, LAGS)
    s.loc[s.F_trel.isna(), "ev"] = -1
    s = s.drop(columns=["F_trel"]).dropna().rename(columns={c: cl(c) for c in need})
    if len(s) < 500:
        continue
    fml = f"{cl(ocol)} ~ i(ev, ref=-1) + " + " + ".join(cl(c) for c in CONTROLS) + " | fips + year"
    try:
        m = pf.feols(fml, data=s, vcov={"CRV1": "state_fips"})
    except Exception as e:
        print(f"  R3 {okey}: {e}"); continue
    for idx, r in m.tidy().iterrows():
        if not str(idx).startswith("ev::"):
            continue
        e = int(float(str(idx).split("::")[1]))
        b, se = float(r["Estimate"]), float(r["Std. Error"])
        es_rows.append({"model": "R3", "outcome": okey, "event_time": e, "beta": b,
                        "se_state": se, "p_state": float(r["Pr(>|t|)"]),
                        "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se, "N": int(m._N)})
es = pd.DataFrame(es_rows)
es.to_csv(os.path.join(OUT, f"{TODAY}_R3_event_study.csv"), index=False)
if len(es):
    pre = es[es.event_time < -1]
    print("\nR3 pre-trend: significant pre-period coefs at 5%")
    print(pre.groupby("outcome").apply(lambda g: f"{int((g.p_state < .05).sum())}/{len(g)}").to_string())

# ============================== R4 LP-DiD ====================================
lp_rows = []
for okey, ocol in OUTCOMES.items():
    pan = df[df.year.between(2012, 2023)][["fips", "year", "state_fips", "F_cohort", ocol] + CONTROLS]
    pan = pan.rename(columns={c: cl(c) for c in pan.columns})
    for arm, ctrls in [("no controls", []), ("step-8 at t-1", [cl(c) for c in CONTROLS])]:
        r = lpdid(pan, cl(ocol), cohort="F_cohort", horizons=range(-LEADS, LAGS + 1),
                  controls=ctrls)
        if r.empty:
            continue
        r.insert(0, "arm", arm); r.insert(0, "outcome", okey); r.insert(0, "model", "R4")
        lp_rows.append(r)
lp = pd.concat(lp_rows, ignore_index=True) if lp_rows else pd.DataFrame()
lp.to_csv(os.path.join(OUT, f"{TODAY}_R4_lpdid.csv"), index=False)
if len(lp):
    print("\nR4 LP-DiD, h=0 and pre-trend")
    for (o, a), g in lp.groupby(["outcome", "arm"]):
        h0 = g.loc[g.h == 0, "beta"]
        pr = g[g.h < -1]
        print(f"  {o:26s} {a:14s} h0={h0.iloc[0] if len(h0) else np.nan:+.4f}  "
              f"pre sig={int((pr.p < .05).sum())}/{len(pr)}  treated={g.n_treated.max()}")
print(f"\nscript4r COMPLETE -> {OUT}")
