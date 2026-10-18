#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4w-honestdid-export.py  --  inputs for the Rambachan-Roth sensitivity analysis (added 2026-10-17)

Refits the BASELINE TWFE event study of script4u (county + year FE, 8 controls, state-clustered covariance)
for each treatment definition and outcome, and writes what R's HonestDiD package needs:
    <design>__<outcome>__beta.csv     event_time, beta               (pre-periods then post-periods, reference e=-1 omitted)
    <design>__<outcome>__sigma.csv    full state-clustered covariance matrix of those coefficients
    manifest.csv                      one row per file pair with the number of pre and post periods

Only a CONTIGUOUS run of event times is exported (HonestDiD indexes periods by position): leads run back
from e=-2 as far as every lead is present, lags run forward from e=0 as far as every lag is present.

NOTE: this repeats the sample construction of script4u (copied, not imported, because script4u runs on
import). Merging the two is on the clean-up list.

Then run script4w-honestdid.R.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf

from script4_treatment import load_panel, OUTCOMES, COVARIATE_ORDER, CENSUS_YEARS, tables_dir, os

OUT = os.path.join(tables_dir, "script4w", "honestdid_inputs"); os.makedirs(OUT, exist_ok=True)
CONTROLS = COVARIATE_ORDER[:8]
CLIP = 8


def clean(name):
    return (name.replace("%", "pct").replace("-", "_").replace(" ", "_").replace("(", "")
                .replace(")", "").replace("/", "_").replace("+", "p").replace(",", "_"))


def slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()


df = load_panel().sort_values(["fips", "year"]).reset_index(drop=True)
waves = df[df.year.isin(CENSUS_YEARS)].pivot(index="fips", columns="year", values="cafo_dairy_large")
never_large = set(waves.index[~(waves.fillna(0) > 0).any(axis=1)])
entry = set(df.loc[df["entry_year"].notna(), "fips"].unique())
DESIGNS = {"first expansion": (df, "add_year"),
           "first entry": (df[df["fips"].isin(never_large | entry)], "entry_year")}

manifest = []
for dname, (data, cohort) in DESIGNS.items():
    data = data.copy(); data["t_rel"] = data["year"] - data[cohort]
    for okey, ocol in OUTCOMES.items():
        need = ["fips", "year", "state_fips", ocol, "t_rel"] + CONTROLS
        es = data[need].copy()
        es["ev"] = es["t_rel"].clip(-CLIP, CLIP)
        es.loc[es["t_rel"].isna(), "ev"] = -1
        es = es.drop(columns=["t_rel"]).dropna().rename(columns={c: clean(c) for c in es.columns if c != "t_rel"})
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
        pd.DataFrame({"event_time": order, "beta": b}).to_csv(os.path.join(OUT, stem + "__beta.csv"), index=False)
        pd.DataFrame(V).to_csv(os.path.join(OUT, stem + "__sigma.csv"), index=False, header=False)
        manifest.append({"design": dname, "outcome": okey, "stem": stem, "num_pre": len(pre), "num_post": len(post),
                         "pre_range": f"{pre[0]}..{pre[-1]}", "post_range": f"{post[0]}..{post[-1]}", "N": int(m._N)})
        print(f"  {dname:16s} {okey:26s} pre {pre[0]}..{pre[-1]} ({len(pre)})  post {post[0]}..{post[-1]} ({len(post)})")
pd.DataFrame(manifest).to_csv(os.path.join(OUT, "manifest.csv"), index=False)
print(f"\nwrote {len(manifest)} input pairs -> {OUT}")
