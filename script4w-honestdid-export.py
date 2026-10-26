#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4w-honestdid-export.py -- inputs for the Rambachan-Roth sensitivity analysis (the R half is R/honestdid.R).

Refits the baseline TWFE event study for each design and outcome and writes the coefficients and the state-clustered covariance matrix.
Reads:   panel via designs.load_designs()
Writes:  Data/output/tables/script4w/honestdid_inputs/<design>__<outcome>__{beta, sigma}.csv and manifest.csv
Run:     python3 script4w-honestdid-export.py && Rscript R/honestdid.R
Method:  METHODS.md section 3. Only a contiguous run of event times is exported (HonestDiD indexes periods by position).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf

from estimators import clean, event_study_frame
from designs import load_designs
from script4_treatment import OUTCOMES, COVARIATE_ORDER, tables_dir, os

OUT = os.path.join(tables_dir, "script4w", "honestdid_inputs"); os.makedirs(OUT, exist_ok=True)
CONTROLS = COVARIATE_ORDER[:8]
CLIP = 8


def slug(s):
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()


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
        pd.DataFrame({"event_time": order, "beta": b}).to_csv(os.path.join(OUT, stem + "__beta.csv"), index=False)
        pd.DataFrame(V).to_csv(os.path.join(OUT, stem + "__sigma.csv"), index=False, header=False)
        manifest.append({"design": dname, "outcome": okey, "stem": stem, "num_pre": len(pre), "num_post": len(post),
                         "pre_range": f"{pre[0]}..{pre[-1]}", "post_range": f"{post[0]}..{post[-1]}", "N": int(m._N)})
        print(f"  {dname:16s} {okey:26s} pre {pre[0]}..{pre[-1]} ({len(pre)})  post {post[0]}..{post[-1]} ({len(post)})")
pd.DataFrame(manifest).to_csv(os.path.join(OUT, "manifest.csv"), index=False)
print(f"\nwrote {len(manifest)} input pairs -> {OUT}")
