#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4c-covariate-stability.py  --  synthesis of script4b. NO new estimation.

PURPOSE
-------
script4b ran 840 regressions: for each treatment x outcome x FE spec it walked
the covariate set in a fixed order, one variable at a time, step 0 -> step 14.
That produced 56 "paths" (4 treatments x 7 outcomes x 2 FE specs).

This script asks one question of each path: DOES THE ANSWER DEPEND ON WHERE YOU
STOP ADDING COVARIATES?

A path is FRAGILE if any of the following holds between step 0 and step 14:
    - the sign of beta flips
    - the 5% significance verdict changes (either direction)
    - |total change in beta| exceeds 25%

WHY THIS MATTERS, AND WHAT IT IS NOT
------------------------------------
This is a measure of researcher degrees of freedom, not a model-selection rule.
If a path flips sign or gains significance only once controls are added, then
"the effect" is a statement about the control set as much as about the treatment.
That must be disclosed, not resolved by picking the step with the nicest result.

Choosing a specification BECAUSE of this table and then reporting conventional
p-values on it is post-selection inference. The reported specification stays
pre-committed (see CONTROL-DECISION-REGISTER.md). This is an appendix exhibit.

STEP 8 vs STEP 14
-----------------
Both are reported. Step 8 is the last step with the sample intact (all eight
full-coverage covariates in, N ~ 65,578). Step 9 brings SAHIE, whose 2008 start
truncates the window, so from there the covariate is confounded with the sample
restriction. A path that is stable to step 8 but moves after it is telling you
about SAMPLE, not about confounding.

Input  <- Data/output/tables/script4b/<date>_covariate_sequence.csv
Output -> Data/output/tables/script4c/
            <date>_path_stability.csv     one row per path (56)
            <date>_stability_summary.csv  aggregate by FE spec / outcome
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import numpy as np, pandas as pd

from script4_treatment import tables_dir, os, date

IN_DIR  = os.path.join(tables_dir, "script4b")
OUT     = os.path.join(tables_dir, "script4c"); os.makedirs(OUT, exist_ok=True)
TODAY   = date.today().strftime("%Y-%m-%d")

ALPHA          = 0.05   # significance verdict used for the flip test
BIG_MOVE_PCT   = 25.0   # |total change| above this counts as fragile
FULL_SAMPLE_STEP = 8    # last step before SAHIE truncates the window
SE_COL, P_COL  = "se_county", "p_county"   # county-clustered is the headline vcov

# ---------------------------------------------------------------- load
src = sorted(glob.glob(os.path.join(IN_DIR, "*_covariate_sequence.csv")))[-1]
d = pd.read_csv(src)
print(f"source : {os.path.basename(src)}  ({len(d):,} regressions)")

KEY = ["treatment_id", "treatment", "outcome", "fe_spec"]

# ---------------------------------------------------------------- per-path
rows = []
for key, g in d.groupby(KEY, sort=False):
    g = g.sort_values("step")
    s0 = g[g.step == 0].iloc[0]                 # no covariates
    sN = g[g.step == g.step.max()].iloc[0]      # all covariates
    s8 = g[g.step == FULL_SAMPLE_STEP]
    s8 = s8.iloc[0] if len(s8) else None

    b0, bN = s0["beta"], sN["beta"]
    p0, pN = s0[P_COL], sN[P_COL]

    # % change is undefined against a zero baseline; guard it
    tot_pct = 100.0 * (bN - b0) / abs(b0) if b0 != 0 else np.nan

    sign_flip = bool(np.sign(b0) != np.sign(bN) and b0 != 0 and bN != 0)
    sig0, sigN = bool(p0 < ALPHA), bool(pN < ALPHA)

    if   sig0 and not sigN: sig_change = "loses significance"
    elif sigN and not sig0: sig_change = "gains significance"
    else:                   sig_change = ""

    # the widest excursion anywhere along the path, not just the endpoints
    max_excursion = (100.0 * (g["beta"] - b0).abs().max() / abs(b0)) if b0 != 0 else np.nan

    rec = dict(zip(KEY, key))
    rec.update({
        "beta_step0":            b0,
        "beta_step8":            s8["beta"] if s8 is not None else np.nan,
        "beta_final":            bN,
        "p_step0":               p0,
        "p_final":               pN,
        "total_change_pct":      tot_pct,
        "max_excursion_pct":     max_excursion,
        # step 8 isolates confounding from sample loss
        "change_to_step8_pct":   (100.0 * (s8["beta"] - b0) / abs(b0))
                                 if (s8 is not None and b0 != 0) else np.nan,
        "sign_flip":             sign_flip,
        "sig_change":            sig_change,
        "vif_max":               g["vif_max"].max(),
        "N_step0":               s0["N"],
        "N_final":               sN["N"],
        "N_lost_pct":            100.0 * (s0["N"] - sN["N"]) / s0["N"],
        "switchers_step0":       s0["n_switchers"],
        "switchers_final":       sN["n_switchers"],
    })
    rec["fragile"] = bool(sign_flip or sig_change
                          or (pd.notna(tot_pct) and abs(tot_pct) > BIG_MOVE_PCT))
    rows.append(rec)

paths = pd.DataFrame(rows)
paths = paths.sort_values("total_change_pct", key=lambda s: s.abs(), ascending=False)

# ---------------------------------------------------------------- aggregates
def _block(g):
    return pd.Series({
        "n_paths":            len(g),
        "median_abs_change":  g["total_change_pct"].abs().median(),
        "sign_flips":         int(g["sign_flip"].sum()),
        "sig_changes":        int((g["sig_change"] != "").sum()),
        "n_fragile":          int(g["fragile"].sum()),
        "pct_fragile":        100.0 * g["fragile"].mean(),
        "vif_max":            g["vif_max"].max(),
    })

by_fe      = paths.groupby("fe_spec").apply(_block).reset_index()
by_outcome = paths.groupby("outcome").apply(_block).reset_index()
by_outcome = by_outcome.sort_values("median_abs_change", ascending=False)
by_tr      = paths.groupby("treatment").apply(_block).reset_index()

by_fe.insert(0, "grouping", "fe_spec")
by_outcome.insert(0, "grouping", "outcome")
by_tr.insert(0, "grouping", "treatment")
by_fe = by_fe.rename(columns={"fe_spec": "level"})
by_outcome = by_outcome.rename(columns={"outcome": "level"})
by_tr = by_tr.rename(columns={"treatment": "level"})
summary = pd.concat([by_fe, by_outcome, by_tr], ignore_index=True)

# ---------------------------------------------------------------- write
p1 = os.path.join(OUT, f"{TODAY}_path_stability.csv")
p2 = os.path.join(OUT, f"{TODAY}_stability_summary.csv")
paths.to_csv(p1, index=False)
summary.to_csv(p2, index=False)

# ---------------------------------------------------------------- report
print(f"\npaths  : {len(paths)}   fragile: {int(paths.fragile.sum())} "
      f"({100*paths.fragile.mean():.0f}%)")
print(f"         sign flips {int(paths.sign_flip.sum())} | "
      f"significance changes {int((paths.sig_change!='').sum())} | "
      f"max VIF {paths.vif_max.max():.2f}")

print("\nBY FE SPEC")
print(by_fe[["level","median_abs_change","sign_flips","sig_changes","pct_fragile"]]
      .to_string(index=False, float_format=lambda x: f"{x:.1f}"))

print("\nBY OUTCOME (most fragile first)")
print(by_outcome[["level","median_abs_change","sign_flips","sig_changes","pct_fragile"]]
      .to_string(index=False, float_format=lambda x: f"{x:.1f}"))

print("\nTEN LEAST STABLE PATHS")
show = ["treatment_id","outcome","fe_spec","beta_step0","beta_step8","beta_final",
        "total_change_pct","sign_flip","sig_change"]
print(paths[show].head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print(f"\nwrote {p1}")
print(f"wrote {p2}")
