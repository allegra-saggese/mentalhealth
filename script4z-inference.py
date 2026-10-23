"""script4z-inference.py -- inference for the confirmatory family (D-42): wild cluster bootstrap, Holm, Romano-Wolf.

Plan:    code-audit/plans/2026-10-22_PLAN_inference-upgrade.md (Parts 1 and 3); tools in inference_tools.py.
Reads:   panel via script4_treatment.load_panel(); script4a's A2 grid (to check this script reproduces it exactly).
Writes:  Data/output/tables/script4z/YYYY-MM-DD_{confirmatory_family,exploratory_family,bootstrap_correlation}.csv
Run:     PYTHONHASHSEED=0 python3 script4z-inference.py          (--quick: 499 draws, separate output folder)
Runtime: about a minute per bootstrap at 9,999 draws (8 bootstraps).

Confirmatory family F1 = Poor MH Days and Frequent Mental Distress x {T1, T3, T4, E2}: TWFE, county + year FE,
the pre-committed 8 controls, state clusters. Wild cluster bootstrap-t (null imposed, Webb weights), then
  - Holm over all 8 wild-bootstrap p-values (family-wise error across outcomes and treatments),
  - Romano-Wolf step-down WITHIN each outcome (its four treatments share one sample, so one seed gives every
    treatment the same state weights in each draw, which is what the step-down needs).
Exploratory family F2 = all 28 reported A2 cells: Benjamini-Hochberg on the analytic state-clustered p-values.
"""
import sys

import numpy as np
import pandas as pd
import pyfixest as pf

import inference_tools as it
from did_tools import clean as _clean
from functions import latest_file_glob
from script4_treatment import (
    load_panel, OUTCOMES, CORE_TREATMENTS, SPEC_EXTRA_REGRESSORS, COVARIATE_ORDER, tables_dir, os, date,
)

QUICK = "--quick" in sys.argv
SEED = 20261022
REPS = 499 if QUICK else 9999
WEIGHTS = "webb"                                  # six-point weights: state sizes are very uneven
FAMILY_OUTCOMES = ["Poor MH Days", "Frequent Mental Distress"]
FAMILY_TREATMENTS = ["T1", "T3", "T4", "E2"]
TREATMENT_COLUMN = {**{t: CORE_TREATMENTS[t] for t in ("T1", "T3", "T4")}, "E2": "tr_e2_add_absorb"}
CONTROLS = COVARIATE_ORDER[:8]                    # pre-committed (D-16, D-31)
TOLERANCE = 1e-6                                  # replication-protocol: estimates and p-values at a fixed seed

OUT = os.path.join(tables_dir, "script4z_quick" if QUICK else "script4z")
os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")


def prepare(df, columns):
    """Complete cases on `columns`, with names cleaned for formulas (as in script4a)."""
    columns = list(dict.fromkeys(columns))
    sub = df[columns].dropna().copy().reset_index(drop=True)
    # wildboottest compiles with numba and fails on an object-dtype cluster column; integer codes give the same clusters
    sub["state_fips"] = pd.factorize(sub["state_fips"])[0]
    return sub.rename(columns={c: _clean(c) for c in columns})


print("=" * 78)
print(f"script4z -- inference for the confirmatory family   reps={REPS}  weights={WEIGHTS}  seed={SEED}")
print("=" * 78)
df = load_panel()
print(f"panel: {os.path.basename(df.attrs['panel_path'])}  {len(df):,} rows, {df.fips.nunique():,} counties")

grid = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4a"), "*_A1_A2_treatment_grid.csv"))
a2 = grid[grid.registry == "A2"].set_index(["treatment_id", "outcome"])

rows, boot_corr = [], []
for okey in FAMILY_OUTCOMES:
    ocol = OUTCOMES[okey]
    t_obs, t_boot, samples, fits = {}, {}, {}, {}
    for tid in FAMILY_TREATMENTS:
        tcol, extra = TREATMENT_COLUMN[tid], SPEC_EXTRA_REGRESSORS.get(tid, [])
        sub = prepare(df, ["fips", "year", "state_fips", ocol, tcol] + list(extra) + CONTROLS)
        names = [_clean(c) for c in [tcol] + list(extra) + CONTROLS]
        fit = pf.feols(f"{_clean(ocol)} ~ " + " + ".join(names) + " | fips + year", data=sub,
                       vcov={"CRV1": "state_fips"})
        tidy = fit.tidy().loc[_clean(tcol)]
        ref = a2.loc[(tid, okey)]
        # the self-check: this script must reproduce script4a's A2 coefficient and p-value exactly
        assert np.isclose(tidy["Estimate"], ref["beta"], rtol=TOLERANCE, atol=1e-8), f"beta differs: {tid} {okey}"
        assert np.isclose(tidy["Pr(>|t|)"], ref["p_state"], rtol=TOLERANCE, atol=1e-8), f"p differs: {tid} {okey}"
        res = it.wild_cluster_test(fit, _clean(tcol), reps=REPS, seed=SEED, weights_type=WEIGHTS)
        t_obs[tid], t_boot[tid], samples[tid] = res["t"], res["t_boot"], sub[["fips", "year"]]
        fits[tid] = {"beta": tidy["Estimate"], "se_state": tidy["Std. Error"], "p_state": tidy["Pr(>|t|)"],
                     "N": int(fit._N), "p_wild": res["p"], "n_switchers": ref["n_switchers"]}
        print(f"  {okey:26s} {tid}: beta {tidy['Estimate']:9.4f}  analytic p {tidy['Pr(>|t|)']:.4f}  "
              f"wild p {res['p']:.4f}   (reproduces script4a)")

    same_sample = all(samples[t].equals(samples[FAMILY_TREATMENTS[0]]) for t in FAMILY_TREATMENTS)
    t_matrix = np.column_stack([t_boot[t] for t in FAMILY_TREATMENTS])
    corr = np.corrcoef(t_matrix, rowvar=False)
    print(f"  {okey}: identical sample across the four treatments: {same_sample}; "
          f"smallest correlation between bootstrap t-statistics {corr[np.triu_indices(4, 1)].min():.2f}")
    for i, a in enumerate(FAMILY_TREATMENTS):
        for j, b in enumerate(FAMILY_TREATMENTS):
            if j > i:
                boot_corr.append({"outcome": okey, "treatment_a": a, "treatment_b": b, "correlation": corr[i, j]})
    rw = (it.romano_wolf([t_obs[t] for t in FAMILY_TREATMENTS], t_matrix) if same_sample
          else np.full(len(FAMILY_TREATMENTS), np.nan))
    for k, tid in enumerate(FAMILY_TREATMENTS):
        rows.append({"outcome": okey, "treatment_id": tid, "treatment_column": TREATMENT_COLUMN[tid],
                     **fits[tid], "t_wild": t_obs[tid], "p_romano_wolf_within_outcome": rw[k],
                     "same_sample_across_treatments": same_sample, "reps": REPS, "weights_type": WEIGHTS, "seed": SEED})

conf = pd.DataFrame(rows)
conf["p_holm_over_8"] = it.holm(conf.p_wild.to_numpy())
conf["family"] = "F1 confirmatory (D-42)"
conf.to_csv(os.path.join(OUT, f"{TODAY}_confirmatory_family.csv"), index=False)
pd.DataFrame(boot_corr).to_csv(os.path.join(OUT, f"{TODAY}_bootstrap_correlation.csv"), index=False)

expl = grid[(grid.registry == "A2") & grid.treatment_id.isin(FAMILY_TREATMENTS)][
    ["treatment_id", "outcome", "beta", "se_state", "p_state", "N", "n_switchers"]].reset_index(drop=True)
expl["p_bh_over_28"] = it.benjamini_hochberg(expl.p_state.to_numpy())
expl["family"] = "F2 exploratory (all reported A2 cells)"
expl.to_csv(os.path.join(OUT, f"{TODAY}_exploratory_family.csv"), index=False)

pd.set_option("display.width", 200)
show = ["outcome", "treatment_id", "beta", "p_state", "p_wild", "p_holm_over_8", "p_romano_wolf_within_outcome"]
print("\nF1 confirmatory family (8 tests):")
print(conf[show].round(4).to_string(index=False))
print(f"\nF2 exploratory family ({len(expl)} cells): significant at 5%  unadjusted {int((expl.p_state < .05).sum())},"
      f"  Benjamini-Hochberg {int((expl.p_bh_over_28 < .05).sum())}")
print(f"\nwrote {OUT}/{TODAY}_*.csv")
