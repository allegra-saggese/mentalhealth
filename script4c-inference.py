"""script4c-inference.py -- inference for the cohort and confirmatory-family results: wild bootstrap and permutation tests.

Merges the former script4z-inference.py (D-42, D-43) and script4ze-permutation-inference.py (D-47); the code of each is
moved unchanged into one function. Tools are in inference_tools.py; the sample builder is designs.usable_sample.

Sections (select with --only; default runs both, `wild` first):
  wild         run_wild()         confirmatory family F1 = Poor MH Days and Frequent Mental Distress x {T1, T3, T4, E2}:
                                  TWFE, county + year FE, the pre-committed 8 controls, state clusters. Wild cluster
                                  bootstrap-t (null imposed, Webb weights), Holm over all 8 p-values, Romano-Wolf step-down
                                  within each outcome. Exploratory family F2 = all 28 reported A2 cells: Benjamini-Hochberg
                                  on the analytic p-values.
                                  Reads script4a's A2 grid (checked: this section reproduces it exactly).
                                  Writes Data/output/tables/script4z/YYYY-MM-DD_{confirmatory_family,exploratory_family,
                                  bootstrap_correlation}.csv       (--quick: script4z_quick, 499 draws)
  permutation  run_permutation()  TWFE coefficient of the cohort designs (first expansion E2, first entry E3), all 7
                                  outcomes: first-treated years shuffled WITHIN each state, never-treated counties fixed;
                                  p = (1 + #{|null| >= |observed|}) / (1 + draws). Tests the sharp null of no effect.
                                  Reads the `wild` section's confirmatory family (script4z folder, shown for reference) and
                                  script4za's overall estimates (the observed coefficient is checked to 1e-6).
                                  Writes Data/output/tables/script4ze/YYYY-MM-DD_{permutation_pvalues,null_draws}[_<design>].csv
                                  (--quick: script4ze_quick, 49 draws, first outcome only)

Run:     PYTHONHASHSEED=0 python3 script4c-inference.py [--only wild|permutation] [--quick] [--design expansion|entry]
         --design applies to `permutation` only (run the two designs as two processes in a full run).
Runtime: `wild` about a minute per bootstrap at 9,999 draws (8 bootstraps); `permutation` about 0.12 s per draw,
         2,000 draws x 7 outcomes x 2 designs is roughly an hour.
Seeds:   wild 20261022; permutation 20261023 (one stream per design x outcome cell, order-independent).
"""
import argparse
import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import pyfixest as pf

import inference_tools as it
from designs import load_designs, usable_sample
from estimators import clean as _clean
from functions import latest_file_glob
from script4_treatment import (
    load_panel, OUTCOMES, CORE_TREATMENTS, SPEC_EXTRA_REGRESSORS, COVARIATE_ORDER, tables_dir,
)

SECTIONS = ("wild", "permutation")
CONTROLS = COVARIATE_ORDER[:8]                    # pre-committed (D-16, D-31)
TOLERANCE = 1e-6                                  # replication-protocol: estimates and p-values at a fixed seed
TODAY = date.today().strftime("%Y-%m-%d")

# --- wild section constants (former script4z)
WILD_SEED = 20261022
WILD_WEIGHTS = "webb"                             # six-point weights: state sizes are very uneven
FAMILY_OUTCOMES = ["Poor MH Days", "Frequent Mental Distress"]
FAMILY_TREATMENTS = ["T1", "T3", "T4", "E2"]
TREATMENT_COLUMN = {**{t: CORE_TREATMENTS[t] for t in ("T1", "T3", "T4")}, "E2": "tr_e2_add_absorb"}

# --- permutation section constants (former script4ze)
PERM_SEED = 20261023
XCOLS = [f"x{i + 1}" for i in range(len(CONTROLS))]
FORMULA = "y ~ D + " + " + ".join(XCOLS) + " | fips + year"


# =====================================================================================================================
# Section `wild` -- former script4z-inference.py
# =====================================================================================================================
def prepare(df, columns):
    """Complete cases on `columns`, with names cleaned for formulas (as in script4a)."""
    columns = list(dict.fromkeys(columns))
    sub = df[columns].dropna().copy().reset_index(drop=True)
    # wildboottest compiles with numba and fails on an object-dtype cluster column; integer codes give the same clusters
    sub["state_fips"] = pd.factorize(sub["state_fips"])[0]
    return sub.rename(columns={c: _clean(c) for c in columns})


def run_wild(quick):
    SEED, WEIGHTS = WILD_SEED, WILD_WEIGHTS
    REPS = 499 if quick else 9999
    OUT = os.path.join(tables_dir, "script4z_quick" if quick else "script4z")
    os.makedirs(OUT, exist_ok=True)

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
                         "same_sample_across_treatments": same_sample, "reps": REPS, "weights_type": WEIGHTS,
                         "seed": SEED})

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


# =====================================================================================================================
# Section `permutation` -- former script4ze-permutation-inference.py
# =====================================================================================================================
def twfe(d):
    """TWFE coefficient and state-clustered t-statistic on the treatment dummy."""
    t = pf.feols(FORMULA, data=d, vcov={"CRV1": "state_fips"}).tidy().loc["D"]
    return float(t["Estimate"]), float(t["t value"])


def run_permutation(quick, only_design):
    warnings.filterwarnings("ignore")
    ONLY_DESIGN = only_design
    DRAWS = 49 if quick else 2000
    SEED = PERM_SEED
    OUT = os.path.join(tables_dir, "script4ze" + ("_quick" if quick else ""))
    os.makedirs(OUT, exist_ok=True)

    print("=" * 78)
    print(f"script4ze -- permutation inference   draws={DRAWS}  seed={SEED}  scheme: first-treated years shuffled within state")
    print("=" * 78)
    _df, DESIGNS, _poly, _counts = load_designs()
    if ONLY_DESIGN:
        DESIGNS = {k: v for k, v in DESIGNS.items() if k.startswith("first " + ONLY_DESIGN)}
    outcomes = dict(list(OUTCOMES.items())[:1]) if quick else OUTCOMES
    tri = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4za"), "*_overall_estimates.csv"))
    tri = tri[tri.estimator == "TWFE (static, reference)"].set_index(["design", "outcome"])
    wild = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4z"), "*_confirmatory_family.csv"))
    wild = wild[wild.treatment_id == "E2"].set_index("outcome")

    rows, draws_out = [], []
    for di, (dname, D) in enumerate(DESIGNS.items()):
        for oi, (okey, ocol) in enumerate(outcomes.items()):
            d = usable_sample(D["data"], ocol, D["treat"], D["cohort"], CONTROLS)
            obs_b, obs_t = twfe(d)
            ref = tri.loc[(dname, okey)]
            assert np.isclose(obs_b, ref["est"], rtol=TOLERANCE, atol=1e-8), f"observed TWFE differs from script4za: {dname} {okey}"
            county = d.drop_duplicates("fips").set_index("fips")[["cohort", "state_fips"]]
            treated = county[county.cohort.notna()]
            shuffleable = int((treated.groupby("state_fips").cohort.nunique() > 1).sum())
            rng = np.random.default_rng(np.random.SeedSequence([SEED, di, oi]))         # one stream per cell: order-independent
            null_b, null_t = np.empty(DRAWS), np.empty(DRAWS)
            for b in range(DRAWS):
                perm = it.permute_cohorts(county.cohort, county.state_fips, rng)
                dd = d.assign(D=((d.year >= d.fips.map(perm)) & d.fips.map(perm).notna()).astype(float))
                null_b[b], null_t[b] = twfe(dd)
            p_b, p_t = it.permutation_pvalue(obs_b, null_b), it.permutation_pvalue(obs_t, null_t)
            rows.append({"design": dname, "outcome": okey, "observed_coef": obs_b, "observed_t": obs_t, "p_perm_coef": p_b,
                         "p_perm_t": p_t, "draws": DRAWS, "seed": SEED, "treated_counties": len(treated), "counties": len(county),
                         "states_with_shuffle": shuffleable, "null_sd": float(null_b.std()),
                         "p_wild_bootstrap_other_sample": float(wild.loc[okey, "p_wild"]) if dname == "first expansion" and okey in wild.index else np.nan})
            draws_out.append(pd.DataFrame({"design": dname, "outcome": okey, "draw": np.arange(DRAWS), "coef": null_b, "t": null_t}))
            print(f"  {dname:16s} {okey:26s} coef {obs_b:+.4g} (t {obs_t:+.2f})  null sd {null_b.std():.3g}  p_perm {p_b:.4f} (t: {p_t:.4f})  "
                  f"{len(treated)} treated, {shuffleable} shufflable states", flush=True)

    tag = f"_{ONLY_DESIGN}" if ONLY_DESIGN else ""
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{TODAY}_permutation_pvalues{tag}.csv"), index=False)
    pd.concat(draws_out).to_csv(os.path.join(OUT, f"{TODAY}_null_draws{tag}.csv"), index=False)
    print(f"\n{len(rows)} cells written to {OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=SECTIONS, help="run one section (default: both, wild first)")
    parser.add_argument("--quick", action="store_true", help="wild: 499 draws; permutation: 49 draws, first outcome; *_quick folders")
    parser.add_argument("--design", choices=("expansion", "entry"), help="permutation only: one design")
    args = parser.parse_args()
    if args.only in (None, "wild"):
        run_wild(args.quick)
    if args.only in (None, "permutation"):
        run_permutation(args.quick, args.design)


if __name__ == "__main__":
    main()
