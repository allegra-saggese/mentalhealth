"""script4ze-permutation-inference.py -- permutation test of the TWFE coefficient for the cohort designs (D-47).

Question: if first-treated years were assigned at random WITHIN each state (never-treated counties fixed), how often would the
TWFE coefficient be as large as the observed one? Tests the sharp null of no effect in any county; needs no sampling model.
Designs: first expansion (E2, primary) and first entry (E3, secondary); all 7 outcomes; samples as in script4za
(designs.usable_sample: complete cases, treated counties need a pre-period, the 8 pre-committed controls).
Statistic: TWFE coefficient on the treatment dummy (county + year fixed effects, controls, state-clustered SE). The primary p-value is
for the coefficient, the secondary for the clustered t-statistic: p = (1 + #{|null| >= |observed|}) / (1 + draws).
Reads:   panel via designs.load_designs(); script4z confirmatory family (wild-bootstrap p, a different sample, shown for reference);
         script4za overall estimates (the observed coefficient is checked against its TWFE reference row to 1e-6).
Writes:  Data/output/tables/script4ze/YYYY-MM-DD_{permutation_pvalues, null_draws}.csv     (--quick: script4ze_quick)
Run:     PYTHONHASHSEED=0 python3 script4ze-permutation-inference.py [--design expansion|entry] [--quick]
Runtime: about 0.12 s per draw; 2,000 draws x 7 outcomes x 2 designs is roughly an hour (run the two designs as two processes).
"""
import sys
import warnings

import numpy as np
import pandas as pd
import pyfixest as pf

import inference_tools as it
from designs import load_designs, usable_sample
from functions import latest_file_glob
from script4_treatment import OUTCOMES, COVARIATE_ORDER, tables_dir, os, date

warnings.filterwarnings("ignore")
ARGV = sys.argv[1:]
QUICK = "--quick" in ARGV
ONLY_DESIGN = next((ARGV[i + 1] for i, a in enumerate(ARGV) if a == "--design"), None)
DRAWS = 49 if QUICK else 2000
SEED = 20261023
CONTROLS = COVARIATE_ORDER[:8]                          # pre-committed (D-16, D-31)
XCOLS = [f"x{i + 1}" for i in range(len(CONTROLS))]
FORMULA = "y ~ D + " + " + ".join(XCOLS) + " | fips + year"
TOLERANCE = 1e-6
OUT = os.path.join(tables_dir, "script4ze" + ("_quick" if QUICK else ""))
os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")


def twfe(d):
    """TWFE coefficient and state-clustered t-statistic on the treatment dummy."""
    t = pf.feols(FORMULA, data=d, vcov={"CRV1": "state_fips"}).tidy().loc["D"]
    return float(t["Estimate"]), float(t["t value"])


print("=" * 78)
print(f"script4ze -- permutation inference   draws={DRAWS}  seed={SEED}  scheme: first-treated years shuffled within state")
print("=" * 78)
_df, DESIGNS, _poly, _counts = load_designs()
if ONLY_DESIGN:
    DESIGNS = {k: v for k, v in DESIGNS.items() if k.startswith("first " + ONLY_DESIGN)}
outcomes = dict(list(OUTCOMES.items())[:1]) if QUICK else OUTCOMES
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
