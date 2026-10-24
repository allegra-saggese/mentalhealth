"""Tests for inference_tools.py (run: python3 tests/test_inference_tools.py).

Known-answer checks for Holm and Benjamini-Hochberg, size and power checks for Romano-Wolf and the wild
cluster bootstrap on simulated data with a known truth, and invariants of the permutation helpers.
Exit code 0 if everything passes. Takes a few minutes.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import inference_tools as it  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20261022
failures = []


def check(label, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + label + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(label)


# 1. known answers (worked by hand: sorted p = .005 .01 .03 .04, m = 4)
p = [0.01, 0.04, 0.03, 0.005]
check("holm known answer", np.allclose(it.holm(p), [0.03, 0.06, 0.06, 0.02]), str(it.holm(p).round(3)))
check("benjamini-hochberg known answer", np.allclose(it.benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02]),
      str(it.benjamini_hochberg(p).round(3)))
check("adjusted never below unadjusted", bool(np.all(it.holm(p) >= np.array(p))))

# 2. Romano-Wolf: family-wise error under the null, power, and the structural properties
rng = np.random.default_rng(SEED)
K, B, R = 5, 1000, 400
cov = 0.5 * np.ones((K, K)) + 0.5 * np.eye(K)
chol = np.linalg.cholesky(cov)
false_rejections = 0
for _ in range(R):
    t_obs = chol @ rng.standard_normal(K)
    t_boot = rng.standard_normal((B, K)) @ chol.T
    false_rejections += it.romano_wolf(t_obs, t_boot).min() < 0.05
fwer = false_rejections / R
check("romano-wolf controls family-wise error", fwer <= 0.09, f"FWER {fwer:.3f} over {R} null families")
t_obs = chol @ rng.standard_normal(K)
t_obs[0] += 6.0
t_boot = rng.standard_normal((B, K)) @ chol.T
adj = it.romano_wolf(t_obs, t_boot)
check("romano-wolf detects a planted effect", adj[0] < 0.05, f"adjusted p {adj[0]:.4f}")
check("romano-wolf keeps the input order and is monotone in |t|",
      bool(np.all(np.diff(adj[np.argsort(-np.abs(t_obs))]) >= -1e-12)))

# 3. wild cluster bootstrap on simulated two-way fixed-effects panels with a known effect
def simulate(effect, seed):
    g = np.random.default_rng(seed)
    n, T, S = 48, 8, 12
    df = pd.DataFrame({"i": np.repeat(np.arange(n), T), "t": np.tile(np.arange(T), n)})
    df["state"] = df.i % S
    first = np.where(df.i % 2 == 0, 3 + (df.i % 3), 99)           # staggered first-treated year
    df["d"] = (df.t >= first).astype(int)
    state_shock = g.normal(size=(S, T))[df.state, df.t]            # shocks shared within a state-year
    df["y"] = effect * df.d + g.normal(size=len(df)) + state_shock
    return df

def reject_rate(effect, sims):
    hits = 0
    for s in range(sims):
        fit = pf.feols("y ~ d | i + t", data=simulate(effect, SEED + s), vcov={"CRV1": "state"})
        hits += it.wild_cluster_test(fit, "d", reps=199, seed=s)["p"] < 0.05
    return hits / sims

size = reject_rate(0.0, 100)
power = reject_rate(2.0, 40)   # at effect 2.0 the analytic cluster-robust test rejects 99 of 100
check("wild cluster bootstrap size near 5% under the null", size <= 0.13, f"rejection rate {size:.2f} over 100 panels")
check("wild cluster bootstrap detects a planted effect", power >= 0.9, f"rejection rate {power:.2f} over 40 panels")
fit = pf.feols("y ~ d | i + t", data=simulate(2.0, SEED), vcov={"CRV1": "state"})
a = it.wild_cluster_test(fit, "d", reps=199, seed=7)
b = it.wild_cluster_test(fit, "d", reps=199, seed=7)
check("wild cluster bootstrap is reproducible with a seed", a["p"] == b["p"] and np.array_equal(a["t_boot"], b["t_boot"]))

try:
    it.wild_cluster_test(fit, "d", reps=99, seed=1, weights_type="normal")
    check("wild_cluster_test rejects the unsupported 'normal' weights", False)
except ValueError:
    check("wild_cluster_test rejects the unsupported 'normal' weights", True)

# 4. permutation helpers
g = np.random.default_rng(SEED)
cohort = pd.Series([2007, 2012, np.nan, 2007, 2017, 2012, np.nan, 2022], index=list("abcdefgh"), dtype=float)
state = pd.Series([1, 1, 1, 1, 2, 2, 2, 2], index=cohort.index)
perm = it.permute_cohorts(cohort, state, g)
check("permute_cohorts keeps never-treated units fixed", perm[cohort.isna()].isna().all())
same = all(sorted(cohort[(state == s) & cohort.notna()]) == sorted(perm[(state == s) & perm.notna()]) for s in (1, 2))
check("permute_cohorts keeps each state's cohort multiset", same)
check("permutation p-value has the +1 correction", it.permutation_pvalue(10.0, np.zeros(99)) == 1 / 100)
check("permutation p-value is two-sided", it.permutation_pvalue(-3.0, np.array([-4.0, 1.0, 2.0, 3.5])) == 3 / 5)
tbl = it.family_table(["a", "b"], [0.01, 0.04])
check("family_table has both columns", list(tbl.columns) == ["test", "p_unadjusted", "p_holm"])

# 5. within-state cohort permutation test (the scheme of script4ze): size under no effect, power under a planted effect
N_STATE, PER_STATE, YEARS = 20, 8, 12


def staggered_panel(effect, seed):
    """Counties nested in states; half are ever treated, cohorts 5, 8 or 11 assigned at random; state-year shocks on top."""
    r = np.random.default_rng(seed)
    n = N_STATE * PER_STATE
    c = r.choice([5, 8, 11, 0], size=n, p=[.2, .15, .15, .5]).astype(float)
    c[c == 0] = np.nan
    p = pd.DataFrame({"fips": np.repeat(np.arange(n), YEARS), "year": np.tile(np.arange(1, YEARS + 1), n)})
    p["state_fips"] = p.fips // PER_STATE
    p["cohort"] = c[p.fips]
    shock = r.normal(scale=0.5, size=(N_STATE, YEARS))[p.state_fips, p.year - 1]
    p["D"] = (p.year >= p.cohort).astype(float)
    p["y"] = r.normal(size=n)[p.fips] + shock + effect * p.D + r.normal(size=len(p))
    return p


def permutation_p(p, draws, seed):
    stat = lambda q: float(pf.feols("y ~ D | fips + year", data=q, vcov={"CRV1": "state_fips"}).coef()["D"])  # noqa: E731
    county = p.drop_duplicates("fips").set_index("fips")[["cohort", "state_fips"]]
    r = np.random.default_rng(seed)
    null = []
    for _ in range(draws):
        perm = it.permute_cohorts(county.cohort, county.state_fips, r)
        null.append(stat(p.assign(D=(p.year >= p.fips.map(perm)).astype(float))))
    return it.permutation_pvalue(stat(p), null), np.array(null)


size_p = [permutation_p(staggered_panel(0.0, 100 + s), 99, 200 + s)[0] for s in range(40)]
size = float(np.mean(np.array(size_p) <= 0.05))
check("permutation test size near 5% under no effect", size <= 0.15, f"rejection rate {size:.2f} over 40 panels, 99 draws")
power_p = [permutation_p(staggered_panel(1.0, 300 + s), 99, 400 + s)[0] for s in range(15)]
power = float(np.mean(np.array(power_p) <= 0.05))
check("permutation test detects a planted effect", power >= 0.9, f"rejection rate {power:.2f} over 15 panels")
a = permutation_p(staggered_panel(0.0, 7), 29, 11)
b = permutation_p(staggered_panel(0.0, 7), 29, 11)
check("permutation test is reproducible with a seed", a[0] == b[0] and np.array_equal(a[1], b[1]))

print(f"{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
