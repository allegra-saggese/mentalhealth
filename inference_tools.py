"""Shared inference tools: multiple-testing corrections, wild cluster bootstrap, permutation inference.

Single place for inference beyond the state-clustered standard errors that pyfixest reports. Plan and
the family definitions: code-audit/plans/2026-10-22_PLAN_inference-upgrade.md (decision D-42).
Tests: tests/test_inference_tools.py.

    holm(p), benjamini_hochberg(p)    adjusted p-values / q-values, same order as the input
    romano_wolf(t_obs, t_boot)        step-down adjusted p-values from JOINT null-imposed bootstrap t-statistics
    wild_cluster_test(fit, ...)       wild cluster bootstrap-t for one coefficient of a fitted pyfixest model
    permute_cohorts(cohort, group, rng)   reassign first-treated years among ever-treated units within a group
    permutation_pvalue(obs, null)     two-sided permutation p-value with the +1 correction

Run any stochastic function with a seeded numpy Generator (or the seed argument), never an unseeded draw.
"""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

WILD_WEIGHTS = {"rademacher", "mammen", "webb"}   # accepted by wildboottest 0.3.2 and tested here


def holm(pvalues):
    """Holm step-down adjusted p-values (family-wise error control), in the input order."""
    p = np.asarray(pvalues, dtype=float)
    return multipletests(p, method="holm")[1]


def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg adjusted p-values (false discovery rate control), in the input order."""
    p = np.asarray(pvalues, dtype=float)
    return multipletests(p, method="fdr_bh")[1]


def romano_wolf(t_obs, t_boot):
    """Romano-Wolf step-down adjusted p-values.

    t_obs  : (K,) observed t-statistics for the K tests in the family.
    t_boot : (B, K) bootstrap t-statistics under the null for the SAME K tests, where draw b uses the same
             resampling (for a cluster bootstrap: the same cluster weights) for every test. Without joint
             draws the step-down loses its dependence correction and the result is not valid.
    Returns (K,) adjusted p-values in the input order.
    """
    t_abs = np.abs(np.asarray(t_obs, dtype=float))
    boot_abs = np.abs(np.asarray(t_boot, dtype=float))
    if boot_abs.ndim != 2 or boot_abs.shape[1] != t_abs.size:
        raise ValueError("t_boot must have shape (B, K) with K = len(t_obs)")
    order = np.argsort(-t_abs)                         # most significant first
    adj_sorted = np.empty(t_abs.size)
    for step in range(t_abs.size):
        remaining = order[step:]                       # tests not yet rejected
        max_boot = boot_abs[:, remaining].max(axis=1)  # largest bootstrap |t| among them, per draw
        adj_sorted[step] = np.mean(max_boot >= t_abs[order[step]])
    adj_sorted = np.maximum.accumulate(adj_sorted)     # enforce monotone adjusted p-values
    adj = np.empty(t_abs.size)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return adj


def wild_cluster_test(fit, param, reps=9999, seed=None, weights_type="webb", bootstrap_type="11", impose_null=True):
    """Wild cluster bootstrap-t for one coefficient of a fitted pyfixest `feols` model.

    The clustering variable is the one the model was fitted with (state in this project) and must be numeric:
    wildboottest compiles with numba and fails on an object-dtype column (use pd.factorize first). Webb six-point
    weights are the default because state sizes are very uneven; use weights_type="rademacher" as a check.
    bootstrap_type is one of "11", "31", "13", "33" (wildboottest). Returns a dict with the observed t,
    the bootstrap p-value, the bootstrap t draws, and the settings used.
    """
    if weights_type not in WILD_WEIGHTS:
        raise ValueError(f"weights_type must be one of {sorted(WILD_WEIGHTS)} (pyfixest's docs say 'normal', which fails)")
    result, t_draws = fit.wildboottest(
        param=param, reps=reps, seed=seed, weights_type=weights_type, bootstrap_type=bootstrap_type,
        impose_null=impose_null, return_bootstrapped_t_stats=True)
    return {"param": param, "t": float(result["t value"]), "p": float(result["Pr(>|t|)"]), "reps": reps,
            "weights_type": weights_type, "bootstrap_type": bootstrap_type, "impose_null": impose_null,
            "seed": seed, "t_boot": np.asarray(t_draws)}


def permute_cohorts(cohort, group, rng):
    """Reassign first-treated years among the ever-treated units, within each group (for example a state).

    cohort : pandas Series indexed by unit; NaN or inf = never treated (kept as is).
    group  : pandas Series with the same index (the state of each unit).
    Within every group the multiset of cohort values among treated units is preserved and shuffled.
    """
    out = cohort.copy()
    treated = cohort.notna() & np.isfinite(cohort.astype(float))
    for _, idx in cohort[treated].groupby(group[treated]).groups.items():
        idx = list(idx)
        out.loc[idx] = rng.permutation(cohort.loc[idx].to_numpy())
    return out


def permutation_pvalue(observed, null_stats):
    """Two-sided permutation p-value with the +1 correction: (1 + #{|null| >= |obs|}) / (1 + B)."""
    null_abs = np.abs(np.asarray(null_stats, dtype=float))
    return float((1 + np.sum(null_abs >= abs(float(observed)))) / (1 + null_abs.size))


def family_table(labels, pvalues, correction="holm"):
    """Unadjusted and adjusted p-values side by side, for the table a family is reported in."""
    adjust = {"holm": holm, "bh": benjamini_hochberg}[correction]
    p = np.asarray(pvalues, dtype=float)
    return pd.DataFrame({"test": list(labels), "p_unadjusted": p, f"p_{correction}": adjust(p)})
