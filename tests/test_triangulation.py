"""Tests for the staggered-DiD triangulation estimators in estimators.py (run: python3 tests/test_triangulation.py).

A simulated staggered panel with a known truth (cohorts 5, 9, 13 of 16 years; effects that differ by cohort and grow
with time since treatment; two time-varying controls). Every estimator must recover the event-time effects and its
own overall ATT; static TWFE must be shown to miss (so the test can fail); a second design with a sharp jump pins every
estimator's event-time indexing (effect 0 at e = 0, effect 5 from e = 1). Needs Rscript, R packages polars and
DIDmultiplegtDYN for the dCDH check. Exit code 0 if everything passes.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import estimators as est  # noqa: E402

warnings.filterwarnings("ignore")
T, COHORT_EFFECT, SLOPE = 16, {5: 1.0, 9: 2.0, 13: 3.0}, 0.3
CONTROLS = ["x1", "x2"]
failures = []


def check(label, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + label + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(label)


def simulate(seed=1, n=800, jump=False):
    g = np.random.default_rng(seed)
    coh = g.choice([5, 9, 13, 0], size=n, p=[.25, .25, .2, .3])
    ufe = g.normal(size=n)
    df = pd.DataFrame({"fips": np.repeat(np.arange(n), T), "year": np.tile(np.arange(1, T + 1), n)})
    df["cohort"] = np.where(coh[df.fips] > 0, coh[df.fips], np.nan)
    df["state_fips"] = (np.arange(n) % 20)[df.fips]
    df["x1"] = 0.3 * ufe[df.fips] + g.normal(size=len(df))
    df["x2"] = g.normal(size=len(df))
    e = df.year - df.cohort
    df["D"] = ((e >= 0) & df.cohort.notna()).astype(int)
    if jump:
        tau = np.where(e >= 1, 5.0, 0.0)
    else:
        tau = np.where(e >= 0, df.cohort.map(COHORT_EFFECT) + SLOPE * e, 0.0)
    df["tau"] = np.where(df.cohort.isna(), 0.0, tau)
    df["y"] = ufe[df.fips] + 0.2 * df.year + 0.5 * df.x1 + 0.3 * df.x2 + df.tau * df.D + g.normal(size=len(df))
    return df


def run_all(df):
    kw = dict(y="y", cohort="cohort", unit="fips", time="year", cluster="state_fips", controls=CONTROLS)
    return {"Stacked DiD": est.stacked_did(df, **kw),
            "Imputation (did2s)": est.did2s_estimates(df, treat="D", **kw),
            "Sun-Abraham": est.saturated_estimates(df, **kw),
            "LP-DiD": est.lpdid_estimates(df, **kw),
            "de Chaisemartin-D'Haultfoeuille": est.dcdh_estimates(df, y="y", treat="D", unit="fips", time="year",
                                                                 cluster="state_fips", controls=CONTROLS)}


def at(table, e):
    row = table[(table.kind == "event") & (table.event_time == e)]
    return (float(row.est.iloc[0]), float(row.se.iloc[0])) if len(row) else (np.nan, np.nan)


# ---- design 1: heterogeneous, growing effects
df = simulate()
tr = df[df.D == 1].assign(e=lambda d: d.year - d.cohort)
truth_e = tr.groupby("e").tau.mean()
truth_all, truth_win = tr.tau.mean(), tr[tr.e <= 8].tau.mean()
twfe = float(pf.feols("y ~ D + x1 + x2 | fips + year", data=df, vcov={"CRV1": "state_fips"}).coef()["D"])
check("static TWFE misses the truth (the test can fail)", abs(twfe - truth_all) > 0.3, f"TWFE {twfe:.2f} vs truth {truth_all:.2f}")

results = run_all(df)
for name, tab in results.items():
    ov = tab[tab.kind == "overall"].iloc[0]
    want = truth_win if name in ("LP-DiD", "de Chaisemartin-D'Haultfoeuille") else truth_all    # these two stop at e = 8
    check(f"{name}: overall ATT near the truth", abs(ov.est - want) < 0.35, f"{ov.est:.2f} (se {ov.se:.2f}) vs {want:.2f}")
    bad = []
    for e in (0, 1, 2, 3, 4):
        b, se = at(tab, e)
        if not (abs(b - truth_e[e]) <= max(0.3, 3.5 * se)):
            bad.append(f"e={e}: {b:.2f} vs {truth_e[e]:.2f}")
    check(f"{name}: event-time effects e=0..4 near the truth", not bad, "; ".join(bad))
    pre = [at(tab, e) for e in (-2, -3, -4)]
    check(f"{name}: pre-period placebos near zero", all(abs(b) <= max(0.3, 3.5 * se) for b, se in pre if not np.isnan(b)),
          ", ".join(f"{b:.2f}" for b, _ in pre))

# ---- design 2: a sharp jump pins the event-time indexing of every estimator
jump = simulate(seed=2, jump=True)
for name, tab in run_all(jump).items():
    b0, _ = at(tab, 0)
    b1, _ = at(tab, 1)
    check(f"{name}: event-time indexing (effect 0 at e=0, 5 at e=1)", abs(b0) < 0.5 and abs(b1 - 5) < 0.6, f"e=0 {b0:.2f}, e=1 {b1:.2f}")

print(f"{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
