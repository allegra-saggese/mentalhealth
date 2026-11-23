#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4h-gp.py -- how much spatial dependence is left in the A2 residuals, measured with the NN-RCM Gaussian-process model (D-61; plan 2026-11-22).

NN-RCM (Grenier, github.com/igrenier/NNRCM) fits one spatial cross-section: a Gaussian process with a Matern covariance, conditioned on the 10 nearest
neighbours. This script does NOT change any estimate. It takes the residuals of the reported A2 regression (county + year fixed effects, the control set,
first expansion E2 of large dairies, as in the Moran's I section of script4g) and asks, year by year: how far does the remaining dependence reach, how large
is it relative to noise, and does it differ by region? The answer is read against the Conley cut-offs (100/250/500 km) and the 100 km weights of D-58.

    export    write the county geometry (Albers km) and the residuals to Data/clean/gp/ for the R step
    summary   read the R fits (R/nnrcm_fit.R) and the simulation check (R/nnrcm_sim_check.R), convert the fitted decay to a range in km, print and save tables

Run order:
    python3 script4h-gp.py --only export
    Rscript R/nnrcm_sim_check.R            # recovers a known range on the county geometry (calibrates what the fitted number means)
    Rscript R/nnrcm_fit.R                  # fits every outcome x series x year (and division) -> Data/output/tables/<date>_4h-gp_fits.csv
    python3 script4h-gp.py --only summary
R needs the NNRCM package built from a scratch copy (no Imports beyond spNNGP, RcppParallel, fields; see the plan). Set NNRCM_LIB to its library folder.

Reading the numbers (Grenier 2021, Ch. 3): covariance sigma^2 (Matern(d; nu) + xi^2 I); phi = nu is a length scale, exp(-d/nu) at smoothness 0.5, so `range_km` is the
e-folding distance (3x it is the practical range); xi is the nugget as a multiple of sigma^2, spatial share of variance = 1/(1+xi).
Limits: Alaska and Hawaii are left out (the Albers projection is for the contiguous states). County residuals are areal and the CDC series are model
estimates (OUTCOME-SMOOTHING note), so a long fitted range partly reflects the smoothing in the data, not only real spatial dependence.
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import importlib.util
import os
from datetime import date

import numpy as np
import pandas as pd

from designs import load_designs, cafo_designs
from packages import db_data, out_file, out_glob, tables_dir
from script4_treatment import OUTCOMES

TODAY = date.today().strftime("%Y-%m-%d")
GP_DIR = os.path.join(db_data, "clean", "gp")
R_EARTH = 6371.0088
UNIT_KM = 100.0                                   # coordinates are written in units of 100 km so the optimiser's starting values (0.5) are on the right scale


def _load_g():
    """script4g-spillovers.py as a module (its file name has a hyphen), so the A2 sample and the TWFE fit are the same code as the Moran section."""
    spec = importlib.util.spec_from_file_location("script4g", os.path.join(os.path.dirname(os.path.abspath(__file__)), "script4g-spillovers.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def albers_km(lat, lon, lat1=29.5, lat2=45.5, lat0=37.5, lon0=-96.0):
    """Albers equal-area conic (spherical) for the contiguous U.S.: planar x, y in km, so Euclidean distance (what NNRCM uses) is a distance on the map."""
    p1, p2, p0, la, lo = map(np.radians, (lat1, lat2, lat0, np.asarray(lat, float), np.asarray(lon, float)))
    lo0 = np.radians(lon0)
    n = (np.sin(p1) + np.sin(p2)) / 2
    C = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho0 = np.sqrt(C - 2 * n * np.sin(p0)) / n
    rho = np.sqrt(C - 2 * n * np.sin(la)) / n
    th = n * (lo - lo0)
    return R_EARTH * rho * np.sin(th), R_EARTH * (rho0 - rho * np.cos(th))


def do_export():
    g = _load_g()
    df, _designs, _p, _c = load_designs()
    df, _ = cafo_designs(df, "dairy", "large")                    # the rebuilt columns equal the panel's (checked in designs.py)
    geo, _D = g.load_geometry(df)
    geo = geo[~geo.fips.str[:2].isin(["02", "15"])].reset_index(drop=True)
    x, y = albers_km(geo.lat.values, geo.lon.values)
    geo["x"], geo["y"] = x / UNIT_KM, y / UNIT_KM
    os.makedirs(GP_DIR, exist_ok=True)

    # check the projection against great-circle distance on a sample of pairs (Albers distortion should be a few per cent)
    rng = np.random.default_rng(20261122)
    i, j = rng.integers(0, len(geo), 2000), rng.integers(0, len(geo), 2000)
    ok = i != j
    gc = g.sp.haversine_km(geo.lat.values, geo.lon.values)[i[ok], j[ok]]
    pl = np.hypot(x[i[ok]] - x[j[ok]], y[i[ok]] - y[j[ok]])
    ratio = pl / gc
    print(f"projection check: planar / great-circle distance, 2,000 random pairs: median {np.median(ratio):.3f}, 1st-99th pct {np.percentile(ratio, 1):.3f}-{np.percentile(ratio, 99):.3f}")
    assert 0.95 < np.percentile(ratio, 1) and np.percentile(ratio, 99) < 1.05, "Albers distortion above 5% on the sample"
    geo.to_csv(os.path.join(GP_DIR, "gp_geometry.csv"), index=False)

    rows = []
    for okey, ocol in OUTCOMES.items():
        tcol, extra = g.TREAT_COL["E2"], []
        sub, mp = g.a2_sample(df, geo[["fips", "lat", "lon"]], ocol, tcol, extra)
        m = g.fit_twfe(sub, g.formula(mp, ocol, tcol, extra), keep_singletons=True)
        assert m._N == len(sub), f"{okey}: row mismatch between fit and sample"
        sub["resid"] = np.asarray(m.resid()).ravel()
        sub = sub.merge(geo[["fips", "x", "y"]], on="fips")
        sub["raw"] = sub[mp[ocol]]
        sub["outcome"] = okey
        rows.append(sub[["outcome", "fips", "year", "state_fips", "division", "x", "y", "resid", "raw"]])
        print(f"  {okey}: {len(sub):,} county-years, {sub.year.nunique()} years, {sub.fips.nunique():,} counties")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(GP_DIR, "gp_input.csv"), index=False)
    print(f"wrote {GP_DIR}/gp_geometry.csv ({len(geo):,} counties) and gp_input.csv ({len(out):,} rows)")


def latest(tag, name):
    f = out_glob(tables_dir, tag, f"*_{name}")
    if not f:
        raise FileNotFoundError(f"no {tag} {name} table yet: run the R step first")
    return f[-1]


def do_summary():
    fits = pd.concat([pd.read_csv(f) for f in out_glob(tables_dir, "4h-gp", "*_fits__*.csv")], ignore_index=True)
    print(f"{len(fits):,} fits from {fits.outcome.nunique()} outcomes; {int((fits.convergence != 0).sum())} not converged or failed")
    sim = pd.read_csv(latest("4h-gp", "sim_check.csv"))
    print("\nsimulation check (known exponential range on the county geometry, nugget sd 0.5):")
    print(sim.to_string(index=False))
    # calibration: the fitted decay phi (100-km units) rises with the true range but flattens beyond about 250 km, so a single factor would mislead.
    # Fitted phi is mapped to km by interpolating the simulated curve (median fitted phi at each true range); outside the simulated span it is flagged.
    cal = sim.groupby("true_range_km").agg(phi_hat=("phi", "median"), sigma_hat=("sigma", "median"), nug_hat=("xi", "median")).reset_index()
    print("\ncalibration (median of the simulated fits at each true range):")
    print(cal.to_string(index=False))
    assert cal.phi_hat.is_monotonic_increasing, "fitted phi is not monotone in the true range: the calibration cannot be inverted"
    lo, hi = cal.phi_hat.min(), cal.phi_hat.max()
    fits["range_km"] = np.interp(fits.phi, cal.phi_hat, cal.true_range_km)
    fits["range_flag"] = np.where(fits.phi < lo, "below simulated span", np.where(fits.phi > hi, "above simulated span", ""))
    fits.loc[fits.kappa != 0.5, ["range_km", "range_flag"]] = [np.nan, "not calibrated (smoothness 1.5)"]   # the simulation used smoothness 0.5 only
    fits["range_km_ok"] = fits.convergence == 0
    fits.to_csv(out_file(tables_dir, "4h-gp", f"{TODAY}_fits_with_range.csv"), index=False)
    ok = fits[fits.range_km_ok]
    summ = (ok.groupby(["outcome", "series", "region", "kappa"]).agg(
        years=("year", "size"), range_km_median=("range_km", "median"), range_km_min=("range_km", "min"), range_km_max=("range_km", "max"),
        sigma_median=("sigma", "median"), nugget_median=("xi", "median")).reset_index())
    summ.to_csv(out_file(tables_dir, "4h-gp", f"{TODAY}_range_summary.csv"), index=False)
    print(summ[summ.region == "US"].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", choices=["export", "summary"], required=True)
    args = ap.parse_args()
    {"export": do_export, "summary": do_summary}[args.only]()


if __name__ == "__main__":
    main()
