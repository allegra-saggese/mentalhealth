#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4x-species-forest.py -- per-species TWFE and Callaway-Sant'Anna estimates and forest plots.

Species: dairy, beef, hogs, chickens. Scope: large operations, and all sizes. Estimates: T1, T3, T4, E2 (TWFE) and CS-adjusted, for 7 outcomes.
Reads:   panel via script4_treatment.load_panel()
Writes:  Data/output/tables/script4x/<date>_species_results.csv and figs/script4x/<date>_forest_<species>.png
Run:     python3 script4x-species-forest.py     (about 25 minutes; --plot-only redraws the figures; --quick is a test run)
Method:  METHODS.md sections 1, 2 and 5. Dairy / large is checked against script4a. Plot x-axis = coefficient / outcome within-county SD.
"""
import warnings; warnings.filterwarnings("ignore")
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from estimators import run_cs, clean
from designs import species_treatments
from script4_treatment import load_panel, OUTCOMES, COVARIATE_ORDER, figs_dir, tables_dir, os, date

QUICK = "--quick" in sys.argv
PLOT_ONLY = "--plot-only" in sys.argv
XLIM = 1.0          # figures are clipped at +/- this many outcome SDs; arrows mark intervals that run off
N_BOOT = 20 if QUICK else 300
SPECIES = ["dairy"] if QUICK else ["dairy", "beef", "hogs", "chickens"]
SCOPES = {"large": ["large"], "all sizes": ["small", "medium", "large"]}
OUTCOME_LIST = dict(list(OUTCOMES.items())[:2]) if QUICK else OUTCOMES
CONTROLS = COVARIATE_ORDER[:8]
OUT_T = os.path.join(tables_dir, "script4x" + ("_quick" if QUICK else "")); os.makedirs(OUT_T, exist_ok=True)
OUT_F = os.path.join(figs_dir, "script4x" + ("_quick" if QUICK else "")); os.makedirs(OUT_F, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
ESTS = [("T1", "T1  presence (0/1)"), ("T3", "T3  per 10,000 residents"), ("T4", "T4  log(1 + count)"),
        ("E2", "E2  first expansion (0/1)"), ("CS", "CS  Callaway-Sant'Anna")]


def one_task(args):
    species, scope = args
    df = load_panel().sort_values(["fips", "year"]).reset_index(drop=True)
    tr = species_treatments(df, species, SCOPES[scope])
    data = pd.concat([df[["fips", "year", "state_fips", "population", "log_pop"] + CONTROLS + list(OUTCOME_LIST.values())], tr], axis=1)
    check = None
    if species == "dairy" and scope == "large":                                # validation against script4_treatment
        pairs = {"T1": "tr_e1_lg_bin", "T3": "tr_i4_lg_p10k", "T4": "tr_c3_log_lg", "E2": "tr_e2_add_absorb"}
        check = {k: float(np.nanmax(np.abs(tr[k].to_numpy(float) - df[v].to_numpy(float)))) for k, v in pairs.items()}
        check["add_year_mismatch"] = int((tr["add_year"].fillna(-1) != df["add_year"].fillna(-1)).sum())
    rows = []
    n_treated = int(data.loc[data["add_year"].notna(), "fips"].nunique())
    for okey, ocol in OUTCOME_LIST.items():
        for est in ("T1", "T3", "T4", "E2"):
            extra = {"T4": ["log_pop"]}.get(est, [])
            need = list(dict.fromkeys(["fips", "year", "state_fips", ocol, est] + extra + CONTROLS))
            sub = data[need].dropna().rename(columns={c: clean(c) for c in need})
            if len(sub) < 500 or sub[clean(est)].nunique() < 2:
                continue
            nsw = int((sub.groupby("fips")[clean(est)].nunique() > 1).sum())
            f = f"{clean(ocol)} ~ " + " + ".join([clean(est)] + [clean(x) for x in extra] + [clean(c) for c in CONTROLS]) + " | fips + year"
            try:
                m = pf.feols(f, data=sub, vcov={"CRV1": "state_fips"}); t = m.tidy().loc[clean(est)]
                rows.append({"species": species, "scope": scope, "estimator": est, "outcome": okey,
                             "estimate": float(t["Estimate"]), "se": float(t["Std. Error"]), "p": float(t["Pr(>|t|)"]),
                             "N": int(m._N), "n_switchers": nsw, "n_treated_counties": n_treated})
            except Exception:
                pass
        gt, ag = run_cs(data, ocol, "add_year", CONTROLS, n_boot=N_BOOT)
        if len(ag):
            o = ag[ag.agg_level == "overall_ATT"].iloc[0]
            rows.append({"species": species, "scope": scope, "estimator": "CS", "outcome": okey,
                         "estimate": float(o.att), "se": float(o.se_state), "p": float(o.p_state),
                         "N": np.nan, "n_switchers": np.nan, "n_treated_counties": int(o.n_treated),
                         "n_cohorts": int(o.n_cohorts), "n_fallback_cells": int(o.n_fallback_cells)})
    return species, scope, rows, check


def plot_species(res, sd, species):
    names = list(OUTCOME_LIST)
    colors = {"T1": "#2a78d6", "T3": "#eb6834", "T4": "#1baf7a", "E2": "#eda100", "CS": "#e87ba4"}
    marks = {"T1": "o", "T3": "s", "T4": "^", "E2": "D", "CS": "*"}
    gap = 0.9
    nrow = len(names) * (len(ESTS) + gap)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.30 * nrow + 2.4), sharey=True, sharex=True, facecolor="#fcfcfb")
    fig.suptitle(f"{species.capitalize()} CAFOs: estimates for all outcomes", x=0.01, ha="left", fontsize=13,
                 fontweight="bold", color="#0b0b0b", y=0.995)
    fig.text(0.01, 0.962, "x = coefficient / the outcome's within-county SD (one common scale). Treatment units differ by row. "
             "Bars = 95% intervals (state-clustered; CS: state bootstrap); arrows = interval runs past +/-1 SD.", fontsize=8.5, color="#52514e")
    for ax, scope in zip(axes, SCOPES):
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#52514e"); ax.tick_params(colors="#52514e", labelsize=8, length=3)
        ax.axvline(0, color="#52514e", linewidth=0.9)
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.7); ax.set_axisbelow(True)
        for gi, o in enumerate(names):
            y0 = gi * (len(ESTS) + gap)
            for k, (e, lab) in enumerate(ESTS):
                r = res[(res.species == species) & (res.scope == scope) & (res.outcome == o) & (res.estimator == e)]
                y = y0 + k
                if r.empty:
                    continue
                r = r.iloc[0]; b, s_ = r.estimate / sd[o], r.se / sd[o]
                lo, hi = b - 1.96 * s_, b + 1.96 * s_
                ax.plot([max(lo, -XLIM), min(hi, XLIM)], [y, y], color=colors[e], linewidth=1.4, solid_capstyle="butt")
                if abs(b) <= XLIM:
                    ax.plot(b, y, marker=marks[e], color=colors[e], markersize=8 if e == "CS" else 5.5,
                            markeredgecolor="#fcfcfb", markeredgewidth=1.0, linestyle="none")
                if lo < -XLIM:
                    ax.plot(-XLIM, y, marker="<", color=colors[e], markersize=5, linestyle="none")
                if hi > XLIM:
                    ax.plot(XLIM, y, marker=">", color=colors[e], markersize=5, linestyle="none")
            if scope == "large":
                ax.text(-0.02, y0 - 0.95, o, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9,
                        fontweight="bold", color="#0b0b0b")
        ax.set_xlim(-XLIM * 1.04, XLIM * 1.04)
        ax.set_title(f"{scope}", fontsize=10, loc="left", color="#0b0b0b")
        ax.set_xlabel("effect in within-county SDs of the outcome", fontsize=8.5, color="#52514e")
    ytl = [y0 + k for y0 in [g * (len(ESTS) + gap) for g in range(len(names))] for k in range(len(ESTS))]
    axes[0].set_yticks(ytl); axes[0].set_yticklabels([lab for _ in names for _, lab in ESTS], fontsize=8, color="#0b0b0b")
    axes[0].set_ylim(nrow - gap, -1.8)
    handles = [plt.Line2D([0], [0], marker=marks[e], color=colors[e], linestyle="none", markersize=7) for e, _ in ESTS]
    fig.legend(handles, [lab for _, lab in ESTS], loc="lower center", ncol=5, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955))
    fig.savefig(os.path.join(OUT_F, f"{TODAY}_forest_{species}.png"), dpi=140, facecolor="#fcfcfb"); plt.close(fig)


if __name__ == "__main__" and PLOT_ONLY:
    import glob
    res = pd.read_csv(sorted(glob.glob(os.path.join(OUT_T, "*_species_results.csv")))[-1])
    sdf = pd.read_csv(sorted(glob.glob(os.path.join(OUT_T, "*_outcome_within_sd.csv")))[-1], index_col=0).iloc[:, 0]
    for sp in SPECIES:
        plot_species(res, sdf.to_dict(), sp)
    print("redrew figures for", SPECIES, "->", OUT_F)
    sys.exit(0)

if __name__ == "__main__":
    tasks = [(s, sc) for s in SPECIES for sc in SCOPES]
    allrows, checks = [], {}
    with ProcessPoolExecutor(max_workers=min(len(tasks), 8)) as ex:
        for species, scope, rows, check in ex.map(one_task, tasks):
            allrows += rows
            if check:
                checks = check
            print(f"done: {species} / {scope}  ({len(rows)} estimates)", flush=True)
    res = pd.DataFrame(allrows)
    res.to_csv(os.path.join(OUT_T, f"{TODAY}_species_results.csv"), index=False)

    # within-county SD of each outcome (for the common x-axis)
    df = load_panel()
    sd = {o: float((df[c] - df.groupby("fips")[c].transform("mean")).std()) for o, c in OUTCOME_LIST.items()}
    pd.Series(sd).rename("within_county_sd").to_csv(os.path.join(OUT_T, f"{TODAY}_outcome_within_sd.csv"))
    for sp in SPECIES:
        plot_species(res, sd, sp)

    # ---------------- validation: dairy / large against script4_treatment and script4a ----------------
    print("\nVALIDATION  dairy / large")
    print("  treatment variables vs script4_treatment (max abs diff):", {k: float(f"{v:.2e}") if isinstance(v, float) else v for k, v in checks.items()})
    import glob
    g = sorted(glob.glob(os.path.join(tables_dir, "script4a", "*_A1_A2_treatment_grid.csv")))
    cs = sorted(glob.glob(os.path.join(tables_dir, "script4a", "*_A6_cs_aggregated.csv")))
    if g:
        grid = pd.read_csv(g[-1]); grid = grid[grid.registry == "A2"]
        for est, tid in [("T1", "T1"), ("T3", "T3"), ("T4", "T4"), ("E2", "E2")]:
            a = res[(res.species == "dairy") & (res.scope == "large") & (res.estimator == est)].set_index("outcome")["estimate"]
            b = grid[grid.treatment_id == tid].set_index("outcome")["beta"]
            idx = a.index.intersection(b.index)
            print(f"  TWFE {est} vs script4a: max abs diff {float((a[idx] - b[idx]).abs().max()):.2e} over {len(idx)} outcomes")
    if cs:
        c = pd.read_csv(cs[-1]); c = c[(c.arm == "covariate-adjusted") & (c.agg_level == "overall_ATT")].set_index("outcome")
        a = res[(res.species == "dairy") & (res.scope == "large") & (res.estimator == "CS")].set_index("outcome")
        idx = a.index.intersection(c.index)
        print(f"  CS overall ATT vs script4a: max abs diff {float((a.loc[idx, 'estimate'] - c.loc[idx, 'att']).abs().max()):.2e}; "
              f"SE diff {float((a.loc[idx, 'se'] - c.loc[idx, 'se_state']).abs().max()):.2e} (SEs equal only with 300 draws)")
    print(f"\nSaved -> {OUT_T}\n         {OUT_F}")
