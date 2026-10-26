#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4a-estimates.py -- every design-based estimate of the reported set, in four sections (Stage D of the script consolidation).

  designs   (was script4a-design-estimates.py)  TWFE grid A1/A2, dairy x FSIS A4, event study A5, Callaway-Sant'Anna A6, LP-DiD A7, VIF audit
  cohorts   (was script4u-did-by-treatment.py)  first expansion and first entry: TWFE, event study, Callaway-Sant'Anna, pre-trend adjustments
  species   (was script4x-species-forest.py)    per-species TWFE and Callaway-Sant'Anna estimates and forest plots (about 25 minutes)
  fsis      (was script4r-fsis-treatment.py)    slaughterhouses (FSIS plants) as the treatment

Each section's code is the old script's code, moved unchanged. Outputs go to the flat Data/output/tables/ and figures/ folders,
named <date>_<tag>_<stem> with tag 4a-designs, 4a-cohorts, 4a-species, 4a-fsis ("-quick" in --quick runs; naming: packages.py).
Run:     PYTHONHASHSEED=0 python3 script4a-estimates.py [--only designs cohorts species fsis]
         flags passed through to the sections: --quick (cohorts, species), --no-cs (cohorts), --plot-only (species), --legacy-grid (designs)
Order:   designs first (cohorts and species check themselves against its output); the default run does all four in that order.
Reads:   panel via script4_treatment.load_panel() / designs.load_designs(); Data/merged/*_panel_fsis.csv (designs A4, fsis)
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
from scipy.stats import norm as _norm

from estimators import run_cs, clean
from designs import species_treatments
from script4_treatment import load_panel, OUTCOMES, COVARIATE_ORDER, figs_dir, tables_dir, os, date
from packages import out_file, out_glob, out_latest, tagged

# ---------------------------------------------------------------------------------------------------------------------
# section `species` (was script4x-species-forest.py): constants and process-pool workers stay at module level, because
# worker processes re-import this module and `one_task` must be importable from it
# ---------------------------------------------------------------------------------------------------------------------
QUICK = "--quick" in sys.argv
PLOT_ONLY = "--plot-only" in sys.argv
XLIM = 1.0          # figures are clipped at +/- this many outcome SDs; arrows mark intervals that run off
N_BOOT = 20 if QUICK else 300
SPECIES = ["dairy"] if QUICK else ["dairy", "beef", "hogs", "chickens"]
SCOPES = {"large": ["large"], "all sizes": ["small", "medium", "large"]}
OUTCOME_LIST = dict(list(OUTCOMES.items())[:2]) if QUICK else OUTCOMES
CONTROLS = COVARIATE_ORDER[:8]
TAG_SP = "4a-species" + ("-quick" if QUICK else "")    # output-name tag (packages.py); was folder script4x
OUT_T, OUT_F = tables_dir, figs_dir
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
    fig.savefig(out_file(figs_dir, TAG_SP, f"{TODAY}_forest_{species}.png"), dpi=140, facecolor="#fcfcfb"); plt.close(fig)


def section_species():
    """Per-species TWFE and Callaway-Sant'Anna estimates and forest plots.

    Old script script4x-species-forest.py (docstring below), code moved unchanged; the worker functions are above.

    script4x-species-forest.py -- per-species TWFE and Callaway-Sant'Anna estimates and forest plots.

    Species: dairy, beef, hogs, chickens. Scope: large operations, and all sizes. Estimates: T1, T3, T4, E2 (TWFE) and CS-adjusted, for 7 outcomes.
    Reads:   panel via script4_treatment.load_panel()
    Writes:  Data/output/tables/<date>_4a-species_species_results.csv and figures/<date>_4a-species_forest_<species>.png
    Run:     python3 script4x-species-forest.py     (about 25 minutes; --plot-only redraws the figures; --quick is a test run)
    Method:  METHODS.md sections 1, 2 and 5. Dairy / large is checked against script4a. Plot x-axis = coefficient / outcome within-county SD.
    """
    if PLOT_ONLY:
        import glob
        res = pd.read_csv(out_glob(tables_dir, TAG_SP, "*_species_results.csv")[-1])
        sdf = pd.read_csv(out_glob(tables_dir, TAG_SP, "*_outcome_within_sd.csv")[-1], index_col=0).iloc[:, 0]
        for sp in SPECIES:
            plot_species(res, sdf.to_dict(), sp)
        print("redrew figures for", SPECIES, "->", OUT_F)
        return

    if True:
        tasks = [(s, sc) for s in SPECIES for sc in SCOPES]
        allrows, checks = [], {}
        with ProcessPoolExecutor(max_workers=min(len(tasks), 8)) as ex:
            for species, scope, rows, check in ex.map(one_task, tasks):
                allrows += rows
                if check:
                    checks = check
                print(f"done: {species} / {scope}  ({len(rows)} estimates)", flush=True)
        res = pd.DataFrame(allrows)
        res.to_csv(out_file(tables_dir, TAG_SP, f"{TODAY}_species_results.csv"), index=False)

        # within-county SD of each outcome (for the common x-axis)
        df = load_panel()
        sd = {o: float((df[c] - df.groupby("fips")[c].transform("mean")).std()) for o, c in OUTCOME_LIST.items()}
        pd.Series(sd).rename("within_county_sd").to_csv(out_file(tables_dir, TAG_SP, f"{TODAY}_outcome_within_sd.csv"))
        for sp in SPECIES:
            plot_species(res, sd, sp)

        # ---------------- validation: dairy / large against script4_treatment and script4a ----------------
        print("\nVALIDATION  dairy / large")
        print("  treatment variables vs script4_treatment (max abs diff):", {k: float(f"{v:.2e}") if isinstance(v, float) else v for k, v in checks.items()})
        import glob
        g = out_glob(tables_dir, "4a-designs", "*_A1_A2_treatment_grid.csv")
        cs = out_glob(tables_dir, "4a-designs", "*_A6_cs_aggregated.csv")
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




def section_designs():
    """TWFE grid, dairy x FSIS, event study, Callaway-Sant'Anna.

    Old script script4a-design-estimates.py (docstring below), code moved unchanged.

    script4a-design-estimates.py -- main design-based estimates: TWFE (A2), dairy x FSIS (A4), event study (A5), Callaway-Sant'Anna (A6).

    Reads:   panel via script4_treatment.load_panel(); Data/merged/*_panel_fsis.csv (A4 only)
    Writes:  Data/output/tables/<date>_4a-designs_{A1_A2_treatment_grid, A4_fsis_interaction, A5_event_study, A6_cs_att_gt,
             A6_cs_aggregated, HEADLINE_twfe_vs_cs, VIF_audit}.csv and figures/<date>_4a-designs_*.png
    Run:     python3 script4a-design-estimates.py     (about 7 minutes; 300 bootstrap draws)
    Method:  METHODS.md sections 1-2; state-clustered SEs; CS functions come from estimators.py.
    Notes:   A1 (pooled state FE) and A7 (LP-DiD) are computed but not reported (D-32).
    """
    import warnings; warnings.filterwarnings("ignore")
    import sys
    import numpy as np
    import pandas as pd
    import pyfixest as pf
    import matplotlib.pyplot as plt
    from scipy.stats import norm as _norm

    from script4_treatment import (
        load_panel, OUTCOMES, TREATMENTS, CORE_TREATMENTS, CORE_TREATMENT_LABELS,
        SPEC_EXTRA_REGRESSORS, COVARIATE_ORDER, REPORT_TREATMENTS,
        db_data, figs_dir, tables_dir, os, date,
    )
    from functions import latest_file_glob
    from estimators import lpdid, conley_se, load_centroids, att_gt_table, base_period_covariates, clean as _clean

    OUT_FIGS, OUT_TABLES = figs_dir, tables_dir    # output-name tag "4a-designs" (packages.py); was folder script4a
    TODAY = date.today().strftime("%Y-%m-%d")

    # D-16 / D-29: the step-8 set is the ONLY control set run. CONTROL_PRETREAT was
    # dropped as a robustness arm on 2026-09-29 -- not on coverage grounds (on the
    # headline sample it costs only 15% of rows) but because each of its ten extra
    # members is excluded FOR CAUSE: two colliders, one measure break, one unusable
    # series, one with no variation, and five that cannot be tested for pre-trends.
    # Reporting a set that contains known colliders as a "robustness check" would
    # imply the exclusions were arbitrary. They are not. See D-29.
    CONTROLS_HEADLINE = COVARIATE_ORDER[:8]
    CONTROL_SETS = {"step8 (pre-committed, D-16)": CONTROLS_HEADLINE}

    USE_LEGACY_GRID = "--legacy-grid" in sys.argv
    GRID_TREATMENTS = (TREATMENTS if USE_LEGACY_GRID
                       else {k: (v, SPEC_EXTRA_REGRESSORS.get(k, []), CORE_TREATMENT_LABELS[k])
                             for k, v in CORE_TREATMENTS.items()})
    # E2 (tr_e2_add_absorb) is NOT a core treatment, but the A2-vs-A6 comparison
    # below requires it: it is the binary absorbing treatment that CS's cohort
    # definition implies. Added to the grid so the two estimators can be compared on
    # the SAME treatment. It is reported only in that comparison table.
    if not USE_LEGACY_GRID:
        GRID_TREATMENTS["E2"] = ("tr_e2_add_absorb", [],
                                 "First positive change in large dairy ops (absorbing)")

    N_BOOT = 300                              # cluster-bootstrap reps for A6
    RNG    = np.random.default_rng(20260928)


    # =============================================================================
    # Helpers
    # =============================================================================
    def prep(df, cols):
        """Subset to `cols`, drop incomplete rows, return (frame, {orig: clean})."""
        cols = list(dict.fromkeys(cols))
        sub = df[cols].dropna().copy()
        return sub.rename(columns={c: _clean(c) for c in cols}), {c: _clean(c) for c in cols}


    def variation(sub, tcol):
        """Counties whose treatment changes, and how many changes. This is what
    actually identifies a within-county coefficient."""
        n_units = sub["fips"].nunique()
        nun = sub.groupby("fips")[tcol].nunique()
        sw = nun[nun > 1].index
        if len(sw) == 0:
            return dict(n_counties=n_units, n_states=sub["state_fips"].nunique(),
                        n_switchers=0, n_transitions=0)
        e = sub[sub["fips"].isin(sw)].sort_values(["fips", "year"]).copy()
        e["_d"] = e.groupby("fips")[tcol].diff()
        return dict(n_counties=n_units, n_states=sub["state_fips"].nunique(),
                    n_switchers=int(len(sw)), n_transitions=int((e["_d"].abs() > 0).sum()))


    def rhs_vif(sub, rhs_cols, within=True):
        """VIF over the FULL right-hand side of THIS spec, not just the control block.

    Computed on WITHIN-TRANSFORMED data when `within`, because that is the
    variation a county+year FE regression actually uses. Raw-level VIF is the
    wrong diagnostic: collinearity among these controls is almost entirely
    cross-sectional and county FE removes it (children_in_poverty 9.10 raw ->
    1.13 within).

    n_zero_var counts regressors with NO within-county variation -- absorbed by
    the FE, contributing nothing. This is how the old C4 spec was caught
    conditioning on a county-constant variable.
    """
        W = sub[rhs_cols].astype(float).copy()
        if within:
            for c in rhs_cols:
                W[c] = (W[c] - sub.groupby("fips")[c].transform("mean")
                            - sub.groupby("year")[c].transform("mean") + sub[c].mean())
        zero_var = [c for c in rhs_cols if W[c].std() <= 1e-10]
        keep = [c for c in rhs_cols if c not in zero_var]
        if len(keep) < 2:
            return {"vif_max": np.nan, "vif_max_var": "", "vif_treat": np.nan,
                    "n_zero_var": len(zero_var)}
        try:
            R = np.corrcoef(W[keep].values, rowvar=False)
            v = pd.Series(np.diag(np.linalg.pinv(R)), index=keep)
        except Exception:
            return {"vif_max": np.nan, "vif_max_var": "", "vif_treat": np.nan,
                    "n_zero_var": len(zero_var)}
        return {"vif_max": float(v.max()), "vif_max_var": str(v.idxmax()),
                "vif_treat": float(v.get(rhs_cols[0], np.nan)), "n_zero_var": len(zero_var)}


    def fit(sub, y, tcol, extra, controls_clean, fe, label):
        """One regression, three variance estimators. Returns a flat dict or None."""
        fml = f"{y} ~ " + " + ".join([tcol] + list(extra) + controls_clean) + f" | {fe}"
        if len(sub) < 200:
            return None
        out = {}
        try:
            for key, vc in [("state", {"CRV1": "state_fips"}),
                            ("county", {"CRV1": "fips"}),
                            ("hetero", "hetero")]:
                m = pf.feols(fml, data=sub, vcov=vc)
                t = m.tidy()
                if tcol not in t.index:
                    return None
                r = t.loc[tcol]
                out[f"se_{key}"] = float(r["Std. Error"])
                out[f"p_{key}"]  = float(r["Pr(>|t|)"])
                if key == "state":
                    out["beta"] = float(r["Estimate"])
                    out["N"] = int(m._N)
                    out["r2_within"] = float(getattr(m, "_r2_within", np.nan))
                    for x in extra:
                        if x in t.index:
                            out[f"cond_{x}_beta"] = float(t.loc[x, "Estimate"])
                            out[f"cond_{x}_se"]   = float(t.loc[x, "Std. Error"])
                            out[f"cond_{x}_p"]    = float(t.loc[x, "Pr(>|t|)"])
        except Exception as e:
            print(f"      [{label}] failed: {type(e).__name__}: {e}")
            return None
        out["ci_lo"] = out["beta"] - 1.96 * out["se_state"]
        out["ci_hi"] = out["beta"] + 1.96 * out["se_state"]
        return out


    # =============================================================================
    print("=" * 78)
    print("script4a -- PART A design-based estimates  (A1, A2, A4, A5, A6)")
    print("=" * 78)

    df = load_panel()
    print(f"panel: {os.path.basename(df.attrs['panel_path'])}  {df.shape[0]:,} rows, "
          f"{df.fips.nunique():,} counties, {int(df.year.min())}-{int(df.year.max())}")
    print(f"treatment grid: {'LEGACY 12' if USE_LEGACY_GRID else 'CORE T1-T4'}  "
          f"({len(GRID_TREATMENTS)} definitions) x {len(OUTCOMES)} outcomes")
    print(f"controls: step-8 pre-committed ({len(CONTROLS_HEADLINE)}), D-16 -- single set, D-29")
    print(f"cohorts: {df[df.cohort.notna()].groupby('cohort').fips.nunique().to_dict()}")

    VIF_AUDIT = []


    # =============================================================================
    # A1 + A2 : the treatment grid, both FE structures, both control sets
    # =============================================================================
    print("\n" + "=" * 78)
    print("A1 (pooled state+year) / A2 (within county+year)")
    print("=" * 78)

    FE_SPECS = {"A1": ("state_fips + year", False), "A2": ("fips + year", True)}

    grid_rows = []
    for tid, spec in GRID_TREATMENTS.items():
        tcol, extra, tlabel = spec
        for okey, ocol in OUTCOMES.items():
            for cs_label, controls in CONTROL_SETS.items():
                need = ["fips", "year", "state_fips", ocol, tcol] + list(extra) + controls
                sub, mp = prep(df, need)
                if len(sub) < 200:
                    continue
                cc = [mp[c] for c in controls]
                var = variation(sub, mp[tcol])
                for reg, (fe, within) in FE_SPECS.items():
                    r = fit(sub, mp[ocol], mp[tcol], [mp[x] for x in extra], cc, fe,
                            f"{reg}|{tid}|{okey}|{cs_label}")
                    if not r:
                        continue
                    vif = rhs_vif(sub, [mp[tcol]] + [mp[x] for x in extra] + cc, within=within)
                    grid_rows.append({"registry": reg, "treatment_id": tid, "treatment": tlabel,
                                      "outcome": okey, "control_set": cs_label,
                                      **r, **var, **vif})
                    VIF_AUDIT.append({"model": reg, "treatment_id": tid, "outcome": okey,
                                      "control_set": cs_label, "within": within, **vif})
        print(f"  {tid} done")

    grid = pd.DataFrame(grid_rows)
    gpath = out_file(tables_dir, "4a-designs", f"{TODAY}_A1_A2_treatment_grid.csv")
    grid.to_csv(gpath, index=False)
    print(f"\nSaved: {gpath}   ({len(grid):,} rows)")

    hl = grid[(grid.registry == "A2") & (grid.control_set.str.startswith("step8"))
              & (grid.outcome == "Poor MH Days")]
    if len(hl):
        print("\n--- A2 within, step-8 controls, Poor MH Days ---")
        print(hl[["treatment_id", "beta", "se_state", "p_state", "N",
                  "n_switchers", "vif_max"]].to_string(index=False))


    # =============================================================================
    # A4 : dairy x FSIS interaction   (was A6)
    # =============================================================================
    # Is the dairy association concentrated where a slaughterhouse is also present?
    # ONE regression per treatment x outcome, containing the dairy term, the FSIS
    # term and their interaction -- not three separate regressions, which would not
    # identify the interaction. This is a TWFE interaction model, NOT an event study
    # (that is A5); it has no leads or lags.
    #
    # RUNS ACROSS ALL FOUR TREATMENTS (fixed 2026-09-28). The previous version
    # hard-coded tr_e1_lg_bin (T1) alone, which made A4 the only model in Part A not
    # estimated over the decided treatment set. There is no reason for the
    # restriction: T2/T3/T4 x FSIS-presence are all well defined (count x presence,
    # per-capita x presence, log-count x presence).
    #
    # SAMPLE: 2017-2023 only, because that is the FSIS coverage window.
    #
    # THE IDENTIFICATION PROBLEM THIS WINDOW CREATES (D-22). Treatment is measured at
    # ag-census waves and forward-filled, so it moves ONLY in 2007, 2012, 2017 and
    # 2022 (163 / 123 / 101 / 89 counties respectively). Restricting to 2017-2023
    # leaves exactly ONE wave transition inside the window: 2022.
    #   - mental health outcomes run to 2023  -> 2022 is inside  -> 89 switchers
    #   - crime outcomes end in 2021          -> NO transition   -> 0 switchers
    #   - Deaths of Despair ends in 2020      -> NO transition   -> 0 switchers
    # With zero switchers, county FE absorbs the dairy term completely and the
    # interaction is identified only off counties whose FSIS status changes -- 19 at
    # best. Those coefficients are NOT INTERPRETABLE and are written out with
    # identified=False rather than reported as nulls: a null claims power this design
    # does not have. Only the two mental-health outcomes are reportable.
    print("\n" + "=" * 78)
    print("A4: dairy x FSIS interaction (2017-2023 only, exploratory)")
    print("=" * 78)

    fsis_rows = []
    try:
        _fp = latest_file_glob(os.path.join(db_data, "merged"), "*_panel_fsis.csv")
        _f = pd.read_csv(_fp, low_memory=False)[["fips", "year", "n_unique_establishments_fsis"]]
        _f["fips"] = _f["fips"].astype(str).str.zfill(5)
        dff = df.merge(_f, on=["fips", "year"], how="left")
        cov = dff["year"].between(2017, 2023)
        # Outside 2017-2023 the FSIS panel simply does not cover: NaN, never 0.
        dff["any_fsis"] = np.where(cov, (dff["n_unique_establishments_fsis"].fillna(0) > 0)
                                   .astype(float), np.nan)
        for tid, tcol in CORE_TREATMENTS.items():
            extra_spec = SPEC_EXTRA_REGRESSORS.get(tid, [])
            ix = f"{tcol}_x_fsis"
            dff[ix] = dff[tcol] * dff["any_fsis"]
            for okey, ocol in OUTCOMES.items():
                need = (["fips", "year", "state_fips", ocol, tcol, "any_fsis", ix,
                         "cafo_dairy_large"] + list(extra_spec) + CONTROLS_HEADLINE)
                sub, mp = prep(dff, need)
                if len(sub) < 200:
                    continue
                cc  = [mp[c] for c in CONTROLS_HEADLINE]
                var = variation(sub, mp[tcol])
                # IDENTIFICATION TEST -- on the RAW CAFO COUNT, not on the treatment
                # variable. T3 is count/population, so its within-county variance is
                # non-zero whenever POPULATION moves, even when no CAFO ever opens or
                # closes. In 2017-2021 the raw count changes in 0 counties but T3
                # "switches" in 603 -- that variation is denominator drift, not
                # exposure, and a coefficient identified off it is not a dairy effect.
                # Testing the numerator catches this for every functional form at once.
                raw_sw = int((sub.groupby("fips")[mp["cafo_dairy_large"]].nunique() > 1).sum())
                identified = raw_sw > 0
                r = fit(sub, mp[ocol], mp[ix],
                        [mp[tcol], mp["any_fsis"]] + [mp[x] for x in extra_spec],
                        cc, "fips + year", f"A4|{tid}|{okey}")
                if r:
                    vif = rhs_vif(sub, [mp[ix], mp[tcol], mp["any_fsis"]]
                                  + [mp[x] for x in extra_spec] + cc)
                    fsis_rows.append({"registry": "A4", "treatment_id": tid,
                                      "treatment": CORE_TREATMENT_LABELS[tid],
                                      "outcome": okey, "term": f"{tid} x FSIS",
                                      "identified": identified,
                                      "n_raw_count_switchers": raw_sw,
                                      "not_identified_reason": ("" if identified else
                                          "raw large-dairy count never changes inside this "
                                          "outcome's year range -- no ag-census wave "
                                          "transition. Any apparent variation in T3 is "
                                          "population-denominator drift, not exposure."),
                                      **r, **var, **vif})
                    VIF_AUDIT.append({"model": "A4", "treatment_id": tid, "outcome": okey,
                                      "control_set": "step8", "within": True, **vif})
        print(f"  FSIS merged from {os.path.basename(_fp)}")
    except Exception as e:
        print(f"  FSIS panel unavailable ({type(e).__name__}: {e}) -- A4 skipped")

    fs = pd.DataFrame(fsis_rows)
    if len(fs):
        fs.to_csv(out_file(tables_dir, "4a-designs", f"{TODAY}_A4_fsis_interaction.csv"), index=False)
        ok = fs[fs.identified]
        print(f"\n  REPORTABLE ({len(ok)} of {len(fs)} specs -- the rest have 0 switchers):")
        if len(ok):
            print(ok[["treatment_id", "outcome", "beta", "se_state", "p_state", "N",
                      "n_switchers", "n_raw_count_switchers"]].to_string(index=False))
        bad = sorted(fs.loc[~fs.identified, "outcome"].unique())
        print(f"\n  NOT IDENTIFIED, suppressed: {', '.join(bad)}")


    # =============================================================================
    # A5 : TWFE event study around the first positive change   (was A7)
    # =============================================================================
    # Leads/lags relative to the county's first positive change in large-dairy ops,
    # omitted category t_rel = -1. Control group = never-treated pooled with
    # not-yet-treated (the standard TWFE event study). 95% CI throughout.
    #
    # READ THE PRE-TREND COEFFICIENTS BEFORE THE POST ONES. Treatment timing carries
    # up to 5 years of measurement error (ag census every 5 years, forward-filled),
    # so leads close to zero are partly contaminated by already-treated periods.
    #
    # This estimator inherits the forbidden-comparison problem that A6 exists to fix.
    # Read the two together, never A5 alone.
    print("\n" + "=" * 78)
    print("A5: TWFE event study around first positive change")
    print("=" * 78)

    LEADS, LAGS = 8, 8
    CLEAN_HEAD = [_clean(c) for c in CONTROLS_HEADLINE]

    # Baseline log population: the county's log population in 2000 (first panel
    # year, before every cohort's entry), so interacting it with year adjusts for
    # differential population TRENDS without conditioning on post-entry population,
    # which CAFO employment could itself move.
    _lp0 = (df[df.year == df.year.min()].set_index("fips")["log_pop"])
    df["log_pop_base"] = df["fips"].map(_lp0)


    def run_event_study(ycol, ylabel, controls, registry, extra_terms="", extra_cols=()):
        """TWFE event study around the first positive change. Returns rows."""
        rows = []
        need = ["fips", "year", "state_fips", ycol, "t_rel"] + list(controls) + list(extra_cols)
        sub = df[list(dict.fromkeys(need))].copy()
        sub["ev"] = sub["t_rel"].clip(-LEADS, LAGS)
        # never-treated (t_rel NaN) are pooled into the omitted category so pyfixest
        # KEEPS them as the comparison group instead of dropping them.
        sub.loc[sub["t_rel"].isna(), "ev"] = -1
        sub = sub.drop(columns=["t_rel"]).dropna()
        sub = sub.rename(columns={c: _clean(c) for c in sub.columns})
        if len(sub) < 500:
            return rows, None
        cc = [_clean(c) for c in controls]
        rhs = " + ".join(["i(ev, ref=-1)"] + cc) + extra_terms
        fml = f"{_clean(ycol)} ~ {rhs} | fips + year"
        try:
            m = pf.feols(fml, data=sub, vcov={"CRV1": "state_fips"})
            t = m.tidy()
        except Exception as e:
            print(f"  [{registry}|{ylabel}] event study failed: {type(e).__name__}: {e}")
            return rows, None
        for idx, r in t.iterrows():
            if not str(idx).startswith("ev::"):
                continue
            try:
                e = int(float(str(idx).split("::")[1]))
            except (ValueError, IndexError):
                continue
            b, se = float(r["Estimate"]), float(r["Std. Error"])
            rows.append({"registry": registry, "outcome": ylabel, "event_time": e,
                         "beta": b, "se_state": se, "p_state": float(r["Pr(>|t|)"]),
                         "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se, "N": int(m._N)})
        pre = [x for x in rows if x["event_time"] < -1]
        nsig = sum(1 for x in pre if x["p_state"] < 0.05)
        print(f"  {registry:12s} {ylabel:26s} pre-period coefs={len(pre):2d}  significant={nsig}  "
              f"{'PRE-TREND CONCERN' if nsig else 'pre-trend flat'}")
        return rows, sub


    es_rows = []
    for okey, ocol in OUTCOMES.items():
        rows, sub = run_event_study(ocol, okey, CONTROLS_HEADLINE, "A5")
        es_rows += rows
        if sub is not None:
            # VIF for A5 is computed on the COVARIATE BLOCK ONLY. The event-time
            # indicators are a mutually exclusive partition of the sample, so they
            # are collinear with each other BY CONSTRUCTION.
            vif5 = rhs_vif(sub, CLEAN_HEAD, within=True)
            VIF_AUDIT.append({"model": "A5", "treatment_id": "E2", "outcome": okey,
                              "control_set": "step8", "within": True,
                              "note": "covariate block only; event-time dummies excluded "
                                      "(collinear by construction)", **vif5})
        # (a) baseline log population x year -- adjusts for population trends
        rows, _ = run_event_study(ocol, okey, CONTROLS_HEADLINE, "A5_popbase",
                                  extra_terms=" + i(year, log_pop_base)",
                                  extra_cols=["log_pop_base"])
        es_rows += rows

    # (b) depopulation placebo: log population as the outcome, no controls
    rows, _ = run_event_study("log_pop", "Log population (placebo)", [], "A5_logpop")
    es_rows += rows

    es = pd.DataFrame(es_rows)
    es.to_csv(out_file(tables_dir, "4a-designs", f"{TODAY}_A5_event_study.csv"), index=False)


    # =============================================================================
    # A6 : CALLAWAY-SANT'ANNA STAGGERED DiD   (was A18)
    # =============================================================================
    # WHY THIS EXISTS. A2/A5 are two-way FE estimators. Under staggered adoption with
    # heterogeneous effects, TWFE is a variance-weighted average of 2x2 DiDs in which
    # ALREADY-TREATED units serve as controls for later-treated ones. Those
    # "forbidden comparisons" can carry NEGATIVE weights, so the TWFE coefficient
    # need not lie inside the range of the true county-level effects. CS removes them
    # by construction.
    #
    # ESTIMATOR, written out rather than called from a package (no `differences` or
    # `csdid` installed, and an explicit implementation is reviewable):
    #
    #   For cohort g (first positive change) and calendar year t, base period g-1:
    #       ATT(g,t) = E[Y_t - Y_{g-1} | G = g]  -  E[Y_t - Y_{g-1} | control]
    #   Control = NOT-YET-TREATED at t (cohort > t) plus NEVER-TREATED. No unit that
    #   is already treated at t is ever used as a control.
    #
    #   ATT(e) for event time e = t - g is the cohort-size-weighted mean of
    #   ATT(g, g+e). Overall ATT is the cohort-size-weighted mean of post ATT(g,t).
    #
    # INFERENCE: nonparametric cluster bootstrap resampling STATES with replacement
    # (300 reps), matching the state-clustered SEs used elsewhere in this file.
    #
    # TWO ARMS, BOTH RUN HERE (merged from the former script4i, 2026-09-29):
    #   unconditional       ATT(g,t) is a raw difference of long differences.
    #                       Parallel trends assumed UNCONDITIONALLY -- a STRONGER
    #                       assumption than A2 makes, since A2 conditions on X.
    #   covariate-adjusted  outcome-regression adjustment using the step-8 set
    #                       measured at base period g-1. Parallel trends assumed
    #                       CONDITIONAL on X, matching A2.
    # Running both in one place is the point: the headline TWFE-vs-CS table uses the
    # adjusted arm, so the comparison isolates the ESTIMATOR rather than confounding
    # it with the conditioning set. Both arms share the same cells and the same
    # bootstrap draws.
    #
    # A doubly-robust version would additionally need a propensity model fitted
    # inside each (g,t) cell; the cells are too thin for that and it is not attempted.
    # Cells too thin even for the outcome regression fall back to the unconditional
    # estimate and are counted in n_fallback_cells -- check that column before
    # describing an aggregate as "adjusted".
    print("\n" + "=" * 78)
    print("A6: Callaway-Sant'Anna staggered DiD (not-yet-treated + never-treated controls)")
    print("=" * 78)

    COHORTS = sorted(df.loc[df["cohort"].notna(), "cohort"].unique())
    print(f"  cohorts: {[int(c) for c in COHORTS]}")
    print(f"  control group: not-yet-treated (cohort > t) + never-treated")
    print(f"  inference: {N_BOOT}-rep cluster bootstrap over states")
    print(f"  covariates: NONE -- unconditional parallel trends. See docstring.\n")


    X_CS = [c for c in CONTROLS_HEADLINE if c in df.columns]
    print(f"  covariate-adjusted arm uses the step-8 set ({len(X_CS)} vars) at the earlier of year t and base period g-1 (D-44)\n")
    print(f"  {'outcome':26s} {'ATT uncond':>11s} {'ATT cond':>10s} {'se':>8s} {'cells':>6s} {'fallback':>9s}")

    cs_rows, csagg_rows = [], []
    for okey, ocol in OUTCOMES.items():
        d = df[["fips", "year", "state_fips", "cohort", ocol] + X_CS].dropna(subset=[ocol])
        if d.empty:
            continue
        wide      = d.pivot_table(index="fips", columns="year", values=ocol, aggfunc="first")
        cohort_of = d.groupby("fips")["cohort"].first()
        state_of  = d.groupby("fips")["state_fips"].first()
        years     = sorted(d["year"].unique())
        Xbase     = base_period_covariates(d, COHORTS, X_CS)

        states   = state_of.dropna().unique()
        by_state = {s: state_of.index[state_of == s] for s in states}
        line = {}
        # BOTH ARMS, same cells, same bootstrap draws -> the only difference between
        # the two columns is the conditioning, not the sample or the resampling.
        for arm, use_cov in [("unconditional", False), ("covariate-adjusted", True)]:
            pt = att_gt_table(wide, cohort_of, COHORTS, years, Xbase=Xbase, use_cov=use_cov)
            if not pt:
                continue
            keys = [(r["cohort"], r["year"]) for r in pt]
            boot, boot_overall = {k: [] for k in keys}, []
            boot_e = {}                                  # event-time aggregates per draw
            rng = np.random.default_rng(20260928)      # same draws for both arms
            for _b in range(N_BOOT):
                draw   = rng.choice(states, size=len(states), replace=True)
                fips_b = pd.Index(np.concatenate([by_state[s].values for s in draw]))
                pb = att_gt_table(wide, cohort_of, COHORTS, years,
                                  keep_fips=pd.Index(pd.unique(fips_b)),
                                  Xbase=Xbase, use_cov=use_cov)
                m = {(r["cohort"], r["year"]): r["att"] for r in pb}
                for k in keys:
                    if k in m:
                        boot[k].append(m[k])
                # event-time aggregate computed INSIDE the draw, with the same
                # cohort-size weights as the point estimate, so its SE is the
                # bootstrap SD of the aggregate itself (added 2026-10-12)
                _pbd = pd.DataFrame(pb)
                if len(_pbd):
                    for _e, _g in _pbd.groupby("event_time"):
                        boot_e.setdefault(int(_e), []).append(
                            float(np.average(_g["att"], weights=_g["n_treated"].astype(float))))
                post_b = [r for r in pb if r["event_time"] >= 0]
                if post_b:
                    w = np.array([r["n_treated"] for r in post_b], float)
                    boot_overall.append(float(np.average([r["att"] for r in post_b], weights=w)))

            for r in pt:
                b  = boot[(r["cohort"], r["year"])]
                se = float(np.std(b, ddof=1)) if len(b) > 10 else np.nan
                cs_rows.append({"registry": "A6", "arm": arm, "outcome": okey, **r,
                                "se_state": se,
                                "ci_lo": r["att"] - 1.96 * se, "ci_hi": r["att"] + 1.96 * se,
                                "p_state": float(2 * (1 - _norm.cdf(abs(r["att"] / se))))
                                if se and se > 0 else np.nan})

            pdf = pd.DataFrame(pt)
            for e, grp in pdf.groupby("event_time"):
                w = grp["n_treated"].astype(float)
                att_e = float(np.average(grp["att"], weights=w))
                be = boot_e.get(int(e), [])
                se_e = float(np.std(be, ddof=1)) if len(be) > 10 else np.nan
                csagg_rows.append({"registry": "A6", "arm": arm, "outcome": okey,
                                   "agg_level": "event_time", "event_time": int(e),
                                   "att": att_e, "se_state": se_e,
                                   "ci_lo": att_e - 1.96 * se_e, "ci_hi": att_e + 1.96 * se_e,
                                   "p_state": (float(2 * (1 - _norm.cdf(abs(att_e / se_e))))
                                               if se_e and se_e > 0 else np.nan),
                                   "n_cohorts": int(grp["cohort"].nunique()),
                                   "n_treated": int(grp["n_treated"].sum())})
            post = pdf[pdf.event_time >= 0]
            if len(post):
                overall = float(np.average(post["att"], weights=post["n_treated"].astype(float)))
                se_o = float(np.std(boot_overall, ddof=1)) if len(boot_overall) > 10 else np.nan
                nfb  = int(pdf.get("fallback", pd.Series(0, index=pdf.index)).sum())
                csagg_rows.append({"registry": "A6", "arm": arm, "outcome": okey,
                                   "agg_level": "overall_ATT",
                                   "event_time": np.nan, "att": overall, "se_state": se_o,
                                   "ci_lo": overall - 1.96 * se_o, "ci_hi": overall + 1.96 * se_o,
                                   "n_cohorts": int(post["cohort"].nunique()),
                                   "n_treated": int(post["n_treated"].sum()),
                                   "n_fallback_cells": nfb})
                line[arm] = (overall, se_o, len(pdf), nfb)

        if line:
            u = line.get("unconditional", (np.nan,)*4)
            c = line.get("covariate-adjusted", (np.nan,)*4)
            star = "*" if (c[1] and abs(c[0]) > 1.96 * c[1]) else " "
            print(f"  {okey:26s} {u[0]:+11.4f} {c[0]:+10.4f}{star}{c[1]:8.4f} "
                  f"{c[2]:6d} {c[3]:9d}")

    cs   = pd.DataFrame(cs_rows)
    csag = pd.DataFrame(csagg_rows)
    cs.to_csv(out_file(tables_dir, "4a-designs", f"{TODAY}_A6_cs_att_gt.csv"), index=False)
    csag.to_csv(out_file(tables_dir, "4a-designs", f"{TODAY}_A6_cs_aggregated.csv"), index=False)


    # =============================================================================
    # A7 : LP-DiD  (Dube, Girardi, Jorda & Taylor 2023)   -- added 2026-10-12
    # =============================================================================
    # Same E2 cohorts as A5/A6. For each horizon h, a stacked regression of the long
    # difference Y_{t+h} - Y_{t-1} on a newly-treated dummy, event-year FE, and the
    # step-8 controls measured at t-1, on newly treated + clean (not-yet / never
    # treated) counties only. Regression-based inference, state-clustered. h < -1 is
    # the pre-trend test. Treatment timing is still only observed at census waves,
    # so h is in years but the identifying events occur every 5 years.
    print("\n" + "=" * 78)
    print("A7: LP-DiD (clean controls: not-yet-treated + never-treated)")
    print("=" * 78)
    lp_rows = []
    for okey, ocol in OUTCOMES.items():
        pan = df[["fips", "year", "state_fips", "cohort", ocol] + CONTROLS_HEADLINE].copy()
        pan = pan.rename(columns={c: _clean(c) for c in pan.columns})
        for arm, ctrls in [("no controls", []), ("step-8 at t-1", CLEAN_HEAD)]:
            r = lpdid(pan, _clean(ocol), horizons=range(-LEADS, LAGS + 1), controls=ctrls)
            if r.empty:
                continue
            r.insert(0, "arm", arm); r.insert(0, "outcome", okey); r.insert(0, "registry", "A7")
            lp_rows.append(r)
            pre = r[r.h < -1]
            print(f"  {okey:26s} {arm:14s} pre sig={int((pre.p < .05).sum())}/{len(pre)}  "
                  f"h=0: {r.loc[r.h == 0, 'beta'].squeeze() if (r.h == 0).any() else float('nan'):+.4f}")
    lp_df = pd.concat(lp_rows, ignore_index=True) if lp_rows else pd.DataFrame()
    lp_df.to_csv(out_file(tables_dir, "4a-designs", f"{TODAY}_A7_lpdid.csv"), index=False)


    # =============================================================================
    # CONLEY SPATIAL SEs on the A2 grid   -- added 2026-10-12
    # =============================================================================
    # State clustering allows arbitrary correlation inside a state and none across a
    # state line. Conley SEs instead let correlation decay with distance (Bartlett,
    # cutoffs 100/250/500 km, within year) plus serial correlation within county.
    # Needs county centroids; skipped with a message if the Gazetteer file is absent.
    cent = load_centroids(db_data)
    if cent is None:
        print("\nConley SEs skipped: no county Gazetteer file in Data/raw/gazetteer/ "
              "(download 2020_Gaz_counties_national.txt from census.gov).")
    else:
        print("\n" + "=" * 78)
        print("Conley spatial SEs, A2 grid")
        print("=" * 78)
        conley_rows = []
        for tid, spec in GRID_TREATMENTS.items():
            tcol, extra, _ = spec
            for okey, ocol in OUTCOMES.items():
                need = ["fips", "year", "state_fips", ocol, tcol] + list(extra) + CONTROLS_HEADLINE
                sub, mp = prep(df, need)
                sub = sub.merge(cent, on="fips", how="inner").reset_index(drop=True)
                if len(sub) < 200:
                    continue
                fml = (f"{mp[ocol]} ~ " + " + ".join([mp[tcol]] + [mp[x] for x in extra]
                       + [mp[c] for c in CONTROLS_HEADLINE]) + " | fips + year")
                try:
                    m = pf.feols(fml, data=sub, vcov="hetero", fixef_rm="none")
                except Exception as e:
                    print(f"  [{tid}|{okey}] {e}"); continue
                if m._N != len(sub):
                    print(f"  [{tid}|{okey}] row mismatch, skipped"); continue
                rec = {"treatment_id": tid, "outcome": okey,
                       "beta": float(m.coef()[mp[tcol]]), "N": int(m._N)}
                for km in (100, 250, 500):
                    rec[f"se_conley_{km}km"] = conley_se(m, sub, mp[tcol], sub["lat"].values,
                                                         sub["lon"].values, cutoff_km=km)
                conley_rows.append(rec)
            print(f"  {tid} done")
        pd.DataFrame(conley_rows).to_csv(
            out_file(tables_dir, "4a-designs", f"{TODAY}_A2_conley_se.csv"), index=False)


    # =============================================================================
    # HEADLINE COMPARISON: A2 (TWFE) vs A6 (Callaway-Sant'Anna)
    # =============================================================================
    # The comparison is only meaningful if the TREATMENT DEFINITION is held fixed and
    # the ONLY thing that changes is the estimator. A gap is then the
    # forbidden-comparison problem showing up.
    #
    # WHICH TREATMENT CS ACTUALLY USES (corrected 2026-09-28). CS is defined on
    # cohorts, and `cohort = add_year` = the first ag-census wave at which the county
    # records a POSITIVE change in its large-dairy count. That treatment is BINARY
    # and ABSORBING -- once treated, always treated. It is `tr_e2_add_absorb`, NOT
    # T1.
    #
    # T1 (tr_e1_lg_bin) is contemporaneous PRESENCE and can switch back off when a
    # county loses its last large dairy. The two disagree on 10.6% of rows
    # (T1=1/absorb=0 in 5,462 rows; T1=0/absorb=1 in 430) and cover different county
    # sets (759 vs 572 ever-treated; 365 vs 572 switchers).
    #
    # Comparing A2(T1) against A6(cohort) would therefore change the estimator AND
    # the treatment at once, and any gap could not be attributed to either. The TWFE
    # side of this table is estimated on tr_e2_add_absorb for that reason. T1 results
    # live in the A1/A2 grid above and are not comparable to CS.
    print("\n" + "=" * 78)
    print("HEADLINE: A2 (TWFE) vs A6 (Callaway-Sant'Anna)")
    print("  treatment held fixed at tr_e2_add_absorb (the CS cohort definition)")
    print("  CS arm = covariate-adjusted, so both sides condition on the step-8 set")
    print("=" * 78)

    CMP_TID = "E2"
    cmp_rows = []
    for okey in OUTCOMES:
        a2 = grid[(grid.registry == "A2") & (grid.treatment_id == CMP_TID)
                  & (grid.outcome == okey) & (grid.control_set.str.startswith("step8"))]
        # Use the COVARIATE-ADJUSTED arm: it conditions on the same step-8 set that
        # the TWFE side uses, so the only difference between the columns is the
        # estimator. The unconditional arm is in the output for reference.
        a6 = (csag[(csag["agg_level"] == "overall_ATT") & (csag.outcome == okey)
                   & (csag.arm == "covariate-adjusted")]
              if len(csag) else pd.DataFrame())
        if a2.empty or a6.empty:
            continue
        a2, a6 = a2.iloc[0], a6.iloc[0]
        sig2 = a2.p_state < 0.05
        sig6 = bool(a6.se_state and abs(a6.att) > 1.96 * a6.se_state)
        cmp_rows.append({
            "outcome": okey,
            "TWFE_beta": a2.beta, "TWFE_se": a2.se_state, "TWFE_sig": sig2,
            "CS_ATT": a6.att, "CS_se": a6.se_state, "CS_sig": sig6,
            "ratio_CS_over_TWFE": a6.att / a2.beta if a2.beta else np.nan,
            "verdict": ("survives CS" if (sig2 and sig6) else
                        "TWFE-only (likely forbidden comparisons)" if sig2 else
                        "CS-only" if sig6 else "null in both"),
            "n_switchers_TWFE": a2.n_switchers, "n_treated_CS": a6.n_treated,
        })
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(out_file(tables_dir, "4a-designs", f"{TODAY}_HEADLINE_twfe_vs_cs.csv"), index=False)
    if len(cmp):
        print(cmp[["outcome", "TWFE_beta", "TWFE_se", "TWFE_sig",
                   "CS_ATT", "CS_se", "CS_sig", "verdict"]].to_string(index=False))


    # =============================================================================
    # VIF AUDIT
    # =============================================================================
    # VIF IS COMPUTED PER MODEL, NOT ONCE FOR THE COVARIATE SET. Two reasons, both
    # demonstrated on this panel:
    #   1. THE FE STRUCTURE CHANGES IT. Collinearity among these controls is almost
    #      entirely CROSS-SECTIONAL, and county FE removes it. On the headline
    #      sample median_household_income is VIF 2.98 in raw levels but 1.16
    #      within-transformed; children_in_poverty 2.86 -> 1.13; the block max falls
    #      2.98 -> 1.45. Quoting a raw-level VIF for a county-FE model overstates
    #      collinearity by roughly a factor of two.
    #   2. THE RIGHT-HAND SIDE CHANGES IT. The treatment and any conditioning terms
    #      are part of the design matrix. Adding T2 plus log_pop moves the block max
    #      1.45 -> 1.57. A3's conditioning sets and A4's interaction move it further.
    # So each row below is the VIF of the ACTUAL design matrix that produced the
    # corresponding coefficient, within-transformed wherever the model uses county FE.
    va = pd.DataFrame(VIF_AUDIT)
    va.to_csv(out_file(tables_dir, "4a-designs", f"{TODAY}_VIF_audit.csv"), index=False)
    print("\n" + "=" * 78)
    print("VIF AUDIT -- per model, on the actual design matrix")
    print("=" * 78)
    if len(va):
        summ = (va.groupby("model")
                  .agg(specs=("vif_max", "size"), transform=("within", "first"),
                       vif_max=("vif_max", "max"), vif_median=("vif_max", "median"),
                       over10=("vif_max", lambda s: int((s > 10).sum())),
                       absorbed=("n_zero_var", "sum"))
                  .reset_index())
        summ["transform"] = np.where(summ["transform"], "within", "raw levels")
        print(summ.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print(f"\n  TOTAL specs audited: {len(va):,}   max VIF anywhere: {va.vif_max.max():.2f}"
              f"  on {va.loc[va.vif_max.idxmax(),'vif_max_var']}")
        print(f"  specs above 10 (the usual rule of thumb): {int((va.vif_max > 10).sum())}")
        print(f"  regressors absorbed by the FE (zero within-variance): {int(va.n_zero_var.sum())}")
        print("\n  A6 has no VIF: Callaway-Sant'Anna estimates cell means, not a regression.")


    # =============================================================================
    # FIGURES
    # =============================================================================
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

    outs = [o for o in OUTCOMES if o in set(grid.outcome)]

    # --- F1: A2 within coefficients, step-8 controls, REPORTED treatments ---------
    # The estimation above covers every treatment; the figure shows the reported set
    # (script4_treatment.REPORT_TREATMENTS, 2026-10-15).
    if len(outs):
        n = len(outs); ncol = 4; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
        for ax, okey in zip(axes.ravel(), outs):
            v = grid[(grid.registry == "A2") & (grid.outcome == okey)
                     & (grid.control_set.str.startswith("step8"))
                     & grid.treatment_id.isin(REPORT_TREATMENTS)].sort_values("treatment_id")
            if v.empty:
                ax.axis("off"); continue
            y = np.arange(len(v))
            ax.errorbar(v.beta, y, xerr=1.96 * v.se_state, fmt="o", ms=4, lw=1, capsize=2)
            ax.axvline(0, color="grey", lw=0.8, ls="--")
            ax.set_yticks(y); ax.set_yticklabels(v.treatment_id, fontsize=8)
            ax.set_title(okey, fontsize=9)
        for ax in axes.ravel()[len(outs):]:
            ax.axis("off")
        fig.suptitle("A2 within-county, step-8 pre-committed controls (95% CI, state-clustered)",
                     fontsize=10)
        fig.tight_layout()
        f1 = out_file(figs_dir, "4a-designs", f"{TODAY}_F1_A2_treatment_grid.png")
        fig.savefig(f1, bbox_inches="tight"); plt.close(fig)
        print("\nSaved:", f1)

    # --- F2: A5 event studies ----------------------------------------------------
    if len(es):
        eo = [o for o in OUTCOMES if o in set(es.outcome)]
        n = len(eo); ncol = 4; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
        for ax, okey in zip(axes.ravel(), eo):
            v = es[es.outcome == okey].sort_values("event_time")
            ax.errorbar(v.event_time, v.beta, yerr=1.96 * v.se_state, fmt="o-", ms=3, lw=1, capsize=2)
            ax.axhline(0, color="grey", lw=0.8, ls="--")
            ax.axvline(-1, color="red", lw=0.8, ls=":")
            ax.set_title(okey, fontsize=9); ax.set_xlabel("event time", fontsize=8)
        for ax in axes.ravel()[len(eo):]:
            ax.axis("off")
        fig.suptitle("A5 TWFE event study (95% CI, state-clustered; ref = -1)", fontsize=10)
        fig.tight_layout()
        f2 = out_file(figs_dir, "4a-designs", f"{TODAY}_F2_A5_event_study.png")
        fig.savefig(f2, bbox_inches="tight"); plt.close(fig)
        print("Saved:", f2)

    # --- F3: A6 event-time ATT ---------------------------------------------------
    if len(csag):
        ev = csag[(csag["agg_level"] == "event_time") & (csag.arm == "covariate-adjusted")]
        eo = [o for o in OUTCOMES if o in set(ev.outcome)]
        if eo:
            n = len(eo); ncol = 4; nrow = int(np.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
            for ax, okey in zip(axes.ravel(), eo):
                v = ev[ev.outcome == okey].sort_values("event_time")
                ax.plot(v.event_time, v.att, "o-", ms=3, lw=1)
                ax.axhline(0, color="grey", lw=0.8, ls="--")
                ax.axvline(0, color="red", lw=0.8, ls=":")
                ax.set_title(okey, fontsize=9); ax.set_xlabel("event time", fontsize=8)
            for ax in axes.ravel()[len(eo):]:
                ax.axis("off")
            fig.suptitle("A6 Callaway-Sant'Anna ATT by event time (cohort-size weighted)",
                         fontsize=10)
            fig.tight_layout()
            f3 = out_file(figs_dir, "4a-designs", f"{TODAY}_F3_A6_cs_event_time.png")
            fig.savefig(f3, bbox_inches="tight"); plt.close(fig)
            print("Saved:", f3)

    print("\n" + "=" * 78)
    print("script4a COMPLETE")
    print(f"  tables -> {OUT_TABLES}")
    print(f"  figs   -> {OUT_FIGS}")
    print("=" * 78)


def section_cohorts():
    """First expansion and first entry designs.

    Old script script4u-did-by-treatment.py (docstring below), code moved unchanged.

    script4u-did-by-treatment.py -- TWFE, event study and Callaway-Sant'Anna for first expansion and first entry, with pre-trend adjustments.

    Designs: first expansion (all rural counties); first entry (entry counties plus never-large counties as controls; D-32).
    Adjustments: county trends, state x year FE, baseline characteristics x time polynomial, past outcomes in CS.
    Reads:   panel via designs.load_designs()
    Writes:  Data/output/tables/<date>_4a-cohorts_{static_twfe, event_study, event_study_leads_joint, cs_att_gt, cs_aggregated, validation_vs_script4a}.csv
    Run:     python3 script4u-did-by-treatment.py   (about 30 minutes; --no-cs skips CS; --quick is a test run)
    Method:  METHODS.md sections 2-3. The first-expansion baseline is checked against script4a; the CS code is in estimators.py.
    """
    import warnings; warnings.filterwarnings("ignore")
    import glob
    import sys

    import numpy as np
    import pandas as pd
    import pyfixest as pf
    from scipy.stats import chi2

    from estimators import run_cs, clean, joint_wald, fe_ols, event_study_frame, stars as st
    from designs import load_designs
    from script4_treatment import OUTCOMES, COVARIATE_ORDER, tables_dir, os, date

    QUICK = "--quick" in sys.argv
    NO_CS = "--no-cs" in sys.argv
    N_BOOT = 20 if QUICK else 300
    OUT, TAG = tables_dir, "4a-cohorts" + ("-quick" if QUICK else "")    # was folder script4u
    TODAY = date.today().strftime("%Y-%m-%d")
    CONTROLS = COVARIATE_ORDER[:8]
    CLIP = 8
    OUTCOME_LIST = dict(list(OUTCOMES.items())[:2]) if QUICK else OUTCOMES


    df, DESIGNS, POLY_COLS, _counts = load_designs(with_polynomial=True)
    PF_SPECS = [("baseline", "fips + year", []),
                ("state x year FE", "fips + state_fips^year", []),
                ("baseline chars x time polynomial", "fips + year", POLY_COLS)]

    print(f"entry design: {_counts['n_entry']} entry counties + {_counts['n_never_large']} never-large counties "
          f"= {_counts['n_entry_sample']} of {_counts['n_all']} counties")

    static_rows, es_rows, joint_rows, gt_all, agg_all, fe_checks = [], [], [], [], [], []

    for dname, D in DESIGNS.items():
        data, treat, cohort = D["data"], D["treat"], D["cohort"]
        print(f"\n##### {dname}: {data[cohort].notna().groupby(data.fips).first().sum() if False else data.loc[data[cohort].notna(),'fips'].nunique()} treated counties")
        for okey, ocol in OUTCOME_LIST.items():
            # ----------------------------- static TWFE -----------------------------
            for spec, fe, extra in PF_SPECS:
                need = list(dict.fromkeys(["fips", "year", "state_fips", ocol, treat] + CONTROLS + extra))
                sub = data[need].dropna().rename(columns={c: clean(c) for c in need})
                nsw = int((sub.groupby("fips")[clean(treat)].nunique() > 1).sum())
                xc = [clean(treat)] + [clean(c) for c in CONTROLS] + [clean(c) for c in extra]
                f = f"{clean(ocol)} ~ " + " + ".join(xc) + f" | {fe}"
                try:
                    m = pf.feols(f, data=sub, vcov={"CRV1": "state_fips"}); t = m.tidy().loc[clean(treat)]
                    static_rows.append({"design": dname, "spec": spec, "outcome": okey, "beta": float(t["Estimate"]),
                                        "se_state": float(t["Std. Error"]), "p_state": float(t["Pr(>|t|)"]),
                                        "N": int(m._N), "n_switchers": nsw})
                    if spec == "baseline":
                        # internal check: fe_ols without trends must reproduce pyfixest
                        chk, _, _ = fe_ols(sub.reset_index(drop=True), clean(ocol), xc, trend=False)
                        fe_checks.append({"design": dname, "outcome": okey,
                                          "beta_diff": abs(chk.loc[clean(treat), "beta"] - float(t["Estimate"])),
                                          "se_diff": abs(chk.loc[clean(treat), "se"] - float(t["Std. Error"])) / float(t["Std. Error"])})
                        res, _, N_ = fe_ols(sub.reset_index(drop=True), clean(ocol), xc, trend=True)
                        r = res.loc[clean(treat)]
                        static_rows.append({"design": dname, "spec": "county trends", "outcome": okey, "beta": float(r.beta),
                                            "se_state": float(r.se), "p_state": float(r.p), "N": int(N_), "n_switchers": nsw})
                except Exception as e:
                    print(f"  static [{dname}|{spec}|{okey}] failed: {type(e).__name__}: {e}")
            # ----------------------------- event study -----------------------------
            for spec, fe, extra in PF_SPECS:
                try:
                    es = event_study_frame(data, ocol, CONTROLS, extra, CLIP)
                    f = (f"{clean(ocol)} ~ i(ev, ref=-1) + " + " + ".join(clean(c) for c in CONTROLS + extra) + f" | {fe}")
                    m = pf.feols(f, data=es, vcov={"CRV1": "state_fips"}); tid = m.tidy()
                    leads = []
                    for idx, r in tid.iterrows():
                        if not str(idx).startswith("ev::"):
                            continue
                        e = int(float(str(idx).split("::")[1]))
                        es_rows.append({"design": dname, "spec": spec, "outcome": okey, "event_time": e,
                                        "beta": float(r["Estimate"]), "se_state": float(r["Std. Error"]),
                                        "p_state": float(r["Pr(>|t|)"]), "N": int(m._N)})
                        if e < -1:
                            leads.append(str(idx))
                    stat, k, p = joint_wald(m, leads)
                    joint_rows.append({"design": dname, "spec": spec, "outcome": okey, "n_leads": k,
                                       "wald_chi2": stat, "joint_p": p, "N": int(m._N)})
                except Exception as e:
                    print(f"  event [{dname}|{spec}|{okey}] failed: {type(e).__name__}: {e}")
            # county trends: event-time dummies built by hand, residualized with county trends
            try:
                es = event_study_frame(data, ocol, CONTROLS, [], CLIP)
                evs = sorted(int(x) for x in es["ev"].unique() if x != -1)
                D_ = pd.DataFrame({f"ev_{e}": (es["ev"] == e).astype(float) for e in evs}, index=es.index)
                ee = pd.concat([es.reset_index(drop=True), D_.reset_index(drop=True)], axis=1)
                xc = list(D_.columns) + [clean(c) for c in CONTROLS]
                res, Vc, N_ = fe_ols(ee, clean(ocol), xc, trend=True)
                lead_idx = [i for i, e in enumerate(evs) if e < -1]
                bl = res.beta.to_numpy()[lead_idx]; Vl = Vc[np.ix_(lead_idx, lead_idx)]
                stat = float(bl @ np.linalg.solve(Vl, bl))
                for e in evs:
                    r = res.loc[f"ev_{e}"]
                    es_rows.append({"design": dname, "spec": "county trends", "outcome": okey, "event_time": e,
                                    "beta": float(r.beta), "se_state": float(r.se), "p_state": float(r.p), "N": int(N_)})
                joint_rows.append({"design": dname, "spec": "county trends", "outcome": okey, "n_leads": len(lead_idx),
                                   "wald_chi2": stat, "joint_p": float(1 - chi2.cdf(stat, len(lead_idx))), "N": int(N_)})
            except Exception as e:
                print(f"  event [{dname}|county trends|{okey}] failed: {type(e).__name__}: {e}")
            # ----------------------------- Callaway-Sant'Anna -----------------------------
            for spec, lags in ([] if NO_CS else [("controls", 0), ("controls + past outcomes", 2)]):
                gt, ag = run_cs(data, ocol, cohort, CONTROLS, n_boot=N_BOOT, outcome_lags=lags)
                if len(gt):
                    gt.insert(0, "outcome", okey); gt.insert(0, "spec", spec); gt.insert(0, "design", dname)
                    ag.insert(0, "outcome", okey); ag.insert(0, "spec", spec); ag.insert(0, "design", dname)
                    gt_all.append(gt); agg_all.append(ag)
            print(f"  done: {okey}", flush=True)

    static = pd.DataFrame(static_rows); es_df = pd.DataFrame(es_rows); joint = pd.DataFrame(joint_rows)
    outputs = [("static_twfe", static), ("event_study", es_df), ("event_study_leads_joint", joint)]
    if not NO_CS:
        gt_df = pd.concat(gt_all, ignore_index=True); agg_df = pd.concat(agg_all, ignore_index=True)
        outputs += [("cs_att_gt", gt_df), ("cs_aggregated", agg_df)]
    for name, x in outputs:
        x.to_csv(out_file(tables_dir, TAG, f"{TODAY}_{name}.csv"), index=False)

    # ---------------------------- validation against script4a (first expansion, baseline) ------------------
    val = []
    def newest(pat):
        f = out_glob(tables_dir, "4a-designs", pat)
        return pd.read_csv(f[-1]) if f else None
    grid, a5, a6 = newest("*_A1_A2_treatment_grid.csv"), newest("*_A5_event_study.csv"), newest("*_A6_cs_aggregated.csv")
    mine = "first expansion"
    if grid is not None:
        g = grid[(grid.registry == "A2") & (grid.treatment_id == "E2")].set_index("outcome")
        s = static[(static.design == mine) & (static.spec == "baseline")].set_index("outcome")
        common = s.index.intersection(g.index)
        val.append({"check": "static TWFE beta", "max_abs_diff": float((s.loc[common, "beta"] - g.loc[common, "beta"]).abs().max()),
                    "n": len(common)})
        val.append({"check": "static TWFE se", "max_abs_diff": float((s.loc[common, "se_state"] - g.loc[common, "se_state"]).abs().max()),
                    "n": len(common)})
    if a5 is not None:
        a = a5[a5.registry == "A5"][["outcome", "event_time", "beta"]]
        m_ = es_df[(es_df.design == mine) & (es_df.spec == "baseline")].merge(a, on=["outcome", "event_time"], suffixes=("", "_4a"))
        val.append({"check": "event study beta", "max_abs_diff": float((m_.beta - m_.beta_4a).abs().max()), "n": len(m_)})
    if a6 is not None and not NO_CS:
        a = a6[(a6.arm == "covariate-adjusted") & (a6.agg_level == "overall_ATT")][["outcome", "att", "se_state"]]
        m_ = agg_df[(agg_df.design == mine) & (agg_df.spec == "controls") & (agg_df.agg_level == "overall_ATT")] \
            .merge(a, on="outcome", suffixes=("", "_4a"))
        val.append({"check": "CS overall ATT", "max_abs_diff": float((m_.att - m_.att_4a).abs().max()), "n": len(m_)})
        val.append({"check": "CS overall ATT bootstrap se", "max_abs_diff": float((m_.se_state - m_.se_state_4a).abs().max()), "n": len(m_)})
    fc = pd.DataFrame(fe_checks)
    val.append({"check": "fe_ols (no trend) vs pyfixest, static beta", "max_abs_diff": float(fc.beta_diff.max()), "n": len(fc)})
    val.append({"check": "fe_ols (no trend) vs pyfixest, static se (RELATIVE diff)", "max_abs_diff": float(fc.se_diff.max()), "n": len(fc)})
    val = pd.DataFrame(val)
    val.to_csv(out_file(tables_dir, TAG, f"{TODAY}_validation_vs_script4a.csv"), index=False)


    print("\n" + "=" * 100 + "\nVALIDATION: first-expansion baseline vs script4a (should be ~0)\n" + "=" * 100)
    print(val.to_string(index=False))
    print("\n" + "=" * 100 + "\nSTATIC TWFE   beta (stars)   [county FE + year FE, SE clustered by state]\n" + "=" * 100)
    for dname in DESIGNS:
        print(f"\n{dname}")
        for o in OUTCOME_LIST:
            r = static[(static.design == dname) & (static.outcome == o)]
            print(f"  {o[:24]:24s} " + " | ".join(f"{x.spec}: {x.beta:+.3g}{st(x.p_state)} (z{x.beta / x.se_state:+.1f})" for x in r.itertuples())
                  + f" | N={int(r.N.iloc[0]):,} switchers={int(r.n_switchers.iloc[0])}")
    print("\n" + "=" * 100 + "\nEVENT STUDY: joint test that all leads = 0 (p)\n" + "=" * 100)
    print(joint.pivot_table(index=["design", "outcome"], columns="spec", values="joint_p").round(3).to_string())
    print("\n" + "=" * 100 + "\nCS OVERALL ATT   att (stars) [bootstrap over states]\n" + "=" * 100)
    ov = agg_df[agg_df.agg_level == "overall_ATT"] if not NO_CS else pd.DataFrame(columns=["design", "outcome"])
    for dname in DESIGNS:
        print(f"\n{dname}")
        for o in OUTCOME_LIST:
            r = ov[(ov.design == dname) & (ov.outcome == o)]
            print(f"  {o[:24]:24s} " + " | ".join(f"{x.spec}: {x.att:+.3g}{st(x.p_state)} (z{x.att / x.se_state:+.1f}, fb={int(x.n_fallback_cells)})" for x in r.itertuples()))
    print(f"\nSaved -> {OUT}")


def section_fsis():
    """Slaughterhouses (FSIS plants) as the treatment.

    Old script script4r-fsis-treatment.py (docstring below), code moved unchanged.

    script4r-fsis-treatment.py -- slaughterhouses (FSIS plants) as the treatment.

    Treatments: F1 = number of FSIS plants (2017-2023); F2 = first year the count rises. Models R0-R3 (event counts, TWFE on F1 and F2, event study).
    Reads:   panel via script4_treatment.load_panel(); Data/merged/*_panel_fsis.csv
    Writes:  Data/output/tables/<date>_4a-fsis_{R0_entry_events, R1_R2_twfe, R3_event_study, R4_lpdid}.csv
    Run:     python3 script4r-fsis-treatment.py
    Notes:   FSIS starts in 2017, so entries before 2017 are invisible; R4 (LP-DiD) is computed but not reported.
    """
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, pandas as pd, pyfixest as pf
    from script4_treatment import load_panel, OUTCOMES, COVARIATE_ORDER, db_data, tables_dir, os, date
    from functions import latest_file_glob
    from estimators import lpdid

    OUT = tables_dir    # output-name tag "4a-fsis" (packages.py); was folder script4r
    TODAY = date.today().strftime("%Y-%m-%d")
    CONTROLS = COVARIATE_ORDER[:8]
    SLAUGHTER = "n_slaughterhouse_present_establishments_fsis"
    FSIS_YEARS = (2017, 2023)


    def cl(c):
        return (c.replace("%", "pct").replace("-", "_").replace(" ", "_").replace("(", "")
                 .replace(")", "").replace("/", "_").replace("+", "p").replace(",", "_"))


    df = load_panel()
    fp = latest_file_glob(os.path.join(db_data, "merged"), "*_panel_fsis.csv")
    f = pd.read_csv(fp, low_memory=False, usecols=["fips", "year", SLAUGHTER])
    f["fips"] = f["fips"].astype(str).str.zfill(5)
    df = df.merge(f, on=["fips", "year"], how="left")
    inwin = df.year.between(*FSIS_YEARS)
    df["F1_count"] = np.where(inwin, df[SLAUGHTER].fillna(0), np.nan)

    # entry: first year the count rises, observed 2018+
    w = df[inwin].sort_values(["fips", "year"])
    w["d"] = w.groupby("fips")["F1_count"].diff()
    first_up = w[w.d > 0].groupby("fips")["year"].min()
    df["F_cohort"] = df["fips"].map(first_up)
    df["F2_entry"] = np.where(df.F_cohort.notna() & (df.year >= df.F_cohort), 1.0, 0.0)
    df["F_trel"] = df.year - df.F_cohort
    print(f"FSIS merged from {os.path.basename(fp)}")

    # ============================== R0 event counts ==============================
    ev = w.assign(up=w.d > 0, down=w.d < 0)
    r0 = pd.DataFrame({
        "counties_first_entry": first_up.value_counts().sort_index(),
        "counties_any_increase": ev[ev.up].groupby("year").fips.nunique(),
        "counties_any_decrease": ev[ev.down].groupby("year").fips.nunique(),
    }).fillna(0).astype(int)
    r0.index.name = "year"
    r0.loc["total"] = r0.sum()
    r0["note"] = ""
    base17 = w[w.year == 2017]
    print("\nR0  entry events by year")
    print(r0.to_string())
    print(f"  counties with >=1 slaughter plant in 2017: {(base17.F1_count > 0).sum()} of {len(base17)}")
    # usable cohorts per outcome: cohort g needs outcome data at g-1 and g
    use = []
    for okey, ocol in OUTCOMES.items():
        yrs = set(df.loc[df[ocol].notna(), "year"])
        for g, n in first_up.value_counts().sort_index().items():
            use.append({"outcome": okey, "cohort": int(g), "counties": int(n),
                        "usable": (g - 1 in yrs) and (g in yrs)})
    use = pd.DataFrame(use)
    r0.to_csv(out_file(tables_dir, "4a-fsis", f"{TODAY}_R0_entry_events.csv"))
    use.to_csv(out_file(tables_dir, "4a-fsis", f"{TODAY}_R0_cohorts_by_outcome.csv"), index=False)
    print("\n  usable entry counties by outcome:")
    print(use[use.usable].groupby("outcome").counties.sum().to_string())


    # ============================== R1 / R2 TWFE =================================
    def twfe(sub, y, t, label):
        fml = f"{y} ~ {t} + " + " + ".join(cl(c) for c in CONTROLS) + " | fips + year"
        m = pf.feols(fml, data=sub, vcov={"CRV1": "state_fips"})
        r = m.tidy().loc[t]
        sw = int((sub.groupby("fips")[t].nunique() > 1).sum())
        sdw = float((sub[y] - sub.groupby("fips")[y].transform("mean")).std())
        se = float(r["Std. Error"])
        return {"model": label, "beta": float(r["Estimate"]), "se_state": se,
                "p_state": float(r["Pr(>|t|)"]), "N": int(m._N),
                "n_counties": int(sub.fips.nunique()), "n_states": int(sub.state_fips.nunique()),
                "n_switchers": sw, "sd_within": sdw,
                "MDE_sd_within": 2.8016 * se / sdw if sdw > 0 else np.nan}

    rows = []
    for okey, ocol in OUTCOMES.items():
        need = ["fips", "year", "state_fips", ocol] + CONTROLS
        a = df[df.year.between(*FSIS_YEARS)][need + ["F1_count"]].dropna()
        a = a.rename(columns={c: cl(c) for c in a.columns})
        if len(a) > 200 and a.F1_count.std() > 0:
            try:
                rows.append({"outcome": okey, "treatment": "F1 count", **twfe(a, cl(ocol), "F1_count", "R1")})
            except Exception as e:
                print(f"  R1 {okey}: {e}")
        b = df[df.year.between(2012, 2023)][need + ["F2_entry"]].dropna()
        b = b.rename(columns={c: cl(c) for c in b.columns})
        if len(b) > 200 and b.F2_entry.std() > 0:
            try:
                rows.append({"outcome": okey, "treatment": "F2 entry", **twfe(b, cl(ocol), "F2_entry", "R2")})
            except Exception as e:
                print(f"  R2 {okey}: {e}")
    tw = pd.DataFrame(rows)
    tw.to_csv(out_file(tables_dir, "4a-fsis", f"{TODAY}_R1_R2_twfe.csv"), index=False)
    print("\nR1/R2 TWFE")
    print(tw[["model", "outcome", "beta", "se_state", "p_state", "N", "n_switchers",
              "MDE_sd_within"]].to_string(index=False))

    # ============================== R3 event study ===============================
    LEADS, LAGS = 5, 5
    es_rows = []
    for okey, ocol in OUTCOMES.items():
        need = ["fips", "year", "state_fips", ocol, "F_trel"] + CONTROLS
        s = df[df.year.between(2012, 2023)][need].copy()
        s["ev"] = s["F_trel"].clip(-LEADS, LAGS)
        s.loc[s.F_trel.isna(), "ev"] = -1
        s = s.drop(columns=["F_trel"]).dropna().rename(columns={c: cl(c) for c in need})
        if len(s) < 500:
            continue
        fml = f"{cl(ocol)} ~ i(ev, ref=-1) + " + " + ".join(cl(c) for c in CONTROLS) + " | fips + year"
        try:
            m = pf.feols(fml, data=s, vcov={"CRV1": "state_fips"})
        except Exception as e:
            print(f"  R3 {okey}: {e}"); continue
        for idx, r in m.tidy().iterrows():
            if not str(idx).startswith("ev::"):
                continue
            e = int(float(str(idx).split("::")[1]))
            b, se = float(r["Estimate"]), float(r["Std. Error"])
            es_rows.append({"model": "R3", "outcome": okey, "event_time": e, "beta": b,
                            "se_state": se, "p_state": float(r["Pr(>|t|)"]),
                            "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se, "N": int(m._N)})
    es = pd.DataFrame(es_rows)
    es.to_csv(out_file(tables_dir, "4a-fsis", f"{TODAY}_R3_event_study.csv"), index=False)
    if len(es):
        pre = es[es.event_time < -1]
        print("\nR3 pre-trend: significant pre-period coefs at 5%")
        print(pre.groupby("outcome").apply(lambda g: f"{int((g.p_state < .05).sum())}/{len(g)}").to_string())

    # ============================== R4 LP-DiD ====================================
    lp_rows = []
    for okey, ocol in OUTCOMES.items():
        pan = df[df.year.between(2012, 2023)][["fips", "year", "state_fips", "F_cohort", ocol] + CONTROLS]
        pan = pan.rename(columns={c: cl(c) for c in pan.columns})
        for arm, ctrls in [("no controls", []), ("step-8 at t-1", [cl(c) for c in CONTROLS])]:
            r = lpdid(pan, cl(ocol), cohort="F_cohort", horizons=range(-LEADS, LAGS + 1),
                      controls=ctrls)
            if r.empty:
                continue
            r.insert(0, "arm", arm); r.insert(0, "outcome", okey); r.insert(0, "model", "R4")
            lp_rows.append(r)
    lp = pd.concat(lp_rows, ignore_index=True) if lp_rows else pd.DataFrame()
    lp.to_csv(out_file(tables_dir, "4a-fsis", f"{TODAY}_R4_lpdid.csv"), index=False)
    if len(lp):
        print("\nR4 LP-DiD, h=0 and pre-trend")
        for (o, a), g in lp.groupby(["outcome", "arm"]):
            h0 = g.loc[g.h == 0, "beta"]
            pr = g[g.h < -1]
            print(f"  {o:26s} {a:14s} h0={h0.iloc[0] if len(h0) else np.nan:+.4f}  "
                  f"pre sig={int((pr.p < .05).sum())}/{len(pr)}  treated={g.n_treated.max()}")
    print(f"\nscript4a fsis COMPLETE -> {OUT} (tag 4a-fsis)")


SECTIONS = {"designs": section_designs, "cohorts": section_cohorts, "species": section_species, "fsis": section_fsis}


def main():
    args = sys.argv[1:]
    only = []
    if "--only" in args:
        for a in args[args.index("--only") + 1:]:
            if a.startswith("--"):
                break
            only.append(a)
    unknown = [o for o in only if o not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    for name in (only or list(SECTIONS)):
        print(f"\n{'=' * 78}\nscript4a-estimates: section {name}\n{'=' * 78}", flush=True)
        SECTIONS[name]()


if __name__ == "__main__":
    main()
