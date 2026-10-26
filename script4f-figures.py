#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4f-figures.py -- the estimator-comparison figures (exhibit script; reads result files, no estimation).

Sections (run in this order; select with --only, e.g. --only triangulation permutation):
  triangulation       forest plot of the estimator triangulation, one figure per design     (was script4zb)
  treatment-by-model  every treatment against every model: one figure per model + one overall (was script4zd)
  permutation         permutation null distributions and p-values                            (was script4zf)
Reads:   triangulation: tables/script4za/*_{overall_estimates,agreement,samples}.csv, tables/script4x/*_outcome_within_sd.csv
         treatment-by-model: tables/script4zc/*_treatment_by_model.csv, tables/script4x/*_outcome_within_sd.csv
         permutation: tables/script4ze/*_permutation_pvalues*.csv and *_null_draws*.csv
         (--quick reads script4zc_quick / script4ze_quick and writes script4zd_quick / script4zf_quick; triangulation has no quick mode)
Writes:  figs/script4zb/YYYY-MM-DD_triangulation_{first_expansion,first_entry}.{pdf,png}
         figs/script4zd/YYYY-MM-DD_treatment_by_model_{model, all_models}.{pdf,png}
         figs/script4zf/YYYY-MM-DD_permutation_inference.{pdf,png}
Run:     PYTHONHASHSEED=0 python3 script4f-figures.py [--only triangulation|treatment-by-model|permutation ...] [--quick]
Style:   figstyle.py (Nature/Science), checked with scripts/figure_check.py. Output folders keep their old names so no reader moves.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

import figstyle
from functions import latest_file_glob
from script4_treatment import OUTCOMES, figs_dir, tables_dir

# Outcome labels shared by the treatment-by-model and permutation figures.

SHORT_OUTCOME = {"Poor MH Days": "Poor MH Days", "Frequent Mental Distress": "Freq. mental distress", "Deaths of Despair": "Deaths of despair",
                 "Aggravated Assault": "Aggravated assault", "Assault, all severities": "Assault, all", "Violence index (partial)": "Violence index",
                 "NIBRS curated total": "NIBRS total"}


# =============================================================== section triangulation (was script4zb)
CS, TW = "Callaway-Sant'Anna (benchmark)", "TWFE (static, reference)"
ROWS = [(CS, "Callaway-Sant'Anna", "black"), (TW, "TWFE (reference)", "grey"),
        ("LP-DiD", "LP-DiD", "blue"), ("Sun-Abraham", "Sun-Abraham", "vermillion"),
        ("Imputation (did2s)", "Imputation", "green"), ("de Chaisemartin-D'Haultfoeuille", "dCDH", "orange"),
        ("Stacked DiD", "Stacked", "purple")]
ESTIMATOR_COLOUR = {**figstyle.PALETTE, "grey": "#8c8c8c"}
DESIGNS = {"first expansion": "first_expansion", "first entry (clean controls)": "first_entry"}


def triangulation_panel(ax, data, ag, sd, outcome, letter, n_text):
    cs = data[data.estimator == CS].iloc[0]
    ax.axvspan(cs.ci_lo / sd, cs.ci_hi / sd, color=ESTIMATOR_COLOUR["sky"], alpha=0.25, lw=0)
    ax.axvline(0, color="black", lw=0.5, ls=":")
    for k, (name, _, colour) in enumerate(ROWS):
        r = data[data.estimator == name]
        if r.empty:
            continue
        r = r.iloc[0]
        a = ag[ag.estimator == name]
        filled = name in (CS,) or (len(a) and bool(a.agrees.iloc[0]))
        marker = "D" if name == CS else ("s" if name == TW else "o")
        ax.errorbar(r.est / sd, k, xerr=[[(r.est - r.ci_lo) / sd], [(r.ci_hi - r.est) / sd]], fmt=marker, ms=3.2, lw=0.8,
                    capsize=1.2, color=ESTIMATOR_COLOUR[colour], mfc=ESTIMATOR_COLOUR[colour] if filled else "white", mew=0.8)
    ax.set_yticks(range(len(ROWS)), [lab for _, lab, _ in ROWS])
    ax.set_ylim(len(ROWS) - 0.4, -0.6)
    ax.set_xlabel("Effect (within-county SD of the outcome)")
    ax.set_title(outcome, loc="left", fontsize=6.5, pad=2)
    figstyle.panel_label(ax, letter)
    ax.text(1.0, 1.0, n_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=5, color="#555555")   # above the axes: no overlap



def triangulation():
    """One forest-plot figure per design: a panel per outcome, a row per estimator, in within-county SD of the outcome."""
    za = os.path.join(tables_dir, "script4za")
    overall = pd.read_csv(latest_file_glob(za, "*_overall_estimates.csv"))
    agree = pd.read_csv(latest_file_glob(za, "*_agreement.csv"))
    samples = pd.read_csv(latest_file_glob(za, "*_samples.csv"))
    within_sd = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4x"), "*_outcome_within_sd.csv"),
                            index_col=0)["within_county_sd"]
    figstyle.apply_style()

    for design, tag in DESIGNS.items():
        fig, axes = figstyle.new_grid(4, 2, width="double", height_mm=185.0)
        fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.925))
        flat = list(axes.ravel())
        for i, outcome in enumerate(OUTCOMES):
            d = overall[(overall.design == design) & (overall.outcome == outcome)]
            ag = agree[(agree.design == design) & (agree.outcome == outcome)]
            s = samples[(samples.design == design) & (samples.outcome == outcome)].iloc[0]
            triangulation_panel(flat[i], d, ag, float(within_sd[outcome]), outcome, "abcdefg"[i],
                  f"{int(s.counties):,} counties, {int(s.treated_counties)} treated")
        key = flat[-1]
        key.axis("off")
        pm = agree[(agree.design == design) & (agree.outcome == "Poor MH Days")]
        key.text(0.0, 0.95, "How to read", fontsize=6.5, fontweight="bold", va="top", transform=key.transAxes)
        key.text(0.0, 0.80, "Filled marker: agrees with Callaway-Sant'Anna\n(same sign, inside its 95% interval; D-45).\n"
                 "Open marker: does not agree.\nShaded band: Callaway-Sant'Anna 95% interval.\nDiamond: benchmark. Square: TWFE, for reference.",
                 fontsize=5.8, va="top", transform=key.transAxes, linespacing=1.4)
        key.text(0.0, 0.18, f"Poor MH Days: {int(pm.agrees.sum())} of {len(pm)} estimators agree.", fontsize=6, va="bottom",
                 transform=key.transAxes)
        fig.text(0.005, 0.005,
                 "Overall post-treatment ATT with 95% interval, one common complete-case sample per outcome; standard errors clustered by state "
                 "(CS: state bootstrap,\n300 draws); imputation drops counties with a single untreated observation; LP-DiD and dCDH stop at event "
                 "time 8. Design: " + design + ".\nSource: script4za-estimator-triangulation.py; agreement rule D-45; scale: within-county SD "
                 "(script4x).", fontsize=5, va="bottom", ha="left", color="#333333")
        pdf, png = figstyle.save_figure(fig, os.path.join(figs_dir, "script4zb"), f"triangulation_{tag}")
        print("wrote", pdf, png)


# =============================================================== section treatment-by-model (was script4zd)
MODELS = ["TWFE", "Callaway-Sant'Anna", "LP-DiD", "Sun-Abraham", "Imputation", "dCDH", "Stacked DiD"]
TREATMENTS = ["E2", "E3", "T1", "T3", "T4"]
SHORT_TREATMENT = {"E2": "E2 first expansion", "E3": "E3 first entry", "T1": "T1 presence", "T3": "T3 per 10k", "T4": "T4 log count"}
TREATMENT_COLOUR = {"E2": "vermillion", "E3": "black", "T1": "blue", "T3": "green", "T4": "purple"}   # five hues that stay distinct



def robust_xlim(rows):
    """Axis limits from the bulk of the intervals: an interval more than 4x the median half-width is drawn clipped."""
    lo, hi = np.array([r.ci_lo for r in rows]), np.array([r.ci_hi for r in rows])
    half = (hi - lo) / 2
    keep = half <= 4 * np.median(half)
    a, b = min(lo[keep].min(), 0.0), max(hi[keep].max(), 0.0)
    pad = 0.06 * (b - a)
    return a - pad, b + pad


def draw(ax, y, row, xlim):
    """One estimate on row position y; an interval beyond the axis is clipped with an arrow and its values are printed."""
    c = figstyle.PALETTE[TREATMENT_COLOUR[row.treatment_id]]
    filled = row.ci_lo > 0 or row.ci_hi < 0
    clipped = row.ci_lo < xlim[0] or row.ci_hi > xlim[1]
    if not clipped:
        ax.errorbar(row.est, y, xerr=[[row.est - row.ci_lo], [row.ci_hi - row.est]], fmt="o", ms=3.2, lw=0.8, capsize=1.2,
                    color=c, mfc=c if filled else "white", mew=0.8)
        return
    ax.plot([max(row.ci_lo, xlim[0]), min(row.ci_hi, xlim[1])], [y, y], color=c, lw=0.8)
    if xlim[0] <= row.est <= xlim[1]:
        ax.plot(row.est, y, "o", ms=3.2, color=c, mfc=c if filled else "white", mew=0.8)
    if row.ci_hi > xlim[1]:
        ax.plot(xlim[1], y, ">", ms=3.2, color=c, clip_on=False)
    if row.ci_lo < xlim[0]:
        ax.plot(xlim[0], y, "<", ms=3.2, color=c, clip_on=False)
    ax.text(xlim[1], y - 0.3, f"{row.est:.2f} ({row.ci_lo:.2f}, {row.ci_hi:.2f})", fontsize=4.5, color=c, ha="right", va="bottom")


def style(ax, letter, title, n_rows, xlim):
    ax.set_xlim(xlim)
    ax.axvline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Effect (within-county SD of the outcome)")
    ax.set_title(title, loc="left", fontsize=6.5, pad=2)
    figstyle.panel_label(ax, letter)
    ax.set_ylim(n_rows - 0.4, -0.6)


def key_panel(ax, lines, colours=True):
    ax.axis("off")
    ax.text(0.0, 0.97, "How to read", fontsize=6.5, fontweight="bold", va="top", transform=ax.transAxes)
    ax.text(0.0, 0.85, "Filled marker: the 95% interval excludes zero.\nOpen marker: it includes zero.\nArrow: interval runs past the axis (values printed).", fontsize=5.8, va="top",
            transform=ax.transAxes, linespacing=1.4)
    if colours:
        for k, tid in enumerate(TREATMENTS):
            yy = 0.58 - 0.09 * k
            ax.plot([0.02], [yy], "o", color=figstyle.PALETTE[TREATMENT_COLOUR[tid]], ms=3.5, transform=ax.transAxes, clip_on=False)
            ax.text(0.07, yy, SHORT_TREATMENT[tid], fontsize=5.8, va="center", transform=ax.transAxes)
    for k, line in enumerate(lines):
        ax.text(0.0, 0.10 - 0.07 * k, line, fontsize=5.5, va="top", transform=ax.transAxes, color="#444444")




def treatment_by_model(quick=False):
    """Format A: one figure per model (panel per outcome, line per treatment). Format B: one figure, a panel per model."""
    OUT_DIR = os.path.join(figs_dir, "script4zd" + ("_quick" if quick else ""))

    res = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4zc" + ("_quick" if quick else "")), "*_treatment_by_model.csv"))
    sd = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4x"), "*_outcome_within_sd.csv"), index_col=0)["within_county_sd"]
    for c in ("est", "ci_lo", "ci_hi"):
        res[c] = res[c] / res.outcome.map(sd)                       # in within-county SDs of the outcome
    chg = res[(res.model == "dCDH") & res.note.notna()].copy()
    chg["avg"] = chg.note.str.extract(r"average first change ([+-]?[\d.]+)").astype(float)
    SWITCH = chg.groupby("treatment_id").avg.mean().to_dict()      # average first change by treatment, for the note
    figstyle.apply_style()
    NOTE = ("Overall post-treatment effect with 95% interval, in within-county SD of the outcome; a filled marker means the interval excludes zero. "
            "TWFE: per one unit of the treatment.\ndCDH for T1, T3, T4: effect of the treatment changes counties experienced (average first change: "
            + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(SWITCH.items())) + "), not per unit. Cohort-based models exist for E2 and E3 only. "
            "Units of T1, T3, T4, E2, E3 differ, so\nline-to-line magnitudes are not comparable. Sources: script4zc (this table), script4za, script4a; scale: script4x.")


    # ---------------- format A: one figure per model, a panel per outcome, a line per treatment
    for model in MODELS:
        m = res[res.model == model]
        treats = [t for t in TREATMENTS if t in set(m.treatment_id)]
        if not treats:
            continue
        k = len(treats)
        fig, axes = figstyle.new_grid(4, 2, width="double", height_mm=min(4 * (5 * k + 24) + 22, 240))
        fig.get_layout_engine().set(rect=(0, 0.085, 1, 0.915))
        flat = list(axes.ravel())
        for i, outcome in enumerate(OUTCOMES):
            ax = flat[i]
            sub = m[(m.outcome == outcome) & m.treatment_id.isin(treats)]
            xl = robust_xlim(list(sub.itertuples()))
            for j, tid in enumerate(treats):
                r = m[(m.treatment_id == tid) & (m.outcome == outcome)]
                if len(r):
                    draw(ax, j, r.iloc[0], xl)
            ax.set_yticks(range(k), [SHORT_TREATMENT[t] for t in treats])
            style(ax, "abcdefg"[i], outcome, k, xl)
        key_panel(flat[-1], [f"Model: {model}"], colours=False)
        fig.text(0.005, 0.005, NOTE, fontsize=5, va="bottom", ha="left", color="#333333")
        stem = "treatment_by_model_" + re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
        print("wrote", *figstyle.save_figure(fig, OUT_DIR, stem))

    # ---------------- format B: one figure, a panel per model, colour = treatment, a row per outcome
    fig, axes = figstyle.new_grid(4, 2, width="double", height_mm=232.0)
    fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.925))
    flat = list(axes.ravel())
    outs = list(OUTCOMES)
    for i, model in enumerate(MODELS):
        ax = flat[i]
        m = res[res.model == model]
        treats = [t for t in TREATMENTS if t in set(m.treatment_id)]
        offsets = np.linspace(-0.3, 0.3, len(TREATMENTS))
        xl = robust_xlim(list(m[m.treatment_id.isin(treats)].itertuples()))
        for o, outcome in enumerate(outs):
            for tid in treats:
                r = m[(m.treatment_id == tid) & (m.outcome == outcome)]
                if len(r):
                    draw(ax, o + offsets[TREATMENTS.index(tid)], r.iloc[0], xl)
        ax.set_yticks(range(len(outs)), [SHORT_OUTCOME[o] for o in outs] if i % 2 == 0 else [""] * len(outs))
        style(ax, "abcdefg"[i], model, len(outs), xl)
    key_panel(flat[-1], ["Rows: outcomes. Each row has up to five", "colours, one per treatment."], colours=True)
    fig.text(0.005, 0.005, NOTE, fontsize=5, va="bottom", ha="left", color="#333333")
    print("wrote", *figstyle.save_figure(fig, OUT_DIR, "treatment_by_model_all_models"))


# =============================================================== section permutation (was script4zf)
EXPANSION, ENTRY = "first expansion", "first entry (clean controls)"



def permutation(quick=False):
    """Panels a, b: null distributions of the TWFE coefficient for the two confirmatory outcomes; panel c: permutation p-values."""
    IN = os.path.join(tables_dir, "script4ze" + ("_quick" if quick else ""))
    OUT_DIR = os.path.join(figs_dir, "script4zf" + ("_quick" if quick else ""))

    def newest(pattern):
        """Concatenate the newest dated file per design suffix (the two designs may be written by separate runs)."""
        files = sorted(glob.glob(os.path.join(IN, pattern)))
        dates = sorted({os.path.basename(f)[:10] for f in files})
        return pd.concat([pd.read_csv(f) for f in files if os.path.basename(f).startswith(dates[-1])], ignore_index=True)


    pv = newest("*_permutation_pvalues*.csv")
    nd = newest("*_null_draws*.csv")
    figstyle.apply_style()
    fig, axes = figstyle.new_grid(2, 2, width="double", height_mm=120.0)
    fig.get_layout_engine().set(rect=(0, 0.1, 1, 0.9))
    ax_a, ax_b, ax_c, ax_note = axes.ravel()

    for ax, outcome, letter in ((ax_a, "Poor MH Days", "a"), (ax_b, "Frequent Mental Distress", "b")):
        null = nd[(nd.design == EXPANSION) & (nd.outcome == outcome)].coef.to_numpy()
        row = pv[(pv.design == EXPANSION) & (pv.outcome == outcome)].iloc[0]
        ax.hist(null, bins=40, color="#BBBBBB", edgecolor="white", lw=0.3)
        ax.axvline(row.observed_coef, color=figstyle.PALETTE["vermillion"], lw=1.0)
        ax.axvline(0, color="black", lw=0.5, ls=":")
        ax.set_xlabel("TWFE coefficient (" + ("days per month" if outcome == "Poor MH Days" else "per 100,000") + ")")
        ax.set_ylabel("Permutations")
        ax.set_title(outcome, loc="left", fontsize=6.5, pad=2)
        ax.text(0.03, 0.95, f"Observed {row.observed_coef:.3g}\nPermutation mean {null.mean():.3g}\np = {row.p_perm_coef:.3f}", transform=ax.transAxes, ha="left", va="top", bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
                fontsize=5.8, color=figstyle.PALETTE["vermillion"])
        figstyle.panel_label(ax, letter)

    outs = list(OUTCOMES)
    for k, (design, colour, label) in enumerate(((EXPANSION, "vermillion", "First expansion (E2)"), (ENTRY, "black", "First entry (E3)"))):
        sub = pv[pv.design == design].set_index("outcome")
        ys = [i + (-0.15 if k == 0 else 0.15) for i, o in enumerate(outs) if o in sub.index]
        ps = [sub.loc[o, "p_perm_coef"] for o in outs if o in sub.index]
        ax_c.plot(ps, ys, "o", ms=3.2, color=figstyle.PALETTE[colour], mfc=figstyle.PALETTE[colour] if False else "white", mew=0.8,
                  label=label)
        filled = [i for i, p in enumerate(ps) if p <= 0.05]
        ax_c.plot([ps[i] for i in filled], [ys[i] for i in filled], "o", ms=3.2, color=figstyle.PALETTE[colour])
    ax_c.axvline(0.05, color="black", lw=0.5, ls=":")
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.006, 1.3)
    ax_c.set_yticks(range(len(outs)), [SHORT_OUTCOME[o] for o in outs])
    ax_c.set_ylim(len(outs) - 0.4, -0.6)
    ax_c.set_xlabel("Permutation p-value (two-sided, log scale)")
    figstyle.panel_label(ax_c, "c")
    ax_note.axis("off")
    ax_note.text(0.0, 0.95, "How to read", fontsize=6.5, fontweight="bold", va="top", transform=ax_note.transAxes)
    ax_note.text(0.0, 0.82, "Grey bars: coefficients obtained when each state's first-treated\nyears are shuffled among its treated counties "
                 "(never-treated\ncounties fixed). Red line: the observed coefficient. The p-value\nis the share of shuffles at least as large in absolute value.\n"
                 "The permutation mean is not zero, so the test compares the observed\ncoefficient with this distribution, not with zero.\nPanel c: filled marker, p <= 0.05; open marker, p > 0.05.", fontsize=5.6, va="top", transform=ax_note.transAxes, linespacing=1.4)
    n_draws, n_states = int(pv.draws.iloc[0]), int(pv.states_with_shuffle.min())
    fig.text(0.005, 0.005, f"TWFE with county and year fixed effects and the 8 pre-committed controls; samples as in script4za. {n_draws:,} permutations, seed "
             f"{int(pv.seed.iloc[0])}; p = (1 + #{{|null| >= |observed|}}) / (1 + {n_draws:,}).\nTests the sharp null of no effect in any county, conditional on each state's "
             "treated counties and cohorts; only states with two or more different cohorts can be shuffled.\nSources: script4ze (this table), script4za (samples).",
             fontsize=5, va="bottom", ha="left", color="#333333")
    h, l = ax_c.get_legend_handles_labels()
    ax_note.legend(h, l, loc="lower left", fontsize=5.8, frameon=False)
    print("wrote", *figstyle.save_figure(fig, OUT_DIR, "permutation_inference"))


SECTIONS = {"triangulation": triangulation, "treatment-by-model": treatment_by_model, "permutation": permutation}
QUICK_SECTIONS = {"treatment-by-model", "permutation"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(SECTIONS), help="sections to run (default: all, in file order)")
    ap.add_argument("--quick", action="store_true", help="read the script4zc_quick / script4ze_quick results (no effect on triangulation)")
    args = ap.parse_args()
    for name in args.only or SECTIONS:
        print(f"=== {name}")
        SECTIONS[name](quick=args.quick) if name in QUICK_SECTIONS else SECTIONS[name]()


if __name__ == "__main__":
    main()
