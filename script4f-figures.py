#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4f-figures.py -- the estimator-comparison figures (exhibit script; reads result files, no estimation).

Sections (run in this order; select with --only, e.g. --only triangulation permutation):
  triangulation       forest plot of the estimator triangulation, one figure per design     (was script4zb)
  treatment-by-model  every treatment against every model: one figure per model + one overall (was script4zd)
  permutation         permutation null distributions and p-values                            (was script4zf)
  pretrends-leads     outcome leads for T1, T3, T4, E2, E3 with a dotted straight-line fit    (new 2026-10-26, D-55)
  pretrends-controls  pre-trend coefficients of the 8 selected controls, x-y panels with fits (new, D-55)
  pretrends-raw       raw pre-period paths, treated vs never-treated counties, outcomes        (new, D-55)
  pretrends-controls-raw  the same raw paths for the 8 selected controls                      (new, D-55)
Reads:   triangulation: tables/script4za/*_{overall_estimates,agreement,samples}.csv, tables/script4x/*_outcome_within_sd.csv
         treatment-by-model: tables/script4zc/*_treatment_by_model.csv, tables/script4x/*_outcome_within_sd.csv
         permutation: tables/script4ze/*_permutation_pvalues*.csv and *_null_draws*.csv
         pretrends-*: tables/script4t/*_leads_coefficients.csv, tables/script4a/*_{control_pretrend_lags,control_pretrends}.csv,
                      tables/script4b/*_{pretrend_raw_means,pretrend_fits}.csv   (written by script4b-diagnostics.py --only controls pretrends)
         (--quick reads script4zc_quick / script4ze_quick and writes script4zd_quick / script4zf_quick; triangulation has no quick mode)
Writes:  figs/script4zb/YYYY-MM-DD_triangulation_{first_expansion,first_entry}.{pdf,png}
         figs/script4zd/YYYY-MM-DD_treatment_by_model_{model, all_models}.{pdf,png}
         figs/script4zf/YYYY-MM-DD_permutation_inference.{pdf,png}
         figs/script4f/YYYY-MM-DD_pretrends_{leads_mental_health, leads_crime, controls, raw_<design>_outcomes, raw_<design>_controls}.{pdf,png}
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
from matplotlib.ticker import MaxNLocator
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


# =============================================================== sections pretrends-* (new 2026-10-26, D-55)
# The dotted lines are ordinary least-squares straight lines through the plotted pre-period points, computed in
# script4b-diagnostics.py (pretrend_fits.csv). They describe the drift; the pre-trend tests are in leads_joint_tests.csv.
PRE_FIG_DIR = os.path.join(figs_dir, "script4f")
MENTAL_OUTCOMES = ["Poor MH Days", "Frequent Mental Distress", "Deaths of Despair"]
CRIME_OUTCOMES = ["Aggravated Assault", "Assault, all severities", "Violence index (partial)", "NIBRS curated total"]
DOTTED = dict(ls=":", lw=1.1)


def _fit_line(ax, fits, x_range, colour, **where):
    """Dotted line from the saved slope and intercept; `where` picks the row of pretrend_fits.csv."""
    m = fits
    for k, v in where.items():
        m = m[m[k] == v]
    if m.empty or m.slope.isna().iloc[0]:
        return
    x = np.array(x_range, float)
    ax.plot(x, m.intercept.iloc[0] + m.slope.iloc[0] * x, color=colour, **DOTTED)


def _pre_inputs():
    fits = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4b"), "*_pretrend_fits.csv"))
    return fits


def pretrends_leads():
    """Lead coefficients of the outcome on each treatment format (rows: outcomes, columns: treatments), with a dotted fit."""
    figstyle.apply_style()
    lc = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4t"), "*_leads_coefficients.csv"))
    lc["x"] = np.where(lc.design == "lead test", -lc.lead, lc.lead)          # a lead k years ahead sits at -k
    fits = _pre_inputs()
    for tag, outs, h in (("mental_health", MENTAL_OUTCOMES, 92), ("crime", CRIME_OUTCOMES, 118)):
        fig, axes = figstyle.new_grid(len(outs), len(TREATMENTS), width="double", height_mm=h)
        fig.get_layout_engine().set(rect=(0, 0.11, 1, 0.89))
        for i, o in enumerate(outs):
            for j, t in enumerate(TREATMENTS):
                ax, g = axes[i, j], lc[(lc.treatment == t) & (lc.outcome == o)].sort_values("x")
                col = figstyle.PALETTE[TREATMENT_COLOUR[t]]
                ax.axhline(0, color="black", lw=0.4)
                ax.errorbar(g.x, g.beta, yerr=1.96 * g.se_state, fmt="o", ms=2.6, lw=0.7, capsize=1, color=col)
                _fit_line(ax, fits, [g.x.min(), g.x.max()], col, source="lead_coefficients", design=t, variable=o)
                ax.set_xlim(-8.6, -1.4) if t in ("E2", "E3") else ax.set_xlim(-4.6, -0.4)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if i == 0:
                    ax.set_title(SHORT_TREATMENT[t], fontsize=6.5, pad=2)
                if j == 0:
                    ax.set_ylabel(SHORT_OUTCOME[o], fontsize=6)
                if i == len(outs) - 1:
                    ax.set_xlabel("Years before treatment")
        fig.text(0.005, 0.005, "Coefficient on treatment k years ahead (T1, T3, T4: lead test; plotted at minus k) or on event time -8 to -2 (E2, E3: "
                 "event study), outcome units, 95% interval, SE clustered by state;\ncounty and year fixed effects, 8 pre-committed controls. Dotted line: "
                 "least-squares line through the plotted points (descriptive; tests in leads_joint_tests.csv). Each panel has its own y-axis.\n"
                 "Source: script4b-diagnostics.py (leads, pretrends); script4f-figures.py.", fontsize=5, va="bottom", ha="left", color="#333333")
        print("wrote", *figstyle.save_figure(fig, PRE_FIG_DIR, f"pretrends_leads_{tag}"))


def pretrends_controls():
    """Each selected control as the dependent variable in the event-study design: pre-period coefficients in SD, x-y panels."""
    figstyle.apply_style()
    lag = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4a"), "*_control_pretrend_lags.csv"))
    ver = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4a"), "*_control_pretrends.csv")).set_index("variable")
    fits = _pre_inputs()
    from script4_treatment import COVARIATE_ORDER
    ctrl = COVARIATE_ORDER[:8]
    fig, axes = figstyle.new_grid(2, 4, width="double", height_mm=100)
    fig.get_layout_engine().set(rect=(0, 0.09, 1, 0.91))
    col = figstyle.PALETTE["blue"]
    for ax, c, letter in zip(axes.ravel(), ctrl, "abcdefgh"):
        g = lag[(lag.variable == c) & (lag.lag < -1)].sort_values("lag")
        ax.axhline(0, color="black", lw=0.4)
        ax.errorbar(g.lag, g.estimate / g.outcome_sd, yerr=1.96 * g.se / g.outcome_sd, fmt="o", ms=2.6, lw=0.7, capsize=1, color=col)
        ax.plot([-1], [0], "o", ms=2.6, mfc="white", color=col, mew=0.7)           # reference year, zero by construction
        _fit_line(ax, fits, [g.lag.min(), -1], col, source="control_lag_coefficients", variable=c)
        ax.set_xlim(-6.6, -0.4)
        ax.set_title(f"{ver.loc[c, 'description']} ({ver.loc[c, 'n_pre_significant']} of {ver.loc[c, 'n_pre_coefs']} sig.)", fontsize=6, pad=2, loc="left")
        figstyle.panel_label(ax, letter)
        ax.set_xlabel("Years before first expansion")
        if ax in axes[:, 0]:
            ax.set_ylabel("Coefficient (SD of the control)")
    fig.text(0.005, 0.005, "Each control is the dependent variable of the E2 event study (first expansion), county and year fixed effects, 95% interval, SE "
             "clustered by state; reference year -1 (open marker).\nDotted line: least-squares line through the plotted points (descriptive). "
             "Title: pre-period coefficients significant at 5% out of those estimated.\nSource: script4b-diagnostics.py (controls); script4f-figures.py.",
             fontsize=5, va="bottom", ha="left", color="#333333")
    print("wrote", *figstyle.save_figure(fig, PRE_FIG_DIR, "pretrends_controls"))


def _raw_figures(kind):
    """Raw change since the year before treatment, treated vs never-treated counties, event times -8 to -1, dotted pre-period fits."""
    figstyle.apply_style()
    raw = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4b"), "*_pretrend_raw_means.csv"))
    fits = _pre_inputs()
    ctrl_desc = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4a"), "*_control_pretrends.csv")).set_index("variable").description
    names = {"E2": "first expansion (E2; also the timing of T3 and T4)", "E3": "first entry (E3; also the timing of T1)"}
    comparison = {"E2": "counties that never add or enter", "E3": "counties that never had a large dairy"}
    for d in ("E2", "E3"):
        sub = raw[(raw.design == d) & (raw.kind == kind) & (raw.event_time < 0)]
        labels = [v for v in sub.variable.unique()]
        fig, axes = figstyle.new_grid(2, 4, width="double", height_mm=100)
        fig.get_layout_engine().set(rect=(0, 0.1, 1, 0.9))
        for ax, v, letter in zip(axes.ravel(), labels, "abcdefgh"):
            g = sub[sub.variable == v].sort_values("event_time")
            ax.axhline(0, color="black", lw=0.4)
            for series, colour, ls, lab in (("treated", "vermillion", "-", "treated"), ("comparison", "blue", "-", "comparison")):
                c = figstyle.PALETTE[colour]
                ax.plot(g.event_time, g[series], color=c, lw=0.8, marker="o", ms=2.2, label=lab)
                _fit_line(ax, fits, [g.event_time.min(), -2], c, source="raw_mean", design=d, variable=v, series=series)
            ax.set_xlim(-8.6, -0.4)
            ax.set_title(SHORT_OUTCOME.get(v, ctrl_desc.get(v, v)), fontsize=6, pad=2, loc="left")
            figstyle.panel_label(ax, letter)
            ax.set_xlabel("Years before treatment")
            if ax in axes[:, 0]:
                ax.set_ylabel("Change since year -1")
        axes.ravel()[0].legend(frameon=False, fontsize=5.5, loc="best")
        for ax in axes.ravel()[len(labels):]:
            ax.axis("off")                                              # no empty axes with default ticks
        fig.text(0.005, 0.005, f"Design: {names[d]}, against {comparison[d]}. Mean change in the variable since the year before treatment,\n"
                 "averaged over cohorts weighted by treated counties (cohort-years with fewer than 10 treated or comparison counties are dropped). "
                 "No controls, no fixed effects.\nDotted lines: least-squares lines through event times -8 to -2 (descriptive). "
                 "Source: script4b-diagnostics.py (pretrends); script4f-figures.py.", fontsize=5, va="bottom", ha="left", color="#333333")
        print("wrote", *figstyle.save_figure(fig, PRE_FIG_DIR, f"pretrends_raw_{d}_{kind}s"))


def pretrends_raw():
    _raw_figures("outcome")


def pretrends_controls_raw():
    _raw_figures("control")


SECTIONS = {"triangulation": triangulation, "treatment-by-model": treatment_by_model, "permutation": permutation,
            "pretrends-leads": pretrends_leads, "pretrends-controls": pretrends_controls, "pretrends-raw": pretrends_raw,
            "pretrends-controls-raw": pretrends_controls_raw}
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
