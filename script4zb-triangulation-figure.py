"""script4zb-triangulation-figure.py -- forest plots of the estimator triangulation (exhibit script; reads results, no estimation).

One figure per design: a panel per outcome, a row per estimator, overall post-treatment ATT with its 95% interval in
within-county standard deviations of the outcome (so panels share a scale of meaning), the Callaway-Sant'Anna 95%
interval as a shaded band, a filled marker when the estimator agrees with CS under D-45 (same sign, inside the band).
Reads:   Data/output/tables/script4za/*_{overall_estimates,agreement,samples}.csv, script4x/*_outcome_within_sd.csv
Writes:  Data/output/figs/script4zb/YYYY-MM-DD_triangulation_{first_expansion,first_entry}.{pdf,png}
Run:     python3 script4zb-triangulation-figure.py        Style: figstyle.py (Nature/Science), checked with /figure-check.
"""
import os

import numpy as np
import pandas as pd

import figstyle
from functions import latest_file_glob
from script4_treatment import OUTCOMES, figs_dir, tables_dir

CS, TW = "Callaway-Sant'Anna (benchmark)", "TWFE (static, reference)"
ROWS = [(CS, "Callaway-Sant'Anna", "black"), (TW, "TWFE (reference)", "grey"),
        ("LP-DiD", "LP-DiD", "blue"), ("Sun-Abraham", "Sun-Abraham", "vermillion"),
        ("Imputation (did2s)", "Imputation", "green"), ("de Chaisemartin-D'Haultfoeuille", "dCDH", "orange"),
        ("Stacked DiD", "Stacked", "purple")]
COLOUR = {**figstyle.PALETTE, "grey": "#8c8c8c"}
DESIGNS = {"first expansion": "first_expansion", "first entry (clean controls)": "first_entry"}

za = os.path.join(tables_dir, "script4za")
overall = pd.read_csv(latest_file_glob(za, "*_overall_estimates.csv"))
agree = pd.read_csv(latest_file_glob(za, "*_agreement.csv"))
samples = pd.read_csv(latest_file_glob(za, "*_samples.csv"))
within_sd = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4x"), "*_outcome_within_sd.csv"),
                        index_col=0)["within_county_sd"]
figstyle.apply_style()


def panel(ax, data, ag, sd, outcome, letter, n_text):
    cs = data[data.estimator == CS].iloc[0]
    ax.axvspan(cs.ci_lo / sd, cs.ci_hi / sd, color=COLOUR["sky"], alpha=0.25, lw=0)
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
                    capsize=1.2, color=COLOUR[colour], mfc=COLOUR[colour] if filled else "white", mew=0.8)
    ax.set_yticks(range(len(ROWS)), [lab for _, lab, _ in ROWS])
    ax.set_ylim(len(ROWS) - 0.4, -0.6)
    ax.set_xlabel("Effect (within-county SD of the outcome)")
    ax.set_title(outcome, loc="left", fontsize=6.5, pad=2)
    figstyle.panel_label(ax, letter)
    ax.text(1.0, 1.0, n_text, transform=ax.transAxes, ha="right", va="bottom", fontsize=5, color="#555555")   # above the axes: no overlap


for design, tag in DESIGNS.items():
    fig, axes = figstyle.new_grid(4, 2, width="double", height_mm=185.0)
    fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.925))
    flat = list(axes.ravel())
    for i, outcome in enumerate(OUTCOMES):
        d = overall[(overall.design == design) & (overall.outcome == outcome)]
        ag = agree[(agree.design == design) & (agree.outcome == outcome)]
        s = samples[(samples.design == design) & (samples.outcome == outcome)].iloc[0]
        panel(flat[i], d, ag, float(within_sd[outcome]), outcome, "abcdefg"[i],
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
