"""script4zd-treatment-by-model-figures.py -- every treatment against every model (exhibit script; reads results, no estimation).

Format A: one figure per model; a panel per outcome; a line per treatment.
Format B: one figure; a panel per model; colour = treatment (key, so lines need no labels); a row per outcome.
Estimates are overall post-treatment effects with 95% intervals, divided by the outcome's within-county SD (script4x). A marker is filled
when the interval excludes zero. TWFE is per one unit of the treatment; dCDH for T1, T3, T4 is the effect of the treatment changes
counties experienced (D-46), not per unit; the units of T1, T3, T4, E2, E3 differ, so lines are not comparable in magnitude.
Reads:   Data/output/tables/script4zc/*_treatment_by_model.csv, script4x/*_outcome_within_sd.csv   (--quick reads script4zc_quick)
Writes:  Data/output/figs/script4zd/YYYY-MM-DD_treatment_by_model_{model, all_models}.{pdf,png}
Run:     python3 script4zd-treatment-by-model-figures.py        Style: figstyle.py (Nature/Science), checked with scripts/figure_check.py.
"""
import re
import sys

import numpy as np
import pandas as pd

import figstyle
from functions import latest_file_glob
from script4_treatment import OUTCOMES, figs_dir, tables_dir, os

QUICK = "--quick" in sys.argv
MODELS = ["TWFE", "Callaway-Sant'Anna", "LP-DiD", "Sun-Abraham", "Imputation", "dCDH", "Stacked DiD"]
TREATMENTS = ["E2", "E3", "T1", "T3", "T4"]
SHORT = {"E2": "E2 first expansion", "E3": "E3 first entry", "T1": "T1 presence", "T3": "T3 per 10k", "T4": "T4 log count"}
COLOUR = {"E2": "vermillion", "E3": "black", "T1": "blue", "T3": "green", "T4": "purple"}   # five hues that stay distinct
OUT_DIR = os.path.join(figs_dir, "script4zd" + ("_quick" if QUICK else ""))
SHORT_OUTCOME = {"Poor MH Days": "Poor MH Days", "Frequent Mental Distress": "Freq. mental distress", "Deaths of Despair": "Deaths of despair",
                 "Aggravated Assault": "Aggravated assault", "Assault, all severities": "Assault, all", "Violence index (partial)": "Violence index",
                 "NIBRS curated total": "NIBRS total"}

res = pd.read_csv(latest_file_glob(os.path.join(tables_dir, "script4zc" + ("_quick" if QUICK else "")), "*_treatment_by_model.csv"))
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
    c = figstyle.PALETTE[COLOUR[row.treatment_id]]
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
            ax.plot([0.02], [yy], "o", color=figstyle.PALETTE[COLOUR[tid]], ms=3.5, transform=ax.transAxes, clip_on=False)
            ax.text(0.07, yy, SHORT[tid], fontsize=5.8, va="center", transform=ax.transAxes)
    for k, line in enumerate(lines):
        ax.text(0.0, 0.10 - 0.07 * k, line, fontsize=5.5, va="top", transform=ax.transAxes, color="#444444")


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
        ax.set_yticks(range(k), [SHORT[t] for t in treats])
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
