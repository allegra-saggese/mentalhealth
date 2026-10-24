"""script4zf-permutation-figure.py -- permutation null distributions and p-values (exhibit script; reads results, no estimation).

Panels a, b: null distribution of the TWFE coefficient when first-treated years are shuffled within state (first expansion), with the
observed coefficient marked, for the two confirmatory mental-health outcomes (D-42). Panel c: permutation p-value of every outcome under
first expansion and first entry; the dotted line marks 0.05.
Reads:   Data/output/tables/script4ze/*_permutation_pvalues*.csv and *_null_draws*.csv   (--quick reads script4ze_quick)
Writes:  Data/output/figs/script4zf/YYYY-MM-DD_permutation_inference.{pdf,png}
Run:     python3 script4zf-permutation-figure.py        Style: figstyle.py (Nature/Science), checked with scripts/figure_check.py.
"""
import glob
import sys

import numpy as np
import pandas as pd

import figstyle
from script4_treatment import OUTCOMES, figs_dir, tables_dir, os

QUICK = "--quick" in sys.argv
IN = os.path.join(tables_dir, "script4ze" + ("_quick" if QUICK else ""))
OUT_DIR = os.path.join(figs_dir, "script4zf" + ("_quick" if QUICK else ""))
EXPANSION, ENTRY = "first expansion", "first entry (clean controls)"
SHORT = {"Poor MH Days": "Poor MH Days", "Frequent Mental Distress": "Freq. mental distress", "Deaths of Despair": "Deaths of despair",
         "Aggravated Assault": "Aggravated assault", "Assault, all severities": "Assault, all", "Violence index (partial)": "Violence index",
         "NIBRS curated total": "NIBRS total"}


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
ax_c.set_yticks(range(len(outs)), [SHORT[o] for o in outs])
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
