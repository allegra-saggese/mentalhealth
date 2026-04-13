#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script2h-threshold-presence.py

Motivated by the theory that effects on mental health / deaths of despair stem
from LARGE CAFO presence and slaughterhouse operations specifically — not simply
from a higher count of any CAFO operation.

The analysis is non-linear by design: we do NOT assume more operations → worse
outcomes. Instead we test:
  (a) Does presence of large operations (binary) predict worse outcomes?
  (b) Is there a dose-response threshold by count tier?
  (c) Does the signal concentrate in large-size operations rather than small?
  (d) Does FSIS slaughterhouse/processing presence drive the effect?

Outputs saved to: Dropbox/Mental/Data/merged/figs/threshold-presence/

Figures:
  O_presence_vs_absence.png   — Mean outcomes ± 95% CI: large CAFO presence (0/1)
                                 and FSIS presence (0/1), side by side
  P_large_cafo_vs_outcomes.png — Binned scatter: log(large CAFO per 10k + 1) vs
                                  all 6 outcomes (compare with A1 for total CAFO)
  Q_dose_response_tiers.png   — Outcomes by large CAFO count tier (0 / 1–3 / 4–10 / 10+)
                                  and by FSIS count tier
  R_small_vs_large_scatter.png — Side-by-side binned scatter: small CAFO (left)
                                  vs large CAFO (right) for key outcomes — tests
                                  whether the signal concentrates in large ops
"""

from packages import *
from functions import *
import scipy.stats as stats

# ── Directories ──────────────────────────────────────────────────────────────
merged_dir = os.path.join(db_data, "merged")
out_dir    = os.path.join(db_data, "merged", "figs", "threshold-presence")
os.makedirs(out_dir, exist_ok=True)
today_str  = date.today().strftime("%Y-%m-%d")

# ── Load panel ───────────────────────────────────────────────────────────────
df = pd.read_csv(latest_file_glob(merged_dir, "*_full_merged.csv"), low_memory=False)
print(f"Panel loaded: {len(df):,} rows | {df['fips'].nunique():,} counties")

POP_COL     = "population"
PANEL_YEARS = (2010, 2015)   # primary MH window
FSIS_YEARS  = (2017, 2023)   # FSIS availability window

# ── Analysis windows ─────────────────────────────────────────────────────────
df_panel = df[df["year"].between(*PANEL_YEARS)].copy()
df_fsis  = df[df["year"].between(*FSIS_YEARS)].copy()

# ── Outcomes ─────────────────────────────────────────────────────────────────
OUTCOMES = {
    "Poor Mental Health Days\n(days/month)":       "poor_mental_health_days",
    "Excessive Drinking\n(per 100k)":              "excessive_drinking_per100k",
    "Violent Crime\n(per 100k)":                   "violent_crime",
    "Homicides\n(per 100k)":                       "homicides",
    "Aggravated Assault\n(per 100k)":              "aggravated_assault_per100k",
    "Deaths of Despair\n(crude rate per 100k)":    "crude_rate_despair",
}
FOCUS_OUTCOMES = {
    "Deaths of Despair\n(per 100k)": "crude_rate_despair",
    "Poor Mental Health Days\n(days/month)": "poor_mental_health_days",
    "Aggravated Assault\n(per 100k)": "aggravated_assault_per100k",
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def log_per10k(series, pop):
    x   = pd.to_numeric(series, errors="coerce")
    pop = pd.to_numeric(pop,    errors="coerce").replace(0, np.nan)
    return np.log1p((x / pop) * 10_000)

def mean_ci(series, alpha=0.05):
    """Return (mean, lower_95_ci, upper_95_ci) for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return np.nan, np.nan, np.nan
    m  = s.mean()
    se = stats.sem(s)
    h  = se * stats.t.ppf(1 - alpha / 2, df=len(s) - 1)
    return m, m - h, m + h

def binned_scatter_ax(ax, x_vals, y_vals, label_x, label_y,
                      n_bins=20, color="#2166ac", title=None):
    """Binned-mean scatter with 95% CI error bars and OLS fit line."""
    x = pd.Series(x_vals, dtype=float)
    y = pd.Series(y_vals, dtype=float)
    mask = x.notna() & y.notna()
    x, y = x[mask].values, y[mask].values
    n = int(mask.sum())

    if n < n_bins * 3:
        ax.text(0.5, 0.5, f"Insufficient data\n(n={n:,})",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="grey")
        ax.set_xlabel(label_x, fontsize=8)
        ax.set_ylabel(label_y, fontsize=8)
        return

    bins   = pd.qcut(x, q=n_bins, duplicates="drop")
    df_bin = pd.DataFrame({"x": x, "y": y, "bin": bins})
    grp    = df_bin.groupby("bin", observed=True)
    x_mid  = grp["x"].mean()
    y_mean = grp["y"].mean()
    y_se   = grp["y"].sem()
    n_b    = grp["y"].count()
    t_crit = stats.t.ppf(0.975, df=np.maximum(n_b - 1, 1))
    y_ci   = y_se * t_crit

    ax.errorbar(x_mid, y_mean, yerr=y_ci, fmt="o", color=color,
                ecolor=color, alpha=0.85, markersize=5, capsize=3,
                elinewidth=1.2, label="Bin mean ± 95% CI")

    # OLS fit
    from numpy.polynomial import polynomial as P
    mask_fit = np.isfinite(x_mid) & np.isfinite(y_mean)
    if mask_fit.sum() >= 3:
        coef = np.polyfit(x_mid[mask_fit], y_mean[mask_fit], 1)
        x_line = np.linspace(x_mid[mask_fit].min(), x_mid[mask_fit].max(), 100)
        ax.plot(x_line, np.polyval(coef, x_line), "--", color=color, alpha=0.6,
                linewidth=1.5, label=f"OLS β={coef[0]:.3f}")
        ax.legend(fontsize=7, loc="best")

    ax.set_xlabel(label_x, fontsize=8)
    ax.set_ylabel(label_y, fontsize=8)
    ax.set_title(title or "", fontsize=9)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
BLUE  = "#2166ac"
GREEN = "#1b7837"
RED   = "#d6604d"
TEAL  = "#41b6c4"


# =============================================================================
# FIGURE O: Presence vs. absence
# Mean outcome ± 95% CI for counties with/without large CAFO presence,
# and with/without any FSIS establishment. One panel per presence variable.
# =============================================================================
# Build binary FSIS presence
df_panel["fsis_any"] = (
    pd.to_numeric(df_panel.get("n_unique_establishments_fsis", np.nan), errors="coerce")
    .gt(0)
    .astype("Int64")
)
df_fsis["fsis_any"] = (
    pd.to_numeric(df_fsis.get("n_unique_establishments_fsis", np.nan), errors="coerce")
    .gt(0)
    .astype("Int64")
)
df_fsis["fsis_slaughter_any"] = (
    pd.to_numeric(df_fsis.get("n_slaughterhouse_present_establishments_fsis", np.nan), errors="coerce")
    .gt(0)
    .astype("Int64")
)

_PRESENCE_SPECS = [
    # (label, binary_col, data_df, group_labels, color_0, color_1, window_str)
    ("Any Large CAFO\n(county presence)",
     "any_large_cafo", df_panel, ["No large CAFO", "Large CAFO present"],
     "#aec7e8", GREEN, f"{PANEL_YEARS[0]}–{PANEL_YEARS[1]}"),
    ("Any FSIS Establishment\n(slaughter or processing)",
     "fsis_any", df_fsis, ["No FSIS", "FSIS present"],
     "#aec7e8", TEAL, f"{FSIS_YEARS[0]}–{FSIS_YEARS[1]}"),
    ("Any FSIS Slaughterhouse\n(slaughter specifically)",
     "fsis_slaughter_any", df_fsis, ["No slaughterhouse", "Slaughterhouse present"],
     "#aec7e8", RED, f"{FSIS_YEARS[0]}–{FSIS_YEARS[1]}"),
]

n_outcomes = len(OUTCOMES)
n_pres     = len(_PRESENCE_SPECS)
fig_o, axes_o = plt.subplots(n_outcomes, n_pres,
                              figsize=(n_pres * 4.5, n_outcomes * 2.8))

for ci, (pres_label, bin_col, data_df, grp_labels, c0, c1, win_str) in enumerate(_PRESENCE_SPECS):
    for ri, (out_label, out_col) in enumerate(OUTCOMES.items()):
        ax = axes_o[ri, ci]
        if out_col not in data_df.columns or bin_col not in data_df.columns:
            ax.set_visible(False)
            continue

        results = []
        for gval, glabel, gcolor in [(0, grp_labels[0], c0), (1, grp_labels[1], c1)]:
            sub = data_df[data_df[bin_col] == gval][out_col]
            m, lo, hi = mean_ci(sub)
            results.append((glabel, m, lo, hi, gcolor))

        x_pos = np.arange(len(results))
        for xi, (glabel, m, lo, hi, gcolor) in enumerate(results):
            ax.bar(xi, m, color=gcolor, alpha=0.85, edgecolor="white", width=0.5)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(xi, m, yerr=[[m - lo], [hi - m]],
                            fmt="none", color="black", capsize=5, linewidth=1.5)
            n_sub = data_df[data_df[bin_col] == gval][out_col].notna().sum()
            ax.text(xi, (lo if np.isfinite(lo) else m) * 0.97,
                    f"n={n_sub:,}", ha="center", va="top", fontsize=6, color="grey")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(grp_labels, fontsize=7)
        ax.set_ylabel(out_label.replace("\n", " "), fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

        if ri == 0:
            ax.set_title(f"{pres_label}\n({win_str})", fontsize=9, fontweight="bold")

fig_o.suptitle(
    "Mean Outcomes by Facility Presence — Rural US Counties\n"
    "Theory: effect driven by LARGE CAFO presence and slaughterhouse operations, not total count",
    fontsize=11, y=1.01,
)
plt.tight_layout()
o_path = os.path.join(out_dir, f"{today_str}_O_presence_vs_absence.png")
fig_o.savefig(o_path, dpi=200, bbox_inches="tight")
plt.close(fig_o)
print("Saved:", o_path)


# =============================================================================
# FIGURE P: Binned scatter — LARGE CAFO only vs all outcomes
# Mirrors Figure A1 but uses log(large_cafo per 10k + 1) as treatment.
# Compare with A1 (total CAFO) to see if signal concentrates in large ops.
# =============================================================================
_large_col = "large_cafo"   # total large ops across all animals, already in panel
df_panel["x_large_cafo"] = log_per10k(df_panel[_large_col], df_panel[POP_COL])

n_cols_p = 3
n_rows_p = int(np.ceil(len(OUTCOMES) / n_cols_p))
fig_p, axes_p = plt.subplots(n_rows_p, n_cols_p, figsize=(n_cols_p * 5.5, n_rows_p * 4.5))
axes_p = axes_p.flatten()

for i, (ylabel, col) in enumerate(OUTCOMES.items()):
    ax = axes_p[i]
    if col not in df_panel.columns:
        ax.set_visible(False)
        continue
    binned_scatter_ax(
        ax,
        x_vals=df_panel["x_large_cafo"],
        y_vals=pd.to_numeric(df_panel[col], errors="coerce"),
        label_x="log(Large CAFO Ops per 10k Population + 1)",
        label_y=ylabel,
        n_bins=20,
        color=GREEN,
    )

for j in range(i + 1, len(axes_p)):
    axes_p[j].set_visible(False)

fig_p.suptitle(
    f"Large CAFO Operations vs. Outcomes — Rural US Counties, {PANEL_YEARS[0]}–{PANEL_YEARS[1]}\n"
    "Treatment: log(large-size ops per 10,000 residents + 1)   |   Large size class ONLY   |   "
    "Outcomes: per 100k or days/month   |   Bin means ± 95% CI   |   OLS fit\n"
    "Compare with Figure A1 (total CAFO) to assess whether signal concentrates in large operations",
    fontsize=10, y=1.02,
)
plt.tight_layout()
p_path = os.path.join(out_dir, f"{today_str}_P_large_cafo_vs_outcomes.png")
fig_p.savefig(p_path, dpi=200, bbox_inches="tight")
plt.close(fig_p)
print("Saved:", p_path)


# =============================================================================
# FIGURE Q: Dose-response by count tier
# Outcomes by large CAFO count tier (0 / 1–3 / 4–10 / 10+) and
# by FSIS establishment count tier (0 / 1 / 2–3 / 4+).
# X = tier category; Y = mean outcome ± 95% CI; one panel per outcome.
# =============================================================================
def assign_tier(series, cuts, labels):
    """Assign categorical tier labels based on numeric cuts."""
    s = pd.to_numeric(series, errors="coerce")
    return pd.cut(s, bins=cuts, labels=labels, right=True)

_TIER_SPECS = [
    # (label, col, data_df, cuts, tier_labels, color, window_str)
    ("Large CAFO Ops\n(count tier)",
     "large_cafo", df_panel,
     [-0.001, 0, 3, 10, 1e9],
     ["None (0)", "1–3", "4–10", "11+"],
     GREEN, f"{PANEL_YEARS[0]}–{PANEL_YEARS[1]}"),
    ("FSIS Establishments\n(count tier)",
     "n_unique_establishments_fsis", df_fsis,
     [-0.001, 0, 1, 3, 1e9],
     ["None (0)", "1", "2–3", "4+"],
     TEAL, f"{FSIS_YEARS[0]}–{FSIS_YEARS[1]}"),
    ("FSIS Slaughterhouses\n(count tier)",
     "n_slaughterhouse_present_establishments_fsis", df_fsis,
     [-0.001, 0, 1, 3, 1e9],
     ["None (0)", "1", "2–3", "4+"],
     RED, f"{FSIS_YEARS[0]}–{FSIS_YEARS[1]}"),
]

n_tiers   = len(_TIER_SPECS)
n_out_q   = len(OUTCOMES)
fig_q, axes_q = plt.subplots(n_out_q, n_tiers, figsize=(n_tiers * 4.5, n_out_q * 2.8))

for ci, (tier_label, tier_col, data_df, cuts, tier_labels, color, win_str) in enumerate(_TIER_SPECS):
    # Assign tiers
    data_df = data_df.copy()
    data_df["_tier"] = assign_tier(data_df[tier_col] if tier_col in data_df.columns else pd.Series(dtype=float),
                                   cuts, tier_labels)

    for ri, (out_label, out_col) in enumerate(OUTCOMES.items()):
        ax = axes_q[ri, ci]
        if out_col not in data_df.columns or "_tier" not in data_df.columns:
            ax.set_visible(False)
            continue

        x_pos  = np.arange(len(tier_labels))
        means, lows, highs, ns = [], [], [], []
        for tlabel in tier_labels:
            sub = data_df[data_df["_tier"] == tlabel][out_col]
            m, lo, hi = mean_ci(sub)
            means.append(m); lows.append(lo); highs.append(hi)
            ns.append(sub.notna().sum())

        ax.bar(x_pos, means, color=color, alpha=0.75, edgecolor="white", width=0.6)
        for xi, (m, lo, hi) in enumerate(zip(means, lows, highs)):
            if np.isfinite(lo) and np.isfinite(hi) and np.isfinite(m):
                ax.errorbar(xi, m, yerr=[[m - lo], [hi - m]],
                            fmt="none", color="black", capsize=4, linewidth=1.5)
        for xi, n in enumerate(ns):
            if np.isfinite(means[xi]):
                ax.text(xi, 0, f"n={n:,}", ha="center", va="bottom",
                        fontsize=5.5, color="grey", rotation=0)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(tier_labels, fontsize=7, rotation=15)
        ax.set_ylabel(out_label.replace("\n", " "), fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if ri == 0:
            ax.set_title(f"{tier_label}\n({win_str})", fontsize=9, fontweight="bold")

fig_q.suptitle(
    "Dose-Response by Facility Count Tier — Rural US Counties\n"
    "Mean outcome ± 95% CI by count category | Tests for threshold / non-linear effects",
    fontsize=11, y=1.01,
)
plt.tight_layout()
q_path = os.path.join(out_dir, f"{today_str}_Q_dose_response_tiers.png")
fig_q.savefig(q_path, dpi=200, bbox_inches="tight")
plt.close(fig_q)
print("Saved:", q_path)


# =============================================================================
# FIGURE R: Small vs. Large CAFO — side-by-side binned scatter
# For each of 3 key outcomes: left panel = small CAFO, right panel = large CAFO.
# Tests whether the signal concentrates in large operations.
# =============================================================================
_small_col = "small_cafo"
df_panel["x_small"] = log_per10k(df_panel[_small_col], df_panel[POP_COL])
df_panel["x_large"] = log_per10k(df_panel[_large_col], df_panel[POP_COL])

n_focus = len(FOCUS_OUTCOMES)
fig_r, axes_r = plt.subplots(n_focus, 2, figsize=(11, n_focus * 4.0))

for ri, (out_label, out_col) in enumerate(FOCUS_OUTCOMES.items()):
    y_vals = pd.to_numeric(df_panel.get(out_col, pd.Series(dtype=float)), errors="coerce")

    # Left: small CAFO
    ax_s = axes_r[ri, 0]
    binned_scatter_ax(
        ax_s,
        x_vals=df_panel["x_small"],
        y_vals=y_vals,
        label_x="log(Small CAFO Ops per 10k + 1)",
        label_y=out_label,
        n_bins=20,
        color="#aec7e8",
        title="Small CAFO Operations" if ri == 0 else None,
    )

    # Right: large CAFO
    ax_l = axes_r[ri, 1]
    binned_scatter_ax(
        ax_l,
        x_vals=df_panel["x_large"],
        y_vals=y_vals,
        label_x="log(Large CAFO Ops per 10k + 1)",
        label_y="",
        n_bins=20,
        color=GREEN,
        title="Large CAFO Operations" if ri == 0 else None,
    )
    # Match y-axis scales across the row for direct comparison
    y_min = min(ax_s.get_ylim()[0], ax_l.get_ylim()[0])
    y_max = max(ax_s.get_ylim()[1], ax_l.get_ylim()[1])
    ax_s.set_ylim(y_min, y_max)
    ax_l.set_ylim(y_min, y_max)

fig_r.suptitle(
    f"Small vs. Large CAFO — Key Outcomes, Rural US Counties, {PANEL_YEARS[0]}–{PANEL_YEARS[1]}\n"
    "If signal concentrates in large ops: right panels will show steeper OLS slope than left\n"
    "Treatment: log(ops per 10,000 residents + 1)   |   Y-axes matched within each row",
    fontsize=11, y=1.02,
)
plt.tight_layout()
r_path = os.path.join(out_dir, f"{today_str}_R_small_vs_large_scatter.png")
fig_r.savefig(r_path, dpi=200, bbox_inches="tight")
plt.close(fig_r)
print("Saved:", r_path)


print(f"\nAll threshold/presence figures saved to: {out_dir}")
