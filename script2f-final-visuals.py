#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2f: Core descriptive visualizations — CAFO / FSIS exposure vs. outcomes.

Purpose:
    Produces analytically useful figures rather than QA diagnostics.
    All X-axis treatment variables are expressed as log(ops per 10k population + 1)
    to remove the mechanical correlation between raw counts and county size.
    Outcome variables from County Health Rankings are already population-standardized
    (rates or percentages) and are used as-is.

Figures produced:
    A1 — Binned scatter: CAFO total ops (per capita, log) vs. all 6 outcomes.
         Pooled 2010–2015. 20 equal-count bins, mean ± 95% CI, OLS fit.
    A2 — Binned scatter: CAFO by animal type (cattle / hogs / chickens)
         vs. 3 core outcomes. Same period.
    A3 — Binned scatter: FSIS establishments (per capita, log) vs. all outcomes.
         2017 cross-section only (FSIS data not available before 2017).
    B  — County time series: 1 representative county per state, 2010–2015.
         CAFO exposure on left axis, Poor Mental Health Days on right axis.
    C  — Outcome cross-correlation heatmap (Spearman, 2010–2015).
         Shows which outcomes co-move before committing to a primary Y variable.
    D  — Violin plots: outcome distributions by CAFO intensity quartile.
         Pooled 2010–2015. Inner lines show Q1/median/Q3 within each violin.
    E  — Box-and-whisker (median, no mean): same CAFO quartile grouping.
         Overlaid with jitter of raw county observations. Median line in red.
    F1 — Choropleth map: log(CAFO ops per 10k pop + 1), rural US counties, 2012.
    F2 — Choropleth map: Poor Mental Health Days, rural US counties, 2012.
         (F maps saved as both interactive HTML and static PNG)

Outputs to: Dropbox/Mental/Data/merged/figs/core-visuals/

Dependencies: packages.py, functions.py (shared project utilities)
"""

from packages import *
from functions import *
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import plotly.express as px
except Exception:
    px = None

# =============================================================================
# Configuration
# =============================================================================
merged_dir = os.path.join(db_data, "merged")
out_dir = os.path.join(merged_dir, "figs", "core-visuals")
os.makedirs(out_dir, exist_ok=True)
today_str = date.today().strftime("%Y-%m-%d")

PANEL_YEARS = (2010, 2015)   # window where MH outcome coverage is adequate
POP_COL     = "population_population_full"
CAFO_TOTAL  = "cafo_total_ops_all_animals"

# Outcome columns with clean display labels and units
OUTCOMES = {
    "Poor Mental Health Days\n(avg days/month, CHR)":
        "poor_mental_health_days_raw_value_mentalhealthrank_full",
    "Excessive Drinking\n(% adults, CHR)":
        "excessive_drinking_raw_value_mentalhealthrank_full",
    "Violent Crime Rate\n(per 100k, CHR)":
        "violent_crime_raw_value_mentalhealthrank_full",
    "Homicide Rate\n(per 100k, CHR)":
        "homicides_raw_value_mentalhealthrank_full",
    "Aggravated Assault\n(log UCR count + 1)":
        "aggravated_assault_crime_fips_level_final",
    "Deaths of Despair\nCrude Rate (per 100k, CDC)":
        "crude_rate_cdc_county_year_deathsofdespair",
}

# Columns that need their own log transform (raw counts, not already rates)
LOG_Y_COLS = {"aggravated_assault_crime_fips_level_final"}

# CAFO by animal type; each sub-dict maps size label to column
CAFO_TYPES = {
    "Cattle":   {"small": "cafo_cattle_small",   "medium": "cafo_cattle_medium",   "large": "cafo_cattle_large"},
    "Hogs":     {"small": "cafo_hogs_small",     "medium": "cafo_hogs_medium",     "large": "cafo_hogs_large"},
    "Chickens": {"small": "cafo_chickens_small", "medium": "cafo_chickens_medium", "large": "cafo_chickens_large"},
}
CAFO_TYPE_COLORS = {"Cattle": "#1b7837", "Hogs": "#762a83", "Chickens": "#b35806"}

FSIS_TOTAL    = "n_unique_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip"
FSIS_SLAUGHTER = "n_slaughterhouse_present_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip"

# 3 outcomes used for the by-type comparison (A2) — those with best coverage
CORE_OUTCOMES_A2 = {
    "Poor Mental Health Days\n(avg days/month)":
        "poor_mental_health_days_raw_value_mentalhealthrank_full",
    "Excessive Drinking\n(% adults)":
        "excessive_drinking_raw_value_mentalhealthrank_full",
    "Violent Crime Rate\n(per 100k)":
        "violent_crime_raw_value_mentalhealthrank_full",
}

# State FIPS → abbreviation (for labeling)
STATE_FIPS_TO_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE",
    "11":"DC","12":"FL","13":"GA","15":"HI","16":"ID","17":"IL","18":"IN","19":"IA",
    "20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA","26":"MI","27":"MN",
    "28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM",
    "36":"NY","37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI",
    "45":"SC","46":"SD","47":"TN","48":"TX","49":"UT","50":"VT","51":"VA","53":"WA",
    "54":"WV","55":"WI","56":"WY",
}

# Plot aesthetics
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})
BLUE = "#2166ac"
RED  = "#d6604d"
GREY = "#636363"

# =============================================================================
# Helpers
# =============================================================================
to_num    = to_numeric_series
load_file = latest_file_glob


def log_per10k(series, pop_series):
    """
    Compute log(x per 10k population + 1).
    Handles zero population and zero ops cleanly.
    """
    x   = to_num(series)
    pop = to_num(pop_series).replace(0, np.nan)
    return np.log1p((x / pop) * 10_000)


def binned_scatter_ax(ax, x_vals, y_vals, label_x, label_y,
                      n_bins=20, color=BLUE, title=None):
    """
    Draw binned-mean scatter with 95% CI error bars and OLS fit line.

    Parameters
    ----------
    ax       : matplotlib Axes
    x_vals   : array-like (NaN-safe)
    y_vals   : array-like (NaN-safe, aligned with x_vals)
    label_x  : str — x-axis label
    label_y  : str — y-axis label
    n_bins   : int — number of equal-count quantile bins
    color    : str — color for bin-mean dots
    title    : str or None — panel title
    """
    x = pd.Series(x_vals, dtype=float)
    y = pd.Series(y_vals, dtype=float)
    mask = x.notna() & y.notna()
    x, y = x[mask].values, y[mask].values
    n = int(mask.sum())

    if n < n_bins * 3:
        ax.text(0.5, 0.5, f"Insufficient data\n(n={n:,})",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color=GREY)
        ax.set_xlabel(label_x, fontsize=8)
        ax.set_ylabel(label_y, fontsize=8)
        if title:
            ax.set_title(title, fontsize=9)
        return

    # Quantile bins on X
    try:
        bins = pd.qcut(x, q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.cut(x, bins=n_bins)

    bin_df = pd.DataFrame({"x": x, "y": y, "bin": bins})
    agg = (
        bin_df.groupby("bin", observed=True)["y"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )
    agg["x_mid"] = bin_df.groupby("bin", observed=True)["x"].mean().values
    agg["ci95"]  = 1.96 * agg["sem"]
    agg = agg.dropna(subset=["x_mid", "mean"])

    ax.errorbar(
        agg["x_mid"], agg["mean"], yerr=agg["ci95"],
        fmt="o", color=color, ms=5, lw=1.2, capsize=3, elinewidth=1,
        label=f"Bin mean ± 95% CI  (n={n:,})",
        zorder=3,
    )

    # OLS fit line
    slope, intercept, r, p, _ = scipy_stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    p_str  = "p<0.001" if p < 0.001 else f"p={p:.3f}"
    ax.plot(
        x_line, intercept + slope * x_line,
        color=RED, lw=1.5, ls="--",
        label=f"OLS: β={slope:.3f}, r={r:.2f}, {p_str}",
        zorder=4,
    )

    ax.set_xlabel(label_x, fontsize=8)
    ax.set_ylabel(label_y, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="best", framealpha=0.6)
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold")


# =============================================================================
# Load and prepare panel
# =============================================================================
merged_path = load_file(merged_dir, "*_full_merged.csv")
print("Loading:", merged_path)

df = pd.read_csv(merged_path, low_memory=False)
df = normalize_panel_key(df)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.drop_duplicates(subset=["fips", "year"], keep="first").reset_index(drop=True)
df["state_fips"] = df["fips"].astype("string").str[:2]

# Coerce all analysis columns to numeric up front
all_analysis_cols = [
    POP_COL, CAFO_TOTAL, FSIS_TOTAL, FSIS_SLAUGHTER,
    *[c for d in CAFO_TYPES.values() for c in d.values()],
    *OUTCOMES.values(),
]
for col in all_analysis_cols:
    if col in df.columns:
        df[col] = to_num(df[col])

print(f"Panel loaded: {df.shape[0]:,} rows | "
      f"{df['fips'].nunique():,} counties | "
      f"years {int(df['year'].min())}–{int(df['year'].max())}")


# =============================================================================
# FIGURE A1
# Binned scatter: CAFO total ops (log per 10k pop) vs all 6 outcomes
# Pooled 2010–2015 (window where poor_mental_health_days coverage is ≥90%)
# =============================================================================
df_panel = df[df["year"].between(*PANEL_YEARS)].copy()
df_panel["x_cafo"] = log_per10k(df_panel[CAFO_TOTAL], df_panel[POP_COL])

n_outcomes = len(OUTCOMES)
n_cols_a1  = 3
n_rows_a1  = int(np.ceil(n_outcomes / n_cols_a1))

fig, axes = plt.subplots(n_rows_a1, n_cols_a1, figsize=(n_cols_a1 * 5.5, n_rows_a1 * 4.5))
axes = axes.flatten()

for i, (ylabel, col) in enumerate(OUTCOMES.items()):
    ax = axes[i]
    if col not in df_panel.columns:
        ax.set_visible(False)
        continue
    y_vals = np.log1p(df_panel[col]) if col in LOG_Y_COLS else df_panel[col]
    binned_scatter_ax(
        ax,
        x_vals=df_panel["x_cafo"],
        y_vals=y_vals,
        label_x="log(Total CAFO Ops per 10k Population + 1)",
        label_y=ylabel,
        n_bins=20,
        color=BLUE,
    )

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"CAFO Operations vs. Outcomes — Rural US Counties, {PANEL_YEARS[0]}–{PANEL_YEARS[1]}\n"
    "Treatment: log(total CAFO ops per 10,000 residents + 1)   |   "
    "Bin means ± 95% CI   |   OLS fit",
    fontsize=11, y=1.01,
)
plt.tight_layout()
a1_path = os.path.join(out_dir, f"{today_str}_A1_cafo_total_vs_outcomes.png")
fig.savefig(a1_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", a1_path)


# =============================================================================
# FIGURE A2
# Binned scatter: CAFO by animal type (cattle / hogs / chickens) vs 3 outcomes
# Rows = outcomes, Columns = animal type
# =============================================================================
animal_list   = list(CAFO_TYPES.keys())
outcome_list  = list(CORE_OUTCOMES_A2.items())
n_rows_a2     = len(outcome_list)
n_cols_a2     = len(animal_list)

fig, axes = plt.subplots(n_rows_a2, n_cols_a2, figsize=(n_cols_a2 * 5.5, n_rows_a2 * 4.5))

for row_i, (ylabel, y_col) in enumerate(outcome_list):
    for col_j, animal in enumerate(animal_list):
        ax = axes[row_i][col_j]

        # Sum across size classes for this animal type
        size_cols   = [c for c in CAFO_TYPES[animal].values() if c in df_panel.columns]
        animal_ops  = df_panel[size_cols].sum(axis=1, min_count=1)
        x_vals      = log_per10k(animal_ops, df_panel[POP_COL])
        y_vals      = df_panel[y_col] if y_col in df_panel.columns else pd.Series(dtype=float)

        # Column header on first row only
        title = animal if row_i == 0 else None

        # Y-axis label on first column only
        ylabel_ax = ylabel if col_j == 0 else ""

        binned_scatter_ax(
            ax,
            x_vals=x_vals,
            y_vals=y_vals,
            label_x=f"log({animal} CAFO Ops per 10k pop + 1)",
            label_y=ylabel_ax,
            n_bins=20,
            color=CAFO_TYPE_COLORS[animal],
            title=title,
        )

fig.suptitle(
    f"CAFO by Animal Type vs. Outcomes — Rural US Counties, {PANEL_YEARS[0]}–{PANEL_YEARS[1]}\n"
    "Columns: animal type   |   Rows: outcome",
    fontsize=11, y=1.01,
)
plt.tight_layout()
a2_path = os.path.join(out_dir, f"{today_str}_A2_cafo_by_type_vs_outcomes.png")
fig.savefig(a2_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", a2_path)


# =============================================================================
# FIGURE A3
# Binned scatter: FSIS establishments (per capita, log) vs all 6 outcomes
# Panel: 2017–2023 (all years where FSIS county linkage is available)
# =============================================================================
_FSIS_YEARS = (2017, 2023)
df_fsis = df[df["year"].between(*_FSIS_YEARS)].copy()

if FSIS_TOTAL in df_fsis.columns and df_fsis[FSIS_TOTAL].notna().sum() > 50:
    df_fsis["x_fsis"] = log_per10k(df_fsis[FSIS_TOTAL], df_fsis[POP_COL])

    fig, axes = plt.subplots(n_rows_a1, n_cols_a1, figsize=(n_cols_a1 * 5.5, n_rows_a1 * 4.5))
    axes = axes.flatten()

    for i, (ylabel, col) in enumerate(OUTCOMES.items()):
        ax = axes[i]
        if col not in df_fsis.columns:
            ax.set_visible(False)
            continue
        y_vals = np.log1p(df_fsis[col]) if col in LOG_Y_COLS else df_fsis[col]
        binned_scatter_ax(
            ax,
            x_vals=df_fsis["x_fsis"],
            y_vals=y_vals,
            label_x="log(FSIS Establishments per 10k Population + 1)",
            label_y=ylabel,
            n_bins=20,
            color="#41b6c4",
        )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"FSIS Slaughter & Processing Establishments vs. Outcomes — Rural US Counties, {_FSIS_YEARS[0]}–{_FSIS_YEARS[1]}\n"
        "Treatment: log(total FSIS establishments per 10,000 residents + 1)   |   "
        "County-year panel (FSIS available 2017–present only)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    a3_path = os.path.join(out_dir, f"{today_str}_A3_fsis_vs_outcomes_{_FSIS_YEARS[0]}_{_FSIS_YEARS[1]}.png")
    fig.savefig(a3_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", a3_path)
else:
    print("FSIS panel data insufficient — skipping Figure A3.")


# =============================================================================
# FIGURE B
# County time series: 1 representative county per state, 2010–2015
#
# Selection criteria:
#   1. Has CAFO data and ≥4 years of poor_mental_health_days in 2010–2015.
#   2. Falls in the top tercile of CAFO exposure (per capita) within its state.
#   3. Among those, pick the county closest to the state median CAFO exposure
#      (avoids choosing an extreme outlier as the representative).
#
# Plot: dual-axis per county — CAFO ops per 10k pop (left, blue) and
#       poor mental health days (right, red), 2010–2015.
# =============================================================================
MH_COL = "poor_mental_health_days_raw_value_mentalhealthrank_full"

df_b = df[df["year"].between(*PANEL_YEARS)].copy()
df_b["cafo_per10k"] = (df_b[CAFO_TOTAL] / df_b[POP_COL].replace(0, np.nan)) * 10_000
df_b[MH_COL] = to_num(df_b[MH_COL])

# Summarize per county
county_stats = (
    df_b.groupby("fips")
    .agg(
        state_fips=("state_fips", "first"),
        mean_cafo_per10k=("cafo_per10k", "mean"),
        n_mh_years=(MH_COL, lambda s: s.notna().sum()),
    )
    .reset_index()
)

# Filter: ≥4 years of MH data
county_stats = county_stats[county_stats["n_mh_years"] >= 4].copy()

# State p67 CAFO threshold (top tercile)
county_stats["state_p67"] = county_stats.groupby("state_fips")["mean_cafo_per10k"].transform(
    lambda s: s.quantile(0.67)
)
county_stats = county_stats[county_stats["mean_cafo_per10k"] >= county_stats["state_p67"]].copy()

# Pick the county closest to its state's median CAFO (within top tercile)
selected_fips = {}
for st, grp in county_stats.groupby("state_fips"):
    if grp.empty:
        continue
    state_median = grp["mean_cafo_per10k"].median()
    best_idx = (grp["mean_cafo_per10k"] - state_median).abs().idxmin()
    selected_fips[st] = grp.loc[best_idx, "fips"]

print(f"Figure B: {len(selected_fips)} counties selected (1 per state).")

df_ts = df_b[df_b["fips"].isin(selected_fips.values())].copy()

state_list = sorted(selected_fips.keys())
n_states   = len(state_list)
n_cols_b   = 6
n_rows_b   = int(np.ceil(n_states / n_cols_b))

fig, axes = plt.subplots(n_rows_b, n_cols_b, figsize=(n_cols_b * 4, n_rows_b * 3.5))
axes = axes.flatten()

for i, st in enumerate(state_list):
    ax = axes[i]
    fips_val = selected_fips[st]
    county_df = df_ts[df_ts["fips"] == fips_val].sort_values("year").copy()

    if county_df.empty:
        ax.set_visible(False)
        continue

    years     = county_df["year"].astype(int).values
    cafo_vals = county_df["cafo_per10k"].values
    mh_vals   = county_df[MH_COL].values

    # Primary axis: CAFO per 10k (blue)
    color_cafo = BLUE
    color_mh   = RED
    ax.plot(years, cafo_vals, color=color_cafo, lw=1.8, marker="o", ms=4, label="CAFO / 10k pop")
    ax.set_ylabel("CAFO ops\nper 10k pop", color=color_cafo, fontsize=7)
    ax.tick_params(axis="y", labelcolor=color_cafo, labelsize=6)
    ax.tick_params(axis="x", labelsize=6)

    # Secondary axis: Poor MH days (red)
    ax2 = ax.twinx()
    ax2.plot(years, mh_vals, color=color_mh, lw=1.8, marker="s", ms=4,
             ls="--", label="Poor MH Days")
    ax2.set_ylabel("Poor MH Days", color=color_mh, fontsize=7)
    ax2.tick_params(axis="y", labelcolor=color_mh, labelsize=6)
    ax2.spines["top"].set_visible(False)

    # County label
    st_abbr = STATE_FIPS_TO_ABBR.get(st, st)
    ax.set_title(f"{st_abbr}\n(FIPS {fips_val})", fontsize=8)
    ax.set_xlabel("Year", fontsize=7)
    ax.spines["top"].set_visible(False)

for j in range(n_states, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"County-Level Trends: CAFO Exposure (blue) vs. Poor Mental Health Days (red dashed)\n"
    f"1 county per state — top-tercile CAFO density, ≥4 years MH data, {PANEL_YEARS[0]}–{PANEL_YEARS[1]}",
    fontsize=11, y=1.01,
)
plt.tight_layout()
b_path = os.path.join(out_dir, f"{today_str}_B_county_time_series_by_state.png")
fig.savefig(b_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", b_path)

# Save county selection metadata
selection_meta = (
    county_stats[county_stats["fips"].isin(selected_fips.values())]
    .copy()
    .sort_values("state_fips")
)
selection_meta.to_csv(
    os.path.join(out_dir, f"{today_str}_B_county_selection_metadata.csv"), index=False
)


# =============================================================================
# FIGURE C
# Outcome cross-correlation heatmap — Spearman, 2010–2015
# Purpose: show how outcomes co-move before committing to a primary Y variable.
# =============================================================================
corr_outcomes = {
    "Poor MH Days (CHR)":        "poor_mental_health_days_raw_value_mentalhealthrank_full",
    "Excessive Drinking (CHR)":  "excessive_drinking_raw_value_mentalhealthrank_full",
    "Violent Crime (CHR)":       "violent_crime_raw_value_mentalhealthrank_full",
    "Homicides (CHR)":           "homicides_raw_value_mentalhealthrank_full",
    "Agg. Assault (UCR)":        "aggravated_assault_crime_fips_level_final",
    "Despair Rate (CDC)":        "crude_rate_cdc_county_year_deathsofdespair",
}

corr_cols   = {lbl: col for lbl, col in corr_outcomes.items() if col in df_panel.columns}
corr_data   = pd.DataFrame({lbl: to_num(df_panel[col]) for lbl, col in corr_cols.items()})
corr_matrix = corr_data.corr(method="spearman", min_periods=200)

mask_upper = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    corr_matrix,
    ax=ax,
    mask=mask_upper,
    cmap="RdBu_r",
    center=0, vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 9},
    linewidths=0.5,
    cbar_kws={"label": "Spearman ρ"},
    square=True,
)
ax.set_title(
    f"Outcome Cross-Correlations — Spearman ρ\n"
    f"Rural US Counties, {PANEL_YEARS[0]}–{PANEL_YEARS[1]} (min 200 pairwise obs)",
    fontsize=11, pad=10,
)
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
c_path = os.path.join(out_dir, f"{today_str}_C_outcome_crosscorrelation.png")
fig.savefig(c_path, dpi=200, bbox_inches="tight")
plt.close(fig)
corr_matrix.to_csv(os.path.join(out_dir, f"{today_str}_C_outcome_crosscorrelation.csv"))
print("Saved:", c_path)

# =============================================================================
# FIGURES D & E
# D — Violin plots: outcome distributions by CAFO intensity quartile
# E — Box-and-whisker (median only, no mean): same grouping
#
# Quartiles derived from log(CAFO total ops per 10k pop + 1) in df_panel.
# Violins show the full empirical density; inner tick lines mark Q1/median/Q3.
# Boxes show IQR with median highlighted in red; raw county dots underlaid.
#
# Rationale: binned scatters (A1-A2) show conditional means along a continuous
# treatment gradient. Quartile plots complement these by showing distributional
# heterogeneity — whether the outcome variance differs across CAFO intensity
# levels, not just the conditional mean. Violin + box together give a reviewer
# both the shape (violin) and the robust summary statistics (box/median).
# =============================================================================
df_panel["cafo_log_per10k"] = log_per10k(df_panel[CAFO_TOTAL], df_panel[POP_COL])
df_panel["cafo_quartile"] = pd.qcut(
    df_panel["cafo_log_per10k"],
    q=4,
    labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"],
    duplicates="drop",
)

n_cols_de = 3
n_rows_de = int(np.ceil(len(OUTCOMES) / n_cols_de))
QUARTILE_PALETTE = ["#d1e5f0", "#92c5de", "#4393c3", "#2166ac"]

# --- Figure D: Violin ---
fig, axes = plt.subplots(n_rows_de, n_cols_de, figsize=(n_cols_de * 5, n_rows_de * 4.5))
axes = axes.flatten()

for i, (ylabel, col) in enumerate(OUTCOMES.items()):
    ax = axes[i]
    if col not in df_panel.columns:
        ax.set_visible(False)
        continue

    plot_df = df_panel[["cafo_quartile", col]].dropna().copy()
    if plot_df.empty or plot_df["cafo_quartile"].nunique() < 2:
        ax.set_visible(False)
        continue

    plot_df["y"] = np.log1p(plot_df[col]) if col in LOG_Y_COLS else plot_df[col]

    sns.violinplot(
        data=plot_df, x="cafo_quartile", y="y", ax=ax,
        palette=QUARTILE_PALETTE,
        inner="quartile",   # tick lines at Q1, median, Q3 inside the violin
        linewidth=0.8,
        cut=0,              # do not extrapolate beyond observed data range
    )
    ax.set_xlabel("CAFO Intensity Quartile\n(log ops per 10k pop)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)

    # Annotate median value above each violin body
    y_top = ax.get_ylim()[1]
    for j, q_label in enumerate(plot_df["cafo_quartile"].cat.categories):
        med = plot_df.loc[plot_df["cafo_quartile"] == q_label, "y"].median()
        ax.text(j, y_top * 0.98, f"{med:.2f}", ha="center", va="top",
                fontsize=6.5, color=GREY, style="italic")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"Outcome Distributions by CAFO Intensity Quartile — Rural US Counties, "
    f"{PANEL_YEARS[0]}–{PANEL_YEARS[1]}\n"
    "Violin width = density   |   Inner ticks: Q1 / median / Q3   |   "
    "CAFO quartiles: log(total ops per 10k pop + 1)",
    fontsize=11, y=1.01,
)
plt.tight_layout()
d_path = os.path.join(out_dir, f"{today_str}_D_violin_cafo_quartile_vs_outcomes.png")
fig.savefig(d_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", d_path)


# --- Figure E: Box-and-whisker (median, no mean) ---
fig, axes = plt.subplots(n_rows_de, n_cols_de, figsize=(n_cols_de * 5, n_rows_de * 4.5))
axes = axes.flatten()

for i, (ylabel, col) in enumerate(OUTCOMES.items()):
    ax = axes[i]
    if col not in df_panel.columns:
        ax.set_visible(False)
        continue

    plot_df = df_panel[["cafo_quartile", col]].dropna().copy()
    if plot_df.empty or plot_df["cafo_quartile"].nunique() < 2:
        ax.set_visible(False)
        continue

    plot_df["y"] = np.log1p(plot_df[col]) if col in LOG_Y_COLS else plot_df[col]

    # Raw county observations behind the boxes (alpha low, small dots)
    sns.stripplot(
        data=plot_df, x="cafo_quartile", y="y", ax=ax,
        color=GREY, alpha=0.12, size=1.8, jitter=True, zorder=1,
    )
    # Box: IQR + whiskers (1.5×IQR). Fliers suppressed — visible via strip.
    sns.boxplot(
        data=plot_df, x="cafo_quartile", y="y", ax=ax,
        palette=QUARTILE_PALETTE,
        linewidth=1.0,
        width=0.5,
        flierprops={"marker": ""},           # hide flier dots (already in strip)
        medianprops={"color": RED, "lw": 2.5},
        showmeans=False,                     # median only — no mean marker
        zorder=2,
    )
    ax.set_xlabel("CAFO Intensity Quartile\n(log ops per 10k pop)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"Outcome Distributions by CAFO Intensity Quartile — Rural US Counties, "
    f"{PANEL_YEARS[0]}–{PANEL_YEARS[1]}\n"
    "Box = IQR   |   Red line = median (no mean shown)   |   "
    "Whiskers = 1.5×IQR   |   Dots = individual county-years",
    fontsize=11, y=1.01,
)
plt.tight_layout()
e_path = os.path.join(out_dir, f"{today_str}_E_boxplot_cafo_quartile_vs_outcomes.png")
fig.savefig(e_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", e_path)


# =============================================================================
# FIGURES F1 & F2
# Choropleth maps — rural US counties, 2012 (Census year; best joint coverage)
# F1: log(CAFO ops per 10k pop + 1) — shows geographic concentration of CAFOs
# F2: Poor Mental Health Days (avg days/month, CHR)
#
# Saved as interactive HTML (always) and static PNG (if kaleido is installed).
# The side-by-side geographic view is the primary storytelling figure for
# presentations: it lets a reader visually compare CAFO and MH geographies
# before any modelling, revealing whether the two patterns spatially overlap.
# =============================================================================
if px is not None:
    try:
        from urllib.request import urlopen
        import json as _json

        _geojson_url = (
            "https://raw.githubusercontent.com/plotly/datasets"
            "/master/geojson-counties-fips.json"
        )
        with urlopen(_geojson_url) as _r:
            _counties_geojson = _json.load(_r)

        df_2012 = df[df["year"] == 2012].copy()
        df_2012["cafo_log_per10k"] = log_per10k(df_2012[CAFO_TOTAL], df_2012[POP_COL])
        df_2012["fips_str"] = df_2012["fips"].astype("string").str.zfill(5)
        _mh_col = "poor_mental_health_days_raw_value_mentalhealthrank_full"

        _map_specs = [
            (
                "cafo_log_per10k",
                "log(CAFO Ops per 10k Population + 1)",
                "YlOrRd",
                "F1_cafo_intensity_map_2012",
            ),
            (
                _mh_col,
                "Poor Mental Health Days (avg days/month, CHR)",
                "RdPu",
                "F2_poor_mental_health_map_2012",
            ),
        ]

        for _col, _label, _cscale, _stub in _map_specs:
            if _col not in df_2012.columns:
                print(f"Skipping map {_stub} — column '{_col}' not found.")
                continue

            _map_df = df_2012[["fips_str", _col]].dropna()
            _fig_map = px.choropleth(
                _map_df,
                geojson=_counties_geojson,
                locations="fips_str",
                color=_col,
                color_continuous_scale=_cscale,
                scope="usa",
                title=f"{_label} — Rural US Counties, 2012",
                labels={_col: _label},
            )
            _fig_map.update_layout(
                margin={"r": 0, "t": 50, "l": 0, "b": 0},
                title_font_size=13,
                geo={"showlakes": False},
            )

            # Always save interactive HTML
            _html_path = os.path.join(out_dir, f"{today_str}_{_stub}.html")
            _fig_map.write_html(_html_path)
            print("Saved HTML:", _html_path)

            # Save static PNG if kaleido is available
            try:
                _png_path = os.path.join(out_dir, f"{today_str}_{_stub}.png")
                _fig_map.write_image(_png_path, width=1400, height=800, scale=2)
                print("Saved PNG: ", _png_path)
            except Exception as _e_img:
                print(f"  PNG export skipped (install kaleido for static export): {_e_img}")

    except Exception as _e_map:
        print(f"Figures F1/F2 (choropleth) skipped: {_e_map}")
else:
    print("Figures F1/F2 skipped — plotly not available. Install plotly to enable maps.")


print("\nAll figures saved to:", out_dir)
