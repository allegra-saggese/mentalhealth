#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script2g-cafo-composition.py

CAFO composition and geographic concentration visualizations.
Outputs saved to: Dropbox/Mental/Data/merged/figs/cafo-composition/

Figures produced:
  K1_cattle_top20_by_year.png     — Top 20 counties: total cattle ops per 10k, each census year
  K2_hogs_top20_by_year.png       — Top 20 counties: total hog ops per 10k, each census year
  K3_chickens_top20_by_year.png   — Top 20 counties: total chicken ops per 10k, each census year
  L1_cattle_large_top20.png       — Top 20 counties: large cattle only per 10k, each census year
  L2_hogs_large_top20.png         — Top 20 counties: large hogs only
  L3_chickens_large_top20.png     — Top 20 counties: large chickens only
  M1_cattle_size_composition.png  — Stacked 100% bar: cattle S/M/L share across census years
  M2_hogs_size_composition.png    — Same for hogs
  M3_chickens_size_composition.png— Same for chickens
  N_all_animals_size_opacity.png  — All animals × S/M/L with opacity-graded stacked bars
"""

from packages import *
from functions import *

# ── Directories ──────────────────────────────────────────────────────────────
merged_dir = os.path.join(db_data, "merged")
out_dir    = os.path.join(db_data, "merged", "figs", "cafo-composition")
os.makedirs(out_dir, exist_ok=True)
today_str  = date.today().strftime("%Y-%m-%d")

# ── Load panel ───────────────────────────────────────────────────────────────
df = pd.read_csv(latest_file_glob(merged_dir, "*_full_merged.csv"), low_memory=False)
print(f"Panel loaded: {len(df):,} rows | {df['fips'].nunique():,} counties | "
      f"years {int(df['year'].min())}–{int(df['year'].max())}")

POP_COL      = "population"
CENSUS_YEARS = [2002, 2007, 2012, 2017]

# ── Helpers ──────────────────────────────────────────────────────────────────
def log_per10k(series, pop):
    x   = pd.to_numeric(series, errors="coerce")
    pop = pd.to_numeric(pop,    errors="coerce").replace(0, np.nan)
    return np.log1p((x / pop) * 10_000)

def top_n_counties(df_yr, val_col, n=20, label_col="county_label"):
    """Return top-N rows by val_col, with a formatted county label."""
    sub = df_yr[[val_col, label_col, POP_COL]].copy()
    sub = sub[sub[val_col].notna() & (sub[val_col] > 0)]
    sub["_log"] = log_per10k(sub[val_col], sub[POP_COL])
    return sub.nlargest(n, "_log")

# Build a county label: "County Name, ST" from state column and fips
def _add_county_label(df):
    out = df.copy()
    # Use state abbreviation if available; fall back to state_code
    _state_col = "state" if "state" in out.columns else "state_code"
    # fips → county name not in panel directly, use fips as fallback
    out["county_label"] = out["fips"].astype("string").str.zfill(5)
    if _state_col in out.columns:
        out["county_label"] = out["county_label"] + " (" + out[_state_col].astype("string") + ")"
    return out

df = _add_county_label(df)

# ── Plot aesthetics ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

ANIMAL_COLORS  = {"Cattle": "#1b7837", "Hogs": "#762a83", "Chickens": "#b35806"}
SIZE_ALPHAS    = {"Small": 0.35, "Medium": 0.65, "Large": 1.0}
SIZE_HATCHES   = {"Small": "///", "Medium": "", "Large": ""}


# =============================================================================
# FIGURES K1–K3: Top-20 counties by total animal ops per 10k — each census year
# =============================================================================
_ANIMAL_TOTALS = {
    "Cattle":   ("cafo_cattle_small",   "cafo_cattle_medium",   "cafo_cattle_large"),
    "Hogs":     ("cafo_hogs_small",     "cafo_hogs_medium",     "cafo_hogs_large"),
    "Chickens": ("cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large"),
}

TOP_N = 20

for k_idx, (animal, size_cols) in enumerate(_ANIMAL_TOTALS.items(), start=1):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    color = ANIMAL_COLORS[animal]

    for ax, yr in zip(axes, CENSUS_YEARS):
        df_yr = df[df["year"] == yr].copy()
        # Sum S+M+L to get total for this animal
        _total_col = f"_total_{animal.lower()}"
        df_yr[_total_col] = df_yr[[c for c in size_cols if c in df_yr.columns]].sum(axis=1, min_count=1)

        top = top_n_counties(df_yr, _total_col, n=TOP_N)
        top = top.sort_values("_log")  # ascending so largest is at top of horizontal bar

        ax.barh(top["county_label"], top["_log"], color=color, alpha=0.8, edgecolor="white")
        ax.set_xlabel("log(Total Ops per 10,000 Residents + 1)", fontsize=9)
        ax.set_title(str(yr), fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(left=0)

    fig.suptitle(
        f"Top {TOP_N} Counties by {animal} CAFO Concentration — Rural US, Census Years\n"
        "Treatment: log(total ops per 10,000 residents + 1) | Small + Medium + Large combined",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f"{today_str}_K{k_idx}_{animal.lower()}_top{TOP_N}_by_year.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


# =============================================================================
# FIGURES L1–L3: Top-20 counties by LARGE-size only, per 10k — each census year
# =============================================================================
_ANIMAL_LARGE = {
    "Cattle":   "cafo_cattle_large",
    "Hogs":     "cafo_hogs_large",
    "Chickens": "cafo_chickens_large",
}

for l_idx, (animal, large_col) in enumerate(_ANIMAL_LARGE.items(), start=1):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    color = ANIMAL_COLORS[animal]

    for ax, yr in zip(axes, CENSUS_YEARS):
        df_yr = df[df["year"] == yr].copy()
        if large_col not in df_yr.columns:
            ax.set_visible(False)
            continue
        top = top_n_counties(df_yr, large_col, n=TOP_N)
        top = top.sort_values("_log")

        ax.barh(top["county_label"], top["_log"], color=color, alpha=0.9, edgecolor="white")
        ax.set_xlabel("log(Large Ops per 10,000 Residents + 1)", fontsize=9)
        ax.set_title(str(yr), fontsize=11, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(left=0)

    fig.suptitle(
        f"Top {TOP_N} Counties by Large {animal} CAFO Concentration — Rural US, Census Years\n"
        "Treatment: log(large-operation ops per 10,000 residents + 1) | Large size class only",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f"{today_str}_L{l_idx}_{animal.lower()}_large_top{TOP_N}_by_year.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


# =============================================================================
# FIGURES M1–M3: Stacked 100% bar — S/M/L share within each animal, by census year
# Each figure = one animal type. X = census year, Y = % share summing to 100%.
# Based on national totals (sum across all rural counties).
# =============================================================================
_SIZE_COLS_BY_ANIMAL = {
    "Cattle":   {"Small": "cafo_cattle_small",   "Medium": "cafo_cattle_medium",   "Large": "cafo_cattle_large"},
    "Hogs":     {"Small": "cafo_hogs_small",     "Medium": "cafo_hogs_medium",     "Large": "cafo_hogs_large"},
    "Chickens": {"Small": "cafo_chickens_small", "Medium": "cafo_chickens_medium", "Large": "cafo_chickens_large"},
}
SIZE_COLORS = {"Small": "#d1e5f0", "Medium": "#4393c3", "Large": "#053061"}

for m_idx, (animal, size_map) in enumerate(_SIZE_COLS_BY_ANIMAL.items(), start=1):
    # Aggregate national totals per census year
    records = []
    for yr in CENSUS_YEARS:
        df_yr = df[df["year"] == yr]
        row = {"year": yr}
        for size, col in size_map.items():
            if col in df_yr.columns:
                row[size] = df_yr[col].sum(skipna=True)
            else:
                row[size] = 0.0
        total = sum(row[s] for s in ["Small", "Medium", "Large"])
        for size in ["Small", "Medium", "Large"]:
            row[f"{size}_pct"] = (row[size] / total * 100) if total > 0 else 0.0
        records.append(row)

    comp_df = pd.DataFrame(records).set_index("year")

    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = np.zeros(len(CENSUS_YEARS))
    x_pos   = np.arange(len(CENSUS_YEARS))

    for size in ["Small", "Medium", "Large"]:
        vals = comp_df[f"{size}_pct"].values
        ax.bar(
            x_pos, vals, bottom=bottoms,
            color=SIZE_COLORS[size], label=size,
            edgecolor="white", linewidth=0.8, width=0.55,
        )
        # Label each segment if ≥5%
        for xi, (bot, val) in enumerate(zip(bottoms, vals)):
            if val >= 5:
                ax.text(xi, bot + val / 2, f"{val:.0f}%",
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in CENSUS_YEARS], fontsize=10)
    ax.set_ylabel("Share of Total Operations (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(f"{animal} CAFO Composition by Size Class — Rural US Counties", fontsize=11)
    ax.legend(title="Size Class", loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{today_str}_M{m_idx}_{animal.lower()}_size_composition.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


# =============================================================================
# FIGURE N: All animals × S/M/L — grouped stacked bar, opacity by size class
# X = census year; 3 color groups (cattle/hogs/chickens); within each group,
# stacked S/M/L with opacity varying (S=light, M=mid, L=full).
# Y = national total operation counts (log scale for readability).
# =============================================================================
_ALL_COMBOS = [
    ("Cattle",   "Small",  "cafo_cattle_small"),
    ("Cattle",   "Medium", "cafo_cattle_medium"),
    ("Cattle",   "Large",  "cafo_cattle_large"),
    ("Hogs",     "Small",  "cafo_hogs_small"),
    ("Hogs",     "Medium", "cafo_hogs_medium"),
    ("Hogs",     "Large",  "cafo_hogs_large"),
    ("Chickens", "Small",  "cafo_chickens_small"),
    ("Chickens", "Medium", "cafo_chickens_medium"),
    ("Chickens", "Large",  "cafo_chickens_large"),
]

# National sums per (animal, size, year)
agg_records = []
for animal, size, col in _ALL_COMBOS:
    for yr in CENSUS_YEARS:
        df_yr = df[df["year"] == yr]
        total = df_yr[col].sum(skipna=True) if col in df_yr.columns else 0.0
        agg_records.append({"animal": animal, "size": size, "year": yr, "total_ops": total})
agg_df = pd.DataFrame(agg_records)

fig, ax = plt.subplots(figsize=(12, 6))

n_years   = len(CENSUS_YEARS)
n_animals = 3
group_w   = 0.22     # width of each animal group
bar_gap   = 0.04     # gap between animal groups
total_w   = n_animals * group_w + (n_animals - 1) * bar_gap

year_positions = np.arange(n_years)
animal_order   = ["Cattle", "Hogs", "Chickens"]
size_order     = ["Small", "Medium", "Large"]

legend_handles = []

for ai, animal in enumerate(animal_order):
    base_color = ANIMAL_COLORS[animal]
    # Convert hex to RGB for opacity manipulation via matplotlib
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(base_color)

    x_offset = (ai - 1) * (group_w + bar_gap)  # center the 3-group cluster on each year tick

    for yr_i, yr in enumerate(CENSUS_YEARS):
        sub = agg_df[(agg_df["animal"] == animal) & (agg_df["year"] == yr)].set_index("size")
        bottom = 0.0
        for size in size_order:
            val = sub.loc[size, "total_ops"] if size in sub.index else 0.0
            alpha = SIZE_ALPHAS[size]
            face_color = (*rgb, alpha)
            bar = ax.bar(
                yr_i + x_offset, val,
                bottom=bottom,
                width=group_w,
                color=face_color,
                edgecolor="white",
                linewidth=0.5,
            )
            bottom += val

# Build legend: animal color patches + size opacity patches
import matplotlib.patches as mpatches
for animal in animal_order:
    legend_handles.append(
        mpatches.Patch(facecolor=ANIMAL_COLORS[animal], label=animal)
    )
legend_handles.append(mpatches.Patch(color="none", label=""))  # spacer
for size in size_order:
    legend_handles.append(
        mpatches.Patch(facecolor="grey", alpha=SIZE_ALPHAS[size], label=f"{size} size")
    )

ax.set_xticks(year_positions)
ax.set_xticklabels([str(y) for y in CENSUS_YEARS], fontsize=11)
ax.set_ylabel("Total Operations (count, national rural counties)", fontsize=10)
ax.set_title(
    "CAFO Operations by Animal Type and Size Class — Rural US Counties, Census Years\n"
    "Color = animal type   |   Opacity = size class (light = small, full = large)",
    fontsize=11,
)
ax.legend(handles=legend_handles, loc="upper right", fontsize=9, ncol=2,
          framealpha=0.8, title="Animal / Size")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_N_all_animals_size_opacity.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)


print(f"\nAll CAFO composition figures saved to: {out_dir}")
