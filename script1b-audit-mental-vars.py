#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mental-health coverage and panel-integrity audit for the merged county-year panel.
Outputs to: Dropbox/Mental/Data/merged/figs/panel-sumstats-by-farms

Quick purpose:
- Audits panel key integrity (`fips, year`) and source-duplication markers.
- Inventories mental-health variables and their year-by-year coverage.
- Checks cross-source duplication consistency for key mental outcomes.
- Writes QA tables used for summary-stat and data-readiness review.
"""

from packages import *
from functions import *


# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------
merged_dir = os.path.join(db_data, "merged")
out_dir = os.path.join(merged_dir, "figs", "panel-sumstats-by-farms")
os.makedirs(out_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")

MENTAL_KEYWORDS = (
    "poor_mental_health_days",
    "frequent_mental_distress",
    "mental_health_providers",
    "suicide",
    "depress",
    "anxiety",
    "excessive_drinking",
    "drug_overdose",
)

CORE_MENTAL_OUTCOMES = [
    "poor_mental_health_days_raw_value_mentalhealthrank_full",
    "frequent_mental_distress_raw_value_mentalhealthrank_full",
    "mental_health_providers_raw_value_mentalhealthrank_full",
    "excessive_drinking_raw_value_mentalhealthrank_full",
]

CORE_CONTEXT_COLS = [
    "deaths_cdc_county_year_deathsofdespair",
    "crude_rate_cdc_county_year_deathsofdespair",
    "population_population_full",
    "n_unique_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
latest_file = latest_file_glob
to_num = to_numeric_series
normalize_key = normalize_panel_key


def fill_pct(series):
    return float(series.notna().mean() * 100)


# ---------------------------------------------------------------------
# Load merged panel
# ---------------------------------------------------------------------
merged_path = latest_file(merged_dir, "*_full_merged.csv")
df = pd.read_csv(merged_path, low_memory=False)
df = normalize_key(df)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()
df = df.sort_values(["fips", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# 1) Panel integrity + source-duplication guards
# ---------------------------------------------------------------------
integrity = pd.DataFrame(
    [
        {
            "merged_file": os.path.basename(merged_path),
            "n_rows": int(len(df)),
            "n_unique_counties": int(df["fips"].nunique()),
            "n_unique_years": int(df["year"].nunique()),
            "year_min": int(df["year"].min()),
            "year_max": int(df["year"].max()),
            "n_duplicate_fips_year_rows": int(df.duplicated(["fips", "year"]).sum()),
        }
    ]
)
integrity.to_csv(os.path.join(out_dir, "qa_panel_integrity_mental_focus.csv"), index=False)

nrow_cols = sorted([c for c in df.columns if c.startswith("n_rows_")])
src_dup_rows = []
for c in nrow_cols:
    s = to_num(df[c])
    gt1 = (s > 1)
    src_dup_rows.append(
        {
            "source_n_rows_col": c,
            "rows_with_n_rows_gt_1": int(gt1.sum()),
            "pct_rows_with_n_rows_gt_1": float(gt1.mean() * 100),
            "max_n_rows": int(s.max()) if s.notna().any() else np.nan,
        }
    )
src_dup = pd.DataFrame(src_dup_rows)
src_dup.to_csv(os.path.join(out_dir, "qa_panel_source_duplication_counts.csv"), index=False)


# ---------------------------------------------------------------------
# 2) Mental variable inventory (focused)
# ---------------------------------------------------------------------
mh_rank_cols = [c for c in df.columns if c.endswith("_mentalhealthrank_full")]
mh_focus_cols = []
for c in mh_rank_cols:
    base = c[: -len("_mentalhealthrank_full")]
    if any(k in base for k in MENTAL_KEYWORDS):
        mh_focus_cols.append(c)

mh_focus_cols = sorted(set(mh_focus_cols))

inv_rows = []
for c in mh_focus_cols:
    raw = df[c]
    non = raw.notna()
    by_year_fill = non.groupby(df["year"]).mean() * 100
    years_with_data = df.loc[non, "year"]
    inv_rows.append(
        {
            "variable": c,
            "dtype": str(raw.dtype),
            "non_missing_n": int(non.sum()),
            "fill_pct": fill_pct(raw),
            "first_year_with_data": int(years_with_data.min()) if non.any() else np.nan,
            "last_year_with_data": int(years_with_data.max()) if non.any() else np.nan,
            "n_years_with_data": int(years_with_data.nunique()) if non.any() else 0,
            "yearly_fill_pct_min_positive": float(by_year_fill[by_year_fill > 0].min()) if (by_year_fill > 0).any() else 0.0,
            "yearly_fill_pct_max": float(by_year_fill.max()),
        }
    )

inventory = pd.DataFrame(inv_rows).sort_values(["fill_pct", "variable"], ascending=[False, True])
inventory.to_csv(os.path.join(out_dir, "qa_mental_focus_variable_inventory.csv"), index=False)


# ---------------------------------------------------------------------
# 3) Core mental outcomes: overall + by-year + county reach
# ---------------------------------------------------------------------
core_cols = [c for c in CORE_MENTAL_OUTCOMES if c in df.columns]
context_cols = [c for c in CORE_CONTEXT_COLS if c in df.columns]

overall_rows = []
for c in core_cols:
    s = to_num(df[c])
    by_year = s.notna().groupby(df["year"]).mean() * 100
    years = by_year[by_year > 0].index.tolist()
    overall_rows.append(
        {
            "variable": c,
            "non_missing_n": int(s.notna().sum()),
            "fill_pct_overall_panel": fill_pct(s),
            "first_year_with_data": int(min(years)) if years else np.nan,
            "last_year_with_data": int(max(years)) if years else np.nan,
            "n_years_with_data": int(len(years)),
            "min_positive_year_fill_pct": float(by_year[by_year > 0].min()) if years else 0.0,
            "max_year_fill_pct": float(by_year.max()),
            "median": float(s.median()) if s.notna().any() else np.nan,
            "p90": float(s.quantile(0.90)) if s.notna().any() else np.nan,
        }
    )
overall = pd.DataFrame(overall_rows).sort_values("variable")
overall.to_csv(os.path.join(out_dir, "qa_mental_outcome_core_overall.csv"), index=False)

county_any_rows = []
for c in core_cols:
    s = to_num(df[c])
    county_any_rows.append(
        {
            "variable": c,
            "n_counties_with_any_data": int(df.loc[s.notna(), "fips"].nunique()),
            "n_years_with_any_data": int(df.loc[s.notna(), "year"].nunique()),
        }
    )
county_any = pd.DataFrame(county_any_rows).sort_values("variable")
county_any.to_csv(os.path.join(out_dir, "qa_mental_outcome_counties_any.csv"), index=False)

year_rows = []
for yr, g in df.groupby("year", as_index=False):
    row = {"year": int(yr), "n_county_rows": int(len(g))}
    for c in [*core_cols, *context_cols]:
        s = to_num(g[c])
        row[f"fill_pct__{c}"] = float(s.notna().mean() * 100)
        row[f"non_missing_n__{c}"] = int(s.notna().sum())
    year_rows.append(row)

by_year = pd.DataFrame(year_rows).sort_values("year")
by_year.to_csv(os.path.join(out_dir, "qa_mental_outcome_core_by_year.csv"), index=False)


# ---------------------------------------------------------------------
# 4) Cross-source duplicate checks
# ---------------------------------------------------------------------
pairs = [
    (
        "poor_mental_health_days_raw_value_mentalhealthrank_full",
        "poor_mental_health_days_raw_value_mh_mortality_fips_yr",
    ),
    (
        "frequent_mental_distress_raw_value_mentalhealthrank_full",
        "frequent_mental_distress_raw_value_mh_mortality_fips_yr",
    ),
]

dupe_rows = []
for a, b in pairs:
    if a not in df.columns or b not in df.columns:
        continue
    sa = to_num(df[a])
    sb = to_num(df[b])
    m = sa.notna() & sb.notna()
    n = int(m.sum())
    if n == 0:
        dupe_rows.append(
            {
                "var_a": a,
                "var_b": b,
                "n_overlap": 0,
                "corr_on_overlap": np.nan,
                "exact_match_pct_on_overlap": np.nan,
                "median_abs_diff_on_overlap": np.nan,
                "p90_abs_diff_on_overlap": np.nan,
            }
        )
        continue

    diff = (sa[m] - sb[m]).abs()
    dupe_rows.append(
        {
            "var_a": a,
            "var_b": b,
            "n_overlap": n,
            "corr_on_overlap": float(sa[m].corr(sb[m])),
            "exact_match_pct_on_overlap": float((diff < 1e-12).mean() * 100),
            "median_abs_diff_on_overlap": float(diff.median()),
            "p90_abs_diff_on_overlap": float(diff.quantile(0.90)),
        }
        )

dupe_df = pd.DataFrame(dupe_rows)
if dupe_df.empty:
    dupe_df = pd.DataFrame(
        [
            {
                "var_a": pd.NA,
                "var_b": pd.NA,
                "n_overlap": 0,
                "corr_on_overlap": np.nan,
                "exact_match_pct_on_overlap": np.nan,
                "median_abs_diff_on_overlap": np.nan,
                "p90_abs_diff_on_overlap": np.nan,
                "note": "No duplicated mental-outcome sources present in merged panel (expected with direct CDC merge path).",
            }
        ]
    )
dupe_df.to_csv(os.path.join(out_dir, "qa_mental_crosssource_duplicate_check.csv"), index=False)


# ---------------------------------------------------------------------
# 5) Short memo
# ---------------------------------------------------------------------
memo_lines = [
    f"# Mental Coverage Audit ({today_str})",
    "",
    "## Scope",
    f"- Merged input: `{os.path.basename(merged_path)}`",
    f"- Unit: county-year (`fips`, `year`)",
    f"- Rows: `{len(df):,}` | Counties: `{df['fips'].nunique():,}` | Years: `{int(df['year'].min())}`-`{int(df['year'].max())}`",
    "",
    "## Outputs",
    "- `qa_panel_integrity_mental_focus.csv`",
    "- `qa_panel_source_duplication_counts.csv`",
    "- `qa_mental_focus_variable_inventory.csv`",
    "- `qa_mental_outcome_core_overall.csv`",
    "- `qa_mental_outcome_core_by_year.csv`",
    "- `qa_mental_outcome_counties_any.csv`",
    "- `qa_mental_crosssource_duplicate_check.csv`",
    "",
    "## Notes",
    "- Overall fill rates are over the full panel window (2000-2023), so variables that start in later years will look lower in the overall metric.",
    "- Use `qa_mental_outcome_core_by_year.csv` for year-appropriate coverage interpretation.",
]

memo_path = os.path.join(out_dir, "qa_mental_coverage_memo.md")
with open(memo_path, "w", encoding="utf-8") as f:
    f.write("\n".join(memo_lines) + "\n")

print("Saved mental coverage audit outputs to:", out_dir)


# =============================================================================
# SECTION 6: QA VISUALIZATIONS
# Added to give reviewers immediate visual diagnostics rather than requiring
# manual inspection of CSV outputs. Sections 6a–6e are self-contained and
# append outputs to the same out_dir.
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


# -----------------------------------------------------------------------------
# 6a: Coverage heatmap — outcome × year
# The most important QA figure: shows at a glance which outcomes are usable
# in which years. Replaces having to read multiple coverage CSVs manually.
# -----------------------------------------------------------------------------
COVERAGE_OUTCOMES = {
    "Poor Mental Health Days (CHR)": "poor_mental_health_days_raw_value_mentalhealthrank_full",
    "Frequent Mental Distress (CHR)": "frequent_mental_distress_raw_value_mentalhealthrank_full",
    "Excessive Drinking (CHR)": "excessive_drinking_raw_value_mentalhealthrank_full",
    "Violent Crime Rate (CHR)": "violent_crime_raw_value_mentalhealthrank_full",
    "Homicide Rate (CHR)": "homicides_raw_value_mentalhealthrank_full",
    "Aggravated Assault (UCR)": "aggravated_assault_crime_fips_level_final",
    "Deaths of Despair Crude Rate (CDC)": "crude_rate_cdc_county_year_deathsofdespair",
    "CAFO Total Ops": "cafo_total_ops_all_animals",
    "FSIS Establishments": "n_unique_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "County Population": "population_population_full",
}

years_all = sorted(df["year"].dropna().astype(int).unique())
cov_matrix = {}
for label, col in COVERAGE_OUTCOMES.items():
    row = {}
    for yr in years_all:
        if col not in df.columns:
            row[yr] = np.nan
            continue
        s = to_num(df.loc[df["year"] == yr, col])
        row[yr] = round(s.notna().mean() * 100, 1)
    cov_matrix[label] = row

cov_df = pd.DataFrame(cov_matrix).T
cov_df.columns = [str(y) for y in cov_df.columns]

fig, ax = plt.subplots(figsize=(max(18, len(years_all) * 0.95), max(4.5, len(COVERAGE_OUTCOMES) * 0.6)))
sns.heatmap(
    cov_df.astype(float),
    ax=ax,
    cmap="YlGn",
    vmin=0, vmax=100,
    annot=True, fmt=".0f", annot_kws={"size": 7},
    linewidths=0.3,
    cbar_kws={"label": "% Non-Missing"},
)
ax.set_title(
    f"Data Coverage by Outcome and Year — % Non-Missing\n"
    f"Rural (non-large-metro) counties only | n ≈ {df['fips'].nunique():,} counties",
    fontsize=11, pad=10,
)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=9)
plt.tight_layout()
heatmap_path = os.path.join(out_dir, f"{today_str}_qa_coverage_heatmap.png")
fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")
plt.close(fig)
cov_df.to_csv(os.path.join(out_dir, f"{today_str}_qa_coverage_matrix.csv"))
print("Saved:", heatmap_path)


# -----------------------------------------------------------------------------
# 6b: Panel balance — N counties observed per year, and overall balance
# -----------------------------------------------------------------------------
balance_by_year = []
for yr, g in df.groupby("year"):
    balance_by_year.append({
        "year": int(yr),
        "n_county_year_obs": int(len(g)),
        "n_unique_fips": int(g["fips"].nunique()),
    })
balance_by_year_df = pd.DataFrame(balance_by_year).sort_values("year")

# Counties that appear in every year of the panel
all_panel_years = set(df["year"].dropna().astype(int).unique())
fips_balanced = (
    df.groupby("fips")["year"]
    .apply(lambda s: all_panel_years.issubset(set(s.astype(int).tolist())))
)
n_balanced = int(fips_balanced.sum())
n_total_fips = int(df["fips"].nunique())

balance_summary = pd.DataFrame([{
    "n_fips_total": n_total_fips,
    "n_fips_in_all_years": n_balanced,
    "pct_balanced": round(n_balanced / n_total_fips * 100, 1) if n_total_fips else np.nan,
    "year_min": int(df["year"].min()),
    "year_max": int(df["year"].max()),
    "n_years_total": int(df["year"].nunique()),
}])

balance_by_year_df.to_csv(os.path.join(out_dir, f"{today_str}_qa_panel_balance_by_year.csv"), index=False)
balance_summary.to_csv(os.path.join(out_dir, f"{today_str}_qa_panel_balance_summary.csv"), index=False)
print("Saved: panel balance tables")


# -----------------------------------------------------------------------------
# 6c: CDC suppression audit
# The CDC deaths-of-despair crude rate has very low coverage in rural counties
# because small cell sizes are flagged as unreliable and suppressed. This
# section documents that explicitly so it is not mistaken for a data error.
# Note: The CDC deaths-of-despair measure is a composite (drug overdose +
# suicide + alcohol-related deaths combined). There is NO dedicated suicide
# column in the current pipeline. A separate CDC WONDER pull by ICD-10 cause
# codes X60-X84 would be required for suicide-specific analysis.
# -----------------------------------------------------------------------------
CDC_DEATHS_COL = "deaths_cdc_county_year_deathsofdespair"
CDC_CRUDE_COL = "crude_rate_cdc_county_year_deathsofdespair"
CDC_UNRELIABLE_COL = "is_unreliable_cdc_county_year_deathsofdespair"

cdc_cols_present = all(c in df.columns for c in [CDC_DEATHS_COL, CDC_CRUDE_COL, CDC_UNRELIABLE_COL])
if cdc_cols_present:
    supp_rows = []
    for yr, g in df.groupby("year"):
        deaths = to_num(g[CDC_DEATHS_COL])
        crude = to_num(g[CDC_CRUDE_COL])
        unreliable = to_num(g[CDC_UNRELIABLE_COL])
        n_with_deaths = int(deaths.notna().sum())
        supp_rows.append({
            "year": int(yr),
            "n_county_rows": int(len(g)),
            "n_with_deaths_data": n_with_deaths,
            "n_unreliable_flagged": int((unreliable == 1).sum()),
            "pct_unreliable_of_deaths_obs": (
                round((unreliable == 1).sum() / n_with_deaths * 100, 1) if n_with_deaths else np.nan
            ),
            "n_with_crude_rate": int(crude.notna().sum()),
            "pct_crude_rate_coverage": round(crude.notna().mean() * 100, 1),
        })
    supp_df = pd.DataFrame(supp_rows).sort_values("year")
    supp_path = os.path.join(out_dir, f"{today_str}_qa_cdc_suppression_audit.csv")
    supp_df.to_csv(supp_path, index=False)
    print(f"Saved: {supp_path}")
    print(
        f"  CDC crude rate avg coverage: {supp_df['pct_crude_rate_coverage'].mean():.1f}% — low by design "
        f"(small rural counties suppressed by CDC)."
    )
else:
    print("CDC suppression audit skipped: required columns not found.")


# -----------------------------------------------------------------------------
# 6d: Outcome distribution histograms — 2012 reference cross-section
# Shows skew and outlier structure for each outcome before visualization.
# 2012 chosen as reference: good CAFO coverage (97%) and partial MH coverage.
# -----------------------------------------------------------------------------
HIST_OUTCOMES = {
    "Poor Mental Health Days\n(avg days/month)": "poor_mental_health_days_raw_value_mentalhealthrank_full",
    "Excessive Drinking\n(% adults)": "excessive_drinking_raw_value_mentalhealthrank_full",
    "Violent Crime Rate\n(per 100k, CHR)": "violent_crime_raw_value_mentalhealthrank_full",
    "Homicide Rate\n(per 100k, CHR)": "homicides_raw_value_mentalhealthrank_full",
    "Aggravated Assault\n(UCR raw count)": "aggravated_assault_crime_fips_level_final",
    "Deaths of Despair\nCrude Rate (per 100k)": "crude_rate_cdc_county_year_deathsofdespair",
}

df_2012 = df[df["year"] == 2012].copy()
valid_hist = {lbl: col for lbl, col in HIST_OUTCOMES.items() if col in df_2012.columns}

n_cols_h = 3
n_rows_h = int(np.ceil(len(valid_hist) / n_cols_h))
fig, axes = plt.subplots(n_rows_h, n_cols_h, figsize=(n_cols_h * 4.5, n_rows_h * 3.5))
axes = axes.flatten()

for i, (lbl, col) in enumerate(valid_hist.items()):
    ax = axes[i]
    s = to_num(df_2012[col]).dropna()
    if s.empty:
        ax.set_visible(False)
        continue
    ax.hist(s, bins=40, color="#4292c6", edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.axvline(s.median(), color="#d6604d", lw=1.5, ls="--", label=f"Median: {s.median():.2f}")
    ax.axvline(s.mean(), color="#f7a35c", lw=1.2, ls=":", label=f"Mean: {s.mean():.2f}")
    ax.set_title(lbl, fontsize=9, pad=4)
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("N Counties", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.text(0.97, 0.97, f"n={len(s):,}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7, color="gray")

for j in range(len(valid_hist), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Outcome Variable Distributions — 2012 Cross-Section (Rural Counties)\n"
    "Check for skew and outlier structure before visualization and modeling.",
    fontsize=11, y=1.01,
)
plt.tight_layout()
dist_path = os.path.join(out_dir, f"{today_str}_qa_outcome_distributions_2012.png")
fig.savefig(dist_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", dist_path)


# -----------------------------------------------------------------------------
# 6e: CAFO per-capita distribution — raw vs. log-transformed
# Demonstrates why log(ops per 10k pop + 1) is the appropriate transformation
# for scatter plots: raw CAFO counts are heavily right-skewed, driven by
# county population size, and compress most counties to the left of any plot.
# -----------------------------------------------------------------------------
POP_COL = "population_population_full"
CAFO_COL = "cafo_total_ops_all_animals"

if POP_COL in df.columns and CAFO_COL in df.columns:
    cafo_pop = df[df["year"] == 2012].copy()
    pop = to_num(cafo_pop[POP_COL])
    cafo = to_num(cafo_pop[CAFO_COL])
    valid_mask = pop.notna() & cafo.notna() & (pop > 0)
    cafo_per10k = (cafo[valid_mask] / pop[valid_mask]) * 10_000
    log_cafo_per10k = np.log1p(cafo_per10k)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Raw CAFO count
    axes[0].hist(cafo[valid_mask], bins=50, color="#74c476", edgecolor="white", linewidth=0.3)
    axes[0].set_title("Raw CAFO Total Ops (2012)", fontsize=10)
    axes[0].set_xlabel("Total CAFO Operations (All Animals)")
    axes[0].set_ylabel("N Counties")
    axes[0].axvline(cafo[valid_mask].median(), color="#d6604d", lw=1.5, ls="--",
                    label=f"Median: {cafo[valid_mask].median():.0f}")
    axes[0].legend(fontsize=8)

    # Per 10k pop
    axes[1].hist(cafo_per10k, bins=50, color="#41ab5d", edgecolor="white", linewidth=0.3)
    axes[1].set_title("CAFO Ops per 10k Population (2012)", fontsize=10)
    axes[1].set_xlabel("CAFO Ops per 10k Residents")
    axes[1].set_ylabel("N Counties")
    axes[1].axvline(cafo_per10k.median(), color="#d6604d", lw=1.5, ls="--",
                    label=f"Median: {cafo_per10k.median():.1f}")
    axes[1].legend(fontsize=8)

    # Log per 10k — the preferred transformation
    axes[2].hist(log_cafo_per10k, bins=50, color="#238b45", edgecolor="white", linewidth=0.3)
    axes[2].set_title("log(CAFO Ops per 10k pop + 1) (2012)\nPreferred X-axis for scatter plots", fontsize=10)
    axes[2].set_xlabel("log(CAFO Ops per 10k pop + 1)")
    axes[2].set_ylabel("N Counties")
    axes[2].axvline(log_cafo_per10k.median(), color="#d6604d", lw=1.5, ls="--",
                    label=f"Median: {log_cafo_per10k.median():.2f}")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "CAFO Density Transformation — Why Log Per-Capita?\n"
        "Raw count (left) is right-skewed and confounded by county size. "
        "Log per-capita (right) is approximately symmetric.",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
    cafo_dist_path = os.path.join(out_dir, f"{today_str}_qa_cafo_percapita_distribution.png")
    fig.savefig(cafo_dist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Flag top 1% outliers by CAFO density
    outlier_df = cafo_pop.loc[valid_mask, ["fips", "year"]].copy()
    outlier_df["cafo_total_ops"] = cafo[valid_mask].values
    outlier_df["population"] = pop[valid_mask].values
    outlier_df["cafo_per10k_pop"] = cafo_per10k.values
    outlier_df["log_cafo_per10k"] = log_cafo_per10k.values
    p99_thresh = cafo_per10k.quantile(0.99)
    outlier_df = outlier_df[outlier_df["cafo_per10k_pop"] >= p99_thresh].sort_values(
        "cafo_per10k_pop", ascending=False
    )
    outlier_path = os.path.join(out_dir, f"{today_str}_qa_cafo_percapita_outliers_p99.csv")
    outlier_df.to_csv(outlier_path, index=False)
    print(f"Saved: {cafo_dist_path}")
    print(f"Saved: {outlier_path}  (p99 threshold: {p99_thresh:.1f} ops per 10k pop)")


# -----------------------------------------------------------------------------
# Append addendum to existing QA memo
# -----------------------------------------------------------------------------
memo_addendum = [
    "",
    "---",
    f"## QA Extension — Sections 6a–6e ({today_str})",
    "",
    "### 6a: Coverage Heatmap",
    f"- File: `{today_str}_qa_coverage_heatmap.png` + `{today_str}_qa_coverage_matrix.csv`",
    "- Key findings:",
    "  - `poor_mental_health_days` (CHR): usable from 2010 (~90%) onward.",
    "  - `frequent_mental_distress` (CHR): only available 2016+; not suitable for 2005–2015 panel.",
    "  - `excessive_drinking` (CHR): starts 2011 (~80%), full 2016+.",
    "  - `violent_crime_raw_value` (CHR): starts 2010 (43%), reliable from 2012 (~90%).",
    "  - `crude_rate` deaths of despair (CDC): <15% most years — see Section 6c.",
    "  - FSIS establishments: only 2017 — cross-section only.",
    "  - Recommended panel window for outcome analysis: **2010–2015**.",
    "",
    "### 6b: Panel Balance",
    f"- Files: `{today_str}_qa_panel_balance_by_year.csv`, `{today_str}_qa_panel_balance_summary.csv`",
    "",
    "### 6c: CDC Suppression Audit",
    f"- File: `{today_str}_qa_cdc_suppression_audit.csv`",
    "- **Important — low coverage is expected, not a bug**: CDC suppresses crude rates for counties",
    "  with fewer than 10 deaths in a cell. Most small rural counties fall below this threshold,",
    "  so the crude rate is missing for ~85–93% of observations. The raw death count is retained.",
    "- **There is no dedicated suicide-cause column in the current pipeline.**",
    "  The CDC 'deaths of despair' files are a composite measure: drug overdose + suicide +",
    "  alcohol-related deaths. For suicide-specific analysis, a separate pull from CDC WONDER",
    "  by ICD-10 cause code (X60–X84 intentional self-harm) is required.",
    "",
    "### 6d: Outcome Distributions",
    f"- File: `{today_str}_qa_outcome_distributions_2012.png` (2012 reference cross-section)",
    "- Check right skew in assault and homicide variables before regression modeling.",
    "",
    "### 6e: CAFO Per-Capita Distribution",
    f"- Files: `{today_str}_qa_cafo_percapita_distribution.png`,",
    f"  `{today_str}_qa_cafo_percapita_outliers_p99.csv`",
    "- Raw CAFO op counts are heavily right-skewed. Preferred X-axis transformation for all",
    "  scatter plots and regressions: **log(CAFO ops per 10k population + 1)**.",
    "  This corrects for county size and reduces leverage of extreme outliers.",
]

memo_path = os.path.join(out_dir, "qa_mental_coverage_memo.md")
with open(memo_path, "a", encoding="utf-8") as f:
    f.write("\n".join(memo_addendum) + "\n")

print("Updated: qa_mental_coverage_memo.md")
print("Done — all QA extension outputs saved to:", out_dir)
