#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:56:54 2026

Summary statistics + missingness QA + visualization outputs for merged county-year panel.
"""

from packages import *
from functions import *


# Paths
merged_dir = os.path.join(db_data, "merged")
clean_dir = os.path.join(db_data, "clean")
figs_dir = os.environ.get("MENTAL_FIGS_DIR", os.path.join(merged_dir, "figs"))
os.makedirs(figs_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")


def latest_file(folder, pattern):
    hits = glob.glob(os.path.join(folder, pattern))
    if not hits:
        raise RuntimeError(f"No files found for pattern: {pattern} in {folder}")
    return max(hits, key=os.path.getmtime)


def normalize_key(df):
    df = clean_cols(df.copy())

    if "fips" not in df.columns and "fips_generated" in df.columns:
        df = df.rename(columns={"fips_generated": "fips"})
    if "year" not in df.columns and "yr" in df.columns:
        df = df.rename(columns={"yr": "year"})

    if "fips" not in df.columns or "year" not in df.columns:
        raise KeyError("Required key columns fips/year are missing.")

    df["fips"] = (
        df["fips"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


def to_numeric_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s2 = (
        s.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "(d)": pd.NA, "(z)": pd.NA, "na": pd.NA, "n/a": pd.NA})
    )
    return pd.to_numeric(s2, errors="coerce")


def numeric_candidates(df, cols, min_parse_rate=0.8):
    out = {}
    rows = []
    for c in cols:
        s = df[c]
        n_raw = s.notna().sum()
        s_num = to_numeric_series(s)
        n_num = s_num.notna().sum()
        parse_rate = (n_num / n_raw) if n_raw else 0
        is_numeric_like = pd.api.types.is_numeric_dtype(s) or (parse_rate >= min_parse_rate and n_num > 0)
        rows.append(
            {
                "variable": c,
                "raw_non_missing_n": int(n_raw),
                "numeric_non_missing_n": int(n_num),
                "parse_rate": parse_rate,
                "numeric_like": int(is_numeric_like),
            }
        )
        if is_numeric_like:
            out[c] = s_num
    meta = pd.DataFrame(rows).sort_values(["numeric_like", "parse_rate"], ascending=[False, False])
    return out, meta


def pick_high_coverage_variables(summary_numeric, n_pick=12):
    keep = summary_numeric[
        (summary_numeric["fill_pct"] >= 95)
        & (summary_numeric["nunique"] >= 15)
        & (~summary_numeric["variable"].isin(["year", "non_large_metro"]))
    ].copy()
    if keep.empty:
        keep = summary_numeric[
            (summary_numeric["fill_pct"] >= 85)
            & (summary_numeric["nunique"] >= 10)
            & (~summary_numeric["variable"].isin(["year", "non_large_metro"]))
        ].copy()
    keep = keep.sort_values(["fill_pct", "std"], ascending=[False, False])
    return keep["variable"].head(n_pick).tolist()


def safe_plot_close():
    plt.tight_layout()
    plt.close()


def write_excel_or_csv(base_path_xlsx, sheets_dict):
    """
    Try writing one Excel workbook. If openpyxl is unavailable, write one CSV per sheet.
    """
    try:
        with pd.ExcelWriter(base_path_xlsx) as xw:
            for sheet_name, obj in sheets_dict.items():
                if isinstance(obj, pd.Series):
                    obj = obj.to_frame("value")
                obj.to_excel(xw, sheet_name=sheet_name, index=False)
        print("Saved workbook:", base_path_xlsx)
        return
    except ModuleNotFoundError as e:
        if "openpyxl" not in str(e).lower():
            raise
        print("openpyxl not installed; writing CSV files instead.")

    base_no_ext = os.path.splitext(base_path_xlsx)[0]
    for sheet_name, obj in sheets_dict.items():
        csv_path = f"{base_no_ext}_{sheet_name}.csv"
        if isinstance(obj, pd.Series):
            obj = obj.to_frame("value")
        obj.to_csv(csv_path, index=False)
        print("Saved:", csv_path)


# ---------------------------------------------------------------------
# 1) Load merged panel
# ---------------------------------------------------------------------
merged_path = latest_file(merged_dir, "*_full_merged.csv")
print("Using merged file:", merged_path)

df = pd.read_csv(merged_path, low_memory=False)
df = normalize_key(df)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.dropna(subset=["fips", "year"]).copy()

dup_mask = df.duplicated(subset=["fips", "year"], keep=False)
dup_count = int(dup_mask.sum())
print("Duplicate (fips,year) rows:", dup_count)
if dup_count:
    dup_path = os.path.join(figs_dir, f"{today_str}_merged_duplicate_fips_year_rows.csv")
    df.loc[dup_mask].sort_values(["fips", "year"]).to_csv(dup_path, index=False)
    print("Saved duplicate-key rows for review:", dup_path)
    df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()

df = df.sort_values(["fips", "year"]).reset_index(drop=True)
print("Final panel rows (county-year):", len(df))


# ---------------------------------------------------------------------
# 2) Missingness tables
# ---------------------------------------------------------------------
key_cols = ["fips", "year"]
analysis_cols = [c for c in df.columns if c not in key_cols]
n_obs = len(df)

missing_overall = pd.DataFrame(
    {
        "variable": analysis_cols,
        "non_missing_n": [int(df[c].notna().sum()) for c in analysis_cols],
        "missing_n": [int(df[c].isna().sum()) for c in analysis_cols],
    }
)
missing_overall["fill_pct"] = missing_overall["non_missing_n"] / n_obs * 100
missing_overall["missing_pct"] = missing_overall["missing_n"] / n_obs * 100
missing_overall["dtype"] = [str(df[c].dtype) for c in analysis_cols]
missing_overall["nunique"] = [int(df[c].nunique(dropna=True)) for c in analysis_cols]
missing_overall = missing_overall.sort_values("missing_pct", ascending=False)

missing_by_year = (df.groupby("year")[analysis_cols].apply(lambda x: x.isna().mean() * 100)).sort_index()
fill_by_year = 100 - missing_by_year

row_fill_pct = df[analysis_cols].notna().mean(axis=1) * 100
row_fill_summary = row_fill_pct.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).to_frame("value")
row_fill_distribution = pd.DataFrame({"fips": df["fips"], "year": df["year"], "row_fill_pct": row_fill_pct})

missing_xlsx = os.path.join(figs_dir, f"{today_str}_merged_missingness_tables.xlsx")
write_excel_or_csv(
    missing_xlsx,
    {
        "missing_overall": missing_overall,
        "missing_pct_by_year": missing_by_year.reset_index(),
        "fill_pct_by_year": fill_by_year.reset_index(),
        "row_fill_summary": row_fill_summary.reset_index(),
        "row_fill_distribution": row_fill_distribution,
    },
)


# Optional heatmap preview for Excel-oriented missingness matrix
heat_vars = (
    missing_overall.loc[missing_overall["missing_pct"].between(1, 99), "variable"]
    .head(60)
    .tolist()
)
if heat_vars:
    plt.figure(figsize=(20, 8))
    sns.heatmap(
        missing_by_year[heat_vars],
        cmap="mako_r",
        cbar_kws={"label": "Missing %"},
    )
    plt.title("Missingness Heatmap by Year (Top 60 Variable Candidates)")
    plt.xlabel("Variable")
    plt.ylabel("Year")
    plt.xticks(rotation=90, fontsize=7)
    heatmap_path = os.path.join(figs_dir, f"{today_str}_missingness_heatmap_top60.png")
    plt.savefig(heatmap_path, dpi=220, bbox_inches="tight")
    safe_plot_close()
    print("Saved heatmap preview:", heatmap_path)


# ---------------------------------------------------------------------
# 3) Numeric summary statistics
# ---------------------------------------------------------------------
numeric_map, numeric_meta = numeric_candidates(df, analysis_cols, min_parse_rate=0.8)
numeric_df = pd.DataFrame(numeric_map, index=df.index)

stats_rows = []
for c in numeric_df.columns:
    s = numeric_df[c]
    stats_rows.append(
        {
            "variable": c,
            "count": int(s.notna().sum()),
            "missing_n": int(s.isna().sum()),
            "fill_pct": float(s.notna().mean() * 100),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "std": float(s.std()) if s.notna().any() else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "p01": float(s.quantile(0.01)) if s.notna().any() else np.nan,
            "p05": float(s.quantile(0.05)) if s.notna().any() else np.nan,
            "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
            "median": float(s.median()) if s.notna().any() else np.nan,
            "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
            "p95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
            "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "nunique": int(s.nunique(dropna=True)),
            "zeros_n": int((s == 0).sum()) if s.notna().any() else 0,
        }
    )

summary_numeric = pd.DataFrame(stats_rows).sort_values(["fill_pct", "std"], ascending=[False, False])
summary_all = missing_overall.merge(
    summary_numeric,
    how="left",
    on="variable",
    suffixes=("", "_num"),
)

summary_xlsx = os.path.join(figs_dir, f"{today_str}_merged_summary_stats.xlsx")
write_excel_or_csv(
    summary_xlsx,
    {
        "summary_all": summary_all,
        "summary_numeric": summary_numeric,
        "numeric_parse_meta": numeric_meta,
    },
)


# ---------------------------------------------------------------------
# 4) Violin plots for high-coverage variables
# ---------------------------------------------------------------------
high_cov_vars = pick_high_coverage_variables(summary_numeric, n_pick=12)
print("High-coverage variables selected for violin plots:", high_cov_vars)

violin_dir = os.path.join(figs_dir, "violin_high_coverage")
os.makedirs(violin_dir, exist_ok=True)

for var in high_cov_vars:
    s = numeric_df[var].dropna()
    if s.empty:
        continue
    plot_df = pd.DataFrame({"value": s})
    plt.figure(figsize=(6.5, 6))
    sns.violinplot(y="value", data=plot_df, inner="quartile", cut=0, color="#2a9d8f")
    plt.title(f"Distribution: {var}")
    plt.ylabel(var)
    plt.xlabel("")
    out = os.path.join(violin_dir, f"{today_str}_violin_{re.sub(r'[^a-zA-Z0-9_]+', '_', var)[:120]}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    safe_plot_close()


# ---------------------------------------------------------------------
# 5) CAFO-specific plots (commodity x class x size)
# ---------------------------------------------------------------------
cafo_path = latest_file(clean_dir, "*_cafo_ops_by_size_compact.csv")
print("Using CAFO compact file:", cafo_path)
cafo = pd.read_csv(cafo_path, low_memory=False)
cafo = normalize_key(cafo)

# keep only rural key rows from merged panel
allowed_keys = df[["fips", "year"]].drop_duplicates()
cafo = cafo.merge(allowed_keys, on=["fips", "year"], how="inner")

for c in ["small", "medium", "large"]:
    if c not in cafo.columns:
        cafo[c] = 0
    cafo[c] = pd.to_numeric(cafo[c], errors="coerce")

if "commodity_desc" in cafo.columns:
    cafo["commodity_desc"] = cafo["commodity_desc"].astype("string").str.lower().str.strip()
if "class_desc" in cafo.columns:
    cafo["class_desc"] = cafo["class_desc"].astype("string").str.lower().str.strip()
else:
    cafo["class_desc"] = "all_classes"

cafo_long = cafo.melt(
    id_vars=["fips", "year", "commodity_desc", "class_desc"],
    value_vars=["small", "medium", "large"],
    var_name="size_bucket",
    value_name="ops_count",
)
cafo_long["ops_count"] = pd.to_numeric(cafo_long["ops_count"], errors="coerce")
cafo_long = cafo_long.dropna(subset=["ops_count"]).copy()

cafo_violin_dir = os.path.join(figs_dir, "cafo_violin")
os.makedirs(cafo_violin_dir, exist_ok=True)

for commodity in ["cattle", "chickens", "hogs"]:
    sub = cafo_long[cafo_long["commodity_desc"] == commodity].copy()
    if sub.empty:
        continue
    top_classes = (
        sub.groupby("class_desc")["ops_count"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index
        .tolist()
    )
    sub = sub[sub["class_desc"].isin(top_classes)].copy()
    plt.figure(figsize=(14, 7))
    sns.violinplot(
        data=sub,
        x="class_desc",
        y="ops_count",
        hue="size_bucket",
        cut=0,
        inner="quartile",
    )
    plt.title(f"CAFO Distribution by Class and Size: {commodity.title()}")
    plt.xlabel("Class")
    plt.ylabel("Operations Count")
    plt.xticks(rotation=45, ha="right")
    out = os.path.join(cafo_violin_dir, f"{today_str}_cafo_violin_{commodity}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    safe_plot_close()

# Higher-level comparison across commodity types and size buckets
if not cafo_long.empty:
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=cafo_long,
        x="commodity_desc",
        y="ops_count",
        hue="size_bucket",
        cut=0,
        inner="quartile",
    )
    plt.title("CAFO Ops Distribution by Commodity and Size")
    plt.xlabel("Commodity")
    plt.ylabel("Operations Count")
    out = os.path.join(cafo_violin_dir, f"{today_str}_cafo_violin_commodity_compare.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    safe_plot_close()


# ---------------------------------------------------------------------
# 6) Trends by CAFO size and selected covariates
# ---------------------------------------------------------------------
trend_dir = os.path.join(figs_dir, "trends")
os.makedirs(trend_dir, exist_ok=True)

size_small_col = "small_cafo_ops_by_size_compact"
size_medium_col = "medium_cafo_ops_by_size_compact"
size_large_col = "large_cafo_ops_by_size_compact"

for c in [size_small_col, size_medium_col, size_large_col]:
    if c not in df.columns:
        df[c] = np.nan
    df[c] = to_numeric_series(df[c])

df["cafo_size_group"] = "no_cafo"
df.loc[df[size_small_col].fillna(0) > 0, "cafo_size_group"] = "small_only"
df.loc[df[size_medium_col].fillna(0) > 0, "cafo_size_group"] = "medium_or_more"
df.loc[df[size_large_col].fillna(0) > 0, "cafo_size_group"] = "large_present"

trend_vars = [
    "some_college_raw_value_mh_mortality_fips_yr",
    "aggravated_assault_crime_fips_level_final",
    "violent_crime_raw_value_mh_mortality_fips_yr",
    "population_population_full",
]

for var in trend_vars:
    if var not in df.columns:
        continue
    tmp = df[["year", "cafo_size_group", var]].copy()
    tmp[var] = to_numeric_series(tmp[var])
    tmp = tmp.dropna(subset=[var, "year"])
    if tmp.empty:
        continue
    agg = tmp.groupby(["year", "cafo_size_group"], as_index=False)[var].mean()
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=agg, x="year", y=var, hue="cafo_size_group", marker="o")
    plt.title(f"Trend: {var} by CAFO Size Group")
    plt.xlabel("Year")
    plt.ylabel(f"Mean {var}")
    out = os.path.join(trend_dir, f"{today_str}_trend_{re.sub(r'[^a-zA-Z0-9_]+', '_', var)[:120]}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    safe_plot_close()

# Trend comparison by commodity x size from CAFO compact
if not cafo_long.empty:
    cafo_trend = (
        cafo_long.groupby(["year", "commodity_desc", "size_bucket"], as_index=False)["ops_count"]
        .mean()
    )
    g = sns.relplot(
        data=cafo_trend,
        x="year",
        y="ops_count",
        hue="commodity_desc",
        col="size_bucket",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
        height=4,
        aspect=1.2,
    )
    g.fig.subplots_adjust(top=0.83)
    g.fig.suptitle("Average Operations by Commodity Across CAFO Size Buckets")
    out = os.path.join(trend_dir, f"{today_str}_trend_cafo_size_by_commodity.png")
    g.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(g.fig)


print("Completed script2 outputs.")
print("Figures + tables written to:", figs_dir)
