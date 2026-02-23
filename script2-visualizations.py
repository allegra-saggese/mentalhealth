#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:56:54 2026

Large summary-statistics and visualization pipeline for merged county-year data.
Observation unit throughout: (fips, year).
"""

from packages import *
from functions import *
import re

try:
    import plotly.express as px
except Exception:
    px = None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
merged_dir = os.path.join(db_data, "merged")
clean_dir = os.path.join(db_data, "clean")
figs_dir = os.environ.get("MENTAL_FIGS_DIR", os.path.join(merged_dir, "figs"))
os.makedirs(figs_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")
COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
MAX_TREND_VARS = os.environ.get("MAX_TREND_VARS")  # optional cap for very large trend batch
STATE_PLOT_LIMIT = int(os.environ.get("STATE_PLOT_LIMIT", "12"))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
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
        n_raw = int(s.notna().sum())
        s_num = to_numeric_series(s)
        n_num = int(s_num.notna().sum())
        parse_rate = (n_num / n_raw) if n_raw else 0
        is_numeric_like = pd.api.types.is_numeric_dtype(s) or (parse_rate >= min_parse_rate and n_num > 0)
        rows.append(
            {
                "variable": c,
                "raw_non_missing_n": n_raw,
                "numeric_non_missing_n": n_num,
                "parse_rate": parse_rate,
                "numeric_like": int(is_numeric_like),
            }
        )
        if is_numeric_like:
            out[c] = s_num
    meta = pd.DataFrame(rows).sort_values(["numeric_like", "parse_rate"], ascending=[False, False])
    return out, meta


def safe_plot_close():
    plt.tight_layout()
    plt.close()


def safe_name(s, n=110):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s)).strip("_")[:n]


def write_csv_bundle(base_name, tables):
    """
    tables: dict{name -> DataFrame}
    """
    for name, obj in tables.items():
        path = os.path.join(figs_dir, f"{today_str}_{base_name}_{name}.csv")
        if isinstance(obj, pd.Series):
            obj = obj.to_frame("value")
        obj.to_csv(path, index=False)
        print("Saved:", path)


def chunked(items, k):
    for i in range(0, len(items), k):
        yield i // k + 1, items[i : i + k]


def weighted_mean(x, w):
    m = x.notna() & w.notna() & (w > 0)
    if not m.any():
        return np.nan
    return np.average(x[m], weights=w[m])


STATE_FIPS_TO_NAME = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California",
    "08": "Colorado", "09": "Connecticut", "10": "Delaware", "11": "District of Columbia",
    "12": "Florida", "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
    "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon",
    "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia",
    "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}


# ---------------------------------------------------------------------
# 1) Load merged panel
# ---------------------------------------------------------------------
merged_path = latest_file(merged_dir, "*_full_merged.csv")
print("Using merged file:", merged_path)

df = pd.read_csv(merged_path, low_memory=False)
df = normalize_key(df)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.dropna(subset=["fips", "year"]).copy()
df = df.sort_values(["fips", "year"]).reset_index(drop=True)

dup_mask = df.duplicated(subset=["fips", "year"], keep=False)
if dup_mask.any():
    dup_path = os.path.join(figs_dir, f"{today_str}_merged_duplicate_fips_year_rows.csv")
    df.loc[dup_mask].to_csv(dup_path, index=False)
    print("Saved duplicate-key rows:", dup_path)
    df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()

print("Final panel rows (county-year):", len(df))

key_cols = ["fips", "year"]
analysis_cols = [c for c in df.columns if c not in key_cols]
n_obs = len(df)


# ---------------------------------------------------------------------
# 2) Missingness tables + family heatmaps + row completeness
# ---------------------------------------------------------------------
missing_overall = pd.DataFrame(
    {
        "variable": analysis_cols,
        "non_missing_n": [int(df[c].notna().sum()) for c in analysis_cols],
        "missing_n": [int(df[c].isna().sum()) for c in analysis_cols],
        "dtype": [str(df[c].dtype) for c in analysis_cols],
        "nunique": [int(df[c].nunique(dropna=True)) for c in analysis_cols],
    }
)
missing_overall["fill_pct"] = missing_overall["non_missing_n"] / n_obs * 100
missing_overall["missing_pct"] = missing_overall["missing_n"] / n_obs * 100
missing_overall = missing_overall.sort_values("missing_pct", ascending=False)

missing_by_year = (df.groupby("year")[analysis_cols].apply(lambda x: x.isna().mean() * 100)).sort_index()
fill_by_year = 100 - missing_by_year

row_fill_pct = df[analysis_cols].notna().mean(axis=1) * 100
row_fill_distribution = pd.DataFrame({"fips": df["fips"], "year": df["year"], "row_fill_pct": row_fill_pct})
row_fill_summary = row_fill_pct.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame("value")

write_csv_bundle(
    "merged_missingness",
    {
        "overall": missing_overall,
        "pct_by_year": missing_by_year.reset_index(),
        "fill_by_year": fill_by_year.reset_index(),
        "row_fill_distribution": row_fill_distribution,
        "row_fill_summary": row_fill_summary.reset_index(),
    },
)

# Top-60 heatmap
top60 = missing_overall.loc[missing_overall["missing_pct"].between(1, 99), "variable"].head(60).tolist()
if top60:
    plt.figure(figsize=(20, 8))
    sns.heatmap(missing_by_year[top60], cmap="mako_r", cbar_kws={"label": "Missing %"})
    plt.title("Missingness Heatmap by Year (Top 60 Variables)")
    plt.xlabel("Variable")
    plt.ylabel("Year")
    plt.xticks(rotation=90, fontsize=7)
    p = os.path.join(figs_dir, f"{today_str}_missingness_heatmap_top60.png")
    plt.savefig(p, dpi=220, bbox_inches="tight")
    safe_plot_close()
    print("Saved:", p)

# Missingness by variable family (separate heatmaps)
family_suffixes = [
    "cafo_ops_by_size_compact",
    "crime_fips_level_final",
    "mh_mortality_fips_yr",
    "population_full",
]


def detect_family(col):
    if col == "non_large_metro":
        return "rural_key"
    for suf in family_suffixes:
        if col.endswith(f"_{suf}"):
            return suf
    return "other"


family_df = pd.DataFrame({"variable": analysis_cols})
family_df["family"] = family_df["variable"].map(detect_family)
family_summary = (
    family_df.groupby("family")["variable"]
    .count()
    .rename("n_variables")
    .reset_index()
    .sort_values("n_variables", ascending=False)
)
write_csv_bundle("missingness_family", {"variable_family_counts": family_summary, "variable_family_map": family_df})

fam_dir = os.path.join(figs_dir, "missingness_family_heatmaps")
os.makedirs(fam_dir, exist_ok=True)

for family in sorted(family_df["family"].unique()):
    fam_vars = family_df.loc[family_df["family"] == family, "variable"].tolist()
    if not fam_vars:
        continue
    for chunk_idx, cols_chunk in chunked(fam_vars, 35):
        plt.figure(figsize=(max(10, 0.45 * len(cols_chunk)), 7))
        sns.heatmap(missing_by_year[cols_chunk], cmap="magma_r", cbar_kws={"label": "Missing %"})
        plt.title(f"Missingness by Year - Family: {family} (chunk {chunk_idx})")
        plt.xlabel("Variable")
        plt.ylabel("Year")
        plt.xticks(rotation=90, fontsize=7)
        out = os.path.join(fam_dir, f"{today_str}_missingness_{safe_name(family)}_chunk{chunk_idx}.png")
        plt.savefig(out, dpi=220, bbox_inches="tight")
        safe_plot_close()

# Row-completeness histograms
quality_dir = os.path.join(figs_dir, "quality")
os.makedirs(quality_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
sns.histplot(row_fill_distribution["row_fill_pct"], bins=40, color="#1f77b4")
plt.title("Row Completeness Distribution (All County-Year Rows)")
plt.xlabel("Row Fill %")
plt.ylabel("Count")
out = os.path.join(quality_dir, f"{today_str}_row_completeness_hist_all.png")
plt.savefig(out, dpi=220, bbox_inches="tight")
safe_plot_close()

plt.figure(figsize=(12, 7))
sns.boxplot(data=row_fill_distribution, x="year", y="row_fill_pct", color="#2a9d8f")
plt.title("Row Completeness by Year")
plt.xlabel("Year")
plt.ylabel("Row Fill %")
plt.xticks(rotation=45)
out = os.path.join(quality_dir, f"{today_str}_row_completeness_box_by_year.png")
plt.savefig(out, dpi=220, bbox_inches="tight")
safe_plot_close()


# County coverage maps by year (HTML interactive)
map_dir = os.path.join(figs_dir, "coverage_maps")
os.makedirs(map_dir, exist_ok=True)
if px is None:
    print("Plotly not available; skipping county coverage maps.")
else:
    for yr in sorted(row_fill_distribution["year"].dropna().astype(int).unique()):
        d = row_fill_distribution[row_fill_distribution["year"] == yr].copy()
        if d.empty:
            continue
        d["fips"] = d["fips"].astype(str).str.zfill(5)
        fig = px.choropleth(
            d,
            geojson=COUNTY_GEOJSON_URL,
            locations="fips",
            color="row_fill_pct",
            color_continuous_scale="YlGnBu",
            range_color=(0, 100),
            scope="usa",
            hover_data={"fips": True, "row_fill_pct": ":.1f"},
            title=f"County Coverage (Row Fill %) - {yr}",
        )
        fig.update_geos(fitbounds="locations", visible=False)
        out = os.path.join(map_dir, f"{today_str}_coverage_map_{yr}.html")
        fig.write_html(out, include_plotlyjs="cdn")
        print("Saved:", out)


# ---------------------------------------------------------------------
# 3) Summary stats for all variables
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
summary_all = missing_overall.merge(summary_numeric, how="left", on="variable")

write_csv_bundle(
    "merged_summary_stats",
    {
        "all_variables": summary_all,
        "numeric_variables": summary_numeric,
        "numeric_parse_meta": numeric_meta,
    },
)


# ---------------------------------------------------------------------
# 4) High-coverage variable violin plots
# ---------------------------------------------------------------------
high_cov = summary_numeric[
    (summary_numeric["fill_pct"] >= 90) & (summary_numeric["nunique"] >= 20)
].sort_values(["fill_pct", "std"], ascending=[False, False])
if high_cov.empty:
    high_cov = summary_numeric[
        (summary_numeric["fill_pct"] >= 60) & (summary_numeric["nunique"] >= 8)
    ].sort_values(["fill_pct", "std"], ascending=[False, False])
high_cov_vars = high_cov["variable"].head(20).tolist()
print("High-coverage vars selected for violins:", len(high_cov_vars))

violin_dir = os.path.join(figs_dir, "violin_high_coverage")
os.makedirs(violin_dir, exist_ok=True)

for var in high_cov_vars:
    s = numeric_df[var].dropna()
    if s.empty:
        continue
    plt.figure(figsize=(6.5, 6))
    sns.violinplot(y=s, inner="quartile", cut=0, color="#2a9d8f")
    plt.title(f"Distribution: {var}")
    plt.ylabel(var)
    plt.xlabel("")
    out = os.path.join(violin_dir, f"{today_str}_violin_{safe_name(var)}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    safe_plot_close()


# ---------------------------------------------------------------------
# 5) CAFO data for commodity/class/size analyses
# ---------------------------------------------------------------------
cafo_path = latest_file(clean_dir, "*_cafo_ops_by_size_compact.csv")
print("Using CAFO compact file:", cafo_path)
cafo = pd.read_csv(cafo_path, low_memory=False)
cafo = normalize_key(cafo)
cafo = cafo.merge(df[["fips", "year"]].drop_duplicates(), on=["fips", "year"], how="inner")

for col in ["small", "medium", "large"]:
    if col not in cafo.columns:
        cafo[col] = 0
    cafo[col] = pd.to_numeric(cafo[col], errors="coerce").fillna(0)

for col in ["commodity_desc", "class_desc"]:
    if col not in cafo.columns:
        cafo[col] = "unknown"
    cafo[col] = cafo[col].astype("string").str.lower().str.strip()

cafo_long = cafo.melt(
    id_vars=["fips", "year", "commodity_desc", "class_desc"],
    value_vars=["small", "medium", "large"],
    var_name="size_bucket",
    value_name="ops_count",
).dropna(subset=["ops_count"])

cafo_violin_dir = os.path.join(figs_dir, "cafo_violin")
os.makedirs(cafo_violin_dir, exist_ok=True)

# Violin: each commodity, class-level by size bucket
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
    sub = sub[sub["class_desc"].isin(top_classes)]
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=sub, x="class_desc", y="ops_count", hue="size_bucket", cut=0, inner="quartile")
    plt.title(f"CAFO Operations Distribution by Class and Size: {commodity.title()}")
    plt.xlabel("Class")
    plt.ylabel("Operations")
    plt.xticks(rotation=45, ha="right")
    out = os.path.join(cafo_violin_dir, f"{today_str}_cafo_violin_{commodity}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    safe_plot_close()

# Violin: cross-commodity by size
plt.figure(figsize=(10, 6))
sns.violinplot(data=cafo_long, x="commodity_desc", y="ops_count", hue="size_bucket", cut=0, inner="quartile")
plt.title("CAFO Operations by Commodity and Size")
plt.xlabel("Commodity")
plt.ylabel("Operations")
out = os.path.join(cafo_violin_dir, f"{today_str}_cafo_violin_commodity_compare.png")
plt.savefig(out, dpi=220, bbox_inches="tight")
safe_plot_close()


# ---------------------------------------------------------------------
# 6) Annual composition plots
# ---------------------------------------------------------------------
comp_dir = os.path.join(figs_dir, "cafo_composition")
os.makedirs(comp_dir, exist_ok=True)

# A) For each size: share by commodity over time
size_commodity = (
    cafo_long.groupby(["year", "size_bucket", "commodity_desc"], as_index=False)["ops_count"]
    .sum()
)
size_commodity["pct_within_size_year"] = (
    size_commodity["ops_count"]
    / size_commodity.groupby(["year", "size_bucket"])["ops_count"].transform("sum")
    * 100
)
size_commodity.to_csv(os.path.join(comp_dir, f"{today_str}_size_commodity_share.csv"), index=False)

for size in ["small", "medium", "large"]:
    d = size_commodity[size_commodity["size_bucket"] == size].copy()
    if d.empty:
        continue
    pivot = d.pivot_table(index="year", columns="commodity_desc", values="pct_within_size_year", aggfunc="sum").fillna(0)
    plt.figure(figsize=(10, 6))
    plt.stackplot(pivot.index, [pivot[c] for c in pivot.columns], labels=pivot.columns, alpha=0.85)
    plt.title(f"Commodity Share (%) Within {size.title()} CAFO Operations by Year")
    plt.xlabel("Year")
    plt.ylabel("Share (%)")
    plt.legend(loc="upper left")
    out = os.path.join(comp_dir, f"{today_str}_share_commodity_within_{size}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    safe_plot_close()

# B) For each commodity: share small/medium/large over time
commodity_size = (
    cafo_long.groupby(["year", "commodity_desc", "size_bucket"], as_index=False)["ops_count"]
    .sum()
)
commodity_size["pct_within_commodity_year"] = (
    commodity_size["ops_count"]
    / commodity_size.groupby(["year", "commodity_desc"])["ops_count"].transform("sum")
    * 100
)
commodity_size.to_csv(os.path.join(comp_dir, f"{today_str}_commodity_size_share.csv"), index=False)

for commodity in ["cattle", "chickens", "hogs"]:
    d = commodity_size[commodity_size["commodity_desc"] == commodity].copy()
    if d.empty:
        continue
    pivot = d.pivot_table(index="year", columns="size_bucket", values="pct_within_commodity_year", aggfunc="sum").fillna(0)
    pivot = pivot[[c for c in ["small", "medium", "large"] if c in pivot.columns]]
    plt.figure(figsize=(10, 6))
    plt.stackplot(pivot.index, [pivot[c] for c in pivot.columns], labels=pivot.columns, alpha=0.85)
    plt.title(f"Size Share (%) Within {commodity.title()} Operations by Year")
    plt.xlabel("Year")
    plt.ylabel("Share (%)")
    plt.legend(loc="upper left")
    out = os.path.join(comp_dir, f"{today_str}_share_size_within_{commodity}.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    safe_plot_close()


# ---------------------------------------------------------------------
# 7) Ridge-like plots for crime and mental health
# ---------------------------------------------------------------------
ridge_dir = os.path.join(figs_dir, "ridge")
os.makedirs(ridge_dir, exist_ok=True)

ridge_targets = [
    "aggravated_assault_crime_fips_level_final",
    "violent_crime_raw_value_mh_mortality_fips_yr",
    "poor_mental_health_days_raw_value_mh_mortality_fips_yr",
    "frequent_mental_distress_raw_value_mh_mortality_fips_yr",
]

for var in ridge_targets:
    if var not in df.columns:
        continue
    tmp = df[["year", var]].copy()
    tmp[var] = to_numeric_series(tmp[var])
    tmp = tmp.dropna()
    if tmp.empty:
        continue
    tmp["year"] = tmp["year"].astype(int).astype(str)
    # Keep years with enough support to draw KDE
    yr_counts = tmp["year"].value_counts()
    keep_years = yr_counts[yr_counts >= 40].index.tolist()
    tmp = tmp[tmp["year"].isin(keep_years)]
    if tmp.empty:
        continue
    g = sns.FacetGrid(tmp, row="year", hue="year", aspect=12, height=0.35, palette="viridis")
    g.map(sns.kdeplot, var, fill=True, alpha=0.9, linewidth=1)
    g.map(sns.kdeplot, var, color="white", lw=0.5)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.fig.subplots_adjust(hspace=-0.6)
    g.fig.suptitle(f"Ridgeline-style KDE by Year: {var}", y=1.02)
    out = os.path.join(ridge_dir, f"{today_str}_ridge_{safe_name(var)}.png")
    g.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(g.fig)


# ---------------------------------------------------------------------
# 8) Massive trend plots: every numeric variable vs CAFO counts
# ---------------------------------------------------------------------
trend_dir = os.path.join(figs_dir, "trends_all_variables")
os.makedirs(trend_dir, exist_ok=True)

cafo_count = (
    cafo.groupby(["fips", "year", "commodity_desc"], as_index=False)[["small", "medium", "large"]]
    .sum()
)

pop_col = "population_population_full" if "population_population_full" in df.columns else None
if pop_col is not None:
    df[pop_col] = to_numeric_series(df[pop_col])

skip_vars = {
    "year",
    "non_large_metro",
    "small_cafo_ops_by_size_compact",
    "medium_cafo_ops_by_size_compact",
    "large_cafo_ops_by_size_compact",
    "any_large_cafo_cafo_ops_by_size_compact",
    "any_medium_or_large_cafo_cafo_ops_by_size_compact",
}
outcome_vars = [c for c in numeric_df.columns if c not in skip_vars and not c.startswith("n_rows")]

if MAX_TREND_VARS:
    try:
        cap = int(MAX_TREND_VARS)
        if cap > 0:
            outcome_vars = outcome_vars[:cap]
    except Exception:
        pass

pd.DataFrame({"variable": outcome_vars}).to_csv(
    os.path.join(trend_dir, f"{today_str}_trend_variable_list.csv"), index=False
)
print("Trend variables to plot:", len(outcome_vars))

for idx, var in enumerate(outcome_vars, start=1):
    if var not in df.columns:
        continue
    df[var] = to_numeric_series(df[var])
    if df[var].notna().sum() < 30:
        continue

    for commodity in ["cattle", "chickens", "hogs"]:
        cc = cafo_count[cafo_count["commodity_desc"] == commodity][["fips", "year", "small", "medium", "large"]].copy()
        if cc.empty:
            continue
        rhs_cols = ["fips", "year", var]
        if pop_col and pop_col != var:
            rhs_cols.append(pop_col)
        sub = cc.merge(df[rhs_cols], on=["fips", "year"], how="left")
        if sub[var].notna().sum() < 30:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        size_cols = ["small", "medium", "large"]
        size_titles = ["Small CAFO count", "Medium CAFO count", "Large CAFO count"]

        for i, (size_col, stitle) in enumerate(zip(size_cols, size_titles)):
            panel = sub[["year", var, size_col] + ([pop_col] if pop_col else [])].copy()
            panel = panel.dropna(subset=["year"])
            agg = panel.groupby("year", as_index=False).agg(
                outcome_mean=(var, "mean"),
                cafo_count=(size_col, "sum"),
            )
            if pop_col:
                w = panel.dropna(subset=[var, pop_col]).copy()
                if not w.empty:
                    w["weighted_val"] = w[var] * w[pop_col]
                    wmean = (
                        w.groupby("year", as_index=False)
                        .agg(weighted_sum=("weighted_val", "sum"), weight_sum=(pop_col, "sum"))
                    )
                    wmean["outcome_wmean"] = wmean["weighted_sum"] / wmean["weight_sum"]
                    wmean = wmean[["year", "outcome_wmean"]]
                    agg = agg.merge(wmean, on="year", how="left")
                else:
                    agg["outcome_wmean"] = np.nan
            else:
                agg["outcome_wmean"] = np.nan

            ax = axes[i]
            ax.plot(agg["year"], agg["outcome_mean"], marker="o", color="#1f77b4", label="Outcome mean")
            if agg["outcome_wmean"].notna().any():
                ax.plot(agg["year"], agg["outcome_wmean"], marker="s", linestyle="--", color="#2ca02c", label="Outcome weighted mean")
            ax.set_title(stitle)
            ax.set_xlabel("Year")
            if i == 0:
                ax.set_ylabel(f"{var} (mean)")

            ax2 = ax.twinx()
            ax2.plot(agg["year"], agg["cafo_count"], marker="^", color="#d62728", alpha=0.8, label="CAFO count")
            if i == 2:
                ax2.set_ylabel("CAFO operations")

        handles, labels = axes[0].get_legend_handles_labels()
        h2, l2 = axes[0].twinx().get_legend_handles_labels()
        fig.legend(handles + h2, labels + l2, loc="upper center", ncol=3, frameon=False)
        fig.suptitle(f"{commodity.title()}: {var} vs CAFO size counts over time", y=1.03)
        fig.tight_layout()
        out = os.path.join(trend_dir, f"{today_str}_trend_{safe_name(commodity)}_{safe_name(var)}.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)

    if idx % 20 == 0:
        print(f"Trend progress: {idx}/{len(outcome_vars)} variables")


# ---------------------------------------------------------------------
# 9) Binned scatters for key outcomes
# ---------------------------------------------------------------------
binned_dir = os.path.join(figs_dir, "binned_scatter")
os.makedirs(binned_dir, exist_ok=True)

key_outcomes = [
    "aggravated_assault_crime_fips_level_final",
    "violent_crime_raw_value_mh_mortality_fips_yr",
    "some_college_raw_value_mh_mortality_fips_yr",
    "population_population_full",
]

for var in key_outcomes:
    if var not in df.columns:
        continue
    v = df[["fips", "year", var]].copy()
    v[var] = to_numeric_series(v[var])
    for commodity in ["cattle", "chickens", "hogs"]:
        cc = cafo_count[cafo_count["commodity_desc"] == commodity][["fips", "year", "small", "medium", "large"]].copy()
        if cc.empty:
            continue
        m = cc.merge(v, on=["fips", "year"], how="left")
        for size_col in ["small", "medium", "large"]:
            d = m[[size_col, var]].dropna().copy()
            d = d[d[size_col] > 0]
            if len(d) < 50 or d[size_col].nunique() < 10:
                continue
            try:
                d["bin"] = pd.qcut(d[size_col], q=20, duplicates="drop")
            except Exception:
                continue
            g = d.groupby("bin", observed=True).agg(x=(size_col, "mean"), y=(var, "mean"), n=("bin", "size")).reset_index(drop=True)
            if g.empty:
                continue
            plt.figure(figsize=(7, 5))
            sns.scatterplot(data=g, x="x", y="y", size="n", legend=False, color="#1f77b4")
            sns.lineplot(data=g, x="x", y="y", color="#d62728")
            plt.title(f"Binned scatter: {var} vs {commodity} {size_col} CAFO count")
            plt.xlabel(f"{commodity} {size_col} CAFO count (bin mean)")
            plt.ylabel(f"{var} (bin mean)")
            out = os.path.join(
                binned_dir, f"{today_str}_binned_{safe_name(var)}_{safe_name(commodity)}_{size_col}.png"
            )
            plt.savefig(out, dpi=220, bbox_inches="tight")
            safe_plot_close()


# ---------------------------------------------------------------------
# 10) State-level trends for states with at least one large CAFO
# ---------------------------------------------------------------------
state_dir = os.path.join(figs_dir, "state_trends_large_cafo")
os.makedirs(state_dir, exist_ok=True)

cafo_state = cafo_count.copy()
cafo_state["state_fips"] = cafo_state["fips"].astype(str).str[:2]
cafo_state["state_name"] = cafo_state["state_fips"].map(STATE_FIPS_TO_NAME).fillna(cafo_state["state_fips"])

# States with at least one large CAFO county-year observation
state_large = (
    cafo_state.groupby("state_fips", as_index=False)["large"]
    .sum()
    .rename(columns={"large": "large_ops_total"})
)
state_large = state_large[state_large["large_ops_total"] > 0].copy()
state_large["state_name"] = state_large["state_fips"].map(STATE_FIPS_TO_NAME).fillna(state_large["state_fips"])
state_large = state_large.sort_values("large_ops_total", ascending=False)
state_large.to_csv(os.path.join(state_dir, f"{today_str}_states_with_large_cafo.csv"), index=False)

# Plot several states (top by large-CAFO intensity)
selected_states = state_large["state_fips"].head(STATE_PLOT_LIMIT).tolist()
print("States with large CAFO (total):", len(state_large))
print("States selected for state trend plots:", len(selected_states))

# Outcome variables to compare against CAFO quantity at state-year level
state_outcomes = [
    "aggravated_assault_crime_fips_level_final",
    "violent_crime_raw_value_mh_mortality_fips_yr",
    "poor_mental_health_days_raw_value_mh_mortality_fips_yr",
    "some_college_raw_value_mh_mortality_fips_yr",
    "population_population_full",
]
state_outcomes = [v for v in state_outcomes if v in df.columns]

df_state = df.copy()
df_state["state_fips"] = df_state["fips"].astype(str).str[:2]
df_state["state_name"] = df_state["state_fips"].map(STATE_FIPS_TO_NAME).fillna(df_state["state_fips"])

for v in state_outcomes:
    df_state[v] = to_numeric_series(df_state[v])

for st in selected_states:
    st_name = STATE_FIPS_TO_NAME.get(st, st)
    for commodity in ["cattle", "chickens", "hogs"]:
        ca = cafo_state[(cafo_state["state_fips"] == st) & (cafo_state["commodity_desc"] == commodity)].copy()
        if ca.empty:
            continue
        ca_agg = ca.groupby("year", as_index=False)[["small", "medium", "large"]].sum()
        ca_agg["total_cafo_ops"] = ca_agg[["small", "medium", "large"]].sum(axis=1)

        for out_var in state_outcomes:
            y = (
                df_state[df_state["state_fips"] == st]
                .groupby("year", as_index=False)[out_var]
                .mean()
                .rename(columns={out_var: "outcome_mean"})
            )
            z = ca_agg.merge(y, on="year", how="inner").dropna(subset=["outcome_mean"])
            if z.empty:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
            for i, size_col in enumerate(["small", "medium", "large"]):
                ax = axes[i]
                ax.plot(z["year"], z["outcome_mean"], marker="o", color="#1f77b4", label="Outcome mean")
                ax.set_title(f"{size_col.title()} CAFO count")
                ax.set_xlabel("Year")
                if i == 0:
                    ax.set_ylabel(out_var)
                ax2 = ax.twinx()
                ax2.plot(z["year"], z[size_col], marker="^", color="#d62728", label=f"{size_col} ops")
                if i == 2:
                    ax2.set_ylabel("CAFO ops count")

            handles, labels = axes[0].get_legend_handles_labels()
            h2, l2 = axes[0].twinx().get_legend_handles_labels()
            fig.legend(handles + h2, labels + l2, loc="upper center", ncol=2, frameon=False)
            fig.suptitle(f"{st_name} ({st}) - {commodity.title()} CAFO vs {out_var}", y=1.03)
            fig.tight_layout()
            out = os.path.join(
                state_dir,
                f"{today_str}_state_{safe_name(st_name)}_{safe_name(commodity)}_{safe_name(out_var)}.png",
            )
            fig.savefig(out, dpi=180, bbox_inches="tight")
            plt.close(fig)


print("Completed script2 outputs.")
print("Output folder:", figs_dir)
