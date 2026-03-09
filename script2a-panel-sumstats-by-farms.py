#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick summary-stats and mapping pipeline for county-year merged panel.
Outputs to: Dropbox/Mental/Data/merged/figs/panel-sumstats-by-farms
"""

from packages import *
from functions import *

try:
    import plotly.express as px
except Exception:
    px = None


# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------
clean_dir = os.path.join(db_data, "clean")
merged_dir = os.path.join(db_data, "merged")
figs_dir = os.path.join(merged_dir, "figs")
out_dir = os.path.join(figs_dir, "panel-sumstats-by-farms")
maps_dir = os.path.join(out_dir, "maps")
plots_dir = os.path.join(out_dir, "plots")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(maps_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")

STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA",
    "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def latest_file(folder, pattern):
    hits = glob.glob(os.path.join(folder, pattern))
    if not hits:
        raise RuntimeError(f"No files found for pattern {pattern} in {folder}")
    return max(hits, key=os.path.getmtime)


def normalize_key(df):
    df = clean_cols(df.copy())
    if "fips" not in df.columns and "fips_generated" in df.columns:
        df = df.rename(columns={"fips_generated": "fips"})
    if "year" not in df.columns and "yr" in df.columns:
        df = df.rename(columns={"yr": "year"})
    if "fips" not in df.columns or "year" not in df.columns:
        raise KeyError("Missing key columns fips/year")

    df["fips"] = (
        df["fips"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fips", "year"]).copy()
    return df


def to_num(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(
        s.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def summarize_numeric(df, columns):
    rows = []
    n_total = len(df)
    for c in columns:
        s = to_num(df[c]) if c in df.columns else pd.Series(dtype="float64")
        nonmiss = int(s.notna().sum())
        rows.append(
            {
                "variable": c,
                "n_total": n_total,
                "non_missing_n": nonmiss,
                "fill_pct": (nonmiss / n_total * 100) if n_total else np.nan,
                "mean": float(s.mean()) if nonmiss else np.nan,
                "p50": float(s.quantile(0.50)) if nonmiss else np.nan,
                "p90": float(s.quantile(0.90)) if nonmiss else np.nan,
                "min": float(s.min()) if nonmiss else np.nan,
                "max": float(s.max()) if nonmiss else np.nan,
                "gt0_pct_among_nonmissing": float((s > 0).mean() * 100) if nonmiss else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_state_choropleths(state_year_df, value_col, years, title_prefix, out_prefix):
    if px is None:
        print(f"Plotly unavailable. Skipping maps: {out_prefix}")
        return
    for yr in years:
        d = state_year_df[state_year_df["year"] == yr].copy()
        d = d[d[value_col].notna()].copy()
        if d.empty:
            continue
        fig = px.choropleth(
            d,
            locations="state_abbrev",
            locationmode="USA-states",
            color=value_col,
            scope="usa",
            color_continuous_scale="OrRd",
            hover_data={"state_fips": True, value_col: ":,.2f"},
            title=f"{title_prefix} ({yr})",
        )
        out = os.path.join(maps_dir, f"{out_prefix}_{yr}.html")
        fig.write_html(out, include_plotlyjs="cdn")


# ---------------------------------------------------------------------
# Load latest merged panel
# ---------------------------------------------------------------------
merged_path = latest_file(merged_dir, "*_full_merged.csv")
print("Using merged panel:", merged_path)

df = pd.read_csv(merged_path, low_memory=False)
df = normalize_key(df)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()
df = df.sort_values(["fips", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Required columns check
# ---------------------------------------------------------------------
cafo_cols_new = [
    "cafo_cattle_small", "cafo_cattle_medium", "cafo_cattle_large",
    "cafo_hogs_small", "cafo_hogs_medium", "cafo_hogs_large",
    "cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large",
    "cafo_total_ops_all_animals", "cafo_total_ops_chickens",
]
missing_cafo = [c for c in cafo_cols_new if c not in df.columns]
if missing_cafo:
    raise RuntimeError(
        "Merged panel is missing new CAFO animal-size columns. "
        "Please rerun script1b-merge-dataclean.py first. "
        f"Missing: {missing_cafo}"
    )

for c in cafo_cols_new:
    df[c] = to_num(df[c])

fsis_col = "n_unique_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip"
fsis_size_cols = [
    "n_size_bucket_1_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_2_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_3_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_4_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_5_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_missing_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
]
for c in [fsis_col, *fsis_size_cols]:
    if c not in df.columns:
        raise RuntimeError(f"Missing required FSIS column in merged panel: {c}")
    df[c] = to_num(df[c])

poor_mental_col = "poor_mental_health_days_raw_value_mentalhealthrank_full"
if poor_mental_col not in df.columns:
    raise RuntimeError(f"Missing required mental-health column in merged panel: {poor_mental_col}")
df[poor_mental_col] = to_num(df[poor_mental_col])

df["state_fips"] = df["fips"].astype("string").str[:2]
df["state_abbrev"] = df["state_fips"].map(STATE_FIPS_TO_ABBR)


# ---------------------------------------------------------------------
# 1) county_coverage_check.csv
# ---------------------------------------------------------------------
rural_path = latest_file(clean_dir, "*rural-key*.csv")
rural_df = read_and_prepare(rural_path)
rural_df = normalize_key(rural_df)
if "non_large_metro" not in rural_df.columns:
    raise RuntimeError(f"'non_large_metro' not found in rural-key file: {rural_path}")
rural_df["non_large_metro"] = pd.to_numeric(rural_df["non_large_metro"], errors="coerce").astype("Int64")

rural_kept = rural_df.loc[rural_df["non_large_metro"] == 1, ["fips", "year"]].drop_duplicates()
merged_keys = df[["fips", "year"]].drop_duplicates()

coverage_row = pd.DataFrame(
    [
        {
            "run_date": today_str,
            "merged_file": os.path.basename(merged_path),
            "rural_key_file": os.path.basename(rural_path),
            "merged_rows": int(len(df)),
            "merged_unique_counties": int(df["fips"].nunique()),
            "rural_kept_rows": int(len(rural_kept)),
            "rural_kept_unique_counties": int(rural_kept["fips"].nunique()),
            "county_count_match": int(df["fips"].nunique() == rural_kept["fips"].nunique()),
            "key_set_exact_match": int(
                merged_keys.merge(rural_kept, on=["fips", "year"], how="outer", indicator=True)["_merge"].eq("both").all()
            ),
        }
    ]
)
coverage_row.to_csv(os.path.join(out_dir, "county_coverage_check.csv"), index=False)


# ---------------------------------------------------------------------
# 2) CAFO sumstats (county-year and state-year)
# ---------------------------------------------------------------------
cafo_county_stats = summarize_numeric(df, cafo_cols_new)
cafo_county_stats.to_csv(os.path.join(out_dir, "cafo_county_year_sumstats.csv"), index=False)

state_year_cafo = (
    df.groupby(["state_fips", "state_abbrev", "year"], as_index=False)[cafo_cols_new]
    .sum(min_count=1)
)
cafo_state_stats = summarize_numeric(state_year_cafo, cafo_cols_new)
cafo_state_stats.to_csv(os.path.join(out_dir, "cafo_state_year_sumstats.csv"), index=False)


# ---------------------------------------------------------------------
# 3) FSIS reliable-only county-year sumstats
# ---------------------------------------------------------------------
fsis_cols_reliable = [fsis_col, *fsis_size_cols]
fsis_stats = summarize_numeric(df, fsis_cols_reliable)
fsis_stats.to_csv(os.path.join(out_dir, "fsis_county_year_sumstats.csv"), index=False)


# ---------------------------------------------------------------------
# 4) mental_fill_share.csv
# ---------------------------------------------------------------------
mental_cols = [c for c in df.columns if c.endswith("_mentalhealthrank_full") and "mental" in c.lower()]
any_mental = df[mental_cols].notna().any(axis=1) if mental_cols else pd.Series(False, index=df.index)
mental_fill = pd.DataFrame(
    [
        {
            "run_date": today_str,
            "merged_file": os.path.basename(merged_path),
            "mental_column_count": int(len(mental_cols)),
            "n_rows": int(len(df)),
            "rows_with_any_mental_data": int(any_mental.sum()),
            "share_rows_with_any_mental_data_pct": float(any_mental.mean() * 100),
        }
    ]
)
mental_fill.to_csv(os.path.join(out_dir, "mental_fill_share.csv"), index=False)


# ---------------------------------------------------------------------
# 5) Maps
# ---------------------------------------------------------------------
# FSIS state choropleths: 2017-2023
fsis_state_year = (
    df.groupby(["year", "state_fips", "state_abbrev"], as_index=False)[fsis_col]
    .sum(min_count=1)
)
save_state_choropleths(
    fsis_state_year,
    value_col=fsis_col,
    years=list(range(2017, 2024)),
    title_prefix="FSIS Establishments (State-Year Total)",
    out_prefix="fsis_state_total_establishments",
)

# CAFO state choropleths: 2010-2015
cafo_state_year_map = (
    df.groupby(["year", "state_fips", "state_abbrev"], as_index=False)["cafo_total_ops_all_animals"]
    .sum(min_count=1)
)
save_state_choropleths(
    cafo_state_year_map,
    value_col="cafo_total_ops_all_animals",
    years=list(range(2010, 2016)),
    title_prefix="CAFO Total Ops (State-Year Total)",
    out_prefix="cafo_state_total_ops",
)


# ---------------------------------------------------------------------
# 6) State-by-state county dot plots for 2017
# ---------------------------------------------------------------------
dot_df = df[(df["year"] == 2017)].copy()
dot_df = dot_df[
    dot_df["cafo_total_ops_all_animals"].notna()
    & (dot_df["cafo_total_ops_all_animals"] > 0)
    & dot_df[poor_mental_col].notna()
].copy()

for st in sorted(dot_df["state_fips"].dropna().unique()):
    d = dot_df[dot_df["state_fips"] == st].copy()
    if d.empty:
        continue
    st_lbl = d["state_abbrev"].dropna().iloc[0] if d["state_abbrev"].notna().any() else st
    plt.figure(figsize=(7.2, 5.4))
    sns.scatterplot(
        data=d,
        x="cafo_total_ops_all_animals",
        y=poor_mental_col,
        s=24,
        alpha=0.65,
        edgecolor=None,
    )
    plt.title(f"{st_lbl} County Dots (2017): Poor Mental Health vs Total CAFO Ops")
    plt.xlabel("Total CAFO Operations (All Animals)")
    plt.ylabel("Poor Mental Health Days (Raw Value)")
    out = os.path.join(plots_dir, f"state_county_dots_{st_lbl}_2017.png")
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# 7) Chickens scatter facets (2010-2020)
# ---------------------------------------------------------------------
facet = df[df["year"].between(2010, 2020, inclusive="both")].copy()
facet = facet[
    facet["cafo_total_ops_chickens"].notna()
    & (facet["cafo_total_ops_chickens"] > 0)
    & facet[poor_mental_col].notna()
].copy()

facet["chicken_size_class"] = pd.NA
facet.loc[facet["cafo_chickens_large"] > 0, "chicken_size_class"] = "large_present"
facet.loc[(facet["cafo_chickens_large"] <= 0) & (facet["cafo_chickens_medium"] > 0), "chicken_size_class"] = "medium_present"
facet.loc[
    (facet["cafo_chickens_large"] <= 0)
    & (facet["cafo_chickens_medium"] <= 0)
    & (facet["cafo_chickens_small"] > 0),
    "chicken_size_class",
] = "small_only"
facet = facet[facet["chicken_size_class"].notna()].copy()

if not facet.empty:
    palette = {"small_only": "#6baed6", "medium_present": "#fd8d3c", "large_present": "#cb181d"}
    g = sns.relplot(
        data=facet,
        x="cafo_total_ops_chickens",
        y=poor_mental_col,
        hue="chicken_size_class",
        col="year",
        col_wrap=4,
        kind="scatter",
        s=18,
        alpha=0.65,
        palette=palette,
        height=3.0,
        aspect=1.15,
    )
    g.set_axis_labels("Chicken CAFO Operations (Total)", "Poor Mental Health Days (Raw Value)")
    g.fig.suptitle("Chicken CAFOs vs Poor Mental Health Days (2010-2020)", y=1.02)
    g.savefig(
        os.path.join(plots_dir, "chickens_cafo_vs_poor_mental_health_faceted_2010_2020.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(g.fig)

    facet[
        [
            "fips",
            "state_fips",
            "state_abbrev",
            "year",
            "cafo_total_ops_chickens",
            poor_mental_col,
            "cafo_chickens_small",
            "cafo_chickens_medium",
            "cafo_chickens_large",
            "chicken_size_class",
        ]
    ].to_csv(
        os.path.join(out_dir, "chickens_scatter_data_2010_2020.csv"),
        index=False,
    )

print("Saved outputs in:", out_dir)
