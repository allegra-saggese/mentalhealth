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
COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

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
latest_file = latest_file_glob
to_num = to_numeric_series
normalize_key = normalize_panel_key


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


def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s)).strip("_").lower()


def save_county_map_one_year(df_year, value_col, year, title_prefix, out_prefix):
    if px is None:
        return
    d = df_year[["fips", value_col]].copy()
    d[value_col] = to_num(d[value_col])
    d = d[d["fips"].notna() & d[value_col].notna()].copy()
    if d.empty:
        return None

    vals = d[value_col].replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return None
    p99 = float(vals.quantile(0.99))
    vmax = p99 if np.isfinite(p99) and p99 > 0 else float(vals.max())
    range_color = (0, vmax) if np.isfinite(vmax) and vmax > 0 else None

    fig = px.choropleth(
        d,
        geojson=COUNTY_GEOJSON_URL,
        locations="fips",
        color=value_col,
        color_continuous_scale="OrRd",
        range_color=range_color,
        scope="usa",
        hover_data={"fips": True, value_col: ":,.2f"},
        title=f"{title_prefix} ({year})",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    out = os.path.join(maps_dir, f"{out_prefix}_{safe_name(value_col)}_{year}.html")
    fig.write_html(out, include_plotlyjs="cdn")
    return out


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

fsis_col = "n_unique_establishments_fsis"
fsis_size_cols = [
    "n_size_bucket_1_fsis",
    "n_size_bucket_2_fsis",
    "n_size_bucket_3_fsis",
    "n_size_bucket_4_fsis",
    "n_size_bucket_5_fsis",
    "n_size_bucket_missing_fsis",
]
for c in [fsis_col, *fsis_size_cols]:
    if c not in df.columns:
        raise RuntimeError(f"Missing required FSIS column in merged panel: {c}")
    df[c] = to_num(df[c])

poor_mental_col = "poor_mental_health_days"
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
mental_cols = [c for c in df.columns if c.endswith("") and "mental" in c.lower()]
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
# 6) County-level maps for 2017 (FSIS + CAFO animal x size)
# ---------------------------------------------------------------------
map_year = 2017
df_2017 = df[df["year"] == map_year].copy()

fsis_2017_cols = [
    "n_unique_establishments_fsis",
    "n_unique_est_size_combos_fsis",
    "n_slaughterhouse_present_establishments_fsis",
    "n_processing_present_establishments_fsis",
    "n_meat_slaughter_establishments_fsis",
    "n_poultry_slaughter_establishments_fsis",
    "n_size_bucket_1_fsis",
    "n_size_bucket_2_fsis",
    "n_size_bucket_3_fsis",
    "n_size_bucket_4_fsis",
    "n_size_bucket_5_fsis",
    "n_size_bucket_missing_fsis",
]
for c in fsis_2017_cols:
    if c not in df_2017.columns:
        raise RuntimeError(f"Missing FSIS map column in merged panel: {c}")

fsis_map_meta = []
for c in fsis_2017_cols:
    s = to_num(df_2017[c])
    row = {
        "year": map_year,
        "variable": c,
        "non_missing_n": int(s.notna().sum()),
        "positive_n": int((s > 0).sum()),
        "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
        "p90": float(s.quantile(0.90)) if s.notna().any() else np.nan,
        "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
        "max": float(s.max()) if s.notna().any() else np.nan,
    }
    fsis_map_meta.append(row)
    save_county_map_one_year(
        df_2017,
        value_col=c,
        year=map_year,
        title_prefix="FSIS County-Level Density",
        out_prefix="county_fsis",
    )
pd.DataFrame(fsis_map_meta).to_csv(
    os.path.join(out_dir, "fsis_county_map_2017_metric_summary.csv"),
    index=False,
)

cafo_animal_size_cols = [
    "cafo_cattle_small", "cafo_cattle_medium", "cafo_cattle_large",
    "cafo_hogs_small", "cafo_hogs_medium", "cafo_hogs_large",
    "cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large",
]

cafo_map_meta = []
for c in cafo_animal_size_cols:
    s = to_num(df_2017[c])
    row = {
        "year": map_year,
        "variable": c,
        "non_missing_n": int(s.notna().sum()),
        "positive_n": int((s > 0).sum()),
        "p50": float(s.quantile(0.50)) if s.notna().any() else np.nan,
        "p90": float(s.quantile(0.90)) if s.notna().any() else np.nan,
        "p99": float(s.quantile(0.99)) if s.notna().any() else np.nan,
        "max": float(s.max()) if s.notna().any() else np.nan,
    }
    cafo_map_meta.append(row)
    save_county_map_one_year(
        df_2017,
        value_col=c,
        year=map_year,
        title_prefix="CAFO County-Level Animal x Size",
        out_prefix="county_cafo_animal_size",
    )
pd.DataFrame(cafo_map_meta).to_csv(
    os.path.join(out_dir, "cafo_county_map_2017_metric_summary.csv"),
    index=False,
)


# ---------------------------------------------------------------------
# 7) State-by-state county dot plots for 2017
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
# 8) Chickens scatter facets (2010-2020)
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


# ---------------------------------------------------------------------
# 9) CAFO unit/count cross-check against pre-merged compact panel
# ---------------------------------------------------------------------
pre_cafo_path = latest_file(clean_dir, "*_cafo_ops_by_size_compact.csv")
pre = pd.read_csv(pre_cafo_path, low_memory=False)
pre = normalize_key(pre)

for c in ["commodity_desc", "small", "medium", "large"]:
    if c not in pre.columns:
        raise RuntimeError(f"Missing column in pre-merged CAFO compact file: {c}")
pre["commodity_desc"] = pre["commodity_desc"].astype("string").str.strip().str.lower()
pre = pre[pre["commodity_desc"].isin(["cattle", "hogs", "chickens"])].copy()
if "class_desc" in pre.columns:
    pre["class_desc"] = pre["class_desc"].astype("string").str.strip().str.lower()
    canonical_class_map = {"cattle": "incl calves", "hogs": "all classes", "chickens": "layers"}
    pre = pre[pre["class_desc"] == pre["commodity_desc"].map(canonical_class_map)].copy()
for c in ["small", "medium", "large"]:
    pre[c] = to_num(pre[c])

# Restrict pre-merged comparison to the same rural keys.
pre = pre.merge(rural_kept, on=["fips", "year"], how="inner")

pre_grp = (
    pre.groupby(["fips", "year", "commodity_desc"], as_index=False)[["small", "medium", "large"]]
    .sum(min_count=1)
)
pre_wide = pre_grp.pivot_table(
    index=["fips", "year"],
    columns="commodity_desc",
    values=["small", "medium", "large"],
    aggfunc="sum",
)
pre_wide.columns = [f"cafo_{commodity}_{size}" for size, commodity in pre_wide.columns]
pre_wide = pre_wide.reset_index()

for commodity in ["cattle", "hogs", "chickens"]:
    for size in ["small", "medium", "large"]:
        col = f"cafo_{commodity}_{size}"
        if col not in pre_wide.columns:
            pre_wide[col] = np.nan

pre_animal_cols = [
    "cafo_cattle_small", "cafo_cattle_medium", "cafo_cattle_large",
    "cafo_hogs_small", "cafo_hogs_medium", "cafo_hogs_large",
    "cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large",
]
pre_wide["cafo_total_ops_all_animals"] = pre_wide[pre_animal_cols].sum(axis=1, min_count=1)
pre_wide["cafo_total_ops_chickens"] = pre_wide[
    ["cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large"]
].sum(axis=1, min_count=1)

merged_cmp = df[["fips", "year", *pre_animal_cols, "cafo_total_ops_all_animals", "cafo_total_ops_chickens"]].copy()
cmp = merged_cmp.merge(pre_wide, on=["fips", "year"], how="left", suffixes=("_merged", "_pre"))

cross_rows = []
for col in [*pre_animal_cols, "cafo_total_ops_all_animals", "cafo_total_ops_chickens"]:
    m = to_num(cmp[f"{col}_merged"])
    p = to_num(cmp[f"{col}_pre"])
    both = m.notna() & p.notna()
    diff = (m - p).abs()
    cross_rows.append(
        {
            "variable": col,
            "n_keys_compared": int(both.sum()),
            "pct_exact_match_on_compared_keys": float((diff[both] == 0).mean() * 100) if both.any() else np.nan,
            "max_abs_diff": float(diff[both].max()) if both.any() else np.nan,
            "merged_sum": float(m.sum(skipna=True)),
            "premerged_sum": float(p.sum(skipna=True)),
            "merged_p90": float(m.quantile(0.90)) if m.notna().any() else np.nan,
            "premerged_p90": float(p.quantile(0.90)) if p.notna().any() else np.nan,
            "merged_max": float(m.max()) if m.notna().any() else np.nan,
            "premerged_max": float(p.max()) if p.notna().any() else np.nan,
        }
    )

crosswalk = pd.DataFrame(cross_rows)
crosswalk.to_csv(
    os.path.join(out_dir, "cafo_animal_size_crosscheck_vs_premerged.csv"),
    index=False,
)

unit_note = pd.DataFrame(
    [
        {
            "source_file": os.path.basename(pre_cafo_path),
            "inference": "CAFO animal-size values are operation/establishment counts, not animal head counts.",
            "evidence_1": "script0b-ag-raw.py builds ops_in_bin from value and labels summary columns as small/medium/large operation totals.",
            "evidence_2": "script0b-ag-raw.py comments: 'Keep operations counts in each inventory bin.'",
            "evidence_3": "Merged CAFO animal-size columns match pre-merged compact county-year totals exactly on compared keys.",
        }
    ]
)
unit_note.to_csv(os.path.join(out_dir, "cafo_unit_crosscheck_note.csv"), index=False)

print("Saved outputs in:", out_dir)
