#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script2a-panel-qa.py

Panel QA, missingness, composition, and summary statistics for the merged
county-year panel. Combines data-quality diagnostics with descriptive
composition analysis. All choropleth/county maps are in script2g-maps.py.

Sections:
  1  — CDC crude-rate sense check
  2  — Correlation tables + FSIS column QA + QA memo
  3  — FSIS size bins vs. poor mental health days (2017 cross-section)
  4  — Missingness tables, heatmaps, row completeness
  5  — Summary statistics for all numeric variables
  6  — Violin plots: high-coverage variables
  7  — CAFO composition from compact (violins + census-year stacked bars)
  8  — CAFO concentration: HHI, Gini, entry/exit by year
  9  — Ridge plots + trend plots + binned scatter (slow; cap with MAX_TREND_VARS)
  10 — State-level CAFO trends
  11 — Targeted sumstats: county coverage check, CAFO/FSIS/MH sumstats, crosscheck
  12 — State dot plots + chickens size-class facets

Inputs:  Data/merged/*_panel.csv, Data/merged/*_panel_fsis.csv,
         Data/clean/*_cafo_ops_by_size_compact.csv, Data/clean/*rural-key*.csv
Outputs: Data/output/tables/panel-qa/  (CSVs + QA memo)
         Data/output/figs/panel-qa/    (PNGs from sections 1–3)
         Data/output/figs/...          (subfolders for sections 4–12)
"""

from packages import *
from functions import *
import re

try:
    import plotly.express as px
except Exception:
    px = None

# ── Directories ───────────────────────────────────────────────────────────────
merged_dir     = os.path.join(db_data, "merged")
clean_dir      = os.path.join(db_data, "clean")
qa_dir         = os.path.join(tables_dir, "panel-qa")
plots_dir      = os.path.join(qa_dir, "plots")
figs_qa_dir    = os.path.join(figs_dir, "panel-qa", "plots")
for _d in (qa_dir, plots_dir, figs_qa_dir):
    os.makedirs(_d, exist_ok=True)

MAX_TREND_VARS   = os.environ.get("MAX_TREND_VARS")
STATE_PLOT_LIMIT = int(os.environ.get("STATE_PLOT_LIMIT", "12"))

STATE_FIPS_TO_NAME = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa", "20": "Kansas",
    "21": "Kentucky", "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", "28": "Mississippi",
    "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma",
    "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina",
    "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah",
    "50": "Vermont", "51": "Virginia", "53": "Washington", "54": "West Virginia",
    "55": "Wisconsin", "56": "Wyoming",
}
STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}

# ── Column name constants ─────────────────────────────────────────────────────
CDC_DEATHS  = "deaths_despair"
CDC_POP     = "population_despair"    # dropped from panel after cleanup; handled gracefully
CDC_CRUDE   = "crude_rate_despair"
COUNTY_NAME = "county_name_despair"   # dropped from panel after cleanup; handled gracefully
CENSUS_POP  = "population"
MIN_CORR_N  = 200
POP_COL     = "population"

# ── Helper functions ──────────────────────────────────────────────────────────
def numeric_candidates(df, cols, min_parse_rate=0.8):
    out, rows = {}, []
    for c in cols:
        s = df[c]
        n_raw = int(s.notna().sum())
        s_num = to_numeric_series(s)
        n_num = int(s_num.notna().sum())
        parse_rate = (n_num / n_raw) if n_raw else 0
        is_numeric_like = pd.api.types.is_numeric_dtype(s) or (parse_rate >= min_parse_rate and n_num > 0)
        rows.append({"variable": c, "raw_non_missing_n": n_raw, "numeric_non_missing_n": n_num,
                     "parse_rate": parse_rate, "numeric_like": int(is_numeric_like)})
        if is_numeric_like:
            out[c] = s_num
    meta = pd.DataFrame(rows).sort_values(["numeric_like", "parse_rate"], ascending=[False, False])
    return out, meta


def safe_plot_close():
    plt.tight_layout()
    plt.close()


def safe_name(s, n=110):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(s)).strip("_").lower()[:n]


def write_csv_bundle(base_name, tables):
    for name, obj in tables.items():
        path = os.path.join(tables_dir, f"{today_str}_{base_name}_{name}.csv")
        if isinstance(obj, pd.Series):
            obj = obj.to_frame("value")
        obj.to_csv(path, index=False)
        print("Saved:", path)


def chunked(items, k):
    for i in range(0, len(items), k):
        yield i // k + 1, items[i: i + k]


def get_series(df_like, col):
    obj = df_like[col]
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return obj


def gini_from_values(values):
    x = np.array(values, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def summarize_numeric(df, columns):
    rows = []
    n_total = len(df)
    for c in columns:
        s = to_numeric_series(df[c]) if c in df.columns else pd.Series(dtype="float64")
        nonmiss = int(s.notna().sum())
        rows.append({"variable": c, "n_total": n_total, "non_missing_n": nonmiss,
                     "fill_pct": (nonmiss / n_total * 100) if n_total else np.nan,
                     "mean": float(s.mean()) if nonmiss else np.nan,
                     "p50": float(s.quantile(0.50)) if nonmiss else np.nan,
                     "p90": float(s.quantile(0.90)) if nonmiss else np.nan,
                     "min": float(s.min()) if nonmiss else np.nan,
                     "max": float(s.max()) if nonmiss else np.nan,
                     "gt0_pct_among_nonmissing": float((s > 0).mean() * 100) if nonmiss else np.nan})
    return pd.DataFrame(rows)


def detect_family(col):
    if col.startswith("cafo_"):
        return "cafo"
    if col.endswith("_fsis"):
        return "fsis"
    if col.endswith("_crime") or col in {"violent_crime", "homicides"}:
        return "crime"
    if "_despair" in col or col in {"cdc_in_query", "deaths_is_zero",
                                    "crude_rate_from_census_pop", "deaths_per_10k_census_pop"}:
        return "cdc"
    if col == "population" or col.startswith("%_"):
        return "demographics"
    return "other"


def _safe_float(v):
    try:
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan


# ── Load panels ───────────────────────────────────────────────────────────────
merged_path = latest_file_glob(merged_dir, "*_panel.csv")
df = pd.read_csv(merged_path, low_memory=False)
df = normalize_panel_key(df, dropna=False)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.dropna(subset=["fips", "year"]).copy()
dup_mask = df.duplicated(subset=["fips", "year"], keep=False)
if dup_mask.any():
    df.loc[dup_mask].to_csv(
        os.path.join(tables_dir, f"{today_str}_merged_duplicate_fips_year_rows.csv"), index=False)
    df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()
df = df.sort_values(["fips", "year"]).reset_index(drop=True)
print(f"Panel:      {len(df):,} rows | {df['fips'].nunique():,} counties | "
      f"{int(df['year'].min())}–{int(df['year'].max())}")

fsis_path = latest_file_glob(merged_dir, "*_panel_fsis.csv")
df_fsis = pd.read_csv(fsis_path, low_memory=False)
df_fsis = normalize_panel_key(df_fsis)
df_fsis = df_fsis.loc[:, ~df_fsis.columns.duplicated()].copy()
df_fsis = df_fsis.sort_values(["fips", "year"]).reset_index(drop=True)
print(f"FSIS panel: {len(df_fsis):,} rows | {df_fsis['fips'].nunique():,} counties | "
      f"{int(df_fsis['year'].min())}–{int(df_fsis['year'].max())}")

cafo_path = latest_file_glob(clean_dir, "*_cafo_ops_by_size_compact.csv")
cafo = pd.read_csv(cafo_path, low_memory=False)
cafo = normalize_panel_key(cafo)
cafo = cafo.merge(df[["fips", "year"]].drop_duplicates(), on=["fips", "year"], how="inner")
for col in ["small", "medium", "large"]:
    if col not in cafo.columns:
        cafo[col] = 0
    cafo[col] = pd.to_numeric(cafo[col], errors="coerce").fillna(0)
for col in ["commodity_desc", "class_desc"]:
    if col not in cafo.columns:
        cafo[col] = "unknown"
    cafo[col] = cafo[col].astype("string").str.lower().str.strip()
_CANONICAL = {"cattle": "incl calves", "hogs": "all classes", "chickens": "layers"}
if "class_desc" in cafo.columns:
    _mask = cafo["commodity_desc"].map(_CANONICAL) == cafo["class_desc"]
    cafo = cafo[_mask | ~cafo["commodity_desc"].isin(_CANONICAL)].copy()
cafo_long = cafo.melt(
    id_vars=["fips", "year", "commodity_desc", "class_desc"],
    value_vars=["small", "medium", "large"],
    var_name="size_bucket", value_name="ops_count",
).dropna(subset=["ops_count"])
print(f"CAFO compact: {len(cafo):,} rows")

key_cols      = ["fips", "year"]
analysis_cols = [c for c in df.columns if c not in key_cols]
n_obs         = len(df)
df["state_fips"]  = df["fips"].astype("string").str[:2]
df["state_abbrev"] = df["state_fips"].map(STATE_FIPS_TO_ABBR)


# =============================================================================
# SECTION 1 — CDC crude-rate sense check
# =============================================================================
print("\n--- Section 1: CDC crude-rate sense check ---")

def _summarize_cdc(g):
    d      = g["diff_census_minus_cdc"]
    ad     = g["abs_diff_census_minus_cdc"]
    z      = g["z_diff_from_zero_sd"]
    poppct = g["population_pct_diff_census_vs_cdc"]
    recalc = g["crude_rate_recalc_census_pop"]
    cdc    = g["cdc_crude_rate"]
    return pd.Series({
        "n_rows":                    int(len(g)),
        "n_valid_diff":              int(d.notna().sum()),
        "mean_diff":                 float(d.mean())   if d.notna().any()  else np.nan,
        "median_diff":               float(d.median()) if d.notna().any()  else np.nan,
        "sd_diff":                   float(d.std(ddof=1)) if d.notna().sum() > 1 else np.nan,
        "mae_diff":                  float(ad.mean())  if ad.notna().any() else np.nan,
        "rmse_diff":                 float(np.sqrt(np.mean(d.dropna()**2))) if d.notna().any() else np.nan,
        "p90_abs_diff":              float(ad.quantile(0.90)) if ad.notna().any() else np.nan,
        "p95_abs_diff":              float(ad.quantile(0.95)) if ad.notna().any() else np.nan,
        "share_abs_diff_le_0_5":     float((ad <= 0.5).mean()*100) if ad.notna().any() else np.nan,
        "share_abs_diff_le_1_0":     float((ad <= 1.0).mean()*100) if ad.notna().any() else np.nan,
        "share_abs_diff_le_2_0":     float((ad <= 2.0).mean()*100) if ad.notna().any() else np.nan,
        "share_abs_z_gt_1":          float((z.abs()>1).mean()*100) if z.notna().any() else np.nan,
        "share_abs_z_gt_2":          float((z.abs()>2).mean()*100) if z.notna().any() else np.nan,
        "share_abs_z_gt_3":          float((z.abs()>3).mean()*100) if z.notna().any() else np.nan,
        "mean_population_pct_diff":  float(poppct.mean())   if poppct.notna().any() else np.nan,
        "median_population_pct_diff":float(poppct.median()) if poppct.notna().any() else np.nan,
        "corr_recalc_vs_cdc":        float(recalc.corr(cdc)) if recalc.notna().sum()>1 and cdc.notna().sum()>1 else np.nan,
    })

required = [CDC_DEATHS, CDC_CRUDE, CENSUS_POP]
missing  = [c for c in required if c not in df.columns]
if missing:
    print(f"WARNING: skipping CDC sense check — missing columns: {missing}")
else:
    df["cdc_deaths"]        = to_numeric_series(df[CDC_DEATHS])
    df["cdc_population"]    = to_numeric_series(df[CDC_POP]) if CDC_POP in df.columns else np.nan
    df["cdc_crude_rate"]    = to_numeric_series(df[CDC_CRUDE])
    df["census_population"] = to_numeric_series(df[CENSUS_POP])

    df["crude_rate_recalc_census_pop"] = np.where(
        df["census_population"] > 0,
        df["cdc_deaths"] / df["census_population"] * 100_000.0, np.nan)
    df["crude_rate_recalc_cdc_pop"] = np.where(
        df["cdc_population"] > 0,
        df["cdc_deaths"] / df["cdc_population"] * 100_000.0, np.nan)
    df["diff_census_minus_cdc"]     = df["crude_rate_recalc_census_pop"] - df["cdc_crude_rate"]
    df["abs_diff_census_minus_cdc"] = df["diff_census_minus_cdc"].abs()
    df["pct_diff_census_vs_cdc"]    = np.where(
        df["cdc_crude_rate"] != 0,
        (df["diff_census_minus_cdc"] / df["cdc_crude_rate"]) * 100.0, np.nan)
    df["population_diff_census_minus_cdc"]  = df["census_population"] - df["cdc_population"]
    df["population_pct_diff_census_vs_cdc"] = np.where(
        df["cdc_population"] != 0,
        (df["population_diff_census_minus_cdc"] / df["cdc_population"]) * 100.0, np.nan)
    sd_all = df["diff_census_minus_cdc"].std(ddof=1)
    df["z_diff_from_zero_sd"] = (
        df["diff_census_minus_cdc"] / sd_all if (pd.notna(sd_all) and sd_all > 0) else np.nan)
    year_sd = df.groupby("year")["diff_census_minus_cdc"].transform(lambda s: s.std(ddof=1))
    df["z_diff_from_zero_sd_within_year"] = np.where(year_sd > 0,
                                                      df["diff_census_minus_cdc"] / year_sd, np.nan)

    rename_map = {
        "crude_rate_recalc_census_pop":      "crude_rate_recalc_censuspop_cdcsense",
        "crude_rate_recalc_cdc_pop":         "crude_rate_recalc_cdcpop_cdcsense",
        "diff_census_minus_cdc":             "crude_rate_diff_census_minus_cdc_cdcsense",
        "abs_diff_census_minus_cdc":         "crude_rate_abs_diff_census_minus_cdc_cdcsense",
        "pct_diff_census_vs_cdc":            "crude_rate_pct_diff_census_vs_cdc_cdcsense",
        "z_diff_from_zero_sd":               "crude_rate_diff_z_from_zero_sd_cdcsense",
        "z_diff_from_zero_sd_within_year":   "crude_rate_diff_z_from_zero_sd_within_year_cdcsense",
        "population_diff_census_minus_cdc":  "population_diff_census_minus_cdc_cdcsense",
        "population_pct_diff_census_vs_cdc": "population_pct_diff_census_vs_cdc_cdcsense",
    }
    for src, dst in rename_map.items():
        df[dst] = df[src]

    cmp = df[
        df["cdc_deaths"].notna() & df["cdc_crude_rate"].notna()
        & df["census_population"].notna() & (df["census_population"] > 0)
    ].copy()

    overall = _summarize_cdc(cmp).to_frame().T
    overall.insert(0, "source_merged_file", os.path.basename(merged_path))
    overall.insert(1, "year_min", int(cmp["year"].min()) if not cmp.empty else np.nan)
    overall.insert(2, "year_max", int(cmp["year"].max()) if not cmp.empty else np.nan)

    by_year = (
        cmp.groupby("year", as_index=False)
        .apply(lambda g: _summarize_cdc(g), include_groups=False)
        .reset_index().drop(columns=["index"], errors="ignore")
        .sort_values("year")
    )

    outliers = cmp.copy()
    outliers["abs_z"] = outliers["crude_rate_diff_z_from_zero_sd_cdcsense"].abs()
    keep_cols = ["fips", "year",
                 COUNTY_NAME if COUNTY_NAME in outliers.columns else None,
                 "cdc_deaths", "census_population", "cdc_population", "cdc_crude_rate",
                 "crude_rate_recalc_censuspop_cdcsense", "crude_rate_recalc_cdcpop_cdcsense",
                 "crude_rate_diff_census_minus_cdc_cdcsense",
                 "crude_rate_abs_diff_census_minus_cdc_cdcsense",
                 "crude_rate_pct_diff_census_vs_cdc_cdcsense",
                 "crude_rate_diff_z_from_zero_sd_cdcsense",
                 "crude_rate_diff_z_from_zero_sd_within_year_cdcsense",
                 "population_pct_diff_census_vs_cdc_cdcsense", "abs_z"]
    keep_cols = [c for c in keep_cols if c is not None and c in outliers.columns]
    outliers  = outliers[keep_cols].sort_values("abs_z", ascending=False).head(1000)

    # Re-export panel + slices with CDC diagnostic columns appended
    _panel_no_fsis = df[[c for c in df.columns if not c.endswith("_fsis")]]
    _panel_no_fsis.to_csv(os.path.join(merged_dir, f"{today_str}_panel.csv"), index=False)
    _panel_no_fsis[_panel_no_fsis["year"].between(2005, 2010, inclusive="both")].to_csv(
        os.path.join(merged_dir, f"{today_str}_panel_05_10.csv"), index=False)
    _panel_no_fsis[_panel_no_fsis["year"].between(2010, 2020, inclusive="both")].to_csv(
        os.path.join(merged_dir, f"{today_str}_panel_10_20.csv"), index=False)
    _panel_no_fsis[_panel_no_fsis["year"].isin([2002, 2007, 2012, 2017, 2022])].to_csv(
        os.path.join(merged_dir, f"{today_str}_panel_census_years.csv"), index=False)

    overall.to_csv(os.path.join(qa_dir, f"{today_str}_qa_cdc_cruderate_sensecheck_overall.csv"), index=False)
    by_year.to_csv(os.path.join(qa_dir, f"{today_str}_qa_cdc_cruderate_sensecheck_by_year.csv"), index=False)
    outliers.to_csv(os.path.join(qa_dir, f"{today_str}_qa_cdc_cruderate_sensecheck_outliers_top1000.csv"), index=False)
    print(f"  Rows compared: {len(cmp):,} | Global SD(diff): {float(sd_all):.4f}")


# =============================================================================
# SECTION 2 — Correlation tables + FSIS column QA + QA memo
# =============================================================================
print("\n--- Section 2: Correlation tables + FSIS QA + memo ---")

cafo_cols = [
    "cafo_cattle_small", "cafo_cattle_medium", "cafo_cattle_large",
    "cafo_hogs_small",   "cafo_hogs_medium",   "cafo_hogs_large",
    "cafo_chickens_small","cafo_chickens_medium","cafo_chickens_large",
    "cafo_total_ops_all_animals", "cafo_total_ops_chickens",
]
mental_cols_corr = ["poor_mental_health_days", "frequent_mental_distress_per100k"]
population_cols  = [POP_COL]
fsis_12_cols = [
    "n_unique_establishments_fsis", "n_unique_est_size_combos_fsis",
    "n_slaughterhouse_present_establishments_fsis", "n_processing_present_establishments_fsis",
    "n_meat_slaughter_establishments_fsis", "n_poultry_slaughter_establishments_fsis",
    "n_size_bucket_1_fsis", "n_size_bucket_2_fsis", "n_size_bucket_3_fsis",
    "n_size_bucket_4_fsis", "n_size_bucket_5_fsis", "n_size_bucket_missing_fsis",
]

all_corr_cols  = [*cafo_cols, *mental_cols_corr, *population_cols]
existing_cols  = [c for c in all_corr_cols if c in df.columns]
for c in existing_cols:
    df[c] = to_numeric_series(df[c])

manifest = pd.DataFrame([
    {"variable": c, "family": (
        "cafo"       if c in cafo_cols       else
        "mental"     if c in mental_cols_corr else
        "population"
    )} for c in existing_cols
])
manifest.to_csv(os.path.join(qa_dir, "correlation_vars_manifest.csv"), index=False)

# Coverage by year
coverage_rows = []
for yr, g in df.groupby("year", as_index=False):
    row = {"year": int(yr)}
    for c in ["cafo_total_ops_all_animals", "cafo_total_ops_chickens",
              "poor_mental_health_days", "frequent_mental_distress_per100k",
              "deaths_despair", POP_COL]:
        if c in g.columns:
            row[f"fill_pct__{c}"] = float(g[c].notna().mean() * 100)
    coverage_rows.append(row)
fsis_cov = []
for yr, g in df_fsis.groupby("year", as_index=False):
    fsis_cov.append({"year": int(yr),
                     "fill_pct__n_unique_establishments_fsis":
                         float(g["n_unique_establishments_fsis"].notna().mean() * 100)
                         if "n_unique_establishments_fsis" in g.columns else np.nan})
fsis_cov_df = pd.DataFrame(fsis_cov)
cov_df = pd.DataFrame(coverage_rows).sort_values("year").merge(fsis_cov_df, on="year", how="left")
cov_df.to_csv(os.path.join(qa_dir, "qa_key_variable_coverage_by_year.csv"), index=False)

# FSIS 12-column QA (from panel_fsis)
fsis_q_rows = []
for c in fsis_12_cols:
    if c not in df_fsis.columns:
        continue
    s       = to_numeric_series(df_fsis[c])
    nonmiss = s.notna().sum()
    p99     = _safe_float(s.quantile(0.99)) if nonmiss else np.nan
    vmax    = _safe_float(s.max())          if nonmiss else np.nan
    fsis_q_rows.append({
        "variable": c, "non_missing_n": int(nonmiss),
        "fill_pct": float(s.notna().mean() * 100),
        "p50": _safe_float(s.quantile(0.50)) if nonmiss else np.nan,
        "p90": _safe_float(s.quantile(0.90)) if nonmiss else np.nan,
        "p99": p99, "max": vmax,
        "share_gt_1000_pct": _safe_float((s > 1000).mean() * 100) if nonmiss else np.nan,
        "qa_reliable_for_now": int((pd.notna(p99) and p99 <= 1000) and (pd.notna(vmax) and vmax <= 1000)),
    })
fsis_qa = pd.DataFrame(fsis_q_rows).sort_values(["qa_reliable_for_now", "share_gt_1000_pct"],
                                                  ascending=[False, True])
fsis_qa.to_csv(os.path.join(qa_dir, "qa_fsis_12_column_check.csv"), index=False)

# Pairwise correlations
def _pairwise_corr(df_num, columns, method, min_n=MIN_CORR_N):
    rows = []
    for i, a in enumerate(columns):
        sa = df_num[a]
        for b in columns[i + 1:]:
            sb = df_num[b]
            mask = sa.notna() & sb.notna()
            n = int(mask.sum())
            if n < min_n:
                continue
            corr = sa[mask].corr(sb[mask], method=method)
            if pd.isna(corr):
                continue
            rows.append({"var_a": a, "var_b": b, "corr": float(corr),
                         "abs_corr": abs(float(corr)), "n_obs_pairwise": n, "method": method})
    out = pd.DataFrame(rows)
    return out.sort_values("abs_corr", ascending=False).reset_index(drop=True) if not out.empty else out

windows = {
    "county_year_2010_2020": df[df["year"].between(2010, 2020, inclusive="both")].copy(),
    "county_year_2017_2023": df[df["year"].between(2017, 2023, inclusive="both")].copy(),
    "county_year_2017":      df[df["year"] == 2017].copy(),
}
for wname, sub in windows.items():
    use_cols = [c for c in existing_cols if c in sub.columns and sub[c].notna().any()]
    if not use_cols:
        continue
    sub_num = sub[use_cols].copy()
    for c in use_cols:
        sub_num[c] = to_numeric_series(sub_num[c])
    for method in ("pearson", "spearman"):
        long = _pairwise_corr(sub_num, use_cols, method=method)
        long.to_csv(os.path.join(qa_dir, f"correlation_long_{wname}_{method}.csv"), index=False)
        long.head(30).to_csv(os.path.join(qa_dir, f"correlation_top30_{wname}_{method}.csv"), index=False)
        sub_num[use_cols].corr(method=method, min_periods=MIN_CORR_N).to_csv(
            os.path.join(qa_dir, f"correlation_matrix_{wname}_{method}.csv"), index=True)

fsis_reliable = fsis_qa.loc[fsis_qa["qa_reliable_for_now"] == 1, "variable"].tolist()
fsis_flagged  = fsis_qa.loc[fsis_qa["qa_reliable_for_now"] == 0, "variable"].tolist()
coverage_check_path = os.path.join(qa_dir, "county_coverage_check.csv")
crosscheck_path     = os.path.join(qa_dir, "cafo_commodity_size_crosscheck_vs_premerged.csv")
coverage_check = pd.read_csv(coverage_check_path) if os.path.exists(coverage_check_path) else pd.DataFrame()
crosscheck     = pd.read_csv(crosscheck_path)     if os.path.exists(crosscheck_path)     else pd.DataFrame()
county_match = "n/a"
key_match    = "n/a"
if not coverage_check.empty:
    county_match = "PASS" if int(coverage_check["county_count_match"].iloc[0]) == 1 else "FAIL"
    key_match    = "PASS" if int(coverage_check["key_set_exact_match"].iloc[0]) == 1 else "FAIL"
cafo_cross = "n/a"
if not crosscheck.empty and "pct_exact_match_on_compared_keys" in crosscheck.columns:
    cafo_cross = (f"{crosscheck['pct_exact_match_on_compared_keys'].min():.1f}%–"
                  f"{crosscheck['pct_exact_match_on_compared_keys'].max():.1f}% exact-match")

memo_lines = [
    f"# QA Memo: Panel + Correlation Tables ({today_str})", "",
    "## 1) Scope",
    f"- Merged file: `{os.path.basename(merged_path)}`",
    f"- Rows: `{len(df):,}` | Counties: `{df['fips'].nunique():,}` | "
    f"Years: `{int(df['year'].min())}–{int(df['year'].max())}`", "",
    "## 2) Key QA Checks",
    f"- County-count check vs rural kept keys: **{county_match}**",
    f"- Exact key-set match vs rural kept keys: **{key_match}**",
    f"- CAFO commodity×size merged-vs-premerged cross-check: **{cafo_cross}**", "",
    "## 3) FSIS 12-Column QA",
    f"- Reliable ({len(fsis_reliable)}): {', '.join([f'`{c}`' for c in fsis_reliable]) or 'none'}",
    f"- Flagged ({len(fsis_flagged)}): {', '.join([f'`{c}`' for c in fsis_flagged]) or 'none'}", "",
    "## 4) Correlation Tables",
    f"- Pairwise minimum n: `{MIN_CORR_N}`. Windows: 2010–2020, 2017–2023, 2017 cross-section.",
    "- FSIS correlations not included (FSIS only in panel_fsis.csv, 2017+).", "",
]
with open(os.path.join(qa_dir, "qa_memo_panel_qa.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(memo_lines) + "\n")
print(f"  Outputs saved to: {qa_dir}")


# =============================================================================
# SECTION 3 — FSIS size bins vs. poor mental health days (2017 cross-section)
# =============================================================================
print("\n--- Section 3: FSIS size bins vs. poor mental health days (2017) ---")

_mental_col  = "poor_mental_health_days"
_size_cols   = ["n_size_bucket_1_fsis", "n_size_bucket_2_fsis", "n_size_bucket_3_fsis",
                "n_size_bucket_4_fsis", "n_size_bucket_5_fsis"]
_missing_s3  = [c for c in [_mental_col, *_size_cols] if c not in df_fsis.columns]
if _missing_s3:
    print(f"  WARNING: skipping — missing columns: {_missing_s3}")
else:
    df_2017 = df_fsis[df_fsis["year"] == 2017].copy()
    df_2017[_mental_col] = to_numeric_series(df_2017[_mental_col])
    for c in _size_cols:
        df_2017[c] = to_numeric_series(df_2017[c]).fillna(0)
    df_2017 = df_2017[df_2017[_mental_col].notna()].copy()

    rename_sz = {c: f"Size Bin {i+1}" for i, c in enumerate(_size_cols)}
    long_df = df_2017.melt(id_vars=["fips", "year", _mental_col], value_vars=_size_cols,
                           var_name="size_bin_col", value_name="fsis_establishments_in_bin")
    long_df["size_bin"] = long_df["size_bin_col"].map(rename_sz)
    long_df.to_csv(os.path.join(plots_dir, f"{today_str}_fsis_size_bins_vs_pmhd_2017_plotdata.csv"), index=False)

    sns.set_theme(style="whitegrid", context="talk")
    g = sns.relplot(data=long_df, x="fsis_establishments_in_bin", y=_mental_col,
                    col="size_bin", col_order=[f"Size Bin {i+1}" for i in range(5)],
                    col_wrap=3, kind="scatter", alpha=0.55, s=22,
                    facet_kws={"sharex": False, "sharey": True},
                    height=4.0, aspect=1.15, color="#2b8cbe")
    g.set_axis_labels("FSIS Establishments in Size Bin (county, 2017)", "Poor Mental Health Days")
    g.set_titles("{col_name}")
    g.fig.suptitle("County-Level FSIS Size-Bin Counts vs Poor Mental Health (2017)", y=1.03)
    fig_path = os.path.join(figs_qa_dir, f"{today_str}_fsis_size_bins_vs_pmhd_2017_facets.png")
    g.savefig(fig_path, dpi=240, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  2017 rows with mental outcome: {len(df_2017):,}")
    print(f"  Saved: {fig_path}")


# =============================================================================
# SECTION 4 — Missingness tables, heatmaps, row completeness
# =============================================================================
print("\n--- Section 4: Missingness ---")

missing_overall = pd.DataFrame({
    "variable":      analysis_cols,
    "non_missing_n": [int(df[c].notna().sum()) for c in analysis_cols],
    "missing_n":     [int(df[c].isna().sum())  for c in analysis_cols],
    "dtype":         [str(df[c].dtype)          for c in analysis_cols],
    "nunique":       [int(df[c].nunique(dropna=True)) for c in analysis_cols],
})
missing_overall["fill_pct"]    = missing_overall["non_missing_n"] / n_obs * 100
missing_overall["missing_pct"] = missing_overall["missing_n"]     / n_obs * 100
missing_overall = missing_overall.sort_values("missing_pct", ascending=False)

missing_by_year = df.groupby("year")[analysis_cols].apply(lambda x: x.isna().mean() * 100).sort_index()
fill_by_year    = 100 - missing_by_year

row_fill_pct = df[analysis_cols].notna().mean(axis=1) * 100
row_fill_dist = pd.DataFrame({"fips": df["fips"], "year": df["year"], "row_fill_pct": row_fill_pct})
row_fill_sum  = row_fill_pct.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame("value")

missing_share_by_year = pd.DataFrame({"year": sorted(df["year"].dropna().astype(int).unique())})
missing_share_by_year["missing_share_pct"] = [
    float(df.loc[df["year"] == y, analysis_cols].isna().mean().mean() * 100)
    for y in missing_share_by_year["year"]
]
missing_share_by_year["non_missing_share_pct"] = 100 - missing_share_by_year["missing_share_pct"]

write_csv_bundle("merged_missingness", {
    "overall":              missing_overall,
    "pct_by_year":          missing_by_year.reset_index(),
    "fill_by_year":         fill_by_year.reset_index(),
    "row_fill_distribution":row_fill_dist,
    "row_fill_summary":     row_fill_sum.reset_index(),
})
write_csv_bundle("missingness_share", {"by_year": missing_share_by_year})

plt.figure(figsize=(10, 5))
sns.lineplot(data=missing_share_by_year, x="year", y="missing_share_pct", marker="o", color="#d62728")
plt.title("Share of Missing Data by Year (All Variables)")
plt.xlabel("Year"); plt.ylabel("Missing share (%)")
plt.savefig(os.path.join(figs_dir, f"{today_str}_missing_share_by_year.png"), dpi=220, bbox_inches="tight")
safe_plot_close()

top60 = missing_overall.loc[missing_overall["missing_pct"].between(1, 99), "variable"].head(60).tolist()
if top60:
    plt.figure(figsize=(20, 8))
    sns.heatmap(missing_by_year[top60], cmap="mako_r", cbar_kws={"label": "Missing %"})
    plt.title("Missingness Heatmap by Year (Top 60 Variables)")
    plt.xlabel("Variable"); plt.ylabel("Year")
    plt.xticks(rotation=90, fontsize=7)
    p = os.path.join(figs_dir, f"{today_str}_missingness_heatmap_top60.png")
    plt.savefig(p, dpi=220, bbox_inches="tight")
    safe_plot_close()
    print("Saved:", p)

family_df = pd.DataFrame({"variable": analysis_cols})
family_df["family"] = family_df["variable"].map(detect_family)
fam_dir = os.path.join(figs_dir, "missingness_family_heatmaps")
os.makedirs(fam_dir, exist_ok=True)
for family in sorted(family_df["family"].unique()):
    fam_vars = family_df.loc[family_df["family"] == family, "variable"].tolist()
    for chunk_idx, cols_chunk in chunked(fam_vars, 35):
        plt.figure(figsize=(max(10, 0.45 * len(cols_chunk)), 7))
        sns.heatmap(missing_by_year[cols_chunk], cmap="magma_r", cbar_kws={"label": "Missing %"})
        plt.title(f"Missingness by Year — Family: {family} (chunk {chunk_idx})")
        plt.xlabel("Variable"); plt.ylabel("Year")
        plt.xticks(rotation=90, fontsize=7)
        plt.savefig(os.path.join(fam_dir, f"{today_str}_missingness_{safe_name(family)}_chunk{chunk_idx}.png"),
                    dpi=220, bbox_inches="tight")
        safe_plot_close()

quality_dir = os.path.join(figs_dir, "quality")
os.makedirs(quality_dir, exist_ok=True)
plt.figure(figsize=(10, 6))
sns.histplot(row_fill_dist["row_fill_pct"], bins=40, color="#1f77b4")
plt.title("Row Completeness Distribution (All County-Year Rows)")
plt.xlabel("Row Fill %"); plt.ylabel("Count")
plt.savefig(os.path.join(quality_dir, f"{today_str}_row_completeness_hist_all.png"), dpi=220, bbox_inches="tight")
safe_plot_close()

plt.figure(figsize=(12, 7))
sns.boxplot(data=row_fill_dist, x="year", y="row_fill_pct", color="#2a9d8f")
plt.title("Row Completeness by Year")
plt.xlabel("Year"); plt.ylabel("Row Fill %"); plt.xticks(rotation=45)
plt.savefig(os.path.join(quality_dir, f"{today_str}_row_completeness_box_by_year.png"), dpi=220, bbox_inches="tight")
safe_plot_close()
print(f"  Missingness outputs complete.")


# =============================================================================
# SECTION 5 — Summary statistics for all numeric variables
# =============================================================================
print("\n--- Section 5: Summary statistics ---")

numeric_map, numeric_meta = numeric_candidates(df, analysis_cols)
numeric_df = pd.DataFrame(numeric_map, index=df.index)

stats_rows = []
for c in numeric_df.columns:
    s = numeric_df[c]
    stats_rows.append({
        "variable": c, "count": int(s.notna().sum()), "missing_n": int(s.isna().sum()),
        "fill_pct": float(s.notna().mean() * 100),
        "mean": float(s.mean()) if s.notna().any() else np.nan,
        "std":  float(s.std())  if s.notna().any() else np.nan,
        "min":  float(s.min())  if s.notna().any() else np.nan,
        "p01":  float(s.quantile(0.01)) if s.notna().any() else np.nan,
        "p25":  float(s.quantile(0.25)) if s.notna().any() else np.nan,
        "median":float(s.median())      if s.notna().any() else np.nan,
        "p75":  float(s.quantile(0.75)) if s.notna().any() else np.nan,
        "p99":  float(s.quantile(0.99)) if s.notna().any() else np.nan,
        "max":  float(s.max())          if s.notna().any() else np.nan,
        "nunique": int(s.nunique(dropna=True)),
        "zeros_n": int((s == 0).sum()) if s.notna().any() else 0,
    })
summary_numeric = pd.DataFrame(stats_rows).sort_values(["fill_pct", "std"], ascending=[False, False])
summary_all = missing_overall.merge(summary_numeric, how="left", on="variable")
write_csv_bundle("merged_summary_stats", {
    "all_variables":    summary_all,
    "numeric_variables":summary_numeric,
    "numeric_parse_meta":numeric_meta,
})

if not numeric_df.empty:
    yearly_means = numeric_df.copy()
    yearly_means["year"] = df["year"].values
    yearly_means = yearly_means.groupby("year").mean(numeric_only=True).sort_index()
    yoy_pct      = yearly_means.pct_change() * 100

    growth_rows = []
    for c in yearly_means.columns:
        s = yearly_means[c].dropna()
        if s.empty:
            growth_rows.append({"variable": c, "first_year": np.nan, "last_year": np.nan,
                                 "first_value": np.nan, "last_value": np.nan,
                                 "abs_change": np.nan, "pct_change": np.nan, "cagr_pct": np.nan})
            continue
        y0, y1 = int(s.index.min()), int(s.index.max())
        v0, v1 = float(s.iloc[0]), float(s.iloc[-1])
        n_yrs  = max(y1 - y0, 1)
        pct_ch = ((v1 / v0 - 1) * 100) if (v0 not in [0, np.nan] and np.isfinite(v0)) else np.nan
        cagr   = (((v1 / v0) ** (1 / n_yrs) - 1) * 100) if (v0 > 0 and v1 > 0) else np.nan
        growth_rows.append({"variable": c, "first_year": y0, "last_year": y1,
                             "first_value": v0, "last_value": v1,
                             "abs_change": v1 - v0, "pct_change": pct_ch, "cagr_pct": cagr})
    growth_summary = pd.DataFrame(growth_rows).sort_values("cagr_pct", ascending=False)
    write_csv_bundle("descriptive_growth", {
        "yearly_means":  yearly_means.reset_index(),
        "yoy_pct_change":yoy_pct.reset_index(),
        "growth_summary":growth_summary,
    })
print("  Summary stats complete.")


# =============================================================================
# SECTION 6 — Violin plots: high-coverage variables
# =============================================================================
print("\n--- Section 6: Violin plots ---")
high_cov = summary_numeric[
    (summary_numeric["fill_pct"] >= 90) & (summary_numeric["nunique"] >= 20)
].sort_values(["fill_pct", "std"], ascending=[False, False])
if high_cov.empty:
    high_cov = summary_numeric[
        (summary_numeric["fill_pct"] >= 60) & (summary_numeric["nunique"] >= 8)
    ].sort_values(["fill_pct", "std"], ascending=[False, False])
high_cov_vars = high_cov["variable"].head(20).tolist()

violin_dir = os.path.join(figs_dir, "violin_high_coverage")
os.makedirs(violin_dir, exist_ok=True)
for var in high_cov_vars:
    s = numeric_df[var].dropna() if var in numeric_df.columns else pd.Series(dtype=float)
    if s.empty:
        continue
    plt.figure(figsize=(6.5, 6))
    sns.violinplot(y=s, inner="quartile", cut=0, color="#2a9d8f")
    plt.title(f"Distribution: {var}"); plt.ylabel(var)
    plt.savefig(os.path.join(violin_dir, f"{today_str}_violin_{safe_name(var)}.png"),
                dpi=220, bbox_inches="tight")
    safe_plot_close()
print(f"  Violin plots: {len(high_cov_vars)} variables.")


# =============================================================================
# SECTION 7 — CAFO composition from compact
# =============================================================================
print("\n--- Section 7: CAFO composition (census years only) ---")

CAFO_CENSUS_YEARS = [2002, 2007, 2012, 2017, 2022]
comp_dir = os.path.join(figs_dir, "cafo_composition")
os.makedirs(comp_dir, exist_ok=True)

# Violin: distribution by commodity × size
cafo_violin_dir = os.path.join(figs_dir, "cafo_violin")
os.makedirs(cafo_violin_dir, exist_ok=True)
for commodity in ["cattle", "chickens", "hogs"]:
    sub = cafo_long[cafo_long["commodity_desc"] == commodity].copy()
    if sub.empty:
        continue
    top_classes = (sub.groupby("class_desc")["ops_count"].sum()
                   .sort_values(ascending=False).head(12).index.tolist())
    sub = sub[sub["class_desc"].isin(top_classes)]
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=sub, x="class_desc", y="ops_count", hue="size_bucket", cut=0, inner="quartile")
    plt.title(f"CAFO Operations Distribution by Class and Size: {commodity.title()}")
    plt.xlabel("Class"); plt.ylabel("Operations"); plt.xticks(rotation=45, ha="right")
    plt.savefig(os.path.join(cafo_violin_dir, f"{today_str}_cafo_violin_{commodity}.png"),
                dpi=220, bbox_inches="tight")
    safe_plot_close()

plt.figure(figsize=(10, 6))
sns.violinplot(data=cafo_long, x="commodity_desc", y="ops_count", hue="size_bucket", cut=0, inner="quartile")
plt.title("CAFO Operations by Commodity and Size")
plt.xlabel("Commodity"); plt.ylabel("Operations")
plt.savefig(os.path.join(cafo_violin_dir, f"{today_str}_cafo_violin_commodity_compare.png"),
            dpi=220, bbox_inches="tight")
safe_plot_close()

# Stacked bars: census years only
cafo_long_census = cafo_long[cafo_long["year"].isin(CAFO_CENSUS_YEARS)].copy()

COMMODITY_COLORS = {"cattle": "#d48f3a", "chickens": "#6aab6e", "hogs": "#e06c6c"}
SIZE_COLORS      = {"small": "#a8c7e8", "medium": "#4f94d0", "large": "#174e86"}

# A) For each size: commodity share
size_commodity = (cafo_long_census.groupby(["year", "size_bucket", "commodity_desc"], as_index=False)
                  ["ops_count"].sum())
size_commodity["pct_within_size_year"] = (
    size_commodity["ops_count"]
    / size_commodity.groupby(["year", "size_bucket"])["ops_count"].transform("sum") * 100)
size_commodity.to_csv(os.path.join(comp_dir, f"{today_str}_size_commodity_share.csv"), index=False)

for size in ["small", "medium", "large"]:
    d = size_commodity[size_commodity["size_bucket"] == size].copy()
    if d.empty:
        continue
    pivot = d.pivot_table(index="year", columns="commodity_desc",
                          values="pct_within_size_year", aggfunc="sum").reindex(CAFO_CENSUS_YEARS)
    commodities = [c for c in ["cattle", "hogs", "chickens"] if c in pivot.columns]
    pivot = pivot[commodities]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CAFO_CENSUS_YEARS)); bottoms = np.zeros(len(CAFO_CENSUS_YEARS))
    for comm in commodities:
        vals = pivot[comm].fillna(0).values
        ax.bar(x, vals, 0.6, bottom=bottoms, label=comm.title(),
               color=COMMODITY_COLORS.get(comm), alpha=0.85)
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in CAFO_CENSUS_YEARS])
    ax.set_xlabel("USDA Census Year"); ax.set_ylabel("Share of operations (%)")
    ax.set_ylim(0, 105); ax.legend(loc="upper right")
    ax.set_title(f"Commodity Share Within {size.title()} CAFO Operations\n"
                 "(USDA Census years only; missing bars = data not available)")
    fig.savefig(os.path.join(comp_dir, f"{today_str}_share_commodity_within_{size}.png"),
                dpi=220, bbox_inches="tight")
    plt.close(fig)

# B) For each commodity: size share
commodity_size = (cafo_long_census.groupby(["year", "commodity_desc", "size_bucket"], as_index=False)
                  ["ops_count"].sum())
commodity_size["pct_within_commodity_year"] = (
    commodity_size["ops_count"]
    / commodity_size.groupby(["year", "commodity_desc"])["ops_count"].transform("sum") * 100)
commodity_size.to_csv(os.path.join(comp_dir, f"{today_str}_commodity_size_share.csv"), index=False)

for commodity in ["cattle", "chickens", "hogs"]:
    d = commodity_size[commodity_size["commodity_desc"] == commodity].copy()
    if d.empty:
        continue
    pivot = d.pivot_table(index="year", columns="size_bucket",
                          values="pct_within_commodity_year", aggfunc="sum").reindex(CAFO_CENSUS_YEARS)
    sizes = [c for c in ["small", "medium", "large"] if c in pivot.columns]
    pivot = pivot[sizes]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(CAFO_CENSUS_YEARS)); bottoms = np.zeros(len(CAFO_CENSUS_YEARS))
    for sz in sizes:
        vals = pivot[sz].fillna(0).values
        ax.bar(x, vals, 0.6, bottom=bottoms, label=sz.title(),
               color=SIZE_COLORS.get(sz), alpha=0.9)
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in CAFO_CENSUS_YEARS])
    ax.set_xlabel("USDA Census Year"); ax.set_ylabel("Share of operations (%)")
    ax.set_ylim(0, 105); ax.legend(loc="upper right")
    ax.set_title(f"Size Share Within {commodity.title()} CAFO Operations\n"
                 "(USDA Census years only; missing bars = data not available)")
    fig.savefig(os.path.join(comp_dir, f"{today_str}_share_size_within_{commodity}.png"),
                dpi=220, bbox_inches="tight")
    plt.close(fig)
print("  CAFO composition complete.")


# =============================================================================
# SECTION 8 — CAFO concentration: HHI, Gini, entry/exit by year
# =============================================================================
print("\n--- Section 8: CAFO concentration and entry/exit ---")

cafo_qc_dir = os.path.join(figs_dir, "cafo_quality")
os.makedirs(cafo_qc_dir, exist_ok=True)

cafo_cty_commodity = (cafo_long.groupby(["year", "fips", "commodity_desc"], as_index=False)
                      ["ops_count"].sum())

concentration_rows = []
for (yr, cmd), g in cafo_cty_commodity.groupby(["year", "commodity_desc"]):
    x     = g["ops_count"].fillna(0).astype(float).values
    total = float(np.sum(x))
    n     = int((x > 0).sum())
    hhi   = float(np.sum((x / total)**2)) if total > 0 else np.nan
    concentration_rows.append({
        "year": int(yr), "commodity_desc": cmd, "total_ops": total,
        "n_counties_positive": n, "hhi": hhi, "gini": gini_from_values(x),
        "top10_county_share_pct": float(np.sort(x)[-10:].sum() / total * 100) if total > 0 else np.nan,
    })
concentration_df = pd.DataFrame(concentration_rows).sort_values(["commodity_desc", "year"])
concentration_df.to_csv(os.path.join(cafo_qc_dir, f"{today_str}_cafo_concentration_by_year_commodity.csv"),
                        index=False)

for metric in ["hhi", "gini", "top10_county_share_pct"]:
    d = concentration_df.dropna(subset=[metric]).copy()
    if d.empty:
        continue
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=d, x="year", y=metric, hue="commodity_desc", marker="o")
    plt.title(f"CAFO Concentration Trend: {metric}")
    plt.xlabel("Year"); plt.ylabel(metric)
    plt.savefig(os.path.join(cafo_qc_dir, f"{today_str}_cafo_concentration_{metric}.png"),
                dpi=220, bbox_inches="tight")
    safe_plot_close()

cafo_overall = cafo.groupby(["fips", "year"], as_index=False)[["small", "medium", "large"]].sum()
cafo_overall["state_size"] = "no_cafo"
cafo_overall.loc[cafo_overall["small"]  > 0, "state_size"] = "small_only"
cafo_overall.loc[cafo_overall["medium"] > 0, "state_size"] = "medium_or_more"
cafo_overall.loc[cafo_overall["large"]  > 0, "state_size"] = "large_present"
cafo_overall = cafo_overall.sort_values(["fips", "year"])
cafo_overall["next_state_size"] = cafo_overall.groupby("fips")["state_size"].shift(-1)
cafo_overall["next_year"]       = cafo_overall.groupby("fips")["year"].shift(-1)
cafo_overall["lag_large"]       = cafo_overall.groupby("fips")["large"].shift(1).fillna(0)
cafo_overall["curr_large"]      = cafo_overall["large"].fillna(0)

transitions = cafo_overall[
    cafo_overall["next_year"].notna() & ((cafo_overall["next_year"] - cafo_overall["year"]) == 1)
].copy()
(transitions.groupby(["state_size", "next_state_size"], as_index=False)
 .size().rename(columns={"size": "n_transitions"})
 .to_csv(os.path.join(cafo_qc_dir, f"{today_str}_cafo_transition_matrix_overall.csv"), index=False))
(transitions.groupby(["year", "state_size", "next_state_size"], as_index=False)
 .size().rename(columns={"size": "n_transitions"})
 .to_csv(os.path.join(cafo_qc_dir, f"{today_str}_cafo_transition_matrix_by_year.csv"), index=False))

entry_exit = cafo_overall.copy()
entry_exit["entry_large"] = ((entry_exit["lag_large"] <= 0) & (entry_exit["curr_large"] > 0)).astype(int)
entry_exit["exit_large"]  = ((entry_exit["lag_large"] > 0)  & (entry_exit["curr_large"] <= 0)).astype(int)
ee_year = (entry_exit.groupby("year", as_index=False)
           .agg(entries_large=("entry_large", "sum"), exits_large=("exit_large", "sum"),
                counties=("fips", "nunique")))
ee_year["entry_rate_pct"] = ee_year["entries_large"] / ee_year["counties"] * 100
ee_year["exit_rate_pct"]  = ee_year["exits_large"]   / ee_year["counties"] * 100
ee_year.to_csv(os.path.join(cafo_qc_dir, f"{today_str}_cafo_entry_exit_large_by_year.csv"), index=False)
plt.figure(figsize=(10, 6))
sns.lineplot(data=ee_year, x="year", y="entry_rate_pct", marker="o", label="Entry rate")
sns.lineplot(data=ee_year, x="year", y="exit_rate_pct",  marker="o", label="Exit rate")
plt.title("Large-CAFO Entry/Exit Rates by Year")
plt.xlabel("Year"); plt.ylabel("Rate (%) of counties")
plt.savefig(os.path.join(cafo_qc_dir, f"{today_str}_cafo_entry_exit_rates.png"),
            dpi=220, bbox_inches="tight")
safe_plot_close()
print("  Concentration and entry/exit complete.")


# =============================================================================
# SECTION 9 — Ridge plots + trend plots + binned scatter (slow)
# =============================================================================
print("\n--- Section 9: Ridge plots + trend plots + binned scatter ---")

ridge_dir = os.path.join(figs_dir, "ridge")
os.makedirs(ridge_dir, exist_ok=True)

ridge_targets = [
    "aggravated_assault_crime", "violent_crime",
    "poor_mental_health_days", "frequent_mental_distress_per100k",
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
    keep_years  = tmp["year"].value_counts()[tmp["year"].value_counts() >= 40].index.tolist()
    tmp = tmp[tmp["year"].isin(keep_years)]
    if tmp.empty:
        continue
    g = sns.FacetGrid(tmp, row="year", hue="year", aspect=12, height=0.35, palette="viridis")
    g.map(sns.kdeplot, var, fill=True, alpha=0.9, linewidth=1)
    g.map(sns.kdeplot, var, color="white", lw=0.5)
    g.set_titles(""); g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.fig.subplots_adjust(hspace=-0.6)
    g.fig.suptitle(f"Ridgeline-style KDE by Year: {var}", y=1.02)
    g.savefig(os.path.join(ridge_dir, f"{today_str}_ridge_{safe_name(var)}.png"),
              dpi=220, bbox_inches="tight")
    plt.close(g.fig)

# Trend plots: all numeric variables vs CAFO counts (controlled by MAX_TREND_VARS)
trend_dir = os.path.join(figs_dir, "trends_all_variables")
os.makedirs(trend_dir, exist_ok=True)
cafo_count = (cafo.groupby(["fips", "year", "commodity_desc"], as_index=False)
              [["small", "medium", "large"]].sum())

skip_vars = {"year", "non_large_metro", "cafo_total_ops_all_animals", "cafo_total_ops_chickens",
             "any_large_cafo", "any_medium_or_large_cafo", "large_cafo", "medium_cafo", "small_cafo"}
outcome_vars = [c for c in numeric_df.columns if c not in skip_vars and not c.startswith("n_rows")]
if MAX_TREND_VARS:
    try:
        cap = int(MAX_TREND_VARS)
        if cap > 0:
            outcome_vars = outcome_vars[:cap]
    except Exception:
        pass
pd.DataFrame({"variable": outcome_vars}).to_csv(
    os.path.join(trend_dir, f"{today_str}_trend_variable_list.csv"), index=False)
print(f"  Trend variables to plot: {len(outcome_vars)}")

for idx, var in enumerate(outcome_vars, start=1):
    if var not in df.columns:
        continue
    df[var] = to_numeric_series(get_series(df, var))
    if df[var].notna().sum() < 30:
        continue
    for commodity in ["cattle", "chickens", "hogs"]:
        cc = cafo_count[cafo_count["commodity_desc"] == commodity][["fips", "year", "small", "medium", "large"]].copy()
        if cc.empty:
            continue
        rhs_cols = ["fips", "year", var, POP_COL] if POP_COL != var else ["fips", "year", var]
        sub = cc.merge(df[[c for c in rhs_cols if c in df.columns]], on=["fips", "year"], how="left")
        if sub[var].notna().sum() < 30:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        for i, (size_col, stitle) in enumerate(zip(["small", "medium", "large"],
                                                    ["Small CAFO", "Medium CAFO", "Large CAFO"])):
            agg = sub.groupby("year", as_index=False).agg(outcome_mean=(var, "mean"),
                                                           cafo_count=(size_col, "sum"))
            ax = axes[i]
            ax.plot(agg["year"], agg["outcome_mean"], marker="o", color="#1f77b4", label="Outcome mean")
            ax.set_title(stitle); ax.set_xlabel("Year")
            if i == 0:
                ax.set_ylabel(f"{var} (mean)")
            ax2 = ax.twinx()
            ax2.plot(agg["year"], agg["cafo_count"], marker="^", color="#d62728", alpha=0.8, label="CAFO count")
            if i == 2:
                ax2.set_ylabel("CAFO operations")
        fig.suptitle(f"{commodity.title()}: {var} vs CAFO size counts", y=1.03)
        fig.tight_layout()
        fig.savefig(os.path.join(trend_dir, f"{today_str}_trend_{safe_name(commodity)}_{safe_name(var)}.png"),
                    dpi=180, bbox_inches="tight")
        plt.close(fig)
    if idx % 20 == 0:
        print(f"  Trend progress: {idx}/{len(outcome_vars)}")

# Binned scatter for key outcomes
binned_dir = os.path.join(figs_dir, "binned_scatter")
os.makedirs(binned_dir, exist_ok=True)
key_outcomes = ["aggravated_assault_crime", "violent_crime", "some_college_per100k", POP_COL]
for var in key_outcomes:
    if var not in df.columns:
        continue
    v = df[["fips", "year", var]].copy()
    v[var] = to_numeric_series(get_series(v, var))
    for commodity in ["cattle", "chickens", "hogs"]:
        cc = cafo_count[cafo_count["commodity_desc"] == commodity][["fips", "year", "small", "medium", "large"]].copy()
        if cc.empty:
            continue
        m = cc.merge(v, on=["fips", "year"], how="left")
        for size_col in ["small", "medium", "large"]:
            d = m[[size_col, var]].dropna()
            d = d[d[size_col] > 0]
            if len(d) < 50 or d[size_col].nunique() < 10:
                continue
            try:
                d["bin"] = pd.qcut(d[size_col], q=20, duplicates="drop")
            except Exception:
                continue
            g = d.groupby("bin", observed=True).agg(x=(size_col, "mean"),
                                                      y=(var, "mean"), n=("bin", "size")).reset_index(drop=True)
            if g.empty:
                continue
            plt.figure(figsize=(7, 5))
            sns.scatterplot(data=g, x="x", y="y", size="n", legend=False, color="#1f77b4")
            sns.lineplot(data=g, x="x", y="y", color="#d62728")
            plt.title(f"Binned scatter: {var} vs {commodity} {size_col}")
            plt.xlabel(f"{commodity} {size_col} CAFO (bin mean)"); plt.ylabel(f"{var} (bin mean)")
            plt.savefig(os.path.join(binned_dir,
                                     f"{today_str}_binned_{safe_name(var)}_{safe_name(commodity)}_{size_col}.png"),
                        dpi=220, bbox_inches="tight")
            safe_plot_close()
print("  Ridge + trend + binned scatter complete.")


# =============================================================================
# SECTION 10 — State-level CAFO trends
# =============================================================================
print("\n--- Section 10: State-level CAFO trends ---")

state_dir = os.path.join(figs_dir, "state_trends_large_cafo")
os.makedirs(state_dir, exist_ok=True)

cafo_count["state_fips"] = cafo_count["fips"].astype(str).str[:2]
state_large = (cafo_count.groupby("state_fips", as_index=False)["large"]
               .sum().rename(columns={"large": "large_ops_total"}))
state_large = state_large[state_large["large_ops_total"] > 0].copy()
state_large["state_name"] = state_large["state_fips"].map(STATE_FIPS_TO_NAME).fillna(state_large["state_fips"])
state_large = state_large.sort_values("large_ops_total", ascending=False)
state_large.to_csv(os.path.join(state_dir, f"{today_str}_states_with_large_cafo.csv"), index=False)

selected_states = state_large["state_fips"].head(STATE_PLOT_LIMIT).tolist()
if "05" in state_large["state_fips"].values and "05" not in selected_states:
    selected_states += ["05"]

state_outcomes = [c for c in ["aggravated_assault_crime", "violent_crime",
                               "poor_mental_health_days", "some_college_per100k", POP_COL]
                  if c in df.columns]
df_state = df.copy()
df_state["state_fips"] = df_state["fips"].astype(str).str[:2]
df_state["state_name"] = df_state["state_fips"].map(STATE_FIPS_TO_NAME).fillna(df_state["state_fips"])
for v in state_outcomes:
    df_state[v] = to_numeric_series(get_series(df_state, v))

for st in selected_states:
    st_name = STATE_FIPS_TO_NAME.get(st, st)
    for commodity in ["cattle", "chickens", "hogs"]:
        ca = cafo_count[(cafo_count["state_fips"] == st) & (cafo_count["commodity_desc"] == commodity)].copy()
        if ca.empty:
            continue
        ca_agg = ca.groupby("year", as_index=False)[["small", "medium", "large"]].sum()
        for out_var in state_outcomes:
            st_slice = df_state[df_state["state_fips"] == st].copy()
            y = st_slice.groupby("year", as_index=False).agg(outcome_mean=(out_var, "mean"))
            z = ca_agg.merge(y, on="year", how="inner").dropna(subset=["outcome_mean"])
            if z.empty:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
            for i, size_col in enumerate(["small", "medium", "large"]):
                ax = axes[i]
                ax.plot(z["year"], z["outcome_mean"], marker="o", color="#1f77b4", label="Outcome mean")
                ax.set_title(f"{size_col.title()} CAFO"); ax.set_xlabel("Year")
                if i == 0:
                    ax.set_ylabel(out_var)
                ax2 = ax.twinx()
                ax2.plot(z["year"], z[size_col], marker="^", color="#d62728", label=f"{size_col} ops")
                if i == 2:
                    ax2.set_ylabel("CAFO ops count")
            fig.suptitle(f"{st_name} — {commodity.title()} CAFO vs {out_var}", y=1.03)
            fig.tight_layout()
            fig.savefig(os.path.join(state_dir,
                                     f"{today_str}_state_{safe_name(st_name)}_{safe_name(commodity)}_{safe_name(out_var)}.png"),
                        dpi=180, bbox_inches="tight")
            plt.close(fig)
print(f"  State trends complete for {len(selected_states)} states.")


# =============================================================================
# SECTION 11 — Targeted sumstats: county coverage check, CAFO/FSIS/MH, crosscheck
# =============================================================================
print("\n--- Section 11: Targeted sumstats + coverage check ---")

cafo_cols_panel = [
    "cafo_cattle_small", "cafo_cattle_medium", "cafo_cattle_large",
    "cafo_hogs_small",   "cafo_hogs_medium",   "cafo_hogs_large",
    "cafo_chickens_small","cafo_chickens_medium","cafo_chickens_large",
    "cafo_total_ops_all_animals", "cafo_total_ops_chickens",
]
missing_cafo = [c for c in cafo_cols_panel if c not in df.columns]
if missing_cafo:
    print(f"  WARNING: missing CAFO columns in panel: {missing_cafo}")
for c in [c for c in cafo_cols_panel if c in df.columns]:
    df[c] = to_numeric_series(df[c])

fsis_col      = "n_unique_establishments_fsis"
fsis_size_cols = ["n_size_bucket_1_fsis", "n_size_bucket_2_fsis", "n_size_bucket_3_fsis",
                  "n_size_bucket_4_fsis", "n_size_bucket_5_fsis", "n_size_bucket_missing_fsis"]
fsis_panel_cols = [fsis_col, *fsis_size_cols]
missing_fsis = [c for c in fsis_panel_cols if c not in df_fsis.columns]
if missing_fsis:
    print(f"  WARNING: missing FSIS columns in panel_fsis: {missing_fsis}")
for c in [c for c in fsis_panel_cols if c in df_fsis.columns]:
    df_fsis[c] = to_numeric_series(df_fsis[c])

poor_mental_col = "poor_mental_health_days"
if poor_mental_col not in df.columns:
    print(f"  WARNING: missing {poor_mental_col} in panel")

# County coverage check vs rural-key
rural_path = latest_file_glob(clean_dir, "*rural-key*.csv")
rural_df   = read_and_prepare(rural_path)
rural_df   = normalize_panel_key(rural_df)
rural_df["non_large_metro"] = pd.to_numeric(rural_df.get("non_large_metro"), errors="coerce").astype("Int64")
rural_kept  = rural_df.loc[rural_df["non_large_metro"] == 1, ["fips", "year"]].drop_duplicates()
merged_keys = df[["fips", "year"]].drop_duplicates()
coverage_row = pd.DataFrame([{
    "run_date": today_str,
    "merged_file": os.path.basename(merged_path),
    "rural_key_file": os.path.basename(rural_path),
    "merged_rows": int(len(df)),
    "merged_unique_counties": int(df["fips"].nunique()),
    "rural_kept_rows": int(len(rural_kept)),
    "rural_kept_unique_counties": int(rural_kept["fips"].nunique()),
    "county_count_match": int(df["fips"].nunique() == rural_kept["fips"].nunique()),
    "key_set_exact_match": int(
        merged_keys.merge(rural_kept, on=["fips", "year"], how="outer", indicator=True)
        ["_merge"].eq("both").all()),
}])
coverage_row.to_csv(os.path.join(qa_dir, "county_coverage_check.csv"), index=False)

# CAFO sumstats
if not missing_cafo:
    summarize_numeric(df, cafo_cols_panel).to_csv(
        os.path.join(qa_dir, "cafo_county_year_sumstats.csv"), index=False)
    state_year_cafo = df.groupby(["state_fips", "state_abbrev", "year"], as_index=False)[cafo_cols_panel].sum(min_count=1)
    summarize_numeric(state_year_cafo, cafo_cols_panel).to_csv(
        os.path.join(qa_dir, "cafo_state_year_sumstats.csv"), index=False)

# FSIS sumstats (from panel_fsis)
if not missing_fsis:
    df_fsis["state_fips"] = df_fsis["fips"].astype("string").str[:2]
    summarize_numeric(df_fsis, [c for c in fsis_panel_cols if c in df_fsis.columns]).to_csv(
        os.path.join(qa_dir, "fsis_county_year_sumstats.csv"), index=False)

# Mental health fill share
mental_fill_cols = [c for c in df.columns if "mental" in c.lower()]
any_mental = df[mental_fill_cols].notna().any(axis=1) if mental_fill_cols else pd.Series(False, index=df.index)
pd.DataFrame([{
    "run_date": today_str, "merged_file": os.path.basename(merged_path),
    "mental_column_count": int(len(mental_fill_cols)), "n_rows": int(len(df)),
    "rows_with_any_mental_data": int(any_mental.sum()),
    "share_rows_with_any_mental_data_pct": float(any_mental.mean() * 100),
}]).to_csv(os.path.join(qa_dir, "mental_fill_share.csv"), index=False)

# CAFO unit/count crosscheck vs pre-merged compact
pre_cafo_path = latest_file_glob(clean_dir, "*_cafo_ops_by_size_compact.csv")
pre = pd.read_csv(pre_cafo_path, low_memory=False)
pre = normalize_panel_key(pre)
for c in ["commodity_desc", "small", "medium", "large"]:
    if c not in pre.columns:
        raise RuntimeError(f"Missing column in CAFO compact: {c}")
pre["commodity_desc"] = pre["commodity_desc"].astype("string").str.strip().str.lower()
pre = pre[pre["commodity_desc"].isin(["cattle", "hogs", "chickens"])].copy()
if "class_desc" in pre.columns:
    pre["class_desc"] = pre["class_desc"].astype("string").str.strip().str.lower()
    canonical_class = {"cattle": "incl calves", "hogs": "all classes", "chickens": "layers"}
    pre = pre[pre["class_desc"] == pre["commodity_desc"].map(canonical_class)].copy()
for c in ["small", "medium", "large"]:
    pre[c] = to_numeric_series(pre[c])
pre = pre.merge(rural_kept, on=["fips", "year"], how="inner")
pre_grp  = pre.groupby(["fips", "year", "commodity_desc"], as_index=False)[["small", "medium", "large"]].sum(min_count=1)
pre_wide = pre_grp.pivot_table(index=["fips", "year"], columns="commodity_desc",
                                values=["small", "medium", "large"], aggfunc="sum")
pre_wide.columns = [f"cafo_{commodity}_{size}" for size, commodity in pre_wide.columns]
pre_wide = pre_wide.reset_index()
for commodity in ["cattle", "hogs", "chickens"]:
    for size in ["small", "medium", "large"]:
        col = f"cafo_{commodity}_{size}"
        if col not in pre_wide.columns:
            pre_wide[col] = np.nan
pre_animal_cols = [f"cafo_{cmd}_{sz}" for cmd in ["cattle", "hogs", "chickens"] for sz in ["small", "medium", "large"]]
pre_wide["cafo_total_ops_all_animals"] = pre_wide[pre_animal_cols].sum(axis=1, min_count=1)
pre_wide["cafo_total_ops_chickens"]    = pre_wide[["cafo_chickens_small", "cafo_chickens_medium",
                                                    "cafo_chickens_large"]].sum(axis=1, min_count=1)
all_check_cols = [*pre_animal_cols, "cafo_total_ops_all_animals", "cafo_total_ops_chickens"]
merged_cmp = df[["fips", "year", *[c for c in all_check_cols if c in df.columns]]].copy()
cmp = merged_cmp.merge(pre_wide, on=["fips", "year"], how="left", suffixes=("_merged", "_pre"))

cross_rows = []
for col in all_check_cols:
    m    = to_numeric_series(cmp.get(f"{col}_merged", pd.Series(dtype=float)))
    p    = to_numeric_series(cmp.get(f"{col}_pre",    pd.Series(dtype=float)))
    both = m.notna() & p.notna()
    diff = (m - p).abs()
    cross_rows.append({
        "variable": col, "n_keys_compared": int(both.sum()),
        "pct_exact_match_on_compared_keys": float((diff[both] == 0).mean() * 100) if both.any() else np.nan,
        "max_abs_diff": float(diff[both].max()) if both.any() else np.nan,
        "merged_sum": float(m.sum(skipna=True)), "premerged_sum": float(p.sum(skipna=True)),
    })
pd.DataFrame(cross_rows).to_csv(
    os.path.join(qa_dir, "cafo_commodity_size_crosscheck_vs_premerged.csv"), index=False)
print("  Sumstats + crosscheck complete.")


# =============================================================================
# SECTION 12 — State dot plots + chickens size-class facets
# =============================================================================
print("\n--- Section 12: State dot plots + chickens facets ---")

dot_plots_dir = os.path.join(figs_dir, "state_county_dots")
os.makedirs(dot_plots_dir, exist_ok=True)

if poor_mental_col in df.columns:
    dot_df = df[(df["year"] == 2017)
                & df["cafo_total_ops_all_animals"].notna()
                & (df["cafo_total_ops_all_animals"] > 0)
                & df[poor_mental_col].notna()].copy()
    for st in sorted(dot_df["state_fips"].dropna().unique()):
        d     = dot_df[dot_df["state_fips"] == st].copy()
        if d.empty:
            continue
        st_lbl = d["state_abbrev"].dropna().iloc[0] if d["state_abbrev"].notna().any() else st
        plt.figure(figsize=(7.2, 5.4))
        sns.scatterplot(data=d, x="cafo_total_ops_all_animals", y=poor_mental_col,
                        s=24, alpha=0.65, edgecolor=None)
        plt.title(f"{st_lbl} County Dots (2017): Poor Mental Health vs Total CAFO Ops")
        plt.xlabel("Total CAFO Operations (All Animals)")
        plt.ylabel("Poor Mental Health Days")
        plt.savefig(os.path.join(dot_plots_dir, f"state_county_dots_{st_lbl}_2017.png"),
                    dpi=220, bbox_inches="tight")
        plt.close()

    facet = df[df["year"].between(2010, 2020, inclusive="both")
               & df["cafo_total_ops_chickens"].notna()
               & (df["cafo_total_ops_chickens"] > 0)
               & df[poor_mental_col].notna()].copy()
    facet["chicken_size_class"] = pd.NA
    facet.loc[facet["cafo_chickens_large"] > 0, "chicken_size_class"] = "large_present"
    facet.loc[(facet["cafo_chickens_large"] <= 0) & (facet["cafo_chickens_medium"] > 0),
              "chicken_size_class"] = "medium_present"
    facet.loc[(facet["cafo_chickens_large"] <= 0) & (facet["cafo_chickens_medium"] <= 0)
              & (facet["cafo_chickens_small"] > 0), "chicken_size_class"] = "small_only"
    facet = facet[facet["chicken_size_class"].notna()].copy()

    if not facet.empty:
        palette = {"small_only": "#6baed6", "medium_present": "#fd8d3c", "large_present": "#cb181d"}
        g = sns.relplot(data=facet, x="cafo_total_ops_chickens", y=poor_mental_col,
                        hue="chicken_size_class", col="year", col_wrap=4, kind="scatter",
                        s=18, alpha=0.65, palette=palette, height=3.0, aspect=1.15)
        g.set_axis_labels("Chicken CAFO Operations (Total)", "Poor Mental Health Days")
        g.fig.suptitle("Chicken CAFOs vs Poor Mental Health Days (2010–2020)", y=1.02)
        g.savefig(os.path.join(figs_dir, "chickens_cafo_vs_poor_mental_health_faceted_2010_2020.png"),
                  dpi=220, bbox_inches="tight")
        plt.close(g.fig)
        facet[["fips", "state_fips", "year", "cafo_total_ops_chickens", poor_mental_col,
               "cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large",
               "chicken_size_class"]].to_csv(
            os.path.join(qa_dir, "chickens_scatter_data_2010_2020.csv"), index=False)

print("  State dots + chickens facets complete.")
print(f"\nscript2a complete. Tables → {qa_dir} | Figs → {figs_dir}")
