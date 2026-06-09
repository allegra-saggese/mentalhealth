#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script2a-qa.py

QA pipeline for the merged county-year panel. Runs three checks sequentially
against the latest full_merged.csv. The panel is loaded once and shared across
all three sections.

  Section 1 — CDC crude-rate sense check
    Verifies that merged CDC crude rates align with rates recomputed from census
    population. Appends diagnostic columns to the merged panel and re-exports
    the full panel plus year-slice CSVs with today's date stamp.
    Outputs: qa_cdc_cruderate_sensecheck_*.csv

  Section 2 — Correlation tables + FSIS column QA + QA memo
    Pairwise Pearson and Spearman correlation tables across three time windows.
    FSIS 12-column upper-tail range check. Markdown QA memo summarizing results.
    Outputs: correlation_*.csv, qa_fsis_12_column_check.csv,
             qa_key_variable_coverage_by_year.csv, qa_memo_panel_sumstats_by_farms.md

  Section 3 — FSIS size bins vs. poor mental health days (2017)
    Faceted scatter of establishment counts by size bucket vs. outcome (2017 cross-section).
    Outputs: fsis_size_bins_vs_poor_mental_health_2017_*.{csv,png}

Tables  → Dropbox/Mental/Data/output/tables/panel-sumstats-by-farms/
Figures → Dropbox/Mental/Data/output/figs/panel-sumstats-by-farms/plots/
"""

from packages import *
from functions import *

# ── Directories ───────────────────────────────────────────────────────────────
merged_dir     = os.path.join(db_data, "merged")
qa_dir         = os.path.join(tables_dir, "panel-sumstats-by-farms")
plots_dir      = os.path.join(qa_dir, "plots")          # CSV plot data
figs_plots_dir = os.path.join(figs_dir, "panel-sumstats-by-farms", "plots")  # PNG
for _d in (qa_dir, plots_dir, figs_plots_dir):
    os.makedirs(_d, exist_ok=True)

# ── Column name constants ─────────────────────────────────────────────────────
CDC_DEATHS  = "deaths_cdc_county_year_deathsofdespair"
CDC_POP     = "population_cdc_county_year_deathsofdespair"
CDC_CRUDE   = "crude_rate_cdc_county_year_deathsofdespair"
COUNTY_NAME = "county_name_cdc_county_year_deathsofdespair"
CENSUS_POP  = "population_population_full"

MIN_CORR_N = 200

# ── Load merged panel once ────────────────────────────────────────────────────
merged_path = latest_file_glob(merged_dir, "*_panel.csv")
df = pd.read_csv(merged_path, low_memory=False)
df = normalize_panel_key(df)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()
df = df.sort_values(["fips", "year"]).reset_index(drop=True)
print(f"Merged panel: {len(df):,} rows | {df['fips'].nunique():,} counties | "
      f"{int(df['year'].min())}–{int(df['year'].max())}")


# =============================================================================
# SECTION 1 — CDC crude-rate sense check
# =============================================================================
print("\n--- Section 1: CDC crude-rate sense check ---")

def _summarize_cdc(g):
    d     = g["diff_census_minus_cdc"]
    ad    = g["abs_diff_census_minus_cdc"]
    z     = g["z_diff_from_zero_sd"]
    poppct = g["population_pct_diff_census_vs_cdc"]
    recalc = g["crude_rate_recalc_census_pop"]
    cdc    = g["cdc_crude_rate"]
    return pd.Series({
        "n_rows":                        int(len(g)),
        "n_valid_diff":                  int(d.notna().sum()),
        "mean_diff":                     float(d.mean())              if d.notna().any()           else np.nan,
        "median_diff":                   float(d.median())            if d.notna().any()           else np.nan,
        "sd_diff":                       float(d.std(ddof=1))         if d.notna().sum() > 1       else np.nan,
        "mae_diff":                      float(ad.mean())             if ad.notna().any()          else np.nan,
        "rmse_diff":                     float(np.sqrt(np.mean(d.dropna()**2))) if d.notna().any() else np.nan,
        "p90_abs_diff":                  float(ad.quantile(0.90))     if ad.notna().any()          else np.nan,
        "p95_abs_diff":                  float(ad.quantile(0.95))     if ad.notna().any()          else np.nan,
        "share_abs_diff_le_0_5":         float((ad <= 0.5).mean()*100) if ad.notna().any()         else np.nan,
        "share_abs_diff_le_1_0":         float((ad <= 1.0).mean()*100) if ad.notna().any()         else np.nan,
        "share_abs_diff_le_2_0":         float((ad <= 2.0).mean()*100) if ad.notna().any()         else np.nan,
        "share_abs_z_gt_1":              float((z.abs()>1).mean()*100) if z.notna().any()          else np.nan,
        "share_abs_z_gt_2":              float((z.abs()>2).mean()*100) if z.notna().any()          else np.nan,
        "share_abs_z_gt_3":              float((z.abs()>3).mean()*100) if z.notna().any()          else np.nan,
        "mean_population_pct_diff":      float(poppct.mean())         if poppct.notna().any()      else np.nan,
        "median_population_pct_diff":    float(poppct.median())       if poppct.notna().any()      else np.nan,
        "corr_recalc_vs_cdc":            float(recalc.corr(cdc))      if recalc.notna().sum()>1 and cdc.notna().sum()>1 else np.nan,
    })

required = [CDC_DEATHS, CDC_POP, CDC_CRUDE, CENSUS_POP]
missing  = [c for c in required if c not in df.columns]
if missing:
    print(f"WARNING: skipping CDC sense check — missing columns: {missing}")
else:
    df["cdc_deaths"]       = to_numeric_series(df[CDC_DEATHS])
    df["cdc_population"]   = to_numeric_series(df[CDC_POP])
    df["cdc_crude_rate"]   = to_numeric_series(df[CDC_CRUDE])
    df["census_population"] = to_numeric_series(df[CENSUS_POP])

    df["crude_rate_recalc_census_pop"] = np.where(
        df["census_population"] > 0,
        df["cdc_deaths"] / df["census_population"] * 100_000.0, np.nan,
    )
    df["crude_rate_recalc_cdc_pop"] = np.where(
        df["cdc_population"] > 0,
        df["cdc_deaths"] / df["cdc_population"] * 100_000.0, np.nan,
    )
    df["diff_census_minus_cdc"]     = df["crude_rate_recalc_census_pop"] - df["cdc_crude_rate"]
    df["abs_diff_census_minus_cdc"] = df["diff_census_minus_cdc"].abs()
    df["pct_diff_census_vs_cdc"]    = np.where(
        df["cdc_crude_rate"] != 0,
        (df["diff_census_minus_cdc"] / df["cdc_crude_rate"]) * 100.0, np.nan,
    )
    df["population_diff_census_minus_cdc"]  = df["census_population"] - df["cdc_population"]
    df["population_pct_diff_census_vs_cdc"] = np.where(
        df["cdc_population"] != 0,
        (df["population_diff_census_minus_cdc"] / df["cdc_population"]) * 100.0, np.nan,
    )
    sd_all = df["diff_census_minus_cdc"].std(ddof=1)
    df["z_diff_from_zero_sd"] = (
        df["diff_census_minus_cdc"] / sd_all if (pd.notna(sd_all) and sd_all > 0) else np.nan
    )
    year_sd = df.groupby("year")["diff_census_minus_cdc"].transform(lambda s: s.std(ddof=1))
    df["z_diff_from_zero_sd_within_year"] = np.where(year_sd > 0, df["diff_census_minus_cdc"] / year_sd, np.nan)

    # Rename diagnostic columns with explicit suffix before re-export
    rename_map = {
        "crude_rate_recalc_census_pop":         "crude_rate_recalc_censuspop_cdcsense",
        "crude_rate_recalc_cdc_pop":            "crude_rate_recalc_cdcpop_cdcsense",
        "diff_census_minus_cdc":                "crude_rate_diff_census_minus_cdc_cdcsense",
        "abs_diff_census_minus_cdc":            "crude_rate_abs_diff_census_minus_cdc_cdcsense",
        "pct_diff_census_vs_cdc":               "crude_rate_pct_diff_census_vs_cdc_cdcsense",
        "z_diff_from_zero_sd":                  "crude_rate_diff_z_from_zero_sd_cdcsense",
        "z_diff_from_zero_sd_within_year":      "crude_rate_diff_z_from_zero_sd_within_year_cdcsense",
        "population_diff_census_minus_cdc":     "population_diff_census_minus_cdc_cdcsense",
        "population_pct_diff_census_vs_cdc":    "population_pct_diff_census_vs_cdc_cdcsense",
    }
    for src, dst in rename_map.items():
        df[dst] = df[src]

    cmp = df[
        df["cdc_deaths"].notna() & df["cdc_crude_rate"].notna()
        & df["census_population"].notna() & (df["census_population"] > 0)
    ].copy()

    overall  = _summarize_cdc(cmp).to_frame().T
    overall.insert(0, "source_merged_file", os.path.basename(merged_path))
    overall.insert(1, "year_min", int(cmp["year"].min()) if not cmp.empty else np.nan)
    overall.insert(2, "year_max", int(cmp["year"].max()) if not cmp.empty else np.nan)

    by_year  = (
        cmp.groupby("year", as_index=False)
        .apply(lambda g: _summarize_cdc(g), include_groups=False)
        .reset_index().drop(columns=["index"], errors="ignore")
        .sort_values("year")
    )

    outliers = cmp.copy()
    outliers["abs_z"] = outliers["crude_rate_diff_z_from_zero_sd_cdcsense"].abs()
    keep_cols = [
        "fips", "year",
        COUNTY_NAME if COUNTY_NAME in outliers.columns else None,
        "cdc_deaths", "census_population", "cdc_population", "cdc_crude_rate",
        "crude_rate_recalc_censuspop_cdcsense", "crude_rate_recalc_cdcpop_cdcsense",
        "crude_rate_diff_census_minus_cdc_cdcsense",
        "crude_rate_abs_diff_census_minus_cdc_cdcsense",
        "crude_rate_pct_diff_census_vs_cdc_cdcsense",
        "crude_rate_diff_z_from_zero_sd_cdcsense",
        "crude_rate_diff_z_from_zero_sd_within_year_cdcsense",
        "population_pct_diff_census_vs_cdc_cdcsense", "abs_z",
    ]
    keep_cols = [c for c in keep_cols if c is not None and c in outliers.columns]
    outliers  = outliers[keep_cols].sort_values("abs_z", ascending=False).head(1000)

    # Re-export panel + year slices with diagnostic columns appended
    df.to_csv(os.path.join(merged_dir, f"{today_str}_panel.csv"), index=False)
    df[df["year"].between(2005, 2010, inclusive="both")].to_csv(
        os.path.join(merged_dir, f"{today_str}_panel_05_10.csv"), index=False)
    df[df["year"].between(2010, 2020, inclusive="both")].to_csv(
        os.path.join(merged_dir, f"{today_str}_panel_10_20.csv"), index=False)
    df[df["year"].isin([2002, 2005, 2007, 2012])].to_csv(
        os.path.join(merged_dir, f"{today_str}_panel_census_years.csv"), index=False)

    overall.to_csv(os.path.join(qa_dir, f"{today_str}_qa_cdc_cruderate_sensecheck_overall.csv"),        index=False)
    by_year.to_csv(os.path.join(qa_dir, f"{today_str}_qa_cdc_cruderate_sensecheck_by_year.csv"),         index=False)
    outliers.to_csv(os.path.join(qa_dir, f"{today_str}_qa_cdc_cruderate_sensecheck_outliers_top1000.csv"), index=False)
    print(f"  Rows compared: {len(cmp)} | Global SD(diff): {float(sd_all):.4f}")


# =============================================================================
# SECTION 2 — Correlation tables + FSIS column QA + QA memo
# =============================================================================
print("\n--- Section 2: Correlation tables + FSIS QA + QA memo ---")

cafo_cols = [
    "cafo_cattle_small", "cafo_cattle_medium", "cafo_cattle_large",
    "cafo_hogs_small",   "cafo_hogs_medium",   "cafo_hogs_large",
    "cafo_chickens_small","cafo_chickens_medium","cafo_chickens_large",
    "cafo_total_ops_all_animals", "cafo_total_ops_chickens",
]
mental_cols = [
    "poor_mental_health_days_raw_value_mentalhealthrank_full",
    "frequent_mental_distress_raw_value_mentalhealthrank_full",
    "poor_mental_health_days_raw_value_mh_mortality_fips_yr",
    "frequent_mental_distress_raw_value_mh_mortality_fips_yr",
]
mortality_cols = ["mortality_total_deaths_mh_mortality_fips_yr"]
population_cols = ["population_population_full"]
fsis_12_cols = [
    "n_unique_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_unique_est_size_combos_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_slaughterhouse_present_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_processing_present_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_meat_slaughter_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_poultry_slaughter_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_1_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_2_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_3_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_4_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_5_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_missing_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
]

all_cols      = [*cafo_cols, *mental_cols, *mortality_cols, *population_cols, *fsis_12_cols]
existing_cols = [c for c in all_cols if c in df.columns]
for c in existing_cols:
    df[c] = to_numeric_series(df[c])

# Variable manifest
manifest = pd.DataFrame([
    {"variable": c, "family": (
        "cafo"       if c in cafo_cols       else
        "fsis"       if c in fsis_12_cols    else
        "mental"     if c in mental_cols     else
        "mortality"  if c in mortality_cols  else
        "population"
    )}
    for c in existing_cols
])
manifest.to_csv(os.path.join(qa_dir, "correlation_vars_manifest.csv"), index=False)

# Coverage by year
coverage_rows = []
for yr, g in df.groupby("year", as_index=False):
    row = {"year": int(yr)}
    for c in [
        "cafo_total_ops_all_animals",
        "cafo_total_ops_chickens",
        "poor_mental_health_days_raw_value_mentalhealthrank_full",
        "frequent_mental_distress_raw_value_mentalhealthrank_full",
        "mortality_total_deaths_mh_mortality_fips_yr",
        "n_unique_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
        "population_population_full",
    ]:
        if c in g.columns:
            row[f"fill_pct__{c}"] = float(g[c].notna().mean() * 100)
    coverage_rows.append(row)
pd.DataFrame(coverage_rows).sort_values("year").to_csv(
    os.path.join(qa_dir, "qa_key_variable_coverage_by_year.csv"), index=False)

# FSIS 12-column upper-tail QA
def _safe_float(v):
    try:
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan

fsis_q_rows = []
for c in fsis_12_cols:
    if c not in df.columns:
        continue
    s      = to_numeric_series(df[c])
    nonmiss = s.notna().sum()
    p99    = _safe_float(s.quantile(0.99)) if nonmiss else np.nan
    vmax   = _safe_float(s.max())          if nonmiss else np.nan
    fsis_q_rows.append({
        "variable":          c,
        "non_missing_n":     int(nonmiss),
        "fill_pct":          float(s.notna().mean() * 100),
        "p50":               _safe_float(s.quantile(0.50)) if nonmiss else np.nan,
        "p90":               _safe_float(s.quantile(0.90)) if nonmiss else np.nan,
        "p99":               p99,
        "max":               vmax,
        "share_gt_1000_pct": _safe_float((s > 1000).mean() * 100) if nonmiss else np.nan,
        "qa_reliable_for_now": int((pd.notna(p99) and p99 <= 1000) and (pd.notna(vmax) and vmax <= 1000)),
    })
fsis_qa = pd.DataFrame(fsis_q_rows).sort_values(
    ["qa_reliable_for_now", "share_gt_1000_pct"], ascending=[False, True])
fsis_qa.to_csv(os.path.join(qa_dir, "qa_fsis_12_column_check.csv"), index=False)

# Pairwise correlation tables
def _pairwise_corr(df_num, columns, method, min_n=MIN_CORR_N):
    rows = []
    for i, a in enumerate(columns):
        sa = df_num[a]
        for b in columns[i + 1:]:
            sb   = df_num[b]
            mask = sa.notna() & sb.notna()
            n    = int(mask.sum())
            if n < min_n:
                continue
            corr = sa[mask].corr(sb[mask], method=method)
            if pd.isna(corr):
                continue
            rows.append({"var_a": a, "var_b": b, "corr": float(corr),
                         "abs_corr": float(abs(corr)), "n_obs_pairwise": n, "method": method})
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
        long   = _pairwise_corr(sub_num, use_cols, method=method)
        long.to_csv(os.path.join(qa_dir, f"correlation_long_{wname}_{method}.csv"), index=False)
        long.head(30).to_csv(os.path.join(qa_dir, f"correlation_top30_{wname}_{method}.csv"), index=False)
        sub_num[use_cols].corr(method=method, min_periods=MIN_CORR_N).to_csv(
            os.path.join(qa_dir, f"correlation_matrix_{wname}_{method}.csv"), index=True)

# QA memo (Markdown)
fsis_reliable = fsis_qa.loc[fsis_qa["qa_reliable_for_now"] == 1, "variable"].tolist()
fsis_flagged  = fsis_qa.loc[fsis_qa["qa_reliable_for_now"] == 0, "variable"].tolist()

coverage_check_path = os.path.join(qa_dir, "county_coverage_check.csv")
crosscheck_path     = os.path.join(qa_dir, "cafo_animal_size_crosscheck_vs_premerged.csv")
coverage_check = pd.read_csv(coverage_check_path) if os.path.exists(coverage_check_path) else pd.DataFrame()
crosscheck     = pd.read_csv(crosscheck_path)     if os.path.exists(crosscheck_path)     else pd.DataFrame()

county_match = "n/a"
key_match    = "n/a"
if not coverage_check.empty:
    county_match = "PASS" if int(coverage_check["county_count_match"].iloc[0]) == 1 else "FAIL"
    key_match    = "PASS" if int(coverage_check["key_set_exact_match"].iloc[0]) == 1 else "FAIL"

cafo_cross = "n/a"
if not crosscheck.empty and "pct_exact_match_on_compared_keys" in crosscheck.columns:
    cafo_cross = (f"{crosscheck['pct_exact_match_on_compared_keys'].min():.1f}% to "
                  f"{crosscheck['pct_exact_match_on_compared_keys'].max():.1f}% exact-match")

memo_lines = [
    f"# QA Memo: Panel + Correlation Tables ({today_str})", "",
    "## 1) Scope",
    f"- Merged file: `{os.path.basename(merged_path)}`",
    f"- Rows: `{len(df):,}` | Counties: `{df['fips'].nunique():,}` | Years: `{int(df['year'].min())}–{int(df['year'].max())}`", "",
    "## 2) Key QA Checks",
    f"- County-count check vs rural kept keys: **{county_match}**",
    f"- Exact key-set match vs rural kept keys: **{key_match}**",
    f"- CAFO animal×size merged-vs-premerged cross-check: **{cafo_cross}**", "",
    "## 3) FSIS 12-Column QA",
    "- Flag rule: `p99 > 1000` or `max > 1000` → unreliable.",
    f"- Reliable ({len(fsis_reliable)}): {', '.join([f'`{c}`' for c in fsis_reliable]) or 'none'}",
    f"- Flagged ({len(fsis_flagged)}): {', '.join([f'`{c}`' for c in fsis_flagged]) or 'none'}", "",
    "## 4) Correlation Tables",
    "- Pearson = linear; Spearman = rank-based (robust to skew).",
    f"- Pairwise minimum n: `{MIN_CORR_N}`.",
    "- Use 2010–2020 tables for CAFO + mental health; 2017 tables when including FSIS.", "",
    "## 5) Outputs",
    "- `correlation_long_*.csv`, `correlation_top30_*.csv`, `correlation_matrix_*.csv`",
    "- `qa_fsis_12_column_check.csv`, `qa_key_variable_coverage_by_year.csv`",
]
with open(os.path.join(qa_dir, "qa_memo_panel_sumstats_by_farms.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(memo_lines) + "\n")

print(f"  Outputs saved to: {qa_dir}")


# =============================================================================
# SECTION 3 — FSIS size bins vs. poor mental health days (2017 cross-section)
# =============================================================================
print("\n--- Section 3: FSIS size bins vs. poor mental health days (2017) ---")

mental_col = "poor_mental_health_days_raw_value_mentalhealthrank_full"
size_cols  = [
    "n_size_bucket_1_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_2_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_3_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_4_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_5_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
]

missing_cols = [c for c in [mental_col, *size_cols] if c not in df.columns]
if missing_cols:
    print(f"  WARNING: skipping — missing columns: {missing_cols}")
else:
    df_2017 = df[df["year"] == 2017].copy()
    df_2017[mental_col] = to_numeric_series(df_2017[mental_col])
    for c in size_cols:
        df_2017[c] = to_numeric_series(df_2017[c]).fillna(0)  # NA = 0 establishments in bin
    df_2017 = df_2017[df_2017[mental_col].notna()].copy()

    rename_map = {c: f"Size Bin {i+1}" for i, c in enumerate(size_cols)}
    long_df = df_2017.melt(
        id_vars=["fips", "year", mental_col],
        value_vars=size_cols,
        var_name="size_bin_col",
        value_name="fsis_establishments_in_bin",
    )
    long_df["size_bin"] = long_df["size_bin_col"].map(rename_map)

    plot_csv = os.path.join(plots_dir, f"{today_str}_fsis_size_bins_vs_poor_mental_health_2017_plotdata.csv")
    long_df.to_csv(plot_csv, index=False)

    sns.set_theme(style="whitegrid", context="talk")
    g = sns.relplot(
        data=long_df,
        x="fsis_establishments_in_bin", y=mental_col,
        col="size_bin",
        col_order=[f"Size Bin {i+1}" for i in range(5)],
        col_wrap=3,
        kind="scatter", alpha=0.55, s=22,
        facet_kws={"sharex": False, "sharey": True},
        height=4.0, aspect=1.15, color="#2b8cbe",
    )
    g.set_axis_labels("FSIS Establishments in Size Bin (county, 2017)", "Poor Mental Health Days")
    g.set_titles("{col_name}")
    g.fig.suptitle("County-Level FSIS Size-Bin Counts vs Poor Mental Health (2017)", y=1.03)

    fig_png = os.path.join(figs_plots_dir, f"{today_str}_fsis_size_bins_vs_poor_mental_health_2017_facets.png")
    g.savefig(fig_png, dpi=240, bbox_inches="tight")
    plt.close(g.fig)

    print(f"  2017 rows with mental outcome: {len(df_2017)}")
    print(f"  Saved: {plot_csv}")
    print(f"  Saved: {fig_png}")

print("\nQA pipeline complete.")
