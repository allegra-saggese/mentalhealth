#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDC crude-rate sense check:
- Recalculate crude rate from census population in merged panel
- Compare to CDC crude rate from deaths-of-despair county-year file
- Add comparison columns to merged panel
- Export QA summaries (overall, by-year, outliers)
"""

import glob
import os
from datetime import date

import numpy as np
import pandas as pd


MERGED_DIR = "/Users/allegrasaggese/Dropbox/Mental/Data/merged"
OUT_QA_DIR = "/Users/allegrasaggese/Dropbox/Mental/Data/merged/figs/panel-sumstats-by-farms"

CDC_DEATHS = "deaths_cdc_county_year_deathsofdespair"
CDC_POP = "population_cdc_county_year_deathsofdespair"
CDC_CRUDE = "crude_rate_cdc_county_year_deathsofdespair"
COUNTY_NAME = "county_name_cdc_county_year_deathsofdespair"
CENSUS_POP = "population_population_full"


def latest_file(folder, pattern):
    hits = glob.glob(os.path.join(folder, pattern))
    if not hits:
        raise FileNotFoundError(f"No files for pattern {pattern} in {folder}")
    return max(hits, key=os.path.getmtime)


def to_num(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s2 = s.astype("string").str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s2, errors="coerce")


def norm_key(df):
    if "fips" not in df.columns and "fips_generated" in df.columns:
        df = df.rename(columns={"fips_generated": "fips"})
    if "year" not in df.columns and "yr" in df.columns:
        df = df.rename(columns={"yr": "year"})
    if "fips" not in df.columns or "year" not in df.columns:
        raise KeyError("Merged panel missing fips/year")
    df["fips"] = (
        df["fips"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df.dropna(subset=["fips", "year"]).copy()


def summarize(g):
    d = g["diff_census_minus_cdc"]
    ad = g["abs_diff_census_minus_cdc"]
    z = g["z_diff_from_zero_sd"]
    poppct = g["population_pct_diff_census_vs_cdc"]
    recalc = g["crude_rate_recalc_census_pop"]
    cdc = g["cdc_crude_rate"]

    out = {
        "n_rows": int(len(g)),
        "n_valid_diff": int(d.notna().sum()),
        "mean_diff": float(d.mean()) if d.notna().any() else np.nan,
        "median_diff": float(d.median()) if d.notna().any() else np.nan,
        "sd_diff": float(d.std(ddof=1)) if d.notna().sum() > 1 else np.nan,
        "mae_diff": float(ad.mean()) if ad.notna().any() else np.nan,
        "rmse_diff": float(np.sqrt(np.mean((d.dropna()) ** 2))) if d.notna().any() else np.nan,
        "p90_abs_diff": float(ad.quantile(0.90)) if ad.notna().any() else np.nan,
        "p95_abs_diff": float(ad.quantile(0.95)) if ad.notna().any() else np.nan,
        "share_abs_diff_le_0_5": float((ad <= 0.5).mean() * 100) if ad.notna().any() else np.nan,
        "share_abs_diff_le_1_0": float((ad <= 1.0).mean() * 100) if ad.notna().any() else np.nan,
        "share_abs_diff_le_2_0": float((ad <= 2.0).mean() * 100) if ad.notna().any() else np.nan,
        "share_abs_z_gt_1": float((z.abs() > 1).mean() * 100) if z.notna().any() else np.nan,
        "share_abs_z_gt_2": float((z.abs() > 2).mean() * 100) if z.notna().any() else np.nan,
        "share_abs_z_gt_3": float((z.abs() > 3).mean() * 100) if z.notna().any() else np.nan,
        "mean_population_pct_diff": float(poppct.mean()) if poppct.notna().any() else np.nan,
        "median_population_pct_diff": float(poppct.median()) if poppct.notna().any() else np.nan,
        "corr_recalc_vs_cdc": float(recalc.corr(cdc)) if recalc.notna().sum() > 1 and cdc.notna().sum() > 1 else np.nan,
    }
    return pd.Series(out)


def main():
    os.makedirs(OUT_QA_DIR, exist_ok=True)

    merged_path = latest_file(MERGED_DIR, "*_full_merged.csv")
    df = pd.read_csv(merged_path, low_memory=False)
    df = norm_key(df)
    df = df.drop_duplicates(["fips", "year"]).sort_values(["fips", "year"]).reset_index(drop=True)

    required = [CDC_DEATHS, CDC_POP, CDC_CRUDE, CENSUS_POP]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Merged panel missing required columns: {missing}")

    # Numeric casts
    df["cdc_deaths"] = to_num(df[CDC_DEATHS])
    df["cdc_population"] = to_num(df[CDC_POP])
    df["cdc_crude_rate"] = to_num(df[CDC_CRUDE])
    df["census_population"] = to_num(df[CENSUS_POP])

    # Recalculations
    df["crude_rate_recalc_census_pop"] = np.where(
        df["census_population"] > 0,
        df["cdc_deaths"] / df["census_population"] * 100000.0,
        np.nan,
    )
    df["crude_rate_recalc_cdc_pop"] = np.where(
        df["cdc_population"] > 0,
        df["cdc_deaths"] / df["cdc_population"] * 100000.0,
        np.nan,
    )

    # Differences
    df["diff_census_minus_cdc"] = df["crude_rate_recalc_census_pop"] - df["cdc_crude_rate"]
    df["abs_diff_census_minus_cdc"] = df["diff_census_minus_cdc"].abs()
    df["pct_diff_census_vs_cdc"] = np.where(
        df["cdc_crude_rate"] != 0,
        (df["diff_census_minus_cdc"] / df["cdc_crude_rate"]) * 100.0,
        np.nan,
    )
    df["population_diff_census_minus_cdc"] = df["census_population"] - df["cdc_population"]
    df["population_pct_diff_census_vs_cdc"] = np.where(
        df["cdc_population"] != 0,
        (df["population_diff_census_minus_cdc"] / df["cdc_population"]) * 100.0,
        np.nan,
    )

    # SD-based distance from zero for crude-rate difference
    sd_all = df["diff_census_minus_cdc"].std(ddof=1)
    if pd.notna(sd_all) and sd_all > 0:
        df["z_diff_from_zero_sd"] = df["diff_census_minus_cdc"] / sd_all
    else:
        df["z_diff_from_zero_sd"] = np.nan

    # Year-specific SD distance
    year_sd = df.groupby("year")["diff_census_minus_cdc"].transform(lambda s: s.std(ddof=1))
    df["z_diff_from_zero_sd_within_year"] = np.where(year_sd > 0, df["diff_census_minus_cdc"] / year_sd, np.nan)

    # Add "built-in" comparison columns with explicit suffix
    rename_map = {
        "crude_rate_recalc_census_pop": "crude_rate_recalc_censuspop_cdcsense",
        "crude_rate_recalc_cdc_pop": "crude_rate_recalc_cdcpop_cdcsense",
        "diff_census_minus_cdc": "crude_rate_diff_census_minus_cdc_cdcsense",
        "abs_diff_census_minus_cdc": "crude_rate_abs_diff_census_minus_cdc_cdcsense",
        "pct_diff_census_vs_cdc": "crude_rate_pct_diff_census_vs_cdc_cdcsense",
        "z_diff_from_zero_sd": "crude_rate_diff_z_from_zero_sd_cdcsense",
        "z_diff_from_zero_sd_within_year": "crude_rate_diff_z_from_zero_sd_within_year_cdcsense",
        "population_diff_census_minus_cdc": "population_diff_census_minus_cdc_cdcsense",
        "population_pct_diff_census_vs_cdc": "population_pct_diff_census_vs_cdc_cdcsense",
    }

    for src, dst in rename_map.items():
        df[dst] = df[src]

    # QA summaries
    cmp = df[
        df["cdc_deaths"].notna()
        & df["cdc_crude_rate"].notna()
        & df["census_population"].notna()
        & (df["census_population"] > 0)
    ].copy()

    overall = summarize(cmp).to_frame().T
    overall.insert(0, "source_merged_file", os.path.basename(merged_path))
    overall.insert(1, "year_min", int(cmp["year"].min()) if not cmp.empty else np.nan)
    overall.insert(2, "year_max", int(cmp["year"].max()) if not cmp.empty else np.nan)

    by_year = (
        cmp.groupby("year", as_index=False)
        .apply(lambda g: summarize(g), include_groups=False)
        .reset_index()
        .drop(columns=["index"], errors="ignore")
        .sort_values("year")
    )

    outliers = cmp.copy()
    outliers["abs_z"] = outliers["z_diff_from_zero_sd"].abs()
    keep_cols = [
        "fips",
        "year",
        COUNTY_NAME if COUNTY_NAME in outliers.columns else None,
        "cdc_deaths",
        "census_population",
        "cdc_population",
        "cdc_crude_rate",
        "crude_rate_recalc_census_pop",
        "crude_rate_recalc_cdc_pop",
        "diff_census_minus_cdc",
        "abs_diff_census_minus_cdc",
        "pct_diff_census_vs_cdc",
        "z_diff_from_zero_sd",
        "z_diff_from_zero_sd_within_year",
        "population_pct_diff_census_vs_cdc",
        "abs_z",
    ]
    keep_cols = [c for c in keep_cols if c is not None and c in outliers.columns]
    outliers = outliers[keep_cols].sort_values("abs_z", ascending=False).head(1000)

    today = date.today().strftime("%Y-%m-%d")

    # Save updated merged + slices with today's date
    out_main = os.path.join(MERGED_DIR, f"{today}_full_merged.csv")
    out_05010 = os.path.join(MERGED_DIR, f"{today}_full_merged_2005_2010.csv")
    out_1020 = os.path.join(MERGED_DIR, f"{today}_full_merged_2010_2020.csv")
    out_census = os.path.join(MERGED_DIR, f"{today}_full_merged_census_years.csv")

    df.to_csv(out_main, index=False)
    df[df["year"].between(2005, 2010, inclusive="both")].to_csv(out_05010, index=False)
    df[df["year"].between(2010, 2020, inclusive="both")].to_csv(out_1020, index=False)
    df[df["year"].isin([2002, 2005, 2007, 2012])].to_csv(out_census, index=False)

    # Save QA files
    qa_overall = os.path.join(OUT_QA_DIR, f"{today}_qa_cdc_cruderate_sensecheck_overall.csv")
    qa_by_year = os.path.join(OUT_QA_DIR, f"{today}_qa_cdc_cruderate_sensecheck_by_year.csv")
    qa_outliers = os.path.join(OUT_QA_DIR, f"{today}_qa_cdc_cruderate_sensecheck_outliers_top1000.csv")
    overall.to_csv(qa_overall, index=False)
    by_year.to_csv(qa_by_year, index=False)
    outliers.to_csv(qa_outliers, index=False)

    print("Saved merged:", out_main)
    print("Saved merged slice:", out_05010)
    print("Saved merged slice:", out_1020)
    print("Saved merged slice:", out_census)
    print("Saved QA:", qa_overall)
    print("Saved QA:", qa_by_year)
    print("Saved QA:", qa_outliers)
    print("Rows compared:", len(cmp))
    print("Global SD(diff):", float(sd_all) if pd.notna(sd_all) else np.nan)


if __name__ == "__main__":
    main()
