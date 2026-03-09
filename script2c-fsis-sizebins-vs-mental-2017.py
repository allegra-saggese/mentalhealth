#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot 2017 county-level FSIS size-bin counts vs poor mental health days.

Outputs:
- Data/merged/figs/panel-sumstats-by-farms/plots/YYYY-MM-DD_fsis_size_bins_vs_poor_mental_health_2017_facets.png
- Data/merged/figs/panel-sumstats-by-farms/plots/YYYY-MM-DD_fsis_size_bins_vs_poor_mental_health_2017_plotdata.csv
"""

import glob
import os
import re
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _latest_file(folder: str, pattern: str) -> str:
    hits = glob.glob(os.path.join(folder, pattern))
    if not hits:
        raise RuntimeError(f"No files found for pattern {pattern} in {folder}")
    return max(hits, key=os.path.getmtime)


def _to_num(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(
        s.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def main():
    db_data = os.path.expanduser("~/Dropbox/Mental/Data")
    merged_dir = os.path.join(db_data, "merged")
    out_dir = os.path.join(merged_dir, "figs", "panel-sumstats-by-farms", "plots")
    os.makedirs(out_dir, exist_ok=True)

    today_str = date.today().strftime("%Y-%m-%d")
    merged_path = _latest_file(merged_dir, "*_full_merged.csv")

    df = pd.read_csv(merged_path, low_memory=False)

    year_col = "year"
    mental_col = "poor_mental_health_days_raw_value_mentalhealthrank_full"
    size_cols = [
        "n_size_bucket_1_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
        "n_size_bucket_2_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
        "n_size_bucket_3_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
        "n_size_bucket_4_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
        "n_size_bucket_5_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    ]

    missing_cols = [c for c in [year_col, mental_col, *size_cols] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df[df[year_col] == 2017].copy()

    df[mental_col] = _to_num(df[mental_col])
    for c in size_cols:
        # Treat NA as zero establishments in that bin for county-year rows.
        df[c] = _to_num(df[c]).fillna(0)

    df = df[df[mental_col].notna()].copy()

    rename_map = {
        size_cols[0]: "Size Bin 1",
        size_cols[1]: "Size Bin 2",
        size_cols[2]: "Size Bin 3",
        size_cols[3]: "Size Bin 4",
        size_cols[4]: "Size Bin 5",
    }

    long_df = df.melt(
        id_vars=["fips", year_col, mental_col],
        value_vars=size_cols,
        var_name="size_bin_col",
        value_name="fsis_establishments_in_bin",
    )
    long_df["size_bin"] = long_df["size_bin_col"].map(rename_map)

    plot_data_out = os.path.join(
        out_dir,
        f"{today_str}_fsis_size_bins_vs_poor_mental_health_2017_plotdata.csv",
    )
    long_df.to_csv(plot_data_out, index=False)

    sns.set_theme(style="whitegrid", context="talk")
    g = sns.relplot(
        data=long_df,
        x="fsis_establishments_in_bin",
        y=mental_col,
        col="size_bin",
        col_order=["Size Bin 1", "Size Bin 2", "Size Bin 3", "Size Bin 4", "Size Bin 5"],
        col_wrap=3,
        kind="scatter",
        alpha=0.55,
        s=22,
        facet_kws={"sharex": False, "sharey": True},
        height=4.0,
        aspect=1.15,
        color="#2b8cbe",
    )

    g.set_axis_labels("FSIS Establishments in Size Bin (county, 2017)", "Poor Mental Health Days")
    g.set_titles("{col_name}")
    g.fig.suptitle("County-Level FSIS Size-Bin Counts vs Poor Mental Health (2017)", y=1.03)

    fig_out = os.path.join(
        out_dir,
        f"{today_str}_fsis_size_bins_vs_poor_mental_health_2017_facets.png",
    )
    g.savefig(fig_out, dpi=240, bbox_inches="tight")
    plt.close(g.fig)

    print("Using merged panel:", merged_path)
    print("Rows in 2017 with mental outcome:", len(df))
    print("Saved:", plot_data_out)
    print("Saved:", fig_out)


if __name__ == "__main__":
    main()
