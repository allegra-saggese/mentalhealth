#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build FSIS establishment-year size/type interim panel and county-year summary.

Input:
- Data/clean/2026-03-07_fsis_establishment_year_all.csv

Outputs (Data/clean):
- YYYY-MM-DD_fsis_establishment_year_fips_size_type_interim.csv
- YYYY-MM-DD_fsis_county_year_fips_est_size_type_summary.csv
"""

import os
import re
from datetime import date

import numpy as np
import pandas as pd


db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
clean_dir = os.path.join(db_data, "clean")
qa_dir = os.path.join(db_data, "FOIA-USDA-request", "qa-fsis")

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(qa_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")
src = os.path.join(clean_dir, "2026-03-07_fsis_establishment_year_all.csv")


def _norm_text(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.lower()
    out = out.str.replace(r"[^a-z]+", "", regex=True)
    return out


def _first_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else pd.NA


def main():
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing source file: {src}")

    df = pd.read_csv(src, low_memory=False)
    if "est_key" not in df.columns or "year" not in df.columns:
        raise KeyError("Source file must contain est_key and year columns.")

    # Normalize key fields
    for c in [
        "est_key",
        "establishment_id",
        "establishment_number",
        "establishment_name",
        "fips_code",
        "county",
        "state",
        "operation_category_year",
        "size_classifier_mode",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["fips_code"] = (
        df["fips_code"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(5)
    )
    df.loc[df["fips_code"].str.len() != 5, "fips_code"] = pd.NA

    # Build size code using numeric size fields first, then cleaned text fallback.
    size_slaughter = pd.to_numeric(df.get("slaughter_volume_category_mode"), errors="coerce")
    size_processing = pd.to_numeric(df.get("processing_volume_category_mode"), errors="coerce")

    size_txt_key = _norm_text(df.get("size_classifier_mode", pd.Series(pd.NA, index=df.index)))
    txt_map = {
        "verysmall": 1,
        "small": 2,
        "medium": 3,
        "large": 4,
        "verylarge": 5,
    }
    size_text = size_txt_key.map(txt_map)

    size_code_final = size_slaughter.combine_first(size_processing).combine_first(size_text)
    size_source = np.where(
        size_slaughter.notna(),
        "slaughter_volume_category_mode",
        np.where(
            size_processing.notna(),
            "processing_volume_category_mode",
            np.where(size_text.notna(), "size_classifier_mode_text", "missing"),
        ),
    )

    df["size_code_final"] = pd.to_numeric(size_code_final, errors="coerce").astype("Int64")
    df["size_source"] = pd.Series(size_source, index=df.index, dtype="string")
    df["size_text_key"] = size_txt_key
    df["size_bucket_final"] = df["size_code_final"].astype("Int64").astype("string").fillna("missing")

    # Ensure operation category exists and make dummies
    allowed_cats = [
        "both_slaughter_and_processing",
        "slaughter_only",
        "processing_only",
        "other_or_unclear",
        "neither_signal",
    ]
    df["operation_category_year"] = df["operation_category_year"].fillna("other_or_unclear")
    df.loc[~df["operation_category_year"].isin(allowed_cats), "operation_category_year"] = "other_or_unclear"

    for cat in allowed_cats:
        col = f"type_{cat}"
        df[col] = (df["operation_category_year"] == cat).astype("Int64")

    # Existing signal columns -> strict 0/1 ints
    signal_cols = [
        "slaughterhouse_present_year",
        "processing_present_year",
        "meat_slaughter_present_year",
        "poultry_slaughter_present_year",
    ]
    for c in signal_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("Int64")
        else:
            df[c] = pd.Series(0, index=df.index, dtype="Int64")

    # Keep one row per establishment-year (defensive dedupe)
    dedupe_cols = ["est_key", "year"]
    prefer_cols = [
        "fips_code",
        "size_code_final",
        "operation_category_year",
        "slaughterhouse_present_year",
        "processing_present_year",
    ]
    df["_non_null_score"] = df[prefer_cols].notna().sum(axis=1).astype("Int64")
    df = df.sort_values(dedupe_cols + ["_non_null_score"], ascending=[True, True, False])
    est_year = df.drop_duplicates(dedupe_cols, keep="first").copy()
    est_year = est_year.drop(columns=["_non_null_score"])

    interim_cols = [
        "est_key",
        "establishment_id",
        "establishment_number",
        "establishment_name",
        "year",
        "fips_code",
        "county",
        "city",
        "state",
        "zip",
        "operation_category_year",
        "type_both_slaughter_and_processing",
        "type_slaughter_only",
        "type_processing_only",
        "type_other_or_unclear",
        "type_neither_signal",
        "slaughterhouse_present_year",
        "processing_present_year",
        "meat_slaughter_present_year",
        "poultry_slaughter_present_year",
        "size_code_final",
        "size_bucket_final",
        "size_source",
        "size_text_key",
        "size_classifier_mode",
        "processing_volume_category_mode",
        "slaughter_volume_category_mode",
        "geo_source_primary",
        "fips_fill_method",
        "geo_fallback_from_year",
        "geo_forward_from_year",
    ]
    interim_cols = [c for c in interim_cols if c in est_year.columns]
    interim = est_year[interim_cols].copy()

    interim_out = os.path.join(clean_dir, f"{today_str}_fsis_establishment_year_fips_size_type_interim.csv")
    interim.to_csv(interim_out, index=False)

    # Build county-year summary from establishments with FIPS.
    county_src = interim.loc[interim["fips_code"].notna() & interim["year"].notna()].copy()
    county_src["year"] = pd.to_numeric(county_src["year"], errors="coerce").astype("Int64")
    county_src = county_src.dropna(subset=["year"])
    county_src["year"] = county_src["year"].astype(int)

    county_src["est_size_combo_key"] = (
        county_src["est_key"].astype("string") + "::" + county_src["size_bucket_final"].astype("string")
    )

    # Size dummies for summary columns
    for b in ["1", "2", "3", "4", "5", "missing"]:
        county_src[f"size_bucket_{b}"] = (county_src["size_bucket_final"] == b).astype("Int64")

    agg_cols = {
        "n_unique_establishments": ("est_key", "nunique"),
        "n_unique_est_size_combos": ("est_size_combo_key", "nunique"),
        "n_slaughterhouse_present_establishments": ("slaughterhouse_present_year", "sum"),
        "n_processing_present_establishments": ("processing_present_year", "sum"),
        "n_meat_slaughter_establishments": ("meat_slaughter_present_year", "sum"),
        "n_poultry_slaughter_establishments": ("poultry_slaughter_present_year", "sum"),
        "n_type_both_slaughter_and_processing": ("type_both_slaughter_and_processing", "sum"),
        "n_type_slaughter_only": ("type_slaughter_only", "sum"),
        "n_type_processing_only": ("type_processing_only", "sum"),
        "n_type_other_or_unclear": ("type_other_or_unclear", "sum"),
        "n_type_neither_signal": ("type_neither_signal", "sum"),
        "n_size_bucket_1": ("size_bucket_1", "sum"),
        "n_size_bucket_2": ("size_bucket_2", "sum"),
        "n_size_bucket_3": ("size_bucket_3", "sum"),
        "n_size_bucket_4": ("size_bucket_4", "sum"),
        "n_size_bucket_5": ("size_bucket_5", "sum"),
        "n_size_bucket_missing": ("size_bucket_missing", "sum"),
        "county_name_any": ("county", _first_non_null),
        "state_any": ("state", _first_non_null),
    }

    county_summary = (
        county_src.groupby(["fips_code", "year"], as_index=False)
        .agg(**agg_cols)
        .rename(columns={"fips_code": "fips"})
        .sort_values(["fips", "year"])
        .reset_index(drop=True)
    )

    final_out = os.path.join(clean_dir, f"{today_str}_fsis_county_year_fips_est_size_type_summary.csv")
    county_summary.to_csv(final_out, index=False)

    # Small QA artifact for quick sanity checks
    qa_metrics = pd.DataFrame(
        {
            "metric": [
                "n_interim_rows",
                "n_interim_unique_est_key",
                "n_interim_missing_fips",
                "n_interim_missing_size",
                "n_county_year_rows",
                "n_county_unique_fips",
                "n_county_year_est_size_combos_total",
            ],
            "value": [
                len(interim),
                int(interim["est_key"].nunique()),
                int(interim["fips_code"].isna().sum()),
                int((interim["size_bucket_final"] == "missing").sum()),
                len(county_summary),
                int(county_summary["fips"].nunique()),
                int(county_summary["n_unique_est_size_combos"].sum()),
            ],
        }
    )
    qa_out = os.path.join(qa_dir, f"{today_str}_fsis_size_type_panel_metrics.csv")
    qa_metrics.to_csv(qa_out, index=False)

    print("Saved:", interim_out)
    print("Saved:", final_out)
    print("Saved QA:", qa_out)
    print("Interim shape:", interim.shape)
    print("County-year shape:", county_summary.shape)


if __name__ == "__main__":
    main()
