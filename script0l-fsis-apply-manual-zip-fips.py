#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply manual ZIP->FIPS mappings to latest FSIS HUD-bulk interim file.

Rules:
- Never overwrite existing fips_code.
- Fill only rows where fips_code is missing and zip5 has a manual mapping.
- Apply across all years (ZIP-level manual fallback).
- Rebuild county-year summary from updated interim.
"""

import os
import re
from datetime import date

import pandas as pd


db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
clean_dir = os.path.join(db_data, "clean")
qa_dir = os.path.join(db_data, "FOIA-USDA-request", "qa-fsis")
os.makedirs(qa_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")


def _latest_file(dirpath: str, pattern: str) -> str:
    pat = re.compile(pattern)
    candidates = []
    for fn in os.listdir(dirpath):
        m = pat.match(fn)
        if m:
            candidates.append((m.group(1), os.path.join(dirpath, fn)))
    if not candidates:
        raise FileNotFoundError(f"No file matching pattern in {dirpath}: {pattern}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _normalize_zip(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.str.extract(r"(\d{5})", expand=False)


def _normalize_fips(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.extract(r"(\d+)", expand=False)
    s = s.where(s.str.len().isin([4, 5]), pd.NA)
    s = s.where(s.str.len() != 4, "0" + s)
    return s


def _first_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else pd.NA


def _build_county_year(df: pd.DataFrame) -> pd.DataFrame:
    src = df[df["fips_code"].notna() & df["year"].notna()].copy()
    src["year"] = pd.to_numeric(src["year"], errors="coerce").astype("Int64")
    src = src.dropna(subset=["year"]).copy()
    src["year"] = src["year"].astype(int)
    src["est_size_combo_key"] = src["est_key"].astype("string") + "::" + src["size_bucket_final"].astype("string")

    # Ensure binary indicators are numeric before aggregation.
    indicator_cols = [
        "slaughterhouse_present_year",
        "processing_present_year",
        "meat_slaughter_present_year",
        "poultry_slaughter_present_year",
        "type_both_slaughter_and_processing",
        "type_slaughter_only",
        "type_processing_only",
        "type_other_or_unclear",
        "type_neither_signal",
    ]
    for c in indicator_cols:
        if c in src.columns:
            src[c] = pd.to_numeric(src[c], errors="coerce").fillna(0).astype("Int64")
        else:
            src[c] = 0

    for b in ["1", "2", "3", "4", "5", "missing"]:
        src[f"size_bucket_{b}"] = (src["size_bucket_final"].astype("string") == b).astype("Int64")

    agg = {
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

    county = (
        src.groupby(["fips_code", "year"], as_index=False)
        .agg(**agg)
        .rename(columns={"fips_code": "fips"})
        .sort_values(["fips", "year"])
        .reset_index(drop=True)
    )
    return county


def main():
    src_interim = _latest_file(
        clean_dir,
        r"^(\d{4}-\d{2}-\d{2})_fsis_establishment_year_fips_size_type_interim_hudbulk\.csv$",
    )
    src_county = _latest_file(
        clean_dir,
        r"^(\d{4}-\d{2}-\d{2})_fsis_county_year_fips_est_size_type_summary_hudbulk\.csv$",
    )
    src_manual = _latest_file(
        qa_dir,
        r"^(\d{4}-\d{2}-\d{2})_fsis_unmatched_unique_zip_for_manual_fips\.xlsx$",
    )

    print("Using interim source:", src_interim)
    print("Using county source:", src_county)
    print("Using manual ZIP template:", src_manual)

    df = pd.read_csv(src_interim, dtype=str, low_memory=False)
    county_before = pd.read_csv(src_county, dtype=str, low_memory=False)
    manual = pd.read_excel(src_manual, dtype=str)

    if "zip5" not in manual.columns or "fips_manual" not in manual.columns:
        raise KeyError("Manual file must include columns: zip5, fips_manual")

    df["zip5"] = _normalize_zip(df["zip5"] if "zip5" in df.columns else df.get("zip", pd.Series([pd.NA] * len(df))))
    df["fips_code"] = _normalize_fips(df["fips_code"])

    manual = manual[["zip5", "fips_manual"]].copy()
    manual["zip5"] = _normalize_zip(manual["zip5"])
    manual["fips_manual"] = _normalize_fips(manual["fips_manual"])
    manual = manual.dropna(subset=["zip5", "fips_manual"]).drop_duplicates(["zip5"], keep="first")

    out = df.copy()
    out = out.merge(manual.rename(columns={"fips_manual": "fips_manual_from_zip"}), on="zip5", how="left")

    mask_fill = out["fips_code"].isna() & out["fips_manual_from_zip"].notna()
    out.loc[mask_fill, "fips_code"] = out.loc[mask_fill, "fips_manual_from_zip"]
    out.loc[mask_fill, "fips_fill_method"] = out.loc[mask_fill, "fips_fill_method"].where(
        out.loc[mask_fill, "fips_fill_method"].notna(), "manual_zip_template"
    )
    out["manual_zip_fips_filled"] = mask_fill.astype("Int64")

    county_before_rebuilt = _build_county_year(df)
    county_after = _build_county_year(out)

    interim_out = os.path.join(clean_dir, f"{today_str}_fsis_establishment_year_fips_size_type_interim_hudbulk_manualzip.csv")
    county_out = os.path.join(clean_dir, f"{today_str}_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip.csv")
    qa_map_out = os.path.join(qa_dir, f"{today_str}_fsis_manual_zip_fips_mapping_used.csv")
    qa_diff_out = os.path.join(qa_dir, f"{today_str}_fsis_manual_zip_fill_diff_metrics.csv")

    out.to_csv(interim_out, index=False)
    county_after.to_csv(county_out, index=False)
    manual.to_csv(qa_map_out, index=False)

    pre_missing = int(df["fips_code"].isna().sum())
    post_missing = int(out["fips_code"].isna().sum())

    qa = pd.DataFrame(
        {
            "metric": [
                "n_rows_interim_before",
                "n_rows_interim_after",
                "n_missing_fips_before",
                "n_missing_fips_after",
                "n_rows_filled_manual_zip",
                "n_unique_manual_zip_in_template",
                "n_unique_manual_zip_used_to_fill",
                "n_county_year_rows_before_source",
                "n_county_year_rows_before_rebuilt",
                "n_county_year_rows_after",
                "delta_county_year_rows_rebuilt",
                "n_unique_county_fips_before_source",
                "n_unique_county_fips_before_rebuilt",
                "n_unique_county_fips_after",
                "delta_unique_county_fips_rebuilt",
            ],
            "value": [
                len(df),
                len(out),
                pre_missing,
                post_missing,
                int(mask_fill.sum()),
                int(manual["zip5"].nunique()),
                int(out.loc[mask_fill, "zip5"].nunique()),
                int(len(county_before)),
                int(len(county_before_rebuilt)),
                int(len(county_after)),
                int(len(county_after) - len(county_before_rebuilt)),
                int(county_before["fips"].astype("string").dropna().nunique()) if "fips" in county_before.columns else pd.NA,
                int(county_before_rebuilt["fips"].astype("string").dropna().nunique()),
                int(county_after["fips"].astype("string").dropna().nunique()),
                int(
                    county_after["fips"].astype("string").dropna().nunique()
                    - county_before_rebuilt["fips"].astype("string").dropna().nunique()
                ),
            ],
        }
    )
    qa.to_csv(qa_diff_out, index=False)

    print("Saved:", interim_out)
    print("Saved:", county_out)
    print("Saved:", qa_map_out)
    print("Saved:", qa_diff_out)
    print("Manual ZIP rows filled:", int(mask_fill.sum()))
    print("Missing FIPS before:", pre_missing, "after:", post_missing)


if __name__ == "__main__":
    main()
