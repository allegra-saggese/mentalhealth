#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply manual county-label -> FIPS mappings to FSIS interim panel.

Expected manual file:
- YYYY-MM-DD_fsis_missing_county_labels_manual_template.csv
- columns include: state, county_label_raw, and either fips or fips_manual

Rules:
- Fill blanks only (never overwrite existing fips_code)
- Match on normalized (state, county label raw)
- Rebuild county-year summary from updated interim
"""

import os
from datetime import date

import pandas as pd
from functions import (
    latest_file_by_regex,
    normalize_fips5,
    normalize_text_upper,
    first_non_null,
    STATE_ABBR_TO_FIPS2,
)


db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
clean_dir = os.path.join(db_data, "clean")
qa_dir = os.path.join(db_data, "FOIA-USDA-request", "qa-fsis")
os.makedirs(qa_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")

_latest_file = latest_file_by_regex
_normalize_fips = normalize_fips5
_normalize_text = normalize_text_upper
_first_non_null = first_non_null


def _build_county_year(df: pd.DataFrame) -> pd.DataFrame:
    src = df[df["fips_code"].notna() & df["year"].notna()].copy()
    src["year"] = pd.to_numeric(src["year"], errors="coerce").astype("Int64")
    src = src.dropna(subset=["year"]).copy()
    src["year"] = src["year"].astype(int)
    src["est_size_combo_key"] = src["est_key"].astype("string") + "::" + src["size_bucket_final"].astype("string")

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

    return (
        src.groupby(["fips_code", "year"], as_index=False)
        .agg(**agg)
        .rename(columns={"fips_code": "fips"})
        .sort_values(["fips", "year"])
        .reset_index(drop=True)
    )


def main():
    src_interim = _latest_file(
        clean_dir,
        r"^(\d{4}-\d{2}-\d{2})_fsis_establishment_year_fips_size_type_interim_hudbulk_manualzip\.csv$",
    )
    src_manual = _latest_file(
        qa_dir,
        r"^(\d{4}-\d{2}-\d{2})_fsis_missing_county_labels_manual_template\.csv$",
    )

    print("Using interim source:", src_interim)
    print("Using manual county template:", src_manual)

    df = pd.read_csv(src_interim, dtype=str, low_memory=False)
    m = pd.read_csv(src_manual, dtype=str)

    if not {"state", "county_label_raw"}.issubset(m.columns):
        raise KeyError("Manual template must include state and county_label_raw")

    fips_col = None
    for c in ["fips", "fips_manual"]:
        if c in m.columns and m[c].astype("string").str.strip().replace({"": pd.NA}).notna().any():
            fips_col = c
            break
    if fips_col is None:
        raise KeyError("Manual template must include a non-empty fips column (fips or fips_manual)")

    orig_fips = _normalize_fips(df["fips_code"])
    out = df.copy()
    out["fips_code"] = orig_fips
    out["state_u"] = _normalize_text(out["state"] if "state" in out.columns else pd.Series([pd.NA] * len(out)))
    out["county_u"] = _normalize_text(out["county"] if "county" in out.columns else pd.Series([pd.NA] * len(out)))
    out["state_code"] = out["state_u"].map(STATE_ABBR_TO_FIPS2)

    mm = m[["state", "county_label_raw", fips_col]].copy()
    mm = mm.rename(columns={fips_col: "manual_fips"})
    mm["state_u"] = _normalize_text(mm["state"])
    mm["county_u"] = _normalize_text(mm["county_label_raw"])
    mm["manual_fips"] = _normalize_fips(mm["manual_fips"])
    mm = mm.dropna(subset=["state_u", "county_u", "manual_fips"]).copy()

    # keep one mapping per state+county label; fail if conflicting fips provided
    conf = mm.groupby(["state_u", "county_u"])["manual_fips"].nunique().reset_index(name="n")
    bad = conf[conf["n"] > 1]
    if not bad.empty:
        raise ValueError("Conflicting manual FIPS for same state+county_label_raw")

    mm = mm.drop_duplicates(["state_u", "county_u"], keep="first")

    out = out.merge(mm[["state_u", "county_u", "manual_fips"]], on=["state_u", "county_u"], how="left")

    # state prefix guard
    manual_ok = out["manual_fips"].notna() & out["state_code"].notna() & out["manual_fips"].str[:2].eq(out["state_code"])
    mask_fill = out["fips_code"].isna() & manual_ok

    out.loc[mask_fill, "fips_code"] = out.loc[mask_fill, "manual_fips"]
    out.loc[mask_fill, "fips_fill_method"] = "manual_county_label_template"
    out["manual_county_label_fips_filled"] = mask_fill.astype("Int64")

    changed_nonmissing = int(((orig_fips.notna()) & (orig_fips != out["fips_code"]) & out["fips_code"].notna()).sum())
    invalid_fips_rows = int((~(out["fips_code"].isna() | out["fips_code"].astype("string").str.fullmatch(r"\d{5}", na=False))).sum())

    county_after = _build_county_year(out)

    interim_out = os.path.join(clean_dir, f"{today_str}_fsis_establishment_year_fips_size_type_interim_hudbulk_manualzip.csv")
    county_out = os.path.join(clean_dir, f"{today_str}_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip.csv")
    qa_map_out = os.path.join(qa_dir, f"{today_str}_fsis_manual_county_label_mapping_used.csv")
    qa_metrics_out = os.path.join(qa_dir, f"{today_str}_fsis_manual_county_label_fill_metrics.csv")

    out.to_csv(interim_out, index=False)
    county_after.to_csv(county_out, index=False)
    mm.to_csv(qa_map_out, index=False)

    qa = pd.DataFrame(
        {
            "metric": [
                "n_rows_input",
                "n_rows_output",
                "n_missing_fips_before",
                "n_missing_fips_after",
                "n_rows_filled_manual_county_label",
                "n_unique_manual_county_labels_in_template",
                "n_unique_manual_county_labels_used",
                "n_changed_preexisting_nonmissing_fips",
                "n_invalid_fips_rows_after",
                "n_county_year_rows_after_rebuilt",
                "n_unique_county_fips_after_rebuilt",
            ],
            "value": [
                len(df),
                len(out),
                int(orig_fips.isna().sum()),
                int(out["fips_code"].isna().sum()),
                int(mask_fill.sum()),
                int(mm[["state_u", "county_u"]].drop_duplicates().shape[0]),
                int(out.loc[mask_fill, ["state_u", "county_u"]].drop_duplicates().shape[0]),
                changed_nonmissing,
                invalid_fips_rows,
                int(len(county_after)),
                int(county_after["fips"].astype("string").dropna().nunique()),
            ],
        }
    )
    qa.to_csv(qa_metrics_out, index=False)

    print("Saved:", interim_out)
    print("Saved:", county_out)
    print("Saved QA:", qa_map_out)
    print("Saved QA:", qa_metrics_out)
    print("Filled rows:", int(mask_fill.sum()))
    print("Missing FIPS before:", int(orig_fips.isna().sum()), "after:", int(out["fips_code"].isna().sum()))
    print("Changed pre-existing nonmissing FIPS rows:", changed_nonmissing)


if __name__ == "__main__":
    main()
