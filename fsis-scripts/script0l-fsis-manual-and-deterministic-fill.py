#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script0l-fsis-manual-and-deterministic-fill.py -- last FIPS-recovery steps for the FSIS establishment panel, in one pass.

Replaces three scripts that each read "the newest" copy of one file name and overwrote it (former script0l, 0m, 0n, run in that order).
All three steps now run in memory in the same order and the result is written once. Every step fills BLANK fips_code values only
and never overwrites an existing one.

  1. Manual ZIP fill      fips_code from the manual ZIP workbook (zip5 -> fips_manual).               [former script0l]
  2. Deterministic fill   ZIP fallback (HUD zip-year > manual > HUD stable > known stable), Connecticut county names,
                          unique city+state, unique establishment-ID history.                          [former script0m]
  3. Manual county fill   fips_code from the manual county-label template (state + county label).      [former script0n]

The order matters: step 2 draws on FIPS filled in step 1, and step 2's establishment-ID carry spreads step-1 fills across years.
Between steps the table is passed through a CSV text round trip (dtype=str), which is exactly what the former file hand-off did, so
the output is identical to the old three-script chain (checked cell by cell; see code-audit/plans D-48).

Reads:   clean/diagnostic/*_fsis_establishment_year_fips_size_type_interim_hudbulk.csv and *_summary_hudbulk.csv (script0k output),
         clean/*_fips_full.csv (script0a), qa-fsis/ manual ZIP workbook, HUD zip-year lookup, manual county-label template.
Writes:  clean/diagnostic/YYYY-MM-DD_fsis_establishment_year_fips_size_type_interim_hudbulk_manualzip.csv
         clean/YYYY-MM-DD_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip.csv   (the descriptor script1b reads)
         qa-fsis/ one QA file per step (manual ZIP mapping and diff metrics, round-2 metrics and audits, county-label metrics).
Run:     python3 fsis-scripts/script0l-fsis-manual-and-deterministic-fill.py
             [--skip-manual-zip] [--skip-manual-county] [--require-manual-zip] [--require-manual-county]
         (script0f-fsis-clean-all.py calls this with the flags it received; a missing template skips its step unless required.)
"""
import argparse
import io
import os
from datetime import date

import pandas as pd

from functions import (
    latest_file_by_regex,
    normalize_zip5,
    normalize_fips5,
    normalize_text_upper,
    first_non_null,
    STATE_ABBR_TO_FIPS2,
)
from packages import db_data, diagnostic_dir

clean_dir = os.path.join(db_data, "clean")
qa_dir = os.path.join(db_data, "FOIA-USDA-request", "qa-fsis")
os.makedirs(diagnostic_dir, exist_ok=True)
os.makedirs(qa_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")

DATED = r"^(\d{4}-\d{2}-\d{2})_"
INTERIM_HUDBULK = DATED + r"fsis_establishment_year_fips_size_type_interim_hudbulk\.csv$"
COUNTY_HUDBULK = DATED + r"fsis_county_year_fips_est_size_type_summary_hudbulk\.csv$"
FIPS_KEY = DATED + r"fips_full\.csv$"
MANUAL_ZIP = DATED + r"fsis_unmatched_unique_zip_for_manual_fips\.xlsx$"
HUD_LOOKUP = DATED + r"fsis_zip_year_fips_lookup_from_hud_bulk\.csv$"
MANUAL_COUNTY = DATED + r"fsis_missing_county_labels_manual_template\.csv$"


def newest(dirpath, pattern):
    """Newest dated file matching pattern, or None when there is none."""
    try:
        return latest_file_by_regex(dirpath, pattern)
    except (FileNotFoundError, OSError):
        return None


def as_read_back(df):
    """The table as the next script used to receive it: written to CSV text and read back with dtype=str."""
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), dtype=str, low_memory=False)


def normalize_county_name(series):
    """Upper-case county name without COUNTY / PARISH / BOROUGH-type words and punctuation (Connecticut matching)."""
    s = series.astype("string").str.upper()
    for token in ["COUNTY", "PARISH", "BOROUGH", "CENSUS AREA", "MUNICIPALITY", "CITY AND BOROUGH", "CITY", "PLANNING REGION"]:
        s = s.str.replace(token, " ", regex=False)
    s = s.str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.where(s.ne(""), pd.NA)


def column_or_na(df, name):
    return df[name] if name in df.columns else pd.Series([pd.NA] * len(df))


def build_county_year(df):
    """County-year establishment counts by type and size bucket, from rows that have a fips_code and a year."""
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
        "county_name_any": ("county", first_non_null),
        "state_any": ("state", first_non_null),
    }

    return (
        src.groupby(["fips_code", "year"], as_index=False)
        .agg(**agg)
        .rename(columns={"fips_code": "fips"})
        .sort_values(["fips", "year"])
        .reset_index(drop=True)
    )


def n_unique_fips(county):
    return int(county["fips"].astype("string").dropna().nunique())


def apply_fill(out, candidate_col, method_label, flag_col):
    """Blank-only fill of fips_code from candidate_col, guarded by the state prefix; returns the number of rows filled."""
    can = normalize_fips5(out[candidate_col])
    state_ok = pd.Series(True, index=out.index)
    has_state = out["state_code"].notna() & can.notna()
    state_ok.loc[has_state] = can.loc[has_state].str[:2].eq(out.loc[has_state, "state_code"])

    mask = out["fips_code"].isna() & can.notna() & state_ok
    out.loc[mask, "fips_code"] = can.loc[mask]
    out.loc[mask, "fips_fill_method"] = method_label
    out[flag_col] = mask.astype("Int64")
    return int(mask.sum())


# ---------------------------------------------------------------------------------------------------------------------
# Step 1: manual ZIP fill (former script0l)
# ---------------------------------------------------------------------------------------------------------------------
def step_manual_zip(df, manual_path, county_source_path):
    manual = pd.read_excel(manual_path, dtype=str)
    if "zip5" not in manual.columns or "fips_manual" not in manual.columns:
        raise KeyError("Manual file must include columns: zip5, fips_manual")

    df = df.copy()
    df["zip5"] = normalize_zip5(df["zip5"] if "zip5" in df.columns else df.get("zip", pd.Series([pd.NA] * len(df))))
    df["fips_code"] = normalize_fips5(df["fips_code"])

    manual = manual[["zip5", "fips_manual"]].copy()
    manual["zip5"] = normalize_zip5(manual["zip5"])
    manual["fips_manual"] = normalize_fips5(manual["fips_manual"])
    manual = manual.dropna(subset=["zip5", "fips_manual"]).drop_duplicates(["zip5"], keep="first")

    out = df.merge(manual.rename(columns={"fips_manual": "fips_manual_from_zip"}), on="zip5", how="left")

    mask_fill = out["fips_code"].isna() & out["fips_manual_from_zip"].notna()
    out.loc[mask_fill, "fips_code"] = out.loc[mask_fill, "fips_manual_from_zip"]
    out.loc[mask_fill, "fips_fill_method"] = out.loc[mask_fill, "fips_fill_method"].where(
        out.loc[mask_fill, "fips_fill_method"].notna(), "manual_zip_template"
    )
    out["manual_zip_fips_filled"] = mask_fill.astype("Int64")

    county_before = (
        pd.read_csv(county_source_path, dtype=str, low_memory=False) if county_source_path is not None else None
    )
    county_before_rebuilt = build_county_year(df)
    county_after = build_county_year(out)

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
                int(df["fips_code"].isna().sum()),
                int(out["fips_code"].isna().sum()),
                int(mask_fill.sum()),
                int(manual["zip5"].nunique()),
                int(out.loc[mask_fill, "zip5"].nunique()),
                int(len(county_before)) if county_before is not None else pd.NA,
                int(len(county_before_rebuilt)),
                int(len(county_after)),
                int(len(county_after) - len(county_before_rebuilt)),
                n_unique_fips(county_before) if county_before is not None and "fips" in county_before.columns else pd.NA,
                n_unique_fips(county_before_rebuilt),
                n_unique_fips(county_after),
                n_unique_fips(county_after) - n_unique_fips(county_before_rebuilt),
            ],
        }
    )
    qa_files = {
        f"{today_str}_fsis_manual_zip_fips_mapping_used.csv": manual,
        f"{today_str}_fsis_manual_zip_fill_diff_metrics.csv": qa,
    }
    print("Step 1 manual ZIP rows filled:", int(mask_fill.sum()), "| missing FIPS before:", int(df["fips_code"].isna().sum()),
          "after:", int(out["fips_code"].isna().sum()))
    return out, qa_files


# ---------------------------------------------------------------------------------------------------------------------
# Step 2: deterministic fill (former script0m)
# ---------------------------------------------------------------------------------------------------------------------
def step_deterministic(base, fips_key_path, manual_zip_path, hud_lookup_path):
    base = base.copy()
    base["fips_code"] = normalize_fips5(base["fips_code"])
    base["zip5"] = normalize_zip5(base["zip5"] if "zip5" in base.columns else base.get("zip", pd.Series([pd.NA] * len(base))))
    base["state_u"] = normalize_text_upper(column_or_na(base, "state"))
    base["city_u"] = normalize_text_upper(column_or_na(base, "city"))
    base["county_u"] = normalize_text_upper(column_or_na(base, "county"))
    base["county_norm"] = normalize_county_name(column_or_na(base, "county"))
    base["state_code"] = base["state_u"].map(STATE_ABBR_TO_FIPS2)

    orig_fips = base["fips_code"].copy()

    out = base.copy()
    # Idempotence: remove helper columns from prior runs before remapping.
    helper_cols = [
        "zip_candidate_hud_year",
        "zip_candidate_manual",
        "zip_candidate_hud_stable",
        "zip_candidate_known_stable",
        "zip_candidate_fips",
        "ct_candidate_fips",
        "ct_mapping_source",
        "city_state_candidate_fips",
        "id_history_candidate_fips",
        "filled_zip_fallback",
        "filled_ct_mapping",
        "filled_city_state_unique",
        "filled_id_history_unique",
        "city_state_ambiguous",
        "residual_reason",
        "county_key",
        "year_key",
        "county_year_key",
        "est_year_key",
    ]
    drop_cols = [c for c in helper_cols if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    # ------------------------------------------------------------------
    # 2a) ZIP fallback pass (deterministic sources only)
    # ------------------------------------------------------------------
    out["zip_candidate_hud_year"] = pd.NA
    out["zip_candidate_manual"] = pd.NA
    out["zip_candidate_hud_stable"] = pd.NA
    out["zip_candidate_known_stable"] = pd.NA

    if hud_lookup_path is not None:
        hud = pd.read_csv(hud_lookup_path, dtype=str)
        # exact zip-year from HUD lookup
        h1 = hud[hud["fips_code_from_hud"].notna()][["year", "zip5", "fips_code_from_hud"]].drop_duplicates()
        out = out.merge(
            h1.rename(columns={"fips_code_from_hud": "zip_candidate_hud_year"}),
            on=["year", "zip5"],
            how="left",
            suffixes=("", "_dup1"),
        )
        if "zip_candidate_hud_year_dup1" in out.columns:
            out["zip_candidate_hud_year"] = out["zip_candidate_hud_year"].where(
                out["zip_candidate_hud_year"].notna(), out["zip_candidate_hud_year_dup1"]
            )
            out = out.drop(columns=["zip_candidate_hud_year_dup1"])

        # stable zip-only from HUD lookup where zip -> one fips across years
        hs = hud[hud["fips_code_from_hud"].notna()][["zip5", "fips_code_from_hud"]].drop_duplicates()
        hs_n = hs.groupby("zip5")["fips_code_from_hud"].nunique().reset_index(name="n")
        hs_ok = hs_n[hs_n["n"] == 1][["zip5"]]
        hs_map = hs.merge(hs_ok, on="zip5", how="inner").drop_duplicates("zip5")
        out = out.merge(
            hs_map.rename(columns={"fips_code_from_hud": "zip_candidate_hud_stable"}),
            on=["zip5"],
            how="left",
            suffixes=("", "_dup2"),
        )
        if "zip_candidate_hud_stable_dup2" in out.columns:
            out["zip_candidate_hud_stable"] = out["zip_candidate_hud_stable"].where(
                out["zip_candidate_hud_stable"].notna(), out["zip_candidate_hud_stable_dup2"]
            )
            out = out.drop(columns=["zip_candidate_hud_stable_dup2"])

    if manual_zip_path is not None:
        man = pd.read_excel(manual_zip_path, dtype=str)
        if {"zip5", "fips_manual"}.issubset(man.columns):
            man = man[["zip5", "fips_manual"]].copy()
            man["zip5"] = normalize_zip5(man["zip5"])
            man["fips_manual"] = normalize_fips5(man["fips_manual"])
            man = man.dropna(subset=["zip5", "fips_manual"]).drop_duplicates(["zip5"], keep="first")
            out = out.merge(
                man.rename(columns={"fips_manual": "zip_candidate_manual"}),
                on=["zip5"],
                how="left",
                suffixes=("", "_dup3"),
            )
            if "zip_candidate_manual_dup3" in out.columns:
                out["zip_candidate_manual"] = out["zip_candidate_manual"].where(
                    out["zip_candidate_manual"].notna(), out["zip_candidate_manual_dup3"]
                )
                out = out.drop(columns=["zip_candidate_manual_dup3"])

    # stable zip from known valid rows in current interim
    known = out[
        out["fips_code"].notna()
        & out["state_code"].notna()
        & out["fips_code"].str[:2].eq(out["state_code"])
        & out["zip5"].notna()
    ][["zip5", "fips_code"]].drop_duplicates()
    kz_n = known.groupby("zip5")["fips_code"].nunique().reset_index(name="n")
    kz_ok = kz_n[kz_n["n"] == 1][["zip5"]]
    kz_map = known.merge(kz_ok, on="zip5", how="inner").drop_duplicates("zip5")
    out = out.merge(
        kz_map.rename(columns={"fips_code": "zip_candidate_known_stable"}),
        on=["zip5"],
        how="left",
        suffixes=("", "_dup4"),
    )
    if "zip_candidate_known_stable_dup4" in out.columns:
        out["zip_candidate_known_stable"] = out["zip_candidate_known_stable"].where(
            out["zip_candidate_known_stable"].notna(), out["zip_candidate_known_stable_dup4"]
        )
        out = out.drop(columns=["zip_candidate_known_stable_dup4"])

    # priority: HUD zip-year, then manual, then HUD stable, then known stable
    out["zip_candidate_fips"] = out["zip_candidate_hud_year"]
    out["zip_candidate_fips"] = out["zip_candidate_fips"].where(out["zip_candidate_fips"].notna(), out["zip_candidate_manual"])
    out["zip_candidate_fips"] = out["zip_candidate_fips"].where(out["zip_candidate_fips"].notna(), out["zip_candidate_hud_stable"])
    out["zip_candidate_fips"] = out["zip_candidate_fips"].where(out["zip_candidate_fips"].notna(), out["zip_candidate_known_stable"])

    n_zip = apply_fill(out, "zip_candidate_fips", "zip_fallback_deterministic", "filled_zip_fallback")

    # ------------------------------------------------------------------
    # 2b) CT normalization pass (auditable mapping layer)
    # ------------------------------------------------------------------
    fips_key = pd.read_csv(fips_key_path, dtype=str)
    ct = fips_key[fips_key["state_code"].astype("string").str.zfill(2) == "09"][["county", "fips"]].copy()
    ct["county_norm"] = normalize_county_name(ct["county"])
    ct["fips5"] = normalize_fips5(ct["fips"])
    ct = ct.dropna(subset=["county_norm", "fips5"]).drop_duplicates(["county_norm", "fips5"])

    # keep only deterministic county_norm -> one fips mapping
    ct_n = ct.groupby("county_norm")["fips5"].nunique().reset_index(name="n")
    ct_ok = ct_n[ct_n["n"] == 1][["county_norm"]]
    ct_map = ct.merge(ct_ok, on="county_norm", how="inner").drop_duplicates(["county_norm"])
    ct_map["mapping_source"] = "ct_legacy_county_name"

    out = out.merge(
        ct_map[["county_norm", "fips5", "mapping_source"]].rename(
            columns={"fips5": "ct_candidate_fips", "mapping_source": "ct_mapping_source"}
        ),
        on=["county_norm"],
        how="left",
    )

    # CT rows only.
    ct_mask = out["state_u"].eq("CT")
    out.loc[~ct_mask, "ct_candidate_fips"] = pd.NA
    n_ct = apply_fill(out, "ct_candidate_fips", "ct_mapping_deterministic", "filled_ct_mapping")

    # ------------------------------------------------------------------
    # 2c) City+State deterministic carry (unique mapping only)
    # ------------------------------------------------------------------
    known_cs = out[
        out["fips_code"].notna()
        & out["state_u"].notna()
        & out["city_u"].notna()
        & out["state_code"].notna()
        & out["fips_code"].str[:2].eq(out["state_code"])
    ][["state_u", "city_u", "fips_code"]].drop_duplicates()

    cs_n = known_cs.groupby(["state_u", "city_u"])["fips_code"].nunique().reset_index(name="n_fips")
    cs_unique = cs_n[cs_n["n_fips"] == 1][["state_u", "city_u"]]
    cs_map = known_cs.merge(cs_unique, on=["state_u", "city_u"], how="inner").drop_duplicates(["state_u", "city_u"])

    out = out.merge(
        cs_map.rename(columns={"fips_code": "city_state_candidate_fips"}),
        on=["state_u", "city_u"],
        how="left",
    )

    n_city = apply_fill(out, "city_state_candidate_fips", "city_state_unique", "filled_city_state_unique")

    # ------------------------------------------------------------------
    # 2d) ID-history deterministic carry (unique FIPS per ID only)
    # ------------------------------------------------------------------
    known_id = out[out["fips_code"].notna() & out["establishment_id"].notna()][["establishment_id", "fips_code"]].drop_duplicates()

    id_n = known_id.groupby("establishment_id")["fips_code"].nunique().reset_index(name="n_fips")
    id_unique = id_n[id_n["n_fips"] == 1][["establishment_id"]]
    id_map = known_id.merge(id_unique, on="establishment_id", how="inner").drop_duplicates(["establishment_id"])

    out = out.merge(
        id_map.rename(columns={"fips_code": "id_history_candidate_fips"}),
        on=["establishment_id"],
        how="left",
    )

    n_id = apply_fill(out, "id_history_candidate_fips", "id_history_unique", "filled_id_history_unique")

    # ------------------------------------------------------------------
    # QA checks and residual reasons
    # ------------------------------------------------------------------
    changed_nonmissing = int(((orig_fips.notna()) & (orig_fips != out["fips_code"]) & out["fips_code"].notna()).sum())
    valid_fips_mask = out["fips_code"].isna() | out["fips_code"].astype("string").str.fullmatch(r"\d{5}", na=False)
    invalid_fips_rows = int((~valid_fips_mask).sum())

    unresolved = out[out["fips_code"].isna()].copy()

    # City-state ambiguity based on known map ambiguity table.
    cs_ambig = cs_n[cs_n["n_fips"] > 1][["state_u", "city_u"]].copy()
    unresolved = unresolved.merge(cs_ambig.assign(city_state_ambiguous=1), on=["state_u", "city_u"], how="left")
    unresolved["city_state_ambiguous"] = unresolved["city_state_ambiguous"].fillna(0).astype("Int64")

    reason = pd.Series("no_geo_signal", index=unresolved.index, dtype="string")
    reason = reason.mask(unresolved["zip5"].isna() & unresolved["county_u"].isna(), "no_zip_no_county")
    reason = reason.mask(
        unresolved["state_u"].eq("CT")
        & unresolved["county_u"].notna()
        & unresolved["county_u"].str.contains("CONNECTICUT|PLANNING REGION", na=False),
        "ct_unmapped_region",
    )
    reason = reason.mask(unresolved["city_state_ambiguous"].eq(1), "city_state_ambiguous")
    unresolved["residual_reason"] = reason

    unresolved["residual_reason"] = unresolved["residual_reason"].fillna("no_geo_signal")
    unresolved["county_key"] = unresolved["county"].astype("string").fillna("NA")
    unresolved["year_key"] = unresolved["year"].astype("string").fillna("NA")
    unresolved["county_year_key"] = unresolved["county_key"] + "::" + unresolved["year_key"]
    unresolved["est_year_key"] = (
        unresolved["establishment_id"].astype("string").fillna("NA") + "::" + unresolved["year_key"]
    )

    residual_counts = (
        unresolved.groupby("residual_reason", as_index=False)
        .agg(
            n_rows=("residual_reason", "size"),
            n_unique_county_labels=("county_key", "nunique"),
            n_unique_years=("year_key", "nunique"),
            n_unique_county_year=("county_year_key", "nunique"),
            n_unique_establishment_year=("est_year_key", "nunique"),
        )
        .sort_values(["n_rows", "residual_reason"], ascending=[False, True])
        .reset_index(drop=True)
    )

    # County-year rebuild + impossible type totals check.
    county_before = build_county_year(base)
    county_after = build_county_year(out)

    impossible = county_after.copy()
    for c in [
        "n_unique_establishments",
        "n_type_both_slaughter_and_processing",
        "n_type_slaughter_only",
        "n_type_processing_only",
        "n_type_other_or_unclear",
        "n_type_neither_signal",
    ]:
        impossible[c] = pd.to_numeric(impossible[c], errors="coerce").fillna(0)
    impossible_rows = impossible[
        (impossible["n_type_both_slaughter_and_processing"] > impossible["n_unique_establishments"])
        | (impossible["n_type_slaughter_only"] > impossible["n_unique_establishments"])
        | (impossible["n_type_processing_only"] > impossible["n_unique_establishments"])
        | (impossible["n_type_other_or_unclear"] > impossible["n_unique_establishments"])
        | (impossible["n_type_neither_signal"] > impossible["n_unique_establishments"])
    ]

    qa_metrics = pd.DataFrame(
        {
            "metric": [
                "n_rows_input",
                "n_rows_output",
                "n_missing_fips_before",
                "n_missing_fips_after",
                "n_filled_total_round2",
                "n_filled_zip_fallback",
                "n_filled_ct_mapping",
                "n_filled_city_state_unique",
                "n_filled_id_history_unique",
                "n_changed_preexisting_nonmissing_fips",
                "n_invalid_fips_rows_after",
                "n_county_year_rows_before_rebuilt",
                "n_county_year_rows_after_rebuilt",
                "delta_county_year_rows_rebuilt",
                "n_unique_county_fips_before_rebuilt",
                "n_unique_county_fips_after_rebuilt",
                "delta_unique_county_fips_rebuilt",
                "n_impossible_type_total_rows_after",
            ],
            "value": [
                len(base),
                len(out),
                int(base["fips_code"].isna().sum()),
                int(out["fips_code"].isna().sum()),
                int(base["fips_code"].isna().sum() - out["fips_code"].isna().sum()),
                n_zip,
                n_ct,
                n_city,
                n_id,
                changed_nonmissing,
                invalid_fips_rows,
                len(county_before),
                len(county_after),
                len(county_after) - len(county_before),
                n_unique_fips(county_before),
                n_unique_fips(county_after),
                n_unique_fips(county_after) - n_unique_fips(county_before),
                len(impossible_rows),
            ],
        }
    )

    fill_method_counts = pd.DataFrame(
        {
            "method": ["zip_fallback_deterministic", "ct_mapping_deterministic", "city_state_unique", "id_history_unique"],
            "n_rows_filled": [n_zip, n_ct, n_city, n_id],
        }
    )

    zip_audit_cols = [
        "year", "zip5", "state", "city", "establishment_id", "establishment_number", "fips_code",
        "zip_candidate_hud_year", "zip_candidate_manual", "zip_candidate_hud_stable", "zip_candidate_known_stable",
        "zip_candidate_fips", "filled_zip_fallback",
    ]
    zip_audit = out[out["zip5"].notna()][[c for c in zip_audit_cols if c in out.columns]].copy()

    ct_audit = out[out["state_u"].eq("CT")][
        [
            c for c in [
                "year", "state", "county", "county_norm", "city", "establishment_id", "establishment_number",
                "fips_code", "ct_candidate_fips", "ct_mapping_source", "filled_ct_mapping",
            ] if c in out.columns
        ]
    ].copy()

    unresolved_rows = unresolved[
        [
            c for c in [
                "year", "state", "county", "city", "zip", "zip5", "establishment_id", "establishment_number",
                "residual_reason", "city_state_ambiguous",
            ] if c in unresolved.columns
        ]
    ]

    qa_files = {
        f"{today_str}_fsis_round2_completion_metrics.csv": qa_metrics,
        f"{today_str}_fsis_round2_fill_method_counts.csv": fill_method_counts,
        f"{today_str}_fsis_round2_zip_fallback_audit.csv": zip_audit,
        f"{today_str}_fsis_round2_ct_mapping_layer.csv": ct_map,
        f"{today_str}_fsis_round2_ct_mapping_audit.csv": ct_audit,
        f"{today_str}_fsis_round2_unresolved_rows.csv": unresolved_rows,
        f"{today_str}_fsis_round2_residual_reason_counts.csv": residual_counts,
    }
    print("Step 2 missing FIPS before:", int(base["fips_code"].isna().sum()), "after:", int(out["fips_code"].isna().sum()))
    print("Step 2 filled by method -> zip:", n_zip, "ct:", n_ct, "city_state:", n_city, "id_history:", n_id)
    print("Step 2 changed pre-existing nonmissing FIPS rows:", changed_nonmissing)
    print("Step 2 impossible type-total rows after:", len(impossible_rows))
    return out, qa_files


# ---------------------------------------------------------------------------------------------------------------------
# Step 3: manual county-label fill (former script0n)
# ---------------------------------------------------------------------------------------------------------------------
def step_manual_county(df, template_path):
    m = pd.read_csv(template_path, dtype=str)

    if not {"state", "county_label_raw"}.issubset(m.columns):
        raise KeyError("Manual template must include state and county_label_raw")

    fips_col = None
    for c in ["fips", "fips_manual"]:
        if c in m.columns and m[c].astype("string").str.strip().replace({"": pd.NA}).notna().any():
            fips_col = c
            break
    if fips_col is None:
        raise KeyError("Manual template must include a non-empty fips column (fips or fips_manual)")

    orig_fips = normalize_fips5(df["fips_code"])
    out = df.copy()
    out["fips_code"] = orig_fips
    out["state_u"] = normalize_text_upper(column_or_na(out, "state"))
    out["county_u"] = normalize_text_upper(column_or_na(out, "county"))
    out["state_code"] = out["state_u"].map(STATE_ABBR_TO_FIPS2)

    mm = m[["state", "county_label_raw", fips_col]].copy()
    mm = mm.rename(columns={fips_col: "manual_fips"})
    mm["state_u"] = normalize_text_upper(mm["state"])
    mm["county_u"] = normalize_text_upper(mm["county_label_raw"])
    mm["manual_fips"] = normalize_fips5(mm["manual_fips"])
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

    county_after = build_county_year(out)

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
                n_unique_fips(county_after),
            ],
        }
    )
    qa_files = {
        f"{today_str}_fsis_manual_county_label_mapping_used.csv": mm,
        f"{today_str}_fsis_manual_county_label_fill_metrics.csv": qa,
    }
    print("Step 3 manual county-label rows filled:", int(mask_fill.sum()), "| missing FIPS before:", int(orig_fips.isna().sum()),
          "after:", int(out["fips_code"].isna().sum()))
    print("Step 3 changed pre-existing nonmissing FIPS rows:", changed_nonmissing)
    return out, qa_files


def main():
    parser = argparse.ArgumentParser(description="FSIS manual ZIP fill, deterministic fill and manual county-label fill, in one pass.")
    parser.add_argument("--skip-manual-zip", action="store_true", help="Skip step 1 (manual ZIP fill); step 2 still uses the workbook if found.")
    parser.add_argument("--skip-manual-county", action="store_true", help="Skip step 3 (manual county-label fill).")
    parser.add_argument("--require-manual-zip", action="store_true", help="Fail if the manual ZIP workbook is not found.")
    parser.add_argument("--require-manual-county", action="store_true", help="Fail if the manual county-label template is not found.")
    args = parser.parse_args()

    src_interim = newest(diagnostic_dir, INTERIM_HUDBULK)
    src_county = newest(diagnostic_dir, COUNTY_HUDBULK)
    src_fips_key = newest(clean_dir, FIPS_KEY)
    src_manual_zip = newest(qa_dir, MANUAL_ZIP)
    src_hud_lookup = newest(qa_dir, HUD_LOOKUP)
    src_manual_county = newest(qa_dir, MANUAL_COUNTY)
    for label, path in (("interim hudbulk file (script0k output)", src_interim), ("fips_full key (script0a output)", src_fips_key)):
        if path is None:
            raise FileNotFoundError(f"Required input not found: {label}")
    if args.require_manual_zip and src_manual_zip is None:
        raise FileNotFoundError("Manual ZIP workbook not found in qa-fsis folder.")
    if args.require_manual_county and src_manual_county is None:
        raise FileNotFoundError("Manual county-label template not found in qa-fsis folder.")

    print("Interim source:", src_interim)
    print("County source (QA only):", src_county)
    print("FIPS key:", src_fips_key)
    print("Manual ZIP workbook:", src_manual_zip)
    print("HUD zip-year lookup:", src_hud_lookup)
    print("Manual county-label template:", src_manual_county)

    qa_files = {}
    out = pd.read_csv(src_interim, dtype=str, low_memory=False)
    n_input = len(out)

    if not args.skip_manual_zip and src_manual_zip is not None:
        out, qa = step_manual_zip(out, src_manual_zip, src_county)
        qa_files.update(qa)
        out = as_read_back(out)
    else:
        print("Step 1 skipped (manual ZIP fill)")

    out, qa = step_deterministic(out, src_fips_key, src_manual_zip, src_hud_lookup)
    qa_files.update(qa)

    if not args.skip_manual_county and src_manual_county is not None:
        out = as_read_back(out)
        out, qa = step_manual_county(out, src_manual_county)
        qa_files.update(qa)
    else:
        print("Step 3 skipped (manual county-label fill)")

    county_after = build_county_year(out)
    interim_out = os.path.join(diagnostic_dir, f"{today_str}_fsis_establishment_year_fips_size_type_interim_hudbulk_manualzip.csv")
    county_out = os.path.join(clean_dir, f"{today_str}_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip.csv")
    out.to_csv(interim_out, index=False)
    county_after.to_csv(county_out, index=False)
    for name, table in qa_files.items():
        table.to_csv(os.path.join(qa_dir, name), index=False)

    assert len(out) == n_input, "row count changed between the input and the output"
    print("Saved:", interim_out)
    print("Saved:", county_out)
    print(f"Saved {len(qa_files)} QA files to {qa_dir}")
    print("Missing FIPS in the final table:", int(out["fips_code"].isna().sum()), "of", len(out), "rows")


if __name__ == "__main__":
    main()
