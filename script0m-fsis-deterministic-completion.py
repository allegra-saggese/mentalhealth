#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Round-next deterministic FSIS FIPS completion.

Sequence (blank-only fills):
1) ZIP fallback
2) CT normalization pass with auditable CT mapping layer
3) City+State deterministic carry (unique only)
4) Establishment-ID history deterministic carry (unique only)

Then rebuild county-year summary and write QA artifacts.
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

STATE_ABBR_TO_FIPS2 = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56", "PR": "72", "VI": "78", "GU": "66",
    "MP": "69", "AS": "60",
}


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


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().str.strip()


def _normalize_county_name(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.upper()
    for token in [
        "COUNTY",
        "PARISH",
        "BOROUGH",
        "CENSUS AREA",
        "MUNICIPALITY",
        "CITY AND BOROUGH",
        "CITY",
        "PLANNING REGION",
    ]:
        s = s.str.replace(token, " ", regex=False)
    s = s.str.replace(r"[^A-Z0-9 ]+", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.where(s.ne(""), pd.NA)


def _first_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else pd.NA


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

    county = (
        src.groupby(["fips_code", "year"], as_index=False)
        .agg(**agg)
        .rename(columns={"fips_code": "fips"})
        .sort_values(["fips", "year"])
        .reset_index(drop=True)
    )
    return county


def _apply_fill(out: pd.DataFrame, candidate_col: str, method_label: str, flag_col: str):
    can = _normalize_fips(out[candidate_col])
    state_ok = pd.Series(True, index=out.index)
    has_state = out["state_code"].notna() & can.notna()
    state_ok.loc[has_state] = can.loc[has_state].str[:2].eq(out.loc[has_state, "state_code"])

    mask = out["fips_code"].isna() & can.notna() & state_ok
    out.loc[mask, "fips_code"] = can.loc[mask]
    out.loc[mask, "fips_fill_method"] = method_label
    out[flag_col] = mask.astype("Int64")
    return int(mask.sum())


def main():
    src_interim = _latest_file(
        clean_dir,
        r"^(\d{4}-\d{2}-\d{2})_fsis_establishment_year_fips_size_type_interim_hudbulk_manualzip\.csv$",
    )
    src_fips_key = _latest_file(
        clean_dir,
        r"^(\d{4}-\d{2}-\d{2})_fips_full\.csv$",
    )

    # Optional sources used in ZIP fallback.
    try:
        src_manual_zip = _latest_file(
            qa_dir,
            r"^(\d{4}-\d{2}-\d{2})_fsis_unmatched_unique_zip_for_manual_fips\.xlsx$",
        )
    except Exception:
        src_manual_zip = None

    try:
        src_hud_lookup = _latest_file(
            qa_dir,
            r"^(\d{4}-\d{2}-\d{2})_fsis_zip_year_fips_lookup_from_hud_bulk\.csv$",
        )
    except Exception:
        src_hud_lookup = None

    print("Using interim source:", src_interim)
    print("Using FIPS key:", src_fips_key)
    print("Using manual ZIP source:", src_manual_zip)
    print("Using HUD lookup source:", src_hud_lookup)

    base = pd.read_csv(src_interim, dtype=str, low_memory=False)
    base["fips_code"] = _normalize_fips(base["fips_code"])
    base["zip5"] = _normalize_zip(base["zip5"] if "zip5" in base.columns else base.get("zip", pd.Series([pd.NA] * len(base))))
    base["state_u"] = _normalize_text(base["state"] if "state" in base.columns else pd.Series([pd.NA] * len(base)))
    base["city_u"] = _normalize_text(base["city"] if "city" in base.columns else pd.Series([pd.NA] * len(base)))
    base["county_u"] = _normalize_text(base["county"] if "county" in base.columns else pd.Series([pd.NA] * len(base)))
    base["county_norm"] = _normalize_county_name(base["county"] if "county" in base.columns else pd.Series([pd.NA] * len(base)))
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
    # 1) ZIP fallback pass (deterministic sources only)
    # ------------------------------------------------------------------
    out["zip_candidate_hud_year"] = pd.NA
    out["zip_candidate_manual"] = pd.NA
    out["zip_candidate_hud_stable"] = pd.NA
    out["zip_candidate_known_stable"] = pd.NA

    if src_hud_lookup is not None:
        hud = pd.read_csv(src_hud_lookup, dtype=str)
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

    if src_manual_zip is not None:
        man = pd.read_excel(src_manual_zip, dtype=str)
        if {"zip5", "fips_manual"}.issubset(man.columns):
            man = man[["zip5", "fips_manual"]].copy()
            man["zip5"] = _normalize_zip(man["zip5"])
            man["fips_manual"] = _normalize_fips(man["fips_manual"])
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

    out["zip_candidate_fips"] = out["zip_candidate_hud_year"]
    out["zip_candidate_fips"] = out["zip_candidate_fips"].where(
        out["zip_candidate_fips"].notna(), out["zip_candidate_manual"]
    )
    out["zip_candidate_fips"] = out["zip_candidate_fips"].where(
        out["zip_candidate_fips"].notna(), out["zip_candidate_hud_stable"]
    )
    out["zip_candidate_fips"] = out["zip_candidate_fips"].where(
        out["zip_candidate_fips"].notna(), out["zip_candidate_known_stable"]
    )

    n_zip = _apply_fill(out, "zip_candidate_fips", "zip_fallback_deterministic", "filled_zip_fallback")

    # ------------------------------------------------------------------
    # 2) CT normalization pass (auditable mapping layer)
    # ------------------------------------------------------------------
    fips_key = pd.read_csv(src_fips_key, dtype=str)
    ct = fips_key[fips_key["state_code"].astype("string").str.zfill(2) == "09"][["county", "fips"]].copy()
    ct["county_norm"] = _normalize_county_name(ct["county"])
    ct["fips5"] = _normalize_fips(ct["fips"])
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
    n_ct = _apply_fill(out, "ct_candidate_fips", "ct_mapping_deterministic", "filled_ct_mapping")

    # ------------------------------------------------------------------
    # 3) City+State deterministic carry (unique mapping only)
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

    n_city = _apply_fill(out, "city_state_candidate_fips", "city_state_unique", "filled_city_state_unique")

    # ------------------------------------------------------------------
    # 4) ID-history deterministic carry (unique FIPS per ID only)
    # ------------------------------------------------------------------
    known_id = out[
        out["fips_code"].notna()
        & out["establishment_id"].notna()
    ][["establishment_id", "fips_code"]].drop_duplicates()

    id_n = known_id.groupby("establishment_id")["fips_code"].nunique().reset_index(name="n_fips")
    id_unique = id_n[id_n["n_fips"] == 1][["establishment_id"]]
    id_map = known_id.merge(id_unique, on="establishment_id", how="inner").drop_duplicates(["establishment_id"])

    out = out.merge(
        id_map.rename(columns={"fips_code": "id_history_candidate_fips"}),
        on=["establishment_id"],
        how="left",
    )

    n_id = _apply_fill(out, "id_history_candidate_fips", "id_history_unique", "filled_id_history_unique")

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
        unresolved["establishment_id"].astype("string").fillna("NA")
        + "::"
        + unresolved["year_key"]
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
    county_before = _build_county_year(base)
    county_after = _build_county_year(out)

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

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    interim_out = os.path.join(clean_dir, f"{today_str}_fsis_establishment_year_fips_size_type_interim_hudbulk_manualzip.csv")
    county_out = os.path.join(clean_dir, f"{today_str}_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip.csv")

    qa_metrics_out = os.path.join(qa_dir, f"{today_str}_fsis_round2_completion_metrics.csv")
    qa_method_out = os.path.join(qa_dir, f"{today_str}_fsis_round2_fill_method_counts.csv")
    qa_zip_out = os.path.join(qa_dir, f"{today_str}_fsis_round2_zip_fallback_audit.csv")
    qa_ct_map_out = os.path.join(qa_dir, f"{today_str}_fsis_round2_ct_mapping_layer.csv")
    qa_ct_audit_out = os.path.join(qa_dir, f"{today_str}_fsis_round2_ct_mapping_audit.csv")
    qa_unresolved_out = os.path.join(qa_dir, f"{today_str}_fsis_round2_unresolved_rows.csv")
    qa_resid_out = os.path.join(qa_dir, f"{today_str}_fsis_round2_residual_reason_counts.csv")

    out.to_csv(interim_out, index=False)
    county_after.to_csv(county_out, index=False)

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
                int(county_before["fips"].astype("string").dropna().nunique()),
                int(county_after["fips"].astype("string").dropna().nunique()),
                int(county_after["fips"].astype("string").dropna().nunique() - county_before["fips"].astype("string").dropna().nunique()),
                len(impossible_rows),
            ],
        }
    )

    fill_method_counts = pd.DataFrame(
        {
            "method": [
                "zip_fallback_deterministic",
                "ct_mapping_deterministic",
                "city_state_unique",
                "id_history_unique",
            ],
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
                "fips_code", "ct_candidate_fips", "ct_mapping_source", "filled_ct_mapping"
            ] if c in out.columns
        ]
    ].copy()

    unresolved[[
        c for c in [
            "year", "state", "county", "city", "zip", "zip5", "establishment_id", "establishment_number",
            "residual_reason", "city_state_ambiguous"
        ] if c in unresolved.columns
    ]].to_csv(qa_unresolved_out, index=False)

    qa_metrics.to_csv(qa_metrics_out, index=False)
    fill_method_counts.to_csv(qa_method_out, index=False)
    zip_audit.to_csv(qa_zip_out, index=False)
    ct_map.to_csv(qa_ct_map_out, index=False)
    ct_audit.to_csv(qa_ct_audit_out, index=False)
    residual_counts.to_csv(qa_resid_out, index=False)

    print("Saved:", interim_out)
    print("Saved:", county_out)
    print("Saved QA:", qa_metrics_out)
    print("Saved QA:", qa_method_out)
    print("Saved QA:", qa_zip_out)
    print("Saved QA:", qa_ct_map_out)
    print("Saved QA:", qa_ct_audit_out)
    print("Saved QA:", qa_unresolved_out)
    print("Saved QA:", qa_resid_out)
    print("Missing FIPS before:", int(base["fips_code"].isna().sum()), "after:", int(out["fips_code"].isna().sum()))
    print("Filled by method -> zip:", n_zip, "ct:", n_ct, "city_state:", n_city, "id_history:", n_id)
    print("Changed pre-existing nonmissing FIPS rows:", changed_nonmissing)
    print("Impossible type-total rows after:", len(impossible_rows))


if __name__ == "__main__":
    main()
