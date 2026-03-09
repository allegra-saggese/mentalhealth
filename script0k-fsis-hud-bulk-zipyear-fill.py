#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bulk HUD USPS ZIP->County FIPS fill for FSIS missing FIPS rows.

Approach:
- Pull HUD crosswalk in bulk by year/quarter using query=All (type=2).
- Build a year+zip lookup (with state-aware tie-break when possible).
- Merge lookup back into interim data for rows still missing FIPS.
"""

import os
import random
import re
import time
from datetime import date
from typing import Dict, Optional, Tuple

import pandas as pd
import requests


db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
clean_dir = os.path.join(db_data, "clean")
qa_dir = os.path.join(db_data, "FOIA-USDA-request", "qa-fsis")
os.makedirs(qa_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")
HUD_URL = "https://www.huduser.gov/hudapi/public/usps"

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
FIPS2_TO_STATE_ABBR = {v: k for k, v in STATE_ABBR_TO_FIPS2.items()}


def _normalize_zip(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.str.extract(r"(\d{5})", expand=False)


def _first_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else pd.NA


def _latest_source_path() -> str:
    pats = [
        re.compile(r"^(\d{4}-\d{2}-\d{2})_fsis_establishment_year_fips_size_type_interim_hudfill\.csv$"),
        re.compile(r"^(\d{4}-\d{2}-\d{2})_fsis_establishment_year_fips_size_type_interim\.csv$"),
    ]
    candidates = []
    for fn in os.listdir(clean_dir):
        for idx, pat in enumerate(pats):
            m = pat.match(fn)
            if m:
                priority = 0 if idx == 0 else 1
                candidates.append((m.group(1), priority, os.path.join(clean_dir, fn)))
                break
    if not candidates:
        raise FileNotFoundError(f"No FSIS interim file found in {clean_dir}")
    candidates.sort(key=lambda x: (x[0], -x[1]))
    newest_date = candidates[-1][0]
    same_date = [c for c in candidates if c[0] == newest_date]
    same_date.sort(key=lambda x: x[1])
    return same_date[0][2]


def _hud_call_all(token: str, year: int, quarter: int, timeout: int = 120) -> requests.Response:
    params = {"type": 2, "query": "All", "year": int(year), "quarter": int(quarter)}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    return requests.get(HUD_URL, params=params, headers=headers, timeout=timeout)


def _coerce_hud_df(results) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=["zip5", "geoid", "score", "state_from_geoid"])

    df = pd.DataFrame(results)
    if "zip" not in df.columns or "geoid" not in df.columns:
        return pd.DataFrame(columns=["zip5", "geoid", "score", "state_from_geoid"])

    df["zip5"] = _normalize_zip(df["zip"])
    df["geoid"] = df["geoid"].astype("string").str.strip()
    df = df[df["zip5"].notna() & df["geoid"].str.fullmatch(r"\d{5}", na=False)].copy()
    if df.empty:
        return pd.DataFrame(columns=["zip5", "geoid", "score", "state_from_geoid"])

    for col in ["tot_ratio", "res_ratio", "bus_ratio", "oth_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    df["score"] = df["tot_ratio"].fillna(df["res_ratio"]).fillna(df["bus_ratio"]).fillna(df["oth_ratio"]).fillna(0)
    df["state_from_geoid"] = df["geoid"].astype("string").str[:2].map(FIPS2_TO_STATE_ABBR)

    return df[["zip5", "geoid", "score", "state_from_geoid"]].copy()


def _fetch_year_crosswalk(token: str, year: int) -> Tuple[pd.DataFrame, Dict]:
    # Prefer Q4, fallback to earlier quarters in same year.
    quarters = [4, 3, 2, 1]
    last_err = ""

    for q in quarters:
        retries = 0
        while retries < 3:
            try:
                r = _hud_call_all(token=token, year=year, quarter=q)
                if r.status_code == 429:
                    wait = min(6.0, (1.8 ** retries) + random.uniform(0.2, 0.7))
                    last_err = "http_429"
                    time.sleep(wait)
                    retries += 1
                    continue
                if r.status_code != 200:
                    last_err = f"http_{r.status_code}"
                    break

                payload = r.json()
                results = payload.get("data", {}).get("results", [])
                hud_df = _coerce_hud_df(results)
                if not hud_df.empty:
                    meta = {
                        "year": int(year),
                        "quarter_used": int(q),
                        "status": "ok",
                        "error": "",
                        "n_rows_hud": int(len(hud_df)),
                        "n_zip_hud": int(hud_df["zip5"].nunique()),
                    }
                    return hud_df, meta

                last_err = "no_results"
                break
            except Exception as e:
                last_err = str(e)[:180]
                time.sleep(0.4 + random.uniform(0.0, 0.4))
                retries += 1

    return pd.DataFrame(columns=["zip5", "geoid", "score", "state_from_geoid"]), {
        "year": int(year),
        "quarter_used": pd.NA,
        "status": "fail",
        "error": last_err,
        "n_rows_hud": 0,
        "n_zip_hud": 0,
    }


def _build_lookup_for_year(hud_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if hud_df.empty:
        cols_state = ["zip5", "state", "fips_code_from_hud", "hud_ratio_score"]
        cols_zip = ["zip5", "fips_code_from_hud", "hud_ratio_score"]
        return pd.DataFrame(columns=cols_state), pd.DataFrame(columns=cols_zip)

    s = (
        hud_df[hud_df["state_from_geoid"].notna()]
        .sort_values(["zip5", "state_from_geoid", "score", "geoid"], ascending=[True, True, False, True])
        .drop_duplicates(["zip5", "state_from_geoid"])
        .rename(columns={"state_from_geoid": "state", "geoid": "fips_code_from_hud", "score": "hud_ratio_score"})
        [["zip5", "state", "fips_code_from_hud", "hud_ratio_score"]]
        .reset_index(drop=True)
    )

    z = (
        hud_df.sort_values(["zip5", "score", "geoid"], ascending=[True, False, True])
        .drop_duplicates(["zip5"])
        .rename(columns={"geoid": "fips_code_from_hud", "score": "hud_ratio_score"})
        [["zip5", "fips_code_from_hud", "hud_ratio_score"]]
        .reset_index(drop=True)
    )

    return s, z


def _build_county_year(df: pd.DataFrame) -> pd.DataFrame:
    src = df[df["fips_code"].notna() & df["year"].notna()].copy()
    src["year"] = pd.to_numeric(src["year"], errors="coerce").astype("Int64")
    src = src.dropna(subset=["year"]).copy()
    src["year"] = src["year"].astype(int)
    src["est_size_combo_key"] = src["est_key"].astype("string") + "::" + src["size_bucket_final"].astype("string")

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
    token = os.environ.get("HUD_API_TOKEN", "").strip()
    if not token:
        raise EnvironmentError("Missing HUD_API_TOKEN in environment.")

    src = _latest_source_path()
    print("Using source:", src)
    df = pd.read_csv(src, low_memory=False)

    required = {"fips_code", "zip", "year"}
    if not required.issubset(df.columns):
        raise KeyError("Source interim must include fips_code, zip, year columns.")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["zip5"] = _normalize_zip(df["zip"])
    if "state" in df.columns:
        df["state"] = df["state"].astype("string").str.upper().str.strip()
    else:
        df["state"] = pd.NA

    missing = df[df["fips_code"].isna()].copy()
    need = missing[missing["zip5"].notna() & missing["year"].notna()].copy()

    req = (
        need.groupby(["year", "zip5"], as_index=False)
        .agg(
            n_rows_missing=("zip5", "size"),
            state_mode=("state", lambda s: _first_non_null(s.dropna().mode())),
        )
        .sort_values(["year", "zip5"])
        .reset_index(drop=True)
    )
    req_out = os.path.join(qa_dir, f"{today_str}_fsis_missing_fips_zip_year_requests.csv")
    req.to_csv(req_out, index=False)

    years = sorted(int(y) for y in req["year"].dropna().unique())

    year_meta_rows = []
    resolved_rows = []

    for idx, y in enumerate(years, start=1):
        hud_df, meta = _fetch_year_crosswalk(token=token, year=y)
        year_meta_rows.append(meta)

        req_y = req[req["year"].astype(int) == int(y)].copy()
        if req_y.empty:
            continue

        state_lookup, zip_lookup = _build_lookup_for_year(hud_df)

        r = req_y.rename(columns={"state_mode": "state"}).merge(state_lookup, on=["zip5", "state"], how="left")
        r = r.merge(
            zip_lookup.rename(
                columns={
                    "fips_code_from_hud": "fips_code_from_hud_zip_fallback",
                    "hud_ratio_score": "hud_ratio_score_zip_fallback",
                }
            ),
            on=["zip5"],
            how="left",
        )

        r["fips_code_from_hud"] = r["fips_code_from_hud"].where(
            r["fips_code_from_hud"].notna(), r["fips_code_from_hud_zip_fallback"]
        )
        r["hud_ratio_score"] = pd.to_numeric(r["hud_ratio_score"], errors="coerce").where(
            pd.to_numeric(r["hud_ratio_score"], errors="coerce").notna(),
            pd.to_numeric(r["hud_ratio_score_zip_fallback"], errors="coerce"),
        )
        r["hud_lookup_method"] = "state_match"
        r.loc[r["fips_code_from_hud"].isna(), "hud_lookup_method"] = "no_match"
        r.loc[
            r["fips_code_from_hud"].notna() & r["fips_code_from_hud_zip_fallback"].notna() & r["fips_code_from_hud"].eq(r["fips_code_from_hud_zip_fallback"]),
            "hud_lookup_method",
        ] = "zip_fallback"

        r["query_year_used"] = int(y)
        r["query_quarter_used"] = meta.get("quarter_used", pd.NA)
        r["hud_year_status"] = meta.get("status", "fail")
        r["hud_year_error"] = meta.get("error", "")

        resolved_rows.append(
            r[
                [
                    "year",
                    "zip5",
                    "state",
                    "n_rows_missing",
                    "fips_code_from_hud",
                    "hud_ratio_score",
                    "hud_lookup_method",
                    "query_year_used",
                    "query_quarter_used",
                    "hud_year_status",
                    "hud_year_error",
                ]
            ]
        )

        print(f"HUD bulk years completed: {idx}/{len(years)} (year={y}, status={meta['status']}, quarter={meta['quarter_used']})")

    year_meta = pd.DataFrame(year_meta_rows)
    year_meta_out = os.path.join(qa_dir, f"{today_str}_fsis_hud_bulk_year_status.csv")
    year_meta.to_csv(year_meta_out, index=False)

    if resolved_rows:
        lookup = pd.concat(resolved_rows, ignore_index=True)
    else:
        lookup = pd.DataFrame(
            columns=[
                "year", "zip5", "state", "n_rows_missing", "fips_code_from_hud", "hud_ratio_score",
                "hud_lookup_method", "query_year_used", "query_quarter_used", "hud_year_status", "hud_year_error",
            ]
        )

    lookup_out = os.path.join(qa_dir, f"{today_str}_fsis_zip_year_fips_lookup_from_hud_bulk.csv")
    lookup.to_csv(lookup_out, index=False)

    out = df.copy()
    out["fips_code"] = out["fips_code"].astype("string")

    lk = lookup[
        [
            "year",
            "zip5",
            "fips_code_from_hud",
            "hud_ratio_score",
            "hud_lookup_method",
            "query_year_used",
            "query_quarter_used",
            "hud_year_status",
        ]
    ].rename(
        columns={
            "fips_code_from_hud": "hud_bulk_fips_code_from_hud",
            "hud_ratio_score": "hud_bulk_ratio_score_lookup",
            "hud_lookup_method": "hud_bulk_lookup_method_lookup",
            "query_year_used": "hud_bulk_query_year_used_lookup",
            "query_quarter_used": "hud_bulk_query_quarter_used_lookup",
            "hud_year_status": "hud_bulk_year_status_lookup",
        }
    )
    out = out.merge(lk, on=["year", "zip5"], how="left")

    mask_fill = out["fips_code"].isna() & out["hud_bulk_fips_code_from_hud"].notna()
    out.loc[mask_fill, "fips_code"] = out.loc[mask_fill, "hud_bulk_fips_code_from_hud"]
    out.loc[mask_fill, "fips_fill_method"] = out.loc[mask_fill, "fips_fill_method"].where(
        out.loc[mask_fill, "fips_fill_method"].notna(), "hud_bulk_zip_year"
    )

    out["hud_bulk_zip_year_fips_filled"] = mask_fill.astype("Int64")
    out["hud_bulk_zip_year_ratio_score"] = pd.to_numeric(out["hud_bulk_ratio_score_lookup"], errors="coerce")
    out["hud_bulk_lookup_method"] = out["hud_bulk_lookup_method_lookup"].astype("string")
    out["hud_bulk_query_year_used"] = pd.to_numeric(out["hud_bulk_query_year_used_lookup"], errors="coerce").astype("Int64")
    out["hud_bulk_query_quarter_used"] = pd.to_numeric(out["hud_bulk_query_quarter_used_lookup"], errors="coerce").astype("Int64")
    out["hud_bulk_year_status"] = out["hud_bulk_year_status_lookup"].astype("string")

    interim_out = os.path.join(clean_dir, f"{today_str}_fsis_establishment_year_fips_size_type_interim_hudbulk.csv")
    out.to_csv(interim_out, index=False)

    county = _build_county_year(out)
    county_out = os.path.join(clean_dir, f"{today_str}_fsis_county_year_fips_est_size_type_summary_hudbulk.csv")
    county.to_csv(county_out, index=False)

    pre_missing = int(df["fips_code"].isna().sum())
    post_missing = int(out["fips_code"].isna().sum())

    qa = pd.DataFrame(
        {
            "metric": [
                "n_rows_source_input",
                "n_rows_missing_fips_before",
                "n_rows_missing_with_zip_year",
                "n_unique_zip_year_requested",
                "n_years_requested",
                "n_years_hud_success",
                "n_lookup_rows_resolved",
                "n_rows_fips_filled_by_hud_bulk_zip_year",
                "n_rows_missing_fips_after",
                "n_county_year_rows_after_hud_bulk_fill",
            ],
            "value": [
                len(df),
                pre_missing,
                len(need),
                len(req),
                len(years),
                int((year_meta["status"] == "ok").sum()) if not year_meta.empty else 0,
                int(lookup["fips_code_from_hud"].notna().sum()) if not lookup.empty else 0,
                int(mask_fill.sum()),
                post_missing,
                len(county),
            ],
        }
    )
    qa_out = os.path.join(qa_dir, f"{today_str}_fsis_hud_bulk_zip_year_fill_metrics.csv")
    qa.to_csv(qa_out, index=False)

    print("Saved:", req_out)
    print("Saved:", year_meta_out)
    print("Saved:", lookup_out)
    print("Saved:", interim_out)
    print("Saved:", county_out)
    print("Saved:", qa_out)
    print("Missing FIPS before:", pre_missing, "after:", post_missing, "filled:", int(mask_fill.sum()))


if __name__ == "__main__":
    main()
