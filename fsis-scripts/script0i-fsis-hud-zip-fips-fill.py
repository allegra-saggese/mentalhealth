#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill missing FSIS FIPS codes via HUD USPS ZIP->County crosswalk API.

Expected token:
- HUD_API_TOKEN in environment

Inputs:
- latest *_fsis_establishment_year_fips_size_type_interim.csv in Data/clean

Outputs:
- Data/clean/YYYY-MM-DD_fsis_establishment_year_fips_size_type_interim_hudfill.csv
- Data/clean/YYYY-MM-DD_fsis_county_year_fips_est_size_type_summary_hudfill.csv
- FOIA QA folder artifacts documenting ZIP list, HUD responses, and fill metrics
"""

import os
import re
import time
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from functions import latest_file_by_regex, normalize_zip5, first_non_null, STATE_ABBR_TO_FIPS2


db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
clean_dir = os.path.join(db_data, "clean")
qa_dir = os.path.join(db_data, "FOIA-USDA-request", "qa-fsis")
os.makedirs(qa_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")
HUD_URL = "https://www.huduser.gov/hudapi/public/usps"


def _latest_interim_path() -> str:
    return latest_file_by_regex(
        clean_dir,
        r"^(\d{4}-\d{2}-\d{2})_fsis_establishment_year_fips_size_type_interim\.csv$",
    )


_normalize_zip = normalize_zip5
_first_non_null = first_non_null


def _hud_call(
    token: str,
    zip5: str,
    year: Optional[int],
    quarter: Optional[int],
    timeout: int = 25,
) -> requests.Response:
    params = {"type": 2, "query": zip5}
    if year is not None:
        params["year"] = int(year)
    if quarter is not None:
        params["quarter"] = int(quarter)
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    return requests.get(HUD_URL, params=params, headers=headers, timeout=timeout)


def _pick_best_result(results, state_abbr: Optional[str]) -> Tuple[Optional[str], Optional[float], bool, int]:
    if not results:
        return None, None, False, 0

    df = pd.DataFrame(results)
    if "geoid" not in df.columns:
        return None, None, False, int(len(df))

    df["geoid"] = df["geoid"].astype("string").str.strip()
    df = df[df["geoid"].str.fullmatch(r"\d{5}", na=False)].copy()
    if df.empty:
        return None, None, False, 0

    for ratio_col in ["tot_ratio", "res_ratio", "bus_ratio", "oth_ratio"]:
        if ratio_col in df.columns:
            df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
        else:
            df[ratio_col] = 0.0
    df["score"] = df["tot_ratio"].fillna(df["res_ratio"]).fillna(df["bus_ratio"]).fillna(df["oth_ratio"]).fillna(0)

    state_match_used = False
    if state_abbr and state_abbr in STATE_ABBR_TO_FIPS2:
        prefix = STATE_ABBR_TO_FIPS2[state_abbr]
        m = df[df["geoid"].str.startswith(prefix)].copy()
        if not m.empty:
            df = m
            state_match_used = True

    df = df.sort_values(["score", "geoid"], ascending=[False, True])
    best = df.iloc[0]
    return str(best["geoid"]), float(best["score"]), state_match_used, int(len(df))


def _resolve_zip_fips(token: str, zip5: str, state_abbr: Optional[str]) -> Dict:
    # Query latest available crosswalk for ZIP; fallback to explicit recent year anchors.
    attempts = [(None, None), (2025, 4), (2024, 4)]
    last_err = ""
    for y, q in attempts:
        try:
            r = _hud_call(token=token, zip5=zip5, year=y, quarter=q)
            if r.status_code != 200:
                last_err = f"http_{r.status_code}"
                continue
            payload = r.json()
            results = payload.get("data", {}).get("results", [])
            geoid, score, state_match_used, cand = _pick_best_result(results, state_abbr=state_abbr)
            if geoid is not None:
                return {
                    "zip5": zip5,
                    "query_year_used": y if y is not None else 0,
                    "query_quarter_used": q if q is not None else 0,
                    "hud_geoid_fips": geoid,
                    "hud_ratio_score": score,
                    "hud_state_match_used": int(state_match_used),
                    "hud_candidates_considered": cand,
                    "hud_status": "ok",
                    "hud_error": "",
                }
            last_err = "no_results"
        except Exception as e:
            last_err = str(e)[:180]
            time.sleep(0.05)

    return {
        "zip5": zip5,
        "query_year_used": pd.NA,
        "query_quarter_used": pd.NA,
        "hud_geoid_fips": pd.NA,
        "hud_ratio_score": pd.NA,
        "hud_state_match_used": 0,
        "hud_candidates_considered": 0,
        "hud_status": "fail",
        "hud_error": last_err,
    }


def main():
    token = os.environ.get("HUD_API_TOKEN", "").strip()
    if not token:
        raise EnvironmentError("Missing HUD_API_TOKEN in environment.")

    src = _latest_interim_path()
    print("Using interim source:", src)
    df = pd.read_csv(src, low_memory=False)

    if "fips_code" not in df.columns or "zip" not in df.columns or "year" not in df.columns:
        raise KeyError("Interim source must include fips_code, zip, year.")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["zip5"] = _normalize_zip(df["zip"])
    if "state" in df.columns:
        df["state"] = df["state"].astype("string").str.upper().str.strip()
    else:
        df["state"] = pd.NA

    missing = df[df["fips_code"].isna()].copy()
    need = missing[missing["zip5"].notna() & missing["year"].notna()].copy()

    # Save establishment rows needing HUD fill (for transparency)
    need_list_cols = [c for c in [
        "est_key", "establishment_id", "establishment_number", "establishment_name",
        "year", "zip", "zip5", "city", "state", "county", "operation_category_year",
        "fips_fill_method"
    ] if c in need.columns]
    need_list_out = os.path.join(qa_dir, f"{today_str}_fsis_missing_fips_establishments_for_hud.csv")
    need[need_list_cols].to_csv(need_list_out, index=False)

    # Unique ZIP list (state mode as tie-break context)
    req = (
        need.groupby(["zip5"], as_index=False)
        .agg(
            n_rows=("est_key", "size"),
            state_mode=("state", lambda s: _first_non_null(s.dropna().mode())),
        )
    )

    # Resolve via HUD
    rows = []
    todo = []
    for _, r in req.iterrows():
        todo.append(
            {
                "zip5": str(r["zip5"]),
                "state_mode": (str(r["state_mode"]) if pd.notna(r["state_mode"]) else None),
                "n_rows": int(r["n_rows"]),
            }
        )

    max_workers = 12
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {
            ex.submit(
                _resolve_zip_fips,
                token,
                item["zip5"],
                item["state_mode"],
            ): item
            for item in todo
        }
        done = 0
        total = len(fut_map)
        for fut in as_completed(fut_map):
            item = fut_map[fut]
            out = fut.result()
            out["n_rows_requested"] = item["n_rows"]
            rows.append(out)
            done += 1
            if done % 500 == 0 or done == total:
                print(f"HUD queries completed: {done}/{total}")

    hud = pd.DataFrame(rows)
    hud_out = os.path.join(qa_dir, f"{today_str}_fsis_hud_zip_county_crosswalk_used.csv")
    hud.to_csv(hud_out, index=False)

    # Merge fill back
    out = df.copy()
    out["fips_code"] = out["fips_code"].astype("string")
    out = out.merge(
        hud[["zip5", "hud_geoid_fips", "hud_ratio_score", "query_year_used", "query_quarter_used", "hud_status"]],
        on=["zip5"],
        how="left",
    )

    mask_fill = out["fips_code"].isna() & out["hud_geoid_fips"].notna()
    out.loc[mask_fill, "fips_code"] = out.loc[mask_fill, "hud_geoid_fips"]
    out.loc[mask_fill, "fips_fill_method"] = out.loc[mask_fill, "fips_fill_method"].where(
        out.loc[mask_fill, "fips_fill_method"].notna(), "hud_zip_year_crosswalk"
    )
    out["hud_zip_fips_filled"] = mask_fill.astype("Int64")
    out["hud_zip_ratio_score"] = pd.to_numeric(out["hud_ratio_score"], errors="coerce")
    out["hud_zip_year_used"] = pd.to_numeric(out["query_year_used"], errors="coerce").astype("Int64")
    out["hud_zip_quarter_used"] = pd.to_numeric(out["query_quarter_used"], errors="coerce").astype("Int64")
    out["hud_zip_status"] = out["hud_status"].astype("string")

    # Save updated interim
    interim_out = os.path.join(clean_dir, f"{today_str}_fsis_establishment_year_fips_size_type_interim_hudfill.csv")
    out.to_csv(interim_out, index=False)

    # Rebuild county-year summary from HUD-filled interim
    src2 = out[out["fips_code"].notna() & out["year"].notna()].copy()
    src2["year"] = pd.to_numeric(src2["year"], errors="coerce").astype("Int64")
    src2 = src2.dropna(subset=["year"]).copy()
    src2["year"] = src2["year"].astype(int)
    src2["est_size_combo_key"] = src2["est_key"].astype("string") + "::" + src2["size_bucket_final"].astype("string")
    for b in ["1", "2", "3", "4", "5", "missing"]:
        src2[f"size_bucket_{b}"] = (src2["size_bucket_final"].astype("string") == b).astype("Int64")

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
        src2.groupby(["fips_code", "year"], as_index=False)
        .agg(**agg)
        .rename(columns={"fips_code": "fips"})
        .sort_values(["fips", "year"])
        .reset_index(drop=True)
    )
    county_out = os.path.join(clean_dir, f"{today_str}_fsis_county_year_fips_est_size_type_summary_hudfill.csv")
    county.to_csv(county_out, index=False)

    # QA metrics
    pre_missing = int(df["fips_code"].isna().sum())
    post_missing = int(out["fips_code"].isna().sum())
    qa = pd.DataFrame(
        {
            "metric": [
                "n_rows_interim_input",
                "n_rows_missing_fips_before",
                "n_rows_with_zip_for_hud",
                "n_unique_zip_queried",
                "n_zip_hud_success",
                "n_rows_fips_filled_by_hud",
                "n_rows_missing_fips_after",
                "n_county_year_rows_after_hud_fill",
            ],
            "value": [
                len(df),
                pre_missing,
                len(need),
                len(req),
                int((hud["hud_status"] == "ok").sum()),
                int(mask_fill.sum()),
                post_missing,
                len(county),
            ],
        }
    )
    qa_out = os.path.join(qa_dir, f"{today_str}_fsis_hud_zip_fill_metrics.csv")
    qa.to_csv(qa_out, index=False)

    print("Saved:", interim_out)
    print("Saved:", county_out)
    print("Saved QA:", need_list_out)
    print("Saved QA:", hud_out)
    print("Saved QA:", qa_out)
    print("Missing FIPS before:", pre_missing, "after:", post_missing, "filled:", int(mask_fill.sum()))


if __name__ == "__main__":
    main()
