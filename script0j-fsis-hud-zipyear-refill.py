#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Year-aware HUD USPS ZIP->County FIPS refill for remaining missing FSIS FIPS.

What this script does:
1) Reads latest HUD-filled FSIS interim if available, otherwise latest interim.
2) Isolates rows where FIPS is still missing but ZIP+year are present.
3) Queries HUD API on unique ZIP-YEAR pairs with strict backoff for 429s.
4) Builds ZIP-YEAR -> FIPS lookup and merges back into interim.
5) Rebuilds county-year summary from the refilled interim.

Required env var:
- HUD_API_TOKEN
"""

import os
import random
import re
import time
from datetime import date
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
from functions import normalize_zip5, first_non_null, STATE_ABBR_TO_FIPS2


db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
clean_dir = os.path.join(db_data, "clean")
qa_dir = os.path.join(db_data, "FOIA-USDA-request", "qa-fsis")
os.makedirs(qa_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")
HUD_URL = "https://www.huduser.gov/hudapi/public/usps"

_normalize_zip = normalize_zip5
_first_non_null = first_non_null


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
                # priority: prefer existing hudfill file over raw interim when same date
                priority = 0 if idx == 0 else 1
                candidates.append((m.group(1), priority, os.path.join(clean_dir, fn)))
                break
    if not candidates:
        raise FileNotFoundError(f"No FSIS interim file found in {clean_dir}")
    candidates.sort(key=lambda x: (x[0], -x[1]))
    # newest date; within date prefer hudfill
    newest_date = candidates[-1][0]
    same_date = [c for c in candidates if c[0] == newest_date]
    same_date.sort(key=lambda x: x[1])
    return same_date[0][2]


def _hud_call(token: str, zip5: str, year: int, quarter: int, timeout: int = 30) -> requests.Response:
    params = {"type": 2, "query": zip5, "year": int(year), "quarter": int(quarter)}
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    return requests.get(HUD_URL, params=params, headers=headers, timeout=timeout)


def _pick_best_result(results, state_abbr: Optional[str]) -> Tuple[Optional[str], Optional[float], int, bool]:
    if not results:
        return None, None, 0, False

    df = pd.DataFrame(results)
    if "geoid" not in df.columns:
        return None, None, int(len(df)), False

    df["geoid"] = df["geoid"].astype("string").str.strip()
    df = df[df["geoid"].str.fullmatch(r"\d{5}", na=False)].copy()
    if df.empty:
        return None, None, 0, False

    for col in ["tot_ratio", "res_ratio", "bus_ratio", "oth_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
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
    return str(best["geoid"]), float(best["score"]), int(len(df)), state_match_used


def _resolve_zip_year(token: str, zip5: str, year: int, state_abbr: Optional[str]) -> Dict:
    # Try exact year only (Q4->Q1). Keep retries shallow to avoid very long runs.
    attempts = [(year, 4), (year, 3), (year, 2), (year, 1)]
    last_err = ""

    for y, q in attempts:
        retry = 0
        while retry < 3:
            try:
                resp = _hud_call(token=token, zip5=zip5, year=y, quarter=q)
                if resp.status_code == 429:
                    wait = min(6.0, (1.8 ** retry) + random.uniform(0.2, 0.7))
                    last_err = "http_429"
                    time.sleep(wait)
                    retry += 1
                    continue
                if resp.status_code != 200:
                    last_err = f"http_{resp.status_code}"
                    break

                payload = resp.json()
                results = payload.get("data", {}).get("results", [])
                geoid, score, cand, state_match_used = _pick_best_result(results, state_abbr=state_abbr)
                if geoid is not None:
                    return {
                        "zip5": zip5,
                        "year": int(year),
                        "query_year_used": int(y),
                        "query_quarter_used": int(q),
                        "hud_geoid_fips": geoid,
                        "hud_ratio_score": score,
                        "hud_candidates_considered": int(cand),
                        "hud_state_match_used": int(state_match_used),
                        "hud_status": "ok",
                        "hud_error": "",
                    }
                last_err = "no_results"
                break
            except Exception as e:
                last_err = str(e)[:180]
                time.sleep(0.4 + random.uniform(0.0, 0.4))
                retry += 1

    return {
        "zip5": zip5,
        "year": int(year),
        "query_year_used": pd.NA,
        "query_quarter_used": pd.NA,
        "hud_geoid_fips": pd.NA,
        "hud_ratio_score": pd.NA,
        "hud_candidates_considered": 0,
        "hud_state_match_used": 0,
        "hud_status": "fail",
        "hud_error": last_err,
    }


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

    # 1) Build ZIP-YEAR request frame from all missing rows with ZIP+year
    req = (
        need.groupby(["zip5", "year"], as_index=False)
        .agg(
            n_rows_missing=("zip5", "size"),
            state_mode=("state", lambda s: _first_non_null(s.dropna().mode())),
        )
        .sort_values(["year", "zip5"])
        .reset_index(drop=True)
    )
    req_out = os.path.join(qa_dir, f"{today_str}_fsis_missing_fips_zip_year_requests.csv")
    req.to_csv(req_out, index=False)

    # 2) Query HUD serially with strict pacing/backoff to avoid 429 floods
    rows = []
    total = len(req)
    for i, r in req.iterrows():
        item = _resolve_zip_year(
            token=token,
            zip5=str(r["zip5"]),
            year=int(r["year"]),
            state_abbr=(str(r["state_mode"]) if pd.notna(r["state_mode"]) else None),
        )
        item["n_rows_requested"] = int(r["n_rows_missing"])
        rows.append(item)

        # Global pacing between requests even when successful.
        time.sleep(0.16 + random.uniform(0.01, 0.05))

        done = i + 1
        if done % 100 == 0 or done == total:
            print(f"HUD ZIP-YEAR queries completed: {done}/{total}")

    hud = pd.DataFrame(rows)
    hud_out = os.path.join(qa_dir, f"{today_str}_fsis_hud_zip_year_county_crosswalk_used.csv")
    hud.to_csv(hud_out, index=False)

    # 3) Build zip<>fips interim lookup from API results
    zip_fips_lookup = (
        hud[hud["hud_status"] == "ok"]
        [["zip5", "year", "hud_geoid_fips", "hud_ratio_score", "query_year_used", "query_quarter_used", "hud_state_match_used"]]
        .drop_duplicates(["zip5", "year"])
        .rename(columns={"hud_geoid_fips": "fips_code_from_hud"})
        .reset_index(drop=True)
    )
    zip_fips_lookup_out = os.path.join(qa_dir, f"{today_str}_fsis_zip_year_fips_lookup_from_hud.csv")
    zip_fips_lookup.to_csv(zip_fips_lookup_out, index=False)

    # 4) Merge fill back into interim by zip5+year
    out = df.copy()
    out["fips_code"] = out["fips_code"].astype("string")
    out = out.merge(
        hud[
            [
                "zip5",
                "year",
                "hud_geoid_fips",
                "hud_ratio_score",
                "query_year_used",
                "query_quarter_used",
                "hud_status",
            ]
        ],
        on=["zip5", "year"],
        how="left",
    )

    mask_fill = out["fips_code"].isna() & out["hud_geoid_fips"].notna()
    out.loc[mask_fill, "fips_code"] = out.loc[mask_fill, "hud_geoid_fips"]
    out.loc[mask_fill, "fips_fill_method"] = out.loc[mask_fill, "fips_fill_method"].where(
        out.loc[mask_fill, "fips_fill_method"].notna(), "hud_zip_year_exact"
    )
    out["hud_zip_year_fips_filled"] = mask_fill.astype("Int64")
    out["hud_zip_year_ratio_score"] = pd.to_numeric(out["hud_ratio_score"], errors="coerce")
    out["hud_zip_year_query_year_used"] = pd.to_numeric(out["query_year_used"], errors="coerce").astype("Int64")
    out["hud_zip_year_query_quarter_used"] = pd.to_numeric(out["query_quarter_used"], errors="coerce").astype("Int64")
    out["hud_zip_year_status"] = out["hud_status"].astype("string")

    interim_out = os.path.join(clean_dir, f"{today_str}_fsis_establishment_year_fips_size_type_interim_hudfill_zipyear.csv")
    out.to_csv(interim_out, index=False)

    # 5) Rebuild county-year summary
    county = _build_county_year(out)
    county_out = os.path.join(clean_dir, f"{today_str}_fsis_county_year_fips_est_size_type_summary_hudfill_zipyear.csv")
    county.to_csv(county_out, index=False)

    pre_missing = int(df["fips_code"].isna().sum())
    post_missing = int(out["fips_code"].isna().sum())

    qa = pd.DataFrame(
        {
            "metric": [
                "n_rows_source_input",
                "n_rows_missing_fips_before",
                "n_rows_missing_with_zip_year",
                "n_unique_zip_missing",
                "n_unique_zip_year_requested",
                "n_zip_year_hud_success",
                "n_rows_fips_filled_by_hud_zip_year",
                "n_rows_missing_fips_after",
                "n_county_year_rows_after_hud_zip_year_fill",
            ],
            "value": [
                len(df),
                pre_missing,
                len(need),
                need["zip5"].nunique(),
                len(req),
                int((hud["hud_status"] == "ok").sum()),
                int(mask_fill.sum()),
                post_missing,
                len(county),
            ],
        }
    )
    qa_out = os.path.join(qa_dir, f"{today_str}_fsis_hud_zip_year_fill_metrics.csv")
    qa.to_csv(qa_out, index=False)

    print("Saved:", req_out)
    print("Saved:", hud_out)
    print("Saved:", zip_fips_lookup_out)
    print("Saved:", interim_out)
    print("Saved:", county_out)
    print("Saved:", qa_out)
    print("Missing FIPS before:", pre_missing, "after:", post_missing, "filled:", int(mask_fill.sum()))


if __name__ == "__main__":
    main()
