#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script0i-fsis-hud-fill.py -- fill missing FSIS FIPS codes from the HUD USPS ZIP->County crosswalk (three strategies, Stage E of the consolidation).

  bulk     (was script0k-fsis-hud-bulk-zipyear-fill.py)  the HUD crosswalk pulled in bulk by year and quarter; the default strategy
  zip      (was script0i-fsis-hud-zip-fips-fill.py)      one HUD request per unique ZIP
  zipyear  (was script0j-fsis-hud-zipyear-refill.py)     year-aware refill of the rows still missing, with backoff for 429s

Each section's code is the old script's code, moved unchanged (the paths now come from packages.py).
Run:     HUD_API_TOKEN=... python3 fsis-scripts/script0i-fsis-hud-fill.py [--strategy bulk|targeted] [--only bulk zip zipyear]
         --strategy bulk (default) = section bulk; --strategy targeted = sections zip then zipyear; --only selects sections directly.
Needs:   HUD_API_TOKEN in the environment and network access (these sections call the HUD API; they were checked by AST comparison with the old
         scripts and a dry run, not re-run here). Called by script0e-fsis-clean-all.py.
"""
import sys


def section_zip():
    """HUD ZIP lookups, one request per ZIP.

    Old script fsis-scripts/script0i-fsis-hud-zip-fips-fill.py (docstring below), code moved unchanged.

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


    from packages import db_base            # the same folder as before, now defined once in packages.py
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


def section_zipyear():
    """Year-aware HUD refill.

    Old script fsis-scripts/script0j-fsis-hud-zipyear-refill.py (docstring below), code moved unchanged.

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


    from packages import db_base            # the same folder as before, now defined once in packages.py
    db_data   = os.path.join(db_base, "Data")
    clean_dir = os.path.join(db_data, "clean")
    diagnostic_dir = os.path.join(db_data, "clean", "diagnostic")
    qa_dir    = os.path.join(db_data, "FOIA-USDA-request", "qa-fsis")
    os.makedirs(diagnostic_dir, exist_ok=True)
    os.makedirs(qa_dir,    exist_ok=True)

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
        for fn in os.listdir(diagnostic_dir):
            for idx, pat in enumerate(pats):
                m = pat.match(fn)
                if m:
                    # priority: prefer existing hudfill file over raw interim when same date
                    priority = 0 if idx == 0 else 1
                    candidates.append((m.group(1), priority, os.path.join(diagnostic_dir, fn)))
                    break
        if not candidates:
            raise FileNotFoundError(f"No FSIS interim file found in {diagnostic_dir}")
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

        interim_out = os.path.join(diagnostic_dir, f"{today_str}_fsis_establishment_year_fips_size_type_interim_hudfill_zipyear.csv")
        out.to_csv(interim_out, index=False)

        # 5) Rebuild county-year summary
        county = _build_county_year(out)
        county_out = os.path.join(diagnostic_dir, f"{today_str}_fsis_county_year_fips_est_size_type_summary_hudfill_zipyear.csv")
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


def section_bulk():
    """Bulk HUD crosswalk fill.

    Old script fsis-scripts/script0k-fsis-hud-bulk-zipyear-fill.py (docstring below), code moved unchanged.

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
    from functions import normalize_zip5, first_non_null, STATE_ABBR_TO_FIPS2


    from packages import db_base            # the same folder as before, now defined once in packages.py
    db_data   = os.path.join(db_base, "Data")
    clean_dir = os.path.join(db_data, "clean")
    diagnostic_dir = os.path.join(db_data, "clean", "diagnostic")
    qa_dir    = os.path.join(db_data, "FOIA-USDA-request", "qa-fsis")
    os.makedirs(diagnostic_dir, exist_ok=True)
    os.makedirs(qa_dir,    exist_ok=True)

    today_str = date.today().strftime("%Y-%m-%d")
    HUD_URL = "https://www.huduser.gov/hudapi/public/usps"

    FIPS2_TO_STATE_ABBR = {v: k for k, v in STATE_ABBR_TO_FIPS2.items()}


    _normalize_zip = normalize_zip5
    _first_non_null = first_non_null


    def _latest_source_path() -> str:
        pats = [
            re.compile(r"^(\d{4}-\d{2}-\d{2})_fsis_establishment_year_fips_size_type_interim_hudfill\.csv$"),
            re.compile(r"^(\d{4}-\d{2}-\d{2})_fsis_establishment_year_fips_size_type_interim\.csv$"),
        ]
        candidates = []
        for fn in os.listdir(diagnostic_dir):
            for idx, pat in enumerate(pats):
                m = pat.match(fn)
                if m:
                    priority = 0 if idx == 0 else 1
                    candidates.append((m.group(1), priority, os.path.join(diagnostic_dir, fn)))
                    break
        if not candidates:
            raise FileNotFoundError(f"No FSIS interim file found in {diagnostic_dir}")
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

        interim_out = os.path.join(diagnostic_dir, f"{today_str}_fsis_establishment_year_fips_size_type_interim_hudbulk.csv")
        out.to_csv(interim_out, index=False)

        county = _build_county_year(out)
        county_out = os.path.join(diagnostic_dir, f"{today_str}_fsis_county_year_fips_est_size_type_summary_hudbulk.csv")
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


SECTIONS = {"zip": section_zip, "zipyear": section_zipyear, "bulk": section_bulk}
STRATEGIES = {"bulk": ["bulk"], "targeted": ["zip", "zipyear"]}


def main():
    args = sys.argv[1:]
    only = []
    if "--only" in args:
        for a in args[args.index("--only") + 1:]:
            if a.startswith("--"):
                break
            only.append(a)
    strategy = args[args.index("--strategy") + 1] if "--strategy" in args else "bulk"
    if strategy not in STRATEGIES:
        raise SystemExit(f"unknown strategy {strategy!r}; choose from {list(STRATEGIES)}")
    unknown = [o for o in only if o not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    for name in (only or STRATEGIES[strategy]):
        print(f"\n{'=' * 78}\nscript0i-fsis-hud-fill: section {name}\n{'=' * 78}", flush=True)
        SECTIONS[name]()


if __name__ == "__main__":
    main()
