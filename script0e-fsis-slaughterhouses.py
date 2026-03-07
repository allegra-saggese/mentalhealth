#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build FSIS slaughterhouse panels from raw demographic + MPI files.

Outputs:
- file inventory (required)
- establishment-year panel (all + slaughterhouse subset)
- county-year slaughterhouse presence panel
- QA artifacts (unmatched geography, geo conflicts, summary metrics)
"""

import os
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Paths (kept aligned with project conventions in packages.py)
db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
raw_dir = os.path.join(db_data, "raw", "fsis")
clean_dir = os.path.join(db_data, "clean")
qa_dir = os.path.join(clean_dir, "qa-fsis")

os.makedirs(clean_dir, exist_ok=True)
os.makedirs(qa_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")


MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

GEO_COLS = [
    "fips_code",
    "county",
    "state",
    "city",
    "zip",
    "latitude",
    "longitude",
    "district",
    "circuit",
]

BASE_KEEP_COLS = [
    "establishment_id",
    "establishment_number",
    "establishment_name",
    "activities",
    "size_classifier",
    "processing_volume_category",
    "slaughter_volume_category",
]

SLAUGHTER_COL_MAP = {
    "beef_slaughter": "beef_slaughter",
    "pork_slaughter": "pork_slaughter",
    "sheep_slaughter": "sheep_slaughter",
    "goat_slaughter": "goat_slaughter",
    "lamb_slaughter": "lamb_slaughter",
    "rabbit_slaughter": "rabbit_slaughter",
    "other_meat_slaughter": "other_meat_slaughter",
    "chicken_slaughter": "chicken_slaughter",
    "turkey_slaughter": "turkey_slaughter",
    "duck_slaughter": "duck_slaughter",
    "goose_slaughter": "goose_slaughter",
    "pheasant_slaughter": "pheasant_slaughter",
    "quail_slaughter": "quail_slaughter",
    "other_poultry_slaughter": "other_poultry_slaughter",
    "meat_slaughter": "meat_slaughter",
    "poultry_slaughter": "poultry_slaughter",
    "slaughter": "slaughter",
}

COL_MAP = {
    # keys
    "establishmentid": "establishment_id",
    "estid": "establishment_id",
    "establishmentnumber": "establishment_number",
    "estnumber": "establishment_number",
    "establishmentname": "establishment_name",
    "company": "establishment_name",
    # common descriptors
    "activities": "activities",
    "size": "size_classifier",
    "processedvolumecategory": "processing_volume_category",
    "processingvolumecategory": "processing_volume_category",
    "slaughtervolumecategory": "slaughter_volume_category",
    "processing_volume_category": "processing_volume_category",
    "slaughter_volume_category": "slaughter_volume_category",
    # geography
    "fipscode": "fips_code",
    "county": "county",
    "state": "state",
    "city": "city",
    "zip": "zip",
    "zipcode": "zip",
    "latitude": "latitude",
    "longitude": "longitude",
    "district": "district",
    "circuit": "circuit",
    # modern names that normalize without underscore
    "establishment_number": "establishment_number",
    "establishment_id": "establishment_id",
    "establishment_name": "establishment_name",
    # slaughter columns (legacy + modern compatible)
    "beefslaughter": "beef_slaughter",
    "porkslaughter": "pork_slaughter",
    "sheepslaughter": "sheep_slaughter",
    "goatslaughter": "goat_slaughter",
    "lambslaughter": "lamb_slaughter",
    "rabbitslaughter": "rabbit_slaughter",
    "othermeatslaughter": "other_meat_slaughter",
    "chickenslaughter": "chicken_slaughter",
    "turkeyslaughter": "turkey_slaughter",
    "duckslaughter": "duck_slaughter",
    "gooseslaughter": "goose_slaughter",
    "pheasantslaughter": "pheasant_slaughter",
    "quailslaughter": "quail_slaughter",
    "otherpoultryslaughter": "other_poultry_slaughter",
    "meatslaughter": "meat_slaughter",
    "poultryslaughter": "poultry_slaughter",
}


def _clean_string(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _snake(s: str) -> str:
    s = _clean_string(s).lower()
    s = s.replace("&", " and ")
    s = s.replace("\n", " ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean_string(s).lower())


def _mode_non_null(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return pd.NA
    vc = s.astype("string").value_counts(dropna=True)
    return vc.index[0] if len(vc) else pd.NA


def _first_non_null(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if len(s) else pd.NA


def _collapse_operation_category(s: pd.Series):
    vals = set(s.dropna().astype("string").tolist())
    if not vals:
        return "neither_signal"
    if "both_slaughter_and_processing" in vals:
        return "both_slaughter_and_processing"
    if "slaughter_only" in vals:
        return "slaughter_only"
    if "processing_only" in vals:
        return "processing_only"
    if "other_or_unclear" in vals:
        return "other_or_unclear"
    return "neither_signal"


def _is_positive(v) -> bool:
    if pd.isna(v):
        return False
    t = str(v).strip().lower()
    if t in {"", "nan", "none", "null", "na", "n/a", "no", "n", "false", "0"}:
        return False
    return True


def _normalize_est_id(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return out


def _normalize_est_number(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.strip().str.upper()
    out = out.str.replace(r"\s+", "", regex=True)
    out = out.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})
    return out


def _normalize_fips(s: pd.Series) -> pd.Series:
    out = s.astype("string").str.replace(r"\.0$", "", regex=True)
    out = out.str.replace(r"\D", "", regex=True).str.zfill(5)
    out = out.where(out.str.len() == 5, pd.NA)
    return out


def _family_from_path(path: str) -> str:
    p = path.lower()
    b = os.path.basename(path).lower()
    if "mpi" in p or "allmpibyestname" in b or "directory_by_establishment" in b:
        return "mpi"
    if "demographic" in p or "demographic" in b:
        return "demographic"
    return "other"


def _infer_time(path: str) -> Tuple[Optional[int], Optional[int], pd.Timestamp]:
    p = path.lower()
    b = os.path.basename(path).lower()
    text = f"{p} {b}"

    # 1) strict YYYYMMDD token
    m = re.search(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            dt = pd.Timestamp(datetime(y, mo, d))
            return y, mo, dt
        except ValueError:
            pass

    # 2) DDMONYY or DDMONYYYY
    m = re.search(r"(?<!\d)(\d{1,2})([a-z]{3,9})(\d{2,4})(?!\d)", text)
    if m:
        d = int(m.group(1))
        mo_txt = m.group(2)[:3]
        yy = m.group(3)
        mo = MONTH_MAP.get(mo_txt)
        if mo is not None:
            y = int(yy)
            if y < 100:
                y = 2000 + y
            try:
                dt = pd.Timestamp(datetime(y, mo, d))
                return y, mo, dt
            except ValueError:
                pass

    # 3) Month + Year patterns
    mo = None
    for k, v in MONTH_MAP.items():
        if re.search(rf"\b{k}\b", text):
            mo = v
            break
    y_match = re.findall(r"(20\d{2})", text)
    y = int(y_match[-1]) if y_match else None
    if y is not None:
        if mo is not None:
            dt = pd.Timestamp(datetime(y, mo, 1))
            return y, mo, dt
        return y, None, pd.NaT

    return None, None, pd.NaT


def _detect_header_row(path: str, family: str, sheet_name: str) -> int:
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=15, dtype=str)
    for i, row in preview.iterrows():
        vals = {_norm(v) for v in row.tolist() if _clean_string(v)}
        if not vals:
            continue
        if family == "demographic":
            if ("estnumber" in vals and "estid" in vals) or (
                "establishmentnumber" in vals and "establishmentid" in vals
            ):
                return int(i)
        if family == "mpi":
            if (
                ("establishmentid" in vals or "estid" in vals)
                and ("county" in vals or "fipscode" in vals or "state" in vals)
            ):
                return int(i)
    return 0


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df

    out = pd.DataFrame(index=df.index)
    for c in pd.Index(df.columns).unique():
        cols = df.loc[:, df.columns == c]
        if cols.shape[1] == 1:
            out[c] = cols.iloc[:, 0]
        else:
            out[c] = cols.bfill(axis=1).iloc[:, 0]
    return out


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    raw_cols = list(df.columns)
    new_cols = []
    for c in raw_cols:
        c_norm = _norm(c)
        if c_norm in COL_MAP:
            new_cols.append(COL_MAP[c_norm])
        else:
            new_cols.append(_snake(c))
    df = df.copy()
    df.columns = new_cols
    df = _coalesce_duplicate_columns(df)
    return df


def _read_tabular(path: str, family: str) -> Tuple[Optional[pd.DataFrame], Dict]:
    ext = os.path.splitext(path)[1].lower()
    meta = {
        "file_ext": ext,
        "header_row_used": pd.NA,
        "sheet_name_used": pd.NA,
        "n_rows_read": pd.NA,
        "n_cols_read": pd.NA,
        "read_status": "ok",
        "error_msg": "",
    }

    try:
        if ext == ".csv":
            df = pd.read_csv(path, dtype=str, low_memory=False)
            meta["header_row_used"] = 0
            meta["sheet_name_used"] = pd.NA
        elif ext in {".xlsx", ".xls"}:
            xl = pd.ExcelFile(path)
            sheet_name = xl.sheet_names[0]
            header_row = _detect_header_row(path, family=family, sheet_name=sheet_name)
            df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, dtype=str)
            meta["header_row_used"] = int(header_row)
            meta["sheet_name_used"] = sheet_name
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    except Exception as e:
        meta["read_status"] = "error"
        meta["error_msg"] = str(e)[:500]
        return None, meta

    # trim and normalize obvious empties
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()
            df[c] = df[c].replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "NaN": pd.NA,
                    "None": pd.NA,
                    "NULL": pd.NA,
                    "null": pd.NA,
                }
            )

    meta["n_rows_read"] = int(df.shape[0])
    meta["n_cols_read"] = int(df.shape[1])
    return df, meta


def _key_style(df: pd.DataFrame) -> str:
    ncols = df.shape[1]
    cols = set(df.columns)
    if ncols >= 240:
        return "modern_253col"
    if ncols >= 25 and (
        "district" in cols or "size_classifier" in cols or "slaughter_volume_category" in cols
    ):
        return "legacy_30ish"
    return "other"


def _build_slaughter_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    slaughter_cols = [c for c in df.columns if "slaughter" in c and "volume_category" not in c]
    keep_slaughter_cols = [c for c in slaughter_cols if c in SLAUGHTER_COL_MAP.values()]
    keep_slaughter_cols += [c for c in slaughter_cols if c.startswith("slaughter_")]
    keep_slaughter_cols = sorted(set(keep_slaughter_cols))

    if keep_slaughter_cols:
        flag_df = df[keep_slaughter_cols].apply(lambda col: col.map(_is_positive))
        any_slaughter_col = flag_df.any(axis=1)
    else:
        any_slaughter_col = pd.Series(False, index=df.index)

    activities_col = df["activities"] if "activities" in df.columns else pd.Series(pd.NA, index=df.index)
    activity_slaughter = activities_col.astype("string").str.contains("slaughter", case=False, na=False)

    vol_col = (
        df["slaughter_volume_category"]
        if "slaughter_volume_category" in df.columns
        else pd.Series(pd.NA, index=df.index)
    )
    vol_slaughter = vol_col.map(_is_positive)

    processing_cols = [c for c in df.columns if "processing" in c and c != "processing_volume_category"]
    if processing_cols:
        processing_flag_from_cols = df[processing_cols].apply(lambda col: col.map(_is_positive)).any(axis=1)
    else:
        processing_flag_from_cols = pd.Series(False, index=df.index)

    activity_processing = activities_col.astype("string").str.contains(r"\bprocess", case=False, na=False)
    vol_processing_col = (
        df["processing_volume_category"]
        if "processing_volume_category" in df.columns
        else pd.Series(pd.NA, index=df.index)
    )
    vol_processing = vol_processing_col.map(_is_positive)

    meat_terms = [
        "meat_slaughter",
        "beef_slaughter",
        "pork_slaughter",
        "sheep_slaughter",
        "goat_slaughter",
        "lamb_slaughter",
        "rabbit_slaughter",
        "other_meat_slaughter",
    ]
    poultry_terms = [
        "poultry_slaughter",
        "chicken_slaughter",
        "turkey_slaughter",
        "duck_slaughter",
        "goose_slaughter",
        "pheasant_slaughter",
        "quail_slaughter",
        "other_poultry_slaughter",
    ]

    meat_cols = [c for c in keep_slaughter_cols if c in meat_terms]
    poultry_cols = [c for c in keep_slaughter_cols if c in poultry_terms]

    meat_flag = (
        df[meat_cols].apply(lambda col: col.map(_is_positive)).any(axis=1)
        if meat_cols
        else pd.Series(False, index=df.index)
    )
    poultry_flag = (
        df[poultry_cols].apply(lambda col: col.map(_is_positive)).any(axis=1)
        if poultry_cols
        else pd.Series(False, index=df.index)
    )

    meat_from_activity = activities_col.astype("string").str.contains("meat slaughter", case=False, na=False)
    poultry_from_activity = activities_col.astype("string").str.contains(
        "poultry slaughter", case=False, na=False
    )

    slaughter_signal = any_slaughter_col | activity_slaughter | vol_slaughter
    processing_signal = processing_flag_from_cols | activity_processing | vol_processing

    df["is_slaughterhouse_row"] = slaughter_signal.astype("Int64")
    df["is_processing_row"] = processing_signal.astype("Int64")
    df["is_meat_slaughter_row"] = (meat_flag | meat_from_activity).astype("Int64")
    df["is_poultry_slaughter_row"] = (poultry_flag | poultry_from_activity).astype("Int64")

    has_indicator_values = pd.Series(False, index=df.index)
    indicator_cols = list(set(keep_slaughter_cols + processing_cols))
    if indicator_cols:
        has_indicator_values = df[indicator_cols].notna().any(axis=1)

    has_context = activities_col.notna() | vol_col.notna() | vol_processing_col.notna() | has_indicator_values

    category = np.where(
        slaughter_signal & processing_signal,
        "both_slaughter_and_processing",
        np.where(
            slaughter_signal & (~processing_signal),
            "slaughter_only",
            np.where(
                (~slaughter_signal) & processing_signal,
                "processing_only",
                np.where(has_context, "other_or_unclear", "neither_signal"),
            ),
        ),
    )
    df["operation_category_row"] = pd.Series(category, index=df.index, dtype="string")
    return df


def _extract_geo_rows(df: pd.DataFrame, family: str, file_path: str, year, snapshot_date) -> pd.DataFrame:
    if "establishment_id" not in df.columns and "establishment_number" not in df.columns:
        return pd.DataFrame()

    cols = [c for c in GEO_COLS if c in df.columns]
    size_cols = [c for c in ["size_classifier", "processing_volume_category", "slaughter_volume_category"] if c in df.columns]
    if not cols and not size_cols:
        return pd.DataFrame()

    keep = [c for c in ["establishment_id", "establishment_number"] if c in df.columns] + cols + size_cols
    out = df[keep].copy()
    out["source_family"] = family
    out["source_file"] = file_path
    out["year"] = pd.to_numeric(year, errors="coerce")
    out["snapshot_date"] = snapshot_date

    for c in GEO_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    out["establishment_id"] = _normalize_est_id(out.get("establishment_id", pd.Series(pd.NA, index=out.index)))
    out["establishment_number"] = _normalize_est_number(
        out.get("establishment_number", pd.Series(pd.NA, index=out.index))
    )
    out["fips_code"] = _normalize_fips(out["fips_code"])
    out["state"] = out["state"].astype("string").str.upper()

    return out


def _geo_completeness(df: pd.DataFrame) -> pd.Series:
    score_cols = ["fips_code", "county", "state", "city", "zip", "latitude", "longitude"]
    return df[score_cols].notna().sum(axis=1).astype("Int64")


def _choose_best_geo_per_key_year(geo_rows: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if geo_rows.empty:
        return geo_rows
    d = geo_rows.loc[geo_rows[key_col].notna()].copy()
    if d.empty:
        return d

    d = d.loc[d["year"].notna()].copy()
    if d.empty:
        return d

    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
    d["geo_score"] = _geo_completeness(d)
    d["snapshot_date"] = pd.to_datetime(d["snapshot_date"], errors="coerce")
    d = d.sort_values(
        [key_col, "year", "geo_score", "snapshot_date"],
        ascending=[True, True, False, False],
    )
    return d.drop_duplicates([key_col, "year"], keep="first")


def _choose_best_geo_undated(geo_rows: pd.DataFrame, key_col: str) -> pd.DataFrame:
    if geo_rows.empty:
        return geo_rows
    d = geo_rows.loc[geo_rows[key_col].notna()].copy()
    if d.empty:
        return d
    d = d.loc[d["year"].isna()].copy()
    if d.empty:
        return d

    d["geo_score"] = _geo_completeness(d)
    d["snapshot_date"] = pd.to_datetime(d["snapshot_date"], errors="coerce")
    d = d.sort_values([key_col, "geo_score", "snapshot_date"], ascending=[True, False, False])
    return d.drop_duplicates([key_col], keep="first")


def _fill_same_year_geo(panel: pd.DataFrame, geo_best: pd.DataFrame, key_col: str, label: str) -> pd.DataFrame:
    if panel.empty or geo_best.empty:
        return panel
    if key_col not in panel.columns or key_col not in geo_best.columns:
        return panel

    geo_keep = [key_col, "year"] + GEO_COLS + ["source_family", "source_file", "snapshot_date"]
    geo_keep = [c for c in geo_keep if c in geo_best.columns]
    m = panel[["row_id", key_col, "year"]].merge(
        geo_best[geo_keep],
        on=[key_col, "year"],
        how="left",
        suffixes=("", "_geo"),
    )

    for c in GEO_COLS:
        if c not in m.columns:
            continue
        before_null = panel[c].isna()
        panel.loc[:, c] = panel[c].where(panel[c].notna(), m[c])
        filled = before_null & panel[c].notna()
        mask_primary = filled & panel["geo_source_primary"].isna()
        mask_file = filled & panel["geo_source_file"].isna()
        mask_family = filled & panel["geo_source_family"].isna()
        panel.loc[mask_primary, "geo_source_primary"] = f"{label}_same_year"
        panel.loc[mask_file, "geo_source_file"] = m.loc[mask_file, "source_file"].values
        panel.loc[mask_family, "geo_source_family"] = m.loc[mask_family, "source_family"].values

    return panel


def _fill_prior_year_geo(panel: pd.DataFrame, geo_best: pd.DataFrame, key_col: str, label: str) -> pd.DataFrame:
    if panel.empty or geo_best.empty or key_col not in panel.columns:
        return panel
    if key_col not in geo_best.columns:
        return panel

    d_geo = geo_best[[key_col, "year"] + GEO_COLS + ["source_family", "source_file"]].copy()
    d_geo = d_geo.loc[d_geo[key_col].notna() & d_geo["year"].notna()].copy()
    if d_geo.empty:
        return panel

    d_geo["year"] = pd.to_numeric(d_geo["year"], errors="coerce").astype("Int64")
    d_geo = d_geo.sort_values([key_col, "year"])

    lookup = {k: g.reset_index(drop=True) for k, g in d_geo.groupby(key_col, sort=False)}
    core_cols = ["fips_code", "county", "state"]

    for i, row in panel.loc[panel[key_col].notna()].iterrows():
        key = row[key_col]
        y = row["year"]
        if pd.isna(y):
            continue
        if key not in lookup:
            continue

        group = lookup[key]
        prior = group.loc[group["year"] < y]
        if prior.empty:
            continue
        best = prior.iloc[-1]

        filled_any = False
        for c in GEO_COLS:
            if c not in panel.columns or c not in best.index:
                continue
            if pd.isna(panel.at[i, c]) and pd.notna(best[c]):
                panel.at[i, c] = best[c]
                filled_any = True

        if filled_any:
            if pd.isna(panel.at[i, "geo_fallback_from_year"]):
                panel.at[i, "geo_fallback_from_year"] = int(best["year"])
            if pd.isna(panel.at[i, "geo_source_primary"]):
                panel.at[i, "geo_source_primary"] = f"{label}_prior_year_fallback"
            if pd.isna(panel.at[i, "geo_source_file"]):
                panel.at[i, "geo_source_file"] = best.get("source_file", pd.NA)
            if pd.isna(panel.at[i, "geo_source_family"]):
                panel.at[i, "geo_source_family"] = best.get("source_family", pd.NA)

            if panel.loc[i, core_cols].isna().all():
                panel.at[i, "geo_source_primary"] = f"{label}_prior_year_fallback"

    return panel


def _fill_undated_geo(panel: pd.DataFrame, geo_undated: pd.DataFrame, key_col: str, label: str) -> pd.DataFrame:
    if panel.empty or geo_undated.empty:
        return panel
    if key_col not in panel.columns or key_col not in geo_undated.columns:
        return panel

    g = geo_undated.set_index(key_col)
    for i, row in panel.loc[panel[key_col].notna()].iterrows():
        key = row[key_col]
        if key not in g.index:
            continue
        best = g.loc[key]
        filled_any = False
        for c in GEO_COLS:
            if c in panel.columns and c in best.index and pd.isna(panel.at[i, c]) and pd.notna(best[c]):
                panel.at[i, c] = best[c]
                filled_any = True
        if filled_any and pd.isna(panel.at[i, "geo_source_primary"]):
            panel.at[i, "geo_source_primary"] = f"{label}_undated_reference"
            panel.at[i, "geo_source_file"] = best.get("source_file", pd.NA)
            panel.at[i, "geo_source_family"] = best.get("source_family", pd.NA)

    return panel


def _build_file_list(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.startswith("~$") or fn == ".DS_Store":
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in {".csv", ".xlsx", ".xls"}:
                out.append(os.path.join(dp, fn))
    return sorted(out)


def main():
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"Missing FSIS raw directory: {raw_dir}")

    files = _build_file_list(raw_dir)
    if not files:
        raise FileNotFoundError(f"No tabular FSIS files found in: {raw_dir}")

    inventory_rows = []
    demo_rows = []
    geo_rows = []

    for p in files:
        family = _family_from_path(p)
        year, month, snapshot_date = _infer_time(p)
        df_raw, meta = _read_tabular(p, family=family)

        inv = {
            "file_path": p,
            "file_name": os.path.basename(p),
            "family": family,
            "file_ext": meta["file_ext"],
            "year_inferred": year if year is not None else pd.NA,
            "month_inferred": month if month is not None else pd.NA,
            "snapshot_date_inferred": snapshot_date,
            "header_row_used": meta["header_row_used"],
            "sheet_name_used": meta["sheet_name_used"],
            "n_rows_read": meta["n_rows_read"],
            "n_cols_read": meta["n_cols_read"],
            "key_style": pd.NA,
            "read_status": meta["read_status"],
            "error_msg": meta["error_msg"],
        }

        if df_raw is None:
            inventory_rows.append(inv)
            continue

        df = _standardize_columns(df_raw)
        inv["key_style"] = _key_style(df)
        inventory_rows.append(inv)

        if "establishment_id" in df.columns:
            df["establishment_id"] = _normalize_est_id(df["establishment_id"])
        else:
            df["establishment_id"] = pd.NA
        if "establishment_number" in df.columns:
            df["establishment_number"] = _normalize_est_number(df["establishment_number"])
        else:
            df["establishment_number"] = pd.NA

        # Geography rows from any readable source family
        geo = _extract_geo_rows(df, family=family, file_path=p, year=year, snapshot_date=snapshot_date)
        if not geo.empty:
            geo_rows.append(geo)

        # Demographic rows for slaughterhouse build
        if family != "demographic":
            continue
        if year is None:
            continue

        d = _build_slaughter_flags(df)
        d["year"] = int(year)
        d["snapshot_date"] = snapshot_date
        d["source_file"] = p

        for c in BASE_KEEP_COLS + list(SLAUGHTER_COL_MAP.values()):
            if c not in d.columns:
                d[c] = pd.NA

        keep = (
            BASE_KEEP_COLS
            + ["year", "snapshot_date", "source_file"]
            + [
                "is_slaughterhouse_row",
                "is_processing_row",
                "is_meat_slaughter_row",
                "is_poultry_slaughter_row",
                "operation_category_row",
            ]
        )
        d = d[keep].copy()
        d = d.loc[d["establishment_id"].notna() | d["establishment_number"].notna()].copy()
        demo_rows.append(d)

    # Save required inventory immediately
    inventory_df = pd.DataFrame(inventory_rows)
    inv_out = os.path.join(qa_dir, f"{today_str}_fsis_file_inventory.csv")
    inventory_df.to_csv(inv_out, index=False)
    print("Saved inventory:", inv_out)

    if not demo_rows:
        raise RuntimeError("No readable demographic rows were extracted. Check inventory read_status/error_msg.")

    demo_df = pd.concat(demo_rows, ignore_index=True, sort=False)
    demo_df["year"] = pd.to_numeric(demo_df["year"], errors="coerce").astype("Int64")
    demo_df = demo_df.dropna(subset=["year"])
    demo_df["year"] = demo_df["year"].astype(int)

    demo_df["est_key"] = np.where(
        demo_df["establishment_id"].notna(),
        "ID:" + demo_df["establishment_id"].astype("string"),
        "NUM:" + demo_df["establishment_number"].astype("string"),
    )

    # Establishment-year collapse (any-month-in-year logic)
    est_year = (
        demo_df.groupby(["est_key", "year"], as_index=False)
        .agg(
            establishment_id=("establishment_id", _first_non_null),
            establishment_number=("establishment_number", _first_non_null),
            establishment_name=("establishment_name", _mode_non_null),
            slaughterhouse_present_year=("is_slaughterhouse_row", "max"),
            processing_present_year=("is_processing_row", "max"),
            meat_slaughter_present_year=("is_meat_slaughter_row", "max"),
            poultry_slaughter_present_year=("is_poultry_slaughter_row", "max"),
            operation_category_year=("operation_category_row", _collapse_operation_category),
            size_classifier_mode=("size_classifier", _mode_non_null),
            processing_volume_category_mode=("processing_volume_category", _mode_non_null),
            slaughter_volume_category_mode=("slaughter_volume_category", _mode_non_null),
            n_source_rows=("source_file", "size"),
            n_files_seen_in_year=("source_file", "nunique"),
            first_snapshot_date=("snapshot_date", "min"),
            last_snapshot_date=("snapshot_date", "max"),
        )
    )

    for c in [
        "slaughterhouse_present_year",
        "processing_present_year",
        "meat_slaughter_present_year",
        "poultry_slaughter_present_year",
    ]:
        est_year[c] = pd.to_numeric(est_year[c], errors="coerce").fillna(0).astype("Int64")

    # Geography build from all readable sources
    geo_df = pd.concat(geo_rows, ignore_index=True, sort=False) if geo_rows else pd.DataFrame()
    if geo_df.empty:
        print("Warning: no geography rows extracted from readable files.")
        for c in GEO_COLS:
            est_year[c] = pd.NA
        est_year["geo_source_primary"] = pd.NA
        est_year["geo_source_family"] = pd.NA
        est_year["geo_source_file"] = pd.NA
        est_year["geo_fallback_from_year"] = pd.NA
    else:
        geo_df["year"] = pd.to_numeric(geo_df["year"], errors="coerce").astype("Int64")
        geo_df["snapshot_date"] = pd.to_datetime(geo_df["snapshot_date"], errors="coerce")
        geo_df["fips_code"] = _normalize_fips(geo_df["fips_code"])
        geo_df["establishment_id"] = _normalize_est_id(geo_df["establishment_id"])
        geo_df["establishment_number"] = _normalize_est_number(geo_df["establishment_number"])

        # QA: conflicting same-year geographies
        conflict_rows = []
        for key_col in ["establishment_id", "establishment_number"]:
            d = geo_df.loc[geo_df[key_col].notna() & geo_df["year"].notna()].copy()
            if d.empty:
                continue
            g = (
                d.groupby([key_col, "year"], as_index=False)
                .agg(
                    n_fips=("fips_code", lambda s: int(s.dropna().astype("string").nunique())),
                    n_county=("county", lambda s: int(s.dropna().astype("string").nunique())),
                    sample_files=("source_file", lambda s: " | ".join(pd.Series(s).dropna().astype(str).head(3))),
                )
            )
            g = g.loc[(g["n_fips"] > 1) | (g["n_county"] > 1)].copy()
            if not g.empty:
                g["key_type"] = key_col
                conflict_rows.append(g)

        geo_conflicts = pd.concat(conflict_rows, ignore_index=True, sort=False) if conflict_rows else pd.DataFrame()
        geo_conf_out = os.path.join(qa_dir, f"{today_str}_fsis_geo_conflicts.csv")
        geo_conflicts.to_csv(geo_conf_out, index=False)
        print("Saved QA:", geo_conf_out)

        geo_best_id = _choose_best_geo_per_key_year(geo_df, key_col="establishment_id")
        geo_best_num = _choose_best_geo_per_key_year(geo_df, key_col="establishment_number")
        geo_undated_id = _choose_best_geo_undated(geo_df, key_col="establishment_id")
        geo_undated_num = _choose_best_geo_undated(geo_df, key_col="establishment_number")

        est_geo = est_year.copy()
        est_geo["row_id"] = np.arange(len(est_geo))
        for c in GEO_COLS:
            est_geo[c] = pd.NA
        est_geo["geo_source_primary"] = pd.NA
        est_geo["geo_source_family"] = pd.NA
        est_geo["geo_source_file"] = pd.NA
        est_geo["geo_fallback_from_year"] = pd.NA

        # Same-year fills
        est_geo = _fill_same_year_geo(est_geo, geo_best_id, key_col="establishment_id", label="id")
        est_geo = _fill_same_year_geo(est_geo, geo_best_num, key_col="establishment_number", label="number")

        # Prior-year fallback
        est_geo = _fill_prior_year_geo(est_geo, geo_best_id, key_col="establishment_id", label="id")
        est_geo = _fill_prior_year_geo(est_geo, geo_best_num, key_col="establishment_number", label="number")

        # Undated reference fallback (from wherever available)
        est_geo = _fill_undated_geo(est_geo, geo_undated_id, key_col="establishment_id", label="id")
        est_geo = _fill_undated_geo(est_geo, geo_undated_num, key_col="establishment_number", label="number")

        est_year = est_geo.drop(columns=["row_id"])

    # Save establishment-year outputs
    est_all_out = os.path.join(clean_dir, f"{today_str}_fsis_establishment_year_all.csv")
    est_year.to_csv(est_all_out, index=False)
    print("Saved:", est_all_out)

    est_slaughter = est_year.loc[est_year["slaughterhouse_present_year"] == 1].copy()
    est_slaughter_out = os.path.join(clean_dir, f"{today_str}_fsis_establishment_year_slaughterhouse.csv")
    est_slaughter.to_csv(est_slaughter_out, index=False)
    print("Saved:", est_slaughter_out)

    # Unmatched geo QA among slaughterhouse establishment-years
    unmatched_geo = est_slaughter.loc[
        est_slaughter["fips_code"].isna() & est_slaughter["county"].isna() & est_slaughter["state"].isna()
    ].copy()
    unmatched_geo_out = os.path.join(qa_dir, f"{today_str}_fsis_establishment_year_unmatched_geo.csv")
    unmatched_geo.to_csv(unmatched_geo_out, index=False)
    print("Saved QA:", unmatched_geo_out)

    # County-year presence panel
    county_src = est_slaughter.loc[est_slaughter["fips_code"].notna()].copy()
    county_src["fips_code"] = _normalize_fips(county_src["fips_code"])
    county_src = county_src.loc[county_src["fips_code"].notna()].copy()

    if county_src.empty:
        county_year = pd.DataFrame(
            columns=[
                "fips",
                "year",
                "slaughterhouse_present",
                "n_establishments",
                "n_meat_establishments",
                "n_poultry_establishments",
                "n_both_ops_establishments",
                "n_slaughter_only_establishments",
            ]
        )
    else:
        county_year = (
            county_src.groupby(["fips_code", "year"], as_index=False)
            .agg(
                n_establishments=("est_key", "nunique"),
                n_meat_establishments=("meat_slaughter_present_year", lambda s: int((s == 1).sum())),
                n_poultry_establishments=("poultry_slaughter_present_year", lambda s: int((s == 1).sum())),
                n_both_ops_establishments=(
                    "operation_category_year",
                    lambda s: int((s.astype("string") == "both_slaughter_and_processing").sum()),
                ),
                n_slaughter_only_establishments=(
                    "operation_category_year",
                    lambda s: int((s.astype("string") == "slaughter_only").sum()),
                ),
            )
            .rename(columns={"fips_code": "fips"})
        )
        county_year["slaughterhouse_present"] = 1
        county_year = county_year[
            [
                "fips",
                "year",
                "slaughterhouse_present",
                "n_establishments",
                "n_meat_establishments",
                "n_poultry_establishments",
                "n_both_ops_establishments",
                "n_slaughter_only_establishments",
            ]
        ]

    county_out = os.path.join(clean_dir, f"{today_str}_fsis_county_year_slaughterhouse_presence.csv")
    county_year.to_csv(county_out, index=False)
    print("Saved:", county_out)

    # Summary QA metrics
    inv_ok = inventory_df.loc[inventory_df["read_status"] == "ok"].copy()
    same_year_geo = est_slaughter["geo_source_primary"].astype("string").str.contains("same_year", na=False).sum()
    fallback_geo = est_slaughter["geo_source_primary"].astype("string").str.contains("prior_year_fallback", na=False).sum()
    undated_geo = est_slaughter["geo_source_primary"].astype("string").str.contains("undated_reference", na=False).sum()
    missing_geo = int(
        (
            est_slaughter["fips_code"].isna()
            & est_slaughter["county"].isna()
            & est_slaughter["state"].isna()
        ).sum()
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "n_files_total",
                "n_files_read_ok",
                "n_files_read_error",
                "n_establishment_year_all",
                "n_establishment_year_slaughterhouse",
                "n_est_year_processing_present",
                "n_est_year_both_ops",
                "n_est_year_processing_only",
                "n_est_year_other_or_unclear",
                "n_est_year_same_year_geo",
                "n_est_year_prior_fallback_geo",
                "n_est_year_undated_geo",
                "n_est_year_missing_geo",
                "n_county_year_rows",
            ],
            "value": [
                len(inventory_df),
                len(inv_ok),
                int((inventory_df["read_status"] == "error").sum()),
                len(est_year),
                len(est_slaughter),
                int((est_year["processing_present_year"] == 1).sum()),
                int((est_year["operation_category_year"].astype("string") == "both_slaughter_and_processing").sum()),
                int((est_year["operation_category_year"].astype("string") == "processing_only").sum()),
                int((est_year["operation_category_year"].astype("string") == "other_or_unclear").sum()),
                int(same_year_geo),
                int(fallback_geo),
                int(undated_geo),
                missing_geo,
                len(county_year),
            ],
        }
    )
    summary_out = os.path.join(qa_dir, f"{today_str}_fsis_summary_metrics.csv")
    summary.to_csv(summary_out, index=False)
    print("Saved QA:", summary_out)


if __name__ == "__main__":
    main()
