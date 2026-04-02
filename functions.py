#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:19:14 2025

@author: allegrasaggese
"""

# purpose: useful functions across scripts 

# load packages - think i can delete
import os
import re
import glob
import json
import pandas as pd
import urllib.error
from urllib.parse import urlencode
from urllib.request import urlopen, Request


##################### STANDARD DEFINITIONS ####################

# set today's date for exported materials 
today_str = date.today().strftime("%Y-%m-%d")


# 50 states + DC (exclude territories for US county panels) - use in script0a
US_STATE_CODES = {
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27",
    "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
    "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53",
    "54", "55", "56"
}

# Census 2024 agesex file uses year codes; this map fills 2021-2024.
YEAR_CODE_2024_MAP = {3: 2021, 4: 2022, 5: 2023, 6: 2024}

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




##################### HELPER FUNCTIONS ####################

# used in: script0a-pop-fips-raw-merge.py, script0c-health-raw.py, script0d-crime-raw.py,
# used in: script0f-nchs-urban.py, script0o-cdc-county-deathsofdespair-panel.py, script3-ridge.py
def read_csv_with_fallback(path, encodings=("utf-8", "latin1"), **kwargs):
    """
    Read CSV with sequential encoding fallbacks.
    If `encoding` is explicitly passed in kwargs, it is used as-is.
    """
    if "encoding" in kwargs:
        return pd.read_csv(path, **kwargs)

    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_error = e
    raise last_error if last_error else RuntimeError(f"Failed to read CSV: {path}")


# used in: script0a-pop-fips-raw-merge.py
def pick_source_by_fragment(source_map, name_fragment):
    """
    Pick one DataFrame from a dict keyed by filename using a name fragment.
    """
    fragment = str(name_fragment).lower()
    matches = [k for k in source_map.keys() if fragment in str(k).lower()]
    if not matches:
        raise FileNotFoundError(f"Source not found for pattern: {name_fragment}")
    if len(matches) > 1:
        print(f"Multiple files matched '{name_fragment}', using first: {matches[0]}")
    return source_map[matches[0]].copy()


# used in: script0a-pop-fips-raw-merge.py
def sniff_delimiter(path):
    """
    Return one of: 'pipe', 'comma', 'ws' based on the first non-empty line.
    """
    sample = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.strip():
                sample = line
                break
    if sample is None:
        raise ValueError(f"Could not sniff delimiter; file appears empty: {path}")

    scores = {
        "pipe": sample.count("|"),
        "comma": sample.count(","),
        "ws": len(re.split(r"\s+", sample.strip())) - 1,
    }
    return max(scores, key=scores.get)


# used in: script0a-pop-fips-raw-merge.py
def expand_years_cross(df, years, year_col="year"):
    """
    Cross-join a DataFrame with a list/range of years.
    """
    year_df = pd.DataFrame({year_col: list(years)})
    out = df.merge(year_df, how="cross")
    return out.reset_index(drop=True)


# used in: script0a-pop-fips-raw-merge.py
def drop_state_level_rows_except_dc(
    df,
    county_code_col="county_code",
    county_name_col="county",
    dc_label="District of Columbia",
):
    """
    Drop state-level rows where county_code == '000', keeping DC county-equivalent.
    """
    out = df.copy()
    county_code = out[county_code_col].astype("string").str.extract(r"(\d+)", expand=False).str.zfill(3)
    county_name = out[county_name_col].astype("string")
    keep_mask = ~((county_code == "000") & (county_name != dc_label))
    return out.loc[keep_mask].copy()


# used in: script0a-pop-fips-raw-merge.py
def standardize_county_identifiers(
    df,
    state_col="state_code",
    county_col="county_code",
    fips_col="fips",
    year_col="year",
    fips_as_string=False,
):
    """
    Standardize county key columns:
    - state_code -> 2-digit string
    - county_code -> 3-digit string
    - fips -> Int64 or zero-padded 5-char string
    - year -> Int64 (if present)
    """
    out = df.copy()
    out[state_col] = (
        out[state_col].astype("string").str.extract(r"(\d+)", expand=False).str.zfill(2)
    )
    out[county_col] = (
        out[county_col].astype("string").str.extract(r"(\d+)", expand=False).str.zfill(3)
    )

    fips_num = pd.to_numeric(out[fips_col], errors="coerce").astype("Int64")
    if fips_as_string:
        out[fips_col] = fips_num.astype("string").str.zfill(5)
    else:
        out[fips_col] = fips_num

    if year_col in out.columns:
        out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
    return out


# used in: script0a-pop-fips-raw-merge.py
def normalize_dc_fips(
    df,
    county_name_col="county",
    state_col="state_code",
    county_col="county_code",
    fips_col="fips",
    fips_value="11001",
):
    """
    Normalize District of Columbia to county-equivalent FIPS across all years.
    """
    out = df.copy()
    dc_mask = out[county_name_col].astype("string").str.contains("District of Columbia", case=False, na=False)
    out.loc[dc_mask, state_col] = "11"
    out.loc[dc_mask, county_col] = "001"

    if pd.api.types.is_numeric_dtype(out[fips_col]):
        out.loc[dc_mask, fips_col] = pd.to_numeric(fips_value, errors="coerce")
    else:
        out.loc[dc_mask, fips_col] = str(fips_value)
    return out


# used in: script0a-pop-fips-raw-merge.py
def filter_to_us_state_codes(df, state_col="state_code", allowed_state_codes=None):
    """
    Keep only rows whose state code is in US_STATE_CODES (50 states + DC by default).
    """
    allowed = allowed_state_codes if allowed_state_codes is not None else US_STATE_CODES
    out = df.copy()
    state_vals = out[state_col].astype("string").str.extract(r"(\d+)", expand=False).str.zfill(2)
    return out.loc[state_vals.isin(allowed)].copy()


# used in: script0c-health-raw.py, script0e-fsis-slaughterhouses.py, script0h-fsis-establishment-size-panel.py,
# used in: script0i-fsis-hud-zip-fips-fill.py, script0j-fsis-hud-zipyear-refill.py, script0k-fsis-hud-bulk-zipyear-fill.py,
# used in: script0l-fsis-apply-manual-zip-fips.py, script0m-fsis-deterministic-completion.py, script0n-fsis-apply-manual-county-fips.py,
# used in: script0o-cdc-county-deathsofdespair-panel.py
def first_non_null(series: pd.Series):
    """
    Return first non-null value from a Series, else pd.NA.
    """
    s = series.dropna()
    return s.iloc[0] if len(s) else pd.NA


# used in: script0i-fsis-hud-zip-fips-fill.py, script0j-fsis-hud-zipyear-refill.py, script0k-fsis-hud-bulk-zipyear-fill.py,
# used in: script0l-fsis-apply-manual-zip-fips.py, script0m-fsis-deterministic-completion.py
def normalize_zip5(series: pd.Series) -> pd.Series:
    """
    Extract 5-digit ZIP code from a string-like series.
    """
    s = series.astype("string").str.strip()
    return s.str.extract(r"(\d{5})", expand=False)


# used in: script0e-fsis-slaughterhouses.py, script0l-fsis-apply-manual-zip-fips.py,
# used in: script0m-fsis-deterministic-completion.py, script0n-fsis-apply-manual-county-fips.py
def normalize_fips5(series: pd.Series) -> pd.Series:
    """
    Normalize to 5-digit FIPS (string); invalid values become NA.
    """
    s = series.astype("string").str.strip().str.extract(r"(\d+)", expand=False)
    s = s.where(s.str.len().isin([4, 5]), pd.NA)
    s = s.where(s.str.len() != 4, "0" + s)
    return s


# used in: script0m-fsis-deterministic-completion.py, script0n-fsis-apply-manual-county-fips.py
def normalize_text_upper(series: pd.Series) -> pd.Series:
    """
    Upper-case trimmed text normalization.
    """
    return series.astype("string").str.upper().str.strip()


# used in: script0o-cdc-county-deathsofdespair-panel.py, script1a-QA-dataclean.py, script2-visualizations.py,
# used in: script2a-panel-sumstats-by-farms.py, script2b-qa-memo-correlations.py, script2c-fsis-sizebins-vs-mental-2017.py,
# used in: script2d-mental-coverage-audit.py
def to_numeric_series(series: pd.Series) -> pd.Series:
    """
    Robust numeric coercion for mixed strings/numeric columns.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    s = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "(d)": pd.NA, "(z)": pd.NA, "na": pd.NA, "n/a": pd.NA})
    )
    return pd.to_numeric(s, errors="coerce")


# used in: script1c-replace-cdc-in-merged.py, script2-visualizations.py, script2a-panel-sumstats-by-farms.py,
# used in: script2b-qa-memo-correlations.py, script2c-fsis-sizebins-vs-mental-2017.py, script2d-mental-coverage-audit.py
def latest_file_glob(folder: str, pattern: str) -> str:
    """
    Return latest path matching a glob pattern by modified time.
    """
    hits = glob.glob(os.path.join(folder, pattern))
    if not hits:
        raise FileNotFoundError(f"No files found for pattern {pattern} in {folder}")
    return max(hits, key=os.path.getmtime)


# used in: script0h-fsis-establishment-size-panel.py, script0i-fsis-hud-zip-fips-fill.py, script0l-fsis-apply-manual-zip-fips.py,
# used in: script0m-fsis-deterministic-completion.py, script0n-fsis-apply-manual-county-fips.py
def latest_file_by_regex(dirpath: str, pattern: str) -> str:
    """
    Return latest file matching regex pattern with a leading date capture group
    in YYYY-MM-DD format.
    """
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


# used in: latest_files_by_descriptor() helper (indirectly script1b-merge-dataclean.py)
def split_dated_descriptor(path: str):
    """
    Returns (descriptor, date_key) where descriptor excludes date prefix.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[-_](.+)$", stem)
    if m:
        return m.group(2), m.group(1)
    return stem, ""


# used in: script1b-merge-dataclean.py
def latest_files_by_descriptor(folder: str):
    """
    Find latest file per descriptor in a folder with dated file naming.
    """
    supported = ("*.csv", "*.parquet", "*.pq", "*.xlsx", "*.xls")
    files = []
    for pat in supported:
        files.extend(glob.glob(os.path.join(folder, pat)))

    latest = {}
    for path in files:
        descriptor, date_key = split_dated_descriptor(path)
        mtime = os.path.getmtime(path)
        score = (date_key, mtime)
        if descriptor not in latest or score > latest[descriptor]["score"]:
            latest[descriptor] = {"path": path, "score": score}
    return {k: v["path"] for k, v in latest.items()}


# used in: script1b-merge-dataclean.py, script1c-replace-cdc-in-merged.py, script2-visualizations.py,
# used in: script2a-panel-sumstats-by-farms.py, script2b-qa-memo-correlations.py, script2d-mental-coverage-audit.py
def normalize_panel_key(df: pd.DataFrame, dropna=True):
    """
    Normalize key columns for panel data to `fips` and `year`.
    """
    out = clean_cols(df.copy())
    if "fips" not in out.columns:
        if "fips_generated" in out.columns:
            out = out.rename(columns={"fips_generated": "fips"})
        elif "geoid" in out.columns:
            out = out.rename(columns={"geoid": "fips"})

    if "year" not in out.columns and "yr" in out.columns:
        out = out.rename(columns={"yr": "year"})

    if "fips" not in out.columns or "year" not in out.columns:
        raise KeyError("Missing key columns fips/year")

    out["fips"] = (
        out["fips"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out.dropna(subset=["fips", "year"]).copy() if dropna else out


# used in: nass_get_counts()/nass_get_data() below (indirectly script0b-ag-raw.py, script0b-ag-raw-v2.py)
def nass_request(base_url: str, endpoint: str, params: dict, user_agent="mentalhealth-ag-script/1.0"):
    """
    Generic USDA NASS API request.
    """
    query = urlencode(params)
    url = f"{base_url}{endpoint}/?{query}"
    req = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NASS API HTTP {e.code} for {url} :: {body}") from e


# used in: script0b-ag-raw.py, script0b-ag-raw-v2.py
def nass_get_counts(base_url: str, params: dict):
    payload = nass_request(base_url, "get_counts", params)
    return int(payload.get("count", 0))


# used in: script0b-ag-raw.py, script0b-ag-raw-v2.py
def nass_get_data(base_url: str, params: dict):
    payload = nass_request(base_url, "api_GET", params)
    data = payload.get("data", [])
    return pd.DataFrame(data)


# FIPS code generator from individual city - state cols 

# used in: script0a-pop-fips-raw-merge.py, script0b-ag-raw.py, script0b-ag-raw-v2.py, script0f-nchs-urban.py
def generate_fips(df, state_col="state", city_col="city"):
    # Pad state to 2 digits _ cities to 3 digits
    df["state_padded"] = df[state_col].astype(str).str.zfill(2)
    df["city_padded"] = df[city_col].astype(str).str.zfill(3)

    # combine cols
    df["FIPS_generated"] = df["state_padded"] + df["city_padded"]

    # QA - Check all are 5 digits
    invalid_fips = df[~df["FIPS_generated"].str.match(r"^\d{5}$")]
    if not invalid_fips.empty:
        print("Warning: Some FIPS codes are not 5 digits.")
        print(invalid_fips[["FIPS_generated", state_col, city_col]])
    else:
        print("All FIPS_generated values are 5 digits.")

    # drop temp columns
    df.drop(columns=["state_padded", "city_padded"], inplace=True)

    return df

# simple / standard colname cleaning

# used in: script0b-ag-raw.py, script0o-cdc-county-deathsofdespair-panel.py,
# used in: normalize_panel_key() helper used across script1/script2 files
def clean_cols(df):
    df.columns = (
        df.columns
        .str.lower()                      # lowercase
        .str.replace("&", "and")          # replace & with and
        .str.replace(r"[.,]", "", regex=True)  # remove . and ,
        .str.strip()                      # trim spaces
    )
    return df


# read in files 
# used in: script1b-merge-dataclean.py, script2a-panel-sumstats-by-farms.py
def read_and_prepare(path):
    """Simple reader: csv/parquet/xlsx -> lower-case cols, strip strings, standardize fips/year."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, dtype=str)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        raise ValueError("unsupported file type")

    # normalize column names and trim string columns
    df.columns = df.columns.str.lower().str.strip()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip()

    # rename common fips/year variants if present
    fips_candidates = [c for c in df.columns if c in ("fips","fips_generated","fips_code","geoid","county_fips")]
    year_candidates = [c for c in df.columns if c in ("year","yr")]

    if fips_candidates:
        df = df.rename(columns={fips_candidates[0]: "fips"})
        # normalize to zero-padded 5-char string where possible
        df["fips"] = df["fips"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(5)

    if year_candidates:
        df = df.rename(columns={year_candidates[0]: "year"})
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df



##### CAFO MAPPING FUNCTIONS ##### 

# make a wrapper to map the inventory the same way in all of them 
# used in: script0b-ag-raw.py
def map_size(df, mapping, unit_match, out_col):
    mask = df['unit_desc'] == unit_match
    df[out_col] = df['domaincat_desc'].map(mapping).where(mask, other=pd.NA).astype("Int64")


# used in: script0b-ag-raw.py, script0b-ag-raw-v2.py
def map_size_class(df, mapping, unit_match, class_match, out_col):
    """
    Map size bins with optional class-level restriction.
    """
    mask = df["unit_desc"] == unit_match
    if class_match is not None:
        mask = mask & (df["class_desc"] == class_match)
    df[out_col] = df["domaincat_desc"].map(mapping).where(mask, other=pd.NA).astype("Int64")
