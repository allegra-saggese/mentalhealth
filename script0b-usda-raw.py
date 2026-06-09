#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script0b-usda-raw.py

Pull USDA NASS Census of Agriculture county-level CAFO data for all census years
(2002, 2007, 2012, 2017, 2022), build the CAFO operations-by-size panel, and save.

Pipeline:
  1. Load each census year from a cached raw API CSV if present; otherwise pull from
     the NASS QuickStats API and save immediately before any filtering.
  2. Harmonize columns and filter to analysis-relevant inventory-bin rows.
  3. Forward-fill census-year values to intervening years.
  4. Merge in FIPS county names (from script0a output *_fips_full*.csv).
  5. Classify operations into small/medium/large bins, run QA, impute suppressed bins.

Inputs:
  - Data/raw/usda/*_usda_{year}_api_cafo_ops_inventory_raw.csv  (cached per-year raw pulls)
  - Data/clean/*_fips_full*.csv                                  (from script0a)

Outputs (Data/clean/):
  - *_ag_annual_df.csv
  - *_cafo_annual_df.csv
  - *_cafo_ops_by_size_long.csv
  - *_cafo_ops_by_size_compact.csv
  - *_qa_*.csv  (QA diagnostics)

Supersedes: script0p-usda-2022-api-only.py
"""

import os
import sys
import time

repo = os.path.dirname(os.path.abspath(__file__))
if repo not in sys.path:
    sys.path.append(repo)

import functions
import packages

print("functions file:", functions.__file__)
print("packages file:",  packages.__file__)

from functions import *
from packages import *

inf  = os.path.join(db_data, "raw")
outf = os.path.join(db_data, "clean")


# =============================================================================
# PART 1 : API SETUP AND CACHE-OR-PULL HELPERS
# =============================================================================

DEFAULT_NASS_API_KEY = "30643212-7739-359A-B451-0EAD3D345DB9"
NASS_API_KEY = os.environ.get("USDA_NASS_API_KEY", DEFAULT_NASS_API_KEY)
if not NASS_API_KEY:
    raise RuntimeError("Missing USDA_NASS_API_KEY env var. Set it before running this script.")

NASS_BASE = "https://quickstats.nass.usda.gov/api/"

source_desc           = "CENSUS"
agg_level_desc        = "COUNTY"
sector_desc           = "ANIMALS & PRODUCTS"
commodity_desc_allow  = ["CATTLE", "CHICKENS", "HOGS"]
unit_desc_allow       = ["OPERATIONS"]
statisticcat_desc_allow = ["INVENTORY"]

# Domain-level splits for requests that exceed the NASS 50k-row cap.
commodity_domain_splits = {
    "CATTLE": [
        "INVENTORY",
        "INVENTORY OF CATTLE, INCL CALVES",
        "INVENTORY OF CATTLE, (EXCL COWS)",
        "INVENTORY OF BEEF COWS",
        "INVENTORY OF MILK COWS",
        "INVENTORY OF CATTLE ON FEED",
    ],
    "CHICKENS": ["INVENTORY"],
    "HOGS": ["INVENTORY OF HOGS", "INVENTORY OF BREEDING HOGS"],
}

# Canonical column shape used by the downstream processing code.
API_CORE_COLS = [
    "group_desc", "commodity_desc", "class_desc", "prodn_practice_desc", "util_practice_desc",
    "statisticcat_desc", "unit_desc", "short_desc", "domain_desc", "domaincat_desc",
    "agg_level_desc", "state_ansi", "state_fips_code", "state_alpha", "state_name",
    "asd_code", "asd_desc", "county_ansi", "county_code", "county_name", "location_desc",
    "year", "freq_desc", "begin_code", "end_code", "reference_period_desc", "value", "cv_",
]


def _harmonize_api_schema(df_api):
    """Reshape a raw API DataFrame to API_CORE_COLS, renaming cv (%) → cv_."""
    if df_api.empty:
        return df_api
    out = df_api.copy()
    out.columns = [c.lower().strip() for c in out.columns]
    if "cv (%)" in out.columns and "cv_" not in out.columns:
        out = out.rename(columns={"cv (%)": "cv_"})
    for c in API_CORE_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[API_CORE_COLS].copy()


def _filter_to_analysis_rows(df_raw):
    """Keep only commodity/unit/domaincat combinations that flow into the CAFO panel."""
    if df_raw.empty:
        return df_raw
    out = df_raw.copy()
    for c in ["domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc",
              "commodity_desc", "group_desc", "class_desc"]:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip().str.lower()

    class_keep_map = {
        "cattle":   {"incl calves", "(excl cows)", "cows, beef", "cows, milk", "calves",
                     "calves, veal", "ge 500 lbs", "heifers, ge 500 lbs, milk replacement"},
        "chickens": {"broilers", "layers", "layers & pullets", "pullets, replacement", "roosters"},
        "hogs":     {"all classes", "breeding"},
    }
    out = out[
        (out["commodity_desc"].isin(["cattle", "chickens", "hogs"]))
        & (out["unit_desc"].isin(["operations", "head"]))
        & (out["statisticcat_desc"].isin(["inventory", "operations"]))
        & (out["domaincat_desc"].str.startswith("inventory", na=False))
    ].copy()
    allowed_pairs = {(c, cl) for c, cls in class_keep_map.items() for cl in cls}
    pair_idx    = pd.MultiIndex.from_frame(out[["commodity_desc", "class_desc"]])
    allowed_idx = pd.MultiIndex.from_tuples(sorted(allowed_pairs))
    return out[pair_idx.isin(allowed_idx)].copy()


def _find_cached_raw(year, raw_dir):
    """Return path to the most recent cached ops-inventory raw CSV for this year, or None."""
    matches = sorted(glob.glob(os.path.join(raw_dir, f"*_usda_{year}_api_cafo_ops_inventory_raw*.csv")))
    return matches[-1] if matches else None


def _pull_raw_year(year, raw_dir):
    """Pull inventory-bin + total-ops rows from NASS API for one census year.
    Saves to raw_dir immediately before returning — never re-pulls if cache exists."""
    from datetime import date as _date
    today_str  = _date.today().strftime("%Y-%m-%d")
    rate_sleep = 0.4

    def _keep_inv(df):
        col = next((c for c in df.columns if c.upper() == "DOMAINCAT_DESC"), None)
        if col is None:
            return df
        return df[df[col].astype(str).str.strip().str.lower().str.startswith("inventory", na=False)]

    # --- inventory-bin rows (split large requests by domain) ---
    frames_bins = []
    for cmd in commodity_desc_allow:
        for unit in unit_desc_allow:
            for stat in statisticcat_desc_allow:
                time.sleep(rate_sleep)
                base_params = {
                    "key": NASS_API_KEY,
                    "source_desc": source_desc,
                    "agg_level_desc": agg_level_desc,
                    "sector_desc": sector_desc,
                    "commodity_desc": cmd,
                    "unit_desc": unit,
                    "statisticcat_desc": stat,
                    "year": year,
                }
                try:
                    count = functions.nass_get_counts(NASS_BASE, base_params)
                except RuntimeError as e:
                    print(f"  Count error {cmd} {year}: {e}"); continue
                if not count:
                    continue
                if count <= 50000:
                    try:
                        df_pull = functions.nass_get_data(NASS_BASE, base_params)
                        frames_bins.append(_keep_inv(df_pull))
                    except RuntimeError as e:
                        print(f"  Data error {cmd} {year}: {e}")
                    continue
                for dd in commodity_domain_splits.get(cmd, []):
                    time.sleep(rate_sleep)
                    dd_params = dict(base_params, domain_desc=dd)
                    try:
                        dd_count = functions.nass_get_counts(NASS_BASE, dd_params)
                        if not dd_count or dd_count > 50000:
                            continue
                        df_pull = functions.nass_get_data(NASS_BASE, dd_params)
                        frames_bins.append(_keep_inv(df_pull))
                    except RuntimeError as e:
                        print(f"  Split {dd} {year}: {e}")

    # --- total-ops rows (QA denominator; CATTLE excluded from unsplit requests above) ---
    frames_totals = []
    for cmd in ["CATTLE", "CHICKENS", "HOGS"]:
        time.sleep(rate_sleep)
        params = {
            "key": NASS_API_KEY,
            "source_desc": "CENSUS",
            "agg_level_desc": "COUNTY",
            "sector_desc": "ANIMALS & PRODUCTS",
            "commodity_desc": cmd,
            "unit_desc": "OPERATIONS",
            "statisticcat_desc": "INVENTORY",
            "domain_desc": "TOTAL",
            "year": year,
        }
        try:
            count = functions.nass_get_counts(NASS_BASE, params)
            if not count:
                continue
            df_pull = functions.nass_get_data(NASS_BASE, params)
            if not df_pull.empty:
                frames_totals.append(df_pull)
                print(f"  {year} {cmd} total-ops: {len(df_pull):,} rows")
        except RuntimeError as e:
            print(f"  Total-ops error {cmd} {year}: {e}")

    all_frames = frames_bins + frames_totals
    if not all_frames:
        raise RuntimeError(f"No data returned from NASS API for year {year}.")

    raw = pd.concat(all_frames, ignore_index=True)
    raw.columns = [c.lower().strip() for c in raw.columns]
    raw = raw.drop_duplicates()

    out_path = os.path.join(raw_dir, f"{today_str}_usda_{year}_api_cafo_ops_inventory_raw.csv")
    raw.to_csv(out_path, index=False)
    print(f"  {year}: saved {len(raw):,} raw rows → {os.path.basename(out_path)}")
    return raw


def _load_or_pull_year(year, raw_dir):
    """Load from cache if available; otherwise pull from API and cache."""
    cached = _find_cached_raw(year, raw_dir)
    if cached:
        print(f"  {year}: loading from cache — {os.path.basename(cached)}")
        df = pd.read_csv(cached, low_memory=False)
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    print(f"  {year}: no cache found — pulling from NASS API ...")
    return _pull_raw_year(year, raw_dir)


# ---- Head-count cache helpers ----

def _find_cached_heads(year, raw_dir):
    """Return path to cached HEAD TOTAL CSV for this year, or None."""
    matches = sorted(glob.glob(os.path.join(raw_dir, f"*_usda_{year}_api_heads_total_raw.csv")))
    return matches[-1] if matches else None


def _pull_heads_year(year, raw_dir):
    """Pull unit=HEAD, domain=TOTAL rows from NASS API for one census year and cache."""
    from datetime import date as _date
    today_str  = _date.today().strftime("%Y-%m-%d")
    rate_sleep = 0.4
    frames = []
    for cmd in ["CATTLE", "CHICKENS", "HOGS"]:
        time.sleep(rate_sleep)
        params = {
            "key": NASS_API_KEY,
            "source_desc": "CENSUS",
            "agg_level_desc": "COUNTY",
            "sector_desc": "ANIMALS & PRODUCTS",
            "commodity_desc": cmd,
            "unit_desc": "HEAD",
            "statisticcat_desc": "INVENTORY",
            "domain_desc": "TOTAL",
            "year": year,
        }
        try:
            count = functions.nass_get_counts(NASS_BASE, params)
            if not count:
                continue
            df_pull = functions.nass_get_data(NASS_BASE, params)
            if not df_pull.empty:
                frames.append(df_pull)
                print(f"  {year} {cmd} total-heads: {len(df_pull):,} rows")
        except RuntimeError as e:
            print(f"  Total-heads error {cmd} {year}: {e}")
    if not frames:
        print(f"  WARNING: no head-count rows for {year}")
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    raw.columns = [c.lower().strip() for c in raw.columns]
    raw = raw.drop_duplicates()
    out_path = os.path.join(raw_dir, f"{today_str}_usda_{year}_api_heads_total_raw.csv")
    raw.to_csv(out_path, index=False)
    print(f"  {year}: saved {len(raw):,} head-count rows → {os.path.basename(out_path)}")
    return raw


def _load_or_pull_heads(year, raw_dir):
    """Load head totals from cache if available; otherwise pull from API and cache."""
    cached = _find_cached_heads(year, raw_dir)
    if cached:
        print(f"  {year}: loading head totals from cache — {os.path.basename(cached)}")
        df = pd.read_csv(cached, low_memory=False)
        df.columns = [c.lower().strip() for c in df.columns]
        return df
    print(f"  {year}: no head-count cache — pulling from NASS API ...")
    return _pull_heads_year(year, raw_dir)


# ---- Canonical animal types: (commodity_desc, class_desc) → output commodity_desc label ----
# cattle = all cattle incl calves (superset); beef and dairy are subsets and intentionally overlap.
CANONICAL_ANIMAL_TYPES = {
    "cattle":   ("cattle",   "incl calves"),
    "beef":     ("cattle",   "cows, beef"),
    "dairy":    ("cattle",   "cows, milk"),
    "hogs":     ("hogs",     "all classes"),
    "chickens": ("chickens", "layers"),
}
# Reverse map for remapping after imputation
_ANIMAL_REMAP = {v: k for k, v in CANONICAL_ANIMAL_TYPES.items()}


# =============================================================================
# PART 2 : LOAD ALL CENSUS YEARS
# =============================================================================

ALL_CENSUS_YEARS = [2002, 2007, 2012, 2017, 2022]
agfolder = os.path.join(inf, "usda")
os.makedirs(diagnostic_dir, exist_ok=True)
print("Loading NASS CAFO data for all census years:")
raw_frames = []
for _yr in ALL_CENSUS_YEARS:
    raw_frames.append(_load_or_pull_year(_yr, agfolder))
combined_raw = pd.concat(raw_frames, ignore_index=True)
print(f"Combined raw rows across {ALL_CENSUS_YEARS}: {len(combined_raw):,}")

# Normalize string cols before splitting bins vs. totals
for _c in ["domain_desc", "domaincat_desc", "unit_desc", "statisticcat_desc", "commodity_desc"]:
    if _c in combined_raw.columns:
        combined_raw[_c] = combined_raw[_c].astype("string").str.strip().str.lower()

# Extract total-ops rows (QA denominator) before the inventory-only filter removes them
combined_api_totals = combined_raw[
    (combined_raw["domain_desc"] == "total")
    & (combined_raw["domaincat_desc"] == "not specified")
    & (combined_raw["unit_desc"] == "operations")
].copy()
print(f"  {len(combined_api_totals):,} total-ops QA rows extracted")

# Harmonize to canonical column shape and filter to inventory-bin rows
_combined_h = _harmonize_api_schema(combined_raw)
combined    = _filter_to_analysis_rows(_combined_h)
combined    = combined.drop_duplicates()
print(f"  {len(combined):,} inventory-bin rows after harmonize + filter + dedup")

combined = combined.loc[:, ~combined.columns.duplicated()].copy()
print(f"Proceeding with API-sourced data for all census years {ALL_CENSUS_YEARS}.")

# ---- Load head-count totals (HEAD, domain=TOTAL) for county-level density measure ----
print("Loading head-count totals for all census years:")
_head_raw_frames = [_load_or_pull_heads(_yr, agfolder) for _yr in ALL_CENSUS_YEARS]
_head_raw_frames = [f for f in _head_raw_frames if not f.empty]
if _head_raw_frames:
    _heads_combined = pd.concat(_head_raw_frames, ignore_index=True)
    _heads_combined.columns = [c.lower().strip() for c in _heads_combined.columns]
    for _c in ["commodity_desc", "class_desc"]:
        if _c in _heads_combined.columns:
            _heads_combined[_c] = _heads_combined[_c].astype("string").str.strip().str.lower()
    _heads_combined = generate_fips(_heads_combined, state_col="state_fips_code", city_col="county_code")
    if "FIPS_generated" in _heads_combined.columns and "fips_generated" not in _heads_combined.columns:
        _heads_combined = _heads_combined.rename(columns={"FIPS_generated": "fips_generated"})
    elif "FIPS_generated" in _heads_combined.columns and "fips_generated" in _heads_combined.columns:
        _heads_combined = _heads_combined.drop(columns=["FIPS_generated"])
    _heads_combined["fips_generated"] = _heads_combined["fips_generated"].astype("string").str.zfill(5)
    _heads_combined["year"] = pd.to_numeric(_heads_combined["year"], errors="coerce").astype("Int64")
    _heads_combined["total_heads"] = pd.to_numeric(
        _heads_combined["value"].astype("string")
        .str.replace(",", "", regex=False).str.strip()
        .replace({"(d)": pd.NA, "(z)": pd.NA, "": pd.NA}),
        errors="coerce",
    )
    heads_agg = (
        _heads_combined[_heads_combined["commodity_desc"].isin(["cattle", "chickens", "hogs"])]
        .groupby(["fips_generated", "year", "commodity_desc"], as_index=False)["total_heads"]
        .sum(min_count=1)
    )
    print(f"  head-count totals: {len(heads_agg):,} rows (fips × year × commodity)")
else:
    heads_agg = pd.DataFrame(columns=["fips_generated", "year", "commodity_desc", "total_heads"])
    print("  WARNING: no head-count data loaded; total_heads will be NA in compact output")

# Quick eyeball QA: row counts by year and unit
chk = combined.copy()
for c in ["commodity_desc", "unit_desc"]:
    chk[c] = chk[c].astype("string").str.strip().str.lower()
chk["year"] = pd.to_numeric(chk["year"], errors="coerce").astype("Int64")
chk = chk[
    chk["commodity_desc"].isin(["cattle", "chickens", "hogs"]) &
    chk["year"].isin(ALL_CENSUS_YEARS)
]
print(chk.groupby(["year", "unit_desc"]).size().unstack(fill_value=0))
print(combined["year"].value_counts().sort_index().tail(10))


# =============================================================================
# PART 3 : QA AND COLUMN CHECK
# =============================================================================

print("Columns:", sorted(combined.columns))
print(combined.dtypes)

ag_iterated = clean_cols(combined).copy()
df = ag_iterated.copy()

for c in ["domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc",
          "commodity_desc", "group_desc", "class_desc"]:
    df[c] = df[c].astype("string").str.strip().str.lower()

comms_of_interest = ["cattle", "chickens", "hogs"]
class_keep_map = {
    "cattle":   {"incl calves","(excl cows)","cows, beef","cows, milk","calves","calves, veal",
                 "ge 500 lbs","heifers, ge 500 lbs, milk replacement"},
    "chickens": {"broilers","layers","layers & pullets","pullets, replacement","roosters"},
    "hogs":     {"all classes","breeding"},
}
df_sub = df[
    (df["commodity_desc"].isin(comms_of_interest)) &
    (df["unit_desc"].isin(["operations", "head"])) &
    (df["statisticcat_desc"].isin(["inventory", "operations"])) &
    (df["domaincat_desc"].str.startswith("inventory", na=False))
].copy()
allowed_pairs = {(commodity, cls) for commodity, classes in class_keep_map.items() for cls in classes}
pair_index    = pd.MultiIndex.from_frame(df_sub[["commodity_desc", "class_desc"]])
allowed_index = pd.MultiIndex.from_tuples(sorted(allowed_pairs))
df_sub = df_sub[pair_index.isin(allowed_index)].copy()
print("df_sub rows:", len(df_sub))


# =============================================================================
# PART 4 : MERGE FIPS COUNTY NAMES
# =============================================================================

matches = (
    glob.glob(os.path.join(outf, "*FIPS_key*.csv")) or
    glob.glob(os.path.join(outf, "*fips_full*.csv"))
)
if matches:
    fips_sense = max(matches, key=os.path.getmtime)
    print("Using FIPS file:", fips_sense)
else:
    raise RuntimeError("No FIPS key CSV found in clean folder. Run script0a first.")

fips_df = pd.read_csv(fips_sense)
fips_df = clean_cols(fips_df)
fips_df["fips"] = pd.to_numeric(fips_df["fips"], errors="coerce").astype("Int64").astype("string").str.zfill(5)
fips_df["year"] = pd.to_numeric(fips_df["year"], errors="coerce").astype("Int64")
fips_key = (
    fips_df[["fips", "year", "county"]]
    .rename(columns={"county": "county_fips_name"})
    .drop_duplicates()
)
dupe_key = fips_key.duplicated(subset=["fips", "year"]).sum()
if dupe_key:
    raise RuntimeError(f"fips_full has duplicate fips-year keys: {dupe_key}")


# =============================================================================
# PART 5 : FORWARD-FILL CENSUS YEARS TO INTERVENING YEARS
# =============================================================================

ag_raw_df = clean_cols(combined)
ag_raw_df = generate_fips(ag_raw_df, state_col="state_fips_code", city_col="county_code")

if "FIPS_generated" in ag_raw_df.columns and "fips_generated" not in ag_raw_df.columns:
    ag_raw_df = ag_raw_df.rename(columns={"FIPS_generated": "fips_generated"})
elif "FIPS_generated" in ag_raw_df.columns and "fips_generated" in ag_raw_df.columns:
    ag_raw_df = ag_raw_df.drop(columns=["FIPS_generated"])

if ag_raw_df.columns.duplicated().any():
    ag_raw_df = ag_raw_df.loc[:, ~ag_raw_df.columns.duplicated()]

if "fips_generated" not in ag_raw_df.columns:
    raise RuntimeError("fips_generated column missing after FIPS generation/normalization")

ag_raw_df["fips_generated"] = ag_raw_df["fips_generated"].astype("string").str.zfill(5)
ag_raw_df["year"] = pd.to_numeric(ag_raw_df["year"], errors="coerce").astype("Int64")

base_years = [2002, 2007, 2012, 2017]
n_forward  = 4
year_col   = "year"

new_frames = []
for b in base_years:
    base = ag_raw_df[ag_raw_df[year_col] == b].copy()
    if base.empty:
        continue
    for y in range(b + 1, b + 1 + n_forward):
        new_frames.append(base.assign(**{year_col: y}))

new_rows = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame(columns=ag_raw_df.columns)
df_big   = pd.concat([ag_raw_df, new_rows], ignore_index=True)

len_df_big_predupe = len(df_big)
len_it_rows        = len(new_rows)
df_big             = df_big.drop_duplicates(ignore_index=True)
len_raw_df         = len(ag_raw_df)
len_df_big_post_dupe = len(df_big)

print("No duplicates found in final dataframe? ",
      (len_df_big_post_dupe == len_df_big_predupe == (len_it_rows + len_raw_df)))

if "_merge" in df_big.columns:
    df_big = df_big.drop(columns=["_merge"])

df_big = df_big.merge(
    fips_key,
    left_on=["fips_generated", "year"],
    right_on=["fips", "year"],
    how="left",
    validate="m:1",
    indicator=True,
)
print("FIPS merge status:")
print(df_big["_merge"].value_counts(dropna=False))


def _norm_county_name(s):
    s = s.astype("string").str.lower().str.strip()
    s = s.str.replace(r"[^a-z0-9 ]", "", regex=True)
    s = s.str.replace(r"\b(county|parish|borough|census area|municipality|city and borough)\b", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


df_big["county_name_norm"]      = _norm_county_name(df_big["county_name"])
df_big["county_fips_name_norm"] = _norm_county_name(df_big["county_fips_name"])

name_mismatch   = df_big[
    df_big["county_name_norm"].notna()
    & df_big["county_fips_name_norm"].notna()
    & (df_big["county_name_norm"] != df_big["county_fips_name_norm"])
].copy()
missing_fips_key = df_big[df_big["_merge"] != "both"].copy()

mismatch_path    = os.path.join(diagnostic_dir, f"{today_str}_ag_fips_name_mismatch.csv")
missing_key_path = os.path.join(diagnostic_dir, f"{today_str}_ag_fips_missing_key.csv")
name_mismatch.to_csv(mismatch_path, index=False)
missing_fips_key.to_csv(missing_key_path, index=False)
print("Saved county-name mismatch rows for manual review:", mismatch_path)
print("Saved missing fips-year key rows for manual review:", missing_key_path)


# =============================================================================
# PART 6 : DEFINE CAFO SIZE-CLASS VARIABLES
# =============================================================================

df_cafo = df_big.copy()
for c in ["domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc",
          "commodity_desc", "group_desc", "class_desc"]:
    if c in df_cafo.columns:
        df_cafo[c] = df_cafo[c].astype("string").str.strip().str.lower()

comms_of_interest = ["cattle", "chickens", "hogs"]

# ---- Extract NASS-reported total operations (QA denominator) ----
# domain_desc="total", domaincat_desc="not specified" rows are stripped from df_cafo
# by the inventory-domaincat filter. We use combined_api_totals (extracted before filtering).
CENSUS_YEARS = {2002, 2007, 2012, 2017, 2022}

_total_canonical_classes = {
    "cattle":   {"incl calves", "cows, beef", "cows, milk"},
    "chickens": {"broilers", "layers"},
    "hogs":     {"all classes"},
}
_total_canonical_pairs = pd.MultiIndex.from_tuples(
    [(c, cl) for c, cls in _total_canonical_classes.items() for cl in cls]
)

df_nass_totals = df_cafo[
    (df_cafo["commodity_desc"].isin(comms_of_interest))
    & (df_cafo["unit_desc"] == "operations")
    & (df_cafo["statisticcat_desc"] == "inventory")
    & (df_cafo["domain_desc"] == "total")
    & (df_cafo["domaincat_desc"] == "not specified")
    & (df_cafo["year"].isin(CENSUS_YEARS))
].copy()
df_nass_totals = df_nass_totals[
    pd.MultiIndex.from_frame(df_nass_totals[["commodity_desc", "class_desc"]]).isin(_total_canonical_pairs)
].copy()

# Append API totals for all census years (stripped from df_cafo by the inventory-domaincat filter)
if not combined_api_totals.empty:
    _api_tot = combined_api_totals.copy()
    for _c in ["domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc", "commodity_desc", "class_desc"]:
        if _c in _api_tot.columns:
            _api_tot[_c] = _api_tot[_c].astype("string").str.strip().str.lower()
    _api_tot = _api_tot[
        (_api_tot["unit_desc"] == "operations")
        & (_api_tot["statisticcat_desc"] == "inventory")
        & (_api_tot["domain_desc"] == "total")
        & (_api_tot["domaincat_desc"] == "not specified")
    ].copy()
    if not _api_tot.empty:
        _api_tot = generate_fips(_api_tot, state_col="state_fips_code", city_col="county_code")
        if "FIPS_generated" in _api_tot.columns and "fips_generated" not in _api_tot.columns:
            _api_tot = _api_tot.rename(columns={"FIPS_generated": "fips_generated"})
        _api_tot["fips_generated"] = _api_tot["fips_generated"].astype("string").str.zfill(5)
        _api_tot["year"] = pd.to_numeric(_api_tot["year"], errors="coerce").astype("Int64")
        _api_tot = _api_tot[
            pd.MultiIndex.from_frame(_api_tot[["commodity_desc", "class_desc"]]).isin(_total_canonical_pairs)
        ].copy()
        df_nass_totals = pd.concat([df_nass_totals, _api_tot], ignore_index=True)

df_nass_totals["nass_total_ops"] = pd.to_numeric(
    df_nass_totals["value"].astype("string")
    .str.replace(",", "", regex=False).str.strip()
    .replace({"(d)": pd.NA, "(z)": pd.NA, "": pd.NA}),
    errors="coerce",
)
nass_totals_agg = (
    df_nass_totals
    .groupby(["fips_generated", "year", "commodity_desc", "class_desc"], as_index=False)["nass_total_ops"]
    .sum(min_count=1)
)
print(f"NASS total-ops QA rows: {len(nass_totals_agg)} "
      f"| years: {sorted(df_nass_totals['year'].dropna().unique().tolist())}")

# ---- Inventory-bin mappings (label text descriptions numerically) ----
class_keep_map = {
    "cattle":   {"incl calves","(excl cows)","cows, beef","cows, milk","calves","calves, veal",
                 "ge 500 lbs","heifers, ge 500 lbs, milk replacement"},
    "chickens": {"broilers","layers","layers & pullets","pullets, replacement","roosters"},
    "hogs":     {"all classes","breeding"},
}

df_sub = df_cafo[
    (df_cafo["commodity_desc"].isin(comms_of_interest))
    & (df_cafo["unit_desc"].isin(["operations", "head"]))
    & (df_cafo["statisticcat_desc"].isin(["inventory", "operations"]))
    & (df_cafo["domaincat_desc"].str.startswith("inventory", na=False))
].copy()

allowed_pairs = {(commodity, cls) for commodity, classes in class_keep_map.items() for cls in classes}
pair_index    = pd.MultiIndex.from_frame(df_sub[["commodity_desc", "class_desc"]])
allowed_index = pd.MultiIndex.from_tuples(sorted(allowed_pairs))
df_sub = df_sub[pair_index.isin(allowed_index)].copy()

print("Stage 2 df_sub rows:", len(df_sub))
print("Stage 2 commodity mix:")
print(df_sub["commodity_desc"].value_counts(dropna=False))
print("Stage 2 unit mix:")
print(df_sub["unit_desc"].value_counts(dropna=False))

layer_map = {
    "inventory: (1 to 49 head)": 1,
    "inventory: (50 to 99 head)": 2,
    "inventory: (100 to 399 head)": 3,
    "inventory: (400 to 3,199 head)": 4,
    "inventory: (3,200 to 9,999 head)": 5,
    "inventory: (10,000 to 19,999 head)": 6,
    "inventory: (20,000 to 49,999 head)": 7,
    "inventory: (50,000 to 99,999 head)": 8,
    "inventory: (100,000 or more head)": 9,
}
cattle_inv_map = {
    "inventory of cattle, incl calves: (1 to 9 head)": 1,
    "inventory of cattle, incl calves: (10 to 19 head)": 2,
    "inventory of cattle, incl calves: (20 to 49 head)": 3,
    "inventory of cattle, incl calves: (50 to 99 head)": 4,
    "inventory of cattle, incl calves: (100 to 199 head)": 5,
    "inventory of cattle, incl calves: (200 to 499 head)": 6,
    "inventory of cattle, incl calves: (500 or more head)": 7,
}
hog_inv_map = {
    "inventory of hogs: (1 to 24 head)": 1,
    "inventory of hogs: (25 to 49 head)": 2,
    "inventory of hogs: (50 to 99 head)": 3,
    "inventory of hogs: (100 to 199 head)": 4,
    "inventory of hogs: (200 to 499 head)": 5,
    "inventory of hogs: (500 to 999 head)": 6,
    "inventory of hogs: (1,000 or more head)": 7,
}
milk_cows_map = {
    "inventory of milk cows: (1 to 9 head)": 1,
    "inventory of milk cows: (10 to 19 head)": 2,
    "inventory of milk cows: (20 to 49 head)": 3,
    "inventory of milk cows: (50 to 99 head)": 4,
    "inventory of milk cows: (100 to 199 head)": 5,
    "inventory of milk cows: (200 to 499 head)": 6,
    "inventory of milk cows: (500 or more head)": 7,
}
breeding_hogs_map = {
    "inventory of breeding hogs: (1 to 24 head)": 1,
    "inventory of breeding hogs: (25 to 49 head)": 2,
    "inventory of breeding hogs: (50 to 99 head)": 3,
    "inventory of breeding hogs: (100 or more head)": 4,
}
cattle_inv_map_no_cows = {
    "inventory of cattle, (excl cows): (1 to 9 head)": 1,
    "inventory of cattle, (excl cows): (10 to 19 head)": 2,
    "inventory of cattle, (excl cows): (100 to 199 head)": 3,
    "inventory of cattle, (excl cows): (20 to 49 head)": 4,
    "inventory of cattle, (excl cows): (200 to 499 head)": 5,
    "inventory of cattle, (excl cows): (50 to 99 head)": 6,
}
cattle_feed_map = {
    "inventory of cattle on feed: (1 to 19 head)": 1,
    "inventory of cattle on feed: (1 to 9 head)": 2,
    "inventory of cattle on feed: (10 to 19 head)": 3,
    "inventory of cattle on feed: (100 to 199 head)": 4,
    "inventory of cattle on feed: (20 to 49 head)": 5,
    "inventory of cattle on feed: (200 to 499 head)": 6,
    "inventory of cattle on feed: (50 to 99 head)": 7,
    "inventory of cattle on feed: (500 or more head)": 8,
}
beef_cows_map = {
    "inventory of beef cows: (1 to 9 head)": 1,
    "inventory of beef cows: (10 to 19 head)": 2,
    "inventory of beef cows: (20 to 49 head)": 3,
    "inventory of beef cows: (50 to 99 head)": 4,
    "inventory of beef cows: (100 to 199 head)": 5,
    "inventory of beef cows: (200 to 499 head)": 6,
    "inventory of beef cows: (500 or more head)": 7,
}

map_size_class(df_sub, layer_map,   unit_match="operations", class_match="layers",   out_col="layer_ops_size")
map_size_class(df_sub, layer_map,   unit_match="operations", class_match="broilers", out_col="broiler_ops_size")
map_size(df_sub, cattle_inv_map,          unit_match="operations", out_col="cattle_ops_size_inv")
map_size(df_sub, hog_inv_map,             unit_match="operations", out_col="hog_ops_size_inv")
map_size(df_sub, milk_cows_map,           unit_match="operations", out_col="dairy_ops_size_inv")
map_size(df_sub, breeding_hogs_map,       unit_match="operations", out_col="breed_hog_ops_size_inv")
map_size(df_sub, cattle_inv_map_no_cows,  unit_match="operations", out_col="cattle_senzcow_ops_size_inv")
map_size(df_sub, cattle_feed_map,         unit_match="operations", out_col="cattle_feed_ops_size_inv")
map_size(df_sub, beef_cows_map,           unit_match="operations", out_col="beef_ops_size_inv")

broiler_cutoff_lrg = 5;  broiler_cutoff_med = 3
layer_cutoff_lrg   = 9;  layer_cutoff_med   = 7
cattle_cutoff_lrg  = 7;  cattle_cutoff_med  = 6
hog_cutoff_lrg     = 7;  hog_cutoff_med     = 6

col_thresholds = {
    "layer_ops_size":              (layer_cutoff_med,  layer_cutoff_lrg),
    "broiler_ops_size":            (broiler_cutoff_med, broiler_cutoff_lrg),
    "cattle_ops_size_inv":         (cattle_cutoff_med, cattle_cutoff_lrg),
    "dairy_ops_size_inv":          (cattle_cutoff_med, cattle_cutoff_lrg),
    "cattle_senzcow_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "cattle_feed_ops_size_inv":    (cattle_cutoff_med, cattle_cutoff_lrg),
    "beef_ops_size_inv":           (cattle_cutoff_med, cattle_cutoff_lrg),
    "hog_ops_size_inv":            (hog_cutoff_med,   hog_cutoff_lrg),
    "breed_hog_ops_size_inv":      (hog_cutoff_med,   hog_cutoff_lrg),
}


def categorize_code(v, med, lrg):
    if pd.isna(v):
        return pd.NA
    if v < med:
        return "small"
    if v < lrg:
        return "medium"
    return "large"


# =============================================================================
# PART 7 : CAFO SIZE CLASSIFICATION AND COMPACT SUMMARY
# =============================================================================

df2 = df_sub.copy()
df2["size_class"]  = pd.Series(pd.NA, index=df2.index, dtype="string")
df2["size_source"] = pd.Series(pd.NA, index=df2.index, dtype="string")

for col, (med, lrg) in col_thresholds.items():
    codes = df2[col]
    take  = codes.notna() & df2["size_source"].isna()
    df2.loc[take, "size_class"]  = codes[take].apply(categorize_code, args=(med, lrg))
    df2.loc[take, "size_source"] = col

df2["ops_in_bin"] = pd.to_numeric(
    df2["value"].astype("string")
    .str.replace(",", "", regex=False).str.strip()
    .replace({"(d)": pd.NA, "(z)": pd.NA, "": pd.NA}),
    errors="coerce",
)
df2["is_large_cafo_row"] = ((df2["size_class"] == "large") & df2["size_source"].notna()).astype("Int8")
df2["is_medium_or_large_cafo_row"] = (
    df2["size_class"].isin(["medium", "large"]) & df2["size_source"].notna()
).astype("Int8")

_summary_src = df2[df2["size_source"].notna()]
summary = (
    _summary_src.groupby(
        ["year", "fips_generated", "county_fips_name", "commodity_desc", "class_desc",
         "statisticcat_desc", "size_source", "size_class"],
        as_index=False,
    ).agg(sum_ops=("ops_in_bin", lambda x: x.sum(min_count=1)))
)

_PIVOT_IDX = ["year", "fips_generated", "county_fips_name", "commodity_desc", "class_desc"]
summary_compact = (
    summary.pivot_table(
        index=_PIVOT_IDX, columns="size_class", values="sum_ops",
        aggfunc="sum", fill_value=0,
    )
    .reset_index()
)
summary_compact.columns.name = None
for size_col in ["small", "medium", "large"]:
    if size_col not in summary_compact.columns:
        summary_compact[size_col] = 0


# =============================================================================
# PART 8 : QA — CATTLE CLASS OVERLAP
# =============================================================================

cattle_compact = summary_compact[summary_compact["commodity_desc"] == "cattle"].copy()
if not cattle_compact.empty:
    cattle_compact["ops_total"] = cattle_compact[["small", "medium", "large"]].sum(axis=1, min_count=1)
    cattle_wide = (
        cattle_compact.pivot_table(
            index=["year", "fips_generated", "county_fips_name"],
            columns="class_desc", values="ops_total", aggfunc="sum",
        ).reset_index()
    )
    cattle_wide.columns.name = None

    canonical_col            = "incl calves"
    noncanonical_cols        = [c for c in cattle_wide.columns if c not in {"year", "fips_generated", "county_fips_name", canonical_col}]
    partition_candidate_cols = [c for c in ["(excl cows)", "cows, beef", "cows, milk"] if c in cattle_wide.columns]

    cattle_wide["canonical_ops_incl_calves"]    = pd.to_numeric(cattle_wide.get(canonical_col), errors="coerce")
    cattle_wide["sum_noncanonical_ops"]          = (cattle_wide[noncanonical_cols].sum(axis=1, min_count=1) if noncanonical_cols else np.nan)
    cattle_wide["sum_partition_candidate_ops"]   = (cattle_wide[partition_candidate_cols].sum(axis=1, min_count=1) if partition_candidate_cols else np.nan)
    cattle_wide["sum_all_class_ops"]             = cattle_wide["canonical_ops_incl_calves"] + cattle_wide["sum_noncanonical_ops"]

    for lhs in ["sum_noncanonical_ops", "sum_partition_candidate_ops", "sum_all_class_ops"]:
        ratio_col = f"ratio_{lhs}_to_canonical"
        diff_col  = f"abs_pct_diff_{lhs}_vs_canonical"
        cattle_wide[ratio_col] = np.where(
            cattle_wide["canonical_ops_incl_calves"] > 0,
            cattle_wide[lhs] / cattle_wide["canonical_ops_incl_calves"], np.nan,
        )
        cattle_wide[diff_col] = np.where(
            cattle_wide["canonical_ops_incl_calves"] > 0,
            (cattle_wide[lhs] - cattle_wide["canonical_ops_incl_calves"]).abs() / cattle_wide["canonical_ops_incl_calves"] * 100, np.nan,
        )

    cattle_overlap_path = os.path.join(diagnostic_dir, f"{today_str}_qa_cattle_class_overlap_county_year.csv")
    cattle_wide.to_csv(cattle_overlap_path, index=False)

    year_diag = (
        cattle_wide.groupby("year", as_index=False).agg(
            county_years=("canonical_ops_incl_calves", "size"),
            canonical_sum=("canonical_ops_incl_calves", "sum"),
            partition_sum=("sum_partition_candidate_ops", "sum"),
            all_class_sum=("sum_all_class_ops", "sum"),
            median_ratio_partition_to_canonical=("ratio_sum_partition_candidate_ops_to_canonical", "median"),
            median_ratio_all_to_canonical=("ratio_sum_all_class_ops_to_canonical", "median"),
        )
    )
    year_diag["ratio_partition_sum_to_canonical_sum"] = np.where(
        year_diag["canonical_sum"] > 0, year_diag["partition_sum"] / year_diag["canonical_sum"], np.nan,
    )
    year_diag["ratio_allclass_sum_to_canonical_sum"] = np.where(
        year_diag["canonical_sum"] > 0, year_diag["all_class_sum"] / year_diag["canonical_sum"], np.nan,
    )
    cattle_overlap_year_path = os.path.join(diagnostic_dir, f"{today_str}_qa_cattle_class_overlap_by_year.csv")
    year_diag.to_csv(cattle_overlap_year_path, index=False)
    print("Saved cattle overlap QA (county-year):", cattle_overlap_path)
    print("Saved cattle overlap QA (year-level):", cattle_overlap_year_path)
    print("Cattle all-class vs canonical (year-level ratio):")
    print(year_diag[["year", "ratio_allclass_sum_to_canonical_sum"]].to_string(index=False))


# =============================================================================
# PART 9 : QA — BIN COVERAGE VS. NASS TOTAL OPS
# =============================================================================

summary_compact["bin_sum"] = (
    summary_compact[["small", "medium", "large"]]
    .apply(pd.to_numeric, errors="coerce")
    .sum(axis=1, min_count=1)
)
qa_coverage = summary_compact[summary_compact["year"].isin(CENSUS_YEARS)].copy()
qa_coverage = qa_coverage.merge(nass_totals_agg, on=["fips_generated", "year", "commodity_desc", "class_desc"], how="left")
_nass_total_float = pd.to_numeric(qa_coverage["nass_total_ops"], errors="coerce")
qa_coverage["coverage_pct"] = np.where(
    _nass_total_float.notna() & (_nass_total_float > 0),
    qa_coverage["bin_sum"] / _nass_total_float * 100, np.nan,
)
qa_coverage["suppressed_ops_estimate"] = qa_coverage["nass_total_ops"] - qa_coverage["bin_sum"]

coverage_path = os.path.join(diagnostic_dir, f"{today_str}_qa_cafo_bin_coverage_vs_nass_total.csv")
qa_coverage.to_csv(coverage_path, index=False)

coverage_year = (
    qa_coverage.groupby(["year", "commodity_desc", "class_desc"], as_index=False).agg(
        n_counties=("fips_generated", "count"),
        bin_sum_total=("bin_sum", "sum"),
        nass_total_total=("nass_total_ops", "sum"),
        pct_counties_with_nass_total=("nass_total_ops", lambda x: x.notna().mean() * 100),
        median_coverage_pct=("coverage_pct", "median"),
    )
)
coverage_year["ratio_bin_to_nass"] = np.where(
    coverage_year["nass_total_total"] > 0,
    coverage_year["bin_sum_total"] / coverage_year["nass_total_total"], np.nan,
)
coverage_year_path = os.path.join(diagnostic_dir, f"{today_str}_qa_cafo_bin_coverage_by_year_commodity.csv")
coverage_year.to_csv(coverage_year_path, index=False)
print("Saved bin coverage QA (county-year):", coverage_path)
print("Saved bin coverage QA (year-level):", coverage_year_path)
print("\nBin coverage ratio (bin_sum / nass_total) by year and commodity:")
print(coverage_year[["year", "commodity_desc", "class_desc", "median_coverage_pct", "ratio_bin_to_nass"]].to_string(index=False))


# =============================================================================
# PART 10 : SUPPRESSED-BIN IMPUTATION
# =============================================================================
# NASS suppresses operation-count bins by omitting rows (no (D) marker in ops data).
# gap = nass_total_ops - bin_sum > 0 means suppressed farms exist; we attribute them to large.
# Tiers (census years only): clean = gap>0 known; dark = nass_total_ops NA; none = no gap.

_imp_frames = []
for _animal, (_commodity, _canon_class) in CANONICAL_ANIMAL_TYPES.items():
    _sc = summary_compact[
        (summary_compact["commodity_desc"] == _commodity)
        & (summary_compact["class_desc"] == _canon_class)
        & (summary_compact["year"].isin(CENSUS_YEARS))
    ].copy()
    if _sc.empty:
        continue

    _sc = _sc.merge(
        nass_totals_agg[
            (nass_totals_agg["commodity_desc"] == _commodity)
            & (nass_totals_agg["class_desc"] == _canon_class)
        ][["fips_generated", "year", "nass_total_ops"]],
        on=["fips_generated", "year"],
        how="left",
    )
    for _col in ["small", "medium", "large"]:
        _sc[_col] = pd.to_numeric(_sc[_col], errors="coerce")
    _sc["nass_total_ops"] = pd.to_numeric(_sc["nass_total_ops"], errors="coerce")
    _sc["bin_sum_imp"]    = _sc[["small", "medium", "large"]].sum(axis=1, min_count=1)
    _sc["gap"]            = _sc["nass_total_ops"] - _sc["bin_sum_imp"]

    _gap_known    = _sc["nass_total_ops"].notna() & _sc["gap"].notna()
    _gap_positive = _gap_known & (_sc["gap"] > 0)

    _sc["imputation_tier"] = "none"
    _sc.loc[_sc["nass_total_ops"].isna(), "imputation_tier"] = "dark"
    _sc.loc[_gap_positive, "imputation_tier"] = "clean"

    _sc["large_imputed"]    = np.where(_sc["imputation_tier"] == "clean", _sc["large"] + _sc["gap"], _sc["large"])
    _sc["large_was_imputed"] = (_sc["imputation_tier"] == "clean").astype("Int8")
    _imp_frames.append(_sc[["fips_generated", "year", "commodity_desc", "class_desc",
                             "gap", "imputation_tier", "large_imputed", "large_was_imputed"]])

if _imp_frames:
    _imp_merge = pd.concat(_imp_frames, ignore_index=True)

    imp_path  = os.path.join(diagnostic_dir, f"{today_str}_qa_suppressed_bin_imputation.csv")
    _imp_merge.to_csv(imp_path, index=False)

    _tier_summary = (
        _imp_merge[_imp_merge["imputation_tier"] != "none"]
        .groupby(["year", "commodity_desc", "class_desc", "imputation_tier"], as_index=False)
        .agg(n_counties=("fips_generated", "count"))
    )
    tier_path = os.path.join(diagnostic_dir, f"{today_str}_qa_imputation_tier_summary.csv")
    _tier_summary.to_csv(tier_path, index=False)
    print("Saved suppression imputation QA:", imp_path)
    print("Saved imputation tier summary:", tier_path)
    print("\nImputation tier counts (census years, canonical classes only):")
    print(_tier_summary.to_string(index=False))

    summary_compact = summary_compact.merge(
        _imp_merge, on=["fips_generated", "year", "commodity_desc", "class_desc"], how="left",
    )
    summary_compact["large_imputed"]    = np.where(summary_compact["large_imputed"].isna(), summary_compact["large"], summary_compact["large_imputed"])
    summary_compact["large_was_imputed"] = summary_compact["large_was_imputed"].fillna(0).astype("Int8")
    summary_compact["imputation_tier"]  = summary_compact["imputation_tier"].fillna("none")

    _ff_mask = ~summary_compact["year"].isin(CENSUS_YEARS) & (summary_compact["large_was_imputed"] == 1)
    summary_compact.loc[_ff_mask, "large_imputed"]    = np.nan
    summary_compact.loc[_ff_mask, "large_was_imputed"] = pd.NA

    summary_compact = summary_compact.sort_values(["fips_generated", "commodity_desc", "class_desc", "year"]).reset_index(drop=True)
    _grp = ["fips_generated", "commodity_desc", "class_desc"]
    summary_compact["large_imputed"]    = summary_compact.groupby(_grp)["large_imputed"].transform("ffill")
    summary_compact["large_was_imputed"] = summary_compact.groupby(_grp)["large_was_imputed"].transform("ffill")
    summary_compact["large_was_imputed"] = summary_compact["large_was_imputed"].fillna(0).astype("Int8")
else:
    summary_compact["large_imputed"]    = summary_compact["large"]
    summary_compact["large_was_imputed"] = 0
    summary_compact["imputation_tier"]  = "none"
    print("No imputation applied (no canonical class rows found or no census years in data).")

print("Stage 2 mapped rows:", int(df2["size_source"].notna().sum()))
print("Stage 2 size class counts:")
print(df2["size_class"].value_counts(dropna=False))
print("Stage 2 compact rows:", len(summary_compact))
print(df2[df2["size_source"].notna()]["year"].value_counts().sort_index().tail(15))
print(summary["year"].min(), summary["year"].max())
print(summary_compact["year"].min(), summary_compact["year"].max())


# =============================================================================
# PART 11 : FINALIZE COMPACT — REMAP, DROP IMPUTATION COLS, MERGE HEADS
# =============================================================================
# Keep only the 5 canonical (commodity_desc, class_desc) pairs; remap commodity_desc
# to the canonical animal label; drop class_desc and imputation diagnostic columns.
# gap=0 confirmed for all census years/commodities — large already holds the correct count.

_canonical_pairs_set = set(CANONICAL_ANIMAL_TYPES.values())
_sc_mask = pd.MultiIndex.from_frame(summary_compact[["commodity_desc", "class_desc"]]).isin(
    pd.MultiIndex.from_tuples(sorted(_canonical_pairs_set))
)
summary_compact = summary_compact[_sc_mask].copy()

summary_compact["commodity_desc"] = [
    _ANIMAL_REMAP.get((r_comm, r_class), r_comm)
    for r_comm, r_class in zip(summary_compact["commodity_desc"], summary_compact["class_desc"])
]
summary_compact = summary_compact.drop(columns=["class_desc"])

_drop_cols = [c for c in ["large_imputed", "large_was_imputed", "imputation_tier", "gap"]
              if c in summary_compact.columns]
summary_compact = summary_compact.drop(columns=_drop_cols)

summary_compact["any_large_cafo"] = (
    pd.to_numeric(summary_compact["large"], errors="coerce") > 0
).astype("Int8")
summary_compact["any_medium_or_large_cafo"] = (
    (pd.to_numeric(summary_compact["medium"], errors="coerce")
     + pd.to_numeric(summary_compact["large"], errors="coerce")) > 0
).astype("Int8")

# Merge total_heads; beef and dairy use total cattle heads for their county/year
_heads_parent_map = {"cattle": "cattle", "beef": "cattle", "dairy": "cattle",
                     "hogs": "hogs", "chickens": "chickens"}
summary_compact["_commodity_parent"] = summary_compact["commodity_desc"].map(_heads_parent_map)
if not heads_agg.empty:
    summary_compact = summary_compact.merge(
        heads_agg.rename(columns={"commodity_desc": "_commodity_parent"}),
        on=["fips_generated", "year", "_commodity_parent"],
        how="left",
    )
    # Forward-fill total_heads from census years to intervening years
    summary_compact = summary_compact.sort_values(
        ["fips_generated", "commodity_desc", "year"]
    ).reset_index(drop=True)
    summary_compact["total_heads"] = (
        summary_compact.groupby(["fips_generated", "commodity_desc"])["total_heads"]
        .transform("ffill")
    )
else:
    summary_compact["total_heads"] = pd.NA
summary_compact = summary_compact.drop(columns=["_commodity_parent"])

print(f"Compact final: {len(summary_compact):,} rows × {summary_compact.shape[1]} cols")
print("Compact animal types:", sorted(summary_compact["commodity_desc"].dropna().unique().tolist()))


# =============================================================================
# PART 12 : EXPORT
# =============================================================================

cafo_compact_path = os.path.join(outf, f"{today_str}_cafo_ops_by_size_compact.csv")
summary_compact.to_csv(cafo_compact_path, index=False)
print("Saved CAFO compact summary:", cafo_compact_path)
