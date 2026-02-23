#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 15:19:00 2025

@author: allegrasaggese
"""

# load packages and workspaces
import sys, importlib.util
from collections import Counter
import re
from functools import reduce
import json
import time
import urllib.error
from urllib.parse import urlencode
from urllib.request import urlopen, Request


# make sure repo root is on sys.path (parent of functions.py / packages/)
repo = "/Users/allegrasaggese/Documents/GitHub/mentalhealth" # change this so any user path should generate correctly 
if repo not in sys.path:
    sys.path.append(repo)

import functions       
import packages         

print("functions file:", functions.__file__)
print("packages file:",  packages.__file__) 

from functions import *     
from packages import *


# other folders
inf = os.path.join(db_data, "raw") # input 
outf = os.path.join(db_data, "clean") #output


# API setup
os.environ["USDA_NASS_API_KEY"] = "30643212-7739-359A-B451-0EAD3D345DB9" # will have to change for user
# Pull USDA NASS Quick Stats data directly instead of reading .dta files.
USE_API = True
NASS_API_KEY = os.environ.get("USDA_NASS_API_KEY") 
if USE_API and not NASS_API_KEY:
    raise RuntimeError("Missing USDA_NASS_API_KEY env var. Set it before running this script.")

NASS_BASE = "https://quickstats.nass.usda.gov/api/"


def _nass_request(endpoint, params):
    query = urlencode(params)
    url = f"{NASS_BASE}{endpoint}/?{query}"
    req = Request(url, headers={"User-Agent": "mentalhealth-ag-script/1.0"})
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NASS API HTTP {e.code} for {url} :: {body}") from e


def nass_get_counts(params):
    payload = _nass_request("get_counts", params)
    return int(payload.get("count", 0))


def nass_get_data(params):
    payload = _nass_request("api_GET", params)
    data = payload.get("data", [])
    return pd.DataFrame(data)


# Filters (user-specified)
source_desc = "CENSUS"
agg_level_desc = "COUNTY"
sector_desc = "ANIMALS & PRODUCTS"
commodity_desc_allow = ["CATTLE", "CHICKENS", "HOGS"]

unit_desc_allow = ["HEAD", "OPERATIONS"]
statisticcat_desc_allow = ["INVENTORY", "OPERATIONS"]

# Census years only
census_years = [2002, 2007, 2012, 2017]

# Commodity-specific domain splits used only when a query exceeds 50k rows.
commodity_domain_splits = {
    "CATTLE": [
        "INVENTORY",
        "INVENTORY OF CATTLE, INCL CALVES",
        "INVENTORY OF CATTLE, (EXCL COWS)",
        "INVENTORY OF BEEF COWS",
        "INVENTORY OF MILK COWS",
        "INVENTORY OF CATTLE ON FEED",
    ],
    "CHICKENS": [
        "INVENTORY",
    ],
    "HOGS": [
        "INVENTORY OF HOGS",
        "INVENTORY OF BREEDING HOGS",
    ],
}


def fetch_ag_data():
    def _keep_inventory_domaincat(df):
        """Keep only rows where domaincat_desc starts with 'inventory' (case-insensitive)."""
        if "domaincat_desc" in df.columns:
            s = df["domaincat_desc"].astype(str).str.strip().str.lower()
            return df[s.str.startswith("inventory", na=False)]
        if "DOMAINCAT_DESC" in df.columns:
            s = df["DOMAINCAT_DESC"].astype(str).str.strip().str.lower()
            return df[s.str.startswith("inventory", na=False)]
        return df

    def _safe_get_counts(params):
        try:
            return nass_get_counts(params)
        except RuntimeError as e:
            print("Skipping combo due to API error:", e)
            return None

    def _safe_get_data(params):
        try:
            out = nass_get_data(params)
        except RuntimeError as e:
            print("Skipping data pull due to API error:", e)
            return pd.DataFrame()
        return _keep_inventory_domaincat(out)

    frames = []
    rate_sleep = 0.4  # basic rate limiting

    for yr in census_years:
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
                        "year": yr,
                    }

                    count = _safe_get_counts(base_params)
                    if count is None or count == 0:
                        continue

                    if count <= 50000:
                        df_pull = _safe_get_data(base_params)
                        if not df_pull.empty:
                            frames.append(df_pull)
                        continue

                    # Still too large: split by commodity-specific domain_desc values.
                    for dd in commodity_domain_splits.get(cmd, []):
                        time.sleep(rate_sleep)
                        dd_params = dict(base_params)
                        dd_params["domain_desc"] = dd
                        dd_count = _safe_get_counts(dd_params)
                        if dd_count is None or dd_count == 0:
                            continue
                        if dd_count > 50000:
                            print(
                                "Skipping domain split still >50k rows: "
                                f"year={yr}, commodity={cmd}, "
                                f"unit={unit}, statisticcat={stat}, domain_desc={dd}, count={dd_count}"
                            )
                            continue
                        df_pull = _safe_get_data(dd_params)
                        if not df_pull.empty:
                            frames.append(df_pull)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# import ag data from API (fallback to local .dta files if API fails)
combined = pd.DataFrame()
if USE_API:
    try:
        combined = fetch_ag_data()
        print(f"Loaded {len(combined):,} rows from USDA NASS Quick Stats API")
    except RuntimeError as e:
        print("API fetch failed; falling back to local .dta files.")
        print("Reason:", e)

if combined.empty:
    agfolder = os.path.join(inf, "usda")
    agfiles = glob.glob(os.path.join(agfolder, "*.dta"))
    if not agfiles:
        raise RuntimeError(f"No .dta files found in {agfolder}")
    agdfs = [pd.read_stata(file) for file in agfiles]
    combined = pd.concat(agdfs, ignore_index=True)
    print(f"Loaded {len(combined):,} rows from local .dta files")


################## DATA CLEANING FOR ALL AG DATA ######################

# basic QA
print("Columns:", sorted(combined.columns))
print(combined.dtypes)
    
ag_iterated = clean_cols(combined).copy()
df = ag_iterated.copy()

for c in [
    "domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc",
    "commodity_desc", "group_desc", "class_desc"
]:
    df[c] = df[c].astype("string").str.strip().str.lower()

comms_of_interest = ["cattle", "chickens", "hogs"]

class_keep_map = {
    "cattle": {"incl calves","(excl cows)","cows, beef","cows, milk","calves","calves, veal","ge 500 lbs","heifers, ge 500 lbs, milk replacement"},
    "chickens": {"broilers","layers","layers & pullets","pullets, replacement","roosters"},
    "hogs": {"all classes","breeding"},
}

df_sub = df[
    (df["commodity_desc"].isin(comms_of_interest)) &
    (df["unit_desc"].isin(["operations", "head"])) &
    (df["statisticcat_desc"].isin(["inventory", "operations"])) &
    (df["domaincat_desc"].str.startswith("inventory", na=False))
].copy()

allowed_pairs = {
    (commodity, cls)
    for commodity, classes in class_keep_map.items()
    for cls in classes
}
pair_index = pd.MultiIndex.from_frame(df_sub[["commodity_desc", "class_desc"]])
allowed_index = pd.MultiIndex.from_tuples(sorted(allowed_pairs))
df_sub = df_sub[pair_index.isin(allowed_index)].copy()

print("df_sub rows:", len(df_sub))

# check value counts 
df_sub["commodity_desc"].value_counts()
df_sub["class_desc"].value_counts().head(20)
df_sub["unit_desc"].value_counts()
df_sub["statisticcat_desc"].value_counts()



########### DATA CLEANING FOR ALL FIPS / AG -- ENSURE MATCH ####################


matches = glob.glob(os.path.join(outf, "*fips_full*.csv")) # pull the most recent fips file 
if matches:
    fips_sense = max(matches, key=os.path.getmtime)
    print("Using:", fips_sense)
else:
    print("No matching file found.")
    
fips_df = pd.read_csv(fips_sense)   # upload fips_df
fips_df = clean_cols(fips_df)

# standardize fips key from external file
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




########### DATA CLEANING FOR ALL AG DATA - ITERATING OVER MISSING YRS ####################

# prep ag raw df for iteration, then export and saving
ag_raw_df = clean_cols(combined)
ag_raw_df.head()

# create total fips code 
ag_raw_df = generate_fips(ag_raw_df, state_col="state_fips_code", city_col="county_code")
ag_raw_df.columns
# normalize possible FIPS column variants and guard against duplicate column names
if "FIPS_generated" in ag_raw_df.columns and "fips_generated" not in ag_raw_df.columns:
    ag_raw_df = ag_raw_df.rename(columns={"FIPS_generated": "fips_generated"})
elif "FIPS_generated" in ag_raw_df.columns and "fips_generated" in ag_raw_df.columns:
    # keep one canonical column name/value path
    ag_raw_df = ag_raw_df.drop(columns=["FIPS_generated"])

if ag_raw_df.columns.duplicated().any():
    ag_raw_df = ag_raw_df.loc[:, ~ag_raw_df.columns.duplicated()]

if "fips_generated" not in ag_raw_df.columns:
    raise RuntimeError("fips_generated column missing after FIPS generation/normalization")

ag_raw_df["fips_generated"] = ag_raw_df["fips_generated"].astype("string").str.zfill(5)
ag_raw_df["year"] = pd.to_numeric(ag_raw_df["year"], errors="coerce").astype("Int64")


# iterate forward
base_years = [2002,2007,2012,2017]
n_forward = 4   # number of years to create after each base year
year_col = "year"

new_frames = []
for b in base_years:
    base = ag_raw_df[ag_raw_df[year_col] == b].copy()
    if base.empty:
        continue
    for y in range(b+1, b+1 + n_forward):
        new_frames.append(base.assign(**{year_col: y}))

new_rows = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame(columns=ag_raw_df.columns)
df_big = pd.concat([ag_raw_df, new_rows], ignore_index=True)
len_df_big_predupe = len(df_big)
len_it_rows = len(new_rows)

df_big = df_big.drop_duplicates(ignore_index=True) # drop dupes
# sense check bind -- comparing lengths 
len_raw_df = len(ag_raw_df)
len_df_big_post_dupe= len(df_big)

print("No duplicates found in final dataframe? ", 
      (len_df_big_post_dupe == len_df_big_predupe == (len_it_rows + len_raw_df))) # INTERPRETATION: 572,656 county observations (after filter for our commodities) from 2002-2021
# recall that USDA aggregates to keep anonymity of the survey, so the FIPS level will not give us the exact number of rows = number of farms 

# attach external fips-year key and county name
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

df_big["county_name_norm"] = _norm_county_name(df_big["county_name"])
df_big["county_fips_name_norm"] = _norm_county_name(df_big["county_fips_name"])

name_mismatch = df_big[
    df_big["county_name_norm"].notna()
    & df_big["county_fips_name_norm"].notna()
    & (df_big["county_name_norm"] != df_big["county_fips_name_norm"])
].copy()

missing_fips_key = df_big[df_big["_merge"] != "both"].copy()


today_str = date.today().strftime("%Y-%m-%d")

clean_ag_census = f"{today_str}_ag_annual_df.csv"
ag_path = os.path.join(outf, clean_ag_census)
df_big.to_csv(ag_path, index=False)
print("Saved iterated AG panel:", ag_path)

review_outf = "/Users/allegrasaggese/Dropbox/Mental/allegra-dropbox-copy/interim-data"
os.makedirs(review_outf, exist_ok=True)
mismatch_path = os.path.join(review_outf, f"{today_str}_ag_fips_name_mismatch.csv")
missing_key_path = os.path.join(review_outf, f"{today_str}_ag_fips_missing_key.csv")
name_mismatch.to_csv(mismatch_path, index=False)
missing_fips_key.to_csv(missing_key_path, index=False)
print("Saved county-name mismatch rows for manual review:", mismatch_path)
print("Saved missing fips-year key rows for manual review:", missing_key_path)



################### STAGE 2: BUILD CAFO SIZE STRUCTURE #####################
# Input for Stage 2 is df_big from Stage 1 (iterated annual panel with FIPS merge).

df_cafo = df_big.copy()

for c in [
    "domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc",
    "commodity_desc", "group_desc", "class_desc"
]:
    if c in df_cafo.columns:
        df_cafo[c] = df_cafo[c].astype("string").str.strip().str.lower()

comms_of_interest = ["cattle", "chickens", "hogs"]

class_keep_map = {
    "cattle": {
        "incl calves",
        "(excl cows)",
        "cows, beef",
        "cows, milk",
        "calves",
        "calves, veal",
        "ge 500 lbs",
        "heifers, ge 500 lbs, milk replacement",
    },
    "chickens": {
        "broilers",
        "layers",
        "layers & pullets",
        "pullets, replacement",
        "roosters",
    },
    "hogs": {
        "all classes",
        "breeding",
    },
}

# Keep inventory-bin rows and both statistic categories requested.
df_sub = df_cafo[
    (df_cafo["commodity_desc"].isin(comms_of_interest))
    & (df_cafo["unit_desc"].isin(["operations", "head"]))
    & (df_cafo["statisticcat_desc"].isin(["inventory", "operations"]))
    & (df_cafo["domaincat_desc"].str.startswith("inventory", na=False))
].copy()

allowed_pairs = {
    (commodity, cls)
    for commodity, classes in class_keep_map.items()
    for cls in classes
}
pair_index = pd.MultiIndex.from_frame(df_sub[["commodity_desc", "class_desc"]])
allowed_index = pd.MultiIndex.from_tuples(sorted(allowed_pairs))
df_sub = df_sub[pair_index.isin(allowed_index)].copy()

print("Stage 2 df_sub rows:", len(df_sub))
print("Stage 2 commodity mix:")
print(df_sub["commodity_desc"].value_counts(dropna=False))
print("Stage 2 unit mix:")
print(df_sub["unit_desc"].value_counts(dropna=False))
print("Stage 2 statistic mix:")
print(df_sub["statisticcat_desc"].value_counts(dropna=False))


def map_size(df, mapping, unit_match, out_col):
    mask = df["unit_desc"] == unit_match
    df[out_col] = df["domaincat_desc"].map(mapping).where(mask, other=pd.NA).astype("Int64")


def map_size_class(df, mapping, unit_match, class_match, out_col):
    mask = (df["unit_desc"] == unit_match) & (df["class_desc"] == class_match)
    df[out_col] = df["domaincat_desc"].map(mapping).where(mask, other=pd.NA).astype("Int64")


# Inventory-bin mappings preserved from script0b-ag-raw-v2.
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

map_size_class(df_sub, layer_map, unit_match="operations", class_match="layers", out_col="layer_ops_size")
map_size_class(df_sub, layer_map, unit_match="operations", class_match="broilers", out_col="broiler_ops_size")

map_size(df_sub, cattle_inv_map, unit_match="operations", out_col="cattle_ops_size_inv")
map_size(df_sub, hog_inv_map, unit_match="operations", out_col="hog_ops_size_inv")
map_size(df_sub, milk_cows_map, unit_match="operations", out_col="dairy_ops_size_inv")
map_size(df_sub, breeding_hogs_map, unit_match="operations", out_col="breed_hog_ops_size_inv")
map_size(df_sub, cattle_inv_map_no_cows, unit_match="operations", out_col="cattle_senzcow_ops_size_inv")
map_size(df_sub, cattle_feed_map, unit_match="operations", out_col="cattle_feed_ops_size_inv")
map_size(df_sub, beef_cows_map, unit_match="operations", out_col="beef_ops_size_inv")

# Numeric-bin thresholds for S/M/L assignment (preserved).
broiler_cutoff_lrg = 5
broiler_cutoff_med = 3
layer_cutoff_lrg = 9
layer_cutoff_med = 7
cattle_cutoff_lrg = 7
cattle_cutoff_med = 6
hog_cutoff_lrg = 7
hog_cutoff_med = 6

col_thresholds = {
    "layer_ops_size": (layer_cutoff_med, layer_cutoff_lrg),
    "broiler_ops_size": (broiler_cutoff_med, broiler_cutoff_lrg),
    "cattle_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "dairy_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "cattle_senzcow_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "cattle_feed_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "beef_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "hog_ops_size_inv": (hog_cutoff_med, hog_cutoff_lrg),
    "breed_hog_ops_size_inv": (hog_cutoff_med, hog_cutoff_lrg),
}


def categorize_code(v, med, lrg):
    if pd.isna(v):
        return pd.NA
    if v < med:
        return "small"
    if v < lrg:
        return "medium"
    return "large"


df2 = df_sub.copy()
df2["size_class"] = pd.Series(pd.NA, index=df2.index, dtype="string")
df2["size_source"] = pd.Series(pd.NA, index=df2.index, dtype="string")

for col, (med, lrg) in col_thresholds.items():
    codes = df2[col]
    take = codes.notna() & df2["size_source"].isna()
    df2.loc[take, "size_class"] = codes[take].apply(categorize_code, args=(med, lrg))
    df2.loc[take, "size_source"] = col

# Keep operations counts in each inventory bin.
df2["ops_in_bin"] = pd.to_numeric(
    df2["value"]
    .astype("string")
    .str.replace(",", "", regex=False)
    .str.strip()
    .replace({"(d)": pd.NA, "(z)": pd.NA, "": pd.NA}),
    errors="coerce",
)

# Row-level CAFO flags (mapped rows only).
df2["is_large_cafo_row"] = ((df2["size_class"] == "large") & df2["size_source"].notna()).astype("Int8")
df2["is_medium_or_large_cafo_row"] = (
    df2["size_class"].isin(["medium", "large"]) & df2["size_source"].notna()
).astype("Int8")

# Compact long summary: county-year-commodity-class-size.
summary = (
    df2[df2["size_source"].notna()]
    .groupby(
        [
            "year",
            "fips_generated",
            "county_fips_name",
            "commodity_desc",
            "class_desc",
            "statisticcat_desc",
            "size_source",
            "size_class",
        ],
        as_index=False,
    )["ops_in_bin"]
    .sum(min_count=1)
    .rename(columns={"ops_in_bin": "sum_ops"})
)

# County-year compact table with small/medium/large columns.
summary_compact = (
    summary.pivot_table(
        index=["year", "fips_generated", "county_fips_name", "commodity_desc", "class_desc"],
        columns="size_class",
        values="sum_ops",
        aggfunc="sum",
        fill_value=0,
    )
    .reset_index()
)
summary_compact.columns.name = None
for size_col in ["small", "medium", "large"]:
    if size_col not in summary_compact.columns:
        summary_compact[size_col] = 0
summary_compact["any_large_cafo"] = (summary_compact["large"] > 0).astype("Int8")
summary_compact["any_medium_or_large_cafo"] = (
    (summary_compact["medium"] + summary_compact["large"]) > 0
).astype("Int8")

print("Stage 2 mapped rows:", int(df2["size_source"].notna().sum()))
print("Stage 2 size class counts:")
print(df2["size_class"].value_counts(dropna=False))
print("Stage 2 compact rows:", len(summary_compact))


################### STAGE 3: EXPORT CAFO OUTPUTS #####################

clean_cafo_row = f"{today_str}_cafo_annual_df.csv"
clean_cafo_long = f"{today_str}_cafo_ops_by_size_long.csv"
clean_cafo_compact = f"{today_str}_cafo_ops_by_size_compact.csv"

cafo_row_path = os.path.join(outf, clean_cafo_row)
cafo_long_path = os.path.join(outf, clean_cafo_long)
cafo_compact_path = os.path.join(outf, clean_cafo_compact)

df2.to_csv(cafo_row_path, index=False)
summary.to_csv(cafo_long_path, index=False)
summary_compact.to_csv(cafo_compact_path, index=False)

print("Saved CAFO row-level panel:", cafo_row_path)
print("Saved CAFO long summary:", cafo_long_path)
print("Saved CAFO compact summary:", cafo_compact_path)
