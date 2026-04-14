#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 15:19:00 2025

@author: allegrasaggese
"""

# ----------------------- SET UP PART 1 : set repos / outputs / purpose  -------------------- -#
# output: This script writes `*_cafo_ops_by_size_compact.csv`, which is consumed downstream

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


# ----------------------- SET UP PART 2 : PULL USDA FROM API  -------------------- -#
# Purpose: pull the data directly from the API (avoiding the existind .dta files)

# set KEY for API 
os.environ["USDA_NASS_API_KEY"] = "30643212-7739-359A-B451-0EAD3D345DB9" # will have to change for user

# pull USDA NASS Quick Stats data directly instead of reading .dta files 
USE_API = True
NASS_API_KEY = os.environ.get("USDA_NASS_API_KEY") 
if USE_API and not NASS_API_KEY:
    raise RuntimeError("Missing USDA_NASS_API_KEY env var. Set it before running this script.")

NASS_BASE = "https://quickstats.nass.usda.gov/api/"


# set filters (user-specified)
source_desc = "CENSUS"
agg_level_desc = "COUNTY"
sector_desc = "ANIMALS & PRODUCTS"
commodity_desc_allow = ["CATTLE", "CHICKENS", "HOGS"]

unit_desc_allow = ["HEAD", "OPERATIONS"]
statisticcat_desc_allow = ["INVENTORY", "OPERATIONS"]

# take the census years only - where data is actually recorded 
census_years = [2002, 2007, 2012, 2017]

# take cmmodity-specific domain splits used only when a query exceeds 50k rows (otherwise its too intensive)
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


# create function to define a call in the API for the ag data 
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
            return functions.nass_get_counts(NASS_BASE, params)
        except RuntimeError as e:
            print("Skipping combo due to API error:", e)
            return None

    def _safe_get_data(params):
        try:
            out = functions.nass_get_data(NASS_BASE, params)
        except RuntimeError as e:
            print("Skipping data pull due to API error:", e)
            return pd.DataFrame()
        return _keep_inventory_domaincat(out)

    frames = []
    rate_sleep = 0.4  # basic rate limiting

    for yr in census_years:
        for cmd in commodity_desc_allow:
            for unit in unit_desc_allow:
                for stat in statisticcat_desc_allow: # call the units we want to pull
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

                    # Still too large: split by commodity-specific domain_desc values
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


# Load .dta files as the authoritative baseline for USDA Census years.
# The API often has gaps for specific commodity × census-year combinations
# (e.g. hogs 2012+, cattle 2017+) even when it returns data for other combos,
# so the old "fallback only if combined.empty" logic silently left those years
# empty. We now ALWAYS load the .dta files and supplement with any API rows.
agfolder = os.path.join(inf, "usda")
agfiles = sorted(glob.glob(os.path.join(agfolder, "*.dta")))
if not agfiles:
    raise RuntimeError(f"No .dta files found in {agfolder}. Cannot build CAFO panel.")
dta_frames = []
for f in agfiles:
    _df = pd.read_stata(f)
    _df.columns = [c.lower().strip() for c in _df.columns]
    dta_frames.append(_df)
combined_dta = pd.concat(dta_frames, ignore_index=True)
print(f"Loaded {len(combined_dta):,} rows from {len(agfiles)} local .dta files: {[os.path.basename(f) for f in agfiles]}")

# Optionally supplement with API rows (adds any rows not already in .dta baseline)
combined_api = pd.DataFrame()
if USE_API:
    try:
        combined_api = fetch_ag_data()
        combined_api.columns = [c.lower().strip() for c in combined_api.columns]
        print(f"Loaded {len(combined_api):,} rows from USDA NASS API")
    except RuntimeError as e:
        print(f"API fetch failed (non-fatal — .dta baseline is sufficient): {e}")

# .dta files are authoritative for all 4 census years; API returns 403 for 2012/2017
# and has extra schema columns (load_time, congr_district_code, etc.) absent in .dta,
# so a full drop_duplicates() cannot remove overlapping rows → ~2x inflation.
# Fix: use .dta exclusively. API supplement disabled to prevent double-counting.
combined = combined_dta.copy()
print(f"Using .dta baseline only ({len(combined):,} rows). API supplement disabled to prevent schema-mismatch duplication.")
    
    
    
# ----------------------- SET UP PART 3 : STANDARDIZE COMBINED DATA   -------------------- -#
# The .dta files loaded above are the authoritative source for all four USDA Census years
# (2002, 2007, 2012, 2017). No donor backfill is needed — the .dta files have complete
# coverage for cattle, hogs, and chickens for all census years.
# The old donor backfill introduced pre-processed rows (with fips_generated already set)
# that conflict with the raw .dta rows (no fips_generated), causing the FIPS merge to fail.
combined = combined.loc[:, ~combined.columns.duplicated()].copy()
print("Proceeding with .dta baseline (+ any API supplement). No donor backfill applied.")

# check we have ops in presence 
chk = combined.copy()
for c in ["commodity_desc", "unit_desc"]:
    chk[c] = chk[c].astype("string").str.strip().str.lower()
chk["year"] = pd.to_numeric(chk["year"], errors="coerce").astype("Int64")
chk = chk[
    chk["commodity_desc"].isin(["cattle", "chickens", "hogs"]) &
    chk["year"].isin([2012, 2017])
]

# quick eye-ball QA 
print(chk.groupby(["year", "unit_desc"]).size().unstack(fill_value=0))

print(combined["year"].value_counts().sort_index().tail(10))



# ----------------------- DATA PART 1 : QUALITY ASSURANCE  -------------------- -#

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

# check subset of the dataframe, checkc to see we have the necessary values in each column 
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





# ----------------------- DATA PART 2 : MATCH FIPS to AG DATA   -------------------- -#

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





# ----------------------- DATA PART 3 : FORWARD FILL MISSING AG DATA   -------------------- -#

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

print("FIPS merge status:") # check merge status 
print(df_big["_merge"].value_counts(dropna=False))

def _norm_county_name(s):
    s = s.astype("string").str.lower().str.strip()
    s = s.str.replace(r"[^a-z0-9 ]", "", regex=True)
    s = s.str.replace(r"\b(county|parish|borough|census area|municipality|city and borough)\b", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

# reset names in the variables 
df_big["county_name_norm"] = _norm_county_name(df_big["county_name"])
df_big["county_fips_name_norm"] = _norm_county_name(df_big["county_fips_name"])

name_mismatch = df_big[
    df_big["county_name_norm"].notna()
    & df_big["county_fips_name_norm"].notna()
    & (df_big["county_name_norm"] != df_big["county_fips_name_norm"])
].copy()

missing_fips_key = df_big[df_big["_merge"] != "both"].copy()



clean_ag_census = f"{today_str}_ag_annual_df.csv"
ag_path = os.path.join(outf, clean_ag_census)
df_big.to_csv(ag_path, index=False) # save a version of the clean, large agriculture census data set as an interim output 
print("Saved iterated AG panel:", ag_path)



review_outf = "/Users/allegrasaggese/Dropbox/Mental/allegra-dropbox-copy/interim-data"
os.makedirs(review_outf, exist_ok=True)
mismatch_path = os.path.join(review_outf, f"{today_str}_ag_fips_name_mismatch.csv")
missing_key_path = os.path.join(review_outf, f"{today_str}_ag_fips_missing_key.csv")
name_mismatch.to_csv(mismatch_path, index=False)
missing_fips_key.to_csv(missing_key_path, index=False)
# print out a QA to see if we need to manually check any fips codes 
print("Saved county-name mismatch rows for manual review:", mismatch_path)
print("Saved missing fips-year key rows for manual review:", missing_key_path)



# ----------------------- ANALYSIS PART 1 : DEFINE CAFO VARIABLES   -------------------- -#
# Input for Stage 2 is df_big from Stage 1 (iterated annual panel with FIPS merge).

# new copy - this way we don't mess up the data cleaning stage 
df_cafo = df_big.copy()

# these are just the vars we need to keep that contain the info relevant for our analysis
for c in [
    "domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc",
    "commodity_desc", "group_desc", "class_desc"
]:
    if c in df_cafo.columns:
        df_cafo[c] = df_cafo[c].astype("string").str.strip().str.lower()

# picking commodities of interest
comms_of_interest = ["cattle", "chickens", "hogs"]


# within each of the commodities, there are sub-categories, so we map these classes 
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

# keeping inventory bins most crucially, and the the other vars within the dataframe
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

# print the QA so we can see how many of each var remain 
print("Stage 2 df_sub rows:", len(df_sub))
print("Stage 2 commodity mix:")
print(df_sub["commodity_desc"].value_counts(dropna=False))
print("Stage 2 unit mix:")
print(df_sub["unit_desc"].value_counts(dropna=False))
print("Stage 2 statistic mix:")
print(df_sub["statisticcat_desc"].value_counts(dropna=False))



# Inventory-bin mappings (preserved from script0b-ag-raw-v2) which just label the text descriptions numerically for ease 
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


# apply mapping to the data 
map_size_class(df_sub, layer_map, unit_match="operations", class_match="layers", out_col="layer_ops_size")
map_size_class(df_sub, layer_map, unit_match="operations", class_match="broilers", out_col="broiler_ops_size")

map_size(df_sub, cattle_inv_map, unit_match="operations", out_col="cattle_ops_size_inv")
map_size(df_sub, hog_inv_map, unit_match="operations", out_col="hog_ops_size_inv")
map_size(df_sub, milk_cows_map, unit_match="operations", out_col="dairy_ops_size_inv")
map_size(df_sub, breeding_hogs_map, unit_match="operations", out_col="breed_hog_ops_size_inv")
map_size(df_sub, cattle_inv_map_no_cows, unit_match="operations", out_col="cattle_senzcow_ops_size_inv")
map_size(df_sub, cattle_feed_map, unit_match="operations", out_col="cattle_feed_ops_size_inv")
map_size(df_sub, beef_cows_map, unit_match="operations", out_col="beef_ops_size_inv")

# Create a numeric-bin thresholds for S/M/L assignment (preserved from CAFO - USDA definition, roughly) 
broiler_cutoff_lrg = 5
broiler_cutoff_med = 3
layer_cutoff_lrg = 9
layer_cutoff_med = 7
cattle_cutoff_lrg = 7
cattle_cutoff_med = 6
hog_cutoff_lrg = 7
hog_cutoff_med = 6

# define them as thresholds 
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

# create a way to categorize them 
def categorize_code(v, med, lrg):
    if pd.isna(v):
        return pd.NA
    if v < med:
        return "small"
    if v < lrg:
        return "medium"
    return "large"


# ----------------------- ANALYSIS PART 2 : CREATE CAFO VARIABLES   -------------------- -#
df2 = df_sub.copy() # make another copy after the classification and before the development of new variables 
df2["size_class"] = pd.Series(pd.NA, index=df2.index, dtype="string")
df2["size_source"] = pd.Series(pd.NA, index=df2.index, dtype="string")

# create categorization
for col, (med, lrg) in col_thresholds.items():
    codes = df2[col]
    take = codes.notna() & df2["size_source"].isna()
    df2.loc[take, "size_class"] = codes[take].apply(categorize_code, args=(med, lrg))
    df2.loc[take, "size_source"] = col

# Keep operations counts in each inventory bin 
df2["ops_in_bin"] = pd.to_numeric(
    df2["value"]
    .astype("string")
    .str.replace(",", "", regex=False)
    .str.strip()
    .replace({"(d)": pd.NA, "(z)": pd.NA, "": pd.NA}),
    errors="coerce",
)

# Row-level CAFO flags (mapped rows only) 
df2["is_large_cafo_row"] = ((df2["size_class"] == "large") & df2["size_source"].notna()).astype("Int8")
df2["is_medium_or_large_cafo_row"] = (
    df2["size_class"].isin(["medium", "large"]) & df2["size_source"].notna()
).astype("Int8")

# Compact long summary: county-year-commodity-class-size 
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

# County-year compact table with small/medium/large columns
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

# QA: cattle canonical vs subgroup overlap diagnostic at county-year level
# Purpose: verify whether cattle subclasses can be summed without overlap.
# Canonical reference = class_desc "incl calves".
cattle_compact = summary_compact[summary_compact["commodity_desc"] == "cattle"].copy()
if not cattle_compact.empty:
    cattle_compact["ops_total"] = cattle_compact[["small", "medium", "large"]].sum(axis=1, min_count=1)
    cattle_wide = (
        cattle_compact.pivot_table(
            index=["year", "fips_generated", "county_fips_name"],
            columns="class_desc",
            values="ops_total",
            aggfunc="sum",
        )
        .reset_index()
    )
    cattle_wide.columns.name = None

    canonical_col = "incl calves"
    noncanonical_cols = [c for c in cattle_wide.columns if c not in {"year", "fips_generated", "county_fips_name", canonical_col}]
    partition_candidate_cols = [c for c in ["(excl cows)", "cows, beef", "cows, milk"] if c in cattle_wide.columns]

    cattle_wide["canonical_ops_incl_calves"] = pd.to_numeric(cattle_wide.get(canonical_col), errors="coerce")
    cattle_wide["sum_noncanonical_ops"] = (
        cattle_wide[noncanonical_cols].sum(axis=1, min_count=1) if noncanonical_cols else np.nan
    )
    cattle_wide["sum_partition_candidate_ops"] = (
        cattle_wide[partition_candidate_cols].sum(axis=1, min_count=1) if partition_candidate_cols else np.nan
    )
    cattle_wide["sum_all_class_ops"] = (
        cattle_wide["canonical_ops_incl_calves"] + cattle_wide["sum_noncanonical_ops"]
    )

    for lhs in ["sum_noncanonical_ops", "sum_partition_candidate_ops", "sum_all_class_ops"]:
        ratio_col = f"ratio_{lhs}_to_canonical"
        diff_col = f"abs_pct_diff_{lhs}_vs_canonical"
        cattle_wide[ratio_col] = np.where(
            cattle_wide["canonical_ops_incl_calves"] > 0,
            cattle_wide[lhs] / cattle_wide["canonical_ops_incl_calves"],
            np.nan,
        )
        cattle_wide[diff_col] = np.where(
            cattle_wide["canonical_ops_incl_calves"] > 0,
            (cattle_wide[lhs] - cattle_wide["canonical_ops_incl_calves"]).abs() / cattle_wide["canonical_ops_incl_calves"] * 100,
            np.nan,
        )

    cattle_overlap_path = os.path.join(outf, f"{today_str}_qa_cattle_class_overlap_county_year.csv")
    cattle_wide.to_csv(cattle_overlap_path, index=False)

    year_diag = (
        cattle_wide.groupby("year", as_index=False)
        .agg(
            county_years=("canonical_ops_incl_calves", "size"),
            canonical_sum=("canonical_ops_incl_calves", "sum"),
            partition_sum=("sum_partition_candidate_ops", "sum"),
            all_class_sum=("sum_all_class_ops", "sum"),
            median_ratio_partition_to_canonical=("ratio_sum_partition_candidate_ops_to_canonical", "median"),
            median_ratio_all_to_canonical=("ratio_sum_all_class_ops_to_canonical", "median"),
        )
    )
    year_diag["ratio_partition_sum_to_canonical_sum"] = np.where(
        year_diag["canonical_sum"] > 0,
        year_diag["partition_sum"] / year_diag["canonical_sum"],
        np.nan,
    )
    year_diag["ratio_allclass_sum_to_canonical_sum"] = np.where(
        year_diag["canonical_sum"] > 0,
        year_diag["all_class_sum"] / year_diag["canonical_sum"],
        np.nan,
    )
    cattle_overlap_year_path = os.path.join(outf, f"{today_str}_qa_cattle_class_overlap_by_year.csv")
    year_diag.to_csv(cattle_overlap_year_path, index=False)

    print("Saved cattle overlap QA (county-year):", cattle_overlap_path)
    print("Saved cattle overlap QA (year-level):", cattle_overlap_year_path)
    print("Cattle all-class vs canonical (year-level ratio):")
    print(year_diag[["year", "ratio_allclass_sum_to_canonical_sum"]].to_string(index=False))


# QA - print on the missing values, and the total values of the vars for manual inspection
print("Stage 2 mapped rows:", int(df2["size_source"].notna().sum()))
print("Stage 2 size class counts:")
print(df2["size_class"].value_counts(dropna=False))
print("Stage 2 compact rows:", len(summary_compact))

# checking no years were dropped
print(df2[df2["size_source"].notna()]["year"].value_counts().sort_index().tail(15))
print(summary["year"].min(), summary["year"].max())
print(summary_compact["year"].min(), summary_compact["year"].max())


# ----------------------- ANALYSIS PART 3: EXPORT   -------------------- -#

# create different versions of the CAFO data, we will use the cafo_ops given ops categorization is most important 
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
