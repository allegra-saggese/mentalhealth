#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 01:55:11 2026

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
repo = "/Users/allegrasaggese/Documents/GitHub/mentalhealth"
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
# Pull USDA NASS Quick Stats data directly instead of reading .dta files.
# Requires env var USDA_NASS_API_KEY to be set in your shell.
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
group_desc_allow = ["LIVESTOCK", "OPERATIONS", "POULTRY", "ANIMAL TOTALS"]
commodity_desc_allow = ["CATTLE", "CHICKENS", "HOGS", "EGGS", "MILK", "POULTRY TOTALS", "ANIMAL TOTALS"]
unit_desc_allow = ["HEAD", "OPERATIONS"]

# Census years only
census_years = [2002, 2007, 2012, 2017]

# State list for chunking if any request exceeds 50k rows
state_alpha = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
    "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
]


def fetch_ag_data():
    frames = []
    rate_sleep = 0.4  # basic rate limiting

    for yr in census_years:
        for grp in group_desc_allow:
            for cmd in commodity_desc_allow:
                for unit in unit_desc_allow:
                    time.sleep(rate_sleep)
                    base_params = {
                        "key": NASS_API_KEY,
                        "source_desc": source_desc,
                        "agg_level_desc": agg_level_desc,
                        "sector_desc": sector_desc,
                        "group_desc": grp,
                        "commodity_desc": cmd,
                        "unit_desc": unit,
                        "year": yr,
                    }

                    try:
                        count = nass_get_counts(base_params)
                    except RuntimeError as e:
                        # skip parameter combos that the API rejects (often 403)
                        print("Skipping combo due to API error:", e)
                        continue

                    if count == 0:
                        continue

                    # If row count is too large, split by state_alpha
                    if count > 50000:
                        for st in state_alpha:
                            time.sleep(rate_sleep)
                            st_params = dict(base_params)
                            st_params["state_alpha"] = st
                            try:
                                st_count = nass_get_counts(st_params)
                            except RuntimeError as e:
                                print("Skipping state combo due to API error:", e)
                                continue
                            if st_count == 0:
                                continue
                            if st_count > 50000:
                                raise RuntimeError(
                                    f"Request still >50k rows after state split: "
                                    f"year={yr}, group={grp}, commodity={cmd}, unit={unit}, state={st}, count={st_count}"
                                )
                            df_st = nass_get_data(st_params)
                            if not df_st.empty:
                                frames.append(df_st)
                    else:
                        df = nass_get_data(base_params)
                        if not df.empty:
                            frames.append(df)

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
    
    
# block --- avoid reloading data if it already exists in enviro 
 
import pandas as pd, os, glob

if "df" not in globals():
    agfolder = "/Users/allegrasaggese/Dropbox/Mental/Data/raw/usda" # will need to make compatible with any enviro / working directory 
    agfiles = sorted(glob.glob(os.path.join(agfolder, "*.dta")))
    dfs = [pd.read_stata(f) for f in agfiles]
    df = pd.concat(dfs, ignore_index=True)

# normalize column names just once
df.columns = df.columns.str.lower().str.strip()


# Block B: filter to operations + inventory bins + commodities

comms_of_interest = ["cattle", "chickens", "milk", "eggs", "hogs"]
groups_exclude = ["specialty", "aquaculture", "animal totals"]

df_sub = df[
    df["commodity_desc"].isin(comms_of_interest)
    & ~df["group_desc"].isin(groups_exclude)
    & (df["statisticcat_desc"] == "operations")
    & (df["unit_desc"] == "operations")
].copy()

# keep only inventory bins
df_sub = df_sub[df_sub["domaincat_desc"].str.startswith("inventory", na=False)].copy()

# ensure FIPS column exists
if "fips_generated" not in df_sub.columns:
    from functions import generate_fips
    df_sub = generate_fips(df_sub, state_col="state_fips_code", city_col="county_code")
    if "FIPS_generated" in df_sub.columns and "fips_generated" not in df_sub.columns:
        df_sub = df_sub.rename(columns={"FIPS_generated": "fips_generated"})

    
# Block C: mappings (same as script)

layer_map = {
 "inventory: (1 to 49 head)":1,
 "inventory: (50 to 99 head)":2,
 "inventory: (100 to 399 head)":3,
 "inventory: (400 to 3,199 head)":4,
 "inventory: (3,200 to 9,999 head)":5,
 "inventory: (10,000 to 19,999 head)":6,
 "inventory: (20,000 to 49,999 head)":7,
 "inventory: (50,000 to 99,999 head)":8,
 "inventory: (100,000 or more head)":9
}

cattle_inv_map = {
 "inventory of cattle, incl calves: (1 to 9 head)":1,
 "inventory of cattle, incl calves: (10 to 19 head)":2,
 "inventory of cattle, incl calves: (20 to 49 head)":3,
 "inventory of cattle, incl calves: (50 to 99 head)":4,
 "inventory of cattle, incl calves: (100 to 199 head)":5,
 "inventory of cattle, incl calves: (200 to 499 head)":6,
 "inventory of cattle, incl calves: (500 or more head)":7
}

hog_inv_map = {
 "inventory of hogs: (1 to 24 head)":1,
 "inventory of hogs: (25 to 49 head)":2,
 "inventory of hogs: (50 to 99 head)":3,
 "inventory of hogs: (100 to 199 head)":4,
 "inventory of hogs: (200 to 499 head)":5,
 "inventory of hogs: (500 to 999 head)":6,
 "inventory of hogs: (1,000 or more head)":7
}

milk_cows_map = {
 "inventory of milk cows: (1 to 9 head)":1,
 "inventory of milk cows: (10 to 19 head)":2,
 "inventory of milk cows: (20 to 49 head)":3,
 "inventory of milk cows: (50 to 99 head)":4,
 "inventory of milk cows: (100 to 199 head)":5,
 "inventory of milk cows: (200 to 499 head)":6,
 "inventory of milk cows: (500 or more head)":7
}

breeding_hogs_map = {
 "inventory of breeding hogs: (1 to 24 head)": 1,
 "inventory of breeding hogs: (25 to 49 head)": 2,
 "inventory of breeding hogs: (50 to 99 head)": 3,
 "inventory of breeding hogs: (100 or more head)": 4
}

cattle_inv_map_no_cows = {
 "inventory of cattle, (excl cows): (1 to 9 head)": 1,
 "inventory of cattle, (excl cows): (10 to 19 head)": 2,
 "inventory of cattle, (excl cows): (100 to 199 head)": 3,
 "inventory of cattle, (excl cows): (20 to 49 head)": 4,
 "inventory of cattle, (excl cows): (200 to 499 head)": 5,
 "inventory of cattle, (excl cows): (50 to 99 head)": 6
}

cattle_feed_map = {
 "inventory of cattle on feed: (1 to 19 head)":1,
 "inventory of cattle on feed: (1 to 9 head)":2,
 "inventory of cattle on feed: (10 to 19 head)":3,
 "inventory of cattle on feed: (100 to 199 head)":4,
 "inventory of cattle on feed: (20 to 49 head)":5,
 "inventory of cattle on feed: (200 to 499 head)":6,
 "inventory of cattle on feed: (50 to 99 head)":7,
 "inventory of cattle on feed: (500 or more head)":8
}

beef_cows_map = {
 "inventory of beef cows: (1 to 9 head)":1,
 "inventory of beef cows: (10 to 19 head)":2,
 "inventory of beef cows: (20 to 49 head)":3,
 "inventory of beef cows: (50 to 99 head)":4,
 "inventory of beef cows: (100 to 199 head)":5,
 "inventory of beef cows: (200 to 499 head)":6,
 "inventory of beef cows: (500 or more head)":7
}

def map_size_class(df, mapping, unit_match, class_match, out_col):
    mask = (df['unit_desc'] == unit_match) & (df['class_desc'] == class_match)
    df[out_col] = df['domaincat_desc'].map(mapping).where(mask, other=pd.NA).astype("Int64")

# operations only
map_size_class(df_sub, layer_map, unit_match="operations", class_match="layers", out_col="layer_ops_size")
map_size_class(df_sub, layer_map, unit_match="operations", class_match="broilers", out_col="broiler_ops_size")
map_size_class(df_sub, cattle_inv_map, unit_match="operations", class_match=None, out_col="cattle_ops_size_inv")
map_size_class(df_sub, hog_inv_map, unit_match="operations", class_match=None, out_col="hog_ops_size_inv")
map_size_class(df_sub, milk_cows_map, unit_match="operations", class_match=None, out_col="dairy_ops_size_inv")
map_size_class(df_sub, breeding_hogs_map, unit_match="operations", class_match=None, out_col="breed_hog_ops_size_inv")
map_size_class(df_sub, cattle_inv_map_no_cows, unit_match="operations", class_match=None, out_col="cattle_senzcow_ops_size_inv")
map_size_class(df_sub, cattle_feed_map, unit_match="operations", class_match=None, out_col="cattle_feed_ops_size_inv")
map_size_class(df_sub, beef_cows_map, unit_match="operations", class_match=None, out_col="beef_ops_size_inv")


# Block D: thresholds + size class

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
    if pd.isna(v): return pd.NA
    if v < med: return "small"
    if v < lrg: return "medium"
    return "large"

df2 = df_sub.copy()
df2["size_class"] = pd.Series(pd.NA, index=df2.index, dtype="string")

for col, (med, lrg) in col_thresholds.items():
    codes = df2[col]
    mask = codes.notna()
    df2.loc[mask, "size_class"] = codes[mask].apply(categorize_code, args=(med, lrg))

# numeric ops count
df2["ops_in_bin"] = pd.to_numeric(df2["value"].astype(str).str.replace(",", ""), errors="coerce")

group_cols = ["year", "fips_generated", "size_class", "unit_desc", "commodity_desc"]
df2["sum_ops"] = df2.groupby(group_cols)["ops_in_bin"].transform("sum")



