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


################## DATA CLEANING FOR ALL AG DATA ######################

# basic QA
print("Columns:", sorted(combined.columns))
print(combined.dtypes)
    



########### DATA CLEANING FOR ALL FIPS / AG -- ENSURING MATCH WILL WORK ####################

# sense check the length of the dataframe against the FIPS data 
# PURPOSE: to see how many observations there are 

matches = glob.glob(os.path.join(outf, "*fips_full*.csv")) # pull the most recent fips file 
if matches:
    fips_sense = max(matches, key=os.path.getmtime)
    print("Using:", fips_sense)
else:
    print("No matching file found.")
    
fips_df = pd.read_csv(fips_sense)   # upload fips_df 

        
        
# sense check the length of the dataframe against the FIPS data to see how many observations there are 
fips_sense = os.path.join(outf, "2025-11-10_fips_full.csv") 
fips_df = pd.read_csv(fips_sense) 

print("\n--- Combined dataframe number of rows ---")
print(len(combined))

# create fips slide df by year (to match w/ ag)
fdf_2002 = fips_df[fips_df["year"] == 2002]     
fdf_2007 = fips_df[fips_df["year"] == 2007]     
fdf_2012 = fips_df[fips_df["year"] == 2012]     
fdf_2017 = fips_df[fips_df["year"] == 2017]     

# checking dupes on fips data - although I"m certain we shouldn"t have any
dup_counts = fdf_2012.groupby(["fips", "year"]).size().reset_index(name="n_rows")
print(dup_counts["n_rows"].value_counts())  # quick frequency check
dup_counts_17 = fdf_2017.groupby(["fips", "year"]).size().reset_index(name="n_rows")
print(dup_counts_17["n_rows"].value_counts())  # quick frequency check



########### DATA CLEANING FOR ALL AG DATA - ITERATING OVER MISSING YRS ####################

# prep ag raw df for iteration, then export and saving
ag_raw_df = clean_cols(combined)
ag_raw_df.head()

# create total fips code 
ag_raw_df = generate_fips(ag_raw_df, state_col="state_fips_code", city_col="county_code")
ag_raw_df.columns


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
      (len_df_big_post_dupe == len_df_big_predupe == (len_it_rows + len_raw_df)))

# INTERPRETATION: about 14 million county observations --  2002-2021
# for QA --- USDA census reports about 1.9 mil farms (in 2022 - data we don"t have), so this number seems fair although maybe a little low) 
# recall that USDA aggregates to keep anonymity of the survey, so the FIPS level will not give us the exact number of rows = number of farms 

# export in this form -- create cafos after 
today_str = date.today().strftime("%Y-%m-%d")

clean_ag_census = f"{today_str}_ag_annual_df.csv"
ag_path = os.path.join(outf, clean_ag_census)
df_big.to_csv(ag_path, index=False)


############ COLLAPSE THE DATA FOR COUNTY LEVEL EASE (creating columns for the CAFOs instead)

# # # # unhash if you are just remaking the CAFOs (i.e. you don"t want to remake the RAW AG DATA!)
match_ag_df = glob.glob(os.path.join(outf, "*ag_annual_df*.csv")) # pull the most recent fips file 
if match_ag_df:
    ag_complete = max(match_ag_df, key=os.path.getmtime)
    print("Using:", ag_complete)
else:
    print("No matching file found.")
    
ag_iterated = pd.read_csv(ag_complete)   # upload ag_complete iterated data
# ag_iterated = df_big.copy()
ag_iterated.columns.tolist()

# remove files from cleaning / iterating ag data that take up a ton of space
del ag_raw_df, b, clean_ag_census, df, df_big
del dup_counts, dup_counts_17, fips_sense, i, matches, n_forward, new_frames, new_rows
del year_col, y



# ASSIGNMENT RULE (CAFO size classification)
# 1) Map USDA inventory size classes (domaincat_desc) to numeric bins per species.
# 2) For each species-specific bin, classify as small/medium/large using thresholds below.
# 3) Final size_class is assigned per row based on the mapped bin and thresholds.
# Note: No filtering by domain_desc; we rely on unit_desc in (HEAD, OPERATIONS).

# CAFO size limits, and can adjust these values for the S/M/L CAFO development by inventory size
broiler_cutoff_lrg = 5
broiler_cutoff_med =  3
layer_cutoff_lrg =  9
layer_cutoff_med =  7
cattle_cutoff_lrg =  7
cattle_cutoff_med = 6
hog_cutoff_lrg =  7
hog_cutoff_med = 6


### MAY DELETE THIS 
# CAFO size limits - head counts - will need to use this once we get AVERAGE SIZE by COUNTY - otherwise not useful 
h_dairy_cutoff_lrg = 700
h_dairy_cutoff_med = 200
h_all_cattle_cutoff_lrg = 1000
h_all_cattle_cutoff_med = 300
h_calf_cutoff_lrg = 1000
h_calf_cutoff_med = 300
h_hogs_cutoff_lrg = 2500
h_hogs_cutoff_med = 750
h_broilers_cutoff_lrg = 125000 # broilers fall in two categories - due to EPA categorization need to make a choice 
h_broilers_cutoff_med = 37500
h_layers_cutoff_lrg = 82000
h_layers_cutoff_med = 25000

thresholds = [
    broiler_cutoff_lrg, broiler_cutoff_med, layer_cutoff_lrg, layer_cutoff_med,
    cattle_cutoff_lrg, cattle_cutoff_med, hog_cutoff_lrg, hog_cutoff_med, 
    h_dairy_cutoff_lrg, h_dairy_cutoff_med, h_all_cattle_cutoff_lrg, 
    h_all_cattle_cutoff_med, h_calf_cutoff_lrg, h_calf_cutoff_med, h_hogs_cutoff_lrg, 
    h_hogs_cutoff_med, h_broilers_cutoff_lrg, h_broilers_cutoff_med, h_layers_cutoff_lrg,
    h_layers_cutoff_med]




# ===== CLEAN + WRAPPERS + FILTER + QA =====

# make a df copy
df = ag_iterated.copy()

# normalize all relevant string columns (strip + lowercase)
for c in [
    "domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc",
    "commodity_desc", "group_desc", "class_desc"
]:
    df[c] = df[c].astype("string").str.strip().str.lower()

# wrappers
def map_size(df, mapping, unit_match, out_col):
    mask = df["unit_desc"] == unit_match
    df[out_col] = df["domaincat_desc"].map(mapping).where(mask, other=pd.NA).astype("Int64")

def map_size_class(df, mapping, unit_match, class_match, out_col):
    mask = (df["unit_desc"] == unit_match) & (df["class_desc"] == class_match)
    df[out_col] = df["domaincat_desc"].map(mapping).where(mask, other=pd.NA).astype("Int64")

# ===== FILTER (updated spec) =====
# 1) group_desc in livestock/poultry/dairy
# 2) no filter on domain_desc
# 3) unit_desc = operations
# 4) statisticcat_desc = inventory
# 5) commodity_desc in cattle/chickens/eggs/hogs/milk
# 6) domaincat_desc starts with inventory

comms_of_interest = ["cattle", "chickens", "eggs", "hogs", "milk"]
groups_keep = ["livestock", "poultry", "dairy"]

df_sub = df[
    (df["group_desc"].isin(groups_keep)) &
    (df["commodity_desc"].isin(comms_of_interest)) &
    (df["unit_desc"] == "operations") &
    (df["statisticcat_desc"] == "inventory")
].copy()

# keep only inventory bins
df_sub = df_sub[df_sub["domaincat_desc"].str.startswith("inventory", na=False)].copy()

# ===== QA =====
print("df_sub rows:", len(df_sub))
print("commodity_desc:", df_sub["commodity_desc"].value_counts().head(10))
print("group_desc:", df_sub["group_desc"].value_counts().head(10))
print("domaincat_desc top 10:")
print(df_sub["domaincat_desc"].value_counts().head(10))




# ===== MAP INVENTORY BINS TO NUMERIC CODES =====

# mappings (same as before)
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

# apply mappings (operations rows)
map_size_class(df_sub, layer_map, unit_match="operations", class_match="layers", out_col="layer_ops_size")
map_size_class(df_sub, layer_map, unit_match="operations", class_match="broilers", out_col="broiler_ops_size")

map_size(df_sub, cattle_inv_map, unit_match="operations", out_col="cattle_ops_size_inv")
map_size(df_sub, hog_inv_map, unit_match="operations", out_col="hog_ops_size_inv")
map_size(df_sub, milk_cows_map, unit_match="operations", out_col="dairy_ops_size_inv")
map_size(df_sub, breeding_hogs_map, unit_match="operations", out_col="breed_hog_ops_size_inv")
map_size(df_sub, cattle_inv_map_no_cows, unit_match="operations", out_col="cattle_senzcow_ops_size_inv")
map_size(df_sub, cattle_feed_map, unit_match="operations", out_col="cattle_feed_ops_size_inv")
map_size(df_sub, beef_cows_map, unit_match="operations", out_col="beef_ops_size_inv")

# ===== THRESHOLDS -> SIZE CLASS =====
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

### FIPS 
from functions import generate_fips

if "fips_generated" not in df2.columns:
    df2 = generate_fips(df2, state_col="state_fips_code", city_col="county_code")
    if "FIPS_generated" in df2.columns and "fips_generated" not in df2.columns:
        df2 = df2.rename(columns={"FIPS_generated": "fips_generated"})


for col, (med, lrg) in col_thresholds.items():
    codes = df2[col]
    mask = codes.notna()
    df2.loc[mask, "size_class"] = codes[mask].apply(categorize_code, args=(med, lrg))

# numeric ops
df2["ops_in_bin"] = pd.to_numeric(df2["value"].astype(str).str.replace(",", ""), errors="coerce")

# aggregate operations by size class
group_cols = ["year", "fips_generated", "size_class", "unit_desc", "commodity_desc"]
df2["sum_ops"] = df2.groupby(group_cols)["ops_in_bin"].transform("sum")

df2["sum_ops"].isna().mean()
df2["size_class"].value_counts(dropna=False)


# compact summary
summary = (
    df2.groupby(
        ["year", "fips_generated", "size_class", "commodity_desc"], 
        as_index=False
    )["ops_in_bin"]
    .sum()
    .rename(columns={"ops_in_bin": "sum_ops"})
)

# export
from datetime import date
today_str = date.today().strftime("%Y-%m-%d")
outpath = f"/Users/allegrasaggese/Dropbox/Mental/Data/clean/{today_str}_cafo_ops_by_size.csv"
summary.to_csv(outpath, index=False)
outpath





# put all mappings together first 
# TO QA: do some of these mappings double count through different statistical items - i.e. ops and heads
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


# ADDITIONAL MAPPINGS CREATED TO INCREASE number of data points in 2025
beef_cows_map = {
 "inventory of beef cows: (1 to 9 head)":1,
 "inventory of beef cows: (10 to 19 head)":2,
 "inventory of beef cows: (20 to 49 head)":3,
 "inventory of beef cows: (50 to 99 head)":4,
 "inventory of beef cows: (100 to 199 head)":5,
 "inventory of beef cows: (200 to 499 head)":6,
 "inventory of beef cows: (500 or more head)":7
}

cattle_inv_map_no_cows = {
"inventory of cattle, (excl cows): (1 to 9 head)": 1,
"inventory of cattle, (excl cows): (10 to 19 head)": 2,
"inventory of cattle, (excl cows): (100 to 199 head)": 3,
"inventory of cattle, (excl cows): (20 to 49 head)": 4,
"inventory of cattle, (excl cows): (200 to 499 head)": 5,
"inventory of cattle, (excl cows): (50 to 99 head)": 6
}


cattle_feed_map ={"inventory of cattle on feed: (1 to 19 head)":1,
"inventory of cattle on feed: (1 to 9 head)":2,
"inventory of cattle on feed: (10 to 19 head)":3,
"inventory of cattle on feed: (100 to 199 head)":4,
"inventory of cattle on feed: (20 to 49 head)":5,
"inventory of cattle on feed: (200 to 499 head)":6,
"inventory of cattle on feed: (50 to 99 head)":7,
"inventory of cattle on feed: (500 or more head)":8}

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


# b/f adding in mappings - check that unit_match and domaincat_desc are comprehensive 
mask = df_sub["domaincat_desc"].isin(cattle_inv_map.keys())
df_sub.loc[mask, "unit_desc"].value_counts()


# apply mappings (operations only)
# chickens: split layers vs broilers using class_desc
map_size_class(df_sub, layer_map, unit_match="operations", class_match="layers", out_col="layer_ops_size")
map_size_class(df_sub, layer_map, unit_match="operations", class_match="broilers", out_col="broiler_ops_size")

map_size(df_sub, cattle_inv_map, unit_match="operations", out_col="cattle_ops_size_inv")
map_size(df_sub, hog_inv_map, unit_match="operations", out_col="hog_ops_size_inv")
map_size(df_sub, milk_cows_map, unit_match="operations", out_col="dairy_ops_size_inv")
map_size(df_sub, breeding_hogs_map, unit_match="operations", out_col="breed_hog_ops_size_inv")
map_size(df_sub, cattle_inv_map_no_cows, unit_match="operations", out_col="cattle_senzcow_ops_size_inv")
map_size(df_sub, cattle_feed_map, unit_match="operations", out_col="cattle_feed_ops_size_inv")
map_size(df_sub, beef_cows_map, unit_match="operations", out_col="beef_ops_size_inv")


# note we're excluding sales in this round - no mapping possible 
#map_size(df_sub, cattle_incl_calves_sales_map, unit_match="operations", out_col="cattle_calves_ops_size_sales")
#map_size(df_sub, cattle_feed_sales_map, unit_match="operations", out_col="cattle_feed_ops_size_sales")
#map_size(df_sub, cattle_500lbs_sales_map, unit_match="operations", out_col="cattle_500lbs_ops_size_sales")
#map_size(df_sub, calves_sales_map, unit_match="operations", out_col="calves_ops_size_sales")


# inspect after mapping
df_small = df_sub.sample(n=500, random_state=1)
df_sub.shape
df_sub.head()

print(type(df_sub))
print(df_sub.shape)
print(df_sub.head())
print(df_sub.columns.tolist())

# drop unneccessary rows (total or unspecified)
df_sub = df_sub[df_sub['domaincat_desc'] != 'unspecified']


### DEEPER DIVE MAPPING - for ambiguous domaincat_desc - try to assign based on short_desc 
#### IN FINAL CODE - WE CAN IGNORE THIS --- USING JUST INVENTORY FOR NOW 
df["short_desc"] = df["short_desc"].astype("string")

def map_ambiguous(df, mapping, keywords, unit_label):
    """
    df: dataframe
    mapping: mapping dict (e.g., sales_map)
    keywords: list of substrings to look for in short_desc related to the type of animal 
    unit_label: suffix for the output column (e.g., 'sales_size')
    """
    for kw in keywords:
        mask = (
            df['domaincat_desc'].isin(mapping.keys()) &
            df['short_desc'].str.contains(kw, case=False, na=False)
        )
        out_col = f"{kw.lower().replace(',', '').replace(' ', '_')}_{unit_label}"
        df[out_col] = df.loc[mask, 'domaincat_desc'].map(mapping).astype('Int64')
        
keywords = ['poultry', 'chickens, layers', 'chickens, broilers', 'cattle', 'hogs']


# mappings 
sales_map = {"sales: (1 to 1,999 head)":1,
 "sales: (100,000 to 199,999 head)":2,
 "sales: (2,000 to 59,999 head)":3,
 "sales: (200,000 to 499,999 head)":4,
 "sales: (500,000 or more head)":5,
 "sales: (60,000 to 99,999 head)":6}


inv_map = {"inventory: (1 to 49 head)": 1,
 "inventory: (50 to 99 head)": 2,
 "inventory: (100 to 399 head)": 3,
 "inventory: (400 to 3,199 head)": 4,
 "inventory: (3,200 to 9,999 head)": 5,
 "inventory: (10,000 to 19,999 head)": 6,
 "inventory: (20,000 to 49,999 head)": 7,
 "inventory: (50,000 to 99,999 head)": 8,
 "inventory: (100,000 or more head)": 9}


# apply mappings for ambiguous categorization
map_ambiguous(df, sales_map, keywords, unit_label='sales_size') # picking up broilers - good 
map_ambiguous(df, inv_map, keywords, unit_label='ops_size') # same number of layers as from other mapping


# quick QA - ambiguous mapping yielded no matches within the sales 
cols = [c for c in df_sub.columns if c.endswith(("_sales_size", "_ops_size"))]
df_sub[cols].describe(include="all")
df_sub[cols].isna().mean().sort_values(ascending=False)  # percent missing
df_sub[cols].notna().sum()



##################### -- QA POST MAPPING --  #########################
df_sub.columns.tolist()

# check outcol fill rate --- if the mapping is working 
out_cols = [
    "layer_ops_size",
    "broiler_ops_size",
    "cattle_ops_size_inv",
    "hog_ops_size_inv",
    "dairy_ops_size_inv",
    "breed_hog_ops_size_inv",
    "cattle_senzcow_ops_size_inv",
    "cattle_feed_ops_size_inv",
    "beef_ops_size_inv"
]


# other qa - check the set of the domaincat_desc 
cols_to_create_mapping_for = set(df_sub["domaincat_desc"].unique()) - set(layer_map.keys())
# other qa - check mean of missing of a few of the cols 
for c in out_cols:
    print(c, df_sub[c].isna().mean() * 100)

# count and percent of missing for each
na_summary = df_sub[out_cols].isna().agg(['sum', 'mean']).T
na_summary['mean'] = na_summary['mean'] * 100  # convert to percent
na_summary.columns = ['n_missing', 'pct_missing']
print(na_summary) # VERY LOW b/c each row should only match on ONE COLUMN - so its not really telling us anything


# check count of each matching --- see you still get plenty of obs for chickens (i.e. neary 170K county observations)
mask = df_sub['domaincat_desc'].isin(layer_map.keys())
print(mask.sum(), "rows match layer_map keys")
print(df_sub.loc[mask, 'domaincat_desc'].unique()[:10])

unmatched = set(df_sub['domaincat_desc'].unique()) - set(layer_map.keys())
print(sorted([x for x in unmatched if "inventory" in x.lower()][:10])) # smaller set of unmatched rows - still need if mapping is not picking up as many operations as we would like 


test_key = "inventory: (1 to 49 head)"
print(test_key in df_sub['domaincat_desc'].unique())
df_sub['mapped_layer'] = df_sub['domaincat_desc'].map(layer_map)
print(df_sub['mapped_layer'].notna().mean()) #1.2% of total df  

# going to compare raw data numbers on those two cols with the mapping output - should be the same count of rows 
matching = df_sub[
    (df_sub['unit_desc'].str.lower() == 'operations')
    & (df_sub['domaincat_desc'].isin(layer_map.keys()))
]
print(len(matching))
print(matching['domaincat_desc'].unique()) # same amount 





###### ###### ###### ###### QA of the threshold development ######### ###### ###### ###### ###### 

# review the interim dataset to understand what the value column is - value is the numeric value assoc. with the unit (inventory, sales, etc)
subset = df_sub.iloc[:500].copy()
subset.to_csv("first_500_rows.csv", index=False)

# set value to numeric 
df_sub["value"] = (
    df_sub["value"]
    .astype(str)                     # ensure string
    .str.strip()                     # remove spaces
    .str.replace(",", "", regex=False)  # remove commas
    .replace({"(D)": pd.NA, "": pd.NA}) # turn (D) and empty to NA
    .pipe(pd.to_numeric, errors="coerce")  # convert to numbers
)

# count of operations in each inventory bin (per row)
df_sub["ops_in_bin"] = df_sub["value"]



# group by year
group_cols = ['FIPS_generated','year']
# add in QA to raise error in the case that these essential columns are not found 
for c in group_cols:
    if c not in df_sub.columns:
        raise KeyError(f"required column '{c}' not found in df")



###### ###### From threshold - now we can sum the columns to get the counts of the total inventory at size level and the count of operations

# test on subset 
temp = (
    df_sub.dropna(subset=['hog_ops_size_inv'])
          .groupby(['FIPS_generated','year','hog_ops_size_inv','unit_desc'], as_index=False)['value']
          .sum()
          .rename(columns={'value': 'hog_ops_size_inv_sum'})
)

tempsmall = temp.head(500) # shows max is 20 farms per fips which seems about right - of a sample recall

# run across all cols 
size_cols = out_cols # already created a vector of cols to sum over

# run loop to create SUMMED values for total count of operations and total 
# count of inventory within all farms at a given size classification
for col in size_cols:
    temp = (
        df_sub.dropna(subset=[col])
              .groupby(['FIPS_generated','year', col, 'unit_desc'], as_index=False)['ops_in_bin']
              .sum()
              .rename(columns={'ops_in_bin': f'{col}_sum'})
    )

    df_sub = df_sub.merge(
        temp,
        on=['FIPS_generated','year', col, 'unit_desc'],
        how='left'
    )

# Now we have summation by each INVENTORY CLASS but we need to use the CAFO thresholds to group those inventory classes 
df_sub.columns.tolist() # list of cols 
df2 = df_sub.copy() # make a copy 
inv_cols = [c for c in df2.columns if c.endswith('_inv')] # all threshold cols 

# ensure size_class column exists even if no mappings were applied
if "size_class" not in df2.columns:
    df2["size_class"] = pd.Series(pd.NA, index=df2.index, dtype="string")

# recall our mapping from above (simplified into a mapping) 
cutoffs = {
    'hog':     {'med': 6, 'lrg': 7},
    'cattle':  {'med': 6, 'lrg': 7},
    'layer':   {'med': 7, 'lrg': 9},
    'broiler': {'med': 3, 'lrg': 5}
}

# create mapping from hogs to cutoff values 
col_thresholds = {
    # layers
    'layer_ops_size':                 (layer_cutoff_med,  layer_cutoff_lrg),
    'broiler_ops_size':               (broiler_cutoff_med, broiler_cutoff_lrg),

    # cattle family (incl. dairy, beef, feedlots, etc.)
    'cattle_ops_size_inv':            (cattle_cutoff_med, cattle_cutoff_lrg),
    'dairy_ops_size_inv':             (cattle_cutoff_med, cattle_cutoff_lrg),
    'cattle_senzcow_ops_size_inv':    (cattle_cutoff_med, cattle_cutoff_lrg),
    'cattle_feed_ops_size_inv':       (cattle_cutoff_med, cattle_cutoff_lrg),
    'beef_ops_size_inv':              (cattle_cutoff_med, cattle_cutoff_lrg),

    # hog family
    'hog_ops_size_inv':               (hog_cutoff_med,    hog_cutoff_lrg),
    'breed_hog_ops_size_inv':         (hog_cutoff_med,    hog_cutoff_lrg),
}


# create function to categorize into text size for labelling 
def categorize_code(v, med, lrg):
    if pd.isna(v): return pd.NA
    if v < med:   return "small"
    if v < lrg:   return "medium"
    return "large"

df2['size_class'] = pd.Series(pd.NA, index=df2.index, dtype='string')
df2['size_class'].isna().sum() # getting .3855 match rate ---- ok 

# create labels in one column
for col, (med, lrg) in col_thresholds.items():
    codes = df2[col]
    mask  = codes.notna()
    df2.loc[mask, 'size_class'] = codes[mask].apply(categorize_code, args=(med, lrg))



# now sum the value cols - total operations by size class x animal classification
group_cols = ['year', 'FIPS_generated', 'size_class', 'unit_desc', 'commodity_desc']
df2['sum_ops'] = df2.groupby(group_cols)['ops_in_bin'].transform('sum')

# QA the range of values for the sums
df2.groupby('unit_desc')['sum_ops'].describe() # reasonable 


# export the CAFO data 
clean_cafo = f"{today_str}_cafo_annual_df.csv"
ag_path2 = os.path.join(outf, clean_cafo)
df2.to_csv(ag_path2, index=False)



### PLOT SUMMARY STATS  


outdir = "/Users/allegrasaggese/Dropbox/Mental/Data/clean/figs"
os.makedirs(outdir, exist_ok=True)

# 1) Total ops over time by commodity (line)
ts = (summary
      .groupby(["year","commodity_desc"], as_index=False)["sum_ops"]
      .sum())

plt.figure(figsize=(10,6))
sns.lineplot(data=ts, x="year", y="sum_ops", hue="commodity_desc", marker="o")
plt.title("Total Operations Over Time by Commodity")
plt.ylabel("Operations (sum)")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "ops_over_time_by_commodity.png"), dpi=200)
plt.close()

# 2) Stacked ops by size_class over time (all commodities combined)
ts_size = (summary
           .groupby(["year","size_class"], as_index=False)["sum_ops"]
           .sum())

pivot = ts_size.pivot(index="year", columns="size_class", values="sum_ops").fillna(0)
pivot = pivot[["small","medium","large"]]  # consistent order

pivot.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Operations by Size Class Over Time (All Commodities)")
plt.ylabel("Operations (sum)")
plt.xlabel("Year")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "ops_by_size_over_time_stacked.png"), dpi=200)
plt.close()

# 3) Latest year snapshot: ops by size_class × commodity
latest_year = summary["year"].max()
snap = summary[summary["year"] == latest_year]
snap_g = (snap.groupby(["commodity_desc","size_class"], as_index=False)["sum_ops"].sum())

plt.figure(figsize=(10,6))
sns.barplot(data=snap_g, x="commodity_desc", y="sum_ops", hue="size_class")
plt.title(f"Operations by Size Class (Year {latest_year})")
plt.ylabel("Operations (sum)")
plt.xlabel("Commodity")
plt.tight_layout()
plt.savefig(os.path.join(outdir, f"ops_by_size_{latest_year}.png"), dpi=200)
plt.close()