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

# import ag data 
agfolder = os.path.join(inf, "usda")
agfiles = glob.glob(os.path.join(agfolder, "*.dta"))

agdfs = [pd.read_stata(file) for file in agfiles]

print(f"Loaded {len(agdfs)} .dta files")


################## DATA CLEANING FOR ALL AG DATA ######################

# check head
agdfs[0].head(20)

# check all columns match across all the years 
base_cols = set(agdfs[0].columns)
all_match = True

for i, df in enumerate(agdfs, 1):
    cols = set(df.columns)
    if cols != base_cols:
        all_match = False
        only_in_base = sorted(base_cols - cols)
        only_in_df   = sorted(cols - base_cols)
        name = os.path.basename(dta_files[i-1]) if 'dta_files' in globals() else f"df_{i}"
        print(f"\nColumns differ for {name}:")
        if only_in_base:
            print("  Missing (present in df_1, absent here):", only_in_base)
        if only_in_df:
            print("  Extra (present here, absent in df_1):  ", only_in_df)

print("\nAll dataframes share identical columns:", all_match) # all match = true 

# list all cols 
all_cols = set().union(*[set(df.columns) for df in agdfs])
unique_cols_sorted = sorted(all_cols)
print("Total unique columns across all DFs:", len(unique_cols_sorted))
print(unique_cols_sorted)

# check col types 
for i, df in enumerate(agdfs, 1):
    print(f"\n--- Dataframe {i} ---")
    print(df.dtypes) # fips code = int
    

# row bind the ag data 
base_set  = set(base_cols)

# verify every df matches the baseline (and align order)
problems = []
dfs_aligned = []
for i, d in enumerate(agdfs, 1):
    cols_set = set(d.columns)
    missing  = base_set - cols_set
    extra    = cols_set - base_set
    if missing or extra:
        problems.append((i, sorted(missing), sorted(extra)))
    # align to baseline order; this also drops any extras if present
    df_i = d.reindex(columns=base_cols)
    dfs_aligned.append(df_i)

# row-bind (union is identical to baseline since we reindexed)
combined = pd.concat(dfs_aligned, ignore_index=True)  

# post-check: confirm columns unchanged by comparing the list of cols 
base_cols = list(base_cols)
same_cols = list(combined.columns) == base_cols
print("Columns identical to baseline after concat:", same_cols) # TRUE - needed to convert base_cols to a list  
len(base_cols) == len(list(combined.columns)) # TRUE 

# manual inspection shows they are the same 
only_in_combined = [c for c in list(combined.columns) if c not in base_cols]
only_in_base = [c for c in base_cols if c not in list(combined.columns)]
print("Columns in combined but not in base:", only_in_combined)
print("Columns in base but not in combined:", only_in_base)


# print any issues found in step 1 -- empty set so yields nothing (but it serves as a QA road block for future analysis)
for i, missing, extra in problems:
    if missing:
        print(f"df_{i} MISSING cols vs baseline: {missing}")
    if extra:
        print(f"df_{i} EXTRA cols vs baseline (dropped by reindex): {extra}")
    

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
=======
        
        
# sense check the length of the dataframe against the FIPS data to see how many observations there are 
fips_sense = os.path.join(outf, "2025-11-10_fips_full.csv") 
fips_df = pd.read_csv(fips_sense) 
>>>>>>> Stashed changes

for i, df in enumerate(agdfs, 1):
    print(f"\n--- Dataframe {i} number of rows ---")
    print(len(df)) 

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

# INTERPRETATION: about 14 million farm level observations --  2002-2021
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
del ag_raw_df, agdfs, agfiles, all_match, b, clean_ag_census, d, df, df_big, df_i
del dup_counts, dup_counts_17, extra, fips_sense, i, matches, missing, n_forward, new_frames, new_rows
del only_in_base, only_in_combined, problems, same_cols, year_col, y

# list of cols that will be created --- MAY REMOVE 
CAFO_cols = ("broiler_cafos_lrg_op",
  "broiler_cafos_med_op",
  "layer_cafos_lrg_op",
  "layer_cafos_med_op",
  "cattle_cafos_INV_lrg_op",
  "cattle_cafos_INV_med_op",
  "cattle_cafos_INV_lrg_head",
  "cattle_cafos_INV_med_head",
  "cattle_cafos_SALES_lrg_op",
  "cattle_cafos_SALES_med_op",
  "cattle_cafos_SALES_lrg_head",
  "cattle_cafos_SALES_med_head",
  "hog_cafos_INV_lrg_op",
  "hog_cafos_INV_med_op",
  "hog_cafos_INV_lrg_head",
  "hog_cafos_INV_med_head",
  "hog_cafos_SALES_lrg_op",
  "hog_cafos_SALES_med_op",     
  "hog_cafos_SALES_lrg_head",
  "hog_cafos_SALES_med_head"   
)

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


# make a df copy for ease 
df = ag_iterated.copy()

# put it all in strings
df[['domaincat_desc','unit_desc']] = df[['domaincat_desc','unit_desc']].astype("string").apply(lambda s: s.str.strip())
df["domaincat_desc"] = df["domaincat_desc"].astype("string").str.strip()

# normalize strings
df[['domaincat_desc','unit_desc']] = (
    df[['domaincat_desc','unit_desc']]
    .astype("string")
    .apply(lambda s: s.str.strip().str.lower())
)


# make a wrapper to map the inventory the same way in all of them 
def map_size(df, mapping, unit_match, out_col):
    mask = df['unit_desc'] == unit_match
    df[out_col] = df['domaincat_desc'].map(mapping).where(mask, other=pd.NA).astype("Int64")
    
    

# check the col for chickens
col = "commodity_desc"
# normalized, sorted uniques (exclude NA)
uniques = sorted(df[col].dropna().astype(str).str.strip().str.lower().unique())
print(len(uniques), "unique")
for v in uniques:
    print(v)


col = "unit_desc"
# normalized, sorted uniques (exclude NA)
uniques = sorted(df[col].dropna().astype(str).str.strip().str.lower().unique())
print(len(uniques), "unique")
for v in uniques:
    print(v)
    

# check unit units 
    col = "unit_desc"
    # normalized, sorted uniques (exclude NA)
    uniques = sorted(df[col].dropna().astype(str).str.strip().str.lower().unique())
    print(len(uniques), "unique")
    for v in uniques:
        print(v)
    
# clean the domain description before making the map 
df["domaincat_desc"] = (
    df["domaincat_desc"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# CREATE SUBSET OF THE DATA so we work only with commodities of interest 
comms_of_interest = ["CATTLE", "CHICKENS", "MILK", "EGGS", "HOGS"]
df_sub = df[df['commodity_desc'].isin(comms_of_interest)]

# subset to only units of interest (count of operations, count of inventory)
units_of_interest = ["head", "operations"]
df_sub = df_sub[df_sub['unit_desc'].isin(units_of_interest)]



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

# sample of the dataframe for output
sample_head = df_sub.loc[mask & (df_sub["unit_desc"] == "head")].sample(20, random_state=1)
sample_ops  = df_sub.loc[mask & (df_sub["unit_desc"] == "operations")].sample(20, random_state=1)
cattletst = pd.concat([sample_head, sample_ops])


# apply mappings
map_size(df_sub, layer_map, unit_match="operations", out_col="layer_ops_size") 
map_size(df_sub, layer_map, unit_match="head", out_col="layer_head_size")

map_size(df_sub, cattle_inv_map, unit_match="operations", out_col="cattle_ops_size_inv")
map_size(df_sub, cattle_inv_map, unit_match="head", out_col="cattle_head_size_inv")

map_size(df_sub, hog_inv_map, unit_match="operations", out_col="hog_ops_size_inv")
map_size(df_sub, hog_inv_map, unit_match="head", out_col="hog_head_size_inv")

map_size(df_sub, milk_cows_map, unit_match="operations", out_col="dairy_ops_size_inv")
map_size(df_sub, milk_cows_map, unit_match="head", out_col="dairy_head_size_inv")

map_size(df_sub, breeding_hogs_map, unit_match="operations", out_col="breed_hog_ops_size_inv")
map_size(df_sub, breeding_hogs_map, unit_match="head", out_col="breed_hog_head_size_inv")

map_size(df_sub, cattle_inv_map_no_cows, unit_match="operations", out_col="cattle_senzcow_ops_size_inv")
map_size(df_sub, cattle_inv_map_no_cows, unit_match="head", out_col="cattle_senzcow_head_size_inv")

map_size(df_sub, cattle_feed_map, unit_match="operations", out_col="cattle_feed_ops_size_inv")
map_size(df_sub, cattle_feed_map, unit_match="head", out_col="cattle_feed_map_head_size_inv")

map_size(df_sub, beef_cows_map, unit_match="operations", out_col="beef_ops_size_inv")
map_size(df_sub, beef_cows_map, unit_match="head", out_col="beef_map_head_size_inv")


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
    "layer_head_size",
    "cattle_ops_size_inv",
    "cattle_head_size_inv",
    "hog_ops_size_inv",
    "hog_head_size_inv",
    "dairy_ops_size_inv",
    "dairy_head_size_inv",
    "breed_hog_ops_size_inv",
    "breed_hog_head_size_inv",
    "cattle_senzcow_ops_size_inv",
    "cattle_senzcow_head_size_inv",
    "cattle_feed_ops_size_inv",
    "cattle_feed_map_head_size_inv",
    "beef_ops_size_inv",
    "beef_map_head_size_inv"
  #  "cattle_calves_ops_size_sales",
  #  "cattle_feed_ops_size_sales",
  #  "cattle_500lbs_ops_size_sales",
  #  "calves_ops_size_sales"
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
              .groupby(['FIPS_generated','year', col, 'unit_desc'], as_index=False)['value']
              .sum()
              .rename(columns={'value': f'{col}_sum'})
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
    'layer_head_size':                (layer_cutoff_med,  layer_cutoff_lrg),

    # cattle family (incl. dairy, beef, feedlots, etc.)
    'cattle_ops_size_inv':            (cattle_cutoff_med, cattle_cutoff_lrg),
    'cattle_head_size_inv':           (cattle_cutoff_med, cattle_cutoff_lrg),
    'dairy_ops_size_inv':             (cattle_cutoff_med, cattle_cutoff_lrg),
    'dairy_head_size_inv':            (cattle_cutoff_med, cattle_cutoff_lrg),
    'cattle_senzcow_ops_size_inv':    (cattle_cutoff_med, cattle_cutoff_lrg),
    'cattle_senzcow_head_size_inv':   (cattle_cutoff_med, cattle_cutoff_lrg),
    'cattle_feed_ops_size_inv':       (cattle_cutoff_med, cattle_cutoff_lrg),
    'cattle_feed_map_head_size_inv':  (cattle_cutoff_med, cattle_cutoff_lrg),
    'beef_ops_size_inv':              (cattle_cutoff_med, cattle_cutoff_lrg),
    'beef_map_head_size_inv':         (cattle_cutoff_med, cattle_cutoff_lrg),

    # hog family
    'hog_ops_size_inv':               (hog_cutoff_med,    hog_cutoff_lrg),
    'hog_head_size_inv':              (hog_cutoff_med,    hog_cutoff_lrg),
    'breed_hog_ops_size_inv':         (hog_cutoff_med,    hog_cutoff_lrg),
    'breed_hog_head_size_inv':        (hog_cutoff_med,    hog_cutoff_lrg),
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



# now sum the value cols - so that we have count of operations and aggregate inventory in CAFO x animal classification
group_cols = ['year', 'FIPS_generated', 'size_class', 'unit_desc', 'commodity_desc']
df2['sum_value'] = df2.groupby(group_cols)['value'].transform('sum')

# QA the range of values for the sums
df2.groupby('unit_desc')['sum_value'].describe() # reasonable 


# export the CAFO data 
clean_cafo = f"{today_str}_cafo_annual_df.csv"
ag_path2 = os.path.join(outf, clean_cafo)
df2.to_csv(ag_path2, index=False)


