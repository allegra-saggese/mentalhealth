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

# post-check: confirm columns unchanged
same_cols = list(combined.columns) == base_cols
combocols = list(combined.columns)
combocols == base_cols
print("Columns identical to baseline after concat:", same_cols) # YIELDING FALSE why? 
len(base_cols) == len(list(combined.columns)) # TRUE 

# manual inspection shows they are the same --- need to check the QA code above 
only_in_combined = [c for c in combocols if c not in base_cols]
only_in_base = [c for c in base_cols if c not in combocols]
print("Columns in combined but not in base:", only_in_combined)
print("Columns in base but not in combined:", only_in_base)

# trying again with the ordering bc i just can't seem to find why there is the fail 
set_combined = set(combocols)
set_base = set(base_cols)

print("Only in combined:", sorted(set_combined - set_base))
print("Only in base:", sorted(set_base - set_combined))
set_combined == set_base # NOW YIELDING TRUE --- it was simply an ordering thing


# print any issues found in step 1
for i, missing, extra in problems:
    if missing:
        print(f"df_{i} MISSING cols vs baseline: {missing}")
    if extra:
        print(f"df_{i} EXTRA cols vs baseline (dropped by reindex): {extra}")
        
        
# sense check the length of the dataframe against the FIPS data to see how many observations there are 
fips_sense = os.path.join(outf, "2025-08-11_fips_full.csv") 
fips_df = pd.read_csv(fips_sense) 

for i, df in enumerate(agdfs, 1):
    print(f"\n--- Dataframe {i} number of rows ---")
    print(len(df)) 

# fips check by year of data
fdf_2002 = fips_df[fips_df["year"] == 2002]     
fdf_2007 = fips_df[fips_df["year"] == 2007]     
fdf_2012 = fips_df[fips_df["year"] == 2012]     
fdf_2017 = fips_df[fips_df["year"] == 2017]     
# length is significantly shorter of course - need to understand the repetition in the USDA data 

########## NEED TO CHECK HERE ON THE FIPS COLS AGAIN! THERE IS AN ISSUE WITH THE 2012, 2017 data 
# multiple entries where they were combined in that year, so need to keep only the unique one 
dup_counts = fdf_2012.groupby(["fips", "year"]).size().reset_index(name="n_rows")
print(dup_counts["n_rows"].value_counts())  # quick frequency check
fdf_2012_dedup = fdf_2012.drop_duplicates(subset=["fips", "year"], keep="first")

dup_counts_17 = fdf_2017.groupby(["fips", "year"]).size().reset_index(name="n_rows")
print(dup_counts_17["n_rows"].value_counts())  # quick frequency check
fdf_2017_dedup = fdf_2017.drop_duplicates(subset=["fips", "year"], keep="first")


# to see what the grouping level is - lets investigate unique values
grouped = combined.groupby(["STATE_FIPS_CODE", "COUNTY_CODE", "YEAR"])

# count unique values for each column within each group (s.t. its combined by state/county/year)
uniq_counts = grouped.nunique() # see table

# see which columns have more than 1 unique value per (fips, year) - most should as its gonna have the stats on all types of farming ops in that county
problem_cols = (uniq_counts > 1).any()
problem_cols = problem_cols[problem_cols].index.tolist()
print("Columns that vary within (fips, year):", problem_cols)

# print unique values for each problem column to help us find the unique CAFO identifier
for col in problem_cols: 
    print(f"\n--- {col} ---")
    print(df[col].unique())


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

df_big = df_big.drop_duplicates(ignore_index=True) # drop dupes - although it should be the same 

# export in this form -- create cafos after 
today_str = date.today().strftime("%Y-%m-%d")

clean_ag_census = f"{today_str}_ag_annual_df.csv"
ag_path = os.path.join(outf, clean_ag_census)
df_big.to_csv(ag_path, index=False)


############ COLLAPSE THE DATA FOR COUNTY LEVEL EASE (creating columns for the CAFOs instead)

# unhash next line if you don't want to remerge the ag data 
ag_path = os.path.join(outf, "2025-10-19_ag_annual_df.csv")
ag_raw = pd.read_csv(ag_path)

ag_raw.columns.tolist()

# list of cols that will be created 
CAFO_cols = c("broiler_cafos_lrg_op",
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
))

# CAFO size limits, and can adjust these values for the S/M/L CAFO development 
broiler_cutoff_lrg <- 5
broiler_cutoff_med <- 3
layer_cutoff_lrg <- 9
layer_cutoff_med <- 7
cattle_cutoff_lrg <- 7
cattle_cutoff_med <- 6
hog_cutoff_lrg <- 7
hog_cutoff_med <- 1


# make a df copy for ease 
df = ag_raw

# put it all in strings
df[['domaincat_desc','unit_desc']] = df[['domaincat_desc','unit_desc']].astype("string").apply(lambda s: s.str.strip())

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


# put all mappings together first 
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

cattle_sales_map = {
 "sales of cattle, incl calves: (1 to 9 head)":1,
 "sales of cattle, incl calves: (10 to 19 head)":2,
 "sales of cattle, incl calves: (20 to 49 head)":3,
 "sales of cattle, incl calves: (50 to 99 head)":4,
 "sales of cattle, incl calves: (100 to 199 head)":5,
 "sales of cattle, incl calves: (200 to 499 head)":6,
 "sales of cattle, incl calves: (500 or more head)":7
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

hog_sales_map = {
 "sales of hogs: (1 to 24 head)":1,
 "sales of hogs: (25 to 49 head)":2,
 "sales of hogs: (50 to 99 head)":3,
 "sales of hogs: (100 to 199 head)":4,
 "sales of hogs: (200 to 499 head)":5,
 "sales of hogs: (500 to 999 head)":6,
 "sales of hogs: (1,000 or more head)":7
}

# broiler/sales maps (same as earlier broiler mapping)
broiler_map = {
 "sales: (1 to 1,999 head)":1,
 "sales: (2,000 to 59,999 head)":2,
 "sales: (60,000 to 99,999 head)":3,
 "sales: (100,000 to 199,999 head)":4,
 "sales: (200,000 to 499,999 head)":5,
 "sales: (500,000 or more head)":6
}

# apply mappings
map_size(df, layer_map, unit_match="operations", out_col="layer_ops_size")
map_size(df, cattle_inv_map, unit_match="operations", out_col="cattle_ops_size_inv")
map_size(df, cattle_sales_map, unit_match="operations", out_col="cattle_ops_size_sales")
map_size(df, hog_inv_map, unit_match="operations", out_col="hog_ops_size_inv")
map_size(df, hog_sales_map, unit_match="operations", out_col="hog_ops_size_sales")

map_size(df, broiler_map, unit_match="head", out_col="broiler_head_size")
map_size(df, layer_map, unit_match="head", out_col="layer_head_size")
map_size(df, cattle_inv_map, unit_match="head", out_col="cattle_head_size_inv")
map_size(df, cattle_sales_map, unit_match="head", out_col="cattle_head_size_sales")
map_size(df, hog_inv_map, unit_match="head", out_col="hog_head_size_inv")
map_size(df, hog_sales_map, unit_match="head", out_col="hog_head_size_sales")
















