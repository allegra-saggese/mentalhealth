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
print("Columns identical to baseline after concat:", same_cols) # YIELDING FALSE why? 
len(base_cols) == len(list(combined.columns)) # TRUE 
# manual inspection shows they are the same --- need to check the QA code above 


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

# to see what the grouping level is - lets investigate unique values
grouped = combined.groupby(["STATE_FIPS_CODE", "COUNTY_CODE", "YEAR"])

# count unique values for each column within each group
uniq_counts = grouped.nunique()

# see which columns have more than 1 unique value per (fips, year)
problem_cols = (uniq_counts > 1).any()
problem_cols = problem_cols[problem_cols].index.tolist()
print("Columns that vary within (fips, year):", problem_cols)

# print unique values for each problem column
for col in problem_cols: 
    print(f"\n--- {col} ---")
    print(df[col].unique())

# check to see if we should use the code
vals = combined["BEGIN_CODE"].dropna().unique()
vals.sort()

print("Unique values:", vals)
print("Is it a full range? ", 
      set(range(vals.min(), vals.max() + 1)) == set(vals))
# FALSE - only contains 0,12

# other levels are class, group, and commodity -- need to decide how to sor
## or if I should summarize by the year <> fips -- as it should be counts / values from all farms within this 
## don't think its possible to simply collapse

# make ag raw df for export 
ag_raw_df = clean_cols(combined)

# create total fips code 
ag_raw_df = generate_fips(ag_raw_df, state_col="state_fips_code", city_col="county_code")


