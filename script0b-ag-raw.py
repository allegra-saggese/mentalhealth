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

######## ISOLATE FOR ONLY AG OF INTEREST ##########

# start with largest categorizations first 






# prep ag raw df for pivot, then export and saving  
ag_raw_df = clean_cols(combined)
ag_raw_df.head()

# create total fips code 
ag_raw_df = generate_fips(ag_raw_df, state_col="state_fips_code", city_col="county_code")
ag_raw_df.columns






keys = ["FIPS_generated", "year", "group_desc"]
cols = ["commodity_desc", "agg_level_desc"]     # subgroups need to differ on the unit description and the statistical description i.e. sales, inventory, or operations are all measuring different things about the CAFO 
val  = "value"                      # the value to spread 

###### ERROR HERE ON THE PIVOT - need to fix COLS for the subgroup --- NEED TO COME BACK TO THIS 
wide_TEST = (
    ag_raw_df.pivot_table(
        index=keys,
        columns=cols,               # <- multiple columns become a MultiIndex
        values=val,
        aggfunc=" "             # will need to sum 
    )
    .reset_index()
)

# optional: if you prefer single-level columns, flatten them:
if isinstance(wide_TEST.columns, pd.MultiIndex):
    wide_TEST.columns = [
        "_".join([str(x) for x in tup if x != ""]) for tup in wide.columns
    ]


# export in this form -- create cafos after 
today_str = date.today().strftime("%Y-%m-%d")

clean_ag_census = f"{today_str}_ag_raw_full.csv"
ag_path = os.path.join(outf, clean_ag_census)
combined.to_csv(ag_path, index=False)



