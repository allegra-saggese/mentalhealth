#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 11:05:15 2025

@author: allegrasaggese
"""


# load packages and workspaces
import sys, importlib.util
from collections import Counter
import re
import difflib

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

crimef = os.path.join(inf, "crime")
crimef = os.path.join(crimef, "total-v1")

csv_files = glob.glob(os.path.join(crimef, "*.csv"))

print(f"Found {len(csv_files)} CSV files")

# read all CSVs into a list of DataFrames
dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file, encoding="utf-8")
        df["source_file"] = os.path.basename(file)   # optional: keep filename
        dfs.append(df)
        print(f"Loaded: {os.path.basename(file)}  shape={df.shape}")
    except Exception as e:
        print(f"failed to read {os.path.basename(file)}: {e}")

# if you want to combine all of them into one big df
combined_df = pd.concat(dfs, ignore_index=True, sort=False)

# HUGE DF - lets preserve memory
combined_df.memory_usage(deep=True).sum() / 1e6  # about 7 GB of memory used 

bytes_used = combined_df.memory_usage(deep=True).sum()
print(f"{bytes_used/1e6:.2f} MB  (~{bytes_used/1024**3:.2f} GiB)")

# big cols
mem_cols = (combined_df.memory_usage(deep=True)
            .sort_values(ascending=False))
print(mem_cols.head(20))

# collapse to year, FIPS, race, sex 

keys = ["year", "state", "county", "fips", "sex_of_arrestee", 
        "race_of_arrestee", "ethnicity_of_arrestee"]

crime_cols = ['aggravated_assault',
              'driving_under_the_influence',
              'fondling_(incident_liberties/child_molest)',
              'incest',
              'intimidation',
              'kidnapping/abduction',
              'rape',
              'sexual_assault_with_an_object',
              'simple_assault',
              'statutory_rape',
              'source_file',
              'human_trafficking_-_commercial_sex_acts',
              'human_trafficking_-_involuntary_servitude']

def norm(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("–","-").replace("—","-")           # normalize unicode dashes
    s = re.sub(r"\s+", "_", s)                        # collapse whitespace to _
    s = re.sub(r"[^\w\-_/()]+", "_", s)               # keep word chars, -, _, /, ()
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# build normalized->actual map from df
colmap = {norm(c): c for c in df.columns}
norm_cols = list(colmap.keys())


# map desired crime cols to actual df columns
wanted_norm = [norm(c) for c in crime_cols]
missing = [orig for orig, n in zip(crime_cols, wanted_norm) if n not in colmap]
agg_cols = [colmap[n] for n in wanted_norm if n in colmap]

if missing:
    print("Not found (after normalization):", missing) # cant find the human trafficking codes 
    
# fuzzy match to find 
targets = [
    "human_trafficking_-_commercial_sex_acts",
    "human_trafficking_-_involuntary_servitude",
]

# 1) fuzzy matches
for t in targets:
    print(f"\nTarget: {t}")
    print(" close:", difflib.get_close_matches(t, norm_cols, n=5, cutoff=0.6))
    # 2) token/substring search
    toks = [r"human", r"traffic", r"traffick", r"commercial", r"sex", r"involuntary", r"servitude"]
    hits = [k for k in norm_cols if all(re.search(tok, k) for tok in ["human","traff"])]
    print(" human+traff hits:", hits[:10])
    any_tok = [k for k in norm_cols if any(re.search(tok, k) for tok in toks)]
    print(" any-token hits:", any_tok[:10])

# find the cols 
traffic_cols = [c for c in combined_df.columns if combined_df.columns.str.contains(r"human|traffic", case=False, regex=True).any()]

summary = (
    combined_df[traffic_cols]
    .isna()
    .sum()
    .rename("n_missing")
    .to_frame()
)
summary["n_filled"] = combined_df[traffic_cols].notna().sum()
summary["pct_filled"] = 100 * summary["n_filled"] / len(combined_df)
summary

# the human trafficking cols exist, but with lower percentage filled 
for c in combined_df.columns:
    if "human" in c.lower():
        print(repr(c))

combined_df.columns = (
    combined_df.columns
    .str.replace(r"[–—]", "-", regex=True)  # normalize dash types
    .str.strip()                            # remove leading/trailing spaces
)

[c for c in combined_df.columns if "human" in c.lower()]


# rename human trafficking columns and remake crime cols 
combined_df = combined_df.rename(columns={
    'human_trafficking_-_commercial_sex_acts ': 'human_traffic_comm_sex',
    'human_trafficking_-_involuntary_servitude ': 'human_traffic_inv_servitude'
})

# check colnames 
print(repr(combined_df.columns.tolist())) # STILL NOT RECGONIZING IT! 

for i, c in enumerate(combined_df.columns):
    if "human" in c.lower():
        print(i, repr(c))
        
# rename by position
combined_df.columns.values[27] = "human_traffic_comm_sex"
combined_df.columns.values[28] = "human_traffic_inv_servitude"
[c for c in combined_df.columns if "human" in c.lower()] # WORKS! 


# now remake the list of crime_cols 
crime_cols = ['aggravated_assault',
              'driving_under_the_influence',
              'fondling_(incident_liberties/child_molest)',
              'incest',
              'intimidation',
              'kidnapping/abduction',
              'rape',
              'sexual_assault_with_an_object',
              'simple_assault',
              'statutory_rape',
              'human_traffic_comm_sex',
              'human_traffic_inv_servitude'
              ]

# COLLAPSE TO RACE/SEX/FIPS/YR with count of incidents only 
collapsed = (
    combined_df.groupby(keys, as_index=False)[crime_cols].sum(min_count=1)
)

# total incidents across all crime types (per group)
collapsed["total_incidents"] = collapsed[crime_cols].sum(axis=1)

# count raw incident rows per group for sanity
collapsed["n_rows"] = df.groupby(keys).size().reset_index(name="n")["n"]

#check avg incident / yr (and demog decomp) collapse size 
7631205/209760 # 36.380649313501145

# check fips uniqueness
# group by year/state/county and count distinct FIPS in each
check = (
    combined_df.groupby(["year", "state", "county"])["fips"]
    .nunique()
    .reset_index(name="n_unique_fips")
)

# find any that violate uniqueness (more than one FIPS)
violations = check[check["n_unique_fips"] > 1]

print(f"Total groups with multiple FIPS: {len(violations)}") # no violations


#### COLLAPSE TO ELIMINATE DISAGGREGATION BY DEMOGRAPHICS 
# keys for the final panel
keys_fips = ["year", "state", "county", "fips"]

# keep only crime columns that exist
crime_cols_present = [c for c in crime_cols if c in collapsed.columns]

# aggregate to fips × year (drop sex/race/ethnicity)
agg_dict = {c: "sum" for c in crime_cols_present}
if "n_rows" in collapsed.columns:
    agg_dict["n_rows"] = "sum"   # optional: sum raw incident rows too

collapsed_fips_only = (
    collapsed.groupby(keys_fips, as_index=False)
             .agg(agg_dict)
)

# recompute total incidents across all crime types
collapsed_fips_only["total_incidents"] = collapsed_fips_only[crime_cols_present].sum(axis=1)



### EXPORT BOTH DATAFRAMES 
today_str = date.today().strftime("%Y-%m-%d")

crime_w_demog = f"{today_str}_crime_demog_final.csv"
cdpath = os.path.join(outf, crime_w_demog)
combined_df.to_csv(cdpath, index=False)


crime_flat= f"{today_str}_crime_fips_level_final.csv"
cpath = os.path.join(outf, crime_flat)
collapsed_fips_only.to_csv(cpath, index=False)










