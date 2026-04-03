#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 11:05:15 2025

@author: allegrasaggese

Crime pipeline:
1) Load cleaned annual crime files (total-v1) and stack into one incident-level table.
2) Select violent-crime columns and collapse to demographic keys:
   fips-year-sex-race-ethnicity (plus county/state labels).
3) Build main county-year panel by collapsing demographics away:
   fips-year totals for selected crime types and total_incidents.
4) Optional extension (off by default): preserve disaggregation by
   sex/race/ethnicity/age bucket and save local extension outputs + QA.
"""

# ----------------------- SET UP PART 1: DEFINE -------------------- -#
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

# write extension artifacts to the explicit Dropbox copy path used for this project round
interim_override = "/Users/allegrasaggese/Dropbox/Mental/allegra-dropbox-copy/interim-data"
qa_dir = os.path.join(interim_override, "qa-crime")
local_ext_dir = os.path.join(interim_override, "crime-disagg-extension")
os.makedirs(qa_dir, exist_ok=True)
os.makedirs(local_ext_dir, exist_ok=True)
today_str = date.today().strftime("%Y-%m-%d")
RUN_CRIME_DISAGG_EXTENSION = os.getenv("RUN_CRIME_DISAGG_EXTENSION", "0").strip().lower() in {"1", "true", "yes", "y"}

crimef = os.path.join(inf, "crime")
crimef = os.path.join(crimef, "total-v1")

csv_files = glob.glob(os.path.join(crimef, "*.csv"))
print(f"Found {len(csv_files)} CSV files")

# read all CSVs into a list of DataFrames
dfs = []
for file in csv_files:
    try:
        df = read_csv_with_fallback(file)
        df["source_file"] = os.path.basename(file)   # optional: keep filename
        dfs.append(df)
        print(f"Loaded: {os.path.basename(file)}  shape={df.shape}")
    except Exception as e:
        print(f"failed to read {os.path.basename(file)}: {e}")

# if you want to combine all of them into one big df
combined_df = pd.concat(dfs, ignore_index=True, sort=False)

# HUGE DF -lets check usage - we want to  preserve memory
combined_df.memory_usage(deep=True).sum() / 1e6  # about 7 GB of memory used 

bytes_used = combined_df.memory_usage(deep=True).sum()
print(f"{bytes_used/1e6:.2f} MB  (~{bytes_used/1024**3:.2f} GiB)")

# big cols
mem_cols = (combined_df.memory_usage(deep=True)
            .sort_values(ascending=False))
print(mem_cols.head(20))

# collapse to year, fips, race, sex 
keys = ["year", "state", "county", "fips", "sex_of_arrestee", 
        "race_of_arrestee", "ethnicity_of_arrestee"]


# ----------------------- DATA PART 1: SELECT KEY CRIME DATA -------------------- -#

# keep only violent crimes - listed / chosen from discussion with reserach team 
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

# create string normalization helper to strip out all extra spaces / marks 
def norm(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("–","-").replace("—","-")           # normalize unicode dashes
    s = re.sub(r"\s+", "_", s)                        # collapse whitespace to _
    s = re.sub(r"[^\w\-_/()]+", "_", s)               # keep word chars, -, _, /, ()
    s = re.sub(r"_+", "_", s).strip("_")
    return s

# build normalized->actual map from existing dataframe
colmap = {norm(c): c for c in combined_df.columns}
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

# find the cols that don't seem to be matching
traffic_cols = [c for c in combined_df.columns if combined_df.columns.str.contains(r"human|traffic", case=False, regex=True).any()]

# create quick summary table to see the number of missing rows
summary = (
    combined_df[traffic_cols]
    .isna()
    .sum()
    .rename("n_missing")
    .to_frame()
)
summary["n_filled"] = combined_df[traffic_cols].notna().sum()
summary["pct_filled"] = 100 * summary["n_filled"] / len(combined_df) # make missing rows into a percentage 
summary

# finding - the human trafficking cols exist, but with lower percentage filled 
for c in combined_df.columns:
    if "human" in c.lower():
        print(repr(c))

# try second round of stripping out issues within the cols 
combined_df.columns = (
    combined_df.columns
    .str.replace(r"[–—]", "-", regex=True)  # normalize dash types
    .str.strip()                            # remove leading/trailing spaces
)

[c for c in combined_df.columns if "human" in c.lower()]


# rename human trafficking columns robustly
rename_map = {}
for c in combined_df.columns:
    nc = norm(c)
    if nc == norm("human_trafficking_-_commercial_sex_acts"):
        rename_map[c] = "human_traffic_comm_sex"
    elif nc == norm("human_trafficking_-_involuntary_servitude"):
        rename_map[c] = "human_traffic_inv_servitude"
if rename_map:
    combined_df = combined_df.rename(columns=rename_map)

# check colnames 
print(repr(combined_df.columns.tolist())) # STILL NOT RECGONIZING IT! 

for i, c in enumerate(combined_df.columns):
    if "human" in c.lower():
        print(i, repr(c))

if "human_traffic_comm_sex" not in combined_df.columns or "human_traffic_inv_servitude" not in combined_df.columns:
    print("Warning: one or both human trafficking columns were not found after normalization.")


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

# ----------------------- DATA PART 2: CREATE CRIME STAT DATA FOR USE  -------------------- -#

# collapse to unit of observation 
collapsed = (
    combined_df.groupby(keys, as_index=False)[crime_cols].sum(min_count=1)
)

# total incidents across all crime types (per group)
collapsed["total_incidents"] = collapsed[crime_cols].sum(axis=1)

# count raw incident rows per group for sanity
collapsed["n_rows"] = combined_df.groupby(keys).size().reset_index(name="n")["n"]

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

# ----------------------- DATA PART 3: COLLAPSE + EXPORT -------------------- -#
# COLLAPSE again to ignore demographic data - NOTE: in future iterations we will want to create an extens
# keys for the final panel (although state-county are redudant)
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
# ----- ADD IN HERE HOW WE CAN TEST TO SEE IF THE CRIME TYPES CHANGE AT ALL 


# output creation
crime_w_demog = f"{today_str}_crime_demog_final.csv"
cdpath = os.path.join(outf, crime_w_demog)
combined_df.to_csv(cdpath, index=False)


crime_flat= f"{today_str}_crime_fips_level_final.csv"
cpath = os.path.join(outf, crime_flat)
collapsed_fips_only.to_csv(cpath, index=False)
print("Saved:", cdpath)
print("Saved:", cpath)


# ----------------------- OPTIONAL EXTENSION: CRIME DISAGG BY SEX/AGE/RACE -------------------- -#
if RUN_CRIME_DISAGG_EXTENSION:
    required_demo = ["sex_of_arrestee", "race_of_arrestee", "ethnicity_of_arrestee", "age_of_arrestee"]
    missing_demo = [c for c in required_demo if c not in combined_df.columns]
    if missing_demo:
        print(f"Skipping crime disaggregation extension; missing columns: {missing_demo}")
    else:
        ext = combined_df.copy()
        ext["year"] = pd.to_numeric(ext["year"], errors="coerce").astype("Int64")
        ext["fips"] = (
            ext["fips"].astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"\D", "", regex=True)
            .str.zfill(5)
        )
        ext["state"] = ext["state"].astype("string").str.strip()
        ext["county"] = ext["county"].astype("string").str.strip()
        ext["sex_of_arrestee"] = ext["sex_of_arrestee"].astype("string").str.strip().str.lower()
        ext["race_of_arrestee"] = ext["race_of_arrestee"].astype("string").str.strip().str.lower()
        ext["ethnicity_of_arrestee"] = ext["ethnicity_of_arrestee"].astype("string").str.strip().str.lower()
        # Source-native age disaggregation: keep the crime data's own age values.
        ext["age_bucket"] = (
            ext["age_of_arrestee"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        )
        ext["age_bucket"] = ext["age_bucket"].replace("", pd.NA).fillna("unknown")
        ext["age_exact"] = pd.to_numeric(ext["age_bucket"], errors="coerce").astype("Int64")

        ext_keys = [
            "year", "state", "county", "fips",
            "sex_of_arrestee", "race_of_arrestee", "ethnicity_of_arrestee",
            "age_bucket"
        ]
        ext_crime_cols = [c for c in crime_cols if c in ext.columns]
        disagg = (
            ext.groupby(ext_keys, as_index=False)[ext_crime_cols]
            .sum(min_count=1)
        )
        disagg["total_incidents"] = disagg[ext_crime_cols].sum(axis=1)
        disagg["n_rows"] = ext.groupby(ext_keys).size().reset_index(name="n")["n"]

        disagg_out = os.path.join(local_ext_dir, f"{today_str}_crime_sex_age_race_disagg_extension.csv")
        disagg.to_csv(disagg_out, index=False)
        print("Saved local extension:", disagg_out)

        panel_keys = ["year", "state", "county", "fips", "sex_of_arrestee", "race_of_arrestee", "ethnicity_of_arrestee", "age_bucket"]
        panel = (
            disagg.groupby(panel_keys, as_index=False)[ext_crime_cols + ["total_incidents", "n_rows"]]
            .sum(min_count=1)
        )
        panel_out = os.path.join(local_ext_dir, f"{today_str}_crime_fips_year_sex_race_agebucket_extension.csv")
        panel.to_csv(panel_out, index=False)
        print("Saved local extension:", panel_out)

        qa_bucket = pd.concat(
            [
                pd.DataFrame({"dimension": "sex_of_arrestee", "value": sorted(panel["sex_of_arrestee"].dropna().unique())}),
                pd.DataFrame({"dimension": "race_of_arrestee", "value": sorted(panel["race_of_arrestee"].dropna().unique())}),
                pd.DataFrame({"dimension": "ethnicity_of_arrestee", "value": sorted(panel["ethnicity_of_arrestee"].dropna().unique())}),
                pd.DataFrame({"dimension": "age_bucket", "value": sorted(panel["age_bucket"].dropna().unique())}),
            ],
            ignore_index=True,
        )
        qa_bucket_out = os.path.join(qa_dir, f"{today_str}_qa_crime_disagg_bucket_values.csv")
        qa_bucket.to_csv(qa_bucket_out, index=False)
        print("Saved QA:", qa_bucket_out)

        qa_keys = pd.DataFrame(
            [
                {
                    "n_rows_disagg": int(len(disagg)),
                    "n_unique_disagg_keys": int(disagg[ext_keys].drop_duplicates().shape[0]),
                    "n_duplicate_disagg_rows": int(disagg.duplicated(ext_keys).sum()),
                    "n_rows_bucket_panel": int(len(panel)),
                    "n_unique_panel_keys": int(panel[panel_keys].drop_duplicates().shape[0]),
                    "n_duplicate_panel_rows": int(panel.duplicated(panel_keys).sum()),
                }
            ]
        )
        qa_keys_out = os.path.join(qa_dir, f"{today_str}_qa_crime_disagg_key_check.csv")
        qa_keys.to_csv(qa_keys_out, index=False)
        print("Saved QA:", qa_keys_out)
else:
    print("Skipping optional crime disaggregation extension (RUN_CRIME_DISAGG_EXTENSION=0).")






