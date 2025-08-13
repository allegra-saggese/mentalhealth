#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:13:23 2025

@author: allegrasaggese
"""
# purpose: ALL MENTAL HEALTH AND CDC DATA UPLOAD! 

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


# upload health files 
## UPLOAD ALL CENSUS DATA - ALL YEARS, RAW 
raw_mh = os.path.join(inf, "mental")
mh_files = glob.glob(os.path.join(raw_mh, "*.csv"))

dfs = [pd.read_csv(file) for file in mh_files]

# check the columns to see if we're missing any of them across years (i.e. changes)
all_cols = set()
for df in dfs:
    all_cols.update(df.columns)

# count how many dfs contain each col
counts = []
for col in sorted(all_cols):
    count = sum(col in df.columns for df in dfs)
    counts.append({"column": col, "dfs_with_col": count})

# make into summary 
summary_df = pd.DataFrame(counts).sort_values(
    by=["dfs_with_col", "column"], ascending=[False, True]
).reset_index(drop=True)

print(summary_df)
out_path = os.path.join(interim, "summary_df.csv")
summary_df.to_csv(out_path, index=False) # save for manual review 

# take the columns that are present in 12+ datasets (create threshold for easy change)
threshold = 12

def get_common_cols(summary_df, threshold):
    """Return list of column names present in >= min_count DataFrames."""
    return summary_df.loc[summary_df["dfs_with_col"] >= threshold, "column"].tolist()

majority_present_cols = get_common_cols(summary_df, 12)

# now take a look at the rest of columns  for similarities 
def find_similar_columns(col_list, threshold=0.7):
    """Find column name pairs with similarity >= threshold."""
    matches = []
    checked = set()

    for i, col1 in enumerate(col_list):
        for j, col2 in enumerate(col_list):
            if i >= j:  # avoid duplicate & self matches
                continue
            pair = tuple(sorted([col1, col2]))
            if pair in checked:
                continue
            ratio = difflib.SequenceMatcher(None, col1, col2).ratio()
            if ratio >= threshold:
                matches.append((col1, col2, round(ratio, 2)))
                checked.add(pair)

    return pd.DataFrame(matches, columns=["col1", "col2", "similarity"])

# run to identify near similar cases 
col_names = summary_df["column"].tolist()
similar_cols_df = find_similar_columns(col_names, threshold=0.7)

print(similar_cols_df)
# RESULT: there's some small punctuation and capitalization issue - need to delete 

def clean_colnames(df):
    df.columns = (
        df.columns
        .str.lower()                      # lowercase
        .str.replace("&", "and")          # replace & with and
        .str.replace(r"[.,]", "", regex=True)  # remove . and ,
        .str.replace(r"\.\d+$", "", regex=True)  # remove .1, .2, etc. at end
        .str.strip()                      # trim spaces
    )
    return df

# Apply to all DataFrames in list
dfs = [clean_colnames(df) for df in dfs]


# manual changes still needed for small varname differences (after all lower): 
    # Limited Access to Healthy Foods <-- Access to Healthy Foods 
    # % female <--- % females 
    # % native hawaiian/other pacific islander <-- % native hawaiian or other pacific islander
    # % american indian and alaska native <--- % american indian or alaska native

# deeper run on a similar cleaning function 
def deep_clean_colnames(df):
    s = df.columns.to_series()

    # base normalization
    s = s.str.replace(r"\s+", " ", regex=True)           # collapse spaces

    # manual fixes (before underscores)
    s = s.str.replace(r"\blimited access to healthy foods\b",
                      "access to healthy foods", regex=True)
    s = s.str.replace(r"\bfemales\b", "female", regex=True)
    s = s.str.replace(r"% native hawaiian\s+or\s+other pacific islander\b",
                      "% native hawaiian/other pacific islander", regex=True)
    s = s.str.replace(r"% american indian\s+or\s+alaska native\b",
                      "% american indian and alaska native", regex=True)

    # finalize
    s = (
        s.str.replace(r"\s+", "_", regex=True)           # spaces -> underscores
         .str.replace(r"_+", "_", regex=True)            # de-dup underscores
         .str.strip("_")
    )

    df.columns = s
    return df

# apply to all dataframes in your list
dfs = [deep_clean_colnames(df) for df in dfs]
    
# check the count again of how many column names are present, manually 
all_cols3 = set()
for df in dfs:
    all_cols3.update(df.columns)
    
counts = []
for col in sorted(all_cols3):
    count = sum(col in df.columns for df in dfs)
    counts.append({"column": col, "dfs_with_col": count})

# make into summary 
summary_df = pd.DataFrame(counts).sort_values(
    by=["dfs_with_col", "column"], ascending=[False, True]
).reset_index(drop=True)

print(summary_df)
out_path = os.path.join(interim, "mental_health_cols_v3.csv")
summary_df.to_csv(out_path, index=False) # save for manual review 
    
# select the cols that are present in 11 (of 14) dataframes -- about 78% 
majority_present_cols = get_common_cols(summary_df, 11)

# so many unmatched cols b/c white, black, hispanic demo disagg for 2018, 2019
# starting in 2020 they add AIAN, Asian Pacific for some of the health issues

# QA -- check the data completeness and duplicates 

def drop_dupes_keep_most_complete(df):
    """drop any dupes - keep highest % of nonmissng values
       prints duplicate info if found."""
    cols_to_keep = {}
    
    for col in df.columns.unique():  # loop over unique col names
        dupes = df.loc[:, df.columns == col]
        
        if dupes.shape[1] > 1:  # duplicate columns found
            comp = dupes.notna().mean() * 100
            best_idx = comp.idxmax()
            print(f"Duplicate col '{col}': keeping '{best_idx}', completeness = {comp.max():.2f}%")
            for other in dupes.columns:
                if other != best_idx:
                    print(f"    Dropped duplicate: '{other}', completeness = {comp[other]:.2f}%")
        else:
            best_idx = dupes.columns[0]
        
        cols_to_keep[col] = best_idx

    return df.loc[:, cols_to_keep.values()]

dfs_clean = []

for i, df in enumerate(dfs, start=1):
    print(f"\nProcessing DataFrame {i}/{len(dfs)}:")
    cleaned_df = drop_dupes_keep_most_complete(df)
    dfs_clean.append(cleaned_df)
# only dupe was with renaming access to healthy food columns - dropped where no data was present 
    
# check completeness 
summaries = []
for i, df in enumerate(dfs_clean, 1):
    summary = percent_missing_vs_filled(df)
    summary = summary.reset_index()  # index -> column
    summary = summary.rename(columns={"index": "column"})
    summary.insert(0, "df_id", f"df_{i}")
    summaries.append(summary)

all_summaries = pd.concat(summaries, ignore_index=True)

#group and average
avg_completeness = (
    all_summaries
    .groupby("column", as_index=False)["filled_pct"]
    .mean()
    .rename(columns={"filled_pct": "avg_filled_pct"})
)

# average for threshold (i.e. present in at least 11 dfs)
avg_completeness_thresh = (
    all_summaries
    .groupby("column")
    .agg(
        avg_filled_pct=("filled_pct", "mean"),
        dfs_with_col=("df_id", "nunique")
    )
    .query("dfs_with_col >= 8") # THRESHOLD VALUE - less years (8 < 11) but higher standard for completeness 
     .reset_index()
)

# now take those with above 80% fill rate with presence in 8 / 14 years
low_missing_cols = (
    avg_completeness_thresh
    .loc[avg_completeness_thresh["avg_filled_pct"] >= 80, "column"]
    .tolist()
)

len(low_missing_cols) # at least 190 cols meet this 

append_cols(majority_present_cols, low_missing_cols) # append to existing list

dfs_shortlist = keep_only_cols(dfs_clean, majority_present_cols) # create short list 









