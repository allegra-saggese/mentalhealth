#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:03:00 2025

@author: allegrasaggese
"""

# load packages and workspaces
from packages import * 
from collections import Counter
import re


# load alt functions
from functions import * 

# other folders
inf = os.path.join(db_data, "raw") # input 
outf = os.path.join(db_data, "clean") #outpit

# load data - CENSUS ALL YEARS RAW 
rawpop = os.path.join(inf, "population")
file_list = glob.glob(os.path.join(rawpop, "*.csv")) + \
            glob.glob(os.path.join(rawpop, "*.txt"))

dfs = []

for file in file_list:
    try:
        df = pd.read_csv(file, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            # Retry with fallback encoding
            df = pd.read_csv(file, encoding="latin1")
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue
    dfs.append(df)
    

# view cols 
col_lists = [df.columns.tolist() for df in dfs]

# issue - only one col in the 1990-2000 data, need to change 
dfs[3].head(50) 
raw_col = dfs[3].iloc[:, 0]
raw_header = dfs[3].columns[0]  # example: "FIPS  STATE_NAME   TOTAL_ HEAD COUNT  AVG_WEIGHT"
new_colnames = raw_header.strip().split()
new_colnames = [name.replace(" ", "") for name in new_colnames]

# split each row on any whitespace (handles multiple spaces)
split_data = raw_col.str.split(expand=True)

# replace the df in place
dfs[3] = split_data
dfs[3].columns = new_colnames
print(dfs[3].head())
 

# get set of colnames 
colsets = [set(df.columns) for df in dfs[:4]]
common_cols = set.intersection(*colsets)
print("Columns common to all 4:", sorted(common_cols)) # none across all four 

all_cols = set.union(*colsets)
print("Total unique columns across all:", len(all_cols)) # 292 total unique 


# count of calls 
all_columns = [col for colset in colsets for col in colset]
col_counts = Counter(all_columns)

# convert to DataFrame and sort
col_presence = pd.DataFrame.from_dict(col_counts, orient="index", columns=["count"])
col_presence = col_presence.reset_index().rename(columns={"index": "column"})
col_presence = col_presence.sort_values(by="count", ascending=False)

print(col_presence) # 2000-2020 data is in wide format, need to convert to long 

# 1990s data - combine to make one census estimate
df1990s = dfs[3]
df1990s = df1990s.apply(pd.to_numeric, errors="coerce")
cols_to_sum = [col for col in df1990s.columns if col.startswith("NH_") or col.startswith("H_")]
df1990s["pop"] = df1990s[cols_to_sum].sum(axis=1)


# 2000 data - wide to long with re
df2000 = dfs[1]

year_cols = [col for col in df2000.columns if re.search(r"20\d{2}$", col)]
manual_cols = ["CENSUS2010POP"] # census pop not meeting the re.search method, manual add 
year_cols = list(set(year_cols + manual_cols)) # combine and update

df2000_long = pd.melt(
    df2000,
    id_vars=[col for col in df2000.columns if col not in year_cols],
    value_vars=year_cols,
    var_name="raw_year_col",
    value_name="value"
)

df2000_long["year"] = df2000_long["raw_year_col"].str.extract(r"(20\d{2})").astype(int)

# drop estimate for 2000 and 2010 (where we have base est, census data)
df2000_long = df2000_long[~df2000_long["raw_year_col"].isin(["POPESTIMATE2000", "POPESTIMATE2010"])]

# 2010 data
