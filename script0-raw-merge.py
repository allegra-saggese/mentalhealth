#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:03:00 2025

@author: allegrasaggese
"""

# load packages and workspaces
from packages import * 
from collections import Counter


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



