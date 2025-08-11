#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:03:00 2025

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
outf = os.path.join(db_data, "clean") #outpit

## UPLOAD ALL CENSUS DATA - ALL YEARS, RAW 
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
    
## UPLOAD ALL FIPS DATA - FOR MATCHING LATER
rawfips = os.path.join(inf, "fips")
fipsfiles = glob.glob(os.path.join(rawfips, "foruse_*.txt"))

def sniff_delim(path):
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.strip():
                s = line
                break
    # count signals for each delimiter
    scores = {
        "pipe": s.count("|"),
        "comma": s.count(","),
        "ws": len(s.split()) - 1  # whitespace tokens
    }
    return max(scores, key=scores.get)

# identify which type of separator exists for each file 
pipe_files, comma_files, ws_files = [], [], []
for p in fipsfiles:
    d = sniff_delim(p)
    if d == "pipe":
        pipe_files.append(p)
    elif d == "comma":
        comma_files.append(p)
    else:
        ws_files.append(p)

# load each bucket (three separate lines/loops)

# whitespace (or fixed-width) files (2000)
ws_dfs = []
for p in ws_files:
    try:
        df = pd.read_fwf(p)  # often best for space-aligned reports
        if df.shape[1] <= 1:  # fallback if not fixed-width
            df = pd.read_table(p, delim_whitespace=True, engine="python")
    except Exception:
        df = pd.read_table(p, delim_whitespace=True, engine="python")
    df["source_file"] = os.path.basename(p)
    ws_dfs.append(df)

# need to break out one col with FIPS / county into two cols 
fips_00 = ws_dfs[0]
fips_00.columns.tolist()
fips_00[["fips", "county"]] = fips_00["FIPS\t\t countyname"].str.strip().str.split(r"\s+", n=1, expand=True)

# drop uneccessary columns + split fips cols for more ID 
fips_00 = fips_00.drop(columns=["FIPS\t\t countyname", "Unnamed: 1", "source_file"])    
fips_00["state_code"] = fips_00["fips"].astype(str).str[:2]
fips_00["county_code"] = fips_00["fips"].astype(str).str[-3:]

# create year col + dupe data for each year where its applicable 
years = list(range(2000, 2010))
year_df = pd.DataFrame({"year": years})
fips_00_expanded = fips_00.merge(year_df, how="cross")
fips_00_expanded = fips_00_expanded.reset_index(drop=True)

# manual creation of the 1990 codes given the states similarities 
 fips_90 = fips_00
# manual changes, REASON: see source data for differences from 1990 to 2000 census
 fips_90["fips"] = fips_90["fips"].replace(12025, 12086)
 fips_90["county_code"] = fips_90["county_code"].replace(025, 086)
 fips_90["county"] = fips_90["county"].replace("Miami-Dade County", "Dade County")
# add in counties that were merged/lost for the 2000 census 
new_row1 = {
    "FIPS": 30113,
    "state_code": 30,
    "county_code": 113,
    "county": "Yellowstone National Park County"
}

new_row2 = {
    "FIPS": 51780,
    "state_code": 51,
    "county_code": 780,
    "county": "South Boston"
}

fips_90 = pd.concat([fips_90, pd.DataFrame([new_row1, new_row2])], ignore_index=True)
# add the years
yrs0 = list(range(1990, 2000))
yrs0df = pd.DataFrame({"year": yrs0})
fips_90_expanded = fips_90.merge(yrs0df, how="cross")
fips_90_expanded = fips_90_expanded.reset_index(drop=True)


# comma file (2010) - doing it manual for ease    
path = "/Users/allegrasaggese/Dropbox/Mental/Data/raw/fips/foruse_FIPScodes2010_w_names.txt"
fips_10 = pd.read_table(path, sep=",", engine="python", encoding="latin1", on_bad_lines="warn")
# create full FIPS
generate_fips(fips_10, state_col="STATEFP", city_col="COUNTYFP")
# add years
yrs2 = list(range(2010, 2020))
yrs2df = pd.DataFrame({"year": yrs2})
fips_10_expanded = fips_10.merge(yrs2df, how="cross")
fips_10_expanded = fips_10_expanded.reset_index(drop=True)
 

# pipe-delimited files (2020)
pipe_dfs = []
for p in pipe_files:
    try:
        df = pd.read_csv(p, sep="|", engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(p, sep="|", engine="python", encoding="latin1")
    df["source_file"] = os.path.basename(p)
    pipe_dfs.append(df)
    
fips_20 = pipe_dfs[0]
generate_fips(fips_20, state_col="STATEFP", city_col="COUNTYFP")
fips_20 = fips_20.drop(columns=["source_file"])   
# add years
yrs3 = list(range(2020, 2025))
yrs3df = pd.DataFrame({"year": yrs3})
fips_20_expanded = fips_20.merge(yrs3df, how="cross")
fips_20_expanded = fips_20_expanded.reset_index(drop=True)

# combine all FIPS data 
fips_annual_full = pd.concat([fips_90_expanded, fips_00_expanded, 
                              fips_10_expanded, fips_20_expanded],
                             axis=0, join="outer", ignore_index=True)


## INVESTIGATE COLS across dataframes for patterns 
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
 

# get all cols across all dfs
colsets = [set(df.columns) for df in dfs[:4]]
common_cols = set.intersection(*colsets)
print("Columns common to all 4:", sorted(common_cols)) # none across all four 

all_cols = set.union(*colsets)
print("Total unique columns across all:", len(all_cols)) # 292 total unique 

# count of columns 
all_columns = [col for colset in colsets for col in colset]
col_counts = Counter(all_columns)

# convert to df, sort 
col_presence = pd.DataFrame.from_dict(col_counts, orient="index", columns=["count"])
col_presence = col_presence.reset_index().rename(columns={"index": "column"})
col_presence = col_presence.sort_values(by="count", ascending=False)

print(col_presence)
# 2000-2020 data is in wide format, need to convert to long 



## MAKE EACH DATA FRAME IN SAME FORMAT FOR THE MERGE

##1990-2000 DATA:
df1990s = dfs[3]   
# need to combine to make one aggregate census estimate (as opposed to disagg)
df1990s = df1990s.apply(pd.to_numeric, errors="coerce")
cols_to_sum = [col for col in df1990s.columns if col.startswith("NH_") or col.startswith("H_")]
df1990s["pop"] = df1990s[cols_to_sum].sum(axis=1)

# drop demographic data
demo_cols = [col for col in df1990s.columns if col.startswith("NH_") or col.startswith("H_")]
df1990s.drop(columns=demo_cols, inplace=True)

# FIPS / YR / POPULATION -- need to merge with county/state names (from  FIPS data)



##2000-2010 DATA:
df2000 = dfs[1]

# make wide to long
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

# rename pop column, drop unneccessary cols, create FIPS code
df2000_long = df2000_long.rename(columns={
    "value": "population",
})
df2000_long = df2000_long.drop(columns=["raw_year_col", "SUMLEV"])
generate_fips(df2000_long, state_col="STATE", city_col="COUNTY")

df2000s = df2000_long


##2010-2020 DATA:
df2010 = dfs[2] 

id_cols = ["STATE", "COUNTY", "DIVISION", "REGION", "STNAME", "CTYNAME"]  # add any others you need
cols_to_keep = id_cols + [col for col in df2010.columns if col.startswith("POPESTIMATE")]
df2010 = df2010[cols_to_keep]

years_c = [col for col in df2010.columns if re.search(r"20\d{2}$", col)]

df2010_long = pd.melt(
    df2010,
    id_vars=[col for col in df2010.columns if col not in years_c],
    value_vars=years_c,
    var_name="raw_year_col",
    value_name="value"
)

df2010_long["year"] = df2010_long["raw_year_col"].str.extract(r"(20\d{2})").astype(int)






