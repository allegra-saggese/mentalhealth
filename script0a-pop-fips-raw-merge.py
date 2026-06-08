#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:03:00 2025

@author: allegrasaggese
"""

# ----------------------- SET UP : set repos / outputs  -------------------- -#


# load packages and workspaces
from collections import Counter
import sys
import json
import time
import urllib.error
from urllib.parse import urlencode
from urllib.request import urlopen, Request

# make sure repo root is on sys.path (parent of functions.py / packages/)
import os
repo = os.path.dirname(os.path.abspath(__file__))
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



# ----------------------- DATA PART 1 : LOAD FIPS DATA  -------------------- -#

## UPLOAD ALL CENSUS DATA - ALL YEARS, RAW 
rawpop = os.path.join(inf, "population")
file_list = sorted(
    glob.glob(os.path.join(rawpop, "*.csv")) +
    glob.glob(os.path.join(rawpop, "*.txt"))
)

dfs = []
pop_sources = {}

for file in file_list:
    try:
        df = read_csv_with_fallback(file)
    except Exception as e:
        print(f"Failed to read {file}: {e}")
        continue
    dfs.append(df)
    pop_sources[os.path.basename(file).lower()] = df


# apply 
df_pop_2024 = pick_source_by_fragment(pop_sources, "cc-est2024-agesex-all")
df_pop_2000s = pick_source_by_fragment(pop_sources, "cc-est00int-tot")
df_pop_2010s = pick_source_by_fragment(pop_sources, "cc-est2010-2020")
df_pop_1990s = pick_source_by_fragment(pop_sources, "cc-est-1990-2000")
    
## UPLOAD ALL FIPS DATA 
rawfips = os.path.join(inf, "fips")
fipsfiles = glob.glob(os.path.join(rawfips, "foruse_*.txt"))

# identify which type of separator exists for each file 
pipe_files, comma_files, ws_files = [], [], []
for p in fipsfiles:
    d = sniff_delimiter(p)
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

# manually check each panel within the fips compendium
fips_00 = ws_dfs[0]
fips_00.columns.tolist()
fips_00[["fips", "county"]] = fips_00["FIPS\t\t countyname"].str.strip().str.split(r"\s+", n=1, expand=True)



# ----------------------- DATA PART 2 : CLEAN UP FIPS DATAFRAMES  -------------------- -#

fips_00 = fips_00.drop(columns=["FIPS\t\t countyname", "Unnamed: 1", "source_file"])    
fips_00["state_code"] = fips_00["fips"].astype(str).str[:2]
fips_00["county_code"] = fips_00["fips"].astype(str).str[-3:]

# create year col + dupe data for each year where its applicable 
fips_00_expanded = expand_years_cross(fips_00, range(2000, 2010))

# manual creation of the 1990 codes given the states similarities 
fips_90 = fips_00


# manual changes, REASON: see source data for differences from 1990 to 2000 census
fips_90["fips"] = fips_90["fips"].replace(12025, 12086)
fips_90["county_code"] = fips_90["county_code"].replace(25, 86)
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
fips_90_expanded = expand_years_cross(fips_90, range(1990, 2000))


# comma file (2010)
if not comma_files:
    raise FileNotFoundError("No comma-delimited 2010 FIPS file found in raw/fips")
comma_2010 = [p for p in comma_files if "2010" in os.path.basename(p).lower()]
comma_target = comma_2010[0] if comma_2010 else comma_files[0]
fips_10 = pd.read_table(comma_target, sep=",", engine="python", encoding="latin1", on_bad_lines="warn")


# GET ALL FIPS <> CTY <> YEAR CODES FROM RAW DATA 
generate_fips(fips_10, state_col="STATEFP", city_col="COUNTYFP")

# collapse on subfips because I want to eliminate this categorization
fips_10 = fips_10.drop(columns=['COUSUBFP', 'COUSUBNAME', 'FUNCSTAT'], errors='ignore')
fips_10 = fips_10.drop_duplicates(subset='FIPS_generated', keep='first')


# add years into the dataframe with empty columns
fips_10_expanded = expand_years_cross(fips_10, range(2010, 2020))
 

# pipe-delimited files (2020)
pipe_dfs = []
for p in pipe_files:
    try:
        df = read_csv_with_fallback(p, sep="|", engine="python")
    except Exception as e:
        print(f"Failed to read {p}: {e}")
        continue
    df["source_file"] = os.path.basename(p)
    pipe_dfs.append(df)
    
fips_20 = pipe_dfs[0]
generate_fips(fips_20, state_col="STATEFP", city_col="COUNTYFP")
fips_20 = fips_20.drop(columns=["source_file"]) 

# similar to fips_10 drop subfips! 
fips_20 = fips_20.drop(columns=['COUSUBFP', 'COUSUBNAME', 'FUNCSTAT'], errors='ignore')
fips_20 = fips_20.drop_duplicates(subset='FIPS_generated', keep='first')

  
# add years into the dataframe with empty columns
fips_20_expanded = expand_years_cross(fips_20, range(2020, 2025))

# change county, state code colnames to ensure merge works 
fips_90_expanded.columns.tolist()
fips_90_expanded.drop(columns=["FIPS"], inplace=True) # empty col to drop
fips_00_expanded.columns.tolist()
fips_10_expanded.columns.tolist()
fips_20_expanded.columns.tolist()

# standard cleaning - make lower case
fips_10_expanded.columns = fips_10_expanded.columns.str.lower()
fips_20_expanded.columns = fips_20_expanded.columns.str.lower()

# rename columns with underscore - for standardization in the future 
fips_10_expanded = fips_10_expanded.rename(columns={
    "statefp": "state_code",
    "countyfp": "county_code",
    "countyname": "county",
    "fips_generated": "fips"
})

fips_20_expanded = fips_20_expanded.rename(columns={
    "statefp": "state_code",
    "countyfp": "county_code",
    "countyname": "county",
    "fips_generated": "fips"
})

# BEFORE MERGE - make sure the same count of cols + col types 
fips_10_expanded = fips_10_expanded.drop(columns=['state'], errors='ignore')
fips_20_expanded = fips_20_expanded.drop(columns=['state', 'cousubns', 'classfp'], errors='ignore')

# standardize col types
for df in [fips_90_expanded, fips_00_expanded, fips_10_expanded, fips_20_expanded]:
    df['fips'] = pd.to_numeric(df['fips'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['state_code'] = df['state_code'].astype(str) # no pad
    df['county_code'] = df['county_code'].astype(str) # no pad 
    df['county'] = df['county'].astype(str)


# combine all FIPS YEAR DATA FRAMES INTO ONE PANEL
fips_annual_full = pd.concat([fips_90_expanded, fips_00_expanded, 
                              fips_10_expanded, fips_20_expanded],
                             axis=0, join="outer", ignore_index=True)




# ----------------------- DATA PART 3 : QUALITY ASSURANCE  -------------------- -#
# QA: find out which year <> county/state combos were missing 


# quick sense check on fill rate 
missing_pct = fips_annual_full.isna().mean() * 100
print(missing_pct)

rows_with_missing = fips_annual_full.isna().any(axis=1).sum()
print(rows_with_missing) # 20 are missing at least one value 
df_missing = fips_annual_full[fips_annual_full.isna().any(axis=1)]


# manually add in the missing fips --- its given by census.gov file 
# https://www2.census.gov/programs-surveys/popest/geographies/1990-2000/90s-fips.txt
mt_fips = 30113
boston_fips = 51780 
# testing on sliced df
df_missing.loc[df_missing['county'].str.contains('Yellowstone', case=False, na=False), 'fips'] = mt_fips
df_missing.loc[df_missing['county'].str.contains('Boston', case=False, na=False), 'fips'] = boston_fips

# do in main 
fips_annual_full.loc[
    fips_annual_full['fips'].isna() & 
    fips_annual_full['county'].str.contains('Yellowstone', case=False, na=False),
    'fips'
] = mt_fips

fips_annual_full.loc[
    fips_annual_full['fips'].isna() & 
    fips_annual_full['county'].str.contains('Boston', case=False, na=False),
    'fips'
] = boston_fips
# recheck missing percentage and its zero! move on 



# remove state-level rows (keeping DC), standardize keys, and normalize DC FIPS
fips_annual_full_v2 = drop_state_level_rows_except_dc(fips_annual_full)
fips_annual_full_v2 = standardize_county_identifiers(fips_annual_full_v2, fips_as_string=False)
fips_annual_full_v2 = normalize_dc_fips(fips_annual_full_v2, fips_value="11001")
fips_annual_full_v2 = filter_to_us_state_codes(fips_annual_full_v2)


# drop historical state-level DC key if present
fips_annual_full_v2 = fips_annual_full_v2[fips_annual_full_v2["fips"] != 11000].copy()

# dedupe and reindex
fips_annual_full_v2 = (
    fips_annual_full_v2
    .sort_values(["year", "fips"])
    .drop_duplicates(["fips", "year"], keep="first")
    .reset_index(drop=True)
)

# stash for population merges below
fips_for_pop = fips_annual_full_v2[["fips", "county", "state_code", "county_code", "year"]].copy()


# export clean data
clean_dir = os.path.join(db_data, "clean")
today_str = date.today().strftime("%Y-%m-%d")
clean_fips_df = f"{today_str}_fips_full.csv"
out_path = os.path.join(clean_dir, clean_fips_df)

# export to csv in clean folder
fips_annual_full_v2.to_csv(out_path, index=False)

# clean up
del fips_00, fips_00_expanded, fips_10, fips_10_expanded, fips_20 
del fips_20_expanded, fips_90, fips_90_expanded, fips_annual_full
del fips_annual_full_v2, missing_pct, mt_fips, new_row1, new_row2, rawfips
del rows_with_missing, boston_fips


# ----------------------- DATA PART 4 : ADD POPULATION IN  -------------------- -#
# Now we add the population data into the fips data for the final key 


## INVESTIGATE COLUMNS ACROSS DISAGGREGATED DFs for patterns 
col_lists = [
    df_pop_2024.columns.tolist(),
    df_pop_2000s.columns.tolist(),
    df_pop_2010s.columns.tolist(),
    df_pop_1990s.columns.tolist(),
]

print(len(df_pop_2024.columns))  # 2024 agesex
print(len(df_pop_2000s.columns)) # 2000-2010
print(len(df_pop_2010s.columns)) # 2010-2020
print(len(df_pop_1990s.columns)) # 1990-2000 text


# issue - only one col in the 1990-2000 data, need to separate by spaces
df_pop_1990s.head(50)
raw_col = df_pop_1990s.iloc[:, 0]
raw_header = df_pop_1990s.columns[0]  # example: "FIPS  STATE_NAME   TOTAL_ HEAD COUNT  AVG_WEIGHT"
new_colnames = raw_header.strip().split()
new_colnames = [name.replace(" ", "") for name in new_colnames]

# split each row on any whitespace (handles multiple spaces)
split_data = raw_col.str.split(expand=True)

# replace the df in place
df_pop_1990s = split_data
df_pop_1990s.columns = new_colnames
print(df_pop_1990s.head())
print(len(df_pop_1990s.columns))  # now 10 cols 


# standardize col names to lower case and remove white spaces
for d in [df_pop_2024, df_pop_2000s, df_pop_2010s, df_pop_1990s]:
    d.columns = d.columns.str.lower().str.replace(r'[\s\-]+', '', regex=True)

# create sets of column names
colsets = [set(df.columns) for df in [df_pop_2024, df_pop_2000s, df_pop_2010s, df_pop_1990s]]
common_cols = set.intersection(*colsets) # print common cols 
print(f"Columns common to all dfs: {len(common_cols)}")
print(common_cols)

# manual review of the columns 
all_cols = set.union(*colsets) # get all unique cols 
print("Total unique columns across all:", len(all_cols)) # 291 total unique 

# count of columns 
all_columns = [col for colset in colsets for col in colset]
col_counts = Counter(all_columns)

# convert to df, sort 
col_presence = pd.DataFrame.from_dict(col_counts, orient="index", columns=["count"])
col_presence = col_presence.reset_index().rename(columns={"index": "column"})
col_presence = col_presence.sort_values(by="count", ascending=False)
print(col_presence) 
# 2000-2020 (dfs[1] = 2010, dfs[2] = 2020, dfs[0] = 2000, data is in wide format, need to convert to long 


# clean up enviro for coding  
del all_cols, all_columns, col_counts, col_lists, col_presence,     
del ws_dfs, ws_files


# ----------------------- DATA PART 5 : REPEAT POP DATA  -------------------- -#

##1990-2000 DATA:
df1990s = df_pop_1990s.copy()
print(df1990s.dtypes) # check col types 
df1990s.columns = df1990s.columns.str.lower() # make all lower 

# need to combine to make one aggregate census estimate (as opposed to disagg by race)
cols_to_sum = [col for col in df1990s.columns if col.startswith("nh_") or col.startswith("h_")]
df1990s[cols_to_sum] = df1990s[cols_to_sum].apply(pd.to_numeric, errors="coerce")
df1990s["pop"] = df1990s[cols_to_sum].sum(axis=1)

df1990s[cols_to_sum].apply(lambda x: x.unique()) # check rows 

# create percentage cols 
for col in cols_to_sum:
    new_col = f"percent_{col.lower()}"
    df1990s[new_col] = df1990s[col] / df1990s["pop"]


# make sep dataframe w/ demographics
demo_cols = [col for col in df1990s.columns if col.startswith("nh_") 
             or col.startswith("h_") 
             or col.startswith("percent")]

df1990s_disaggregated = df1990s.copy() # save a copy of the DF with disagg b/f removing for merge 
df1990s.drop(columns=demo_cols, inplace=True) # drop demo data from main DF 

# pad fips 
df1990s["fips"] = df1990s["fips"].astype(str).str.zfill(5)

# merge with fips county name data
sub_cols = ["fips", "county", "state_code", "county_code", "year"]
fips_sub = fips_for_pop[sub_cols].copy()

fips_sub['fips'] = fips_sub['fips'].astype(int) # set type
df1990s['fips'] = df1990s['fips'].astype(int) # set type as the same for fips
fips_sub['year'] = fips_sub['year'].astype(int)
df1990s['year'] = df1990s['year'].astype(int)

# use fips data to merge in state, county information 
df1990s_full = df1990s.merge(fips_sub, on=["fips", "year"], how="inner")  
df1990s_full = df1990s_full.rename(columns={"pop": "population"})
df1990s_full["fips"] = df1990s_full["fips"].astype(str).str.zfill(5)

print(df1990s_full.shape)


##2000-2010 DATA:
df2000 = df_pop_2000s.copy()

# make wide to long
year_cols = [col for col in df2000.columns if re.search(r"20\d{2}$", col)]
manual_cols = ["census2010pop"] # census pop not meeting the re.search method, manual add 
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
df2000_long = df2000_long[~df2000_long["raw_year_col"].isin(["popestimate2000", "popestimate2010"])]

# rename pop column, drop unneccessary cols, create FIPS code
df2000_long = df2000_long.rename(columns={
    "value": "population",
})
df2000_long = df2000_long.drop(columns=["raw_year_col", "sumlev"])
generate_fips(df2000_long, state_col="state", city_col="county")

df2000s = df2000_long.copy()
df2000s.columns = df2000s.columns.str.lower() # making lower case again 
df2000s["state"] = df2000s["state"].astype(str).str.zfill(2) # pad 
df2000s["county"] = df2000s["county"].astype(str).str.zfill(3) # pad 


df2000s_cleannames = df2000s.rename(columns={
    "stname": "state",
    "ctyname": "county",
    "state": "state_code",
    "county": "county_code",
    "fips_generated": "fips"
})

# drop state-only observations
before = len(df2000s_cleannames)
df2000s_cleannames = df2000s_cleannames[df2000s_cleannames["county"] != df2000s_cleannames["state"]]
dropped = before - len(df2000s_cleannames)
print("Rows dropped:", dropped)

# FINAL OUTPUT for 2000s POP DATA
df2000s_full = df2000s_cleannames


##2010-2020 DATA:
df2010 = df_pop_2010s.copy()

id_cols = ["state", "county", "division", "region", "stname", "ctyname"]
cols_to_keep = id_cols + [col for col in df2010.columns if col.startswith("popestimate")]
df2010 = df2010[cols_to_keep]

years_c = [col for col in df2010.columns if re.search(r"20\d{2}$", col)]

df2010_long = pd.melt(
    df2010,
    id_vars=[col for col in df2010.columns if col not in years_c],
    value_vars=years_c,
    var_name="raw_year_col",
    value_name="value"
)

# standardize colnames
df2010_long["year"] = df2010_long["raw_year_col"].str.extract(r"(20\d{2})").astype(int)
generate_fips(df2010_long, state_col="state", city_col="county")
df2010_long.columns = df2010_long.columns.str.lower()


df2010_full = df2010_long.rename(columns={
    "stname": "state",
    "ctyname": "county",
    "state": "state_code",
    "county": "county_code",
    "fips_generated": "fips",
    "value": "population"
})


df2010_full = df2010_full.drop(columns=["raw_year_col"])

# drop 2010 data as previous census has this data 
df2010_full = df2010_full[df2010_full["year"] != 2010]

# drop state only observations 
before2 = len(df2010_full)
df2010_full = df2010_full[df2010_full["county"] != df2010_full["state"]]
dropped2 = before2 - len(df2010_full)
print("Rows dropped:", dropped2)

# FINAL OUTPUT for 2010s DATA 
df2010_full_v2 = df2010_full.copy()

# 2021-2024 DATA from cc-est2024-agesex-all
df2024 = df_pop_2024.copy()
df2024 = df2024[df2024["sumlev"] == 50].copy()
df2024["year"] = pd.to_numeric(df2024["year"], errors="coerce").map(YEAR_CODE_2024_MAP)
df2024 = df2024.dropna(subset=["year"]).copy()
df2024["year"] = df2024["year"].astype(int)
generate_fips(df2024, state_col="state", city_col="county")


# standardize var names for hte merge 
df2024_full = df2024.rename(columns={
    "stname": "state",
    "ctyname": "county",
    "state": "state_code",
    "county": "county_code",
    "FIPS_generated": "fips",
    "fips_generated": "fips",
    "popestimate": "population"
})

# fill strings 
df2024_full["state_code"] = df2024_full["state_code"].astype(str).str.zfill(2)
df2024_full["county_code"] = df2024_full["county_code"].astype(str).str.zfill(3)
df2024_full = df2024_full[df2024_full["state_code"].isin(US_STATE_CODES)].copy()
df2024_full = df2024_full[df2024_full["fips"].astype(str).str.len() == 5].copy()

# keep only columns shared with annual panel
keep_2024 = [c for c in df2010_full_v2.columns if c in df2024_full.columns]
df2024_full = df2024_full[keep_2024].copy()



# ----------------------- DATA PART 5: MERGE FIPS / POPS  -------------------- -#


# sort columns (alphabetical) for easier comparison
df1990s_full = df1990s_full[sorted(df1990s_full.columns)]
df2000s_full = df2000s_full[sorted(df2000s_full.columns)]
df2010_full_v2 = df2010_full_v2[sorted(df2010_full_v2 .columns)]
df2024_full = df2024_full[sorted(df2024_full.columns)]


# for 1990s data - get state code 
state_key = df2010_full[["state_code", "state"]].drop_duplicates(subset="state_code")
state_key["state_code"] = state_key["state_code"].astype(str).str.zfill(2) # pad 
new_row = {"state_code": "11", "state": "District of Columbia"} # manually add DC 
state_key = pd.concat([state_key, pd.DataFrame([new_row])], ignore_index=True)

df1990s_full["state_code"] = df1990s_full["state_code"].astype(str) # make both string
state_key["state_code"] = state_key["state_code"].astype(str) # make both string
# merge
df1990s_full = df1990s_full.merge(state_key, on="state_code", how="left")


# check all columns for the rowbind 
df1990s_full.columns.tolist()
df2000s_full.columns.tolist()
df2010_full_v2.columns.tolist()


# concanate all population data 
full_pop_df = pd.concat([df1990s_full, df2000s_full, df2010_full_v2, df2024_full], ignore_index=True, sort=False)

full_pop_df = standardize_county_identifiers(full_pop_df, fips_as_string=True)
full_pop_df["population"] = pd.to_numeric(full_pop_df["population"], errors="coerce")

# normalize DC and keep US-only scope
full_pop_df = normalize_dc_fips(full_pop_df, fips_value="11001")
full_pop_df = filter_to_us_state_codes(full_pop_df)
full_pop_df = full_pop_df[full_pop_df["fips"] != "11000"].copy()
full_pop_df = full_pop_df.dropna(subset=["fips", "year"]).copy()
full_pop_df = full_pop_df.sort_values(["fips", "year"]).drop_duplicates(["fips", "year"], keep="last")

# quick QA to see if we have any loss at the year - level 
numeric_cols = full_pop_df.select_dtypes(include=[np.number])
ranges = numeric_cols.agg(['min', 'max']).T

# unique counts
unique_counts = {
    "state_code_unique": full_pop_df["state_code"].nunique(),
    "county_code_unique": full_pop_df["county_code"].nunique()
}

# missing stats, i.e. did the merge break down 
missing_pct = full_pop_df.isna().mean() * 100
rows_with_3plus_missing = (full_pop_df.isna().sum(axis=1) >= 3).sum()

print("Numeric column ranges:\n", ranges)
print("\nUnique counts:\n", unique_counts)
print("\n% missing per column:\n", missing_pct)
print(f"\nRows with ≥3 missing: {rows_with_3plus_missing}")

# only rows with missing data are those without info on region and division
# but strangely - there are 101 state codes --- need to investigate this 
print(full_pop_df["state_code"].unique()[:50])  # sample of unique entries - beginning 
print(full_pop_df["state_code"].unique()[50:])  # sample of unique entries - end 
print("\nData types:", full_pop_df["state_code"].dtype)
# re-ran QA and now we should have 51 unique codes (50 states + DC)




# Final step: EXPORT TO CSV IN THE CLEAN FOLDER 
clean_pop_df = f"{today_str}_population_full.csv"
poppath = os.path.join(clean_dir, clean_pop_df)
full_pop_df.to_csv(poppath, index=False)


# =============================================================================
# SECTION B: USDA NASS CAFO DATA (formerly script0b-ag-raw.py)
# Outputs: *_ag_annual_df.csv, *_cafo_ops_by_size_{row,long,compact}.csv
# =============================================================================
# ----------------------- SET UP PART 2 : PULL USDA FROM API  -------------------- -#
# Purpose: add USDA 2022 Census rows from the API in the same shape used by the
# existing .dta-based CAFO pipeline (rather than downloading broad raw extracts).

# API setup
DEFAULT_NASS_API_KEY = "30643212-7739-359A-B451-0EAD3D345DB9"
USE_API_SUPPLEMENT = True
API_SUPPLEMENT_YEARS = [2022]  # pull only years not present in local .dta files
NASS_API_KEY = os.environ.get("USDA_NASS_API_KEY", DEFAULT_NASS_API_KEY)
if USE_API_SUPPLEMENT and not NASS_API_KEY:
    raise RuntimeError("Missing USDA_NASS_API_KEY env var. Set it before running this script.")

NASS_BASE = "https://quickstats.nass.usda.gov/api/"

# Query filters for the API supplement.
# We keep this narrow: only combinations that can flow into CAFO ops mappings.
source_desc = "CENSUS"
agg_level_desc = "COUNTY"
sector_desc = "ANIMALS & PRODUCTS"
commodity_desc_allow = ["CATTLE", "CHICKENS", "HOGS"]
unit_desc_allow = ["OPERATIONS"]          # downstream CAFO construction uses operations rows
statisticcat_desc_allow = ["INVENTORY"]   # mapped size bins are inventory categories

# Split only when a request exceeds the NASS 50k cap.
commodity_domain_splits = {
    "CATTLE": [
        "INVENTORY",
        "INVENTORY OF CATTLE, INCL CALVES",
        "INVENTORY OF CATTLE, (EXCL COWS)",
        "INVENTORY OF BEEF COWS",
        "INVENTORY OF MILK COWS",
        "INVENTORY OF CATTLE ON FEED",
    ],
    "CHICKENS": [
        "INVENTORY",
    ],
    "HOGS": [
        "INVENTORY OF HOGS",
        "INVENTORY OF BREEDING HOGS",
    ],
}


# Canonical .dta column shape used downstream.
DTA_CORE_COLS = [
    "group_desc", "commodity_desc", "class_desc", "prodn_practice_desc", "util_practice_desc",
    "statisticcat_desc", "unit_desc", "short_desc", "domain_desc", "domaincat_desc",
    "agg_level_desc", "state_ansi", "state_fips_code", "state_alpha", "state_name",
    "asd_code", "asd_desc", "county_ansi", "county_code", "county_name", "location_desc",
    "year", "freq_desc", "begin_code", "end_code", "reference_period_desc", "value", "cv_",
]


def _harmonize_api_to_dta_schema(df_api):
    """
    Keep API pulls in a .dta-compatible schema so downstream code stays unchanged.
    """
    if df_api.empty:
        return df_api

    out = df_api.copy()
    out.columns = [c.lower().strip() for c in out.columns]

    # API uses "CV (%)"; .dta files use "CV_".
    if "cv (%)" in out.columns and "cv_" not in out.columns:
        out = out.rename(columns={"cv (%)": "cv_"})

    for c in DTA_CORE_COLS:
        if c not in out.columns:
            out[c] = pd.NA

    return out[DTA_CORE_COLS].copy()


def _filter_to_rows_used_downstream(df_raw):
    """
    Mirror the script's later CAFO filters early so API rows are already scoped to
    what can actually survive into the analysis tables.
    """
    if df_raw.empty:
        return df_raw

    out = df_raw.copy()
    for c in [
        "domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc",
        "commodity_desc", "group_desc", "class_desc"
    ]:
        if c in out.columns:
            out[c] = out[c].astype("string").str.strip().str.lower()

    comms_of_interest = ["cattle", "chickens", "hogs"]
    class_keep_map = {
        "cattle": {"incl calves", "(excl cows)", "cows, beef", "cows, milk", "calves", "calves, veal", "ge 500 lbs", "heifers, ge 500 lbs, milk replacement"},
        "chickens": {"broilers", "layers", "layers & pullets", "pullets, replacement", "roosters"},
        "hogs": {"all classes", "breeding"},
    }

    out = out[
        (out["commodity_desc"].isin(comms_of_interest))
        & (out["unit_desc"].isin(["operations", "head"]))
        & (out["statisticcat_desc"].isin(["inventory", "operations"]))
        & (out["domaincat_desc"].str.startswith("inventory", na=False))
    ].copy()

    allowed_pairs = {
        (commodity, cls)
        for commodity, classes in class_keep_map.items()
        for cls in classes
    }
    pair_index = pd.MultiIndex.from_frame(out[["commodity_desc", "class_desc"]])
    allowed_index = pd.MultiIndex.from_tuples(sorted(allowed_pairs))
    out = out[pair_index.isin(allowed_index)].copy()

    return out


# create function to define a call in the API for the ag data
def fetch_ag_data(years_to_pull):
    if not years_to_pull:
        return pd.DataFrame()

    def _keep_inventory_domaincat(df):
        """Keep only rows where domaincat_desc starts with 'inventory' (case-insensitive)."""
        if "domaincat_desc" in df.columns:
            s = df["domaincat_desc"].astype(str).str.strip().str.lower()
            return df[s.str.startswith("inventory", na=False)]
        if "DOMAINCAT_DESC" in df.columns:
            s = df["DOMAINCAT_DESC"].astype(str).str.strip().str.lower()
            return df[s.str.startswith("inventory", na=False)]
        return df

    def _safe_get_counts(params):
        try:
            return functions.nass_get_counts(NASS_BASE, params)
        except RuntimeError as e:
            print("Skipping combo due to API error:", e)
            return None

    def _safe_get_data(params):
        try:
            out = functions.nass_get_data(NASS_BASE, params)
        except RuntimeError as e:
            print("Skipping data pull due to API error:", e)
            return pd.DataFrame()
        return _keep_inventory_domaincat(out)

    frames = []
    rate_sleep = 0.4  # basic rate limiting

    for yr in years_to_pull:
        for cmd in commodity_desc_allow:
            for unit in unit_desc_allow:
                for stat in statisticcat_desc_allow: # call the units we want to pull
                    time.sleep(rate_sleep)
                    base_params = {
                        "key": NASS_API_KEY,
                        "source_desc": source_desc,
                        "agg_level_desc": agg_level_desc,
                        "sector_desc": sector_desc,
                        "commodity_desc": cmd,
                        "unit_desc": unit,
                        "statisticcat_desc": stat,
                        "year": yr,
                    }

                    count = _safe_get_counts(base_params)
                    if count is None or count == 0:
                        continue

                    if count <= 50000:
                        df_pull = _safe_get_data(base_params)
                        if not df_pull.empty:
                            frames.append(df_pull)
                        continue

                    # Still too large: split by commodity-specific domain_desc values
                    for dd in commodity_domain_splits.get(cmd, []):
                        time.sleep(rate_sleep)
                        dd_params = dict(base_params)
                        dd_params["domain_desc"] = dd
                        dd_count = _safe_get_counts(dd_params)
                        if dd_count is None or dd_count == 0:
                            continue
                        if dd_count > 50000:
                            print(
                                "Skipping domain split still >50k rows: "
                                f"year={yr}, commodity={cmd}, "
                                f"unit={unit}, statisticcat={stat}, domain_desc={dd}, count={dd_count}"
                            )
                            continue
                        df_pull = _safe_get_data(dd_params)
                        if not df_pull.empty:
                            frames.append(df_pull)

    if not frames:
        return pd.DataFrame()

    pulled = pd.concat(frames, ignore_index=True)
    pulled = _harmonize_api_to_dta_schema(pulled)
    pulled = _filter_to_rows_used_downstream(pulled)
    return pulled


# Load .dta files as the authoritative baseline for historical Census years.
# Then append API rows only for missing Census years (currently 2022).
agfolder = os.path.join(inf, "usda")
agfiles = sorted(glob.glob(os.path.join(agfolder, "*.dta")))
if not agfiles:
    raise RuntimeError(f"No .dta files found in {agfolder}. Cannot build CAFO panel.")
dta_frames = []
for f in agfiles:
    _df = pd.read_stata(f)
    _df.columns = [c.lower().strip() for c in _df.columns]
    dta_frames.append(_df)
combined_dta = pd.concat(dta_frames, ignore_index=True)
print(f"Loaded {len(combined_dta):,} rows from {len(agfiles)} local .dta files: {[os.path.basename(f) for f in agfiles]}")

# Optionally supplement with API rows for years absent from .dta.
dta_years = sorted(pd.to_numeric(combined_dta["year"], errors="coerce").dropna().astype(int).unique().tolist())
years_to_pull_api = [yr for yr in API_SUPPLEMENT_YEARS if yr not in dta_years]
combined_api = pd.DataFrame()
if USE_API_SUPPLEMENT and years_to_pull_api:
    try:
        combined_api = fetch_ag_data(years_to_pull_api)
        print(f"Loaded {len(combined_api):,} filtered rows from USDA NASS API for years {years_to_pull_api}")
    except RuntimeError as e:
        print(f"API fetch failed (non-fatal — proceeding with .dta baseline only): {e}")
elif USE_API_SUPPLEMENT:
    print(f"API supplement skipped: local .dta already covers requested years {API_SUPPLEMENT_YEARS}.")

combined = combined_dta.copy()
if not combined_api.empty:
    before_append = len(combined)
    combined = pd.concat([combined, combined_api], ignore_index=True)
    dedupe_cols = [c for c in DTA_CORE_COLS if c in combined.columns]
    before_dedupe = len(combined)
    combined = combined.drop_duplicates(subset=dedupe_cols, keep="first")
    print(
        "Appended API supplement rows: "
        f"pre-append={before_append:,}, post-append={before_dedupe:,}, "
        f"post-dedupe={len(combined):,}"
    )
else:
    print(f"Using .dta baseline only ({len(combined):,} rows).")
    
    
    
# ----------------------- SET UP PART 3 : STANDARDIZE COMBINED DATA   -------------------- -#
# The .dta files loaded above are authoritative for historical years
# (2002, 2007, 2012, 2017). API rows (if appended) are harmonized to the same
# schema and filtered to the same analysis-relevant subset (currently 2022).
# The old donor backfill introduced pre-processed rows (with fips_generated already set)
# that conflict with the raw .dta rows (no fips_generated), causing the FIPS merge to fail.
combined = combined.loc[:, ~combined.columns.duplicated()].copy()
print("Proceeding with .dta baseline (+ API supplement, if available). No donor backfill applied.")

# check we have ops in presence 
chk = combined.copy()
for c in ["commodity_desc", "unit_desc"]:
    chk[c] = chk[c].astype("string").str.strip().str.lower()
chk["year"] = pd.to_numeric(chk["year"], errors="coerce").astype("Int64")
chk = chk[
    chk["commodity_desc"].isin(["cattle", "chickens", "hogs"]) &
    chk["year"].isin([2012, 2017, 2022])
]

# quick eye-ball QA 
print(chk.groupby(["year", "unit_desc"]).size().unstack(fill_value=0))

print(combined["year"].value_counts().sort_index().tail(10))



# ----------------------- DATA PART 1 : QUALITY ASSURANCE  -------------------- -#

# basic QA
print("Columns:", sorted(combined.columns))
print(combined.dtypes)
    
ag_iterated = clean_cols(combined).copy()
df = ag_iterated.copy()

for c in [
    "domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc",
    "commodity_desc", "group_desc", "class_desc"
]:
    df[c] = df[c].astype("string").str.strip().str.lower()

comms_of_interest = ["cattle", "chickens", "hogs"]

class_keep_map = {
    "cattle": {"incl calves","(excl cows)","cows, beef","cows, milk","calves","calves, veal","ge 500 lbs","heifers, ge 500 lbs, milk replacement"},
    "chickens": {"broilers","layers","layers & pullets","pullets, replacement","roosters"},
    "hogs": {"all classes","breeding"},
}

# check subset of the dataframe, checkc to see we have the necessary values in each column 
df_sub = df[
    (df["commodity_desc"].isin(comms_of_interest)) &
    (df["unit_desc"].isin(["operations", "head"])) &
    (df["statisticcat_desc"].isin(["inventory", "operations"])) &
    (df["domaincat_desc"].str.startswith("inventory", na=False))
].copy()

allowed_pairs = {
    (commodity, cls)
    for commodity, classes in class_keep_map.items()
    for cls in classes
}
pair_index = pd.MultiIndex.from_frame(df_sub[["commodity_desc", "class_desc"]])
allowed_index = pd.MultiIndex.from_tuples(sorted(allowed_pairs))
df_sub = df_sub[pair_index.isin(allowed_index)].copy()

print("df_sub rows:", len(df_sub))

# check value counts 
df_sub["commodity_desc"].value_counts()
df_sub["class_desc"].value_counts().head(20)
df_sub["unit_desc"].value_counts()
df_sub["statisticcat_desc"].value_counts()





# ----------------------- DATA PART 2 : MATCH FIPS to AG DATA   -------------------- -#

matches = glob.glob(os.path.join(outf, "*fips_full*.csv")) # pull the most recent fips file 
if matches:
    fips_sense = max(matches, key=os.path.getmtime)
    print("Using:", fips_sense)
else:
    print("No matching file found.")
    
fips_df = pd.read_csv(fips_sense)   # upload fips_df
fips_df = clean_cols(fips_df)

# standardize fips key from external file
fips_df["fips"] = pd.to_numeric(fips_df["fips"], errors="coerce").astype("Int64").astype("string").str.zfill(5)
fips_df["year"] = pd.to_numeric(fips_df["year"], errors="coerce").astype("Int64")

fips_key = (
    fips_df[["fips", "year", "county"]]
    .rename(columns={"county": "county_fips_name"})
    .drop_duplicates()
)

dupe_key = fips_key.duplicated(subset=["fips", "year"]).sum()
if dupe_key:
    raise RuntimeError(f"fips_full has duplicate fips-year keys: {dupe_key}")





# ----------------------- DATA PART 3 : FORWARD FILL MISSING AG DATA   -------------------- -#

# prep ag raw df for iteration, then export and saving
ag_raw_df = clean_cols(combined)
ag_raw_df.head()

# create total fips code 
ag_raw_df = generate_fips(ag_raw_df, state_col="state_fips_code", city_col="county_code")
ag_raw_df.columns
# normalize possible FIPS column variants and guard against duplicate column names
if "FIPS_generated" in ag_raw_df.columns and "fips_generated" not in ag_raw_df.columns:
    ag_raw_df = ag_raw_df.rename(columns={"FIPS_generated": "fips_generated"})
elif "FIPS_generated" in ag_raw_df.columns and "fips_generated" in ag_raw_df.columns:
    # keep one canonical column name/value path
    ag_raw_df = ag_raw_df.drop(columns=["FIPS_generated"])

if ag_raw_df.columns.duplicated().any():
    ag_raw_df = ag_raw_df.loc[:, ~ag_raw_df.columns.duplicated()]

if "fips_generated" not in ag_raw_df.columns:
    raise RuntimeError("fips_generated column missing after FIPS generation/normalization")

ag_raw_df["fips_generated"] = ag_raw_df["fips_generated"].astype("string").str.zfill(5)
ag_raw_df["year"] = pd.to_numeric(ag_raw_df["year"], errors="coerce").astype("Int64")


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
      (len_df_big_post_dupe == len_df_big_predupe == (len_it_rows + len_raw_df))) # INTERPRETATION: 572,656 county observations (after filter for our commodities) from 2002-2021
# recall that USDA aggregates to keep anonymity of the survey, so the FIPS level will not give us the exact number of rows = number of farms 

# attach external fips-year key and county name
if "_merge" in df_big.columns:
    df_big = df_big.drop(columns=["_merge"])

df_big = df_big.merge(
    fips_key,
    left_on=["fips_generated", "year"],
    right_on=["fips", "year"],
    how="left",
    validate="m:1",
    indicator=True,
)

print("FIPS merge status:") # check merge status 
print(df_big["_merge"].value_counts(dropna=False))

def _norm_county_name(s):
    s = s.astype("string").str.lower().str.strip()
    s = s.str.replace(r"[^a-z0-9 ]", "", regex=True)
    s = s.str.replace(r"\b(county|parish|borough|census area|municipality|city and borough)\b", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

# reset names in the variables 
df_big["county_name_norm"] = _norm_county_name(df_big["county_name"])
df_big["county_fips_name_norm"] = _norm_county_name(df_big["county_fips_name"])

name_mismatch = df_big[
    df_big["county_name_norm"].notna()
    & df_big["county_fips_name_norm"].notna()
    & (df_big["county_name_norm"] != df_big["county_fips_name_norm"])
].copy()

missing_fips_key = df_big[df_big["_merge"] != "both"].copy()



clean_ag_census = f"{today_str}_ag_annual_df.csv"
ag_path = os.path.join(outf, clean_ag_census)
df_big.to_csv(ag_path, index=False) # save a version of the clean, large agriculture census data set as an interim output 
print("Saved iterated AG panel:", ag_path)



review_outf = interim
os.makedirs(review_outf, exist_ok=True)
mismatch_path = os.path.join(review_outf, f"{today_str}_ag_fips_name_mismatch.csv")
missing_key_path = os.path.join(review_outf, f"{today_str}_ag_fips_missing_key.csv")
name_mismatch.to_csv(mismatch_path, index=False)
missing_fips_key.to_csv(missing_key_path, index=False)
# print out a QA to see if we need to manually check any fips codes 
print("Saved county-name mismatch rows for manual review:", mismatch_path)
print("Saved missing fips-year key rows for manual review:", missing_key_path)



# ----------------------- ANALYSIS PART 1 : DEFINE CAFO VARIABLES   -------------------- -#
# Input for Stage 2 is df_big from Stage 1 (iterated annual panel with FIPS merge).

# new copy - this way we don't mess up the data cleaning stage 
df_cafo = df_big.copy()

# these are just the vars we need to keep that contain the info relevant for our analysis
for c in [
    "domaincat_desc", "unit_desc", "statisticcat_desc", "domain_desc",
    "commodity_desc", "group_desc", "class_desc"
]:
    if c in df_cafo.columns:
        df_cafo[c] = df_cafo[c].astype("string").str.strip().str.lower()

# picking commodities of interest
comms_of_interest = ["cattle", "chickens", "hogs"]


# within each of the commodities, there are sub-categories, so we map these classes 
class_keep_map = {
    "cattle": {
        "incl calves",
        "(excl cows)",
        "cows, beef",
        "cows, milk",
        "calves",
        "calves, veal",
        "ge 500 lbs",
        "heifers, ge 500 lbs, milk replacement",
    },
    "chickens": {
        "broilers",
        "layers",
        "layers & pullets",
        "pullets, replacement",
        "roosters",
    },
    "hogs": {
        "all classes",
        "breeding",
    },
}

# keeping inventory bins most crucially, and the the other vars within the dataframe
df_sub = df_cafo[
    (df_cafo["commodity_desc"].isin(comms_of_interest))
    & (df_cafo["unit_desc"].isin(["operations", "head"]))
    & (df_cafo["statisticcat_desc"].isin(["inventory", "operations"]))
    & (df_cafo["domaincat_desc"].str.startswith("inventory", na=False))
].copy()

allowed_pairs = {
    (commodity, cls)
    for commodity, classes in class_keep_map.items()
    for cls in classes
}
pair_index = pd.MultiIndex.from_frame(df_sub[["commodity_desc", "class_desc"]])
allowed_index = pd.MultiIndex.from_tuples(sorted(allowed_pairs))
df_sub = df_sub[pair_index.isin(allowed_index)].copy()

# print the QA so we can see how many of each var remain 
print("Stage 2 df_sub rows:", len(df_sub))
print("Stage 2 commodity mix:")
print(df_sub["commodity_desc"].value_counts(dropna=False))
print("Stage 2 unit mix:")
print(df_sub["unit_desc"].value_counts(dropna=False))
print("Stage 2 statistic mix:")
print(df_sub["statisticcat_desc"].value_counts(dropna=False))



# Inventory-bin mappings (preserved from script0b-ag-raw-v2) which just label the text descriptions numerically for ease 
layer_map = {
    "inventory: (1 to 49 head)": 1,
    "inventory: (50 to 99 head)": 2,
    "inventory: (100 to 399 head)": 3,
    "inventory: (400 to 3,199 head)": 4,
    "inventory: (3,200 to 9,999 head)": 5,
    "inventory: (10,000 to 19,999 head)": 6,
    "inventory: (20,000 to 49,999 head)": 7,
    "inventory: (50,000 to 99,999 head)": 8,
    "inventory: (100,000 or more head)": 9,
}

cattle_inv_map = {
    "inventory of cattle, incl calves: (1 to 9 head)": 1,
    "inventory of cattle, incl calves: (10 to 19 head)": 2,
    "inventory of cattle, incl calves: (20 to 49 head)": 3,
    "inventory of cattle, incl calves: (50 to 99 head)": 4,
    "inventory of cattle, incl calves: (100 to 199 head)": 5,
    "inventory of cattle, incl calves: (200 to 499 head)": 6,
    "inventory of cattle, incl calves: (500 or more head)": 7,
}

hog_inv_map = {
    "inventory of hogs: (1 to 24 head)": 1,
    "inventory of hogs: (25 to 49 head)": 2,
    "inventory of hogs: (50 to 99 head)": 3,
    "inventory of hogs: (100 to 199 head)": 4,
    "inventory of hogs: (200 to 499 head)": 5,
    "inventory of hogs: (500 to 999 head)": 6,
    "inventory of hogs: (1,000 or more head)": 7,
}

milk_cows_map = {
    "inventory of milk cows: (1 to 9 head)": 1,
    "inventory of milk cows: (10 to 19 head)": 2,
    "inventory of milk cows: (20 to 49 head)": 3,
    "inventory of milk cows: (50 to 99 head)": 4,
    "inventory of milk cows: (100 to 199 head)": 5,
    "inventory of milk cows: (200 to 499 head)": 6,
    "inventory of milk cows: (500 or more head)": 7,
}

breeding_hogs_map = {
    "inventory of breeding hogs: (1 to 24 head)": 1,
    "inventory of breeding hogs: (25 to 49 head)": 2,
    "inventory of breeding hogs: (50 to 99 head)": 3,
    "inventory of breeding hogs: (100 or more head)": 4,
}

cattle_inv_map_no_cows = {
    "inventory of cattle, (excl cows): (1 to 9 head)": 1,
    "inventory of cattle, (excl cows): (10 to 19 head)": 2,
    "inventory of cattle, (excl cows): (100 to 199 head)": 3,
    "inventory of cattle, (excl cows): (20 to 49 head)": 4,
    "inventory of cattle, (excl cows): (200 to 499 head)": 5,
    "inventory of cattle, (excl cows): (50 to 99 head)": 6,
}

cattle_feed_map = {
    "inventory of cattle on feed: (1 to 19 head)": 1,
    "inventory of cattle on feed: (1 to 9 head)": 2,
    "inventory of cattle on feed: (10 to 19 head)": 3,
    "inventory of cattle on feed: (100 to 199 head)": 4,
    "inventory of cattle on feed: (20 to 49 head)": 5,
    "inventory of cattle on feed: (200 to 499 head)": 6,
    "inventory of cattle on feed: (50 to 99 head)": 7,
    "inventory of cattle on feed: (500 or more head)": 8,
}

beef_cows_map = {
    "inventory of beef cows: (1 to 9 head)": 1,
    "inventory of beef cows: (10 to 19 head)": 2,
    "inventory of beef cows: (20 to 49 head)": 3,
    "inventory of beef cows: (50 to 99 head)": 4,
    "inventory of beef cows: (100 to 199 head)": 5,
    "inventory of beef cows: (200 to 499 head)": 6,
    "inventory of beef cows: (500 or more head)": 7,
}


# apply mapping to the data 
map_size_class(df_sub, layer_map, unit_match="operations", class_match="layers", out_col="layer_ops_size")
map_size_class(df_sub, layer_map, unit_match="operations", class_match="broilers", out_col="broiler_ops_size")

map_size(df_sub, cattle_inv_map, unit_match="operations", out_col="cattle_ops_size_inv")
map_size(df_sub, hog_inv_map, unit_match="operations", out_col="hog_ops_size_inv")
map_size(df_sub, milk_cows_map, unit_match="operations", out_col="dairy_ops_size_inv")
map_size(df_sub, breeding_hogs_map, unit_match="operations", out_col="breed_hog_ops_size_inv")
map_size(df_sub, cattle_inv_map_no_cows, unit_match="operations", out_col="cattle_senzcow_ops_size_inv")
map_size(df_sub, cattle_feed_map, unit_match="operations", out_col="cattle_feed_ops_size_inv")
map_size(df_sub, beef_cows_map, unit_match="operations", out_col="beef_ops_size_inv")

# Create a numeric-bin thresholds for S/M/L assignment (preserved from CAFO - USDA definition, roughly) 
broiler_cutoff_lrg = 5
broiler_cutoff_med = 3
layer_cutoff_lrg = 9
layer_cutoff_med = 7
cattle_cutoff_lrg = 7
cattle_cutoff_med = 6
hog_cutoff_lrg = 7
hog_cutoff_med = 6

# define them as thresholds 
col_thresholds = {
    "layer_ops_size": (layer_cutoff_med, layer_cutoff_lrg),
    "broiler_ops_size": (broiler_cutoff_med, broiler_cutoff_lrg),
    "cattle_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "dairy_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "cattle_senzcow_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "cattle_feed_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "beef_ops_size_inv": (cattle_cutoff_med, cattle_cutoff_lrg),
    "hog_ops_size_inv": (hog_cutoff_med, hog_cutoff_lrg),
    "breed_hog_ops_size_inv": (hog_cutoff_med, hog_cutoff_lrg),
}

# create a way to categorize them 
def categorize_code(v, med, lrg):
    if pd.isna(v):
        return pd.NA
    if v < med:
        return "small"
    if v < lrg:
        return "medium"
    return "large"


# ----------------------- ANALYSIS PART 2 : CREATE CAFO VARIABLES   -------------------- -#
df2 = df_sub.copy() # make another copy after the classification and before the development of new variables 
df2["size_class"] = pd.Series(pd.NA, index=df2.index, dtype="string")
df2["size_source"] = pd.Series(pd.NA, index=df2.index, dtype="string")

# create categorization
for col, (med, lrg) in col_thresholds.items():
    codes = df2[col]
    take = codes.notna() & df2["size_source"].isna()
    df2.loc[take, "size_class"] = codes[take].apply(categorize_code, args=(med, lrg))
    df2.loc[take, "size_source"] = col

# Keep operations counts in each inventory bin 
df2["ops_in_bin"] = pd.to_numeric(
    df2["value"]
    .astype("string")
    .str.replace(",", "", regex=False)
    .str.strip()
    .replace({"(d)": pd.NA, "(z)": pd.NA, "": pd.NA}),
    errors="coerce",
)

# Row-level CAFO flags (mapped rows only) 
df2["is_large_cafo_row"] = ((df2["size_class"] == "large") & df2["size_source"].notna()).astype("Int8")
df2["is_medium_or_large_cafo_row"] = (
    df2["size_class"].isin(["medium", "large"]) & df2["size_source"].notna()
).astype("Int8")

# Compact long summary: county-year-commodity-class-size 
summary = (
    df2[df2["size_source"].notna()]
    .groupby(
        [
            "year",
            "fips_generated",
            "county_fips_name",
            "commodity_desc",
            "class_desc",
            "statisticcat_desc",
            "size_source",
            "size_class",
        ],
        as_index=False,
    )["ops_in_bin"]
    .sum(min_count=1)
    .rename(columns={"ops_in_bin": "sum_ops"})
)

# County-year compact table with small/medium/large columns
summary_compact = (
    summary.pivot_table(
        index=["year", "fips_generated", "county_fips_name", "commodity_desc", "class_desc"],
        columns="size_class",
        values="sum_ops",
        aggfunc="sum",
        fill_value=0,
    )
    .reset_index()
)
summary_compact.columns.name = None
for size_col in ["small", "medium", "large"]:
    if size_col not in summary_compact.columns:
        summary_compact[size_col] = 0
summary_compact["any_large_cafo"] = (summary_compact["large"] > 0).astype("Int8")
summary_compact["any_medium_or_large_cafo"] = (
    (summary_compact["medium"] + summary_compact["large"]) > 0
).astype("Int8")

# QA: cattle canonical vs subgroup overlap diagnostic at county-year level
# Purpose: verify whether cattle subclasses can be summed without overlap.
# Canonical reference = class_desc "incl calves".
cattle_compact = summary_compact[summary_compact["commodity_desc"] == "cattle"].copy()
if not cattle_compact.empty:
    cattle_compact["ops_total"] = cattle_compact[["small", "medium", "large"]].sum(axis=1, min_count=1)
    cattle_wide = (
        cattle_compact.pivot_table(
            index=["year", "fips_generated", "county_fips_name"],
            columns="class_desc",
            values="ops_total",
            aggfunc="sum",
        )
        .reset_index()
    )
    cattle_wide.columns.name = None

    canonical_col = "incl calves"
    noncanonical_cols = [c for c in cattle_wide.columns if c not in {"year", "fips_generated", "county_fips_name", canonical_col}]
    partition_candidate_cols = [c for c in ["(excl cows)", "cows, beef", "cows, milk"] if c in cattle_wide.columns]

    cattle_wide["canonical_ops_incl_calves"] = pd.to_numeric(cattle_wide.get(canonical_col), errors="coerce")
    cattle_wide["sum_noncanonical_ops"] = (
        cattle_wide[noncanonical_cols].sum(axis=1, min_count=1) if noncanonical_cols else np.nan
    )
    cattle_wide["sum_partition_candidate_ops"] = (
        cattle_wide[partition_candidate_cols].sum(axis=1, min_count=1) if partition_candidate_cols else np.nan
    )
    cattle_wide["sum_all_class_ops"] = (
        cattle_wide["canonical_ops_incl_calves"] + cattle_wide["sum_noncanonical_ops"]
    )

    for lhs in ["sum_noncanonical_ops", "sum_partition_candidate_ops", "sum_all_class_ops"]:
        ratio_col = f"ratio_{lhs}_to_canonical"
        diff_col = f"abs_pct_diff_{lhs}_vs_canonical"
        cattle_wide[ratio_col] = np.where(
            cattle_wide["canonical_ops_incl_calves"] > 0,
            cattle_wide[lhs] / cattle_wide["canonical_ops_incl_calves"],
            np.nan,
        )
        cattle_wide[diff_col] = np.where(
            cattle_wide["canonical_ops_incl_calves"] > 0,
            (cattle_wide[lhs] - cattle_wide["canonical_ops_incl_calves"]).abs() / cattle_wide["canonical_ops_incl_calves"] * 100,
            np.nan,
        )

    cattle_overlap_path = os.path.join(outf, f"{today_str}_qa_cattle_class_overlap_county_year.csv")
    cattle_wide.to_csv(cattle_overlap_path, index=False)

    year_diag = (
        cattle_wide.groupby("year", as_index=False)
        .agg(
            county_years=("canonical_ops_incl_calves", "size"),
            canonical_sum=("canonical_ops_incl_calves", "sum"),
            partition_sum=("sum_partition_candidate_ops", "sum"),
            all_class_sum=("sum_all_class_ops", "sum"),
            median_ratio_partition_to_canonical=("ratio_sum_partition_candidate_ops_to_canonical", "median"),
            median_ratio_all_to_canonical=("ratio_sum_all_class_ops_to_canonical", "median"),
        )
    )
    year_diag["ratio_partition_sum_to_canonical_sum"] = np.where(
        year_diag["canonical_sum"] > 0,
        year_diag["partition_sum"] / year_diag["canonical_sum"],
        np.nan,
    )
    year_diag["ratio_allclass_sum_to_canonical_sum"] = np.where(
        year_diag["canonical_sum"] > 0,
        year_diag["all_class_sum"] / year_diag["canonical_sum"],
        np.nan,
    )
    cattle_overlap_year_path = os.path.join(outf, f"{today_str}_qa_cattle_class_overlap_by_year.csv")
    year_diag.to_csv(cattle_overlap_year_path, index=False)

    print("Saved cattle overlap QA (county-year):", cattle_overlap_path)
    print("Saved cattle overlap QA (year-level):", cattle_overlap_year_path)
    print("Cattle all-class vs canonical (year-level ratio):")
    print(year_diag[["year", "ratio_allclass_sum_to_canonical_sum"]].to_string(index=False))


# QA - print on the missing values, and the total values of the vars for manual inspection
print("Stage 2 mapped rows:", int(df2["size_source"].notna().sum()))
print("Stage 2 size class counts:")
print(df2["size_class"].value_counts(dropna=False))
print("Stage 2 compact rows:", len(summary_compact))

# checking no years were dropped
print(df2[df2["size_source"].notna()]["year"].value_counts().sort_index().tail(15))
print(summary["year"].min(), summary["year"].max())
print(summary_compact["year"].min(), summary_compact["year"].max())


# ----------------------- ANALYSIS PART 3: EXPORT   -------------------- -#

# create different versions of the CAFO data, we will use the cafo_ops given ops categorization is most important 
clean_cafo_row = f"{today_str}_cafo_annual_df.csv"
clean_cafo_long = f"{today_str}_cafo_ops_by_size_long.csv"
clean_cafo_compact = f"{today_str}_cafo_ops_by_size_compact.csv"

cafo_row_path = os.path.join(outf, clean_cafo_row)
cafo_long_path = os.path.join(outf, clean_cafo_long)
cafo_compact_path = os.path.join(outf, clean_cafo_compact)

df2.to_csv(cafo_row_path, index=False)
summary.to_csv(cafo_long_path, index=False)
summary_compact.to_csv(cafo_compact_path, index=False)

print("Saved CAFO row-level panel:", cafo_row_path)
print("Saved CAFO long summary:", cafo_long_path)
print("Saved CAFO compact summary:", cafo_compact_path)
