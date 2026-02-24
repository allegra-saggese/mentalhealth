#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:03:00 2025

@author: allegrasaggese
"""

# load packages and workspaces
from collections import Counter
import sys

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

# 50 states + DC (exclude territories for US county panels).
US_STATE_CODES = {
    "01", "02", "04", "05", "06", "08", "09", "10", "11", "12", "13", "15",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27",
    "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
    "40", "41", "42", "44", "45", "46", "47", "48", "49", "50", "51", "53",
    "54", "55", "56"
}

# Census 2024 agesex file uses year codes; this mapping fills 2021-2024.
YEAR_CODE_2024_MAP = {3: 2021, 4: 2022, 5: 2023, 6: 2024}

# other folders
inf = os.path.join(db_data, "raw") # input 
outf = os.path.join(db_data, "clean") #outpit

# ----------------------- DATA PART 1 : CREATE FIPS KEY  -------------------- -#


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
        df = pd.read_csv(file, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            # Retry with fallback encoding
            df = pd.read_csv(file, encoding="latin1")
        except Exception as e:
            print(f"Failed to read {file}: {e}")
            continue
    dfs.append(df)
    pop_sources[os.path.basename(file).lower()] = df


def _pick_pop_source(name_fragment):
    matches = [k for k in pop_sources.keys() if name_fragment in k]
    if not matches:
        raise FileNotFoundError(f"Population source not found for pattern: {name_fragment}")
    if len(matches) > 1:
        print(f"Multiple files matched '{name_fragment}', using first: {matches[0]}")
    return pop_sources[matches[0]].copy()


df_pop_2024 = _pick_pop_source("cc-est2024-agesex-all")
df_pop_2000s = _pick_pop_source("cc-est00int-tot")
df_pop_2010s = _pick_pop_source("cc-est2010-2020")
df_pop_1990s = _pick_pop_source("cc-est-1990-2000")
    
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
yrs0 = list(range(1990, 2000))
yrs0df = pd.DataFrame({"year": yrs0})
fips_90_expanded = fips_90.merge(yrs0df, how="cross")
fips_90_expanded = fips_90_expanded.reset_index(drop=True)


# comma file (2010)
if not comma_files:
    raise FileNotFoundError("No comma-delimited 2010 FIPS file found in raw/fips")
comma_2010 = [p for p in comma_files if "2010" in os.path.basename(p).lower()]
comma_target = comma_2010[0] if comma_2010 else comma_files[0]
fips_10 = pd.read_table(comma_target, sep=",", engine="python", encoding="latin1", on_bad_lines="warn")
# create full FIPS
generate_fips(fips_10, state_col="STATEFP", city_col="COUNTYFP")

# collapse on subfips because we don't need them 
fips_10 = fips_10.drop(columns=['COUSUBFP', 'COUSUBNAME', 'FUNCSTAT'], errors='ignore')
fips_10 = fips_10.drop_duplicates(subset='FIPS_generated', keep='first')

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

# similar to fips_10 drop subfips! 
fips_20 = fips_20.drop(columns=['COUSUBFP', 'COUSUBNAME', 'FUNCSTAT'], errors='ignore')
fips_20 = fips_20.drop_duplicates(subset='FIPS_generated', keep='first')

  
# add years
yrs3 = list(range(2020, 2025))
yrs3df = pd.DataFrame({"year": yrs3})
fips_20_expanded = fips_20.merge(yrs3df, how="cross")
fips_20_expanded = fips_20_expanded.reset_index(drop=True)

# change county, state code colnames to ensure merge works 
fips_90_expanded.columns.tolist()
fips_90_expanded.drop(columns=["FIPS"], inplace=True) # empty col to drop

fips_00_expanded.columns.tolist()
fips_10_expanded.columns.tolist()
fips_20_expanded.columns.tolist()

# make lower case
fips_10_expanded.columns = fips_10_expanded.columns.str.lower()
fips_20_expanded.columns = fips_20_expanded.columns.str.lower()

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


# combine all FIPS data 
fips_annual_full = pd.concat([fips_90_expanded, fips_00_expanded, 
                              fips_10_expanded, fips_20_expanded],
                             axis=0, join="outer", ignore_index=True)


#### QA before further cleaning 
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

# remove state level values (keeping DC), based on county_code
fips_annual_full_v2 = fips_annual_full[
    ~((fips_annual_full['county_code'] == "000") &
      (fips_annual_full['county'] != "District of Columbia"))
].copy()

# standardize identifier columns
fips_annual_full_v2["state_code"] = (
    fips_annual_full_v2["state_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(2)
)
fips_annual_full_v2["county_code"] = (
    fips_annual_full_v2["county_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(3)
)
fips_annual_full_v2["fips"] = pd.to_numeric(fips_annual_full_v2["fips"], errors="coerce").astype("Int64")

# normalize DC to county-equivalent FIPS 11001 across all years
dc_mask = fips_annual_full_v2["county"].astype(str).str.contains("District of Columbia", case=False, na=False)
fips_annual_full_v2.loc[dc_mask, "state_code"] = "11"
fips_annual_full_v2.loc[dc_mask, "county_code"] = "001"
fips_annual_full_v2.loc[dc_mask, "fips"] = 11001

# keep US counties + DC only
fips_annual_full_v2 = fips_annual_full_v2[fips_annual_full_v2["state_code"].isin(US_STATE_CODES)].copy()

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


# ----------------------- DATA PART 2: POPULATION  -------------------- -#


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


## CLEAN UP ENVIRO 
del all_cols, all_columns, col_counts, col_lists, col_presence,     
del ws_dfs, ws_files
del yrs0, yrs2, yrs0df, yrs2df, yrs3, yrs3df


######### ------- DATA CLEANING - POPULATION ------ ####### ####### ####### 
##------- MAKE THE SAME DATAFRAME FOR ALL FOUR SETS OF POPULATION DATA 

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

df2024_full = df2024.rename(columns={
    "stname": "state",
    "ctyname": "county",
    "state": "state_code",
    "county": "county_code",
    "FIPS_generated": "fips",
    "fips_generated": "fips",
    "popestimate": "population"
})

df2024_full["state_code"] = df2024_full["state_code"].astype(str).str.zfill(2)
df2024_full["county_code"] = df2024_full["county_code"].astype(str).str.zfill(3)
df2024_full = df2024_full[df2024_full["state_code"].isin(US_STATE_CODES)].copy()
df2024_full = df2024_full[df2024_full["fips"].astype(str).str.len() == 5].copy()

# keep only columns shared with annual panel
keep_2024 = [c for c in df2010_full_v2.columns if c in df2024_full.columns]
df2024_full = df2024_full[keep_2024].copy()

###### MERGING ALL TOGETHER ######

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

full_pop_df = pd.concat([df1990s_full, df2000s_full, df2010_full_v2, df2024_full], ignore_index=True, sort=False)

full_pop_df["state_code"] = (
    full_pop_df["state_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(2)
)
full_pop_df["county_code"] = (
    full_pop_df["county_code"].astype(str).str.extract(r"(\d+)")[0].str.zfill(3)
)
full_pop_df["fips"] = (
    pd.to_numeric(full_pop_df["fips"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(5)
)
full_pop_df["year"] = pd.to_numeric(full_pop_df["year"], errors="coerce").astype("Int64")
full_pop_df["population"] = pd.to_numeric(full_pop_df["population"], errors="coerce")

# normalize DC and keep US-only scope
dc_mask2 = full_pop_df["county"].astype(str).str.contains("District of Columbia", case=False, na=False)
full_pop_df.loc[dc_mask2, "state_code"] = "11"
full_pop_df.loc[dc_mask2, "county_code"] = "001"
full_pop_df.loc[dc_mask2, "fips"] = "11001"
full_pop_df = full_pop_df[full_pop_df["state_code"].isin(US_STATE_CODES)].copy()
full_pop_df = full_pop_df[full_pop_df["fips"] != "11000"].copy()
full_pop_df = full_pop_df.dropna(subset=["fips", "year"]).copy()
full_pop_df = full_pop_df.sort_values(["fips", "year"]).drop_duplicates(["fips", "year"], keep="last")

# QA MERGE 
numeric_cols = full_pop_df.select_dtypes(include=[np.number])
ranges = numeric_cols.agg(['min', 'max']).T

# unique counts
unique_counts = {
    "state_code_unique": full_pop_df["state_code"].nunique(),
    "county_code_unique": full_pop_df["county_code"].nunique()
}

# missing stats
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

# export to CSV 
clean_pop_df = f"{today_str}_population_full.csv"
poppath = os.path.join(clean_dir, clean_pop_df)
full_pop_df.to_csv(poppath, index=False)
