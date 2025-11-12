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
from functools import reduce
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
## UPLOAD ALL COUNTY HEALTH DATA - ALL YEARS, RAW 
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

# add the years
year_re = re.compile(r'(19|20)\d{2}(?=\.[A-Za-z0-9]+$)')

# extract years from mh_files in order
years = []
for p in mh_files:
    base = os.path.basename(p)
    m = year_re.search(base)
    if not m:
        raise ValueError(f"Couldn't find a year at end of filename: {base}")
    years.append(int(m.group(0)))

# sanity check
if len(years) != len(dfs):
    raise ValueError(f"Lengths differ: {len(years)} files vs {len(dfs)} dataframes")

# assign year to each df by index -- note the list of names is ordered from mh_files
for i, y in enumerate(years):
    dfs_shortlist[i]["year"] = y


# merge all dfs together 
for i, d in enumerate(dfs_shortlist, 1):
    try:
        _ = d["5-digit_fips_code"]
        _ = d["year"]
        pd.to_numeric(d["year"], errors="raise")  # will raise if not numeric
        print(f"df #{i}: OK  (rows={len(d)})")
    except Exception as e:
        print(f"df #{i}: FAIL -> {type(e).__name__}: {e}")
        
def prep(df):
    """Schange type of key before merging """
    df["5-digit_fips_code"] = df["5-digit_fips_code"].astype(str).str.zfill(5)
    df["year"] = df["year"].astype(int)
    return df

dfs_ready = [prep(d) for d in dfs_shortlist]  # dfs_shortlist already has year set per file

for i, d in enumerate(dfs_ready, 1):
    try:
        _ = d["5-digit_fips_code"]
        _ = d["year"]
        pd.to_numeric(d["year"], errors="raise")  # will raise if not numeric
        print(f"df #{i}: OK  (rows={len(d)})")
    except Exception as e:
        print(f"df #{i}: FAIL -> {type(e).__name__}: {e}")
         
 # still OK after the prep
        
# nothing missing - so try again with a different prep function
def prep2(df: pd.DataFrame) -> pd.DataFrame:
    if "5-digit_fips_code" not in df.columns:
        raise KeyError("Missing column: 5-digit_fips_code")
        print("ERROR fips")
    if "year" not in df.columns:
        raise KeyError("Missing column: year")

    df = df.copy()
    df["5-digit_fips_code"] = df["5-digit_fips_code"].astype(str).str.zfill(5)
    # allow non-numeric/blank → NA; keep nullable integer dtype
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df

dfs_ready_v2 = [prep2(d) for d in dfs_ready]  # no issue 

# failing concat bc of index 
# checking for existing dupes -- in 2010 data 
for i, d in enumerate(dfs_ready_v2 , 1):
    print(
        f"df #{i}: dup_cols={d.columns.duplicated().any()}  "
        f"nonunique_index={not d.index.is_unique}  "
        f"ncols={len(d.columns)}  nrows={len(d)}"
    )
    
dfs_ready_v2[0] = dfs_ready_v2[0].loc[:, ~dfs_ready_v2[0].columns.duplicated()]

# CONCAT - works! 
combined = pd.concat(dfs_ready_v2, ignore_index=True, sort=False)


# ensure one row per FIPS × year (only needed if duplicates exist)
def first_notna(s):
    idx = s.first_valid_index()
    return s.loc[idx] if idx is not None else pd.NA

if combined.duplicated(["5-digit_fips_code", "year"]).any():
    combined = (combined
        .groupby(["5-digit_fips_code", "year"], as_index=False)
        .agg(first_notna))

# tidy order
keys = ["5-digit_fips_code", "year"]
other = [c for c in combined.columns if c not in keys]
combined = combined[keys + other].sort_values(keys).reset_index(drop=True)

# export to CSV file
today_str = date.today().strftime("%Y-%m-%d")
 
clean_MH_data = f"{today_str}_mentalhealthrank_full.csv"
mh_path = os.path.join(outf, clean_MH_data)
combined.to_csv(mh_path, index=False)



### UPLOAD CDC MORTALITY DATA - NOTE THIS DATA IS ONLY FOR FIVE STATES!!!!!! 
#### NEW ALL CAUSE MORTALITY DATA TO BE UPLOADED LATER! 
#upload from raw text files and create merged dataset with year indicators 
cdc_data_folder = "/Users/allegrasaggese/Dropbox/Mental/Data/raw/cdc"
paths = sorted(glob.glob(os.path.join(cdc_data_folder , "*.txt")))
year_re = re.compile(r"(\d{4})(?=\.txt$)")

dfs = []
for p in paths:
    # extract 4-digit year from end of filename
    m = year_re.search(os.path.basename(p))
    if not m:
        raise ValueError(f"No YYYY at end of filename: {os.path.basename(p)}")
    yr = int(m.group(1))

    df = pd.read_csv(p, sep="\t", quotechar='"', engine="python")

    # add year column
    df["year"] = yr

    # --- unreliables -> flags + clean numeric text ---
    unreliable_count = 1
    for col in df.columns:
        if df[col].astype(str).str.contains(r"\(Unreliable\)", regex=True).any():
            new_col = f"Unreliable_{unreliable_count}"
            df[new_col] = df[col].astype(str).str.contains(r"\(Unreliable\)", regex=True).astype(int)
            unreliable_count += 1

            df[col] = (df[col].astype(str)
                               .str.replace(r"\s*\(Unreliable\)$", "", regex=True)
                               .pipe(pd.to_numeric, errors="coerce"))

    dfs.append(df)

#concat each YR dataframe together
combined_1 = pd.concat(dfs, ignore_index=True, sort=False)

# lower case / clean col names 
combined_1.columns = (combined_1.columns
                .str.lower()
                .str.strip()        # remove leading/trailing spaces
                .str.replace(" ", "_")  # replace spaces with underscores
                .str.replace(r"[^\w_]", "", regex=True))  # drop punctuation

# drop the notes col (not needed)
combined_1 = combined_1.drop(columns=["notes"])

# split the county / state out of the county col 
combined_1["state"] = combined_1["county"].str[-3:].str.replace(",", "", regex=False).str.strip()
combined_1["county"] = combined_1["county"].str[:-4].str.strip()

# add padding to fips (4 digits ----> 5 digits)
combined_1["county_code"] = (
    combined_1["county_code"]
    .apply(lambda x: str(int(x)) if pd.notna(x) and re.match(r"^\d+(\.0+)?$", str(x)) else str(x))
    .str.strip()
)

#count the number of entries < 5 digits
num_padded = (combined_1["county_code"].str.len() == 4).sum()

# pad to 5 digits
combined_1["county_code"] = combined_1["county_code"].str.zfill(5)
print(f"Padded {num_padded} county_code entries to 5 digits.") # check number of padding

# our FIPS key has different codes depending on the year (potential for change)
## want to QA this data to see if it records FIP code changes or backfills 
## IF THE FIPS CODES ARE THE SAME ----> WE SHOULD REPLACE WITH THE annual/fips code key
code_counts = (
    combined_1.groupby(["county", "state"])["county_code"]
            .nunique()
            .reset_index(name="n_unique_codes")
) # only 448 observations of FIPS here - quite low for the data, only pulling five states (NC, IA, CA, TX, GA)
print(f"Unique states in mortality data:      {combined_1['state'].nunique()}")
unique_states = sorted(combined_1["state"].dropna().unique().tolist())
print(unique_states)


#flag groups where the code changes over time (n_unique_codes > 1)
changing = code_counts[code_counts["n_unique_codes"] > 1]

# 3) show which codes each (county, state) has (for the changing ones)
codes_per_group = (
    combined_1.groupby(["county", "state"])["county_code"]
            .apply(lambda s: sorted(s.dropna().unique().tolist()))
            .reset_index(name="codes")
)
changing_details = changing.merge(codes_per_group, on=["county","state"])

print(changing_details.sort_values(["state","county"])) # no code changes - will need to merge on names, not fips 

# checking the percentage of deaths column (I think I should make it from scratch)
print(combined_1.columns.tolist()) # looking at all col names 

# to fix the percentage col (object --> num) 
combined_1["_of_total_deaths"].dtype
combined_1["_of_total_deaths"].describe()

combined_1["_of_total_deaths"] = (
    combined_1["_of_total_deaths"]
    .astype(str)                                      # everything to string
    .replace(["", "None", "none", "NULL", "null", "NaN", "nan", "<NA>"], np.nan)  # unify nulls
    .str.replace("%", "", regex=False)                # drop %
    .str.strip()                                      # clean spaces
)

# now convert only valid numeric strings
combined_1["_of_total_deaths"] = pd.to_numeric(combined_1["_of_total_deaths"], errors="coerce")
combined_1["_of_total_deaths"] = combined_1["_of_total_deaths"] / 100
# the percentages don't make a lot of sense, I think there was a read in error - going to delete and calculate myself 
combined_1 = combined_1.drop(columns=["_of_total_deaths"])

combined_1["percent_sex_race_deaths"] = (
    combined_1["deaths"] / combined_1["population"]
)
combined_1["percent_sex_race_deaths"].describe() # matches and is more decimal places 

# eliminate age b/c we never use it 
keys = ["year", "county", "state", "race", "gender", "icd_chapter", "icd_chapter_code"]

# collapse over age: sum deaths & population
collapsed = (
    combined_1
      .groupby(keys, as_index=False)
      .agg(deaths=("deaths","sum"),
           population=("population","sum"))
)

# crude rate per 100k (CDC-style)
collapsed["crude_rate_per_100k"] = (collapsed["deaths"] / collapsed["population"]) * 100_000

# % of total deaths in county-year (share by race/gender/ICD)
totals = collapsed.groupby(["year","county","state"])["deaths"].transform("sum")
collapsed["pct_of_total_deaths"] = (collapsed["deaths"] / totals) * 100

# export the deaths data 
clean_mortality_data = f"{today_str}_mortality_sex_race_disagg.csv"
mpath = os.path.join(outf, clean_mortality_data)
combined_1.to_csv(mpath, index=False)

# check number of unique FIPS before the merge (only 400 observations relative to 3200 in total in the US)
print(f"Unique FIPS in combined mortality data before collapse:      {combined_1['county_code'].nunique()}")

### merge with the mental health data 

# COLLAPSE MORTALITY DATA over age, race, gender first (so one obs per icd_chapter–county–year)
by_chapter = (
    combined_1
    .groupby(["year","state","county","icd_chapter"], as_index=False)
    .agg(deaths=("deaths","sum"),
         population=("population","sum"))
)

#totals per county–year (for denominators)
totals = (
    by_chapter.groupby(["year","state","county"], as_index=False)
              .agg(total_deaths=("deaths","sum"),
                   population=("population","sum"))
)

by_chapter = by_chapter.merge(totals, on=["year","state","county"], how="left")

# compute each ICD’s crude rate & % of total
by_chapter["icd_crude_rate_per_100k"] = (
    by_chapter["deaths"] / by_chapter["population_x"] * 100_000
)
by_chapter["icd_pct_of_total_deaths"] = (
    by_chapter["deaths"] / by_chapter["total_deaths"] * 100
)

#pivot wide to get a col for each type of ISD category of death 
wide = (
    by_chapter
    .pivot_table(
        index=["year","state","county"],
        columns="icd_chapter",
        values=["deaths","icd_crude_rate_per_100k","icd_pct_of_total_deaths"],
        aggfunc="first"
    )
)

# flatten 
wide.columns = [
    f"{icd}_{metric}"
    for metric, icd in wide.columns
]
wide = wide.reset_index()
wide = wide.merge(totals, on=["year","state","county"], how="left")

filled_summary = (1 - wide.isna().sum() / len(wide)) * 100
print(filled_summary)


# LOAD IN FIPS 
fips_clean = os.path.join(outf, "2025-08-11_fips_full.csv")
print("File exists:", os.path.exists(fips_clean))
fips_key = pd.read_csv(fips_clean, low_memory=False)

# upload MH data, review fips 
mh_data = f"{today_str}_mentalhealthrank_full.csv"
mh_path = os.path.join(outf, mh_data)
mh_df = pd.read_csv(mh_path, low_memory=False)

# create fips identifier in the mortality data (from a year-county-state match)
cols_to_add = ["fips", "state_code", "county_code", "year", "county", "state"]
fips_sub = fips_key[cols_to_add] # subset for ease 

# check fips data first 
print("fips_sub dup keys:", fips_sub.duplicated(["year","county","state"]).any()) # dupes present 

keys = ["year","county","state"]

# normalize join fields (trim/case)
for d in (wide, fips_key):
    d["county"] = d["county"].astype(str).str.strip()
    d["state"]  = d["state"].astype(str).str.strip().str.upper()
    
# inspect conflicts - strange finding from bad fips merge 
conflicts = (
    fips_key.groupby(keys)
    .agg(n_rows=("fips","size"),
         n_fips=("fips","nunique"),
         fips_set=("fips", lambda s: sorted(pd.Series(s).dropna().unique().tolist())))
    .query("n_rows > 1")
)
print("dup groups:", len(conflicts), "with >1 FIPS:", (conflicts["n_fips"]>1).sum()) # there are duplicates in the FIPS data - will have to adjust later 


# quick fix fips key 
# identify (year, county, state) groups with n_rows > 1 and n_fips == 1
dup_keys = (
    fips_sub.groupby(["year", "county", "state"])
    .agg(n_rows=("fips", "size"), n_fips=("fips", "nunique"))
    .query("n_rows > 1 and n_fips == 1")
    .reset_index()[["year", "county", "state"]]
)

# drop those redundant rows from fips_sub
before = len(fips_sub)
fips_sub = fips_sub.merge(dup_keys, on=["year", "county", "state"], how="left", indicator=True)
fips_sub = fips_sub[fips_sub["_merge"] == "left_only"].drop(columns="_merge")

after = len(fips_sub)
print(f"Dropped {before - after} redundant duplicate rows (same FIPS).")

# DC has duplicated FIPS for both years 
fips_sub = fips_sub.drop_duplicates(["year", "county", "state"], keep="first")

print("Deduplicated fips_sub — kept first instance for Washington DC (and any others).")
print(f"Remaining rows: {len(fips_sub)}") # good only dropped the 20 obs that were duplicated 


# merge all health data together
# should not have to do this - there is something strange with the MH data, lets hope its a column header issue and not an imerge issue - I will look into it 
mh_df = mh_df[mh_df["5-digit_fips_code"] != "fips_code"] #### TO ADDRESS THIS 


# merge fips and mortality data 
wide_fips = wide.merge(
    fips_sub,
    on=["year", "county", "state"],
    how="left"     # keeps all rows from wide, adds matches from fips_sub
)
print("Shape after merge:", wide_fips.shape)

# merge mental health with mortality data on the fips key now 
for d in (wide_fips, mh_df):
    d["fips"] = d["fips"].astype(str).str.zfill(5)
    d["year"] = d["year"].astype(int)


if "5-digit_fips_code" in mh_df.columns and "fips" not in mh_df.columns:
    mh_df = mh_df.rename(columns={"5-digit_fips_code": "fips"})

def normalize_fips(col: pd.Series):
    s = (col.astype("string")                # unified string dtype
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)  # 6001.0 -> 6001
            .str.replace(r"\D", "", regex=True))   # drop non-digits
    num_padded = (s.str.len() == 4).sum()
    s = s.str.zfill(5)
    return s, int(num_padded)

# apply to both DFs
wide_fips["fips"], padded_wide = normalize_fips(wide_fips["fips"])
mh_df["fips"], padded_mh = normalize_fips(mh_df["fips"])

print(f"Padded in wide_fips: {padded_wide}")
print(f"Padded in mh_df:     {padded_mh}")

# check year type 
wide_fips["year"] = pd.to_numeric(wide_fips["year"], errors="coerce").astype("Int64")
mh_df["year"]     = pd.to_numeric(mh_df["year"], errors="coerce").astype("Int64")

# check the range for the years (theres only 6 years of overlap)
print(mh_df["year"].min(), mh_df["year"].max())
print(wide_fips["year"].min(), wide_fips["year"].max())

# MERGE 
mh_df["fips"] = mh_df["fips"].astype(str).str.zfill(5)
wide_fips["fips"] = wide_fips["fips"].astype(str).str.zfill(5)
mh_df["year"] = mh_df["year"].astype(int)
wide_fips["year"] = wide_fips["year"].astype(int)

# checking again the number of fips to see what % is likely going to spillout in the merge 

# remove the non-county observations 
bad_states = ["US", "STATE", "NAN"]
mh_df[mh_df["state_abbreviation"].isin(bad_states)][["state_abbreviation"]].value_counts()
mh_df = mh_df[~mh_df["state_abbreviation"].isin(bad_states)]

"fips" in mh_df

print(f"Unique FIPS in mh_df:      {mh_df['fips'].nunique()}")
print(f"Unique FIPS in wide_fips:  {wide_fips['fips'].nunique()}")

# full merge
wide_mh_full = pd.merge(
    wide_fips,
    mh_df,
    on=["fips", "year"],
    how="outer",
    sort=True,
    validate="m:m"
)
print(f"Rows: {len(wide_mh_full)}  |  Columns: {len(wide_mh_full.columns)}")


# export clean, complete (UNBALANCED) health data frame 
today_str = date.today().strftime("%Y-%m-%d")
 
fullmh = f"{today_str}_mh_mortality_fips_yr.csv"
path_3 = os.path.join(outf, fullmh)
wide_mh_full.to_csv(path_3, index=False)







