#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 21:56:38 2025

@author: allegrasaggese
"""

# README FILE FOR CAFO INTERPRETATION

## OVERVIEW: 
    ### 1 read in raw agriculture census data (5-year intervals)
    ### 2 ensure census data is consistent across years (it should be given USDA notes on provision) for a rowbind into one dataframe
    ### 3 check all fips codes provided in the USDA census match the fips KEY (previously generated)
    ### 4 FIPS key = list of fips-year used for all other data that will be merged to ag data 
    ### 5 create values for missing years by taking previous census year and generating new rows 
    ### 6 create categories for agriculture feeding ops (CAFOs) based on census data, provided by animal type 
    ### 7 calculate number of these CAFOs (small, med, large) per census area 
    
    
    
## SOURCES:
    ### USDA Agriculture census - methodology, notes
    ### FIPS definition from US census
    ### concentrated agriculture feeding operation (CAFO) definition - taken from US EPA


# STEP 1 - DATA GENERATION + FORMATTING 

## load packages and workspaces

import sys, importlib.util
from collections import Counter
import re
from functools import reduce

### make sure repo root is on sys.path (parent of functions.py / packages/)
repo = “/Users/allegrasaggese/Documents/GitHub/mentalhealth”
if repo not in sys.path:
sys.path.append(repo)

import functions
import packages
from functions import *
from packages import *

### save name of other folders

inf = os.path.join(db_data, “raw”) # input
outf = os.path.join(db_data, “clean”) #output

## import USDA agriculture census data (every five years - 2002, 2007, 2012, 2017)

agfolder = os.path.join(inf, “usda”)
agfiles = glob.glob(os.path.join(agfolder, “*.dta”))
agdfs = [pd.read_stata(file) for file in agfiles]


### check all columns match across all the years and synthesize coltype 

base_cols = set(agdfs[0].columns)
all_match = True

for i, df in enumerate(agdfs, 1):
print(f”\n— Dataframe {i} —”)
print(df.dtypes) # fips code = int

### row bind the agriculture data given all columns are aligned
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

row-bind (union is identical to baseline since we reindexed)
combined = pd.concat(dfs_aligned, ignore_index=True)

### post-check: confirm columns unchanged

same_cols = list(combined.columns) == base_cols
combocols = list(combined.columns)
combocols == base_cols
len(base_cols) == len(list(combined.columns)) # TRUE

## read in FIPS code (county-state identifiers for each census unit)
fips_sense = os.path.join(outf, “2025-08-11_fips_full.csv”)
fips_df = pd.read_csv(fips_sense)

for i, df in enumerate(agdfs, 1):
print(f”\n— Dataframe {i} number of rows —”)
print(len(df))

### fips check by year of data

fdf_2002 = fips_df[fips_df[“year”] == 2002]
fdf_2007 = fips_df[fips_df[“year”] == 2007]
fdf_2012 = fips_df[fips_df[“year”] == 2012]
fdf_2017 = fips_df[fips_df[“year”] == 2017]

# STEP 2: TRANSFORM RAW AG DATA - (every five years) for iteration
## forward-fill data such that 2002 fills 2003-2006, and 2007 fills 2008-2011, and so on. 

base_years = [2002,2007,2012,2017]
n_forward = 4   # number of years to create after each base year
year_col = “year”

new_frames = []
for b in base_years:
base = ag_raw_df[ag_raw_df[year_col] == b].copy()
if base.empty:
continue
for y in range(b+1, b+1 + n_forward):
new_frames.append(base.assign(**{year_col: y}))

new_rows = pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame(columns=ag_raw_df.columns)
df_big = pd.concat([ag_raw_df, new_rows], ignore_index=True)


## export to save the interim, iterated over dataframe 
today_str = date.today().strftime(”%Y-%m-%d”)
clean_ag_census = f”{today_str}_ag_annual_df.csv”
ag_path = os.path.join(outf, clean_ag_census)
df_big.to_csv(ag_path, index=False)



# STEP 3 - CREATE CAFO COLUMN CLASSIFICATION
## list of cols that will be created

CAFO_cols = (“broiler_cafos_lrg_op”,
“broiler_cafos_med_op”,
“layer_cafos_lrg_op”,
“layer_cafos_med_op”,
“cattle_cafos_INV_lrg_op”,
“cattle_cafos_INV_med_op”,
“cattle_cafos_INV_lrg_head”,
“cattle_cafos_INV_med_head”,
“cattle_cafos_SALES_lrg_op”,
“cattle_cafos_SALES_med_op”,
“cattle_cafos_SALES_lrg_head”,
“cattle_cafos_SALES_med_head”,
“hog_cafos_INV_lrg_op”,
“hog_cafos_INV_med_op”,
“hog_cafos_INV_lrg_head”,
“hog_cafos_INV_med_head”,
“hog_cafos_SALES_lrg_op”,
“hog_cafos_SALES_med_op”,
“hog_cafos_SALES_lrg_head”,
“hog_cafos_SALES_med_head”
)

## CAFO size limits, and can adjust these values for the S/M/L CAFO development

broiler_cutoff_lrg = 5
broiler_cutoff_med =  3
layer_cutoff_lrg =  9
layer_cutoff_med =  7
cattle_cutoff_lrg =  7
cattle_cutoff_med = 6
hog_cutoff_lrg =  7
hog_cutoff_med = 1



## FUNCTION TO MAP EACH ROW TO A CAFO SIZE 

# INTUITION: Note that in the USDA data, they state: Domain Desc, Domaincat Desc, and Data Item create unique combinations of 
# observations. In our first analysis we use unit_desc and domaincat desc as the unique identifiers. We may want to ensure that 
# we include data_item (short_desc in our data) and also domain_desc in order to avoid grouping on multiple sets of data 

def map_size(df, mapping, unit_match, out_col):
    mask = df[‘unit_desc’] == unit_match
    df[out_col] = df[‘domaincat_desc’].map(mapping).where(mask, other=pd.NA).astype(“Int64”)


### put all mappings together first

layer_map = {
“inventory: (1 to 49 head)”:1,
“inventory: (50 to 99 head)”:2,
“inventory: (100 to 399 head)”:3,
“inventory: (400 to 3,199 head)”:4,
“inventory: (3,200 to 9,999 head)”:5,
“inventory: (10,000 to 19,999 head)”:6,
“inventory: (20,000 to 49,999 head)”:7,
“inventory: (50,000 to 99,999 head)”:8,
“inventory: (100,000 or more head)”:9
}

cattle_inv_map = {
“inventory of cattle, incl calves: (1 to 9 head)”:1,
“inventory of cattle, incl calves: (10 to 19 head)”:2,
“inventory of cattle, incl calves: (20 to 49 head)”:3,
“inventory of cattle, incl calves: (50 to 99 head)”:4,
“inventory of cattle, incl calves: (100 to 199 head)”:5,
“inventory of cattle, incl calves: (200 to 499 head)”:6,
“inventory of cattle, incl calves: (500 or more head)”:7
}

cattle_sales_map = {
“sales of cattle, incl calves: (1 to 9 head)”:1,
“sales of cattle, incl calves: (10 to 19 head)”:2,
“sales of cattle, incl calves: (20 to 49 head)”:3,
“sales of cattle, incl calves: (50 to 99 head)”:4,
“sales of cattle, incl calves: (100 to 199 head)”:5,
“sales of cattle, incl calves: (200 to 499 head)”:6,
“sales of cattle, incl calves: (500 or more head)”:7
}

hog_inv_map = {
“inventory of hogs: (1 to 24 head)”:1,
“inventory of hogs: (25 to 49 head)”:2,
“inventory of hogs: (50 to 99 head)”:3,
“inventory of hogs: (100 to 199 head)”:4,
“inventory of hogs: (200 to 499 head)”:5,
“inventory of hogs: (500 to 999 head)”:6,
“inventory of hogs: (1,000 or more head)”:7
}

<<<<<<< Updated upstream

#### NOTE AS OF DEC 2025 WE HAVE DROPPED ANY SALES CLASSIFICATION - WE CAN RECONSIDER THIS IN THE FUTURE 

=======
>>>>>>> Stashed changes
hog_sales_map = {
“sales of hogs: (1 to 24 head)”:1,
“sales of hogs: (25 to 49 head)”:2,
“sales of hogs: (50 to 99 head)”:3,
“sales of hogs: (100 to 199 head)”:4,
“sales of hogs: (200 to 499 head)”:5,
“sales of hogs: (500 to 999 head)”:6,
“sales of hogs: (1,000 or more head)”:7
}

### broiler/sales maps (same as earlier broiler mapping)

broiler_map = {
“sales: (1 to 1,999 head)”:1,
“sales: (2,000 to 59,999 head)”:2,
“sales: (60,000 to 99,999 head)”:3,
“sales: (100,000 to 199,999 head)”:4,
“sales: (200,000 to 499,999 head)”:5,
“sales: (500,000 or more head)”:6
}

### apply mappings

map_size(df, layer_map, unit_match=“operations”, out_col=“layer_ops_size”)
map_size(df, cattle_inv_map, unit_match=“operations”, out_col=“cattle_ops_size_inv”)
map_size(df, cattle_sales_map, unit_match=“operations”, out_col=“cattle_ops_size_sales”)
map_size(df, hog_inv_map, unit_match=“operations”, out_col=“hog_ops_size_inv”)
map_size(df, hog_sales_map, unit_match=“operations”, out_col=“hog_ops_size_sales”)

map_size(df, broiler_map, unit_match=“head”, out_col=“broiler_head_size”)
map_size(df, broiler_map, unit_match=“operations”, out_col=“broiler_ops_size”)
map_size(df, layer_map, unit_match=“head”, out_col=“layer_head_size”)
map_size(df, cattle_inv_map, unit_match=“head”, out_col=“cattle_head_size_inv”)
map_size(df, cattle_sales_map, unit_match=“head”, out_col=“cattle_head_size_sales”)
map_size(df, hog_inv_map, unit_match=“head”, out_col=“hog_head_size_inv”)
map_size(df, hog_sales_map, unit_match=“head”, out_col=“hog_head_size_sales”)



## set value to numeric

df[‘value’] = pd.to_numeric(df[‘value’].astype(str).str.replace(’,’, ‘’, regex=False), errors=‘coerce’)

## group for year

group_cols = [‘FIPS_generated’,‘year’]


# STEP 4 - CREATE FUNCTION TO SUM BY THE COUNT OF EACH CLASSIFICATION 
<<<<<<< Updated upstream


# to fill in to create (a) simple string classification and (b) sum total of inventory by classification and count of farms 


=======
def agg_mask(mask, prefix):
    sub = df.loc[mask, group_cols + [‘value’]].copy()
    if sub.empty:
        # return empty frame with correct column names so merges keep year
        return pd.DataFrame(columns=group_cols + [f”{prefix}_value_sum”, f”{prefix}_count”])
    g = (
        sub
        .groupby(group_cols, dropna=False)
        .agg(**{
            f”{prefix}_value_sum”: (‘value’, ‘sum’),
    f”{prefix}_count”:     (‘value’, ‘count’)   # non-missing value count
    })
    .reset_index()
    )
    return g

frames = []

### broiler

frames.append(agg_mask(df[‘broiler_ops_size’] >= broiler_cutoff_lrg, ‘broiler_lrg_op’))
frames.append(agg_mask((df[‘broiler_ops_size’] >= broiler_cutoff_med) & (df[‘broiler_ops_size’] < broiler_cutoff_lrg), ‘broiler_med_op’))

### layer (if present)

if ‘layer_ops_size’ in df.columns:
frames.append(agg_mask(df[‘layer_ops_size’] >= layer_cutoff_lrg, ‘layer_lrg_op’))
frames.append(agg_mask((df[‘layer_ops_size’] >= layer_cutoff_med) & (df[‘layer_ops_size’] < layer_cutoff_lrg), ‘layer_med_op’))

### cattle (inv & sales)

if ‘cattle_ops_size_inv’ in df.columns:
frames.append(agg_mask(df[‘cattle_ops_size_inv’] >= cattle_cutoff_lrg, ‘cattle_lrg_op_inv’))
frames.append(agg_mask((df[‘cattle_ops_size_inv’] >= cattle_cutoff_med) & (df[‘cattle_ops_size_inv’] < cattle_cutoff_lrg), ‘cattle_med_op_inv’))
if ‘cattle_ops_size_sales’ in df.columns:
frames.append(agg_mask(df[‘cattle_ops_size_sales’] >= cattle_cutoff_lrg, ‘cattle_lrg_op_sales’))
frames.append(agg_mask((df[‘cattle_ops_size_sales’] >= cattle_cutoff_med) & (df[‘cattle_ops_size_sales’] < cattle_cutoff_lrg), ‘cattle_med_op_sales’))

## hog (inv & sales)

if ‘hog_ops_size_inv’ in df.columns:
frames.append(agg_mask(df[‘hog_ops_size_inv’] >= hog_cutoff_lrg, ‘hog_lrg_op_inv’))
frames.append(agg_mask((df[‘hog_ops_size_inv’] >= hog_cutoff_med) & (df[‘hog_ops_size_inv’] < hog_cutoff_lrg), ‘hog_med_op_inv’))
if ‘hog_ops_size_sales’ in df.columns:
frames.append(agg_mask(df[‘hog_ops_size_sales’] >= hog_cutoff_lrg, ‘hog_lrg_op_sales’))
frames.append(agg_mask((df[‘hog_ops_size_sales’] >= hog_cutoff_med) & (df[‘hog_ops_size_sales’] < hog_cutoff_lrg), ‘hog_med_op_sales’))


# STEP 5 - FINAL MERGE: merge each aggregate onto all_fips_year

agg_by_fips_year = all_fips_year.copy()
>>>>>>> Stashed changes

### EXPORT 
clean_cafo = f”{today_str}_cafo_annual_df.csv”
ag_path2 = os.path.join(outf, clean_cafo)
agg_by_fips_year.to_csv(ag_path2, index=False)