#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:31:27 2025

@author: allegrasaggese
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 18:55:21 2025

@author: allegrasaggese
"""

# purpose - data review (primary, will not be used in modules)

# load packages
import os
import pandas as pd
from datetime import datetime


# load functions
from functions import * 

# set directories
db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
db_me = os.path.join(db_base, "allegra-dropbox-copy")
interim = os.path.join(db_me, "interim-data") # file for interim datasets or lists used across scripts 


# import data
agg_file_path = os.path.join(db_me, "25-07-01-dta-copy.dta")
df = pd.read_stata(agg_file_path, convert_categoricals=False)

# sort data / review bf iteration 
df_sorted = df.sort_values(by=["STATE", "FIPS", "SURVEY_YEAR"])
df_sorted.head(20)

# save test head for manual review 
test_df_for_iteration = df_sorted.head(500)
save_path = os.path.join(interim, "test_df_for_iteration.csv")
test_df_for_iteration.to_csv(save_path, index=False)

print("File exists:", os.path.exists(save_path)) # check it worked 



## ITERATION

# start by dropping sub-group rows
df_totals = df[df["aggregation_type"] != "DEMOGRAPHICS"].copy()


# create new years
new_years = [2003, 2004, 2005, 2006, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
fips_list = df_totals["FIPS"].dropna().unique()

new_rows = pd.DataFrame(
    [(fips, year) for fips in fips_list for year in new_years],
    columns=["FIPS", "year"]
)

# grab CAIFO cols 
cols = df_totals.columns.tolist()
start_idx = cols.index("all_animal_OP")
end_idx = cols.index("cattle_share_med_CAFO")

# CAIFO cols 
cols_to_copy = cols[start_idx:end_idx + 1]


# check data shape 
print(df_totals["year"].unique())
print(df_totals['FIPS'].nunique()) # 3149

print(df_totals[df_totals["year"] == 2002].shape) # 3111
print(df_totals[df_totals["year"] == 2007].shape) # 3143 missing 6 
print(df_totals[df_totals["year"] == 2012].shape) # 3143 missing 6 

# check types for the merge --- types remain the same (no issue in merge)
print(df_totals.dtypes["FIPS"])
print(new_rows.dtypes["FIPS"])


# Subset for base years
base_2002 = df_totals[df_totals["year"] == 2002][["FIPS"] + cols_to_copy].copy()
base_2007 = df_totals[df_totals["year"] == 2007][["FIPS"] + cols_to_copy].copy()
base_2012 = df_totals[df_totals["year"] == 2012][["FIPS"] + cols_to_copy].copy()


# Filter new rows
new_rows_03_06 = new_rows[new_rows["year"].isin([2003, 2004, 2005, 2006])].copy()
new_rows_08_11 = new_rows[new_rows["year"].isin([2008, 2009, 2010, 2011])].copy()
new_rows_13_16 = new_rows[new_rows["year"].isin([2013, 2014, 2015, 2016])].copy()


# checking FIPS match rate across each match across dataframes
filled_03_06 = new_rows_03_06.merge(
    base_2002, on="FIPS", how="left", indicator=True
)

filled_08_11 = new_rows_08_11.merge(
    base_2007, on="FIPS", how="left", indicator=True
)

filled_13_16 = new_rows_13_16.merge(
    base_2012, on="FIPS", how="left", indicator=True
)

# checking that the merge went through 
print(filled_03_06["_merge"].value_counts())
print(filled_08_11["_merge"].value_counts())
print(filled_13_16["_merge"].value_counts())

print(filled_13_16.head(50))

# concat 
df_full = pd.concat(
    [df, filled_03_06, filled_08_11, filled_13_16],
    ignore_index=True,
    sort=False
)


# save new full iterated df 
today = datetime.today().strftime('%Y%m%d')
filename = f"{today}_annual_CAIFO_df.csv"
df_full.to_csv(os.path.join(db_data, filename), index=False)


#### SCATTERPLOTS 





