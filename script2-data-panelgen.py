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
df_totals = df_sorted[df_sorted["aggregation_type"] == "TOTALS"].copy()

# create new years
new_years = [2003, 2004, 2005, 2006, 2013, 2014, 2015, 2016]
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


# check 2002 data
print(df_totals["year"].unique())
print(df_totals[df_totals["year"] == 2002].shape)
print(df_totals[df_totals["year"] == 2012].shape)

# check types for the merge
print(df_totals.dtypes["FIPS"])
print(new_rows.dtypes["FIPS"])


# Subset for base years
base_2002 = df_totals[df_totals["year"] == 2002][["FIPS"] + cols_to_copy].copy()
base_2012 = df_totals[df_totals["year"] == 2012][["FIPS"] + cols_to_copy].copy()

# Filter new rows
new_rows_03_06 = new_rows[new_rows["year"].isin([2003, 2004, 2005, 2006])].copy()
new_rows_13_16 = new_rows[new_rows["year"].isin([2013, 2014, 2015, 2016])].copy()

# Merge FAILED 
filled_03_06 = new_rows_03_06.merge(base_2002, on="FIPS", how="left")
filled_13_16 = new_rows_13_16.merge(base_2012, on="FIPS", how="left")

print(base_2002.head(30))


# checking that the FIPS codes are available 
filled_03_06 = new_rows_03_06.merge(
    base_2002, on="FIPS", how="left", indicator=True
)

# MERGE FAIL CHECK HERE 
print(filled_03_06["_merge"].value_counts())



