#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 18:55:21 2025

@author: allegrasaggese
"""

# purpose - data review (primary, will not be used in modules)

# load packages, functions, databases 
from packages import *
from functions import * 


# import data
agg_file_path = os.path.join(db_me, "25-07-01-dta-copy.dta")
df = pd.read_stata(agg_file_path, convert_categoricals=False)

# review data 
print(df.columns.tolist())

# save colnames for excel
col_types_df = pd.DataFrame({
    'colname': df.columns,
    'type'   : df.dtypes.astype(str).values
})

# write to CSV (with header row)
path = os.path.join(interim, "column_types.csv")
col_types_df.to_csv(path, index=False)


# check percentage of data availabiltiy 
miss_pct_df = pd.DataFrame({
    "col": df.columns,
    "miss_pct": ((df.isna() | (df == "")).mean() * 100).round(2)
})

miss_pct_df = miss_pct_df.sort_values(by="miss_pct", ascending=False)
print(miss_pct_df)

# check for vars that contain no data 
cols_all_missing = miss_pct_df.loc[miss_pct_df["miss_pct"] == 100, "col"].tolist()
print(cols_all_missing)
len(cols_all_missing)

colpath = os.path.join(interim, "cols_all_missing.csv")
pd.Series(cols_all_missing, name="col").to_csv(colpath, index=False)


# FOR CODEBOOK - checking ranges 
df["fips"].max()
df["fips"].min()

df["SURVEY_YEAR"].max()
df["SURVEY_YEAR"].min()


# before merge - lets check completeness of CAIFO data 
# across current panel structure - its very low (17.5%), lets check collapsed 

df_totals = df[
    (df["aggregation_type"] != "DEMOGRAPHICS") &
    (df["year"]               != 1999)
].copy()

bob = percent_missing_vs_filled(df_totals) # now missing is about 21% 
# check for only relevant years -- 2007, 2012






