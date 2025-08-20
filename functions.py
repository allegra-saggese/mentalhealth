#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:19:14 2025

@author: allegrasaggese
"""

# purpose: useful functions across scripts 

# load packages - think i can delete
import os
import pandas as pd



# data cleaning - checking missing percentages 
def percent_missing_vs_filled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with, for each column in `df`:
      - missing_pct:    percentage of NA values
      - filled_pct:     percentage of non-NA (numeric) values
    """
    # total rows
    total = len(df)

    # count missing per column
    missing_count = df.isna().sum()

    # compute percentages
    missing_pct = missing_count / total * 100
    filled_pct = 100 - missing_pct

    # assemble summary
    summary = pd.DataFrame({
        'missing_pct': missing_pct,
        'filled_pct':  filled_pct
    })

    return summary



# FIPS code generator from individual city - state cols 

def generate_fips(df, state_col="state", city_col="city"):
    # Pad state to 2 digits _ cities to 3 digits
    df["state_padded"] = df[state_col].astype(str).str.zfill(2)
    df["city_padded"] = df[city_col].astype(str).str.zfill(3)

    # combine cols
    df["FIPS_generated"] = df["state_padded"] + df["city_padded"]

    # QA - Check all are 5 digits
    invalid_fips = df[~df["FIPS_generated"].str.match(r"^\d{5}$")]
    if not invalid_fips.empty:
        print("Warning: Some FIPS codes are not 5 digits.")
        print(invalid_fips[["FIPS_generated", state_col, city_col]])
    else:
        print("All FIPS_generated values are 5 digits.")

    # drop temp columns
    df.drop(columns=["state_padded", "city_padded"], inplace=True)

    return df



# Appending lists of column names for sorting

def append_cols(existing_list, new_cols):
    """Append new columns to existing list, avoiding duplicates."""
    for col in new_cols:
        if col not in existing_list:
            existing_list.append(col)
    return existing_list


# Sorting lists of dataframes 

def keep_only_cols(dfs, cols_to_keep):
    """Return new list of DataFrames with only specified columns (keep order)."""
    return [df[[col for col in cols_to_keep if col in df.columns]] for df in dfs]


# simple / standard colname cleaning

def clean_cols(df):
    df.columns = (
        df.columns
        .str.lower()                      # lowercase
        .str.replace("&", "and")          # replace & with and
        .str.replace(r"[.,]", "", regex=True)  # remove . and ,
        .str.strip()                      # trim spaces
    )
    return df










