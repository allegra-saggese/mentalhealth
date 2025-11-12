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


# read in files 
def read_and_prepare(path):
    """Simple reader: csv/parquet/xlsx -> lower-case cols, strip strings, standardize fips/year."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, dtype=str)
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=str)
    else:
        raise ValueError("unsupported file type")

    # normalize column names and trim string columns
    df.columns = df.columns.str.lower().str.strip()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip()

    # rename common fips/year variants if present
    fips_candidates = [c for c in df.columns if c in ("fips","fips_generated","fips_code","geoid","county_fips")]
    year_candidates = [c for c in df.columns if c in ("year","yr")]

    if fips_candidates:
        df = df.rename(columns={fips_candidates[0]: "fips"})
        # normalize to zero-padded 5-char string where possible
        df["fips"] = df["fips"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(5)

    if year_candidates:
        df = df.rename(columns={year_candidates[0]: "year"})
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    return df



##### CAFO MAPPING FUNCTIONS ##### 

# wrapper for ambiguous descriptions 
def map_ambiguous(df, mapping, keywords, unit_label):
    """
    df: dataframe
    mapping: mapping dict (e.g., sales_map)
    keywords: list of substrings to look for in short_desc related to the type of animal 
    unit_label: suffix for the output column (e.g., 'sales_size')
    """
    for kw in keywords:
        mask = (
            df['domaincat_desc'].isin(mapping.keys()) &
            df['short_desc'].str.contains(kw, case=False, na=False)
        )
        out_col = f"{kw.lower().replace(',', '').replace(' ', '_')}_{unit_label}"
        df[out_col] = df.loc[mask, 'domaincat_desc'].map(mapping).astype('Int64')
        
keywords = ['poultry', 'chickens, layers', 'chickens, broilers', 'cattle', 'hogs']



# make a wrapper to map the inventory the same way in all of them 
def map_size(df, mapping, unit_match, out_col):
    mask = df['unit_desc'] == unit_match
    df[out_col] = df['domaincat_desc'].map(mapping).where(mask, other=pd.NA).astype("Int64")



# creating CAFO thresholds 
def assign_size(df, col, med_thresh, lrg_thresh):
    df = df.copy()
    # keep Int64Dtype and use nullable comparisons
    cond_large  = df[col].ge(lrg_thresh)
    cond_medium = df[col].ge(med_thresh) & df[col].lt(lrg_thresh)
    cond_small  = df[col].lt(med_thresh)

    df['size'] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[cond_large.fillna(False), 'size']  = 'large'
    df.loc[cond_medium.fillna(False), 'size'] = 'medium'
    df.loc[cond_small.fillna(False), 'size']  = 'small'

    df['size_col'] = col
    return df









