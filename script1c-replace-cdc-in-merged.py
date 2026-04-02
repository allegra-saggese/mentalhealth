#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace old mh_mortality descriptor block in merged panel with
cdc_county_year_deathsofdespair descriptor block.
"""

import os
import glob
import pandas as pd
from datetime import date


MERGED_DIR = "/Users/allegrasaggese/Dropbox/Mental/Data/merged"
CLEAN_DIR = "/Users/allegrasaggese/Dropbox/Mental/Data/clean"
CDC_SUFFIX = "_cdc_county_year_deathsofdespair"
OLD_SUFFIX = "_mh_mortality_fips_yr"


def latest_file(folder, pattern):
    hits = glob.glob(os.path.join(folder, pattern))
    if not hits:
        raise FileNotFoundError(f"No files for pattern {pattern} in {folder}")
    return max(hits, key=os.path.getmtime)


def norm_key(df):
    if "fips" not in df.columns and "fips_generated" in df.columns:
        df = df.rename(columns={"fips_generated": "fips"})
    if "year" not in df.columns and "yr" in df.columns:
        df = df.rename(columns={"yr": "year"})
    if "fips" not in df.columns or "year" not in df.columns:
        raise KeyError("Missing fips/year")
    df["fips"] = (
        df["fips"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df.dropna(subset=["fips", "year"]).copy()


merged_path = latest_file(MERGED_DIR, "*_full_merged.csv")
cdc_path = latest_file(CLEAN_DIR, "*_cdc_county_year_deathsofdespair.csv")

print("Using merged:", os.path.basename(merged_path))
print("Using CDC panel:", os.path.basename(cdc_path))

m = pd.read_csv(merged_path, low_memory=False)
m = norm_key(m)

old_cols = [c for c in m.columns if c.endswith(OLD_SUFFIX)]
if old_cols:
    m = m.drop(columns=old_cols)
print("Dropped old mh_mortality columns:", len(old_cols))

existing_new = [c for c in m.columns if c.endswith(CDC_SUFFIX)]
if existing_new:
    m = m.drop(columns=existing_new)
print("Dropped pre-existing CDC replacement columns:", len(existing_new))

c = pd.read_csv(cdc_path, low_memory=False)
c = norm_key(c)

if c.duplicated(["fips", "year"]).any():
    num_cols = [x for x in c.columns if x not in ("fips", "year") and pd.api.types.is_numeric_dtype(c[x])]
    txt_cols = [x for x in c.columns if x not in ("fips", "year") and x not in num_cols]
    agg = {x: (lambda s: s.sum(min_count=1)) for x in num_cols}
    agg.update({x: "first" for x in txt_cols})
    c = c.groupby(["fips", "year"], as_index=False).agg(agg)

c["n_rows"] = 1
non_key = [x for x in c.columns if x not in ("fips", "year")]
c = c.rename(columns={x: f"{x}{CDC_SUFFIX}" for x in non_key})

rows_before = len(m)
m = m.merge(c, on=["fips", "year"], how="left")
rows_after = len(m)

if rows_before != rows_after:
    raise RuntimeError(f"Row count changed unexpectedly: {rows_before} -> {rows_after}")

m = m.sort_values(["fips", "year"]).reset_index(drop=True)

today = date.today().strftime("%Y-%m-%d")
out_main = os.path.join(MERGED_DIR, f"{today}_full_merged.csv")
out_05010 = os.path.join(MERGED_DIR, f"{today}_full_merged_2005_2010.csv")
out_1020 = os.path.join(MERGED_DIR, f"{today}_full_merged_2010_2020.csv")
out_census = os.path.join(MERGED_DIR, f"{today}_full_merged_census_years.csv")

m.to_csv(out_main, index=False)
m[m["year"].between(2005, 2010, inclusive="both")].to_csv(out_05010, index=False)
m[m["year"].between(2010, 2020, inclusive="both")].to_csv(out_1020, index=False)
m[m["year"].isin([2002, 2005, 2007, 2012])].to_csv(out_census, index=False)

new_cols = [c for c in m.columns if c.endswith(CDC_SUFFIX)]
left_old = [c for c in m.columns if c.endswith(OLD_SUFFIX)]

print("Rows:", len(m))
print("New CDC columns:", len(new_cols))
print("Old mh_mortality columns remaining:", len(left_old))
print("Saved:", out_main)
print("Saved:", out_05010)
print("Saved:", out_1020)
print("Saved:", out_census)
