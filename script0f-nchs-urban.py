#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:55:22 2026

@author: allegrasaggese
"""

from packages import *
from functions import *


# folders
inf = os.path.join(db_data, "raw", "nchs")
outf = os.path.join(db_data, "clean")

# input file
csv_path = os.path.join(inf, "NCHSurb-rural-codes.csv")

# read (latin1 encoding needed)
df = pd.read_csv(csv_path, encoding="latin1")

# clean columns
df.columns = df.columns.str.lower().str.strip()
df = df.rename(columns={
    "stfips": "state_fips_code",
    "ctyfips": "county_code",
    "st_abbrev": "state_abbrev",
    "ctyname": "county_name",
    "cbsatitle": "cbsa_title",
    "cbsapop": "cbsa_pop",
    "ctypop": "county_pop",
    "code2023": "nchs_code_2023",
    "code2013": "nchs_code_2013",
    "code2006": "nchs_code_2006",
    "code1990": "nchs_code_1990",
})

# generate 5-digit FIPS (functions.generate_fips creates column "FIPS_generated")
df = generate_fips(df, state_col="state_fips_code", city_col="county_code")
if "FIPS_generated" in df.columns and "fips_generated" not in df.columns:
    df = df.rename(columns={"FIPS_generated": "fips_generated"})

# numeric columns
num_cols = ["cbsa_pop", "county_pop", "nchs_code_2023", "nchs_code_2013", "nchs_code_2006", "nchs_code_1990"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

# build long panel
code_cols = ["nchs_code_2023", "nchs_code_2013", "nchs_code_2006", "nchs_code_1990"]
year_map = {
    "nchs_code_2023": 2023,
    "nchs_code_2013": 2013,
    "nchs_code_2006": 2006,
    "nchs_code_1990": 1990,
}

panel = df.melt(
    id_vars=[
        "fips_generated",
        "state_fips_code",
        "county_code",
        "state_abbrev",
        "county_name",
        "cbsa_title",
        "cbsa_pop",
        "county_pop",
    ],
    value_vars=code_cols,
    var_name="nchs_version",
    value_name="nchs_code",
)

panel["year"] = panel["nchs_version"].map(year_map).astype("Int64")
panel["nchs_code"] = pd.to_numeric(panel["nchs_code"], errors="coerce").astype("Int64")

# NCHS labels (standard 6-level scheme)
label_map = {
    1: "Large central metro",
    2: "Large fringe metro",
    3: "Medium metro",
    4: "Small metro",
    5: "Micropolitan",
    6: "Noncore",
}
panel["nchs_label"] = panel["nchs_code"].map(label_map).astype("string")

# binary flag: 0 for large central/fringe metro, 1 otherwise
panel["non_large_metro"] = 1
panel.loc[panel["nchs_code"].isin([1, 2]), "non_large_metro"] = 0

# drop rows without codes
panel = panel.dropna(subset=["nchs_code"]).copy()
panel = panel.sort_values(["fips_generated", "year"])

# expand to discrete annual panel using most recent base year
def expand_years(df, base_year, start_year, end_year):
    base = df[df["year"] == base_year].copy()
    frames = []
    for y in range(start_year, end_year + 1):
        frames.append(base.assign(year=y))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=df.columns)

panel_expanded = pd.concat(
    [
        expand_years(panel, 1990, 1990, 2005),
        expand_years(panel, 2006, 2006, 2012),
        expand_years(panel, 2013, 2013, 2022),
        expand_years(panel, 2023, 2023, 2023),
    ],
    ignore_index=True,
)

# drop years before 2000
panel_expanded = panel_expanded[panel_expanded["year"] >= 2000].copy()
panel_expanded = panel_expanded.sort_values(["fips_generated", "year"])

# export
today_str = date.today().strftime("%Y-%m-%d")
panel_path = os.path.join(outf, f"{today_str}-rural-key.csv")

panel_expanded.to_csv(panel_path, index=False)

print("Saved:", panel_path)
