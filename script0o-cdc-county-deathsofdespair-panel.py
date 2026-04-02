#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build county-year deaths-of-despair panel from CDC WONDER annual county exports.

Expected raw files:
  Data/raw/cdc/cty-level-deathsofdespair-YYYY.csv

Output:
  Data/clean/YYYY-MM-DD_cdc_county_year_deathsofdespair.csv

QA outputs:
  allegra-dropbox-copy/interim-data/qa-health/YYYY-MM-DD_qa_cdc_deathsofdespair_file_inventory.csv
  allegra-dropbox-copy/interim-data/qa-health/YYYY-MM-DD_qa_cdc_deathsofdespair_by_year.csv
  allegra-dropbox-copy/interim-data/qa-health/YYYY-MM-DD_qa_cdc_deathsofdespair_key_check.csv
"""

import re

from packages import *
from functions import *


raw_dir = os.path.join(db_data, "raw", "cdc")
clean_dir = os.path.join(db_data, "clean")
qa_dir = os.path.join(db_me, "interim-data", "qa-health")
os.makedirs(clean_dir, exist_ok=True)
os.makedirs(qa_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")


def _extract_year_from_filename(path):
    base = os.path.basename(path)
    m = re.search(r"(19|20)\d{2}(?=\.csv$)", base)
    if not m:
        raise ValueError(f"Could not infer year from filename: {base}")
    return int(m.group(0))


def _to_num(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    return pd.to_numeric(series.astype("string").str.replace(",", "", regex=False).str.strip(), errors="coerce")


def _first_notna(s):
    idx = s.first_valid_index()
    return s.loc[idx] if idx is not None else pd.NA


files = sorted(glob.glob(os.path.join(raw_dir, "cty-level-deathsofdespair-*.csv")))
if not files:
    raise FileNotFoundError(f"No cty-level deaths-of-despair files found in {raw_dir}")


parts = []
inventory_rows = []

for path in files:
    src_year = _extract_year_from_filename(path)
    d = pd.read_csv(path, low_memory=False)
    d = clean_cols(d.copy())
    d.columns = d.columns.str.replace(r"\s+", "_", regex=True).str.strip("_")

    required = {"county", "county_code", "deaths", "population", "crude_rate"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise KeyError(f"{os.path.basename(path)} missing required columns: {missing}")

    # Keep only tabular rows (drop metadata/footer rows with blank county_code).
    d["county_code_num"] = _to_num(d["county_code"])
    d = d[d["county_code_num"].notna()].copy()
    d["fips"] = d["county_code_num"].round().astype("Int64").astype("string").str.zfill(5)
    d["year"] = src_year

    d["county"] = d["county"].astype("string").str.strip()
    d["state_abbrev"] = d["county"].str.extract(r",\s*([A-Z]{2})$", expand=False)
    d["county_name"] = d["county"].str.replace(r",\s*[A-Z]{2}$", "", regex=True).str.strip()

    d["deaths"] = _to_num(d["deaths"]).astype("Int64")
    d["population"] = _to_num(d["population"]).astype("Int64")
    d["crude_rate_raw"] = d["crude_rate"].astype("string").str.strip()
    d["crude_rate"] = _to_num(d["crude_rate_raw"].str.replace(r"\(.*?\)", "", regex=True))
    d["is_unreliable"] = d["crude_rate_raw"].str.contains("unreliable", case=False, na=False).astype("Int64")

    if "%_of_total_deaths" in d.columns:
        d["pct_of_total_deaths_raw"] = d["%_of_total_deaths"].astype("string").str.strip()
        d["pct_of_total_deaths"] = _to_num(d["pct_of_total_deaths_raw"].str.replace("%", "", regex=False))
    else:
        d["pct_of_total_deaths_raw"] = pd.NA
        d["pct_of_total_deaths"] = pd.NA

    keep = [
        "fips",
        "year",
        "state_abbrev",
        "county_name",
        "county",
        "deaths",
        "population",
        "crude_rate",
        "crude_rate_raw",
        "is_unreliable",
        "pct_of_total_deaths",
        "pct_of_total_deaths_raw",
    ]
    out = d[keep].copy()
    out["source_file"] = os.path.basename(path)
    parts.append(out)

    inventory_rows.append(
        {
            "file": os.path.basename(path),
            "year_from_filename": src_year,
            "rows_raw": int(len(pd.read_csv(path, low_memory=False))),
            "rows_kept_with_county_code": int(len(out)),
            "n_unique_fips_kept": int(out["fips"].nunique()),
            "n_missing_deaths": int(out["deaths"].isna().sum()),
            "n_missing_population": int(out["population"].isna().sum()),
        }
    )


panel = pd.concat(parts, ignore_index=True)

# Guard: should be one row per fips-year; if duplicates exist, collapse deterministically.
if panel.duplicated(["fips", "year"]).any():
    panel = (
        panel.groupby(["fips", "year"], as_index=False)
        .agg(
            state_abbrev=("state_abbrev", _first_notna),
            county_name=("county_name", _first_notna),
            county=("county", _first_notna),
            deaths=("deaths", "sum"),
            population=("population", "sum"),
            crude_rate=("crude_rate", _first_notna),
            crude_rate_raw=("crude_rate_raw", _first_notna),
            is_unreliable=("is_unreliable", "sum"),
            pct_of_total_deaths=("pct_of_total_deaths", _first_notna),
            pct_of_total_deaths_raw=("pct_of_total_deaths_raw", _first_notna),
            source_file=("source_file", _first_notna),
        )
    )

panel = panel.sort_values(["year", "fips"]).reset_index(drop=True)


# Save clean panel
out_clean = os.path.join(clean_dir, f"{today_str}_cdc_county_year_deathsofdespair.csv")
panel.to_csv(out_clean, index=False)


# QA outputs
inv = pd.DataFrame(inventory_rows).sort_values("year_from_filename")
inv_out = os.path.join(qa_dir, f"{today_str}_qa_cdc_deathsofdespair_file_inventory.csv")
inv.to_csv(inv_out, index=False)

by_year = (
    panel.groupby("year", as_index=False)
    .agg(
        n_rows=("fips", "size"),
        n_unique_fips=("fips", "nunique"),
        n_nonmissing_deaths=("deaths", lambda s: int(s.notna().sum())),
        n_nonmissing_population=("population", lambda s: int(s.notna().sum())),
        n_unreliable=("is_unreliable", lambda s: int((s > 0).sum())),
    )
    .sort_values("year")
)
by_year_out = os.path.join(qa_dir, f"{today_str}_qa_cdc_deathsofdespair_by_year.csv")
by_year.to_csv(by_year_out, index=False)

key_check = pd.DataFrame(
    [
        {
            "n_rows": int(len(panel)),
            "n_unique_fips_year": int(panel[["fips", "year"]].drop_duplicates().shape[0]),
            "n_duplicate_fips_year_rows": int(panel.duplicated(["fips", "year"]).sum()),
            "year_min": int(panel["year"].min()),
            "year_max": int(panel["year"].max()),
            "n_unique_years": int(panel["year"].nunique()),
            "n_unique_fips": int(panel["fips"].nunique()),
        }
    ]
)
key_out = os.path.join(qa_dir, f"{today_str}_qa_cdc_deathsofdespair_key_check.csv")
key_check.to_csv(key_out, index=False)

print("Saved clean panel:", out_clean)
print("Saved QA:", inv_out)
print("Saved QA:", by_year_out)
print("Saved QA:", key_out)
