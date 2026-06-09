#!/usr/bin/env python3
"""
QA: Compare 2017 .dta data against 2017 NASS API pull.

For each county-commodity-class-domaincat combination that appears in BOTH sources,
check whether the value matches. Key question:
  - Does the API return (D) for rows where the .dta has a real numeric value?
  - Or does the .dta also have the same (D) / NaN for those rows?

If the .dta has numeric values where the API has (D), the .dta is undisclosed research
data with full counts. If both show (D) / NaN for the same rows, they are the same product.
"""
import os, sys, time
from datetime import date
import pandas as pd
import numpy as np

repo = os.path.dirname(os.path.abspath(__file__))
if repo not in sys.path:
    sys.path.append(repo)
import functions, packages
from packages import *

# ---- API setup (copied from script0a) ----
DEFAULT_NASS_API_KEY = "30643212-7739-359A-B451-0EAD3D345DB9"
NASS_API_KEY = os.environ.get("USDA_NASS_API_KEY", DEFAULT_NASS_API_KEY)
NASS_BASE = "https://quickstats.nass.usda.gov/api/"
RATE_SLEEP = 0.4
COMMODITIES = ["CATTLE", "CHICKENS", "HOGS"]

# ---- Load 2017 .dta ----
dta_path = os.path.join(db_data, "raw", "usda", "2017.dta")
print(f"Loading {dta_path} ...")
dta = pd.read_stata(dta_path)
dta.columns = [c.lower().strip() for c in dta.columns]
for c in ["commodity_desc", "class_desc", "unit_desc", "statisticcat_desc",
          "domain_desc", "domaincat_desc"]:
    if c in dta.columns:
        dta[c] = dta[c].astype(str).str.strip().str.lower()

dta_ops = dta[
    (dta["commodity_desc"].isin(["cattle", "chickens", "hogs"]))
    & (dta["unit_desc"] == "operations")
    & (dta["statisticcat_desc"] == "inventory")
    & (dta["domaincat_desc"].str.startswith("inventory", na=False))
].copy()
print(f"2017 .dta operations/inventory rows: {len(dta_ops):,}")
dta_d = dta_ops[dta_ops["value"].astype(str).str.strip().isin(["(D)", "(Z)", "(d)", "(z)"])]
print(f"2017 .dta rows with (D)/(Z) value: {len(dta_d):,}")
dta_ops["value_numeric_dta"] = pd.to_numeric(
    dta_ops["value"].astype(str).str.replace(",", "", regex=False).str.strip(),
    errors="coerce",
)
print(f"2017 .dta rows with numeric value: {dta_ops['value_numeric_dta'].notna().sum():,}")
print()

# ---- Pull 2017 from API ----
print("Pulling 2017 from NASS API (same request structure as 2022 pull)...")
frames = []
for cmd in COMMODITIES:
    for unit in ["OPERATIONS"]:
        for stat in ["INVENTORY"]:
            time.sleep(RATE_SLEEP)
            params = {
                "key": NASS_API_KEY,
                "source_desc": "CENSUS",
                "agg_level_desc": "COUNTY",
                "sector_desc": "ANIMALS & PRODUCTS",
                "commodity_desc": cmd,
                "unit_desc": unit,
                "statisticcat_desc": stat,
                "year": 2017,
            }
            try:
                count = functions.nass_get_counts(NASS_BASE, params)
            except RuntimeError as e:
                print(f"  Count error {cmd}: {e}"); continue
            if not count:
                print(f"  No rows: {cmd}"); continue
            print(f"  {cmd} {unit} {stat}: {count:,} rows")

            if count <= 50000:
                try:
                    df_pull = functions.nass_get_data(NASS_BASE, params)
                    frames.append(df_pull)
                except RuntimeError as e:
                    print(f"  Data error: {e}")
                continue

            # Split by domain if over 50k
            splits = {
                "CATTLE": ["INVENTORY OF CATTLE, INCL CALVES", "INVENTORY OF CATTLE, (EXCL COWS)",
                           "INVENTORY OF BEEF COWS", "INVENTORY OF MILK COWS",
                           "INVENTORY OF CATTLE ON FEED", "INVENTORY"],
                "CHICKENS": ["INVENTORY"],
                "HOGS": ["INVENTORY OF HOGS", "INVENTORY OF BREEDING HOGS"],
            }
            for dd in splits.get(cmd, []):
                time.sleep(RATE_SLEEP)
                p2 = dict(params, domain_desc=dd)
                try:
                    cnt2 = functions.nass_get_counts(NASS_BASE, p2)
                    if not cnt2 or cnt2 > 50000:
                        continue
                    df_pull = functions.nass_get_data(NASS_BASE, p2)
                    frames.append(df_pull)
                except RuntimeError as e:
                    print(f"    Split {dd} error: {e}")

if not frames:
    print("No API data pulled — exiting.")
    sys.exit(1)

api = pd.concat(frames, ignore_index=True)
api.columns = [c.lower().strip() for c in api.columns]
for c in ["commodity_desc", "class_desc", "unit_desc", "statisticcat_desc",
          "domain_desc", "domaincat_desc"]:
    if c in api.columns:
        api[c] = api[c].astype(str).str.strip().str.lower()

# Keep only inventory/operations rows (domaincat starts with "inventory")
api = api[api["domaincat_desc"].astype(str).str.strip().str.lower().str.startswith("inventory", na=False)].copy()
api = api.drop_duplicates()
print(f"\n2017 API operations/inventory rows (after dedup): {len(api):,}")

api_d = api[api["value"].astype(str).str.strip().isin(["(D)", "(Z)", "(d)", "(z)"])]
print(f"2017 API rows with (D)/(Z) value: {len(api_d):,}")
api["value_numeric_api"] = pd.to_numeric(
    api["value"].astype(str).str.replace(",", "", regex=False).str.strip(),
    errors="coerce",
)
print(f"2017 API rows with numeric value: {api['value_numeric_api'].notna().sum():,}")

# ---- Merge on join keys ----
join_keys = ["state_fips_code", "county_code", "commodity_desc", "class_desc", "domaincat_desc"]
# Normalise key columns
for df_, src in [(dta_ops, "dta"), (api, "api")]:
    df_["state_fips_code"] = df_["state_fips_code"].astype(str).str.strip().str.zfill(2)
    df_["county_code"]     = df_["county_code"].astype(str).str.strip().str.zfill(3)

merged = dta_ops[join_keys + ["value", "value_numeric_dta"]].merge(
    api[join_keys + ["value", "value_numeric_api"]],
    on=join_keys, how="outer", suffixes=("_dta", "_api"),
)
print(f"\nMerged rows: {len(merged):,}")
print("Merge status:")
dta_only = merged["value_numeric_dta"].notna() & merged["value_numeric_api"].isna() & merged["value_api"].isna()
api_only = merged["value_numeric_api"].notna() & merged["value_numeric_dta"].isna() & merged["value_dta"].isna()
both     = merged["value_dta"].notna() & merged["value_api"].notna()
print(f"  Both sources:  {both.sum():,}")
print(f"  .dta only:     {dta_only.sum():,}")
print(f"  API only:      {api_only.sum():,}")

# ---- Key comparison: where API shows (D) ----
api_suppressed = merged["value_api"].astype(str).str.strip().isin(["(D)", "(Z)", "(d)", "(z)"])
print(f"\nRows where API value = (D)/(Z): {api_suppressed.sum():,}")

# Of those, what does the .dta show?
api_d_rows = merged[api_suppressed].copy()
print("For rows where API=(D)/(Z), .dta value breakdown:")
print(api_d_rows["value_dta"].value_counts(dropna=False).head(20))
print(f"  Numeric in .dta: {api_d_rows['value_numeric_dta'].notna().sum():,}")
print(f"  NaN / missing in .dta: {api_d_rows['value_numeric_dta'].isna().sum():,}")

# ---- Key comparison: where .dta shows (D) ----
dta_suppressed = merged["value_dta"].astype(str).str.strip().isin(["(D)", "(Z)", "(d)", "(z)"])
print(f"\nRows where .dta value = (D)/(Z): {dta_suppressed.sum():,}")

# ---- Numeric match for rows present in both ----
both_numeric = merged[both & merged["value_numeric_dta"].notna() & merged["value_numeric_api"].notna()].copy()
both_numeric["match"] = np.isclose(both_numeric["value_numeric_dta"], both_numeric["value_numeric_api"])
print(f"\nRows with numeric value in BOTH sources: {len(both_numeric):,}")
print(f"  Exact matches:    {both_numeric['match'].sum():,}")
print(f"  Mismatches:       {(~both_numeric['match']).sum():,}")
if (~both_numeric["match"]).sum() > 0:
    print("Sample mismatches:")
    print(both_numeric[~both_numeric["match"]][
        join_keys + ["value_numeric_dta", "value_numeric_api"]
    ].head(10).to_string(index=False))

# ---- Save for review ----
out_dir = os.path.join(interim, "qa-cafo-dta-vs-api")
os.makedirs(out_dir, exist_ok=True)
today_str = date.today().strftime("%Y-%m-%d")

# Rows where API=(D) but .dta has a number — the smoking gun
api_d_dta_num = api_d_rows[api_d_rows["value_numeric_dta"].notna()]
api_d_dta_num.to_csv(os.path.join(out_dir, f"{today_str}_api_suppressed_dta_numeric.csv"), index=False)
print(f"\nSaved {len(api_d_dta_num):,} rows where API=(D) but .dta is numeric to qa-cafo-dta-vs-api/")

merged.to_csv(os.path.join(out_dir, f"{today_str}_2017_dta_vs_api_full_merge.csv"), index=False)
print(f"Saved full merge ({len(merged):,} rows) to qa-cafo-dta-vs-api/")
