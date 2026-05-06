#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script0p-usda-2022-api-only.py

Pull USDA NASS Quick Stats county-level CAFO-relevant rows for 2022 only.
No .dta loading and no merge/backfill.

Outputs (raw/usda):
- {date}_usda_2022_api_cafo_ops_inventory_raw.csv
- {date}_usda_2022_api_cafo_ops_inventory_filtered.csv
- {date}_usda_2022_api_pull_log.csv
"""

from packages import *
from functions import *
import time

# ----------------------- Config -----------------------
DEFAULT_NASS_API_KEY = "30643212-7739-359A-B451-0EAD3D345DB9"
NASS_API_KEY = os.environ.get("USDA_NASS_API_KEY", DEFAULT_NASS_API_KEY)
if not NASS_API_KEY:
    raise RuntimeError("Missing USDA_NASS_API_KEY env var")

NASS_BASE = "https://quickstats.nass.usda.gov/api/"
YEAR = 2022

source_desc = "CENSUS"
agg_level_desc = "COUNTY"
sector_desc = "ANIMALS & PRODUCTS"
commodity_desc_allow = ["CATTLE", "CHICKENS", "HOGS"]
unit_desc_allow = ["OPERATIONS"]
statisticcat_desc_allow = ["INVENTORY"]

commodity_domain_splits = {
    "CATTLE": [
        "INVENTORY",
        "INVENTORY OF CATTLE, INCL CALVES",
        "INVENTORY OF CATTLE, (EXCL COWS)",
        "INVENTORY OF BEEF COWS",
        "INVENTORY OF MILK COWS",
        "INVENTORY OF CATTLE ON FEED",
    ],
    "CHICKENS": ["INVENTORY"],
    "HOGS": ["INVENTORY OF HOGS", "INVENTORY OF BREEDING HOGS"],
}

out_dir = os.path.join(db_data, "raw", "usda")
os.makedirs(out_dir, exist_ok=True)
today_str = date.today().strftime("%Y-%m-%d")


# ----------------------- Pull -----------------------
frames = []
log_rows = []

for cmd in commodity_desc_allow:
    for unit in unit_desc_allow:
        for stat in statisticcat_desc_allow:
            time.sleep(0.4)
            base_params = {
                "key": NASS_API_KEY,
                "source_desc": source_desc,
                "agg_level_desc": agg_level_desc,
                "sector_desc": sector_desc,
                "commodity_desc": cmd,
                "unit_desc": unit,
                "statisticcat_desc": stat,
                "year": YEAR,
            }

            try:
                count = nass_get_counts(NASS_BASE, base_params)
            except RuntimeError as e:
                log_rows.append({
                    "commodity_desc": cmd,
                    "unit_desc": unit,
                    "statisticcat_desc": stat,
                    "domain_desc": pd.NA,
                    "count": pd.NA,
                    "status": f"error: {e}",
                })
                continue

            if count == 0:
                log_rows.append({
                    "commodity_desc": cmd,
                    "unit_desc": unit,
                    "statisticcat_desc": stat,
                    "domain_desc": pd.NA,
                    "count": 0,
                    "status": "no_rows",
                })
                continue

            if count <= 50000:
                df_pull = nass_get_data(NASS_BASE, base_params)
                if not df_pull.empty:
                    frames.append(df_pull)
                log_rows.append({
                    "commodity_desc": cmd,
                    "unit_desc": unit,
                    "statisticcat_desc": stat,
                    "domain_desc": pd.NA,
                    "count": count,
                    "status": "pulled_unsplit",
                })
                continue

            # split large request by domain_desc
            for dd in commodity_domain_splits.get(cmd, []):
                time.sleep(0.4)
                dd_params = dict(base_params)
                dd_params["domain_desc"] = dd
                try:
                    dd_count = nass_get_counts(NASS_BASE, dd_params)
                except RuntimeError as e:
                    log_rows.append({
                        "commodity_desc": cmd,
                        "unit_desc": unit,
                        "statisticcat_desc": stat,
                        "domain_desc": dd,
                        "count": pd.NA,
                        "status": f"split_error: {e}",
                    })
                    continue

                if dd_count == 0:
                    log_rows.append({
                        "commodity_desc": cmd,
                        "unit_desc": unit,
                        "statisticcat_desc": stat,
                        "domain_desc": dd,
                        "count": 0,
                        "status": "split_no_rows",
                    })
                    continue

                if dd_count > 50000:
                    log_rows.append({
                        "commodity_desc": cmd,
                        "unit_desc": unit,
                        "statisticcat_desc": stat,
                        "domain_desc": dd,
                        "count": dd_count,
                        "status": "split_too_large",
                    })
                    continue

                df_pull = nass_get_data(NASS_BASE, dd_params)
                if not df_pull.empty:
                    frames.append(df_pull)
                log_rows.append({
                    "commodity_desc": cmd,
                    "unit_desc": unit,
                    "statisticcat_desc": stat,
                    "domain_desc": dd,
                    "count": dd_count,
                    "status": "pulled_split",
                })

if not frames:
    raise RuntimeError("No data returned from USDA API for 2022 with the requested filters.")

raw = pd.concat(frames, ignore_index=True)
raw.columns = [c.lower().strip() for c in raw.columns]

# Filter to rows that match downstream CAFO inventory-bin logic.
filtered = raw.copy()
for c in ["domaincat_desc", "unit_desc", "statisticcat_desc", "commodity_desc", "class_desc"]:
    if c in filtered.columns:
        filtered[c] = filtered[c].astype("string").str.strip().str.lower()

class_keep_map = {
    "cattle": {"incl calves", "(excl cows)", "cows, beef", "cows, milk", "calves", "calves, veal", "ge 500 lbs", "heifers, ge 500 lbs, milk replacement"},
    "chickens": {"broilers", "layers", "layers & pullets", "pullets, replacement", "roosters"},
    "hogs": {"all classes", "breeding"},
}
allowed_pairs = {
    (commodity, cls)
    for commodity, classes in class_keep_map.items()
    for cls in classes
}

filtered = filtered[
    (filtered["commodity_desc"].isin(["cattle", "chickens", "hogs"]))
    & (filtered["unit_desc"].isin(["operations", "head"]))
    & (filtered["statisticcat_desc"].isin(["inventory", "operations"]))
    & (filtered["domaincat_desc"].str.startswith("inventory", na=False))
].copy()

pair_index = pd.MultiIndex.from_frame(filtered[["commodity_desc", "class_desc"]])
allowed_index = pd.MultiIndex.from_tuples(sorted(allowed_pairs))
filtered = filtered[pair_index.isin(allowed_index)].copy()

raw_path = os.path.join(out_dir, f"{today_str}_usda_2022_api_cafo_ops_inventory_raw.csv")
flt_path = os.path.join(out_dir, f"{today_str}_usda_2022_api_cafo_ops_inventory_filtered.csv")
log_path = os.path.join(out_dir, f"{today_str}_usda_2022_api_pull_log.csv")

raw.to_csv(raw_path, index=False)
filtered.to_csv(flt_path, index=False)
pd.DataFrame(log_rows).to_csv(log_path, index=False)

print("Saved raw API rows:", raw_path, "| rows:", len(raw))
print("Saved filtered rows:", flt_path, "| rows:", len(filtered))
print("Saved pull log:", log_path)
