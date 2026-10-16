#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qa-scripts/pull_2022_cattle_totals.py  --  one-off: fetch NASS 2022 county TOTAL operations for cattle.

WHY
---
For 2002-2017 the raw NASS pull contains, for each county, (a) operations by size bin and
(b) NASS's own TOTAL operations (domain_desc = TOTAL). Summing the bins and comparing to
the total shows whether any farms were suppressed. The 2022 raw file on disk has bins but
NO cattle TOTAL rows (it has them for chickens and hogs). That file is dated 2026-05-06,
before script0b gained its total-ops request, and script0b never re-pulls a year once a
cache exists, so the totals may simply never have been requested for 2022.

WHAT THIS DOES
--------------
1. Asks NASS Quick Stats for CATTLE / OPERATIONS / INVENTORY / domain TOTAL / COUNTY / CENSUS 2022
   (the same parameters as script0b's total-ops block). If NASS returns 0 rows, it says so
   and writes nothing: that is the answer to "does NASS publish them".
2. Saves the rows to Data/raw/usda/<date>_usda_2022_api_cafo_ops_totals_raw.csv.
3. Compares bin sums to the totals for milk cows (dairy), beef cows and incl-calves, county
   by county, and prints the result.

THE FILE NAME IS DELIBERATELY DIFFERENT from the script0b cache name
(*_usda_2022_api_cafo_ops_inventory_raw*.csv). script0b takes the newest file matching that
pattern as THE 2022 cache; a totals-only file under that name would silently replace the bin
rows. This script never touches the existing 2022 file and does not modify script0b.

USAGE
-----
    export USDA_NASS_API_KEY=...        # your key; this script reads it from the environment only
    python3 qa-scripts/pull_2022_cattle_totals.py
    python3 qa-scripts/pull_2022_cattle_totals.py --selftest 2017   # no API call; runs the comparison on 2017
"""
import glob
import json
import os
import sys
import time
import urllib.error
from datetime import date
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from packages import db_data  # noqa: E402

# add in key line (will remov ewhen public)
USDA_NASS_API_KEY = "30643212-7739-359A-B451-0EAD3D345DB9"

RAW_DIR = os.path.join(db_data, "raw", "usda")
NASS_BASE = "https://quickstats.nass.usda.gov/api/"
YEAR = 2022
CLASSES = {"cows, milk": "inventory of milk cows",
           "cows, beef": "inventory of beef cows",
           "incl calves": "inventory of cattle, incl calves"}


def nass(endpoint, params):
    req = Request(f"{NASS_BASE}{endpoint}/?{urlencode(params)}",
                  headers={"User-Agent": "mentalhealth-ag-script/1.0"})
    print(f"  requesting NASS {endpoint} ...", flush=True)
    for attempt in (1, 2, 3):
        try:
            # timeout: without one, a stalled NASS connection waits forever.
            with urlopen(req, timeout=90) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # The URL carries the key, so report only the status and NASS's message.
            raise RuntimeError(f"NASS API HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:300]}") from None
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            print(f"  attempt {attempt} failed ({type(e).__name__}); "
                  f"{'retrying' if attempt < 3 else 'giving up'}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"NASS {endpoint} did not respond after 3 attempts of 90 s")


def pull_totals(key):
    params = {"key": key, "source_desc": "CENSUS", "agg_level_desc": "COUNTY",
              "sector_desc": "ANIMALS & PRODUCTS", "commodity_desc": "CATTLE",
              "unit_desc": "OPERATIONS", "statisticcat_desc": "INVENTORY",
              "domain_desc": "TOTAL", "year": YEAR}
    n = int(nass("get_counts", params).get("count", 0))
    print(f"NASS reports {n:,} rows for CATTLE / TOTAL / county / {YEAR}")
    if n == 0:
        return pd.DataFrame()
    time.sleep(0.4)
    df = pd.DataFrame(nass("api_GET", params).get("data", []))
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def county_ops(df, domain):
    """Sum of operations by county for one commodity class (value '(D)' becomes NaN)."""
    d = df.copy()
    for c in ["commodity_desc", "class_desc", "domain_desc", "agg_level_desc"]:
        d[c] = d[c].astype(str).str.strip().str.lower()
    d = d[(d.agg_level_desc == "county") & (d.domain_desc == domain)]
    d["fips"] = d.state_fips_code.astype(str).str.zfill(2) + d.county_code.astype(str).str.zfill(3)
    d["n"] = pd.to_numeric(d.value.astype(str).str.replace(",", "").str.strip(), errors="coerce")
    return d, d.groupby(["class_desc", "fips"])["n"].sum(min_count=1)


def compare(ops):
    """Bin sums vs NASS TOTAL, county by county, for each cattle class."""
    for cls, bin_domain in CLASSES.items():
        bins_rows, bins = county_ops(ops, bin_domain)
        tot_rows, tot = county_ops(ops, "total")
        b = bins[bins.index.get_level_values(0) == cls].droplevel(0).rename("bins")
        t = tot[tot.index.get_level_values(0) == cls].droplevel(0).rename("total")
        print(f"\n--- {cls} ---")
        if t.empty:
            print("  no TOTAL rows for this class")
            continue
        m = pd.concat([b, t], axis=1)
        both = m.dropna()
        print(f"  counties with bins: {b.notna().sum():,} | with NASS total: {t.notna().sum():,} | both: {len(both):,}")
        print(f"  bins == total: {int((both.bins == both.total).sum()):,}"
              f" | bins < total (suppressed farms): {int((both.bins < both.total).sum()):,}"
              f" | bins > total: {int((both.bins > both.total).sum()):,}")
        print(f"  total only (NASS counts farms, no bins): {int((m.bins.isna() & m.total.notna()).sum()):,}"
              f" | bins only (no NASS total): {int((m.bins.notna() & m.total.isna()).sum()):,}")
        print(f"  national sum  bins {b.sum():,.0f}  |  NASS county totals {t.sum():,.0f}"
              f"  |  hidden by omission {max(both.total.sum() - both.bins.sum(), 0):,.0f}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        yr = int(sys.argv[sys.argv.index("--selftest") + 1])
        f = sorted(glob.glob(os.path.join(RAW_DIR, f"*_usda_{yr}_api_cafo_ops_inventory_raw*.csv")))[-1]
        print(f"SELF-TEST on {os.path.basename(f)} (no API call)")
        ops = pd.read_csv(f, low_memory=False, dtype=str)
        ops.columns = [c.lower() for c in ops.columns]
        compare(ops)
        sys.exit(0)

    key = os.environ.get("USDA_NASS_API_KEY")
    if not key:
        sys.exit("USDA_NASS_API_KEY is not set in the environment. Export it and rerun.")

    totals = pull_totals(key)
    if totals.empty:
        sys.exit("NASS returned no 2022 cattle TOTAL rows: nothing saved. "
                 "The 2022 size bins cannot be checked against a county total.")

    out = os.path.join(RAW_DIR, f"{date.today():%Y-%m-%d}_usda_{YEAR}_api_cafo_ops_totals_raw.csv")
    totals.to_csv(out, index=False)
    print(f"saved {len(totals):,} rows -> {out}")

    existing = sorted(glob.glob(os.path.join(RAW_DIR, f"*_usda_{YEAR}_api_cafo_ops_inventory_raw*.csv")))[-1]
    bins = pd.read_csv(existing, low_memory=False, dtype=str)
    bins.columns = [c.lower() for c in bins.columns]
    print(f"comparing against bins in {os.path.basename(existing)}")
    compare(pd.concat([bins, totals.astype(str)], ignore_index=True))
