# Original header of `qa-scripts/pull_2022_cattle_totals.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

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

````
