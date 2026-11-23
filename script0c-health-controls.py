#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script0c-health-controls.py  --  county health, mortality and annual-controls ingestion.

Created Wed Aug 13 2025 (Part 1); Part 2 folded in from script0g-annual-controls.py (2026-10-24).
Inputs : Data/raw/mental (CHR), Data/raw/cdc (CDC WONDER), Data/raw/annual_controls (SAIPE, LAUS, SAHIE, PEP)
Outputs: Data/clean/{today}_mentalhealthrank_full.csv
         Data/clean/{today}_cdc_county_year_deathsofdespair.csv
         Data/clean/{today}_cdc_behavioral_mortality_county_year.csv (when new CDC exports are present)
         Data/clean/{today}_acs_farm_occupation_county_year.csv (when ACS exports are present)
         Data/clean/{today}_annual_controls_county_year.csv      (plus QA files in diagnostic_dir)
Run    : python3 script0c-health-controls.py [--skip-download] [--only health|controls]
         --skip-download  use only files already cached in raw/annual_controls; no network calls
         --only           run one part (default: both, health first)

PART 1  Health pipeline (run_health):
1) Load all raw County Health Rankings mental-health CSVs from raw/mental.
2) Infer year from each filename, normalize column names, dedupe duplicate
   columns by keeping the most complete version, and normalize FIPS.
3) Build MH variable-presence QA, then select MH columns using thresholds:
   - majority columns: present in >= 11 files
   - high-fill columns: present in >= 8 files and avg fill >= 80%
4) Concatenate selected MH columns into a single fips-year panel, collapse
   any duplicate keys with first non-null values, and save clean MH output.
5) Load county-level CDC annual files matching:
   cty-level-deathsofdespair-YYYY.csv
   Parse and keep county-year fields (deaths, population, crude rate,
   reliability flags, optional percent-of-total-deaths), then save clean CDC panel.
6) Produce CDC QA outputs (file inventory, by-year coverage, key integrity).
7) Outer-merge MH + CDC county-year panels on (fips, year), create overlap QA
   by year, and save merged health+mortality output (legacy filename retained).
8) Optional extension (off by default): parse disaggregated CDC tranche files
   and write local extension outputs for future fips-year-race-sex panel work.
9) Optional new sources: clean grouped CDC mortality exports from
   raw/cdc/behavioral_mortality/ and ACS 5-year C24010 farming-fishing-forestry
   occupation counts (fetched from the Census API, needs CENSUS_API_KEY, or
   cached under raw/acs/farm_occ_workers/); each source is saved as its own
   county-year clean file and is merged by script0f when present.

Notes:
- Optional extension is controlled by env var RUN_CDC_DISAGG_EXTENSION
  (set to 1/true/yes/y to enable).
- Disaggregated extension outputs are NOT used in the main county-year panel.

PART 2  Annual controls (run_controls) -- download annual, correctly-dated county controls.

WHY THIS EXISTS
---------------
County Health Rankings dates every measure by its RELEASE year, not the year the
data describes. Confirmed against CHR's own release code and documentation:
`median_household_income`, `children_in_poverty` and `unemployment` are lagged 2
years; `uninsured_adults` is lagged 3. A panel row labelled 2020 therefore holds
2018 income and 2017 uninsured rates, while CAFO treatment is dated by the actual
ag-census year. This part replaces those four with series pulled straight from
the primary source, dated by the year the data actually describes.

SOURCES (all annual, all counties, no population floor)
-------------------------------------------------------
  SAIPE  Small Area Income and Poverty Estimates, Census
         -> median_household_income, children_in_poverty
         est{YY}all.dat  (2000-2002)  /  est{YY}all.xls  (2003+)
  LAUS   Local Area Unemployment Statistics, BLS
         -> unemployment rate
         download.bls.gov (www.bls.gov returns 403 to non-browser clients)
  SAHIE  Small Area Health Insurance Estimates, Census
         -> uninsured adults 18-64
         sahie-{YEAR}-csv.zip
  PEP    Population Estimates Program, Census
         -> county demographic shares (see the PEP section below)

Raw downloads -> Data/raw/annual_controls/<source>/   (cached; a cached file is never re-downloaded)
script0f drops the CHR columns these series replace (CHR_REPLACED_COLS) before merging them in.
"""

# ----------------------- SET UP PART 1 : IMPORT KEY PATHS  -------------------- -#

import argparse
import io
import json
import re
import time
import urllib.request
import warnings
import zipfile
from functions import *
from packages import *


# folders
inf           = os.path.join(db_data, "raw")
outf          = os.path.join(db_data, "clean")
local_ext_dir = os.path.join(interim, "health-disagg-extension")
os.makedirs(diagnostic_dir, exist_ok=True)
os.makedirs(local_ext_dir,  exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")
RUN_CDC_DISAGG_EXTENSION = os.getenv("RUN_CDC_DISAGG_EXTENSION", "0").strip().lower() in {"1", "true", "yes", "y"}


# ----------------------- SET UP PART 2: QA HELPERS  -------------------- -#
qa_fill_frames = []
qa_key_frames = []


def _add_fill(stage, df):
    fill = pd.DataFrame(
        {
            "stage": stage,
            "column": df.columns,
            "n_rows": len(df),
            "n_non_null": [int(df[c].notna().sum()) for c in df.columns],
        }
    )
    fill["fill_pct"] = np.where(fill["n_rows"] > 0, (fill["n_non_null"] / fill["n_rows"]) * 100, np.nan)
    qa_fill_frames.append(fill)


def _add_key_qa(stage, df, keys):
    missing_keys = [k for k in keys if k not in df.columns]
    if missing_keys:
        qa_key_frames.append(
            pd.DataFrame(
                {
                    "stage": [stage],
                    "keys": [",".join(keys)],
                    "n_rows": [len(df)],
                    "n_unique_keys": [np.nan],
                    "n_duplicate_rows": [np.nan],
                    "note": [f"missing key columns: {missing_keys}"],
                }
            )
        )
        return

    dup_rows = int(df.duplicated(keys, keep=False).sum())
    n_unique = int(df[keys].drop_duplicates().shape[0])
    qa_key_frames.append(
        pd.DataFrame(
            {
                "stage": [stage],
                "keys": [",".join(keys)],
                "n_rows": [len(df)],
                "n_unique_keys": [n_unique],
                "n_duplicate_rows": [dup_rows],
                "note": [""],
            }
        )
    )


def _finalize_qa():
    fill_df = pd.concat(qa_fill_frames, ignore_index=True) if qa_fill_frames else pd.DataFrame()
    key_df = pd.concat(qa_key_frames, ignore_index=True) if qa_key_frames else pd.DataFrame()
    fill_path = os.path.join(diagnostic_dir, f"{today_str}_qa_health_stage_fill.csv")
    key_path = os.path.join(diagnostic_dir, f"{today_str}_qa_health_stage_keys.csv")
    fill_df.to_csv(fill_path, index=False)
    key_df.to_csv(key_path, index=False)
    print("Saved QA:", fill_path)
    print("Saved QA:", key_path)


def _clean_mh_cols(cols):
    s = pd.Index(cols).astype(str)
    s = s.str.lower()
    s = s.str.replace("&", "and", regex=False)
    s = s.str.replace(r"[.,]", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    # harmonize common drifted labels
    s = s.str.replace(r"\blimited access to healthy foods\b", "access to healthy foods", regex=True)
    s = s.str.replace(r"\bfemales\b", "female", regex=True)
    s = s.str.replace(
        r"% native hawaiian\s+or\s+other pacific islander\b",
        "% native hawaiian/other pacific islander",
        regex=True,
    )
    s = s.str.replace(
        r"% american indian\s+or\s+alaska native\b",
        "% american indian and alaska native",
        regex=True,
    )

    s = s.str.replace(r"\s+", "_", regex=True).str.replace(r"_+", "_", regex=True).str.strip("_")
    return s


def _dedupe_columns_keep_most_complete(df):
    keep_positions = []
    for col in pd.Index(df.columns).unique():
        pos = np.where(df.columns == col)[0]
        if len(pos) == 1:
            keep_positions.append(pos[0])
            continue
        completeness = [df.iloc[:, i].notna().mean() for i in pos]
        keep_positions.append(pos[int(np.argmax(completeness))])
    keep_positions = sorted(set(keep_positions))
    out = df.iloc[:, keep_positions].copy()
    out.columns = df.columns[keep_positions]
    return out


_first_notna = first_non_null


def _normalize_fips(series):
    s = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(5)
    )
    return s


def _read_grouped_cdc_exports(raw_cdc):
    """Clean CDC WONDER query exports for the grouped mortality outcomes (D-68).

    Expected raw files live in raw/cdc/behavioral_mortality/, one multi-year
    WONDER export per measure, named cdc_<measure>.csv or cdc_<measure>.txt
    (measure = union, suicide, drug_overdose, alcohol_induced). Each query is
    grouped by County and Year and exports County, County Code, Year (Code),
    Deaths, Population, and Crude Rate. WONDER's default export is tab
    delimited with a notes footer; both delimiters are read and footer lines
    are dropped because their County Code is not a 5-digit FIPS.

    The union is its own WONDER query, so WONDER applies its suppression rule
    (1-9 deaths) to the combined count; component series are never summed here.
    Suppressed counts stay missing (not zero) and are flagged; with "Show Zero
    Values" ticked in the query, county-years with no deaths appear as 0.
    """
    folder = os.path.join(raw_cdc, "behavioral_mortality")
    measures = ("union", "suicide", "drug_overdose", "alcohol_induced")
    files = {}
    for path in sorted(glob.glob(os.path.join(folder, "cdc_*.*"))):
        m = re.fullmatch(r"cdc_(union|suicide|drug_overdose|alcohol_induced)\.(csv|txt)", os.path.basename(path))
        if not m:
            continue
        if m.group(1) in files:
            raise ValueError(f"More than one export for measure {m.group(1)} under {folder}")
        files[m.group(1)] = path
    if not files:
        print(f"Grouped CDC mortality inputs not present; expected exports under {folder}")
        return None

    parts = []
    for measure in measures:
        if measure not in files:
            continue
        path = files[measure]
        base = os.path.basename(path)
        with open(path, encoding="latin1") as fh:
            sep = "\t" if "\t" in fh.readline() else ","
        d = read_csv_with_fallback(path, sep=sep, low_memory=False, dtype=str)
        d = clean_cols(d.copy())
        d.columns = d.columns.str.replace(r"\s+", "_", regex=True).str.strip("_")
        year_col = "year_code" if "year_code" in d.columns else "year"
        required = {"county_code", "deaths", "population", "crude_rate", year_col}
        missing = sorted(required - set(d.columns))
        if missing:
            raise KeyError(f"{base} missing required CDC WONDER fields: {missing}; "
                           "group the query by County and Year")

        n_raw = len(d)
        d["fips"] = _normalize_fips(d["county_code"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True))
        d = d[d["fips"].str.fullmatch(r"\d{5}", na=False)].copy()
        d["year"] = to_numeric_series(d[year_col]).astype("Int64")
        deaths_raw = d["deaths"].astype("string").str.strip()
        d[f"deaths_{measure}"] = to_numeric_series(deaths_raw).astype("Int64")
        d[f"deaths_suppressed_{measure}"] = deaths_raw.str.contains("suppress", case=False, na=False).astype("Int64")
        d[f"population_{measure}"] = to_numeric_series(d["population"]).astype("Int64")
        rate_raw = d["crude_rate"].astype("string").str.strip()
        d[f"crude_rate_{measure}_raw"] = rate_raw
        d[f"crude_rate_{measure}"] = to_numeric_series(rate_raw.str.replace(r"\(.*?\)", "", regex=True))
        d[f"rate_unreliable_{measure}"] = rate_raw.str.contains("unreliable", case=False, na=False).astype("Int64")
        d = d[["fips", "year", f"deaths_{measure}", f"deaths_suppressed_{measure}", f"population_{measure}",
               f"crude_rate_{measure}", f"crude_rate_{measure}_raw", f"rate_unreliable_{measure}"]]
        if d["year"].isna().any():
            raise ValueError(f"{base} has county rows without a year; group the query by County and Year")
        if d.duplicated(["fips", "year"]).any():
            raise ValueError(f"{base} has duplicate county-year rows; inspect the WONDER export before cleaning")
        print(f"  {measure}: {n_raw:,} raw lines -> {len(d):,} county-years, years "
              f"{int(d['year'].min())}-{int(d['year'].max())}, "
              f"suppressed {int(d[f'deaths_suppressed_{measure}'].sum()):,}, "
              f"unreliable rates {int(d[f'rate_unreliable_{measure}'].sum()):,}")
        parts.append(d)

    out = parts[0]
    for part in parts[1:]:
        n_before = len(out)
        out = out.merge(part, on=["fips", "year"], how="outer", validate="one_to_one")
        print(f"  outer merge: {n_before:,} + {len(part):,} -> {len(out):,} county-years")
    out = out.sort_values(["fips", "year"]).reset_index(drop=True)
    count_cols = [f"deaths_{m}" for m in measures if f"deaths_{m}" in out.columns]
    print(f"Grouped CDC mortality cleaned: {len(out):,} county-year rows; measures={', '.join(count_cols)}")
    return out


ACS_YEARS = range(2009, 2024)   # release years tried; 2009 = 2005-2009, the first county-level 5-year file


def _acs_get_json(url, tries=3):
    """GET a Census API URL and parse the JSON; None if unavailable. Generic client name, no contact details."""
    req = urllib.request.Request(url, headers={"User-Agent": "academic-research-client"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if k == tries - 1:
                print(f"    ACS request failed ({type(e).__name__}): {url.split('?')[0]}")
                return None
            time.sleep(2)


def _acs_pick_rows(variables):
    """Find the C24010 total, male and female 'Farming, fishing, and forestry occupations' estimate rows by LABEL.

    C24010 (sex by occupation, civilian employed population 16+) is the
    county-level table that carries this occupation group; the detailed
    'agricultural workers' tables are not published at county level (D-70).
    Row numbers differ across vintages (2009 release: 029/066; 2010 onward:
    031/067), so rows are located from each vintage's own labels. Labels are
    split on '!!' and stripped of ':'; total = ['Estimate', 'Total']; farm row =
    last segment equals the occupation group and the third segment is Male or
    Female. Anything but exactly one match per role raises, listing candidates.
    """
    target = "farming, fishing, and forestry occupations"
    picked, labels = {}, {}
    cand = {"total": [], "farm_male": [], "farm_female": []}
    for var, meta in variables.items():
        if not (var.startswith("C24010_") and var.endswith("E")):
            continue
        segs = [s.strip().rstrip(":").strip() for s in meta.get("label", "").split("!!") if s.strip()]
        if len(segs) == 2 and segs[1].lower() == "total":
            cand["total"].append((var, meta["label"]))
        elif len(segs) >= 3 and segs[-1].lower() == target:
            if segs[2].lower() == "male":
                cand["farm_male"].append((var, meta["label"]))
            elif segs[2].lower() == "female":
                cand["farm_female"].append((var, meta["label"]))
    for role, found in cand.items():
        if len(found) != 1:
            raise KeyError(f"C24010 row for '{role}': expected exactly one label match, found {found}")
        picked[role], labels[role] = found[0]
    return picked, labels


def _download_acs_c24010(raw_root):
    """Fetch ACS 5-year C24010 county estimates and margins of error from the Census API.

    Writes raw/acs/farm_occ_workers/acs5_c24010_<release_year>.csv (GEO_ID plus the
    total, male-farm-occupation and female-farm-occupation estimate and margin
    columns) and upserts the sidecar acs5_c24010_rows.csv recording which
    variable and label was used for each role in each year. Cached years are
    not re-fetched. An optional CENSUS_API_KEY in the environment is sent
    if set. With --skip-download nothing is fetched.
    """
    folder = os.path.join(raw_root, "acs", "farm_occ_workers")
    rows_path = os.path.join(folder, "acs5_c24010_rows.csv")
    if SKIP_DOWNLOAD:
        print("ACS C24010: --skip-download, using cached files only")
        return
    os.makedirs(folder, exist_ok=True)
    rows = pd.read_csv(rows_path, dtype={"year": int}) if os.path.exists(rows_path) else \
        pd.DataFrame(columns=["year", "role", "variable", "label"])
    key = os.environ.get("CENSUS_API_KEY")
    key_q = f"&key={key}" if key else ""
    for y in ACS_YEARS:
        dest = os.path.join(folder, f"acs5_c24010_{y}.csv")
        if os.path.exists(dest) and (rows["year"] == y).sum() == 3:
            continue
        meta = _acs_get_json(f"https://api.census.gov/data/{y}/acs/acs5/groups/C24010.json")
        if meta is None:
            print(f"    ACS {y}: C24010 variable list unavailable; skipped")
            continue
        picked, labels = _acs_pick_rows(meta["variables"])
        base_vars = [picked[k][:-1] for k in ("total", "farm_male", "farm_female")]   # C24010_xxx without E/M
        get = ",".join(f"{v}{s}" for v in base_vars for s in ("E", "M"))
        data = _acs_get_json(f"https://api.census.gov/data/{y}/acs/acs5?get=NAME,{get}&for=county:*{key_q}")
        if not data:
            print(f"    ACS {y}: county data unavailable; skipped")
            continue
        d = pd.DataFrame(data[1:], columns=data[0])
        d["GEO_ID"] = "0500000US" + d["state"].str.zfill(2) + d["county"].str.zfill(3)
        d = d[["GEO_ID", "NAME"] + [f"{v}{s}" for v in base_vars for s in ("E", "M")]]
        d.to_csv(dest, index=False)
        rows = rows[rows["year"] != y]
        rows = pd.concat([rows, pd.DataFrame(
            {"year": y, "role": list(picked), "variable": [picked[k][:-1] for k in picked],
             "label": [labels[k] for k in picked]})], ignore_index=True)
        print(f"    ACS {y}: {len(d):,} counties; rows {', '.join(f'{k}={picked[k][:-1]}' for k in picked)}")
    rows.sort_values(["year", "role"]).to_csv(rows_path, index=False)


def _read_acs_farm_occupation(raw_root):
    """Clean ACS 5-year C24010 county files into a farm-occupation share panel (D-67, D-70).

    Raw files (written by _download_acs_c24010 or saved by hand) are
    acs5_c24010_<release_year>.csv, one row per county with GEO_ID and the
    variables named in the sidecar acs5_c24010_rows.csv, which records for
    every release year which Census variable is the total, male farming-
    fishing-forestry and female farming-fishing-forestry row (found by label,
    because C24010 row numbers change across vintages). Numerator: male plus
    female workers in Farming, fishing, and forestry occupations; denominator:
    employed civilians aged 16 and over. Release year = last year of the
    five-year window, so adjacent years overlap by four years.
    """
    folder = os.path.join(raw_root, "acs", "farm_occ_workers")
    rows_path = os.path.join(folder, "acs5_c24010_rows.csv")
    files = sorted(glob.glob(os.path.join(folder, "acs5_c24010_20??.csv")))
    if not files or not os.path.exists(rows_path):
        print(f"ACS farm-occupation inputs not present; expected files and acs5_c24010_rows.csv under {folder}")
        return None
    rows = pd.read_csv(rows_path, dtype={"year": int})

    parts = []
    seen_years = set()
    for path in files:
        base = os.path.basename(path)
        m = re.fullmatch(r"acs5_c24010_(20\d{2})\.csv", base)
        if not m:
            continue
        release_year = int(m.group(1))
        if release_year in seen_years:
            raise ValueError(f"Duplicate ACS C24010 export for {release_year}: {base}")
        seen_years.add(release_year)

        # Variable codes for this vintage, from the sidecar (never hard-coded).
        r = rows[rows["year"] == release_year].set_index("role")["variable"].to_dict()
        for role in ("total", "farm_male", "farm_female"):
            if role not in r:
                raise KeyError(f"acs5_c24010_rows.csv has no '{role}' row for {release_year}")
        d = read_csv_with_fallback(path, dtype=str, low_memory=False)
        needed = {"GEO_ID"} | {f"{r[k]}{s}" for k in r for s in ("E", "M")}
        missing = sorted(needed - set(d.columns))
        if missing:
            raise KeyError(f"{base} missing ACS C24010 fields: {missing}")

        # GEO_ID looks like 0500000US01001; the county FIPS is the last five digits.
        fips = d["GEO_ID"].astype("string").str.extract(r"US(\d{5})$", expand=False)

        d["fips"] = _normalize_fips(fips)
        d = d[d["fips"].str.fullmatch(r"\d{5}", na=False)].copy()
        d["year"] = release_year
        d["period_start"] = release_year - 4
        d["period_end"] = release_year
        # The Census API codes "not available" as large negative sentinels (e.g. -666666666); a count or
        # margin of error is never negative, so negatives become missing rather than entering the share.
        def num(col):
            x = to_numeric_series(d[col])
            return x.where(x >= 0)
        d["employed_civilian_16plus"] = num(f"{r['total']}E")
        d["employed_civilian_16plus_moe"] = num(f"{r['total']}M")
        d["farm_occ_male"] = num(f"{r['farm_male']}E")
        d["farm_occ_female"] = num(f"{r['farm_female']}E")
        d["farm_occ_workers"] = d[["farm_occ_male", "farm_occ_female"]].sum(axis=1, min_count=2)
        male_moe = num(f"{r['farm_male']}M")
        female_moe = num(f"{r['farm_female']}M")
        d["farm_occ_moe"] = np.sqrt(male_moe.pow(2) + female_moe.pow(2))
        d["farm_occ_share"] = (
            d["farm_occ_workers"] / d["employed_civilian_16plus"]
        ).where(d["employed_civilian_16plus"] > 0)
        keep = ["fips", "year", "period_start", "period_end", "farm_occ_workers",
                "farm_occ_male", "farm_occ_female", "farm_occ_moe",
                "employed_civilian_16plus", "employed_civilian_16plus_moe", "farm_occ_share"]
        d = d[keep]
        if d.duplicated(["fips", "year"]).any():
            raise ValueError(f"{base} has duplicate county rows")
        parts.append(d)

    if not parts:
        print(f"No recognized ACS C24010 exports found under {folder}")
        return None
    out = pd.concat(parts, ignore_index=True).sort_values(["fips", "year"]).reset_index(drop=True)
    print(f"ACS farm-occupation workers cleaned: {len(out):,} county-period rows; "
          f"release years={out['year'].min()}–{out['year'].max()}")
    return out


# =============================================================================
# PART 1 -- HEALTH PIPELINE (County Health Rankings + CDC deaths of despair)
# =============================================================================
def run_health():

    # ----------------------- DATA PART 1 : CLEAN MH SURVEY DATA  -------------------- -#

    # ID folder for the raw data 
    raw_mh = os.path.join(inf, "mental")
    mh_files = sorted(glob.glob(os.path.join(raw_mh, "*.csv")))
    if not mh_files:
        raise FileNotFoundError(f"No mental health files found in {raw_mh}") # qa check in case the files go missing! 

    # create blank directories to fill with the individual files 
    mh_file_inventory = []
    mh_dfs = []
    year_re = re.compile(r"(19|20)\d{2}(?=\.[A-Za-z0-9]+$)")

    # read through folder, find each year's surbey
    for p in mh_files:
        d = read_csv_with_fallback(p, low_memory=False)
        base = os.path.basename(p)
        m = year_re.search(base) # take the year from the filename 
        if not m:
            raise ValueError(f"Could not infer year from filename: {base}")
        yr = int(m.group(0))

        d.columns = _clean_mh_cols(d.columns)
        d = _dedupe_columns_keep_most_complete(d)
        d["year"] = yr # create column with the year from the year string

        if "5-digit_fips_code" not in d.columns:
            alt = [c for c in d.columns if "fips" in c]
            if alt:
                d = d.rename(columns={alt[0]: "5-digit_fips_code"})
            else:
                raise KeyError(f"Missing FIPS column in {base}")

        d["5-digit_fips_code"] = _normalize_fips(d["5-digit_fips_code"]) # pad / destring the fips 
        d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")

        mh_file_inventory.append(
            {
                "file": base,
                "year": yr,
                "n_rows": len(d),
                "n_cols": len(d.columns),
                "n_dup_cols_after_clean": int(d.columns.duplicated().sum()),
            }
        )
        mh_dfs.append(d) # append all data frames into long form 

    # convert to CSV 
    pd.DataFrame(mh_file_inventory).sort_values("year").to_csv(
        os.path.join(diagnostic_dir, f"{today_str}_qa_health_mh_file_inventory.csv"), index=False
    )

    # concat all files 
    _add_fill("mh_raw_union_preselect", pd.concat(mh_dfs, ignore_index=True, sort=False))




    # ----------------------- DATA PART 2 : MH - SELECT KEY VARS  -------------------- -#

    # review all cols in the panel 
    all_cols = sorted(set().union(*[set(d.columns) for d in mh_dfs]))

    presence_rows = []
    for col in all_cols:
        dfs_with_col = sum(col in d.columns for d in mh_dfs)
        fill_vals = [d[col].notna().mean() * 100 for d in mh_dfs if col in d.columns] # calculate fill rates
        presence_rows.append(
            {
                "column": col,
                "dfs_with_col": dfs_with_col,
                "avg_fill_pct_when_present": float(np.mean(fill_vals)) if fill_vals else np.nan,
            }
        )

    # create a df that tells us the share of complete rows for a particular column 
    presence_df = pd.DataFrame(presence_rows).sort_values(
        ["dfs_with_col", "avg_fill_pct_when_present", "column"], ascending=[False, False, True]
    )

    # ouput this QA file 
    presence_path = os.path.join(diagnostic_dir, f"{today_str}_qa_health_mh_column_presence.csv")
    presence_df.to_csv(presence_path, index=False)
    print("Saved QA:", presence_path)


    # threshold gen --- use an arbitrarily chosen threshold (adjustable) to determine which cols we should keep 
    # threshold is 2 part: (1) MAJORITY - extensive margin - variable is in most survey years 
    # threshold is 2 part: (2) HIGH FILL - intensive margin - variable is in fewer survey years but is more robust w/in each year

    yr_threshold_ct = 11 # MAJORITY COL: variable name must be present in at least 11 of 15 
    avg_threshold = 80 # HIGH FILL:  choose percentage (80%) of total row fill when slicing for remaining cols 
    yr_threshold_ct_2 = 8 # HIGH FILL: choose a lower percentage (8 of 15) for years being present for a particular variable name 

    # apply the threshold 
    majority_cols = presence_df.loc[presence_df["dfs_with_col"] >= yr_threshold_ct, "column"].tolist()
    high_fill_cols = presence_df.loc[
        (presence_df["dfs_with_col"] >= yr_threshold_ct_2) & (presence_df["avg_fill_pct_when_present"] >= avg_threshold),
        "column",
    ].tolist()
    selected_cols = list(dict.fromkeys(["5-digit_fips_code", "year"] + majority_cols + high_fill_cols))


    # fill in dfs into set of list - copy so earlier data cleaning is traceable 
    mh_selected = []
    for d in mh_dfs:
        keep = [c for c in selected_cols if c in d.columns]
        mh_selected.append(d[keep].copy())

    mh_combined = pd.concat(mh_selected, ignore_index=True, sort=False)
    _add_fill("mh_selected_concat", mh_combined)
    _add_key_qa("mh_selected_concat", mh_combined, ["5-digit_fips_code", "year"])


    # drop dupes if present by kepping the first in a group 
    if mh_combined.duplicated(["5-digit_fips_code", "year"]).any():
        mh_combined = (
            mh_combined.groupby(["5-digit_fips_code", "year"], as_index=False).agg(_first_notna)
        )

    # clean cols - make the year cols numeric, apply fips normalization to fips code 
    mh_combined = mh_combined.rename(columns={"5-digit_fips_code": "fips"})
    mh_combined["fips"] = _normalize_fips(mh_combined["fips"])
    mh_combined["year"] = pd.to_numeric(mh_combined["year"], errors="coerce").astype("Int64")
    mh_combined = mh_combined.sort_values(["fips", "year"]).reset_index(drop=True)

    _add_fill("mh_final", mh_combined)
    _add_key_qa("mh_final", mh_combined, ["fips", "year"])

    # save output
    mh_out = os.path.join(outf, f"{today_str}_mentalhealthrank_full.csv")
    mh_combined.to_csv(mh_out, index=False)
    print("Saved:", mh_out)


    # ----------------------- DATA PART 2 : CLEAN CDC COUNTY-YEAR AGG DATA -------------------- -#
    raw_cdc = os.path.join(inf, "cdc")
    county_files = sorted(glob.glob(os.path.join(raw_cdc, "cty-level-deathsofdespair-*.csv")))
    if not county_files:
        raise FileNotFoundError(
            f"No county-level deaths-of-despair files found in {raw_cdc}. "
            "Expected pattern: cty-level-deathsofdespair-YYYY.csv"
        )

    cdc_parts = []
    cdc_inventory_rows = []

    for p in county_files:
        base = os.path.basename(p)
        m = re.search(r"(19|20)\d{2}(?=\.csv$)", base)
        if not m:
            raise ValueError(f"Could not infer year from filename: {base}")
        src_year = int(m.group(0))

        d = read_csv_with_fallback(p, low_memory=False)
        d = clean_cols(d.copy())
        d.columns = d.columns.str.replace(r"\s+", "_", regex=True).str.strip("_")

        req = {"county", "county_code", "deaths", "population", "crude_rate"}
        missing = sorted(req - set(d.columns))
        if missing:
            raise KeyError(f"{base} missing required columns: {missing}")

        # Keep only true data rows (drop footer metadata lines)
        d["county_code_num"] = to_numeric_series(d["county_code"])
        d = d[d["county_code_num"].notna()].copy()

        d["fips"] = d["county_code_num"].round().astype("Int64").astype("string").str.zfill(5)
        d["year"] = src_year
        d["county"] = d["county"].astype("string").str.strip()
        d["state_abbrev"] = d["county"].str.extract(r",\s*([A-Z]{2})$", expand=False)
        d["county_name"] = d["county"].str.replace(r",\s*[A-Z]{2}$", "", regex=True).str.strip()

        d["deaths"] = to_numeric_series(d["deaths"]).astype("Int64")
        d["population"] = to_numeric_series(d["population"]).astype("Int64")
        d["crude_rate_raw"] = d["crude_rate"].astype("string").str.strip()
        d["crude_rate"] = to_numeric_series(d["crude_rate_raw"].str.replace(r"\(.*?\)", "", regex=True))
        d["is_unreliable"] = d["crude_rate_raw"].str.contains("unreliable", case=False, na=False).astype("Int64")

        if "%_of_total_deaths" in d.columns:
            d["pct_of_total_deaths_raw"] = d["%_of_total_deaths"].astype("string").str.strip()
            d["pct_of_total_deaths"] = to_numeric_series(d["pct_of_total_deaths_raw"].str.replace("%", "", regex=False))
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
        out["source_file"] = base
        cdc_parts.append(out)

        cdc_inventory_rows.append(
            {
                "file": base,
                "year_from_filename": src_year,
                "rows_raw": int(len(read_csv_with_fallback(p, low_memory=False))),
                "rows_kept_with_county_code": int(len(out)),
                "n_unique_fips_kept": int(out["fips"].nunique()),
                "n_missing_deaths": int(out["deaths"].isna().sum()),
                "n_missing_population": int(out["population"].isna().sum()),
            }
        )

    cdc_panel = pd.concat(cdc_parts, ignore_index=True, sort=False)

    if cdc_panel.duplicated(["fips", "year"]).any():
        cdc_panel = (
            cdc_panel.groupby(["fips", "year"], as_index=False)
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

    cdc_panel = cdc_panel.sort_values(["year", "fips"]).reset_index(drop=True)
    _add_fill("cdc_county_panel_final", cdc_panel)
    _add_key_qa("cdc_county_panel_final", cdc_panel, ["fips", "year"])

    cdc_clean_out = os.path.join(outf, f"{today_str}_cdc_county_year_deathsofdespair.csv")
    cdc_panel.to_csv(cdc_clean_out, index=False)
    print("Saved:", cdc_clean_out)

    cdc_inv = pd.DataFrame(cdc_inventory_rows).sort_values("year_from_filename")
    cdc_inv_out = os.path.join(diagnostic_dir, f"{today_str}_qa_cdc_deathsofdespair_file_inventory.csv")
    cdc_inv.to_csv(cdc_inv_out, index=False)
    print("Saved QA:", cdc_inv_out)

    cdc_by_year = (
        cdc_panel.groupby("year", as_index=False)
        .agg(
            n_rows=("fips", "size"),
            n_unique_fips=("fips", "nunique"),
            n_nonmissing_deaths=("deaths", lambda s: int(s.notna().sum())),
            n_nonmissing_population=("population", lambda s: int(s.notna().sum())),
            n_unreliable=("is_unreliable", lambda s: int((s > 0).sum())),
        )
        .sort_values("year")
    )
    cdc_by_year_out = os.path.join(diagnostic_dir, f"{today_str}_qa_cdc_deathsofdespair_by_year.csv")
    cdc_by_year.to_csv(cdc_by_year_out, index=False)
    print("Saved QA:", cdc_by_year_out)

    cdc_key_check = pd.DataFrame(
        [
            {
                "n_rows": int(len(cdc_panel)),
                "n_unique_fips_year": int(cdc_panel[["fips", "year"]].drop_duplicates().shape[0]),
                "n_duplicate_fips_year_rows": int(cdc_panel.duplicated(["fips", "year"]).sum()),
                "year_min": int(cdc_panel["year"].min()),
                "year_max": int(cdc_panel["year"].max()),
                "n_unique_years": int(cdc_panel["year"].nunique()),
                "n_unique_fips": int(cdc_panel["fips"].nunique()),
            }
        ]
    )
    cdc_key_out = os.path.join(diagnostic_dir, f"{today_str}_qa_cdc_deathsofdespair_key_check.csv")
    cdc_key_check.to_csv(cdc_key_out, index=False)
    print("Saved QA:", cdc_key_out)

    # ---------- PART 2B: GROUPED CDC MORTALITY + ACS FARM OCCUPATION ----------
    # These are separate outcomes/inputs from the legacy deaths-of-despair series.
    # The CDC union file must be a WONDER query with all selected causes combined;
    # do not sum code-level files after their individual counts have been suppressed.
    grouped_mortality = _read_grouped_cdc_exports(raw_cdc)
    if grouped_mortality is not None:
        grouped_out = os.path.join(outf, f"{today_str}_cdc_behavioral_mortality_county_year.csv")
        grouped_mortality.to_csv(grouped_out, index=False)
        print("Saved:", grouped_out)
        _add_fill("cdc_behavioral_mortality_final", grouped_mortality)
        _add_key_qa("cdc_behavioral_mortality_final", grouped_mortality, ["fips", "year"])

    _download_acs_c24010(inf)
    acs_farm_occupation = _read_acs_farm_occupation(inf)
    if acs_farm_occupation is not None:
        acs_out = os.path.join(outf, f"{today_str}_acs_farm_occupation_county_year.csv")
        acs_farm_occupation.to_csv(acs_out, index=False)
        print("Saved:", acs_out)
        _add_fill("acs_farm_occupation_final", acs_farm_occupation)
        _add_key_qa("acs_farm_occupation_final", acs_farm_occupation, ["fips", "year"])


    # ---------- PART 3: Merge health + county CDC ----------
    for d in [mh_combined, cdc_panel]:
        d["fips"] = _normalize_fips(d["fips"])
        d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")

    if mh_combined.duplicated(["fips", "year"]).any():
        mh_combined = mh_combined.groupby(["fips", "year"], as_index=False).agg(_first_notna)
    if cdc_panel.duplicated(["fips", "year"]).any():
        cdc_panel = cdc_panel.groupby(["fips", "year"], as_index=False).agg(_first_notna)

    wide_mh_full = pd.merge(
        mh_combined,
        cdc_panel,
        on=["fips", "year"],
        how="outer",
        sort=True,
        validate="1:1",
    )

    wide_mh_full["has_mh"] = wide_mh_full[[c for c in mh_combined.columns if c not in ["fips", "year"]]].notna().any(axis=1)
    wide_mh_full["has_mort"] = wide_mh_full["deaths"].notna()

    overlap_df = (
        wide_mh_full.groupby("year", as_index=False)
        .agg(
            rows=("fips", "size"),
            mh_rows=("has_mh", "sum"),
            mort_rows=("has_mort", "sum"),
            overlap_rows=("has_mh", lambda s: int((s & wide_mh_full.loc[s.index, "has_mort"]).sum())),
        )
    )
    overlap_path = os.path.join(diagnostic_dir, f"{today_str}_qa_health_overlap_by_year.csv")
    overlap_df.to_csv(overlap_path, index=False)
    print("Saved QA:", overlap_path)

    _add_fill("mh_cdc_merged_final", wide_mh_full)
    _add_key_qa("mh_cdc_merged_final", wide_mh_full, ["fips", "year"])

    final_out = os.path.join(diagnostic_dir, f"{today_str}_mh_mortality_fips_yr.csv")
    wide_mh_full.to_csv(final_out, index=False)
    print("Saved:", final_out)


    # ---------- OPTIONAL EXTENSION: CDC DISAGGREGATED PANELS ----------
    if RUN_CDC_DISAGG_EXTENSION:
        # read in the disaggregated (sex, age, ethnic)
        cdc_csv_files = sorted(glob.glob(os.path.join(raw_cdc, "*.csv")))
        disagg_files = [p for p in cdc_csv_files if not os.path.basename(p).startswith("cty-level-deathsofdespair-")]
        # QA for file location
        if not disagg_files:
            print("Disaggregated CDC extension requested, but no tranche files found. Skipping extension.")
        else:
            mort_raw_parts = []
            for p in disagg_files:
                d = read_csv_with_fallback(p, low_memory=False)
                d.columns = ( # helper to clean 
                    pd.Index(d.columns)
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .str.replace(r"\s+", "_", regex=True)
                )
            
                req = ["state", "county", "county_code", "year", "deaths", "population", "race"]
                missing = [c for c in req if c not in d.columns]
                if missing:
                    continue
                # further disagg 
                if "sex" not in d.columns:
                    lower_name = os.path.basename(p).lower()
                    d["sex"] = "male" if "male" in lower_name else ("female" if "female" in lower_name else pd.NA)
                # clean the cols again 
                d["source_file"] = os.path.basename(p)
                d["state"] = d["state"].astype("string").str.strip()
                d["county"] = d["county"].astype("string").str.replace(r",\s*[A-Z]{2}$", "", regex=True).str.strip()
                d["race"] = d["race"].astype("string").str.strip()
                d["sex"] = d["sex"].astype("string").str.strip().str.title()
                d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")
                d["deaths"] = pd.to_numeric(d["deaths"], errors="coerce")
                d["population"] = pd.to_numeric(d["population"], errors="coerce")
                d["county_code"] = pd.to_numeric(d["county_code"], errors="coerce").astype("Int64")
                d["fips"] = d["county_code"].astype("string").str.zfill(5)
                # create crude rate 
                note_cols = [c for c in ["notes", "crude_rate"] if c in d.columns]
                if note_cols:
                    note_str = d[note_cols].astype("string").fillna("").agg(" ".join, axis=1).str.lower()
                    d["is_unreliable"] = note_str.str.contains("unreliable|suppressed").astype("Int64")
                else:
                    d["is_unreliable"] = 0
                    # put all into one dataframe 
                mort_raw_parts.append(
                    d[
                        [
                            "year",
                            "state",
                            "county",
                            "fips",
                            "race",
                            "sex",
                            "deaths",
                            "population",
                            "is_unreliable",
                            "source_file",
                        ]
                    ].copy()
                )

            if mort_raw_parts:
                mort_raw = pd.concat(mort_raw_parts, ignore_index=True, sort=False)
                _add_fill("mort_disagg_raw_extension", mort_raw)
                _add_key_qa("mort_disagg_raw_extension", mort_raw, ["fips", "year", "race", "sex"])

                mort_keys = ["fips", "year", "state", "county", "race", "sex"]
                mort_disagg = (
                    mort_raw.groupby(mort_keys, as_index=False)
                    .agg(
                        deaths=("deaths", "sum"),
                        population=("population", "sum"),
                        unreliable_n=("is_unreliable", "sum"),
                    )
                )
                mort_disagg["crude_rate_per_100k"] = np.where(
                    mort_disagg["population"] > 0, # make sure that we're not calculating rates on missing populations
                    mort_disagg["deaths"] / mort_disagg["population"] * 100000,
                    np.nan,
                )
                # sum up the totals - although note that this is NOT going to sum to the AGG b/c it does not capture all mutually exclusive disaggregated groups 
                totals = mort_disagg.groupby(["fips", "year"])["deaths"].transform("sum")
                mort_disagg["pct_of_county_year_deaths"] = np.where(totals > 0, mort_disagg["deaths"] / totals * 100, np.nan)
                _add_fill("mort_disagg_extension_final", mort_disagg)
                _add_key_qa("mort_disagg_extension_final", mort_disagg, ["fips", "year", "race", "sex"])
                # create output file
                mort_disagg_out = os.path.join(local_ext_dir, f"{today_str}_mortality_sex_race_disagg_extension.csv")
                mort_disagg.to_csv(mort_disagg_out, index=False)
                print("Saved local extension:", mort_disagg_out)

                # MORT TOTALS WILL BE FOR QA ON THE DATASET BUT SHOULD NOT EQUAL DISAGGREGATED PANEL 
                mort_totals = (
                    mort_disagg.groupby(["fips", "year"], as_index=False)
                    .agg(
                        mortality_total_deaths=("deaths", "sum"),
                        mortality_total_population=("population", "sum"),
                        mortality_unreliable_count=("unreliable_n", "sum"),
                    )
                )
                mort_totals["mortality_crude_rate_per_100k"] = np.where(
                    mort_totals["mortality_total_population"] > 0,
                    mort_totals["mortality_total_deaths"] / mort_totals["mortality_total_population"] * 100000,
                    np.nan,
                )
                mort_geo = (
                    mort_disagg.groupby(["fips", "year"], as_index=False)
                    .agg(
                        mortality_state=("state", _first_notna),
                        mortality_county=("county", _first_notna),
                    )
                )
                mort_panel_ext = mort_geo.merge(mort_totals, on=["fips", "year"], how="left")
                mort_panel_out = os.path.join(local_ext_dir, f"{today_str}_mortality_county_year_extension.csv")
                mort_panel_ext.to_csv(mort_panel_out, index=False)
                print("Saved local extension:", mort_panel_out)
            else:
                print("Disaggregated CDC extension requested, but no valid disaggregated rows parsed.")
    else:
        print("Skipping optional disaggregated CDC extension (RUN_CDC_DISAGG_EXTENSION=0).")

    _finalize_qa()


# =============================================================================
# PART 2 -- ANNUAL CONTROLS (SAIPE, LAUS, SAHIE, PEP), dated by true data year
# =============================================================================
RAW   = os.path.join(db_data, "raw", "annual_controls")
YEARS = range(2000, 2024)
UA    = "Mozilla/5.0 (academic research client)"
SKIP_DOWNLOAD = False   # set by --skip-download in main(): fetch() then serves cached files only

def fetch(url, dest, tries=3):
    """Download once and cache. Returns dest, or None if unavailable."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    if SKIP_DOWNLOAD:
        print(f"    MISS {os.path.basename(dest)}: not cached (--skip-download)")
        return None

    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r, open(dest,"wb") as f:
                f.write(r.read())
            return dest
        except Exception as e:
            if k == tries-1:
                print(f"    MISS {os.path.basename(dest)}: {type(e).__name__}")
                return None
            time.sleep(2)

# =============================================================================
# SAIPE -- median household income + children in poverty
# =============================================================================
# Layout differs by vintage: fixed-width .dat before 2003, Excel after. Both are
# state+county files where county_fips == 0 marks the state total row, which is
# dropped. Column POSITIONS in the .dat files follow the published layout.
def saipe():
    print("SAIPE")
    frames=[]
    for y in YEARS:
        yy=f"{y%100:02d}"
        base=f"https://www2.census.gov/programs-surveys/saipe/datasets/{y}/{y}-state-and-county"
        got=None
        for ext in ("xls","dat","txt"):
            url=f"{base}/est{yy}all.{ext}"
            dest=os.path.join(RAW,"saipe",f"est{yy}all.{ext}")
            if fetch(url,dest): got=(dest,ext); break
        if not got:
            print(f"  {y}: unavailable"); continue
        dest,ext=got
        try:
            # Column positions verified against est10all.xls header row 2:
            #   0 State FIPS | 1 County FIPS | 7 Poverty Percent All Ages
            #   10 Poverty ESTIMATE Under 18 (a COUNT) | 13 Poverty PERCENT Under 18
            #   22 Median Household Income
            # Col 13, not 10 -- col 10 is the count and runs into the hundreds of
            # thousands, which is what made the first run's means nonsensical.
            if ext=="xls":
                d=pd.read_excel(dest,header=None,skiprows=3)
                st=d.iloc[:,0]; ct=d.iloc[:,1]
                mhi=pd.to_numeric(d.iloc[:,22],errors="coerce")
                cip=pd.to_numeric(d.iloc[:,13],errors="coerce")
            else:
                # Fixed-width .dat, 2000-2002. Whitespace-delimited in practice:
                # state, county, then the same field order as the .xls layout.
                raw=[l for l in open(dest,"r",errors="ignore").read().splitlines() if l.strip()]
                parts=[l.split() for l in raw]
                st =pd.Series([q[0] if len(q)>0 else None for q in parts])
                ct =pd.Series([q[1] if len(q)>1 else None for q in parts])
                cip=pd.to_numeric(pd.Series([q[11] if len(q)>11 else None for q in parts]),errors="coerce")
                mhi=pd.to_numeric(pd.Series([q[22] if len(q)>22 else None for q in parts]),errors="coerce")
            # FIPS: both files carry these as numbers or space-padded strings, so
            # coerce to int before zero-padding. zfill on "  1" leaves the spaces.
            st=pd.to_numeric(st,errors="coerce"); ct=pd.to_numeric(ct,errors="coerce")
            f=pd.DataFrame({"state":st,"county":ct,
                            "saipe_median_hh_income":mhi,"saipe_pct_children_poverty":cip})
            f=f.dropna(subset=["state","county"])
            f=f[f.county>0]                      # county 0 is the state total row
            f["fips"]=(f.state.astype(int).astype(str).str.zfill(2)
                       + f.county.astype(int).astype(str).str.zfill(3))
            f["year"]=y
            frames.append(f[["fips","year","saipe_median_hh_income","saipe_pct_children_poverty"]])
            print(f"  {y}: {len(f):,} counties")
        except Exception as e:
            print(f"  {y}: parse failed ({type(e).__name__}: {e})")
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

# =============================================================================
# LAUS -- county unemployment rate
# =============================================================================
# www.bls.gov returns 403 to scripted clients; download.bls.gov serves the same
# data. la.data.64.County is the full county time series; la.area maps the LAUS
# area code to a state+county FIPS. Annual averages are period M13.
def laus():
    print("LAUS")
    d1=fetch("https://download.bls.gov/pub/time.series/la/la.data.64.County",
             os.path.join(RAW,"laus","la.data.64.County"))
    d2=fetch("https://download.bls.gov/pub/time.series/la/la.series",
             os.path.join(RAW,"laus","la.series"))
    if not d1 or not d2:
        print("  unavailable"); return pd.DataFrame()
    dat=pd.read_csv(d1,sep="\t",dtype=str).rename(columns=lambda c:c.strip())
    ser=pd.read_csv(d2,sep="\t",dtype=str).rename(columns=lambda c:c.strip())
    dat["series_id"]=dat["series_id"].str.strip()
    ser["series_id"]=ser["series_id"].str.strip()
    # measure_code 03 = unemployment rate
    ser=ser[ser["measure_code"].str.strip()=="03"]
    m=dat.merge(ser[["series_id","area_code"]],on="series_id",how="inner")
    m=m[m["period"].str.strip()=="M13"]                      # annual average
    m["year"]=pd.to_numeric(m["year"],errors="coerce")
    m=m[m.year.between(min(YEARS),max(YEARS))]
    # LAUS county area codes look like CN{SSCCC}0000000000
    m["fips"]=m["area_code"].str.extract(r"^CN(\d{5})")
    m=m.dropna(subset=["fips"])
    m["laus_unemployment_rate"]=pd.to_numeric(m["value"].str.strip(),errors="coerce")
    out=m[["fips","year","laus_unemployment_rate"]].drop_duplicates(["fips","year"])
    print(f"  {len(out):,} county-years, {out.year.min()}-{out.year.max()}")
    return out

# =============================================================================
# SAHIE -- uninsured, adults 18-64
# =============================================================================
def sahie():
    print("SAHIE")
    frames=[]
    for y in YEARS:
        url=f"https://www2.census.gov/programs-surveys/sahie/datasets/time-series/estimates-acs/sahie-{y}-csv.zip"
        dest=os.path.join(RAW,"sahie",f"sahie-{y}.zip")
        if not fetch(url,dest): continue
        try:
            # The preamble length varies by vintage (79 lines in some years, 84-85 in
            # others), so locate the header row rather than hardcoding skiprows.
            with zipfile.ZipFile(dest) as z:
                name=[n for n in z.namelist() if n.lower().endswith(".csv")][0]
                lines=z.open(name).read().decode("latin-1").splitlines()
            # Must be the comma-delimited header, not the "PRIMARY KEY: year version
            # statefips countyfips ..." documentation line in the preamble, which also
            # contains both words and sits ~17 lines earlier.
            hdr=next(i for i,l in enumerate(lines)
                     if l.lower().lstrip().startswith("year,") and "countyfips" in l.lower())
            d=pd.read_csv(io.StringIO("\n".join(lines[hdr:])),dtype=str,low_memory=False)
            d.columns=[c.strip().lower() for c in d.columns]
            # agecat 1 = 18-64; all races, both sexes, all incomes
            for _c in ["agecat","racecat","sexcat","iprcat","statefips","countyfips"]:
                d[_c]=d[_c].astype(str).str.strip()
            # agecat 1 = ages 18-64; all races, both sexes, all income levels
            d=d[(d.agecat=="1")&(d.racecat=="0")&(d.sexcat=="0")&(d.iprcat=="0")]
            d=d[pd.to_numeric(d.countyfips,errors="coerce").fillna(0)>0]
            f=pd.DataFrame({"fips":d.statefips.str.zfill(2)+d.countyfips.str.zfill(3),
                            "year":y,
                            "sahie_pct_uninsured_18_64":pd.to_numeric(d.pctui,errors="coerce")})
            frames.append(f); print(f"  {y}: {len(f):,} counties")
        except Exception as e:
            print(f"  {y}: parse failed ({type(e).__name__}: {e})")
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()



# ---------------------------------------------------------------------------
# FINAL NAMING. The new series take over the EXACT column names the CHR versions
# used, so that no downstream code has to change. script0f drops the CHR columns
# before merging these in (see REPLACES below).
#
# UNITS: the panel's `*_per100k` columns store PERCENT x 1000 -- 8.9% is stored
# as 8900. Verified against the existing CHR columns (unemployment mean 6,363.6
# = 6.36%). SAIPE/LAUS/SAHIE publish plain percentages, so they are scaled by
# 1000 here. `median_household_income` is plain dollars in both and is not scaled.
# ---------------------------------------------------------------------------
REPLACES = {
    "saipe_median_hh_income":        ("median_household_income",      1),
    "saipe_pct_children_poverty":    ("children_in_poverty_per100k",  1000),
    "laus_unemployment_rate":        ("unemployment_per100k",         1000),
    "sahie_pct_uninsured_18_64":     ("uninsured_adults_per100k",     1000),
    # PEP shares are fractions (0-1); the panel stores these as plain percent
    # (mean %_hispanic ~0.08 in the panel), so scale by 1.
    "pep_pct_female":                ("%_female",                              1),
    "pep_pct_hispanic":              ("%_hispanic",                            1),
    "pep_pct_asian":                 ("%_asian",                               1),
    "pep_pct_nhpi":                  ("%_native_hawaiian/other_pacific_islander", 1),
    "pep_pct_black":                 ("%_black",                               1),
    "pep_pct_aian":                  ("%_american_indian_alaskan_native_pep",  1),
    "pep_pct_asian_aian_nhpi":       ("%_asian_aian_nhpi",                     1),
    "pep_pct_65_plus":               ("%_65_and_older",                        1),
    "pep_pct_under_18":              ("%_below_18_years_of_age",               1),
}

def apply_final_names(out):
    """Rename to the panel's own column names and match its units. Keeps the
    source-named originals alongside so the provenance is visible in the file."""
    for src,(dst,scale) in REPLACES.items():
        if src in out.columns:
            out[dst] = out[src] * scale
    return out

# =============================================================================
# PEP -- county demographics, dated by TRUE data year
# =============================================================================
# Replaces the six CHR demographic shares, which are dated by CHR's RELEASE year
# and therefore carry a 2-year lag. Census PEP publishes county population by
# age x sex x race x Hispanic origin ANNUALLY for every county, with no
# population floor, so the lag is removable outright.
#
# THREE VINTAGES, each indexing YEAR as a code rather than a calendar year, and
# each indexing it differently. Verified empirically against known Alabama state
# totals (2010, 2012, 2015, 2018 all matched within 0.4%) rather than taken from
# the documentation:
#
#   co-est00int-alldata-{SS}   2000-2010   code 2..12 -> 2000..2010
#                              (code 1 = 4/1/2000 census, 13 = 4/1/2010 census)
#   CC-EST2020-ALLDATA-{SS}    2010-2020   code 3..13 -> 2010..2020
#                              (code 1 = 4/1/2010 census, 2 = estimates base)
#   cc-est2024-alldata         2020-2024   code 2..6  -> 2020..2024
#                              (code 1 = 4/1/2020 estimates base)
#
# VINTAGE OVERLAP: the 2010-2020 series drifted ~2% BELOW the 2020 census count
# by its final year and was never rebased, so the two vintages disagree at the
# seam. Per team decision, the NEWER vintage wins wherever they overlap. Because
# every variable here is a SHARE, a level error largely cancels between numerator
# and denominator -- but the 2010 and 2020 seams are still checked for jumps
# below, since a spurious break there would read as an event-study result.
#
# AGEGRP: 0 = all ages, 1..18 = 0-4, 5-9, ... 85+.
#   under 18  = AGEGRP 1,2,3 plus 3/5 of AGEGRP 4 (15-19) -- see note below
#   65+       = AGEGRP 14..18
# The 15-19 band straddles 18, so "% below 18" cannot be formed exactly from
# 5-year bands. CHR's own measure uses single-year ages. We take AGEGRP 1-3
# (0-14) plus 3/5 of the 15-19 band as a linear approximation, and record it.
PEP_STATES = [f"{i:02d}" for i in list(range(1,57))]
def pep():
    print("PEP demographics")
    VINT = [
        ("2000-2010", "https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-alldata-{ss}.csv", 1998, range(2,13)),
        ("2010-2020", "https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/asrh/CC-EST2020-ALLDATA-{ss}.csv", 2007, range(3,14)),
        ("2020-2024", "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv", 2018, range(2,7)),
    ]
    frames=[]
    for vname, tmpl, offset, codes in VINT:
        parts=[]
        targets = ["NATIONAL"] if "{ss}" not in tmpl else PEP_STATES
        for ss in targets:
            url = tmpl if ss=="NATIONAL" else tmpl.format(ss=ss)
            dest = os.path.join(RAW,"pep", f"{vname}_{ss}.csv")
            if not fetch(url,dest): continue
            try:
                d=pd.read_csv(dest,encoding="latin-1",low_memory=False)
            except Exception as e:
                print(f"    {vname} {ss}: {type(e).__name__}"); continue
            d.columns=[c.upper().strip() for c in d.columns]
            if "AGEGRP" not in d.columns: continue
            d=d[d["YEAR"].isin(list(codes))]
            if "SUMLEV" in d.columns: d=d[d["SUMLEV"]==50]
            parts.append(d)
        if not parts:
            print(f"  {vname}: no files"); continue
        D=pd.concat(parts,ignore_index=True)
        D["year"]=D["YEAR"]+offset
        D["fips"]=(D["STATE"].astype(int).astype(str).str.zfill(2)
                   + D["COUNTY"].astype(int).astype(str).str.zfill(3))
        # AGEGRP CODING DIFFERS BY VINTAGE -- caught by the seam check below.
        #   co-est00int (2000-2010): AGEGRP 0 = AGE 0 (infants), 99 = TOTAL,
        #                            1 = 1-4, 2 = 5-9, 3 = 10-14, 4 = 15-19 ...
        #   CC-EST2020 / cc-est2024: AGEGRP 0 = TOTAL,
        #                            1 = 0-4, 2 = 5-9, 3 = 10-14, 4 = 15-19 ...
        # Using 0 as the denominator for the 2000-2010 files divided by the infant
        # count instead of the population -- ~72x too small, and inf where zero.
        # 65+ is AGEGRP 14-18 in BOTH codings. Under-18 differs: the older vintage
        # needs AGEGRP 0 included, the newer does not.
        TOTAL_CODE = 99 if D["AGEGRP"].max() == 99 else 0
        U18_LOW    = 0  if TOTAL_CODE == 99 else 1
        tot = D[D.AGEGRP==TOTAL_CODE].set_index(["fips","year"])
        num = lambda f,c: pd.to_numeric(f[c],errors="coerce")
        base = num(tot,"TOT_POP")
        out = pd.DataFrame(index=tot.index)
        out["pep_pct_female"]   = num(tot,"TOT_FEMALE")/base
        out["pep_pct_hispanic"] = (num(tot,"H_MALE")+num(tot,"H_FEMALE"))/base
        out["pep_pct_asian"]    = (num(tot,"AA_MALE")+num(tot,"AA_FEMALE"))/base
        out["pep_pct_nhpi"]     = (num(tot,"NA_MALE")+num(tot,"NA_FEMALE"))/base
        # Added 2026-10-12 (Sept 30 feedback): Black and AIAN shares from the same
        # annual PEP files, so the race block is dated by true data year. CHR's
        # %_non-hispanic_african_american / %_american_indian_and_alaskan_native
        # cover 2011-2019 only. PEP race categories are "alone" and include
        # Hispanic members of each race.
        out["pep_pct_black"]    = (num(tot,"BA_MALE")+num(tot,"BA_FEMALE"))/base
        out["pep_pct_aian"]     = (num(tot,"IA_MALE")+num(tot,"IA_FEMALE"))/base
        # Combined group: Asian + AIAN + NHPI. NHPI alone has no usable variation in
        # non-metro counties, and Asian and AIAN shares are small; one combined
        # share replaces %_asian in the control set.
        out["pep_pct_asian_aian_nhpi"] = (out["pep_pct_asian"] + out["pep_pct_aian"]
                                          + out["pep_pct_nhpi"])
        g = D[D.AGEGRP.between(0,18)].copy()
        g["_pop"]=pd.to_numeric(g["TOT_POP"],errors="coerce")
        o65 = g[g.AGEGRP.between(14,18)].groupby(["fips","year"])["_pop"].sum()
        u15 = g[g.AGEGRP.between(U18_LOW,3)].groupby(["fips","year"])["_pop"].sum()
        b1519 = g[g.AGEGRP==4].groupby(["fips","year"])["_pop"].sum()
        out["pep_pct_65_plus"] = (o65/base).reindex(out.index)
        out["pep_pct_under_18"] = ((u15 + 0.6*b1519)/base).reindex(out.index)
        out=out.reset_index(); out["_vintage"]=vname
        frames.append(out)
        print(f"  {vname}: {out.fips.nunique():,} counties, {int(out.year.min())}-{int(out.year.max())}")
    if not frames: return pd.DataFrame()
    P=pd.concat(frames,ignore_index=True)
    # NEWER VINTAGE WINS on overlap: sort so the latest vintage lands last, keep last.
    order={v:i for i,(v,_,_,_) in enumerate(VINT)}
    P["_ord"]=P["_vintage"].map(order)
    P=P.sort_values(["fips","year","_ord"]).drop_duplicates(["fips","year"],keep="last")
    P=P.drop(columns=["_ord","_vintage"])
    # seam check -- a jump at 2010 or 2020 would read as a spurious event
    for seam in (2010,2020):
        a=P[P.year==seam-1].set_index("fips")["pep_pct_65_plus"]
        b=P[P.year==seam].set_index("fips")["pep_pct_65_plus"]
        j=(b-a.reindex(b.index)).abs().median()
        print(f"  seam {seam-1}->{seam}: median |change| in pct_65_plus = {j:.5f}")
    return P


def run_controls():
    warnings.filterwarnings("ignore")      # script0g ran with warnings off; scoped here to Part 2
    for sub in ("saipe", "laus", "sahie", "pep"):
        os.makedirs(os.path.join(RAW, sub), exist_ok=True)
    print("="*72); print("script0c Part 2 -- annual county controls, dated by TRUE data year"); print("="*72)
    S,L,H,P = saipe(), laus(), sahie(), pep()
    out=None
    for f in (S,L,H,P):
        if f is None or f.empty: continue
        out = f if out is None else out.merge(f,on=["fips","year"],how="outer")
    if out is None or out.empty:
        raise SystemExit("nothing downloaded")
    out["fips"]=out["fips"].astype(str).str.zfill(5)
    out=apply_final_names(out)
    out=out.sort_values(["fips","year"])
    p=os.path.join(outf,f"{today_str}_annual_controls_county_year.csv")
    out.to_csv(p,index=False)
    print("\n"+"="*72)
    print(f"rows {len(out):,} | counties {out.fips.nunique():,} | years {out.year.min()}-{out.year.max()}")
    for c in out.columns:
        if c in ("fips","year"): continue
        print(f"  {c:32s} coverage {out[c].notna().mean():6.1%}")
    print(f"\nSaved: {p}")


def main():
    global SKIP_DOWNLOAD
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--skip-download", action="store_true",
                    help="Part 2: use only files already cached in raw/annual_controls (no network)")
    ap.add_argument("--only", choices=["health", "controls"],
                    help="run only Part 1 (health) or Part 2 (controls); default is both")
    args = ap.parse_args()
    SKIP_DOWNLOAD = args.skip_download
    if args.only in (None, "health"):
        run_health()
    if args.only in (None, "controls"):
        run_controls()


if __name__ == "__main__":
    main()
