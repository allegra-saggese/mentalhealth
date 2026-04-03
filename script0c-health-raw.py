#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:13:23 2025

Health pipeline (current behavior):
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

Notes:
- Optional extension is controlled by env var RUN_CDC_DISAGG_EXTENSION
  (set to 1/true/yes/y to enable).
- Disaggregated extension outputs are NOT used in the main county-year panel.
"""

# ----------------------- SET UP PART 1 : IMPORT KEY PATHS  -------------------- -#

import re
from functions import *
from packages import *


# folders
inf = os.path.join(db_data, "raw")
outf = os.path.join(db_data, "clean")
qa_dir = os.path.join(interim, "qa-health")
local_ext_dir = os.path.join(interim, "health-disagg-extension")
os.makedirs(qa_dir, exist_ok=True)
os.makedirs(local_ext_dir, exist_ok=True)

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
    fill_path = os.path.join(qa_dir, f"{today_str}_qa_health_stage_fill.csv")
    key_path = os.path.join(qa_dir, f"{today_str}_qa_health_stage_keys.csv")
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
    os.path.join(qa_dir, f"{today_str}_qa_health_mh_file_inventory.csv"), index=False
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
presence_path = os.path.join(qa_dir, f"{today_str}_qa_health_mh_column_presence.csv")
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
cdc_inv_out = os.path.join(qa_dir, f"{today_str}_qa_cdc_deathsofdespair_file_inventory.csv")
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
cdc_by_year_out = os.path.join(qa_dir, f"{today_str}_qa_cdc_deathsofdespair_by_year.csv")
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
cdc_key_out = os.path.join(qa_dir, f"{today_str}_qa_cdc_deathsofdespair_key_check.csv")
cdc_key_check.to_csv(cdc_key_out, index=False)
print("Saved QA:", cdc_key_out)


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
overlap_path = os.path.join(qa_dir, f"{today_str}_qa_health_overlap_by_year.csv")
overlap_df.to_csv(overlap_path, index=False)
print("Saved QA:", overlap_path)

_add_fill("mh_cdc_merged_final", wide_mh_full)
_add_key_qa("mh_cdc_merged_final", wide_mh_full, ["fips", "year"])

final_out = os.path.join(outf, f"{today_str}_mh_mortality_fips_yr.csv")
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
