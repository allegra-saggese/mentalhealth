#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:13:23 2025

Health pipeline:
1) County Health Rankings (mental health) consolidation
2) CDC mortality (demographic tranche CSVs) consolidation
3) Merge on fips-year
4) Stage-by-stage QA artifacts for missingness and key integrity
"""

import re

from functions import *
from packages import *


# folders
inf = os.path.join(db_data, "raw")
outf = os.path.join(db_data, "clean")
qa_dir = os.path.join(interim, "qa-health")
os.makedirs(qa_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")


# ---------- QA helpers ----------
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


def _first_notna(s):
    idx = s.first_valid_index()
    return s.loc[idx] if idx is not None else pd.NA


def _normalize_fips(series):
    s = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(5)
    )
    return s


# ---------- PART 1: Mental health ----------
raw_mh = os.path.join(inf, "mental")
mh_files = sorted(glob.glob(os.path.join(raw_mh, "*.csv")))
if not mh_files:
    raise FileNotFoundError(f"No mental health files found in {raw_mh}")

mh_file_inventory = []
mh_dfs = []
year_re = re.compile(r"(19|20)\d{2}(?=\.[A-Za-z0-9]+$)")

for p in mh_files:
    d = pd.read_csv(p, low_memory=False)
    base = os.path.basename(p)
    m = year_re.search(base)
    if not m:
        raise ValueError(f"Could not infer year from filename: {base}")
    yr = int(m.group(0))

    d.columns = _clean_mh_cols(d.columns)
    d = _dedupe_columns_keep_most_complete(d)
    d["year"] = yr

    if "5-digit_fips_code" not in d.columns:
        alt = [c for c in d.columns if "fips" in c]
        if alt:
            d = d.rename(columns={alt[0]: "5-digit_fips_code"})
        else:
            raise KeyError(f"Missing FIPS column in {base}")

    d["5-digit_fips_code"] = _normalize_fips(d["5-digit_fips_code"])
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
    mh_dfs.append(d)

pd.DataFrame(mh_file_inventory).sort_values("year").to_csv(
    os.path.join(qa_dir, f"{today_str}_qa_health_mh_file_inventory.csv"), index=False
)

_add_fill("mh_raw_union_preselect", pd.concat(mh_dfs, ignore_index=True, sort=False))

# Column presence and fill diagnostics
all_cols = sorted(set().union(*[set(d.columns) for d in mh_dfs]))
presence_rows = []
for col in all_cols:
    dfs_with_col = sum(col in d.columns for d in mh_dfs)
    fill_vals = [d[col].notna().mean() * 100 for d in mh_dfs if col in d.columns]
    presence_rows.append(
        {
            "column": col,
            "dfs_with_col": dfs_with_col,
            "avg_fill_pct_when_present": float(np.mean(fill_vals)) if fill_vals else np.nan,
        }
    )
presence_df = pd.DataFrame(presence_rows).sort_values(
    ["dfs_with_col", "avg_fill_pct_when_present", "column"], ascending=[False, False, True]
)
presence_path = os.path.join(qa_dir, f"{today_str}_qa_health_mh_column_presence.csv")
presence_df.to_csv(presence_path, index=False)
print("Saved QA:", presence_path)

# Keep keys + stable columns
majority_cols = presence_df.loc[presence_df["dfs_with_col"] >= 11, "column"].tolist()
high_fill_cols = presence_df.loc[
    (presence_df["dfs_with_col"] >= 8) & (presence_df["avg_fill_pct_when_present"] >= 80),
    "column",
].tolist()
selected_cols = list(dict.fromkeys(["5-digit_fips_code", "year"] + majority_cols + high_fill_cols))

mh_selected = []
for d in mh_dfs:
    keep = [c for c in selected_cols if c in d.columns]
    mh_selected.append(d[keep].copy())

mh_combined = pd.concat(mh_selected, ignore_index=True, sort=False)
_add_fill("mh_selected_concat", mh_combined)
_add_key_qa("mh_selected_concat", mh_combined, ["5-digit_fips_code", "year"])

if mh_combined.duplicated(["5-digit_fips_code", "year"]).any():
    mh_combined = (
        mh_combined.groupby(["5-digit_fips_code", "year"], as_index=False).agg(_first_notna)
    )

mh_combined = mh_combined.rename(columns={"5-digit_fips_code": "fips"})
mh_combined["fips"] = _normalize_fips(mh_combined["fips"])
mh_combined["year"] = pd.to_numeric(mh_combined["year"], errors="coerce").astype("Int64")
mh_combined = mh_combined.sort_values(["fips", "year"]).reset_index(drop=True)

_add_fill("mh_final", mh_combined)
_add_key_qa("mh_final", mh_combined, ["fips", "year"])

mh_out = os.path.join(outf, f"{today_str}_mentalhealthrank_full.csv")
mh_combined.to_csv(mh_out, index=False)
print("Saved:", mh_out)


# ---------- PART 2: CDC mortality ----------
raw_cdc = os.path.join(inf, "cdc")
cdc_csv_files = sorted(glob.glob(os.path.join(raw_cdc, "*.csv")))
if not cdc_csv_files:
    raise FileNotFoundError(f"No CDC tranche CSV files found in {raw_cdc}")

mort_raw_parts = []
for p in cdc_csv_files:
    try:
        d = pd.read_csv(p, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        d = pd.read_csv(p, low_memory=False, encoding="latin1")

    d.columns = (
        pd.Index(d.columns)
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    req = ["state", "county", "county_code", "year", "deaths", "population", "race"]
    missing = [c for c in req if c not in d.columns]
    if missing:
        raise KeyError(f"Missing required mortality columns in {os.path.basename(p)}: {missing}")

    if "sex" not in d.columns:
        lower_name = os.path.basename(p).lower()
        d["sex"] = "male" if "male" in lower_name else ("female" if "female" in lower_name else pd.NA)

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

    if "crude_rate" in d.columns:
        crude = (
            d["crude_rate"]
            .astype("string")
            .str.replace("%", "", regex=False)
            .str.replace(r"\(.*?\)", "", regex=True)
            .str.strip()
        )
        d["crude_rate_raw"] = pd.to_numeric(crude, errors="coerce")
    else:
        d["crude_rate_raw"] = pd.NA

    note_cols = [c for c in ["notes", "crude_rate"] if c in d.columns]
    if note_cols:
        note_str = d[note_cols].astype("string").fillna("").agg(" ".join, axis=1).str.lower()
        d["is_unreliable"] = note_str.str.contains("unreliable|suppressed").astype("Int64")
    else:
        d["is_unreliable"] = 0

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
                "crude_rate_raw",
                "is_unreliable",
                "source_file",
            ]
        ].copy()
    )

mort_raw = pd.concat(mort_raw_parts, ignore_index=True, sort=False)
_add_fill("mort_raw_concat", mort_raw)
_add_key_qa("mort_raw_concat", mort_raw, ["fips", "year", "race", "sex"])

# collapse to demographic panel
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
    mort_disagg["population"] > 0,
    mort_disagg["deaths"] / mort_disagg["population"] * 100000,
    np.nan,
)
totals = mort_disagg.groupby(["fips", "year"])["deaths"].transform("sum")
mort_disagg["pct_of_county_year_deaths"] = np.where(totals > 0, mort_disagg["deaths"] / totals * 100, np.nan)

_add_fill("mort_disagg_final", mort_disagg)
_add_key_qa("mort_disagg_final", mort_disagg, ["fips", "year", "race", "sex"])

mort_disagg_out = os.path.join(outf, f"{today_str}_mortality_sex_race_disagg.csv")
mort_disagg.to_csv(mort_disagg_out, index=False)
print("Saved:", mort_disagg_out)

# county-year mortality panel
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

mort_disagg["race_sex_key"] = (
    mort_disagg["race"].astype("string").str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    + "_"
    + mort_disagg["sex"].astype("string").str.lower().str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
)

deaths_wide = mort_disagg.pivot_table(
    index=["fips", "year"], columns="race_sex_key", values="deaths", aggfunc="sum"
).reset_index()
deaths_wide = deaths_wide.rename(columns={c: f"mort_deaths_{c}" for c in deaths_wide.columns if c not in ["fips", "year"]})

pop_wide = mort_disagg.pivot_table(
    index=["fips", "year"], columns="race_sex_key", values="population", aggfunc="sum"
).reset_index()
pop_wide = pop_wide.rename(columns={c: f"mort_pop_{c}" for c in pop_wide.columns if c not in ["fips", "year"]})

mort_panel = mort_totals.merge(deaths_wide, on=["fips", "year"], how="left").merge(pop_wide, on=["fips", "year"], how="left")
_add_fill("mort_county_year_panel", mort_panel)
_add_key_qa("mort_county_year_panel", mort_panel, ["fips", "year"])


# ---------- PART 3: Merge health + mortality ----------
for d in [mh_combined, mort_panel]:
    d["fips"] = _normalize_fips(d["fips"])
    d["year"] = pd.to_numeric(d["year"], errors="coerce").astype("Int64")

if mh_combined.duplicated(["fips", "year"]).any():
    mh_combined = mh_combined.groupby(["fips", "year"], as_index=False).agg(_first_notna)
if mort_panel.duplicated(["fips", "year"]).any():
    mort_panel = mort_panel.groupby(["fips", "year"], as_index=False).agg(_first_notna)

wide_mh_full = pd.merge(
    mh_combined,
    mort_panel,
    on=["fips", "year"],
    how="outer",
    sort=True,
    validate="1:1",
)

# overlap diagnostics
wide_mh_full["has_mh"] = wide_mh_full[[c for c in mh_combined.columns if c not in ["fips", "year"]]].notna().any(axis=1)
wide_mh_full["has_mort"] = wide_mh_full["mortality_total_deaths"].notna()

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

_add_fill("mh_mortality_merged_final", wide_mh_full)
_add_key_qa("mh_mortality_merged_final", wide_mh_full, ["fips", "year"])

final_out = os.path.join(outf, f"{today_str}_mh_mortality_fips_yr.csv")
wide_mh_full.to_csv(final_out, index=False)
print("Saved:", final_out)

_finalize_qa()
