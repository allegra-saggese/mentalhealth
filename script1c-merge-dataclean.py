#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 01:10:00 2025

Builds one county-year merged panel from the latest version of each clean file.
Filtering rule: keep only (fips, year) where rural-key non_large_metro == 1.

Quick purpose:
- Finds the latest clean outputs by descriptor.
- Normalizes each source to county-year (`fips, year`) and merges them.
- Applies the rural-key filter used in the project’s main analysis panel.
- Pulls county CDC mortality directly from script0c output
  (`*_cdc_county_year_deathsofdespair.csv`) via descriptor matching.
- Writes full merged panel + standard year-range slices.
"""

from packages import *
from functions import *
import re


# Directories
clean_dir = os.path.join(db_data, "clean")
merged_dir = os.path.join(db_data, "merged")

# Merge configuration
BASE_DESCRIPTOR = "cafo_ops_by_size_compact"
MERGE_DESCRIPTORS = {
    "cdc_county_year_deathsofdespair",
    "crime_fips_level_final",
    "fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "mentalhealthrank_full",
    "population_full",
}
RURAL_DESCRIPTOR_HINT = "rural-key"
CAFO_COMMODITIES = ("cattle", "hogs", "chickens")


def _ensure_key(df):
    """
    Ensure df has normalized fips/year key columns.
    """
    try:
        df = normalize_panel_key(df, dropna=True)
    except KeyError:
        return None
    return df


def _safe_descriptor_name(descriptor):
    return re.sub(r"[^a-z0-9]+", "_", descriptor.lower()).strip("_")


def _reduce_to_county_year(df, descriptor):
    """
    Collapse to one row per (fips, year):
      - numeric-like columns: sum
      - text columns: first observed value
      - add n_rows source count
    """
    non_key = [c for c in df.columns if c not in ("fips", "year")]
    if not non_key:
        out = df[["fips", "year"]].drop_duplicates().copy()
        out["n_rows"] = 1
        non_key = ["n_rows"]
    else:
        work = df[["fips", "year"]].copy()
        numeric_cols = []
        text_cols = []

        for c in non_key:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                work[c] = s
                numeric_cols.append(c)
                continue

            s_str = s.astype("string").str.replace(",", "", regex=False).str.strip()
            s_num = pd.to_numeric(s_str, errors="coerce")
            non_null = s_str.notna().sum()
            parse_rate = (s_num.notna().sum() / non_null) if non_null else 0

            if parse_rate >= 0.8 and s_num.notna().any():
                work[c] = s_num
                numeric_cols.append(c)
            else:
                work[c] = s.astype("string")
                text_cols.append(c)

        # Preserve all-missing groups as NA instead of coercing to 0.
        agg = {c: (lambda s: s.sum(min_count=1)) for c in numeric_cols}
        agg.update({c: "first" for c in text_cols})
        out = (
            work.groupby(["fips", "year"], as_index=False)
            .agg(agg)
        )
        n_rows = df.groupby(["fips", "year"]).size().rename("n_rows").reset_index()
        out = out.merge(n_rows, on=["fips", "year"], how="left")
        non_key = [c for c in out.columns if c not in ("fips", "year")]

    tag = _safe_descriptor_name(descriptor)
    out = out.rename(columns={c: f"{c}_{tag}" for c in non_key})
    return out


def _read_filter_reduce(path, descriptor, allowed_keys):
    try:
        df = read_and_prepare(path)
    except Exception as e:
        print(f"Skip {descriptor}: failed to read ({e})")
        return None

    df = _ensure_key(df)
    if df is None:
        print(f"Skip {descriptor}: no fips/year key columns")
        return None

    # runtime reduction: filter to rural key immediately
    df = df.merge(allowed_keys, on=["fips", "year"], how="inner")
    if df.empty:
        print(f"Skip {descriptor}: no rows after rural-key filter")
        return None

    out = _reduce_to_county_year(df, descriptor)
    print(f"Use {descriptor}: {out.shape}")
    return out


def _build_cafo_animal_size_panel(path, allowed_keys):
    """
    Build county-year CAFO animal x size columns from the compact CAFO file.
    Keeps existing CAFO total-size columns intact by adding a separate block.
    """
    try:
        df = read_and_prepare(path)
    except Exception as e:
        print(f"Skip CAFO animal-size block: failed to read base file ({e})")
        return None

    df = _ensure_key(df)
    if df is None:
        print("Skip CAFO animal-size block: missing fips/year")
        return None

    needed = {"commodity_desc", "small", "medium", "large"}
    missing_needed = sorted(list(needed - set(df.columns)))
    if missing_needed:
        print(f"Skip CAFO animal-size block: missing required columns {missing_needed}")
        return None

    df["commodity_desc"] = (
        df["commodity_desc"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    df = df[df["commodity_desc"].isin(CAFO_COMMODITIES)].copy()
    if df.empty:
        print("Skip CAFO animal-size block: no rows after commodity filter")
        return None

    for size_col in ("small", "medium", "large"):
        df[size_col] = pd.to_numeric(df[size_col], errors="coerce")

    # Restrict to rural keys used in merged panel.
    df = df.merge(allowed_keys, on=["fips", "year"], how="inner")
    if df.empty:
        print("Skip CAFO animal-size block: no rows after rural-key filter")
        return None

    grouped = (
        df.groupby(["fips", "year", "commodity_desc"], as_index=False)[["small", "medium", "large"]]
        .sum(min_count=1)
    )

    wide = grouped.pivot_table(
        index=["fips", "year"],
        columns="commodity_desc",
        values=["small", "medium", "large"],
        aggfunc="sum",
    )

    # Flatten pivot columns -> cafo_{commodity}_{size}
    wide.columns = [
        f"cafo_{commodity}_{size}" for size, commodity in wide.columns
    ]
    wide = wide.reset_index()

    # Ensure stable set of output columns even if some commodity is absent.
    animal_size_cols = []
    for commodity in CAFO_COMMODITIES:
        for size in ("small", "medium", "large"):
            col = f"cafo_{commodity}_{size}"
            animal_size_cols.append(col)
            if col not in wide.columns:
                wide[col] = pd.NA

    wide[animal_size_cols] = wide[animal_size_cols].apply(pd.to_numeric, errors="coerce")
    wide["cafo_total_ops_all_animals"] = wide[animal_size_cols].sum(axis=1, min_count=1)
    chickens_cols = [f"cafo_chickens_{s}" for s in ("small", "medium", "large")]
    wide["cafo_total_ops_chickens"] = wide[chickens_cols].sum(axis=1, min_count=1)

    keep_cols = ["fips", "year", *animal_size_cols, "cafo_total_ops_all_animals", "cafo_total_ops_chickens"]
    wide = wide[keep_cols].copy()
    print(f"Use CAFO animal-size block: {wide.shape}")
    return wide


latest = latest_files_by_descriptor(clean_dir)
if not latest:
    raise RuntimeError(f"No supported files found in {clean_dir}")

print("Latest clean files by descriptor:")
for desc, p in sorted(latest.items()):
    print(f" - {desc}: {os.path.basename(p)}")


# Rural-key filter source
rural_candidates = [d for d in latest if RURAL_DESCRIPTOR_HINT in d]
if not rural_candidates:
    raise RuntimeError("Could not find rural-key file in clean folder.")
rural_desc = sorted(rural_candidates)[-1]
rural_path = latest[rural_desc]

rural_df = _ensure_key(read_and_prepare(rural_path))
if rural_df is None:
    raise RuntimeError(f"Rural-key file missing fips/year: {rural_path}")
if "non_large_metro" not in rural_df.columns:
    raise KeyError(f"'non_large_metro' not found in rural-key file: {rural_path}")

rural_df["non_large_metro"] = pd.to_numeric(rural_df["non_large_metro"], errors="coerce").astype("Int64")
allowed_keys = (
    rural_df.loc[rural_df["non_large_metro"] == 1, ["fips", "year"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

if allowed_keys.empty:
    raise RuntimeError("No (fips, year) rows with non_large_metro == 1 in rural-key data.")

print(f"Rural-key filter rows kept (non_large_metro == 1): {len(allowed_keys):,}")


# Target files for this merge run
target_descriptors = {BASE_DESCRIPTOR, *MERGE_DESCRIPTORS}
missing = sorted([d for d in target_descriptors if d not in latest])
if missing:
    raise RuntimeError(f"Missing required clean descriptors: {missing}")

print("Descriptors included in merge:")
for d in sorted(target_descriptors):
    print(f" - {d}: {os.path.basename(latest[d])}")

# Base panel: CAFO compact reduced to one row per fips-year
base = _read_filter_reduce(latest[BASE_DESCRIPTOR], BASE_DESCRIPTOR, allowed_keys)
if base is None or base.empty:
    raise RuntimeError(f"Base dataset {BASE_DESCRIPTOR} is empty after rural-key filter.")

merged_all = allowed_keys.copy()
merged_all["non_large_metro"] = 1
merged_all = merged_all.merge(base, on=["fips", "year"], how="left")

# Add CAFO animal x size block (in addition to legacy total-size columns).
cafo_animal_size = _build_cafo_animal_size_panel(latest[BASE_DESCRIPTOR], allowed_keys)
if cafo_animal_size is not None and not cafo_animal_size.empty:
    merged_all = merged_all.merge(cafo_animal_size, on=["fips", "year"], how="left")


# Merge remaining selected datasets on top of base keys
for descriptor in sorted(MERGE_DESCRIPTORS):
    path = latest[descriptor]
    part = _read_filter_reduce(path, descriptor, allowed_keys)
    if part is None:
        continue

    merged_all = merged_all.merge(part, on=["fips", "year"], how="left")


merged_all = merged_all.sort_values(["fips", "year"]).reset_index(drop=True)
print("Final merged panel shape:", merged_all.shape)


# Export
today_str = date.today().strftime("%Y-%m-%d")
out_name = f"{today_str}_full_merged.csv"
out_path = os.path.join(merged_dir, out_name)
os.makedirs(merged_dir, exist_ok=True)
merged_all.to_csv(out_path, index=False)
print("Saved:", out_path)


# Spliced exports
slice_2005_2010 = merged_all[merged_all["year"].between(2005, 2010, inclusive="both")].copy()
slice_2010_2020 = merged_all[merged_all["year"].between(2010, 2020, inclusive="both")].copy()
slice_census_years = merged_all[merged_all["year"].isin([2002, 2005, 2007, 2012])].copy()

slice_2005_2010_path = os.path.join(merged_dir, f"{today_str}_full_merged_2005_2010.csv")
slice_2010_2020_path = os.path.join(merged_dir, f"{today_str}_full_merged_2010_2020.csv")
slice_census_path = os.path.join(merged_dir, f"{today_str}_full_merged_census_years.csv")

slice_2005_2010.to_csv(slice_2005_2010_path, index=False)
slice_2010_2020.to_csv(slice_2010_2020_path, index=False)
slice_census_years.to_csv(slice_census_path, index=False)

print("Saved:", slice_2005_2010_path, "| rows:", len(slice_2005_2010))
print("Saved:", slice_2010_2020_path, "| rows:", len(slice_2010_2020))
print("Saved:", slice_census_path, "| rows:", len(slice_census_years))
