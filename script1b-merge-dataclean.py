#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 01:10:00 2025

Builds one county-year merged panel from the latest version of each clean file.
Filtering rule: keep only (fips, year) where rural-key non_large_metro == 1.
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
    "crime_fips_level_final",
    "mentalhealthrank_full",
    "mh_mortality_fips_yr",
    "population_full",
}
RURAL_DESCRIPTOR_HINT = "rural-key"


def _split_descriptor(path):
    """
    Returns:
      descriptor: file stem without YYYY-MM-DD_ or YYYY-MM-DD- prefix
      date_key: datetime.date or date.min if no prefix
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^(\d{4}-\d{2}-\d{2})[-_](.+)$", stem)
    if m:
        return m.group(2), date.fromisoformat(m.group(1))
    return stem, date.min


def _latest_files_by_descriptor(folder):
    supported = ("*.csv", "*.parquet", "*.pq", "*.xlsx", "*.xls")
    files = []
    for pat in supported:
        files.extend(glob.glob(os.path.join(folder, pat)))

    latest = {}
    for path in files:
        descriptor, date_key = _split_descriptor(path)
        mtime = os.path.getmtime(path)
        score = (date_key, mtime)
        if descriptor not in latest or score > latest[descriptor]["score"]:
            latest[descriptor] = {"path": path, "score": score}

    return {k: v["path"] for k, v in latest.items()}


def _ensure_key(df):
    """
    Ensure df has normalized fips/year key columns.
    """
    df = clean_cols(df.copy())
    if "fips" not in df.columns:
        if "fips_generated" in df.columns:
            df = df.rename(columns={"fips_generated": "fips"})
        elif "geoid" in df.columns:
            df = df.rename(columns={"geoid": "fips"})

    if "year" not in df.columns and "yr" in df.columns:
        df = df.rename(columns={"yr": "year"})

    if "fips" not in df.columns or "year" not in df.columns:
        return None

    df["fips"] = (
        df["fips"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(5)
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["fips", "year"]).copy()
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

        agg = {c: "sum" for c in numeric_cols}
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


latest = _latest_files_by_descriptor(clean_dir)
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
