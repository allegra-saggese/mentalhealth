#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 18:55:21 2025

Population QA across merged outputs.
"""

from packages import *
from functions import *
import re


merged_dir = os.path.join(db_data, "merged")
out_dir = os.path.join(db_me, "interim-data")
os.makedirs(out_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")


def _find_merged_files():
    files = sorted(glob.glob(os.path.join(merged_dir, "*_full_merged*.csv")))
    if not files:
        raise FileNotFoundError(f"No merged files found in {merged_dir}")
    return files


def _pop_cols(df):
    pat = re.compile(r"(pop|population|popestimate|population_estimate)", re.IGNORECASE)
    return [c for c in df.columns if pat.search(c)]


def _to_num(s):
    return pd.to_numeric(
        s.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def _pick_reference(cols):
    priority = [
        "population_population_full",
        "population_full_population",
        "population",
    ]
    for p in priority:
        if p in cols:
            return p
    pop_full_like = [c for c in cols if "population_full" in c]
    if pop_full_like:
        return sorted(pop_full_like)[0]
    return sorted(cols)[0] if cols else None


summary_rows = []
pair_rows = []
catalog_rows = []

for path in _find_merged_files():
    df = pd.read_csv(path, low_memory=False)
    fname = os.path.basename(path)
    pop_cols = _pop_cols(df)

    for c in pop_cols:
        s_num = _to_num(df[c])
        non_null = int(s_num.notna().sum())
        zero_n = int((s_num == 0).sum())
        summary_rows.append(
            {
                "file": fname,
                "column": c,
                "n_rows": len(df),
                "n_non_null_numeric": non_null,
                "pct_non_null_numeric": round(100 * non_null / len(df), 4),
                "n_zero": zero_n,
                "pct_zero_of_non_null": round(100 * zero_n / non_null, 4) if non_null else np.nan,
                "min": s_num.min(skipna=True),
                "median": s_num.median(skipna=True),
                "max": s_num.max(skipna=True),
            }
        )
        catalog_rows.append({"file": fname, "column": c})

    ref = _pick_reference(pop_cols)
    if ref is None:
        continue

    ref_num = _to_num(df[ref])
    for c in pop_cols:
        if c == ref:
            continue
        cmp_num = _to_num(df[c])
        mask = ref_num.notna() & cmp_num.notna()
        n_overlap = int(mask.sum())
        if n_overlap == 0:
            pair_rows.append(
                {
                    "file": fname,
                    "reference_col": ref,
                    "compare_col": c,
                    "n_overlap": 0,
                    "corr": np.nan,
                    "mean_abs_diff": np.nan,
                    "mean_pct_diff": np.nan,
                }
            )
            continue

        diff = (cmp_num[mask] - ref_num[mask]).astype(float)
        base = ref_num[mask].replace(0, np.nan).astype(float)
        pct_diff = (diff.abs() / base.abs()) * 100
        corr = np.corrcoef(ref_num[mask].astype(float), cmp_num[mask].astype(float))[0, 1]

        pair_rows.append(
            {
                "file": fname,
                "reference_col": ref,
                "compare_col": c,
                "n_overlap": n_overlap,
                "corr": corr,
                "mean_abs_diff": diff.abs().mean(),
                "mean_pct_diff": pct_diff.mean(skipna=True),
            }
        )


summary_df = pd.DataFrame(summary_rows).sort_values(["file", "pct_non_null_numeric", "column"], ascending=[True, False, True])
pair_df = pd.DataFrame(pair_rows).sort_values(["file", "n_overlap", "mean_abs_diff"], ascending=[True, False, True])
catalog_df = pd.DataFrame(catalog_rows).drop_duplicates().sort_values(["column", "file"])

summary_path = os.path.join(out_dir, f"{today_str}_qa_population_columns_summary.csv")
pair_path = os.path.join(out_dir, f"{today_str}_qa_population_columns_pairwise.csv")
catalog_path = os.path.join(out_dir, f"{today_str}_qa_population_columns_catalog.csv")

summary_df.to_csv(summary_path, index=False)
pair_df.to_csv(pair_path, index=False)
catalog_df.to_csv(catalog_path, index=False)

print("Saved:", summary_path)
print("Saved:", pair_path)
print("Saved:", catalog_path)
