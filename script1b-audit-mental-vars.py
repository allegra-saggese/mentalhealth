#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mental-health coverage and panel-integrity audit for the merged county-year panel.
Outputs to: Dropbox/Mental/Data/merged/figs/panel-sumstats-by-farms

Quick purpose:
- Audits panel key integrity (`fips, year`) and source-duplication markers.
- Inventories mental-health variables and their year-by-year coverage.
- Checks cross-source duplication consistency for key mental outcomes.
- Writes QA tables used for summary-stat and data-readiness review.
"""

from packages import *
from functions import *


# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------
merged_dir = os.path.join(db_data, "merged")
out_dir = os.path.join(merged_dir, "figs", "panel-sumstats-by-farms")
os.makedirs(out_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")

MENTAL_KEYWORDS = (
    "poor_mental_health_days",
    "frequent_mental_distress",
    "mental_health_providers",
    "suicide",
    "depress",
    "anxiety",
    "excessive_drinking",
    "drug_overdose",
)

CORE_MENTAL_OUTCOMES = [
    "poor_mental_health_days_raw_value_mentalhealthrank_full",
    "frequent_mental_distress_raw_value_mentalhealthrank_full",
    "mental_health_providers_raw_value_mentalhealthrank_full",
    "excessive_drinking_raw_value_mentalhealthrank_full",
]

CORE_CONTEXT_COLS = [
    "deaths_cdc_county_year_deathsofdespair",
    "crude_rate_cdc_county_year_deathsofdespair",
    "population_population_full",
    "n_unique_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
latest_file = latest_file_glob
to_num = to_numeric_series
normalize_key = normalize_panel_key


def fill_pct(series):
    return float(series.notna().mean() * 100)


# ---------------------------------------------------------------------
# Load merged panel
# ---------------------------------------------------------------------
merged_path = latest_file(merged_dir, "*_full_merged.csv")
df = pd.read_csv(merged_path, low_memory=False)
df = normalize_key(df)
df = df.loc[:, ~df.columns.duplicated()].copy()
df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()
df = df.sort_values(["fips", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# 1) Panel integrity + source-duplication guards
# ---------------------------------------------------------------------
integrity = pd.DataFrame(
    [
        {
            "merged_file": os.path.basename(merged_path),
            "n_rows": int(len(df)),
            "n_unique_counties": int(df["fips"].nunique()),
            "n_unique_years": int(df["year"].nunique()),
            "year_min": int(df["year"].min()),
            "year_max": int(df["year"].max()),
            "n_duplicate_fips_year_rows": int(df.duplicated(["fips", "year"]).sum()),
        }
    ]
)
integrity.to_csv(os.path.join(out_dir, "qa_panel_integrity_mental_focus.csv"), index=False)

nrow_cols = sorted([c for c in df.columns if c.startswith("n_rows_")])
src_dup_rows = []
for c in nrow_cols:
    s = to_num(df[c])
    gt1 = (s > 1)
    src_dup_rows.append(
        {
            "source_n_rows_col": c,
            "rows_with_n_rows_gt_1": int(gt1.sum()),
            "pct_rows_with_n_rows_gt_1": float(gt1.mean() * 100),
            "max_n_rows": int(s.max()) if s.notna().any() else np.nan,
        }
    )
src_dup = pd.DataFrame(src_dup_rows)
src_dup.to_csv(os.path.join(out_dir, "qa_panel_source_duplication_counts.csv"), index=False)


# ---------------------------------------------------------------------
# 2) Mental variable inventory (focused)
# ---------------------------------------------------------------------
mh_rank_cols = [c for c in df.columns if c.endswith("_mentalhealthrank_full")]
mh_focus_cols = []
for c in mh_rank_cols:
    base = c[: -len("_mentalhealthrank_full")]
    if any(k in base for k in MENTAL_KEYWORDS):
        mh_focus_cols.append(c)

mh_focus_cols = sorted(set(mh_focus_cols))

inv_rows = []
for c in mh_focus_cols:
    raw = df[c]
    non = raw.notna()
    by_year_fill = non.groupby(df["year"]).mean() * 100
    years_with_data = df.loc[non, "year"]
    inv_rows.append(
        {
            "variable": c,
            "dtype": str(raw.dtype),
            "non_missing_n": int(non.sum()),
            "fill_pct": fill_pct(raw),
            "first_year_with_data": int(years_with_data.min()) if non.any() else np.nan,
            "last_year_with_data": int(years_with_data.max()) if non.any() else np.nan,
            "n_years_with_data": int(years_with_data.nunique()) if non.any() else 0,
            "yearly_fill_pct_min_positive": float(by_year_fill[by_year_fill > 0].min()) if (by_year_fill > 0).any() else 0.0,
            "yearly_fill_pct_max": float(by_year_fill.max()),
        }
    )

inventory = pd.DataFrame(inv_rows).sort_values(["fill_pct", "variable"], ascending=[False, True])
inventory.to_csv(os.path.join(out_dir, "qa_mental_focus_variable_inventory.csv"), index=False)


# ---------------------------------------------------------------------
# 3) Core mental outcomes: overall + by-year + county reach
# ---------------------------------------------------------------------
core_cols = [c for c in CORE_MENTAL_OUTCOMES if c in df.columns]
context_cols = [c for c in CORE_CONTEXT_COLS if c in df.columns]

overall_rows = []
for c in core_cols:
    s = to_num(df[c])
    by_year = s.notna().groupby(df["year"]).mean() * 100
    years = by_year[by_year > 0].index.tolist()
    overall_rows.append(
        {
            "variable": c,
            "non_missing_n": int(s.notna().sum()),
            "fill_pct_overall_panel": fill_pct(s),
            "first_year_with_data": int(min(years)) if years else np.nan,
            "last_year_with_data": int(max(years)) if years else np.nan,
            "n_years_with_data": int(len(years)),
            "min_positive_year_fill_pct": float(by_year[by_year > 0].min()) if years else 0.0,
            "max_year_fill_pct": float(by_year.max()),
            "median": float(s.median()) if s.notna().any() else np.nan,
            "p90": float(s.quantile(0.90)) if s.notna().any() else np.nan,
        }
    )
overall = pd.DataFrame(overall_rows).sort_values("variable")
overall.to_csv(os.path.join(out_dir, "qa_mental_outcome_core_overall.csv"), index=False)

county_any_rows = []
for c in core_cols:
    s = to_num(df[c])
    county_any_rows.append(
        {
            "variable": c,
            "n_counties_with_any_data": int(df.loc[s.notna(), "fips"].nunique()),
            "n_years_with_any_data": int(df.loc[s.notna(), "year"].nunique()),
        }
    )
county_any = pd.DataFrame(county_any_rows).sort_values("variable")
county_any.to_csv(os.path.join(out_dir, "qa_mental_outcome_counties_any.csv"), index=False)

year_rows = []
for yr, g in df.groupby("year", as_index=False):
    row = {"year": int(yr), "n_county_rows": int(len(g))}
    for c in [*core_cols, *context_cols]:
        s = to_num(g[c])
        row[f"fill_pct__{c}"] = float(s.notna().mean() * 100)
        row[f"non_missing_n__{c}"] = int(s.notna().sum())
    year_rows.append(row)

by_year = pd.DataFrame(year_rows).sort_values("year")
by_year.to_csv(os.path.join(out_dir, "qa_mental_outcome_core_by_year.csv"), index=False)


# ---------------------------------------------------------------------
# 4) Cross-source duplicate checks
# ---------------------------------------------------------------------
pairs = [
    (
        "poor_mental_health_days_raw_value_mentalhealthrank_full",
        "poor_mental_health_days_raw_value_mh_mortality_fips_yr",
    ),
    (
        "frequent_mental_distress_raw_value_mentalhealthrank_full",
        "frequent_mental_distress_raw_value_mh_mortality_fips_yr",
    ),
]

dupe_rows = []
for a, b in pairs:
    if a not in df.columns or b not in df.columns:
        continue
    sa = to_num(df[a])
    sb = to_num(df[b])
    m = sa.notna() & sb.notna()
    n = int(m.sum())
    if n == 0:
        dupe_rows.append(
            {
                "var_a": a,
                "var_b": b,
                "n_overlap": 0,
                "corr_on_overlap": np.nan,
                "exact_match_pct_on_overlap": np.nan,
                "median_abs_diff_on_overlap": np.nan,
                "p90_abs_diff_on_overlap": np.nan,
            }
        )
        continue

    diff = (sa[m] - sb[m]).abs()
    dupe_rows.append(
        {
            "var_a": a,
            "var_b": b,
            "n_overlap": n,
            "corr_on_overlap": float(sa[m].corr(sb[m])),
            "exact_match_pct_on_overlap": float((diff < 1e-12).mean() * 100),
            "median_abs_diff_on_overlap": float(diff.median()),
            "p90_abs_diff_on_overlap": float(diff.quantile(0.90)),
        }
        )

dupe_df = pd.DataFrame(dupe_rows)
if dupe_df.empty:
    dupe_df = pd.DataFrame(
        [
            {
                "var_a": pd.NA,
                "var_b": pd.NA,
                "n_overlap": 0,
                "corr_on_overlap": np.nan,
                "exact_match_pct_on_overlap": np.nan,
                "median_abs_diff_on_overlap": np.nan,
                "p90_abs_diff_on_overlap": np.nan,
                "note": "No duplicated mental-outcome sources present in merged panel (expected with direct CDC merge path).",
            }
        ]
    )
dupe_df.to_csv(os.path.join(out_dir, "qa_mental_crosssource_duplicate_check.csv"), index=False)


# ---------------------------------------------------------------------
# 5) Short memo
# ---------------------------------------------------------------------
memo_lines = [
    f"# Mental Coverage Audit ({today_str})",
    "",
    "## Scope",
    f"- Merged input: `{os.path.basename(merged_path)}`",
    f"- Unit: county-year (`fips`, `year`)",
    f"- Rows: `{len(df):,}` | Counties: `{df['fips'].nunique():,}` | Years: `{int(df['year'].min())}`-`{int(df['year'].max())}`",
    "",
    "## Outputs",
    "- `qa_panel_integrity_mental_focus.csv`",
    "- `qa_panel_source_duplication_counts.csv`",
    "- `qa_mental_focus_variable_inventory.csv`",
    "- `qa_mental_outcome_core_overall.csv`",
    "- `qa_mental_outcome_core_by_year.csv`",
    "- `qa_mental_outcome_counties_any.csv`",
    "- `qa_mental_crosssource_duplicate_check.csv`",
    "",
    "## Notes",
    "- Overall fill rates are over the full panel window (2000-2023), so variables that start in later years will look lower in the overall metric.",
    "- Use `qa_mental_outcome_core_by_year.csv` for year-appropriate coverage interpretation.",
]

memo_path = os.path.join(out_dir, "qa_mental_coverage_memo.md")
with open(memo_path, "w", encoding="utf-8") as f:
    f.write("\n".join(memo_lines) + "\n")

print("Saved mental coverage audit outputs to:", out_dir)
