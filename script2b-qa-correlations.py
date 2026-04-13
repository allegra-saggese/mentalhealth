#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build QA memo + general correlation tables for merged county-year panel.
Outputs to: Dropbox/Mental/Data/merged/figs/panel-sumstats-by-farms
"""

from packages import *
from functions import *


# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------
clean_dir = os.path.join(db_data, "clean")
merged_dir = os.path.join(db_data, "merged")
out_dir = os.path.join(merged_dir, "figs", "panel-sumstats-by-farms")
os.makedirs(out_dir, exist_ok=True)

today_str = date.today().strftime("%Y-%m-%d")
MIN_CORR_N = 200


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
latest_file = latest_file_glob
to_num = to_numeric_series
normalize_key = normalize_panel_key


def pairwise_corr_long(df_num, columns, method="pearson", min_n=MIN_CORR_N):
    rows = []
    for i, a in enumerate(columns):
        sa = df_num[a]
        for b in columns[i + 1 :]:
            sb = df_num[b]
            mask = sa.notna() & sb.notna()
            n = int(mask.sum())
            if n < min_n:
                continue
            corr = sa[mask].corr(sb[mask], method=method)
            if pd.isna(corr):
                continue
            rows.append(
                {
                    "var_a": a,
                    "var_b": b,
                    "corr": float(corr),
                    "abs_corr": float(abs(corr)),
                    "n_obs_pairwise": n,
                    "method": method,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return out


def safe_float(v):
    try:
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


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
# Variable sets
# ---------------------------------------------------------------------
cafo_cols = [
    "cafo_cattle_small", "cafo_cattle_medium", "cafo_cattle_large",
    "cafo_hogs_small", "cafo_hogs_medium", "cafo_hogs_large",
    "cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large",
    "cafo_total_ops_all_animals", "cafo_total_ops_chickens",
]

mental_cols = [
    "poor_mental_health_days_raw_value_mentalhealthrank_full",
    "frequent_mental_distress_raw_value_mentalhealthrank_full",
    "poor_mental_health_days_raw_value_mh_mortality_fips_yr",
    "frequent_mental_distress_raw_value_mh_mortality_fips_yr",
]

mortality_cols = [
    "mortality_total_deaths_mh_mortality_fips_yr",
]

population_cols = [
    "population_population_full",
]

fsis_12_cols = [
    "n_unique_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_unique_est_size_combos_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_slaughterhouse_present_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_processing_present_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_meat_slaughter_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_poultry_slaughter_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_1_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_2_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_3_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_4_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_5_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "n_size_bucket_missing_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
]

all_cols = [*cafo_cols, *mental_cols, *mortality_cols, *population_cols, *fsis_12_cols]
existing_cols = [c for c in all_cols if c in df.columns]
for c in existing_cols:
    df[c] = to_num(df[c])

manifest_rows = []
for c in existing_cols:
    fam = "other"
    if c in cafo_cols:
        fam = "cafo"
    elif c in fsis_12_cols:
        fam = "fsis"
    elif c in mental_cols:
        fam = "mental"
    elif c in mortality_cols:
        fam = "mortality"
    elif c in population_cols:
        fam = "population"
    manifest_rows.append({"variable": c, "family": fam})
manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(os.path.join(out_dir, "correlation_vars_manifest.csv"), index=False)


# ---------------------------------------------------------------------
# Coverage checks for memo
# ---------------------------------------------------------------------
coverage_vars = [
    "cafo_total_ops_all_animals",
    "cafo_total_ops_chickens",
    "poor_mental_health_days_raw_value_mentalhealthrank_full",
    "frequent_mental_distress_raw_value_mentalhealthrank_full",
    "mortality_total_deaths_mh_mortality_fips_yr",
    "n_unique_establishments_fsis_county_year_fips_est_size_type_summary_hudbulk_manualzip",
    "population_population_full",
]
coverage_rows = []
for yr, g in df.groupby("year", as_index=False):
    row = {"year": int(yr)}
    for c in coverage_vars:
        if c in g.columns:
            row[f"fill_pct__{c}"] = float(g[c].notna().mean() * 100)
    coverage_rows.append(row)
coverage_by_year = pd.DataFrame(coverage_rows).sort_values("year")
coverage_by_year.to_csv(os.path.join(out_dir, "qa_key_variable_coverage_by_year.csv"), index=False)


# ---------------------------------------------------------------------
# FSIS 12-column QA summary
# ---------------------------------------------------------------------
fsis_q_rows = []
for c in fsis_12_cols:
    if c not in df.columns:
        continue
    s = to_num(df[c])
    nonmiss = s.notna().sum()
    p99 = safe_float(s.quantile(0.99)) if nonmiss else np.nan
    vmax = safe_float(s.max()) if nonmiss else np.nan
    share_gt_1k = safe_float((s > 1000).mean() * 100) if nonmiss else np.nan
    fsis_q_rows.append(
        {
            "variable": c,
            "non_missing_n": int(nonmiss),
            "fill_pct": float(s.notna().mean() * 100),
            "p50": safe_float(s.quantile(0.50)) if nonmiss else np.nan,
            "p90": safe_float(s.quantile(0.90)) if nonmiss else np.nan,
            "p99": p99,
            "max": vmax,
            "share_gt_1000_pct": share_gt_1k,
            "qa_reliable_for_now": int((pd.notna(p99) and p99 <= 1000) and (pd.notna(vmax) and vmax <= 1000)),
        }
    )

fsis_qa = pd.DataFrame(fsis_q_rows).sort_values(["qa_reliable_for_now", "share_gt_1000_pct"], ascending=[False, True])
fsis_qa.to_csv(os.path.join(out_dir, "qa_fsis_12_column_check.csv"), index=False)


# ---------------------------------------------------------------------
# Correlation tables
# ---------------------------------------------------------------------
windows = {
    "county_year_2010_2020": df[df["year"].between(2010, 2020, inclusive="both")].copy(),
    "county_year_2017_2023": df[df["year"].between(2017, 2023, inclusive="both")].copy(),
    "county_year_2017": df[df["year"] == 2017].copy(),
}

for wname, sub in windows.items():
    use_cols = [c for c in existing_cols if c in sub.columns and sub[c].notna().any()]
    if not use_cols:
        continue
    sub_num = sub[use_cols].copy()
    for c in use_cols:
        sub_num[c] = to_num(sub_num[c])

    for method in ("pearson", "spearman"):
        corr_long = pairwise_corr_long(sub_num, use_cols, method=method, min_n=MIN_CORR_N)
        corr_long.to_csv(
            os.path.join(out_dir, f"correlation_long_{wname}_{method}.csv"),
            index=False,
        )
        top30 = corr_long.head(30).copy() if not corr_long.empty else corr_long
        top30.to_csv(
            os.path.join(out_dir, f"correlation_top30_{wname}_{method}.csv"),
            index=False,
        )

        corr_matrix = sub_num[use_cols].corr(method=method, min_periods=MIN_CORR_N)
        corr_matrix.to_csv(
            os.path.join(out_dir, f"correlation_matrix_{wname}_{method}.csv"),
            index=True,
        )


# ---------------------------------------------------------------------
# QA memo (Markdown)
# ---------------------------------------------------------------------
coverage_check_path = os.path.join(out_dir, "county_coverage_check.csv")
coverage_check = pd.read_csv(coverage_check_path) if os.path.exists(coverage_check_path) else pd.DataFrame()
crosscheck_path = os.path.join(out_dir, "cafo_animal_size_crosscheck_vs_premerged.csv")
crosscheck = pd.read_csv(crosscheck_path) if os.path.exists(crosscheck_path) else pd.DataFrame()

fsis_reliable = fsis_qa.loc[fsis_qa["qa_reliable_for_now"] == 1, "variable"].tolist()
fsis_flagged = fsis_qa.loc[fsis_qa["qa_reliable_for_now"] == 0, "variable"].tolist()

county_match = "n/a"
key_match = "n/a"
if not coverage_check.empty:
    county_match = "PASS" if int(coverage_check["county_count_match"].iloc[0]) == 1 else "FAIL"
    key_match = "PASS" if int(coverage_check["key_set_exact_match"].iloc[0]) == 1 else "FAIL"

cafo_cross = "n/a"
if not crosscheck.empty and "pct_exact_match_on_compared_keys" in crosscheck.columns:
    cafo_cross = f"{crosscheck['pct_exact_match_on_compared_keys'].min():.1f}% to {crosscheck['pct_exact_match_on_compared_keys'].max():.1f}% exact-match"

memo_lines = [
    f"# QA Memo: Panel + Correlation Tables ({today_str})",
    "",
    "## 1) Scope",
    f"- Merged input file used: `{os.path.basename(merged_path)}`",
    f"- Observation unit: county-year (`fips`, `year`)",
    f"- Rows: `{len(df):,}`",
    f"- Unique counties: `{df['fips'].nunique():,}`",
    f"- Year range: `{int(df['year'].min())}` to `{int(df['year'].max())}`",
    "",
    "## 2) Key QA Checks",
    f"- County-count check vs rural kept keys: **{county_match}**",
    f"- Exact key-set match vs rural kept keys: **{key_match}**",
    f"- CAFO animal×size merged-vs-premerged cross-check: **{cafo_cross}**",
    "",
    "## 3) FSIS 12-Column QA (2017 county maps set)",
    "- Rule used here: flag as unreliable-for-now when upper tail explodes (`p99 > 1000` or `max > 1000`).",
    f"- Reliable-for-now columns ({len(fsis_reliable)}):",
]
for c in fsis_reliable:
    memo_lines.append(f"  - `{c}`")

memo_lines.append(f"- Flagged columns ({len(fsis_flagged)}):")
for c in fsis_flagged:
    memo_lines.append(f"  - `{c}`")

memo_lines.extend(
    [
        "",
        "## 4) What Correlation Tables Are",
        "- A correlation table quantifies pairwise association between variables.",
        "- Values are between `-1` and `+1`:",
        "  - `+1`: move together strongly",
        "  - `-1`: move in opposite directions strongly",
        "  - `0`: little/no monotonic/linear association",
        "- **Pearson** correlation: linear association on raw values.",
        "- **Spearman** correlation: rank-based association (more robust to skew/outliers).",
        "- Correlation is **not causal**; it is descriptive screening.",
        "",
        f"- Pairwise minimum sample filter used: `n >= {MIN_CORR_N}`.",
        "",
        "## 5) Correlation Outputs Produced",
        "- `correlation_vars_manifest.csv`",
        "- `correlation_long_county_year_2010_2020_pearson.csv`",
        "- `correlation_long_county_year_2010_2020_spearman.csv`",
        "- `correlation_long_county_year_2017_2023_pearson.csv`",
        "- `correlation_long_county_year_2017_2023_spearman.csv`",
        "- `correlation_long_county_year_2017_pearson.csv`",
        "- `correlation_long_county_year_2017_spearman.csv`",
        "- Matching `top30` and `matrix` files for each window/method.",
        "",
        "## 6) Recommended Use",
        "- Use `2010-2020` tables for broader CAFO + mental structure.",
        "- Use `2017-2023` and `2017` tables when including FSIS fields.",
        "- Prioritize Spearman when heavy right tails are present (counts/density fields).",
    ]
)

memo_path = os.path.join(out_dir, "qa_memo_panel_sumstats_by_farms.md")
with open(memo_path, "w", encoding="utf-8") as f:
    f.write("\n".join(memo_lines) + "\n")

print("Saved outputs to:", out_dir)
