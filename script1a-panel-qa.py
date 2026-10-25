"""script1a-panel-qa.py -- checks that can fail, run on the finished county-year panel (QA tables and QA plots; no estimation).

Sections (run one with --only <name>, several with --only a b; default is all, in this order):
  integrity       panel key integrity and source-duplication counts
  inventory       mental-health variable inventory (fill rate, first and last year)
  core_outcomes   core mental outcomes: overall, by year, county reach
  crosssource     agreement of duplicated mental-outcome sources
  coverage        outcome x year coverage matrix and heatmap
  balance         panel balance by year and across years
  cdc_suppression CDC deaths-of-despair crude-rate coverage by year
  distributions   2012 outcome histograms
  cafo_percapita  CAFO operations per 10,000 residents, raw vs log, and p99 outliers
  cdc_sensecheck  crude rate recomputed from deaths / Census population vs the CDC rate
  coverage_check  panel keys vs the rural-key file's kept keys (county count and exact key set)
  mental_fill     share of rows with any mental-health value
  cafo_crosscheck CAFO commodity x size counts in the panel vs the pre-merged compact file
  correlations    correlation tables, FSIS column QA, QA memo (reads the two checks above)
  missingness     missingness tables, heatmaps, row completeness
  memo            coverage memo (lists the outputs above)
Reads:   Data/merged/*_panel.csv and *_panel_fsis.csv (newest); Data/clean/*rural-key*.csv and *_cafo_ops_by_size_compact.csv;
         the cdc_sensecheck columns stay in the QA tables only.
Writes:  Data/output/tables/script1a/YYYY-MM-DD_*.csv|md   and   Data/output/figs/script1a/YYYY-MM-DD_*.{pdf,png}
         Never writes to Data/merged (the panel has one writer, script0f-generate-panel.py).
Run:     PYTHONHASHSEED=0 python3 script1a-panel-qa.py [--only <section> ...]      Not stochastic, so no seed. Runtime: about 1 min.
"""
import argparse
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

import figstyle
from functions import latest_file_glob, normalize_panel_key, read_and_prepare, to_numeric_series
from packages import db_data, figs_dir, tables_dir

# ---------------------------------------------------------------------------------------------------------------------
# Paths and constants (shared set-up, defined once)
# ---------------------------------------------------------------------------------------------------------------------
SCRIPT = "script1a"
MERGED_DIR = Path(db_data) / "merged"
TABLE_DIR = Path(tables_dir) / SCRIPT
FIG_DIR = Path(figs_dir) / SCRIPT
CLEAN_DIR = Path(db_data) / "clean"
TODAY = date.today().isoformat()

MENTAL_KEYWORDS = ("poor_mental_health_days", "frequent_mental_distress", "mental_health_providers", "suicide", "depress",
                   "anxiety", "excessive_drinking", "drug_overdose")
# Column names as they appear in the renamed panel (script0f strips source suffixes). frequent_mental_distress,
# mental_health_providers and excessive_drinking keep only their _per100k variants.
CORE_MENTAL_OUTCOMES = ["poor_mental_health_days", "frequent_mental_distress_per100k", "mental_health_providers_per100k",
                        "excessive_drinking_per100k"]
# n_unique_establishments_fsis is absent from panel.csv (FSIS-only panel); listed so the fill table shows the absence.
CORE_CONTEXT_COLS = ["deaths_despair", "crude_rate_despair", "population", "n_unique_establishments_fsis"]
DUPLICATE_PAIRS = [("poor_mental_health_days", "poor_mental_health_days_raw_value_mh_mortality_fips_yr"),
                   ("frequent_mental_distress_per100k", "frequent_mental_distress_raw_value_mh_mortality_fips_yr")]
COVERAGE_OUTCOMES = {
    "Poor Mental Health Days (CHR)": "poor_mental_health_days",
    "Frequent Mental Distress per 100k (CHR)": "frequent_mental_distress_per100k",
    "Excessive Drinking per 100k (CHR)": "excessive_drinking_per100k",
    "Violent Crime Rate (CHR)": "violent_crime",
    "Homicide Rate (CHR)": "homicides",
    "Aggravated Assault (UCR)": "aggravated_assault_crime",
    "Deaths of Despair Crude Rate (CDC)": "crude_rate_despair",
    "CAFO Total Ops": "cafo_total_ops_all_animals",
    "FSIS Establishments": "n_unique_establishments_fsis",
    "County Population": "population",
}
HIST_OUTCOMES = {  # label -> (column, x-axis label with unit)
    "Poor mental health days": ("poor_mental_health_days", "Poor mental health days (days per month)"),
    "Excessive drinking": ("excessive_drinking_per100k", "Excessive drinking (per 100,000)"),
    "Violent crime rate": ("violent_crime", "Violent crime rate (per 100,000, CHR)"),
    "Homicide rate": ("homicides", "Homicide rate (per 100,000, CHR)"),
    "Aggravated assault": ("aggravated_assault_crime", "Aggravated assault (UCR count)"),
    "Deaths of despair": ("crude_rate_despair", "Deaths of despair crude rate (per 100,000)"),
}
REFERENCE_YEAR = 2012            # good CAFO coverage (97%) and partial mental-health coverage
HIST_BINS, CAFO_BINS, OUTLIER_QUANTILE = 40, 50, 0.99
MIN_CORR_N = 200
CDC_DEATHS, CDC_POP, CDC_CRUDE, COUNTY_NAME = "deaths_despair", "population_despair", "crude_rate_despair", "county_name_despair"
CENSUS_POP = "population"
CAFO_CORR_COLS = ["cafo_cattle_small", "cafo_cattle_medium", "cafo_cattle_large", "cafo_hogs_small", "cafo_hogs_medium",
                  "cafo_hogs_large", "cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large",
                  "cafo_total_ops_all_animals", "cafo_total_ops_chickens"]
MENTAL_CORR_COLS = ["poor_mental_health_days", "frequent_mental_distress_per100k"]
FSIS_QA_COLS = ["n_unique_establishments_fsis", "n_unique_est_size_combos_fsis",
                "n_slaughterhouse_present_establishments_fsis", "n_processing_present_establishments_fsis",
                "n_meat_slaughter_establishments_fsis", "n_poultry_slaughter_establishments_fsis",
                "n_size_bucket_1_fsis", "n_size_bucket_2_fsis", "n_size_bucket_3_fsis", "n_size_bucket_4_fsis",
                "n_size_bucket_5_fsis", "n_size_bucket_missing_fsis"]
FSIS_RELIABLE_MAX = 1000         # a count above this in a county-year is treated as a data error until checked
CORR_WINDOWS = {"county_year_2010_2020": (2010, 2020), "county_year_2017_2023": (2017, 2023), "county_year_2017": (2017, 2017)}
CAFO_SIZES = ("small", "medium", "large")
CAFO_COMMODITIES = ("cattle", "hogs", "chickens")
CAFO_CANONICAL_CLASS = {"cattle": "incl calves", "hogs": "all classes", "chickens": "layers"}  # one animal class per commodity
FAMILY_CHUNK = 35                # variables per family-heatmap figure
TOP_MISSING_VARS = 60


# ---------------------------------------------------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------------------------------------------------
def write_table(frame, stem, index=False):
    """Write a dated CSV to TABLE_DIR and return its path."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{TODAY}_{stem}.csv"
    frame.to_csv(path, index=index)
    print("Saved:", path)
    return path


def format_stat(value):
    """Median/mean for a legend: two decimals for small values, one for large, thousands separators."""
    return f"{value:,.2f}" if abs(value) < 100 else f"{value:,.1f}"


def pct_filled(series):
    return float(series.notna().mean() * 100)


@lru_cache(maxsize=None)
def merged_path(pattern):
    return latest_file_glob(str(MERGED_DIR), pattern)


@lru_cache(maxsize=None)
def load_panel():
    """Newest *_panel.csv with the key normalised. Reports duplicate (fips, year) rows before dropping them."""
    path = merged_path("*_panel.csv")
    raw = normalize_panel_key(pd.read_csv(path, low_memory=False))
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    n_duplicates = int(raw.duplicated(["fips", "year"]).sum())
    if n_duplicates:
        print(f"WARNING: {n_duplicates:,} duplicate (fips, year) rows in {Path(path).name}; keeping the first of each")
        write_table(raw.loc[raw.duplicated(["fips", "year"], keep=False)], "merged_duplicate_fips_year_rows")
    panel = raw.drop_duplicates(["fips", "year"], keep="first").sort_values(["fips", "year"]).reset_index(drop=True)
    print(f"Panel: {Path(path).name} | {len(panel):,} rows | {panel['fips'].nunique():,} counties | "
          f"{int(panel['year'].min())}-{int(panel['year'].max())}")
    return panel, n_duplicates


@lru_cache(maxsize=None)
def load_fsis_panel():
    path = merged_path("*_panel_fsis.csv")
    frame = normalize_panel_key(pd.read_csv(path, low_memory=False))
    frame = frame.loc[:, ~frame.columns.duplicated()].sort_values(["fips", "year"]).reset_index(drop=True)
    print(f"FSIS panel: {Path(path).name} | {len(frame):,} rows | {frame['fips'].nunique():,} counties")
    return frame


def source_note(extra=""):
    """One-line provenance note printed under each QA figure (the caption is written by the author)."""
    panel, _ = load_panel()
    text = (f"Source: {Path(merged_path('*_panel.csv')).name}, {len(panel):,} county-years, {panel['fips'].nunique():,} counties "
            f"({int(panel['year'].min())}-{int(panel['year'].max())}). Script: script1a-panel-qa.py. {extra}").strip()
    return text


def save_qa_figure(fig, stem, note):
    """Add the source note, reserve room for it, and save PDF + PNG through figstyle."""
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.955))
    fig.text(0.005, 0.004, note, fontsize=5, ha="left", va="bottom", color="0.3", wrap=True)
    pdf, png = figstyle.save_figure(fig, FIG_DIR, stem)
    plt.close(fig)
    print("Saved:", png)
    return png


def safe_name(text, n=110):
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(text)).strip("_").lower()[:n]


def chunked(items, size):
    for i in range(0, len(items), size):
        yield i // size + 1, items[i:i + size]


def detect_family(col):
    """Variable family from its name, used to split the missingness heatmaps."""
    if col.startswith("cafo_"):
        return "cafo"
    if col.endswith("_fsis"):
        return "fsis"
    if col.endswith("_crime") or col in {"violent_crime", "homicides"}:
        return "crime"
    if "_despair" in col or col in {"cdc_in_query", "deaths_is_zero", "crude_rate_from_census_pop", "deaths_per_10k_census_pop"}:
        return "cdc"
    if col == "population" or col.startswith("%_"):
        return "demographics"
    return "other"


def safe_float(value):
    try:
        return float(value) if pd.notna(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------------------------------------------------
# 1a sections: integrity, inventory, core outcomes, cross-source duplicates
# ---------------------------------------------------------------------------------------------------------------------
def run_integrity():
    """Key integrity of the panel and the per-source row counts (n_rows_* > 1 means a source had several rows per key)."""
    df, n_raw_duplicates = load_panel()
    integrity = pd.DataFrame([{
        "merged_file": Path(merged_path("*_panel.csv")).name,
        "n_rows": int(len(df)),
        "n_unique_counties": int(df["fips"].nunique()),
        "n_unique_years": int(df["year"].nunique()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        # Counted before duplicates are dropped, so this check can fail (the retired script counted after dropping them).
        "n_duplicate_fips_year_rows": n_raw_duplicates,
    }])
    write_table(integrity, "qa_panel_integrity_mental_focus")

    rows = []
    for col in sorted(c for c in df.columns if c.startswith("n_rows_")):
        counts = to_numeric_series(df[col])
        above_one = counts > 1
        rows.append({"source_n_rows_col": col, "rows_with_n_rows_gt_1": int(above_one.sum()),
                     "pct_rows_with_n_rows_gt_1": float(above_one.mean() * 100),
                     "max_n_rows": int(counts.max()) if counts.notna().any() else np.nan})
    write_table(pd.DataFrame(rows), "qa_panel_source_duplication_counts")


def run_inventory():
    """Fill rate and year range of every mental-health-related column."""
    df, _ = load_panel()
    rows = []
    for col in sorted(c for c in df.columns if any(k in c for k in MENTAL_KEYWORDS)):
        raw = df[col]
        present = raw.notna()
        fill_by_year = present.groupby(df["year"]).mean() * 100
        years_present = df.loc[present, "year"]
        rows.append({
            "variable": col, "dtype": str(raw.dtype), "non_missing_n": int(present.sum()), "fill_pct": pct_filled(raw),
            "first_year_with_data": int(years_present.min()) if present.any() else np.nan,
            "last_year_with_data": int(years_present.max()) if present.any() else np.nan,
            "n_years_with_data": int(years_present.nunique()) if present.any() else 0,
            "yearly_fill_pct_min_positive": float(fill_by_year[fill_by_year > 0].min()) if (fill_by_year > 0).any() else 0.0,
            "yearly_fill_pct_max": float(fill_by_year.max()),
        })
    inventory = pd.DataFrame(rows).sort_values(["fill_pct", "variable"], ascending=[False, True])
    write_table(inventory, "qa_mental_focus_variable_inventory")


def run_core_outcomes():
    """Core mental outcomes: overall fill, county reach, and fill by year."""
    df, _ = load_panel()
    core_cols = [c for c in CORE_MENTAL_OUTCOMES if c in df.columns]
    context_cols = [c for c in CORE_CONTEXT_COLS if c in df.columns]

    overall_rows, reach_rows = [], []
    for col in core_cols:
        values = to_numeric_series(df[col])
        fill_by_year = values.notna().groupby(df["year"]).mean() * 100
        years = fill_by_year[fill_by_year > 0].index.tolist()
        overall_rows.append({
            "variable": col, "non_missing_n": int(values.notna().sum()), "fill_pct_overall_panel": pct_filled(values),
            "first_year_with_data": int(min(years)) if years else np.nan,
            "last_year_with_data": int(max(years)) if years else np.nan,
            "n_years_with_data": int(len(years)),
            "min_positive_year_fill_pct": float(fill_by_year[fill_by_year > 0].min()) if years else 0.0,
            "max_year_fill_pct": float(fill_by_year.max()),
            "median": float(values.median()) if values.notna().any() else np.nan,
            "p90": float(values.quantile(0.90)) if values.notna().any() else np.nan,
        })
        reach_rows.append({"variable": col, "n_counties_with_any_data": int(df.loc[values.notna(), "fips"].nunique()),
                           "n_years_with_any_data": int(df.loc[values.notna(), "year"].nunique())})
    write_table(pd.DataFrame(overall_rows).sort_values("variable"), "qa_mental_outcome_core_overall")
    write_table(pd.DataFrame(reach_rows).sort_values("variable"), "qa_mental_outcome_counties_any")

    year_rows = []
    for year, group in df.groupby("year", as_index=False):
        row = {"year": int(year), "n_county_rows": int(len(group))}
        for col in [*core_cols, *context_cols]:
            values = to_numeric_series(group[col])
            row[f"fill_pct__{col}"] = float(values.notna().mean() * 100)
            row[f"non_missing_n__{col}"] = int(values.notna().sum())
        year_rows.append(row)
    write_table(pd.DataFrame(year_rows).sort_values("year"), "qa_mental_outcome_core_by_year")


def run_crosssource():
    """Where one outcome arrives from two sources, how closely do they agree on the overlap?"""
    df, _ = load_panel()
    columns = ["var_a", "var_b", "n_overlap", "corr_on_overlap", "exact_match_pct_on_overlap",
               "median_abs_diff_on_overlap", "p90_abs_diff_on_overlap"]
    rows = []
    for a, b in DUPLICATE_PAIRS:
        if a not in df.columns or b not in df.columns:
            continue
        sa, sb = to_numeric_series(df[a]), to_numeric_series(df[b])
        both = sa.notna() & sb.notna()
        n_overlap = int(both.sum())
        if n_overlap == 0:
            rows.append({"var_a": a, "var_b": b, "n_overlap": 0, **{c: np.nan for c in columns[3:]}})
            continue
        diff = (sa[both] - sb[both]).abs()
        rows.append({"var_a": a, "var_b": b, "n_overlap": n_overlap, "corr_on_overlap": float(sa[both].corr(sb[both])),
                     "exact_match_pct_on_overlap": float((diff < 1e-12).mean() * 100),
                     "median_abs_diff_on_overlap": float(diff.median()),
                     "p90_abs_diff_on_overlap": float(diff.quantile(0.90))})
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame([{"var_a": pd.NA, "var_b": pd.NA, "n_overlap": 0, "corr_on_overlap": np.nan,
                                "exact_match_pct_on_overlap": np.nan, "median_abs_diff_on_overlap": np.nan,
                                "p90_abs_diff_on_overlap": np.nan,
                                "note": "No duplicated mental-outcome sources present in merged panel "
                                        "(expected with direct CDC merge path)."}])
    write_table(result, "qa_mental_crosssource_duplicate_check")


# ---------------------------------------------------------------------------------------------------------------------
# 1a QA plots and the tables behind them
# ---------------------------------------------------------------------------------------------------------------------
def run_coverage():
    """Share of counties with a non-missing value, outcome x year: which outcomes are usable in which years."""
    df, _ = load_panel()
    years = sorted(df["year"].dropna().astype(int).unique())
    matrix = {}
    for label, col in COVERAGE_OUTCOMES.items():
        matrix[label] = {yr: (np.nan if col not in df.columns
                              else round(to_numeric_series(df.loc[df["year"] == yr, col]).notna().mean() * 100, 1))
                         for yr in years}
    coverage = pd.DataFrame(matrix).T
    coverage.columns = [str(y) for y in coverage.columns]
    write_table(coverage, "qa_coverage_matrix", index=True)

    fig, ax = figstyle.new_figure("double", height_mm=85)
    sns.heatmap(coverage.astype(float), ax=ax, cmap=figstyle.SEQUENTIAL_CMAP, vmin=0, vmax=100, annot=True, fmt=".0f",
                annot_kws={"size": 4.5}, linewidths=0.3, cbar_kws={"label": "Non-missing county-years (%)", "pad": 0.01})
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    save_qa_figure(fig, "qa_coverage_heatmap", source_note("Rural (non-large-metro) counties only; blank = column not in this panel."))


def run_balance():
    """How many county-years each year holds, and how many counties appear in every year."""
    df, _ = load_panel()
    by_year = (df.groupby("year").agg(n_county_year_obs=("fips", "size"), n_unique_fips=("fips", "nunique"))
               .reset_index().sort_values("year"))
    by_year["year"] = by_year["year"].astype(int)
    all_years = set(df["year"].dropna().astype(int).unique())
    in_every_year = df.groupby("fips")["year"].apply(lambda s: all_years.issubset(set(s.astype(int).tolist())))
    n_balanced, n_total = int(in_every_year.sum()), int(df["fips"].nunique())
    summary = pd.DataFrame([{
        "n_fips_total": n_total, "n_fips_in_all_years": n_balanced,
        "pct_balanced": round(n_balanced / n_total * 100, 1) if n_total else np.nan,
        "year_min": int(df["year"].min()), "year_max": int(df["year"].max()), "n_years_total": int(df["year"].nunique()),
    }])
    write_table(by_year, "qa_panel_balance_by_year")
    write_table(summary, "qa_panel_balance_summary")


def run_cdc_suppression():
    """CDC suppresses small cells, so the crude rate is missing for most rural counties; document it so it is not read as a bug.

    The deaths-of-despair measure is a composite (overdose + suicide + alcohol-related); the panel has no suicide-only column.
    """
    df, _ = load_panel()
    if not all(c in df.columns for c in (CDC_DEATHS, CDC_CRUDE)):
        print("CDC suppression audit skipped: required columns not found.")
        return
    rows = []
    for year, group in df.groupby("year"):
        deaths, crude = to_numeric_series(group[CDC_DEATHS]), to_numeric_series(group[CDC_CRUDE])
        rows.append({"year": int(year), "n_county_rows": int(len(group)), "n_with_deaths_data": int(deaths.notna().sum()),
                     "n_with_crude_rate": int(crude.notna().sum()), "pct_crude_rate_coverage": round(crude.notna().mean() * 100, 1)})
    audit = pd.DataFrame(rows).sort_values("year")
    write_table(audit, "qa_cdc_suppression_audit")
    print(f"  CDC crude rate average coverage: {audit['pct_crude_rate_coverage'].mean():.1f}% (low by design: "
          "small rural counties are suppressed by CDC)")


def run_distributions():
    """Histograms of each outcome in the reference year: skew and outliers before modelling."""
    df, _ = load_panel()
    ref = df[df["year"] == REFERENCE_YEAR]
    present = {label: spec for label, spec in HIST_OUTCOMES.items() if spec[0] in ref.columns}
    n_cols = 3
    n_rows = int(np.ceil(len(present) / n_cols))
    fig, axes = figstyle.new_grid(n_rows, n_cols, width="double", height_mm=45 * n_rows)
    flat = axes.ravel()
    for i, (label, (col, xlabel)) in enumerate(present.items()):
        ax = flat[i]
        values = to_numeric_series(ref[col]).dropna()
        if values.empty:
            ax.set_visible(False)
            continue
        ax.hist(values, bins=HIST_BINS, color=figstyle.PALETTE["blue"], edgecolor="white", linewidth=0.3)
        ax.axvline(values.median(), color=figstyle.PALETTE["vermillion"], lw=0.8, ls="--", label=f"Median {format_stat(values.median())}")
        ax.axvline(values.mean(), color=figstyle.PALETTE["orange"], lw=0.8, ls=":", label=f"Mean {format_stat(values.mean())}")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.5)       # headroom so the legend never sits on the bars
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,g}"))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counties")
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
        ax.text(0.97, 0.97, f"n = {len(values):,}", transform=ax.transAxes, ha="right", va="top", fontsize=5, color="0.35")
        figstyle.panel_label(ax, "abcdefghi"[i])
    for ax in flat[len(present):]:
        ax.set_visible(False)
    save_qa_figure(fig, f"qa_outcome_distributions_{REFERENCE_YEAR}",
                   source_note(f"{REFERENCE_YEAR} cross-section of rural counties; dashed line = median, dotted = mean."))


def run_cafo_percapita():
    """Why log(CAFO operations per 10,000 residents + 1): raw counts are right-skewed and track county size."""
    df, _ = load_panel()
    if "population" not in df.columns or "cafo_total_ops_all_animals" not in df.columns:
        print("CAFO per-capita distribution skipped: required columns not found.")
        return
    ref = df[df["year"] == REFERENCE_YEAR].copy()
    pop, cafo = to_numeric_series(ref["population"]), to_numeric_series(ref["cafo_total_ops_all_animals"])
    valid = pop.notna() & cafo.notna() & (pop > 0)
    per10k = (cafo[valid] / pop[valid]) * 10_000
    log_per10k = np.log1p(per10k)

    fig, axes = figstyle.new_grid(1, 3, width="double", height_mm=60)
    panels = [(cafo[valid], "Total CAFO operations, all animals", "{:.0f}", figstyle.PALETTE["sky"]),
              (per10k, "CAFO operations per 10,000 residents", "{:.1f}", figstyle.PALETTE["blue"]),
              (log_per10k, "log(CAFO operations per 10,000 residents + 1)", "{:.2f}", figstyle.PALETTE["green"])]
    for i, (values, xlabel, fmt, color) in enumerate(panels):
        ax = axes[0, i]
        ax.hist(values, bins=CAFO_BINS, color=color, edgecolor="white", linewidth=0.3)
        ax.axvline(values.median(), color=figstyle.PALETTE["vermillion"], lw=0.8, ls="--", label="Median " + fmt.format(values.median()))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counties")
        ax.legend()
        figstyle.panel_label(ax, "abc"[i])
    save_qa_figure(fig, "qa_cafo_percapita_distribution",
                   source_note(f"{REFERENCE_YEAR} cross-section; log per-capita is the scatter-plot x-axis."))

    outliers = ref.loc[valid, ["fips", "year"]].copy()
    outliers["cafo_total_ops"] = cafo[valid].values
    outliers["population"] = pop[valid].values
    outliers["cafo_per10k_pop"] = per10k.values
    outliers["log_cafo_per10k"] = log_per10k.values
    threshold = per10k.quantile(OUTLIER_QUANTILE)
    outliers = outliers[outliers["cafo_per10k_pop"] >= threshold].sort_values("cafo_per10k_pop", ascending=False)
    write_table(outliers, "qa_cafo_percapita_outliers_p99")
    print(f"  p99 threshold: {threshold:.1f} operations per 10,000 residents")


# ---------------------------------------------------------------------------------------------------------------------
# 2a section 1: CDC crude-rate sense check
# ---------------------------------------------------------------------------------------------------------------------
def summarize_cdc_differences(g):
    """Distribution of (crude rate recomputed from Census population) - (CDC crude rate) for one group of county-years."""
    d = g["diff_census_minus_cdc"]
    ad = g["abs_diff_census_minus_cdc"]
    z = g["z_diff_from_zero_sd"]
    poppct = g["population_pct_diff_census_vs_cdc"]
    recalc = g["crude_rate_recalc_census_pop"]
    cdc = g["cdc_crude_rate"]
    return pd.Series({
        "n_rows": int(len(g)),
        "n_valid_diff": int(d.notna().sum()),
        "mean_diff": float(d.mean()) if d.notna().any() else np.nan,
        "median_diff": float(d.median()) if d.notna().any() else np.nan,
        "sd_diff": float(d.std(ddof=1)) if d.notna().sum() > 1 else np.nan,
        "mae_diff": float(ad.mean()) if ad.notna().any() else np.nan,
        "rmse_diff": float(np.sqrt(np.mean(d.dropna() ** 2))) if d.notna().any() else np.nan,
        "p90_abs_diff": float(ad.quantile(0.90)) if ad.notna().any() else np.nan,
        "p95_abs_diff": float(ad.quantile(0.95)) if ad.notna().any() else np.nan,
        "share_abs_diff_le_0_5": float((ad <= 0.5).mean() * 100) if ad.notna().any() else np.nan,
        "share_abs_diff_le_1_0": float((ad <= 1.0).mean() * 100) if ad.notna().any() else np.nan,
        "share_abs_diff_le_2_0": float((ad <= 2.0).mean() * 100) if ad.notna().any() else np.nan,
        "share_abs_z_gt_1": float((z.abs() > 1).mean() * 100) if z.notna().any() else np.nan,
        "share_abs_z_gt_2": float((z.abs() > 2).mean() * 100) if z.notna().any() else np.nan,
        "share_abs_z_gt_3": float((z.abs() > 3).mean() * 100) if z.notna().any() else np.nan,
        "mean_population_pct_diff": float(poppct.mean()) if poppct.notna().any() else np.nan,
        "median_population_pct_diff": float(poppct.median()) if poppct.notna().any() else np.nan,
        "corr_recalc_vs_cdc": float(recalc.corr(cdc)) if recalc.notna().sum() > 1 and cdc.notna().sum() > 1 else np.nan,
    })


def run_cdc_sensecheck():
    """Recompute deaths / population x 100,000 and compare with the CDC crude rate. Diagnostic columns stay in these tables."""
    panel, _ = load_panel()
    missing = [c for c in (CDC_DEATHS, CDC_CRUDE, CENSUS_POP) if c not in panel.columns]
    if missing:
        print(f"WARNING: skipping CDC sense check; missing columns: {missing}")
        return
    df = panel.copy()                        # the diagnostic columns are added to a copy, never to the panel
    df["cdc_deaths"] = to_numeric_series(df[CDC_DEATHS])
    df["cdc_population"] = to_numeric_series(df[CDC_POP]) if CDC_POP in df.columns else np.nan
    df["cdc_crude_rate"] = to_numeric_series(df[CDC_CRUDE])
    df["census_population"] = to_numeric_series(df[CENSUS_POP])
    df["crude_rate_recalc_census_pop"] = np.where(df["census_population"] > 0,
                                                  df["cdc_deaths"] / df["census_population"] * 100_000.0, np.nan)
    df["crude_rate_recalc_cdc_pop"] = np.where(df["cdc_population"] > 0,
                                               df["cdc_deaths"] / df["cdc_population"] * 100_000.0, np.nan)
    df["diff_census_minus_cdc"] = df["crude_rate_recalc_census_pop"] - df["cdc_crude_rate"]
    df["abs_diff_census_minus_cdc"] = df["diff_census_minus_cdc"].abs()
    df["pct_diff_census_vs_cdc"] = np.where(df["cdc_crude_rate"] != 0,
                                            (df["diff_census_minus_cdc"] / df["cdc_crude_rate"]) * 100.0, np.nan)
    df["population_diff_census_minus_cdc"] = df["census_population"] - df["cdc_population"]
    df["population_pct_diff_census_vs_cdc"] = np.where(
        df["cdc_population"] != 0, (df["population_diff_census_minus_cdc"] / df["cdc_population"]) * 100.0, np.nan)
    sd_all = df["diff_census_minus_cdc"].std(ddof=1)
    df["z_diff_from_zero_sd"] = df["diff_census_minus_cdc"] / sd_all if (pd.notna(sd_all) and sd_all > 0) else np.nan
    year_sd = df.groupby("year")["diff_census_minus_cdc"].transform(lambda s: s.std(ddof=1))
    df["z_diff_from_zero_sd_within_year"] = np.where(year_sd > 0, df["diff_census_minus_cdc"] / year_sd, np.nan)

    rename_map = {
        "crude_rate_recalc_census_pop": "crude_rate_recalc_censuspop_cdcsense",
        "crude_rate_recalc_cdc_pop": "crude_rate_recalc_cdcpop_cdcsense",
        "diff_census_minus_cdc": "crude_rate_diff_census_minus_cdc_cdcsense",
        "abs_diff_census_minus_cdc": "crude_rate_abs_diff_census_minus_cdc_cdcsense",
        "pct_diff_census_vs_cdc": "crude_rate_pct_diff_census_vs_cdc_cdcsense",
        "z_diff_from_zero_sd": "crude_rate_diff_z_from_zero_sd_cdcsense",
        "z_diff_from_zero_sd_within_year": "crude_rate_diff_z_from_zero_sd_within_year_cdcsense",
        "population_diff_census_minus_cdc": "population_diff_census_minus_cdc_cdcsense",
        "population_pct_diff_census_vs_cdc": "population_pct_diff_census_vs_cdc_cdcsense",
    }
    for src, dst in rename_map.items():
        df[dst] = df[src]

    compared = df[df["cdc_deaths"].notna() & df["cdc_crude_rate"].notna()
                  & df["census_population"].notna() & (df["census_population"] > 0)].copy()
    overall = summarize_cdc_differences(compared).to_frame().T
    overall.insert(0, "source_merged_file", Path(merged_path("*_panel.csv")).name)
    overall.insert(1, "year_min", int(compared["year"].min()) if not compared.empty else np.nan)
    overall.insert(2, "year_max", int(compared["year"].max()) if not compared.empty else np.nan)
    by_year = (compared.groupby("year", as_index=False).apply(summarize_cdc_differences, include_groups=False)
               .reset_index().drop(columns=["index"], errors="ignore").sort_values("year"))

    outliers = compared.copy()
    outliers["abs_z"] = outliers["crude_rate_diff_z_from_zero_sd_cdcsense"].abs()
    keep = ["fips", "year", COUNTY_NAME if COUNTY_NAME in outliers.columns else None, "cdc_deaths", "census_population",
            "cdc_population", "cdc_crude_rate", "crude_rate_recalc_censuspop_cdcsense", "crude_rate_recalc_cdcpop_cdcsense",
            "crude_rate_diff_census_minus_cdc_cdcsense", "crude_rate_abs_diff_census_minus_cdc_cdcsense",
            "crude_rate_pct_diff_census_vs_cdc_cdcsense", "crude_rate_diff_z_from_zero_sd_cdcsense",
            "crude_rate_diff_z_from_zero_sd_within_year_cdcsense", "population_pct_diff_census_vs_cdc_cdcsense", "abs_z"]
    keep = [c for c in keep if c is not None and c in outliers.columns]
    outliers = outliers[keep].sort_values("abs_z", ascending=False).head(1000)

    write_table(overall, "qa_cdc_cruderate_sensecheck_overall")
    write_table(by_year, "qa_cdc_cruderate_sensecheck_by_year")
    write_table(outliers, "qa_cdc_cruderate_sensecheck_outliers_top1000")
    print(f"  Rows compared: {len(compared):,} | Global SD(diff): {float(sd_all):.4f}")


# ---------------------------------------------------------------------------------------------------------------------
# Key-set and source checks (2a section 11): checks that can fail, with their own output tables
# ---------------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_rural_kept_keys():
    """(file name, kept keys): the (fips, year) pairs the rural-key file marks non_large_metro == 1."""
    path = latest_file_glob(str(CLEAN_DIR), "*rural-key*.csv")
    rural = normalize_panel_key(read_and_prepare(path))
    rural["non_large_metro"] = pd.to_numeric(rural.get("non_large_metro"), errors="coerce").astype("Int64")
    kept = rural.loc[rural["non_large_metro"] == 1, ["fips", "year"]].drop_duplicates()
    return Path(path).name, kept


def run_coverage_check():
    """Does the panel hold exactly the counties and county-years the rural key keeps? Writes 1/0 flags the memo reads."""
    df, _ = load_panel()
    rural_name, kept = load_rural_kept_keys()
    panel_keys = df[["fips", "year"]].drop_duplicates()
    row = pd.DataFrame([{
        "run_date": TODAY,
        "merged_file": Path(merged_path("*_panel.csv")).name,
        "rural_key_file": rural_name,
        "merged_rows": int(len(df)),
        "merged_unique_counties": int(df["fips"].nunique()),
        "rural_kept_rows": int(len(kept)),
        "rural_kept_unique_counties": int(kept["fips"].nunique()),
        "county_count_match": int(df["fips"].nunique() == kept["fips"].nunique()),
        "key_set_exact_match": int(panel_keys.merge(kept, on=["fips", "year"], how="outer", indicator=True)
                                   ["_merge"].eq("both").all()),
    }])
    write_table(row, "county_coverage_check")
    if not (row["county_count_match"].iloc[0] and row["key_set_exact_match"].iloc[0]):
        print("WARNING: panel keys differ from the rural-key kept keys; see county_coverage_check")


def run_mental_fill():
    """Share of county-year rows with a value in any column whose name contains 'mental'."""
    df, _ = load_panel()
    mental_cols = [c for c in df.columns if "mental" in c.lower()]
    any_mental = df[mental_cols].notna().any(axis=1) if mental_cols else pd.Series(False, index=df.index)
    write_table(pd.DataFrame([{
        "run_date": TODAY, "merged_file": Path(merged_path("*_panel.csv")).name,
        "mental_column_count": int(len(mental_cols)), "n_rows": int(len(df)),
        "rows_with_any_mental_data": int(any_mental.sum()),
        "share_rows_with_any_mental_data_pct": float(any_mental.mean() * 100),
    }]), "mental_fill_share")


def build_premerged_cafo_wide(kept):
    """Rebuild the panel's cafo_{commodity}_{size} columns from the compact file, for rural-kept keys only."""
    path = latest_file_glob(str(CLEAN_DIR), "*_cafo_ops_by_size_compact.csv")
    pre = normalize_panel_key(pd.read_csv(path, low_memory=False))
    for col in ("commodity_desc", *CAFO_SIZES):
        if col not in pre.columns:
            raise RuntimeError(f"Missing column in CAFO compact: {col}")
    pre["commodity_desc"] = pre["commodity_desc"].astype("string").str.strip().str.lower()
    pre = pre[pre["commodity_desc"].isin(CAFO_COMMODITIES)].copy()
    if "class_desc" in pre.columns:
        pre["class_desc"] = pre["class_desc"].astype("string").str.strip().str.lower()
        pre = pre[pre["class_desc"] == pre["commodity_desc"].map(CAFO_CANONICAL_CLASS)].copy()
    for col in CAFO_SIZES:
        pre[col] = to_numeric_series(pre[col])
    pre = pre.merge(kept, on=["fips", "year"], how="inner")
    grouped = pre.groupby(["fips", "year", "commodity_desc"], as_index=False)[list(CAFO_SIZES)].sum(min_count=1)
    wide = grouped.pivot_table(index=["fips", "year"], columns="commodity_desc", values=list(CAFO_SIZES), aggfunc="sum")
    wide.columns = [f"cafo_{commodity}_{size}" for size, commodity in wide.columns]
    wide = wide.reset_index()
    animal_cols = [f"cafo_{cmd}_{sz}" for cmd in CAFO_COMMODITIES for sz in CAFO_SIZES]
    for col in animal_cols:
        if col not in wide.columns:
            wide[col] = np.nan
    wide["cafo_total_ops_all_animals"] = wide[animal_cols].sum(axis=1, min_count=1)
    wide["cafo_total_ops_chickens"] = wide[[f"cafo_chickens_{sz}" for sz in CAFO_SIZES]].sum(axis=1, min_count=1)
    return wide, animal_cols


def run_cafo_crosscheck():
    """Do the panel's CAFO counts equal the same counts rebuilt from the compact file? Exact match is expected."""
    df, _ = load_panel()
    _, kept = load_rural_kept_keys()
    wide, animal_cols = build_premerged_cafo_wide(kept)
    check_cols = [*animal_cols, "cafo_total_ops_all_animals", "cafo_total_ops_chickens"]
    panel_side = df[["fips", "year", *[c for c in check_cols if c in df.columns]]].copy()
    merged = panel_side.merge(wide, on=["fips", "year"], how="left", suffixes=("_merged", "_pre"))
    rows = []
    for col in check_cols:
        in_panel = to_numeric_series(merged.get(f"{col}_merged", pd.Series(dtype=float)))
        rebuilt = to_numeric_series(merged.get(f"{col}_pre", pd.Series(dtype=float)))
        both = in_panel.notna() & rebuilt.notna()
        diff = (in_panel - rebuilt).abs()
        rows.append({"variable": col, "n_keys_compared": int(both.sum()),
                     "pct_exact_match_on_compared_keys": float((diff[both] == 0).mean() * 100) if both.any() else np.nan,
                     "max_abs_diff": float(diff[both].max()) if both.any() else np.nan,
                     "merged_sum": float(in_panel.sum(skipna=True)), "premerged_sum": float(rebuilt.sum(skipna=True))})
    result = pd.DataFrame(rows)
    write_table(result, "cafo_commodity_size_crosscheck_vs_premerged")
    if (result["pct_exact_match_on_compared_keys"] < 100).any():
        print("WARNING: panel CAFO counts differ from the compact file; see cafo_commodity_size_crosscheck_vs_premerged")


# ---------------------------------------------------------------------------------------------------------------------
# 2a section 2: correlation tables, FSIS column QA, QA memo
# ---------------------------------------------------------------------------------------------------------------------
def pairwise_correlations(frame, columns, method, min_n=MIN_CORR_N):
    """Long table of pairwise correlations over the rows both variables share; pairs with fewer than min_n rows are skipped."""
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            both = frame[a].notna() & frame[b].notna()
            n = int(both.sum())
            if n < min_n:
                continue
            corr = frame.loc[both, a].corr(frame.loc[both, b], method=method)
            if pd.isna(corr):
                continue
            rows.append({"var_a": a, "var_b": b, "corr": float(corr), "abs_corr": abs(float(corr)),
                         "n_obs_pairwise": n, "method": method})
    out = pd.DataFrame(rows)
    return out.sort_values("abs_corr", ascending=False).reset_index(drop=True) if not out.empty else out


def fsis_column_qa(fsis):
    """Distribution of the 12 FSIS count columns; a column whose p99 or max exceeds FSIS_RELIABLE_MAX is flagged."""
    rows = []
    for col in FSIS_QA_COLS:
        if col not in fsis.columns:
            continue
        values = to_numeric_series(fsis[col])
        n = values.notna().sum()
        p99 = safe_float(values.quantile(0.99)) if n else np.nan
        vmax = safe_float(values.max()) if n else np.nan
        rows.append({
            "variable": col, "non_missing_n": int(n), "fill_pct": float(values.notna().mean() * 100),
            "p50": safe_float(values.quantile(0.50)) if n else np.nan,
            "p90": safe_float(values.quantile(0.90)) if n else np.nan, "p99": p99, "max": vmax,
            "share_gt_1000_pct": safe_float((values > FSIS_RELIABLE_MAX).mean() * 100) if n else np.nan,
            "qa_reliable_for_now": int((pd.notna(p99) and p99 <= FSIS_RELIABLE_MAX) and (pd.notna(vmax) and vmax <= FSIS_RELIABLE_MAX)),
        })
    return pd.DataFrame(rows).sort_values(["qa_reliable_for_now", "share_gt_1000_pct"], ascending=[False, True])


def run_correlations():
    """Correlation tables for CAFO, mental-health and population variables; FSIS column QA; the QA memo."""
    panel, _ = load_panel()
    fsis = load_fsis_panel()
    corr_cols = [c for c in [*CAFO_CORR_COLS, *MENTAL_CORR_COLS, CENSUS_POP] if c in panel.columns]
    coverage_only_cols = [c for c in ("deaths_despair",) if c in panel.columns]   # in the coverage table, not the correlations
    df = panel[["fips", "year", *corr_cols, *coverage_only_cols]].copy()
    for col in [*corr_cols, *coverage_only_cols]:
        df[col] = to_numeric_series(df[col])

    def family(col):
        return "cafo" if col in CAFO_CORR_COLS else "mental" if col in MENTAL_CORR_COLS else "population"

    write_table(pd.DataFrame([{"variable": c, "family": family(c)} for c in corr_cols]), "correlation_vars_manifest")

    coverage_rows = []
    for year, group in df.groupby("year", as_index=False):
        row = {"year": int(year)}
        for col in ["cafo_total_ops_all_animals", "cafo_total_ops_chickens", "poor_mental_health_days",
                    "frequent_mental_distress_per100k", "deaths_despair", CENSUS_POP]:
            if col in group.columns:
                row[f"fill_pct__{col}"] = float(group[col].notna().mean() * 100)
        coverage_rows.append(row)
    fsis_coverage = pd.DataFrame([
        {"year": int(year), "fill_pct__n_unique_establishments_fsis":
            float(group["n_unique_establishments_fsis"].notna().mean() * 100) if "n_unique_establishments_fsis" in group.columns
            else np.nan}
        for year, group in fsis.groupby("year", as_index=False)])
    write_table(pd.DataFrame(coverage_rows).sort_values("year").merge(fsis_coverage, on="year", how="left"),
                "qa_key_variable_coverage_by_year")

    fsis_qa = fsis_column_qa(fsis)
    write_table(fsis_qa, "qa_fsis_12_column_check")

    for name, (first, last) in CORR_WINDOWS.items():
        sub = df[df["year"].between(first, last, inclusive="both")]
        use_cols = [c for c in corr_cols if sub[c].notna().any()]
        if not use_cols:
            continue
        for method in ("pearson", "spearman"):
            long = pairwise_correlations(sub[use_cols], use_cols, method)
            write_table(long, f"correlation_long_{name}_{method}")
            write_table(long.head(30), f"correlation_top30_{name}_{method}")
            write_table(sub[use_cols].corr(method=method, min_periods=MIN_CORR_N), f"correlation_matrix_{name}_{method}", index=True)

    write_qa_memo(panel, fsis_qa)


def write_qa_memo(panel, fsis_qa):
    """Memo of the key-set checks, FSIS column QA and correlation set-up.

    The key-set lines read this script's own newest county_coverage_check and CAFO cross-check tables
    (run those sections first, as the default order does); "n/a" means they have not been written yet.
    """
    def newest_own_table(stem):
        hits = sorted(TABLE_DIR.glob(f"*_{stem}.csv"))
        return pd.read_csv(hits[-1]) if hits else pd.DataFrame()

    coverage_check = newest_own_table("county_coverage_check")
    crosscheck = newest_own_table("cafo_commodity_size_crosscheck_vs_premerged")
    county_match = key_match = cafo_cross = "n/a"
    if not coverage_check.empty:
        county_match = "PASS" if int(coverage_check["county_count_match"].iloc[0]) == 1 else "FAIL"
        key_match = "PASS" if int(coverage_check["key_set_exact_match"].iloc[0]) == 1 else "FAIL"
    if not crosscheck.empty and "pct_exact_match_on_compared_keys" in crosscheck.columns:
        cafo_cross = (f"{crosscheck['pct_exact_match_on_compared_keys'].min():.1f}%-"
                      f"{crosscheck['pct_exact_match_on_compared_keys'].max():.1f}% exact-match")
    reliable = fsis_qa.loc[fsis_qa["qa_reliable_for_now"] == 1, "variable"].tolist()
    flagged = fsis_qa.loc[fsis_qa["qa_reliable_for_now"] == 0, "variable"].tolist()
    lines = [
        f"# QA Memo: Panel + Correlation Tables ({TODAY})", "",
        "## 1) Scope",
        f"- Merged file: `{Path(merged_path('*_panel.csv')).name}`",
        f"- Rows: `{len(panel):,}` | Counties: `{panel['fips'].nunique():,}` | "
        f"Years: `{int(panel['year'].min())}-{int(panel['year'].max())}`", "",
        "## 2) Key QA Checks",
        f"- County-count check vs rural kept keys: **{county_match}**",
        f"- Exact key-set match vs rural kept keys: **{key_match}**",
        f"- CAFO commodity x size merged-vs-premerged cross-check: **{cafo_cross}**", "",
        "## 3) FSIS 12-Column QA",
        f"- Reliable ({len(reliable)}): {', '.join(f'`{c}`' for c in reliable) or 'none'}",
        f"- Flagged ({len(flagged)}): {', '.join(f'`{c}`' for c in flagged) or 'none'}", "",
        "## 4) Correlation Tables",
        f"- Pairwise minimum n: `{MIN_CORR_N}`. Windows: 2010-2020, 2017-2023, 2017 cross-section.",
        "- FSIS correlations not included (FSIS only in panel_fsis.csv, 2017+).", "",
    ]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{TODAY}_qa_memo_panel_qa.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved:", path)


# ---------------------------------------------------------------------------------------------------------------------
# 2a section 4: missingness
# ---------------------------------------------------------------------------------------------------------------------
def heatmap_of_missing(frame, stem, width_mm_height, note, cmap=figstyle.SEQUENTIAL_CMAP):
    """Year x variable heatmap of missing share (%), labelled with the variable names."""
    fig, ax = figstyle.new_figure("double", height_mm=width_mm_height)
    sns.heatmap(frame, ax=ax, cmap=cmap, vmin=0, vmax=100, xticklabels=True, yticklabels=True,
                cbar_kws={"label": "Missing (%)", "pad": 0.01})
    ax.set_xlabel("Variable")
    ax.set_ylabel("Year")
    ax.tick_params(axis="x", rotation=90, labelsize=4.5)
    ax.tick_params(axis="y", rotation=0)
    save_qa_figure(fig, stem, note)


def run_missingness():
    """Missingness by variable and year, row completeness, and the figures behind them."""
    df, _ = load_panel()
    analysis_cols = [c for c in df.columns if c not in ("fips", "year")]
    n_obs = len(df)

    overall = pd.DataFrame({
        "variable": analysis_cols,
        "non_missing_n": [int(df[c].notna().sum()) for c in analysis_cols],
        "missing_n": [int(df[c].isna().sum()) for c in analysis_cols],
        "dtype": [str(df[c].dtype) for c in analysis_cols],
        "nunique": [int(df[c].nunique(dropna=True)) for c in analysis_cols],
    })
    overall["fill_pct"] = overall["non_missing_n"] / n_obs * 100
    overall["missing_pct"] = overall["missing_n"] / n_obs * 100
    overall = overall.sort_values("missing_pct", ascending=False)

    missing_by_year = df.groupby("year")[analysis_cols].apply(lambda x: x.isna().mean() * 100).sort_index()
    row_fill_pct = df[analysis_cols].notna().mean(axis=1) * 100
    row_fill = pd.DataFrame({"fips": df["fips"], "year": df["year"], "row_fill_pct": row_fill_pct})
    row_fill_summary = row_fill_pct.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_frame("value")
    share_by_year = pd.DataFrame({"year": sorted(df["year"].dropna().astype(int).unique())})
    share_by_year["missing_share_pct"] = [float(df.loc[df["year"] == y, analysis_cols].isna().mean().mean() * 100)
                                          for y in share_by_year["year"]]
    share_by_year["non_missing_share_pct"] = 100 - share_by_year["missing_share_pct"]

    write_table(overall, "merged_missingness_overall")
    write_table(missing_by_year.reset_index(), "merged_missingness_pct_by_year")
    write_table((100 - missing_by_year).reset_index(), "merged_missingness_fill_by_year")
    write_table(row_fill, "merged_missingness_row_fill_distribution")
    write_table(row_fill_summary.reset_index(), "merged_missingness_row_fill_summary")
    write_table(share_by_year, "missingness_share_by_year")

    note = source_note()
    fig, ax = figstyle.new_figure("single", height_mm=55)
    ax.plot(share_by_year["year"], share_by_year["missing_share_pct"], marker="o", color=figstyle.PALETTE["vermillion"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Missing cells, all variables (%)")
    save_qa_figure(fig, "missing_share_by_year", note)

    top = overall.loc[overall["missing_pct"].between(1, 99), "variable"].head(TOP_MISSING_VARS).tolist()
    if top:
        heatmap_of_missing(missing_by_year[top], "missingness_heatmap_top60", 140,
                           f"{note} The {TOP_MISSING_VARS} variables with the most missing data (1-99% missing).")

    families = pd.Series({c: detect_family(c) for c in analysis_cols})
    for fam in sorted(families.unique()):
        for chunk_idx, chunk in chunked(families[families == fam].index.tolist(), FAMILY_CHUNK):
            heatmap_of_missing(missing_by_year[chunk], f"missingness_{safe_name(fam)}_chunk{chunk_idx}", 120,
                               f"{note} Variable family: {fam}, chunk {chunk_idx}.")

    fig, ax = figstyle.new_figure("single", height_mm=55)
    ax.hist(row_fill["row_fill_pct"], bins=HIST_BINS, color=figstyle.PALETTE["blue"], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Non-missing variables in the county-year row (%)")
    ax.set_ylabel("County-years")
    save_qa_figure(fig, "row_completeness_hist_all", note)

    fig, ax = figstyle.new_figure("double", height_mm=70)
    sns.boxplot(data=row_fill, x="year", y="row_fill_pct", color=figstyle.PALETTE["sky"], fliersize=1, linewidth=0.5, ax=ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Non-missing variables in the row (%)")
    ax.tick_params(axis="x", rotation=90)
    save_qa_figure(fig, "row_completeness_box_by_year", note)
    print("  Missingness outputs complete.")


# ---------------------------------------------------------------------------------------------------------------------
# Coverage memo (1a): lists the outputs above
# ---------------------------------------------------------------------------------------------------------------------
def run_memo():
    """Short memo that lists the 1a outputs. The findings paragraphs are written by hand for the panel read on the date shown."""
    df, _ = load_panel()
    d = TODAY
    lines = [
        f"# Mental Coverage Audit ({d})", "",
        "## Scope",
        f"- Merged input: `{Path(merged_path('*_panel.csv')).name}`",
        "- Unit: county-year (`fips`, `year`)",
        f"- Rows: `{len(df):,}` | Counties: `{df['fips'].nunique():,}` | Years: `{int(df['year'].min())}-{int(df['year'].max())}`", "",
        "## Outputs",
        *[f"- `{d}_{stem}.csv`" for stem in (
            "qa_panel_integrity_mental_focus", "qa_panel_source_duplication_counts", "qa_mental_focus_variable_inventory",
            "qa_mental_outcome_core_overall", "qa_mental_outcome_core_by_year", "qa_mental_outcome_counties_any",
            "qa_mental_crosssource_duplicate_check")], "",
        "## Notes",
        "- Overall fill rates are over the full panel window (2000-2023), so variables that start in later years will look "
        "lower in the overall metric.",
        f"- Use `{d}_qa_mental_outcome_core_by_year.csv` for year-appropriate coverage interpretation.",
        "", "---", f"## QA Extension ({d})", "",
        "### Coverage heatmap",
        f"- Files: `{d}_qa_coverage_heatmap.png|pdf` + `{d}_qa_coverage_matrix.csv`",
        "- Findings noted by hand on an earlier panel; re-read the matrix before quoting them:",
        "  - `poor_mental_health_days` (CHR): usable from 2010 (~90%) onward.",
        "  - `frequent_mental_distress` (CHR): only available 2016+; not suitable for 2005-2015 panel.",
        "  - `excessive_drinking` (CHR): starts 2011 (~80%), full 2016+.",
        "  - `violent_crime_raw_value` (CHR): starts 2010 (43%), reliable from 2012 (~90%).",
        "  - `crude_rate` deaths of despair (CDC): <15% most years; see the suppression audit.",
        "  - FSIS establishments: only 2017, cross-section only.",
        "  - Recommended panel window for outcome analysis: **2010-2015**.", "",
        "### Panel balance",
        f"- Files: `{d}_qa_panel_balance_by_year.csv`, `{d}_qa_panel_balance_summary.csv`", "",
        "### CDC suppression audit",
        f"- File: `{d}_qa_cdc_suppression_audit.csv`",
        "- Low coverage is expected, not a bug: CDC suppresses crude rates for counties with fewer than 10 deaths in a cell,",
        "  so the crude rate is missing for most small rural counties. The raw death count is retained.",
        "- There is no dedicated suicide-cause column in the current pipeline. The CDC deaths-of-despair files are a",
        "  composite (drug overdose + suicide + alcohol-related). Suicide-specific analysis needs a separate CDC WONDER pull",
        "  by ICD-10 cause code (X60-X84, intentional self-harm).", "",
        "### Outcome distributions",
        f"- File: `{d}_qa_outcome_distributions_{REFERENCE_YEAR}.png|pdf` ({REFERENCE_YEAR} reference cross-section)",
        "- Check right skew in assault and homicide variables before regression modeling.", "",
        "### CAFO per-capita distribution",
        f"- Files: `{d}_qa_cafo_percapita_distribution.png|pdf`, `{d}_qa_cafo_percapita_outliers_p99.csv`",
        "- Raw CAFO operation counts are heavily right-skewed. Scatter-plot x-axis: log(CAFO operations per 10,000 residents + 1).",
    ]
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{d}_qa_mental_coverage_memo.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved:", path)


SECTIONS = {
    "integrity": run_integrity, "inventory": run_inventory, "core_outcomes": run_core_outcomes,
    "crosssource": run_crosssource, "coverage": run_coverage, "balance": run_balance,
    "cdc_suppression": run_cdc_suppression, "distributions": run_distributions, "cafo_percapita": run_cafo_percapita,
    "cdc_sensecheck": run_cdc_sensecheck, "coverage_check": run_coverage_check, "mental_fill": run_mental_fill,
    "cafo_crosscheck": run_cafo_crosscheck, "correlations": run_correlations, "missingness": run_missingness,
    "memo": run_memo,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=sorted(SECTIONS), metavar="SECTION",
                        help="run only these sections (choices: " + ", ".join(SECTIONS) + ")")
    args = parser.parse_args()
    figstyle.apply_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in (args.only or list(SECTIONS)):
        print(f"\n--- {name} ---")
        SECTIONS[name]()
    print("\nDone. Tables:", TABLE_DIR, "| Figures:", FIG_DIR)


if __name__ == "__main__":
    main()
