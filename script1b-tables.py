#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script1b-tables.py -- every numeric / CSV table of the descriptive stage (no graphics).

Sections (one function each; run one with --only <name>):
  summary        summary statistics of all numeric panel variables + yearly means and growth   (from script2a section 5)
  concentration  CAFO concentration (HHI, Gini, top-10 share), state transitions, large entry/exit (script2a section 8)
  sumstats       summary statistics of CAFO size-bin counts (county-year, state-year) and FSIS counts (script2a section 11)
  species        large-CAFO species overlap: six tables                                        (from script4s)
  register       master variable register: xlsx + csv codebook                                 (from script4k)
Reads:   newest Data/merged/*_panel.csv (all sections), Data/merged/*_panel_fsis.csv (sumstats, register),
         Data/clean/*_cafo_ops_by_size_compact.csv (concentration), Data/clean/*.csv headers (register source map)
Writes:  Data/output/tables/<date>_1b_<stem>.csv ; the codebook to Data/output/codebooks/<date>_VARIABLE_REGISTER.{xlsx,csv}
Run:     PYTHONHASHSEED=0 python3 script1b-tables.py [--only summary concentration sumstats species register]
Rules:   species handling per D-36 (cattle contains dairy and beef, so it is not a species); a missing cafo count is
         zero from 2002 (D-34). The numbers are those of the retired scripts; this file moves them, it does not redefine them.
"""
import argparse
import glob
import warnings
from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd

from packages import os, db_data, codebooks_dir, tables_dir, out_file
from functions import latest_file_glob, normalize_panel_key, to_numeric_series
from script4_treatment import (load_panel, OUTCOMES, TREATMENTS, CONTROL_PRETREAT,
                               CONTROL_COMPREHENSIVE, CONTROL_MEDIATORS, COVARIATE_ORDER,
                               CORE_TREATMENTS, CORE_TREATMENT_LABELS, REPORT_TREATMENTS)

warnings.filterwarnings("ignore")

# =============================================================================
# Shared set-up
# =============================================================================
TODAY = date.today().strftime("%Y-%m-%d")
MERGED_DIR = os.path.join(db_data, "merged")
CLEAN_DIR = os.path.join(db_data, "clean")
OUT_TABLES, TAG = tables_dir, "1b"     # output-name tag (packages.py): <date>_1b_<stem>
# Codebooks keep their own folder (the running record of every variable listing), as script4k did.
OUT_CODEBOOKS = codebooks_dir   # names unchanged: <date>_VARIABLE_REGISTER.{xlsx,csv}

SPECIES = ["dairy", "beef", "hogs", "chickens"]
ALL_COLUMNS = SPECIES + ["cattle"]      # cattle is read only for the appendix check
CENSUS_YEARS = [2002, 2007, 2012, 2017, 2022]
FIRST_YEAR = min(CENSUS_YEARS)          # 2000-01 are unmeasured and left out of the species tables

# Compact CAFO file: keep the canonical class of each commodity (one row per county, year, commodity)
CANONICAL_CLASS = {"cattle": "incl calves", "hogs": "all classes", "chickens": "layers"}


def write_table(frame, stem):
    """Write one dated CSV to the script's table folder and return its path."""
    path = out_file(OUT_TABLES, TAG, f"{TODAY}_{stem}.csv")
    frame.to_csv(path, index=False)
    print(f"  saved: {path}")
    return path


def load_qa_panel(panel_path):
    """The panel as the QA script read it: keys normalised, duplicate columns and key-duplicate rows dropped."""
    df = pd.read_csv(panel_path, low_memory=False)
    df = normalize_panel_key(df, dropna=False)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.dropna(subset=["fips", "year"]).copy()
    n_dup = int(df.duplicated(subset=["fips", "year"], keep=False).sum())
    if n_dup:
        print(f"  WARNING: {n_dup:,} rows share a (fips, year) key; keeping the first of each, as the QA script did")
        df = df.drop_duplicates(subset=["fips", "year"], keep="first").copy()
    df = df.sort_values(["fips", "year"]).reset_index(drop=True)
    print(f"panel: {os.path.basename(panel_path)}  {len(df):,} rows | {df['fips'].nunique():,} counties | "
          f"{int(df['year'].min())}-{int(df['year'].max())}")
    return df


# =============================================================================
# 1. Summary statistics (script2a section 5)
# =============================================================================
def numeric_candidates(df, cols, min_parse_rate=0.8):
    """Columns that are numeric or parse as numeric for >= min_parse_rate of their non-missing cells."""
    out, rows = {}, []
    for c in cols:
        s = df[c]
        n_raw = int(s.notna().sum())
        s_num = to_numeric_series(s)
        n_num = int(s_num.notna().sum())
        parse_rate = (n_num / n_raw) if n_raw else 0
        is_numeric_like = pd.api.types.is_numeric_dtype(s) or (parse_rate >= min_parse_rate and n_num > 0)
        rows.append({"variable": c, "raw_non_missing_n": n_raw, "numeric_non_missing_n": n_num,
                     "parse_rate": parse_rate, "numeric_like": int(is_numeric_like)})
        if is_numeric_like:
            out[c] = s_num
    meta = pd.DataFrame(rows).sort_values(["numeric_like", "parse_rate"], ascending=[False, False])
    return out, meta


def summary_statistics(panel_path):
    """Missingness overview and summary statistics for every non-key column; yearly means, year-on-year change, growth."""
    print("\n--- summary: summary statistics ---")
    df = load_qa_panel(panel_path)
    analysis_cols = [c for c in df.columns if c not in ["fips", "year"]]
    n_obs = len(df)

    missing_overall = pd.DataFrame({
        "variable":      analysis_cols,
        "non_missing_n": [int(df[c].notna().sum()) for c in analysis_cols],
        "missing_n":     [int(df[c].isna().sum()) for c in analysis_cols],
        "dtype":         [str(df[c].dtype) for c in analysis_cols],
        "nunique":       [int(df[c].nunique(dropna=True)) for c in analysis_cols],
    })
    missing_overall["fill_pct"] = missing_overall["non_missing_n"] / n_obs * 100
    missing_overall["missing_pct"] = missing_overall["missing_n"] / n_obs * 100
    missing_overall = missing_overall.sort_values("missing_pct", ascending=False)

    numeric_map, numeric_meta = numeric_candidates(df, analysis_cols)
    numeric_df = pd.DataFrame(numeric_map, index=df.index)

    stats_rows = []
    for c in numeric_df.columns:
        s = numeric_df[c]
        any_obs = s.notna().any()
        stats_rows.append({
            "variable": c, "count": int(s.notna().sum()), "missing_n": int(s.isna().sum()),
            "fill_pct": float(s.notna().mean() * 100),
            "mean": float(s.mean()) if any_obs else np.nan,
            "std": float(s.std()) if any_obs else np.nan,
            "min": float(s.min()) if any_obs else np.nan,
            "p01": float(s.quantile(0.01)) if any_obs else np.nan,
            "p25": float(s.quantile(0.25)) if any_obs else np.nan,
            "median": float(s.median()) if any_obs else np.nan,
            "p75": float(s.quantile(0.75)) if any_obs else np.nan,
            "p99": float(s.quantile(0.99)) if any_obs else np.nan,
            "max": float(s.max()) if any_obs else np.nan,
            "nunique": int(s.nunique(dropna=True)),
            "zeros_n": int((s == 0).sum()) if any_obs else 0,
        })
    summary_numeric = pd.DataFrame(stats_rows).sort_values(["fill_pct", "std"], ascending=[False, False])
    summary_all = missing_overall.merge(summary_numeric, how="left", on="variable")
    write_table(summary_all, "merged_summary_stats_all_variables")
    write_table(summary_numeric, "merged_summary_stats_numeric_variables")
    write_table(numeric_meta, "merged_summary_stats_numeric_parse_meta")

    if numeric_df.empty:
        return
    yearly_means = numeric_df.copy()
    yearly_means["year"] = df["year"].values
    yearly_means = yearly_means.groupby("year").mean(numeric_only=True).sort_index()
    yoy_pct = yearly_means.pct_change() * 100

    growth_rows = []
    for c in yearly_means.columns:
        s = yearly_means[c].dropna()
        if s.empty:
            growth_rows.append({"variable": c, "first_year": np.nan, "last_year": np.nan,
                                "first_value": np.nan, "last_value": np.nan,
                                "abs_change": np.nan, "pct_change": np.nan, "cagr_pct": np.nan})
            continue
        y0, y1 = int(s.index.min()), int(s.index.max())
        v0, v1 = float(s.iloc[0]), float(s.iloc[-1])
        n_yrs = max(y1 - y0, 1)
        pct_ch = ((v1 / v0 - 1) * 100) if (v0 not in [0, np.nan] and np.isfinite(v0)) else np.nan
        cagr = (((v1 / v0) ** (1 / n_yrs) - 1) * 100) if (v0 > 0 and v1 > 0) else np.nan
        growth_rows.append({"variable": c, "first_year": y0, "last_year": y1,
                            "first_value": v0, "last_value": v1,
                            "abs_change": v1 - v0, "pct_change": pct_ch, "cagr_pct": cagr})
    growth_summary = pd.DataFrame(growth_rows).sort_values("cagr_pct", ascending=False)
    write_table(yearly_means.reset_index(), "descriptive_growth_yearly_means")
    write_table(yoy_pct.reset_index(), "descriptive_growth_yoy_pct_change")
    write_table(growth_summary, "descriptive_growth_growth_summary")
    print(f"  variables: {len(analysis_cols)} ({numeric_df.shape[1]} numeric)")


# =============================================================================
# 2. CAFO concentration and entry/exit (script2a section 8)
# =============================================================================
def gini_from_values(values):
    """Gini coefficient of non-negative values (0 if all zero, NaN if none)."""
    x = np.array(values, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = x.size
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def load_cafo_compact(panel_keys):
    """Compact CAFO file restricted to the panel's (fips, year) keys and to the canonical class of each commodity."""
    cafo = pd.read_csv(latest_file_glob(CLEAN_DIR, "*_cafo_ops_by_size_compact.csv"), low_memory=False)
    cafo = normalize_panel_key(cafo)
    cafo = cafo.merge(panel_keys.drop_duplicates(), on=["fips", "year"], how="inner")
    for col in ["small", "medium", "large"]:
        if col not in cafo.columns:
            cafo[col] = 0
        cafo[col] = pd.to_numeric(cafo[col], errors="coerce").fillna(0)
    for col in ["commodity_desc", "class_desc"]:
        if col not in cafo.columns:
            cafo[col] = "unknown"
        cafo[col] = cafo[col].astype("string").str.lower().str.strip()
    is_canonical = cafo["commodity_desc"].map(CANONICAL_CLASS) == cafo["class_desc"]
    cafo = cafo[is_canonical | ~cafo["commodity_desc"].isin(CANONICAL_CLASS)].copy()
    print(f"  CAFO compact: {len(cafo):,} rows")
    return cafo


def cafo_concentration(panel_path):
    """HHI, Gini and top-10 share of operations across counties; state-size transitions; large entry and exit rates."""
    print("\n--- concentration: CAFO concentration and entry/exit ---")
    df = load_qa_panel(panel_path)
    cafo = load_cafo_compact(df[["fips", "year"]])
    cafo_long = cafo.melt(id_vars=["fips", "year", "commodity_desc", "class_desc"],
                          value_vars=["small", "medium", "large"],
                          var_name="size_bucket", value_name="ops_count").dropna(subset=["ops_count"])

    cty_commodity = cafo_long.groupby(["year", "fips", "commodity_desc"], as_index=False)["ops_count"].sum()
    rows = []
    for (yr, cmd), g in cty_commodity.groupby(["year", "commodity_desc"]):
        x = g["ops_count"].fillna(0).astype(float).values
        total = float(np.sum(x))
        rows.append({
            "year": int(yr), "commodity_desc": cmd, "total_ops": total,
            "n_counties_positive": int((x > 0).sum()),
            "hhi": float(np.sum((x / total) ** 2)) if total > 0 else np.nan,
            "gini": gini_from_values(x),
            "top10_county_share_pct": float(np.sort(x)[-10:].sum() / total * 100) if total > 0 else np.nan,
        })
    concentration = pd.DataFrame(rows).sort_values(["commodity_desc", "year"])
    write_table(concentration, "cafo_concentration_by_year_commodity")

    # County state by year: no CAFO / small only / medium or more / large present
    state = cafo.groupby(["fips", "year"], as_index=False)[["small", "medium", "large"]].sum()
    state["state_size"] = "no_cafo"
    state.loc[state["small"] > 0, "state_size"] = "small_only"
    state.loc[state["medium"] > 0, "state_size"] = "medium_or_more"
    state.loc[state["large"] > 0, "state_size"] = "large_present"
    state = state.sort_values(["fips", "year"])
    state["next_state_size"] = state.groupby("fips")["state_size"].shift(-1)
    state["next_year"] = state.groupby("fips")["year"].shift(-1)
    state["lag_large"] = state.groupby("fips")["large"].shift(1).fillna(0)
    state["curr_large"] = state["large"].fillna(0)

    # Consecutive-year transitions only (a gap in the years is not a transition)
    transitions = state[state["next_year"].notna() & ((state["next_year"] - state["year"]) == 1)].copy()
    write_table(transitions.groupby(["state_size", "next_state_size"], as_index=False)
                .size().rename(columns={"size": "n_transitions"}), "cafo_transition_matrix_overall")
    write_table(transitions.groupby(["year", "state_size", "next_state_size"], as_index=False)
                .size().rename(columns={"size": "n_transitions"}), "cafo_transition_matrix_by_year")

    state["entry_large"] = ((state["lag_large"] <= 0) & (state["curr_large"] > 0)).astype(int)
    state["exit_large"] = ((state["lag_large"] > 0) & (state["curr_large"] <= 0)).astype(int)
    entry_exit = (state.groupby("year", as_index=False)
                  .agg(entries_large=("entry_large", "sum"), exits_large=("exit_large", "sum"),
                       counties=("fips", "nunique")))
    entry_exit["entry_rate_pct"] = entry_exit["entries_large"] / entry_exit["counties"] * 100
    entry_exit["exit_rate_pct"] = entry_exit["exits_large"] / entry_exit["counties"] * 100
    write_table(entry_exit, "cafo_entry_exit_large_by_year")
    print(f"  consecutive-year transitions: {len(transitions):,}; years in the compact file: "
          f"{sorted(int(y) for y in state['year'].unique())}")


# =============================================================================
# 2b. Targeted summary statistics: CAFO and FSIS counts (script2a section 11, the sumstats part only)
# =============================================================================
CAFO_SUMSTAT_COLS = [
    "cafo_cattle_small", "cafo_cattle_medium", "cafo_cattle_large",
    "cafo_hogs_small", "cafo_hogs_medium", "cafo_hogs_large",
    "cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large",
    "cafo_total_ops_all_animals", "cafo_total_ops_chickens",
]
FSIS_SUMSTAT_COLS = ["n_unique_establishments_fsis"] + [f"n_size_bucket_{i}_fsis" for i in (1, 2, 3, 4, 5, "missing")]
STATE_FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


def summarize_numeric(df, columns):
    """Count, fill share, mean, median, p90, min, max and share positive for each column."""
    rows = []
    n_total = len(df)
    for c in columns:
        s = to_numeric_series(df[c]) if c in df.columns else pd.Series(dtype="float64")
        nonmiss = int(s.notna().sum())
        rows.append({"variable": c, "n_total": n_total, "non_missing_n": nonmiss,
                     "fill_pct": (nonmiss / n_total * 100) if n_total else np.nan,
                     "mean": float(s.mean()) if nonmiss else np.nan,
                     "p50": float(s.quantile(0.50)) if nonmiss else np.nan,
                     "p90": float(s.quantile(0.90)) if nonmiss else np.nan,
                     "min": float(s.min()) if nonmiss else np.nan,
                     "max": float(s.max()) if nonmiss else np.nan,
                     "gt0_pct_among_nonmissing": float((s > 0).mean() * 100) if nonmiss else np.nan})
    return pd.DataFrame(rows)


def targeted_sumstats(panel_path):
    """Summary statistics of the CAFO size-bin counts (county-year and state-year) and of the FSIS counts."""
    print("\n--- sumstats: CAFO and FSIS count summary statistics ---")
    df = load_qa_panel(panel_path)
    df["state_fips"] = df["fips"].astype("string").str[:2]
    df["state_abbrev"] = df["state_fips"].map(STATE_FIPS_TO_ABBR)

    missing_cafo = [c for c in CAFO_SUMSTAT_COLS if c not in df.columns]
    if missing_cafo:
        print(f"  WARNING: missing CAFO columns in panel, CAFO tables skipped: {missing_cafo}")
    else:
        for c in CAFO_SUMSTAT_COLS:
            df[c] = to_numeric_series(df[c])
        write_table(summarize_numeric(df, CAFO_SUMSTAT_COLS), "cafo_county_year_sumstats")
        # min_count=1: a state-year with every county missing stays missing rather than becoming zero
        state_year = (df.groupby(["state_fips", "state_abbrev", "year"], as_index=False)[CAFO_SUMSTAT_COLS]
                      .sum(min_count=1))
        write_table(summarize_numeric(state_year, CAFO_SUMSTAT_COLS), "cafo_state_year_sumstats")

    fsis = pd.read_csv(latest_file_glob(MERGED_DIR, "*_panel_fsis.csv"), low_memory=False)
    fsis = normalize_panel_key(fsis)
    fsis = fsis.loc[:, ~fsis.columns.duplicated()].copy()
    fsis = fsis.sort_values(["fips", "year"]).reset_index(drop=True)
    missing_fsis = [c for c in FSIS_SUMSTAT_COLS if c not in fsis.columns]
    if missing_fsis:
        print(f"  WARNING: missing FSIS columns in the FSIS panel, FSIS table skipped: {missing_fsis}")
    else:
        for c in FSIS_SUMSTAT_COLS:
            fsis[c] = to_numeric_series(fsis[c])
        write_table(summarize_numeric(fsis, FSIS_SUMSTAT_COLS), "fsis_county_year_sumstats")


# =============================================================================
# 3. Species overlap (script4s)
# =============================================================================
def species_overlap(panel_path):
    """How many counties have large CAFOs of more than one species (D-36; missing counts are zero from 2002, D-34)."""
    print("\n--- species: large-CAFO species overlap ---")
    panel = pd.read_csv(panel_path, low_memory=False)
    panel["fips"] = panel["fips"].astype(str).str.zfill(5)
    assert (panel["rural"] == 1).all(), "panel is expected to be pre-filtered to rural counties"
    print(f"panel: {os.path.basename(panel_path)}  {len(panel):,} rows, {panel.fips.nunique():,} counties, "
          f"{int(panel.year.min())}-{int(panel.year.max())}")
    large = {s: panel[f"cafo_{s}_large"] for s in ALL_COLUMNS}

    def presence(species):
        """0/1 frame: does the county-year have >=1 large operation of each species?
        A missing count is a zero (no large operation). Years before FIRST_YEAR are not
        measured and are dropped, never filled."""
        cols = pd.concat([large[s].rename(s) for s in species], axis=1)
        keep = panel["year"] >= FIRST_YEAR
        return panel.loc[keep, ["fips", "year"]].copy(), (cols.loc[keep].fillna(0) > 0).astype(int)

    # How many missing counts the zero rule recodes, so the choice is auditable.
    measured = panel[panel["year"] >= FIRST_YEAR]
    nan_rows = (measured[["year"]]
                .join(pd.concat([large[s].loc[measured.index].isna().rename(s) for s in ALL_COLUMNS], axis=1))
                .groupby("year").sum().reset_index())
    nan_rows.insert(1, "n_county_years", measured.groupby("year").size().values)
    write_table(nan_rows, "species_nan_as_zero")

    # 1. Counties by number of species present, every year
    keys, pres = presence(SPECIES)
    k = pres.sum(axis=1)
    by_year = pd.crosstab(keys["year"], k).reindex(columns=range(len(SPECIES) + 1), fill_value=0)
    by_year.columns = [f"n_counties_{c}_species" for c in by_year.columns]
    by_year["n_county_years"] = by_year.sum(axis=1)
    by_year["n_multi_species"] = by_year[[f"n_counties_{c}_species" for c in range(2, len(SPECIES) + 1)]].sum(axis=1)
    by_year["share_multi_species"] = by_year["n_multi_species"] / by_year["n_county_years"]
    by_year["census_year"] = by_year.index.isin(CENSUS_YEARS)
    by_year = by_year.reset_index().rename(columns={"index": "year"})
    write_table(by_year, "species_count_by_year")
    n_recoded = int(sum(large[s].loc[measured.index].isna().sum() for s in SPECIES))
    print(f"  species tracked: {SPECIES}   missing counts set to zero (2002 on): {n_recoded:,} of "
          f"{len(measured) * len(SPECIES):,} species-county-years")
    print(by_year[by_year.census_year].to_string(index=False))

    # 2. Exact combinations, census years
    keys["combo"] = pres.apply(lambda r: "+".join(s for s in SPECIES if r[s]) or "none", axis=1)
    census_mask = keys["year"].isin(CENSUS_YEARS)
    combo = (pd.crosstab(keys.loc[census_mask, "combo"], keys.loc[census_mask, "year"])
             .assign(total=lambda d: d.sum(axis=1))
             .sort_values("total", ascending=False).reset_index())
    write_table(combo, "species_combination_census")

    # 3. Pairwise co-occurrence, census years
    # n_both: counties with large operations of BOTH species.
    # p_b_given_a: of counties with species A, the share that also has B (the dairy question is a = dairy).
    pair_rows = []
    for yr in CENSUS_YEARS:
        p = pres[(keys["year"] == yr).values]
        for a, b in combinations(SPECIES, 2):
            n_a, n_b, n_ab = int(p[a].sum()), int(p[b].sum()), int((p[a] & p[b]).sum())
            pair_rows.append({"year": yr, "species_a": a, "species_b": b, "n_county_years": len(p),
                              "n_a": n_a, "n_b": n_b, "n_both": n_ab,
                              "p_b_given_a": n_ab / n_a if n_a else np.nan,
                              "p_a_given_b": n_ab / n_b if n_b else np.nan})
    pairs = pd.DataFrame(pair_rows)
    write_table(pairs, "species_pair_census")

    # 4. County-level summary
    # 'ever'       >=2 species in at least one census wave;  'persistent' >=2 species in at least two census waves
    cens = keys[census_mask].assign(k=k[census_mask.values])
    multi_waves = cens.assign(multi=cens.k >= 2).groupby("fips")["multi"].sum()
    dairy_waves = pres.loc[census_mask.values, "dairy"].groupby(keys.loc[census_mask, "fips"]).sum()
    summary = pd.DataFrame({
        "metric": ["counties in the panel (census-year rows)",
                   "counties with >=2 species in >=1 census wave",
                   "counties with >=2 species in >=2 census waves",
                   "counties with large dairy in >=1 census wave",
                   "  of those, with another species in the same wave at least once"],
        "n": [int(multi_waves.size), int((multi_waves >= 1).sum()), int((multi_waves >= 2).sum()),
              int((dairy_waves >= 1).sum()),
              int(((dairy_waves >= 1) & (multi_waves >= 1)).sum())]})
    write_table(summary, "species_county_summary")
    print(summary.to_string(index=False))

    # 5. Appendix: why cattle is not a species. Cattle should contain dairy and beef, so dairy or beef
    # present with cattle absent should be (near) zero; "cattle only" = large cattle operations with
    # neither a large dairy nor a large beef herd (e.g. feedlot or stocker operations).
    _, pc = presence(ALL_COLUMNS)
    cattle_rows = []
    for yr in CENSUS_YEARS:
        p = pc[(keys["year"] == yr).values]
        cattle_rows.append({"year": yr, "n_county_years": len(p),
                            "n_cattle": int(p["cattle"].sum()),
                            "n_dairy": int(p["dairy"].sum()), "n_beef": int(p["beef"].sum()),
                            "n_dairy_without_cattle": int(((p["dairy"] == 1) & (p["cattle"] == 0)).sum()),
                            "n_beef_without_cattle": int(((p["beef"] == 1) & (p["cattle"] == 0)).sum()),
                            "n_cattle_without_dairy_or_beef":
                                int(((p["cattle"] == 1) & (p["dairy"] == 0) & (p["beef"] == 0)).sum())})
    write_table(pd.DataFrame(cattle_rows), "species_cattle_check")


# =============================================================================
# 4. Master variable register (script4k)
# =============================================================================
# Everything established in the decision register, keyed by panel column.
#   v      CHR measure code
#   src    true data source
#   yrs    YEARS OF DATA (not release year) per the CHR 2025 measures PDF
#   lag    release year minus data year
#   tier   A = re-source annually / B = multi-year, keep with restrictions / C = drop
#   role   causal classification argued in the register
#   ptest  can a pre-trend be MEANINGFULLY tested on this series?
#   pres   pre-trend result (script4j), where testable
#   why    rationale for the causal label
CONTROL_META = {
"%_65_and_older":               dict(v="v053",src="Census PEP",yrs="2023",lag=2,tier="A",role="confounder",ptest="Y",pres="flat (0/5 sig)",why="Age structure predicts agricultural land use and mental-health reporting; not moved by a CAFO."),
"%_asian":                      dict(v="v081",src="Census PEP",yrs="2023",lag=2,tier="A",role="confounder",ptest="Y",pres="flat (0/5 sig)",why="Predetermined composition. Near-zero in non-metro; adds little."),
"%_asian_aian_nhpi":            dict(v="",src="Census PEP",yrs="annual",lag=0,tier="A",role="",ptest="Y",pres="",why="Combined Asian + AIAN + NHPI share, annual Census PEP (script0c); replaced %_asian in the control set (D-31)."),
"%_black":                      dict(v="",src="Census PEP",yrs="annual",lag=0,tier="A",role="",ptest="Y",pres="",why="Black share, annual Census PEP (script0c); added to the control set (D-31)."),
"%_below_18_years_of_age":      dict(v="v052",src="Census PEP",yrs="2023",lag=2,tier="A",role="confounder",ptest="Y",pres="flat (0/5 sig)",why="Age structure; predetermined."),
"%_female":                     dict(v="v057",src="Census PEP",yrs="2023",lag=2,tier="A",role="confounder",ptest="Y",pres="flat (0/5 sig)",why="Predetermined."),
"%_hispanic":                   dict(v="v056",src="Census PEP",yrs="2023",lag=2,tier="A",role="confounder",ptest="Y",pres="flat (0/5 sig)",why="Labelled mediator on labour-recruitment grounds, but pre-trend is FLAT (0.006 SD) -- the mediation concern is NOT supported by the data. Treat as confounder."),
"%_native_hawaiian/other_pacific_islander": dict(v="v080",src="Census PEP",yrs="2023",lag=2,tier="A",role="confounder",ptest="Y",pres="flat (0/5 sig)",why="Predetermined; negligible in non-metro."),
"%_not_proficient_in_english_per100k": dict(v="v059",src="ACS 5-year",yrs="2019-2023",lag="n/a (5-yr)",tier="B",role="confounder",ptest="N",pres="NOT TESTABLE",why="Labelled mediator (CAFO labour in-migration) but pre-trend is flat; and as a 5-year rolling estimate the test is uninformative anyway."),
"%_rural":                      dict(v="v058",src="Decennial Census",yrs="2020",lag="decennial",tier="C",role="confounder",ptest="N",pres="NOT TESTABLE",why="DROP (D-03). 93.7% of values identical to prior year; within-SD share 0.069; and the sample is NCHS 3-6 (non-large-metro), not rural, so it is not a consistency check either."),
"access_to_healthy_foods_per100k": dict(v="v030->v083",src="USDA Food Atlas",yrs="2019",lag=6,tier="C",role="mediator",ptest="N",pres="NOT TESTABLE",why="DROP (D-05). Measure splice v030->v083 at release 2013; USDA publishes ~every 5 years; its 0.898 within-SD share is the splice, not signal."),
"adult_obesity_per100k":        dict(v="v011",src="BRFSS",yrs="2022",lag=3,tier="A",role="confounder",ptest="Y",pres="flat (0/5 sig)",why="Health stock, slow-moving; unlikely to respond within the window."),
"adult_smoking_per100k":        dict(v="v009/v095",src="BRFSS",yrs="2022",lag=3,tier="A",role="collider",ptest="Y",pres="FAILS (2/5 sig)",why="Responds to distress AND to local economic shocks -- caused by outcome and treatment alike. BUT it also has a code blip v009->v095->v009 around 2011, which would produce a spurious pre-trend, so the failed test is NOT independent evidence. Decide on the causal argument. (D-04)"),
"children_in_poverty_per100k":  dict(v="v024",src="SAIPE + ACS 5-yr",yrs="2023 & 2019-2023",lag="mixed",tier="A",role="mediator",ptest="Y",pres="flat (0/5 sig)",why="Local economic conditions respond to agricultural restructuring."),
"children_in_single-parent_households_per100k": dict(v="v082",src="ACS 5-year",yrs="2019-2023",lag="n/a (5-yr)",tier="B",role="confounder",ptest="N",pres="NOT TESTABLE",why="Family structure, slow-moving. Earlier 'FAILS (2/5)' result is WITHDRAWN -- a 5-year rolling estimate cannot show a differential trend."),
"driving_alone_to_work_per100k":dict(v="v067",src="ACS 5-year",yrs="2019-2023",lag="n/a (5-yr)",tier="B",role="confounder",ptest="N",pres="NOT TESTABLE",why="Commuting pattern, proxies labour-market geography. Earlier 'mediator signature (4 sig post)' reading is WITHDRAWN -- invalid for a 5-year rolling estimate."),
"median_household_income":      dict(v="v063",src="SAIPE + ACS 5-yr",yrs="2023 & 2019-2023",lag="mixed",tier="A",role="mediator+confounder",ptest="Y",pres="FAILS (2 pre, 3 post)",why="Does BOTH jobs: 2 significant PRE coefficients (selection into treatment) and a monotonic POST rise to +0.13 SD by t+6 (response to treatment). Conditioning trades one bias for another. Direct CAFO employment is too small to explain the post pattern; more likely a marker for wider agricultural investment."),
"some_college_per100k":         dict(v="v069",src="ACS 5-year",yrs="2019-2023",lag="n/a (5-yr)",tier="B",role="confounder",ptest="N",pres="NOT TESTABLE",why="Education stock, predetermined."),
"teen_births_per100k":          dict(v="v014",src="NCHS Natality (pooled)",yrs="2017-2023",lag="n/a (7-yr pooled)",tier="B",role="collider",ptest="N",pres="NOT TESTABLE",why="KEEP CHR pooled (D-12). NCHS annual alternative rejected: ends 2020 (costs 6,289 rows / 78 switchers), AR(1)=0.984 so MORE smoothed than the pooled measure, and its Bayesian model borrows from neighbouring counties -- hazardous when CAFO counties cluster. Collider status (D-04) remains open."),
"unemployment_per100k":         dict(v="v023",src="BLS LAUS",yrs="2023",lag=2,tier="A",role="mediator",ptest="Y",pres="flat (0/5 sig)",why="A CAFO opening changes local employment directly."),
"uninsured_adults_per100k":     dict(v="v003",src="SAHIE",yrs="2022",lag=3,tier="A",role="confounder",ptest="Y",pres="FAILS (2 pre, 1 post)",why="Labelled mediator (insurance follows employment) but the empirical signature is CONFOUNDER -- diverges before treatment, not after."),
}

OUTCOME_META = {
"poor_mental_health_days":      dict(v="v042",src="BRFSS",yrs="2022",lag=3,note="Primary mental-health outcome. Best coverage, most switchers."),
"frequent_mental_distress_per100k": dict(v="v145",src="BRFSS",yrs="2022",lag=3,note="Exists only from release 2016 -> only 2 CS cohorts, 14 (g,t) cells. Weakest foundation; the one CS-significant result sits here and its pre-period swallows the effect."),
"crude_rate_from_census_pop":   dict(v="CDC WONDER",src="CDC/NCHS mortality",yrs="varies",lag="n/a",note="Deaths of despair. CDC suppresses <10 deaths -> 22.6% coverage. Truncated after 2020 (source coverage falls to zero)."),
"crime_agg_assault":            dict(v="NIBRS",src="FBI NIBRS arrestee segment",yrs="incident year",lag=0,note="ARRESTS not offences (D-09). Largest genuine UCR violent component available."),
"assault_total_per100k":        dict(v="NIBRS",src="FBI NIBRS arrestee segment",yrs="incident year",lag=0,note="ARRESTS. Renamed from crime_assault (D-07). 79% SIMPLE assault, which UCR EXCLUDES from violent crime -- never present as a violent-crime measure. r=0.989 with total_incidents."),
"violence_index_partial":       dict(v="NIBRS",src="FBI NIBRS arrestee segment",yrs="incident year",lag=0,note="ARRESTS. agg assault + rape + intimidation. NON-UCR (robbery and homicide missing). r=0.916 with agg assault -- 67% agg assault by construction."),
"total_incidents_per100k":      dict(v="NIBRS",src="FBI NIBRS arrestee segment",yrs="incident year",lag=0,note="ARRESTS. NOT all crime: a curated 12-offence sum, no property or drug offences (D-08). 86% assault."),
}

DECISION = {
"%_rural":"D-03 drop (REMOVED 2026-09-27)","access_to_healthy_foods_per100k":"D-05 drop (decided)",
"adult_smoking_per100k":"D-04 collider, removal candidate (open)","teen_births_per100k":"D-04 collider (open) / D-12 keep CHR pooled (decided)",
}
for _c in CONTROL_MEDIATORS:
    DECISION.setdefault(_c, "D-01 excluded as mediator")
# The control set: the 8 full-coverage controls, pre-committed (D-16, membership revised D-31).
CONTROL_SET = COVARIATE_ORDER[:8]
for _c in CONTROL_SET:
    DECISION[_c] = "; ".join(x for x in ["D-16/D-31 control set", DECISION.get(_c, "")] if x)
GENERIC_CAFO = ["small_cafo", "medium_cafo", "large_cafo", "bin_sum_cafo", "total_heads_cafo",
                "any_large_cafo", "any_medium_or_large_cafo"]


def decision_of(col):
    """Decision-register ids that bear on a column."""
    ids = [DECISION.get(col, "")]
    if col in GENERIC_CAFO:
        ids.append("D-51 generic CAFO = cattle + hogs + chickens")
    if "cafo" in col.lower():
        ids.append("D-34 missing count = 0 from 2002; D-53 census-year blank = 0 before forward-fill")
    if col.startswith("cafo_cattle"):
        ids.append("D-36 cattle contains dairy and beef; not a species")
    if col == "tr_e2_add_absorb":
        ids.append("D-32 E2 reported")
    return "; ".join(i for i in ids if i)


# Treatment id(s) of each treatment column, and whether it is in the reported set (REPORT_TREATMENTS).
TREATMENT_IDS = {}
for _k, _c in CORE_TREATMENTS.items():
    TREATMENT_IDS.setdefault(_c, []).append(_k)
for _k, (_c, _cond, _lab) in TREATMENTS.items():
    TREATMENT_IDS.setdefault(_c, []).append(_k)
TREATMENT_LABEL = {_c: _lab for _k, (_c, _cond, _lab) in TREATMENTS.items()}
TREATMENT_LABEL.update({_c: CORE_TREATMENT_LABELS[_k] for _k, _c in CORE_TREATMENTS.items()})
INTERACTS = {
"%_hispanic":"corr -0.775 with log_pop (within). Used in dairy x %hispanic interaction in the ML block -- that interaction is hard to read cleanly because of this.",
"median_household_income":"VIF 5.35 raw / 1.19 within; the largest raw-level VIF in the set.",
"children_in_poverty_per100k":"VIF 5.35 raw / 1.17 within.",
}

# Salvaged from 250710-codebook-fillout.xlsx (the superseded hand-filled codebook). Only 3 of its 213 names
# matched the current panel, but its NIBRS offence-code definitions are authoritative and worth carrying
# forward. The crime-against-a-PERSON vs crime-against-SOCIETY split is the principled basis for which
# offences belong in a violence measure -- note DUI is against SOCIETY, not a person.
NIBRS_CODES = {
 "aggravated_assault":            "Group A, assault offense 13A, crime against a PERSON",
 "simple_assault":                "Group A, assault offense 13B, crime against a PERSON",
 "intimidation":                  "Group A, assault offense 13C, crime against a PERSON",
 "rape":                          "Group A, sex offense nonforcible, 36B, crime against a PERSON",
 "incest":                        "Group A, sex offense nonforcible, 36A, crime against a PERSON",
 "sexual_assault_with_an_object": "Group A, sex offense forcible, 11B, crime against a PERSON",
 "kidnapping/abduction":          "Group A, kidnapping/abduction offense 100, crime against a PERSON",
 "driving_under_the_influence":   "Group B, DUI offense 90D, crime against SOCIETY (not against a person)",
}


def nibrs_note(col):
    """Attach the offence-code definition to every derived column of that offence."""
    for stem, desc in NIBRS_CODES.items():
        if col == stem or col.startswith(stem + "_"):
            return desc
    return ""


# Source file map: traces each panel column back to the clean-layer file it came from, and from there to the
# original raw file. The script names are copied as the retired script recorded them.
SOURCE_FILES = {
 "cafo_ops_by_size_compact":        ("script0b", "USDA NASS Quick Stats API (census years only)"),
 "cdc_county_year_deathsofdespair": ("script0c", "CDC WONDER: cty-level-deathsofdespair-YYYY.csv"),
 "crime_fips_level_final":          ("script0d", "FBI NIBRS: nibrs_arrestee_segment_YYYY.csv (ARRESTS)"),
 "fips_full":                       ("script0a", "Census FIPS crosswalk"),
 "mentalhealthrank_full":           ("script0c", "County Health Rankings: analytic_dataYYYY.csv"),
 "population_full":                 ("script0a", "Census Population Estimates Program"),
 "rural-key":                       ("script0a", "NCHS: NCHSurb-rural-codes.csv"),
 "fsis_county_year_fips":           ("script0e", "USDA FSIS establishment file (FOIA) + HUD ZIP crosswalk"),
}


def build_source_map():
    """Exact header match first; then stem-matching, because the panel build renames columns during the merge
    (CHR arrives as `<measure>_raw_value`, the panel keeps `<measure>` or `<measure>_per100k`)."""
    exact, stems = {}, {}
    for stem, (scr, raw) in SOURCE_FILES.items():
        hits = sorted(glob.glob(os.path.join(CLEAN_DIR, f"*{stem}*.csv")))
        if not hits:
            continue
        f = hits[-1]
        try:
            cols = list(pd.read_csv(f, nrows=1, low_memory=False).columns)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        for c in cols:
            exact.setdefault(c, (os.path.basename(f), scr, raw))
        for c in cols:   # index the clean file's measure stems so renamed panel columns still match
            base = (c.replace("_raw_value", "").replace("_numerator", "")
                     .replace("_denominator", "").replace("_ci_high", "").replace("_ci_low", ""))
            stems.setdefault(base, (os.path.basename(f), scr, raw))
    return exact, stems


def source_of(col, source_exact, source_stems):
    """Return (clean_file, script, raw_source) for a panel column; a rule on the name itself is the fallback."""
    if col in source_exact:
        return source_exact[col]
    base = col.replace("_per100k", "")
    if base in source_stems:
        return source_stems[base]
    if col in source_stems:
        return source_stems[col]
    lc = col.lower()
    if lc.endswith("_fsis") or "fsis" in lc:
        return ("*_fsis_county_year_...csv", "script0e", "USDA FSIS establishment file (FOIA) + HUD ZIP crosswalk")
    if "cafo" in lc or lc in ("large_cafo", "medium_cafo", "small_cafo", "bin_sum_cafo", "total_heads_cafo"):
        return ("*_cafo_ops_by_size_compact.csv", "script0b", "USDA NASS Quick Stats API (census years only)")
    if lc.startswith("tr_") or lc.startswith("cond_") or lc in ("cohort", "t_rel", "add_year", "entry_year",
                                                               "build_year", "consol_year", "log_pop"):
        return ("(derived in script4_treatment.py)", "script4_treatment", "derived from the USDA NASS CAFO block")
    if any(k in lc for k in ["assault", "rape", "incest", "intimidation", "kidnap", "fondling",
                             "human_traffic", "statutory", "incident", "violence", "driving_under", "sexual_assault", "crime"]):
        return ("*_crime_fips_level_final.csv", "script0d", "FBI NIBRS: nibrs_arrestee_segment_YYYY.csv (ARRESTS)")
    if "despair" in lc or lc.startswith("crude_rate") or lc in ("deaths", "deaths_is_zero", "cdc_in_query", "homicides"):
        return ("*_cdc_county_year_deathsofdespair.csv", "script0c", "CDC WONDER: cty-level-deathsofdespair-YYYY.csv")
    if lc in ("rural", "non_large_metro", "nchs_code", "nchs_label"):
        return ("*-rural-key.csv", "script0a", "NCHS: NCHSurb-rural-codes.csv")
    if lc.startswith(("pep_", "saipe_", "laus_", "sahie_")) or lc in ("%_asian_aian_nhpi", "%_black") \
            or lc.endswith("_pep"):
        return ("*_annual_controls_county_year.csv", "script0c",
                "Census PEP / SAIPE / BLS LAUS / SAHIE (annual, data-year dated)")
    if lc.endswith("_count_imputed"):
        return ("(derived in script0f-generate-panel.py)", "script0f", "CHR count imputation from rate x population")
    if "pop" in lc:
        return ("*_population_full.csv", "script0a", "Census Population Estimates Program")
    if lc in ("fips", "state", "county", "state_fips", "county_fips", "state_abbrev", "state_abbreviation",
              "name", "state_code"):
        return ("*_fips_full.csv", "script0a", "Census FIPS crosswalk")
    return ("", "", "")


# Interpolation / augmentation: what was done to a variable beyond reading it. Stated per variable because
# a referee will ask.
CAFO_AUG = ("FORWARD-FILLED from ag-census years (2002/2007/2012/2017/2022) to all "
            "intervening years by script0f, AND zero-filled where a rural county is "
            "absent from a census year (confirmed zero operations). Consequence: all "
            "within-county variation occurs at 4 wave transitions; treatment timing "
            "carries up to 5 years of measurement error.")
AUG = {
 "tr_e1_lg_bin":      "DERIVED: 1 if cafo_dairy_large > 0, NaN-preserving. " + CAFO_AUG,
 "tr_e2_add_absorb":  "DERIVED: absorbing indicator, 1 from the county's first wave with d(large)>0. Exits never switch it off. " + CAFO_AUG,
 "tr_e3_entry_absorb":"DERIVED: absorbing indicator, 1 from the county's first 0->positive wave. " + CAFO_AUG,
 "tr_i1_add_cum":     "DERIVED: cumulative sum of POSITIVE wave-to-wave changes only (contractions do not subtract). " + CAFO_AUG,
 "tr_i2_add_nevents": "DERIVED: running count of positive-change waves. " + CAFO_AUG,
 "tr_i3_lg_count":    "DERIVED: raw count of large dairy operations. " + CAFO_AUG,
 "tr_i4_lg_p10k":     "DERIVED: large operations / population x 10,000. " + CAFO_AUG,
 "tr_m1_build":       "DERIVED: absorbing, 1 from first wave with d(large)>0 AND d(total)>0. " + CAFO_AUG,
 "tr_m2_consolidate": "DERIVED: absorbing, 1 from first wave with d(large)>0 AND d(total)<0. " + CAFO_AUG,
 "tr_m3_add_any":     "DERIVED: identical to tr_e2_add_absorb. NEVER ESTIMATED -- duplicate.",
 "tr_c1_lg_share":    "DERIVED: large / total dairy operations. Unstable where total is small (median county has 7 total ops; 40% have <=5). " + CAFO_AUG,
 "tr_c2_lg_share_scr":"DERIVED: tr_c1_lg_share restricted to counties with >=10 total operations. " + CAFO_AUG,
 "tr_c3_log_lg":      "DERIVED: log(1 + large count). Entered JOINTLY with cond_log_sm in the conditional model. " + CAFO_AUG,
 "tr_c5_hhi":         "DERIVED: Herfindahl index across the three size bins. " + CAFO_AUG,
 "cond_log_sm":       "DERIVED: log(1 + small dairy count), contemporaneous. POST-TREATMENT if large operations displace small ones. " + CAFO_AUG,
 "cond_log_sm_base":  "DERIVED: log(1 + small count) at the county's FIRST OBSERVED wave. County-CONSTANT, so county FE absorbs it entirely -- unusable as a level in a FE model.",
 "cond_smbase_x_t":   "DERIVED: cond_log_sm_base x (year - min year). The usable form of baseline conditioning in a FE model: allows differential trends by baseline small-farm structure.",
 "cond_log_tot":      "DERIVED: log(1 + total dairy count). " + CAFO_AUG,
 "add_year":          "DERIVED: first ag-census wave with a positive change in large dairy count.",
 "entry_year":        "DERIVED: first wave going 0 -> positive.",
 "build_year":        "DERIVED: first wave with d(large)>0 AND d(total)>0.",
 "consol_year":       "DERIVED: first wave with d(large)>0 AND d(total)<0.",
 "cohort":            "DERIVED: = add_year. The CS DiD cohort assignment.",
 "t_rel":             "DERIVED: year - cohort. Event time. Carries the 5-year census timing error.",
 "log_pop":           "DERIVED: log(population), population>0 only.",
 "crime_agg_assault": "DERIVED: = aggravated_assault_per100k, promoted to a standalone outcome.",
 "assault_total_per100k": "DERIVED: aggravated + simple assault per 100k, NaN-masked on total_incidents coverage. 79% simple assault.",
 "violence_index_partial":"DERIVED: aggravated assault + rape + intimidation per 100k, NaN-masked on total_incidents coverage. NON-UCR (robbery and homicide absent).",
 "crude_rate_from_census_pop":"CONSTRUCTED in script0f: deaths / census population x 100,000. Used instead of CDC's own crude rate because CDC suppresses it below 10 deaths. Truncated after 2020 (source coverage falls to zero).",
 "total_incidents_per100k":"CONSTRUCTED in script0d: sum of a CURATED 12-offence list, per 100k. Not all crime -- no property or drug offences.",
}
for _c in ["cafo_dairy_small", "cafo_dairy_medium", "cafo_dairy_large", "cafo_dairy_total",
           "cafo_hogs_large", "cafo_beef_large", "cafo_cattle_large", "cafo_chickens_large"]:
    AUG.setdefault(_c, CAFO_AUG)

# Why a variable is NOT used as a control.
EXCLUSION = {
 "income_inequality":"EXCLUDED (D-02): coverage 73.0% on outcome rows, below the 92% threshold. Listwise deletion cost.",
 "social_associations_per100k":"EXCLUDED (D-02): coverage 66.1%.",
 "%_non-hispanic_african_american":"EXCLUDED (D-02): coverage 63.0%.",
 "air_pollution_-_particulate_matter":"EXCLUDED (D-02): coverage 78.0%.",
 "mental_health_providers_per100k":"EXCLUDED (D-02): coverage 79.1%.",
 "primary_care_physicians_per100k":"EXCLUDED (D-02): coverage 89.1%.",
 "poor_physical_health_days":"EXCLUDED (D-01): plausible MEDIATOR -- a health outcome a CAFO could move. Conditioning blocks part of the causal path. Retained in CONTROL_COMPREHENSIVE for the robustness arm.",
 "premature_death":"EXCLUDED (D-01): plausible MEDIATOR. In CONTROL_COMPREHENSIVE only.",
 "preventable_hospital_stays":"EXCLUDED (D-01): plausible MEDIATOR. In CONTROL_COMPREHENSIVE only.",
 "low_birthweight_per100k":"EXCLUDED (D-01): plausible MEDIATOR. In CONTROL_COMPREHENSIVE only.",
 "diabetes_prevalence_per100k":"EXCLUDED (D-01): plausible MEDIATOR. In CONTROL_COMPREHENSIVE only.",
 "physical_inactivity_per100k":"EXCLUDED (D-01): plausible MEDIATOR. In CONTROL_COMPREHENSIVE only.",
 "poor_or_fair_health_per100k":"EXCLUDED (D-01): plausible MEDIATOR. In CONTROL_COMPREHENSIVE only.",
 "%_rural":"FLAGGED FOR REMOVAL (D-03): 93.7% of values identical to the prior year (decennial, carried forward); within-SD share 0.069 so county FE absorbs it; and the sample is NCHS 3-6 (non-large-metro, 18.6% still metropolitan), so it is not a filter consistency check either.",
 "access_to_healthy_foods_per100k":"DROP (D-05): measure splice v030->v083 at release 2013 (two different CHR measures in one column); USDA Food Atlas publishes ~every 5 years; its 0.898 within-SD share is the splice, not signal.",
 "violent_crime":"DROPPED AS AN OUTCOME (D-06): CHR v043 pools across years -- 46.7% of county-year values identical to the prior year -- so it cannot support annual event-study timing. Also release-year dated and discontinued after the 2022 release.",
}

FSIS_NOTE = ("FSIS coverage is 2017-2023 ONLY, at roughly 46-50% of rural counties per "
             "year. Lives in *_panel_fsis.csv, NOT the main panel, to avoid truncating "
             "the 2000-2023 series. Any analysis using an _fsis column must load that "
             "file. The dairy x FSIS interaction (A6) is flagged DO NOT PRESENT: "
             "VIF 34.3 on %_rural in its thin sample.")

REGISTER_ORDER = {"OUTCOME": 0, "TREATMENT": 1, "CONTROL (control set)": 2, "CONTROL (candidate)": 3,
                  "CONTROL (comprehensive only)": 4, "IDENTIFIER": 5, "not used": 6}


def fill_within_window(df, col, years):
    """Share (%) of cells filled INSIDE the variable's own year range."""
    if not len(years):
        return 0.0
    sub = df[df["year"].between(years.min(), years.max())]
    return round(100 * float(sub[col].notna().mean()), 1)


def coverage_shape(df, col, s, years):
    """Classify WHERE the missingness is: years, counties, both, or neither."""
    if not len(years):
        return "no data"
    py = years.nunique() / df["year"].nunique()
    pc = df.loc[s.notna(), "fips"].nunique() / df["fips"].nunique()
    fw = fill_within_window(df, col, years) / 100
    if py >= 0.9 and pc >= 0.9 and fw >= 0.9:
        return "FULL (all years, all counties)"
    if py < 0.9 and pc >= 0.9 and fw >= 0.9:
        return f"YEAR-LIMITED ({years.min()}-{years.max()}, but near-complete within that window)"
    if py >= 0.9 and pc < 0.9:
        return "COUNTY-LIMITED (most years, subset of counties)"
    if py < 0.9 and fw < 0.9:
        return f"BOTH ({years.min()}-{years.max()} AND sparse within that window)"
    if fw < 0.9:
        return "SPARSE WITHIN WINDOW (years fine, cells patchy)"
    return "partial"


def role_of(col):
    """Role of a panel column in the analysis."""
    if col in OUTCOMES.values():
        return "OUTCOME"
    if col in {t[0] for t in TREATMENTS.values()} or col.startswith("tr_"):
        return "TREATMENT"
    if col in CONTROL_SET:
        return "CONTROL (control set)"
    if col in CONTROL_PRETREAT:
        return "CONTROL (candidate)"
    if col in CONTROL_COMPREHENSIVE:
        return "CONTROL (comprehensive only)"
    if col in {"fips", "year", "state_fips", "rural", "population", "log_pop"}:
        return "IDENTIFIER"
    return "not used"


def register_fsis_sheet(main_columns, source_exact, source_stems):
    """FSIS columns live in a SEPARATE panel (`*_panel_fsis.csv`, 2017+ only) so that merging them does not
    truncate the full 2000-2023 series; they are documented on their own tab because their window differs."""
    try:
        fsis_path = latest_file_glob(MERGED_DIR, "*_panel_fsis.csv")
        fsis = pd.read_csv(fsis_path, low_memory=False)
        rows = []
        for c in fsis.columns:
            fs = fsis[c]
            num = pd.api.types.is_numeric_dtype(fs)
            yrs = fsis.loc[fs.notna(), "year"] if "year" in fsis.columns else pd.Series(dtype=float)
            src = source_of(c, source_exact, source_stems)
            rows.append({"colname": c,
                         "in_main_panel": "Y" if c in main_columns else "N (FSIS panel only)",
                         "is_fsis_column": "Y" if c.endswith("_fsis") else "",
                         "type": str(fs.dtype),
                         "coverage_pct": round(100 * float(fs.notna().mean()), 1),
                         "first_year": int(yrs.min()) if len(yrs) else None,
                         "last_year": int(yrs.max()) if len(yrs) else None,
                         "source_file_clean": src[0], "source_script": src[1], "source_raw_original": src[2],
                         "mean": float(fs.mean()) if num else np.nan,
                         "sd": float(fs.std()) if num else np.nan,
                         "notes": FSIS_NOTE if c.endswith("_fsis") else ""})
        sheet = pd.DataFrame(rows)
        sheet["_o"] = (sheet.is_fsis_column != "Y").astype(int)
        return sheet.sort_values(["_o", "colname"]).drop(columns="_o"), os.path.basename(fsis_path)
    except (FileNotFoundError, OSError, ValueError, KeyError) as err:
        return pd.DataFrame([{"colname": "<FSIS panel not found>", "notes": str(err)}]), "n/a"


def variable_register(panel_path):
    """One row per column of the analysis panel carrying every decision and diagnostic of the control-decision
    register, so each inclusion and exclusion can be justified to a referee from a single file (xlsx + csv)."""
    print("\n--- register: master variable register ---")
    os.makedirs(OUT_CODEBOOKS, exist_ok=True)
    source_exact, source_stems = build_source_map()
    df = load_panel()            # the analysis panel with every treatment variable attached
    ref = df[df["poor_mental_health_days"].notna()]
    est = df[["fips", "year"] + [c for c in CONTROL_PRETREAT if c in df.columns]].dropna()

    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.api.types.is_numeric_dtype(s)
        yrs = df.loc[s.notna(), "year"]
        meta, om = CONTROL_META.get(c, {}), OUTCOME_META.get(c, {})
        r = role_of(c)
        is_cov = "Y" if c in CONTROL_PRETREAT else ("comprehensive only" if c in CONTROL_COMPREHENSIVE else "N")
        wshare = np.nan
        if c in est.columns and num and est[c].std() > 0:
            w = est[c] - est.groupby("fips")[c].transform("mean")
            wshare = w.std() / est[c].std()
        inv = [k for k, v in OUTCOMES.items() if v == c]
        src = source_of(c, source_exact, source_stems)
        rows.append({
            "colname": c,
            "role": r,
            "description": (meta.get("why", "") if r.startswith("CONTROL") else
                            om.get("note", "") if r == "OUTCOME" else
                            TREATMENT_LABEL.get(c, AUG.get(c, "")) if r == "TREATMENT" else ""),
            "treatment_id": "/".join(TREATMENT_IDS.get(c, [])),
            "reported_treatment": "Y" if set(TREATMENT_IDS.get(c, [])) & set(REPORT_TREATMENTS) else "",
            "outcome_label": inv[0] if inv else "",
            "type": str(s.dtype),
            "source": meta.get("src", om.get("src", "")),
            "chr_measure_code": meta.get("v", om.get("v", "")),
            "data_years_TRUE": meta.get("yrs", om.get("yrs", "")),
            "year_lag_release_minus_data": meta.get("lag", om.get("lag", "")),
            "source_tier": meta.get("tier", ""),
            "covariate_YN": is_cov,
            "covar_possibility": meta.get("role", ""),
            "covar_rationale": meta.get("why", ""),
            "pretrend_testable": meta.get("ptest", ""),
            "pretrend_result": meta.get("pres", ""),
            # in_control_set: the 8 controls of every reported specification (TWFE and CS-adjusted).
            # in_candidates: CONTROL_PRETREAT, the non-mediator candidates documented by script4b control-set.
            # in_comprehensive: CONTROL_COMPREHENSIVE, the robustness arm that also includes mediators.
            "in_control_set": "Y" if c in CONTROL_SET else "",
            "in_candidates": "Y" if c in CONTROL_PRETREAT else "",
            "in_comprehensive": "Y" if c in CONTROL_COMPREHENSIVE else "",
            "nibrs_offence_code": nibrs_note(c),
            "interactions_collinearity": INTERACTS.get(c, ""),
            "decision_id": decision_of(c),
            "pct_complete_full_panel": float(s.notna().mean()),
            "pct_complete_outcome_rows": float(ref[c].notna().mean()) if c in ref.columns else np.nan,
            "first_year": int(yrs.min()) if len(yrs) else None,
            "last_year": int(yrs.max()) if len(yrs) else None,
            "within_sd_share": wshare,
            "mean": float(s.mean()) if num else np.nan,
            "sd": float(s.std()) if num else np.nan,
            "min": float(s.min()) if num else np.nan,
            "max": float(s.max()) if num else np.nan,
            # Coverage split into its two dimensions: "covers every county but only 5 years" and "covers
            # every year but only half the counties" look identical in one percentage and mean different things.
            #   coverage_pct              cells filled / all county-year cells in the panel
            #   pct_years_covered         years with ANY data / panel years
            #   pct_counties_covered      counties with ANY data / all panel counties
            #   fill_within_active_window cells filled / cells available inside the variable's own year range
            #   coverage_shape            plain-language classification of the above
            "coverage_pct": round(100 * float(s.notna().mean()), 1),
            "n_years_with_data": int(yrs.nunique()) if len(yrs) else 0,
            "pct_years_covered": round(100 * (yrs.nunique() / df["year"].nunique()), 1) if len(yrs) else 0.0,
            "n_counties_with_data": int(df.loc[s.notna(), "fips"].nunique()) if s.notna().any() else 0,
            "pct_counties_covered": (round(100 * (df.loc[s.notna(), "fips"].nunique() / df["fips"].nunique()), 1)
                                     if s.notna().any() else 0.0),
            "fill_within_active_window": fill_within_window(df, c, yrs),
            "coverage_shape": coverage_shape(df, c, s, yrs),
            "coverage_pct_on_outcome_rows": round(100 * float(ref[c].notna().mean()), 1) if c in ref.columns else np.nan,
            "why_not_used": EXCLUSION.get(c, ""),
            "source_file_clean": src[0],
            "source_script": src[1],
            "source_raw_original": src[2],
            "interpolation_augmentation": AUG.get(c, ""),
            "notes": "",
        })
    register = pd.DataFrame(rows)
    register["_o"] = register.role.map(REGISTER_ORDER).fillna(9)
    register = register.sort_values(["_o", "colname"]).drop(columns="_o")

    fsis_sheet, fsis_src = register_fsis_sheet(set(df.columns), source_exact, source_stems)
    readme = pd.DataFrame({"item": [
        "Generated by", "Source panel", "FSIS panel", "Rows", "Purpose",
        "why_not_used", "coverage_pct", "source_file_clean / source_raw_original",
        "interpolation_augmentation", "pretrend_testable", "source_tier", "decision_id"],
     "detail": [
        "script1b-tables.py --only register (rerun to regenerate)",
        os.path.basename(df.attrs["panel_path"]),
        fsis_src,
        f"{len(register)} main panel columns; {len(fsis_sheet)} FSIS panel columns",
        "One row per variable, with a description of the variable, and every inclusion/exclusion decision for control set. "
        "in_control_set = the 8 pre-committed controls (D-16, D-31); reported_treatment = T1, T3, T4, E2 (REPORT_TREATMENTS)",
        "The decision to include or exclude a control variable, based on classification as a mediator or collider, and on review of coverage",
        "Share of rows with a non-missing value. coverage_pct_on_outcome_rows restricts to rows with a non-missing Poor MH Days.",
        "Traces the variable to the Data/clean file it came from and to the original raw source.",
        "Any interpolation or changes that are done to the variable from the original format of the data set, such as forward-fill, zero-fill, derivation formula.",
        "N where the series is pooled or a multi-year rolling estimate: a pre-trend CANNOT be meaningfully tested on it.",
        "A = re-source at true annual data year; B = multi-year, keep with restrictions; C = drop.",
        "Links to code-audit/plans/CONTROL-DECISION-REGISTER.md",
     ]})

    xlsx_path = os.path.join(OUT_CODEBOOKS, f"{TODAY}_VARIABLE_REGISTER.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        sheets = [("README", readme), ("variables", register), ("FSIS_panel", fsis_sheet)]
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name, index=False)
        for name, frame in sheets:
            ws = writer.sheets[name]
            ws.freeze_panes = "B2"
            for i, c in enumerate(frame.columns, 1):
                width = max(len(str(c)), int(frame[c].astype(str).str.len().max() or 0))
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(width + 2, 12), 70)
    csv_path = os.path.join(OUT_CODEBOOKS, f"{TODAY}_VARIABLE_REGISTER.csv")
    register.to_csv(csv_path, index=False)
    print(f"  rows: {len(register)}")
    print(register.role.value_counts().to_string())
    print(f"  saved: {xlsx_path}\n  saved: {csv_path}")


# =============================================================================
# Run
# =============================================================================
SECTIONS = {"summary": summary_statistics, "concentration": cafo_concentration, "sumstats": targeted_sumstats,
            "species": species_overlap, "register": variable_register}


def main():
    parser = argparse.ArgumentParser(description="Descriptive-stage tables (no graphics).")
    parser.add_argument("--only", nargs="+", choices=list(SECTIONS), default=list(SECTIONS),
                        help="run only these sections (default: all)")
    args = parser.parse_args()
    os.makedirs(OUT_TABLES, exist_ok=True)
    panel_path = latest_file_glob(MERGED_DIR, "*_panel.csv")
    for name in args.only:
        SECTIONS[name](panel_path)
    print(f"\nscript1b-tables COMPLETE  ({', '.join(args.only)})\n  tables -> {OUT_TABLES}")


if __name__ == "__main__":
    main()
