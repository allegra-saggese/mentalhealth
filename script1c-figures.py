#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""script1c-figures.py -- every non-map figure of the descriptive stage (stage 1), on figstyle.py.

Sections (python3 script1c-figures.py --only <name> [<name> ...]; no flag runs all, in this order):
  binned-scatter           CAFO and FSIS exposure vs outcomes: binned means, 95% CI, OLS line        (was script2b A1 A2 A3 I J; script2d P R)
  outcome-correlation      Spearman correlations between the six outcomes                            (script2b C)
  quartile-distributions   violin and box plots of each outcome by CAFO-intensity quartile           (script2b D E)
  cafo-top-counties        top-20 counties by operations per 10,000, census years                   (script2c K L)
  cafo-census-composition  size-class shares and national counts by animal, census years            (script2c M N P)
  threshold-presence       presence/absence and count-tier means of outcomes                        (script2d O Q)
  fsis-size-bins           FSIS establishments per size bin vs poor mental health days, 2017        (script2a section 3)
  high-coverage-violins    distribution of the 20 best-covered panel variables                      (script2a section 6)
  cafo-class-size          operations by commodity and size class: violins and shares               (script2a section 7)
  cafo-concentration       HHI, Gini, top-10 share and large-operation entry/exit                   (script2a section 8, plots)
  ridgelines               yearly density of four outcomes, one ridge per year                       (script2a section 9)
  trend-sweep              every numeric variable's yearly mean against CAFO counts by commodity and size (script2a section 9)
  binned-by-size           key outcomes against CAFO counts by commodity and size class                (script2a section 9)
  state-trends             state-level outcome vs CAFO counts, states with most large operations      (script2a section 10)
  state-dots               county dots by state, 2017: poor mental health days vs total CAFO operations (script2a section 12)
  chicken-facets           chicken CAFOs vs poor mental health days by year and size class            (script2a section 12)
  species-overlap          counties by number of species with a large operation                     (script4s)
  control-coverage         percent filled, 2015-2020, of the 24 candidate controls, by type          (new; D-55)
  trends                   raw trends by county group, event-time paths, pre-treatment slopes       (script4v)
Reads:   newest Data/merged/*_panel.csv and *_panel_fsis.csv; Data/clean/*_cafo_ops_by_size_compact.csv; for `trends`
         the treatment definitions in script4_treatment.py and designs.py (no new definitions here)
Window:  each outcome is drawn over its own designs.OUTCOME_WINDOWS entry (outcomes without one: all panel years); exposures use
         all panel years. Descriptive only, not the estimation samples (A21 open)
Writes:  Data/output/figs/script1c/<date>_<stem>.{pdf,png} (183 mm / 89 mm, 600 dpi PNG + vector PDF) and the numbers behind
         each figure in Data/output/tables/script1c/<date>_<stem>.csv (`control-coverage` also writes the same numbers as .txt)
Run:     PYTHONHASHSEED=0 python3 script1c-figures.py            about 4 min (trend-sweep is most of it; --max-trend-vars N shortens it)
Style:   figstyle.py (Nature/Science standards, .claude/rules/figure-and-table-standards.md). Stochastic code: the box-plot
         jitter only, seeded inside the section. Tables of the descriptive stage are written by script1b-tables.py.
"""
import argparse
import functools
import os
import re
import textwrap
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import figstyle
from designs import CONTROL_TYPE, CONTROL_TYPE_COLOUR, FILL_THRESHOLD_PCT, OUTCOME_WINDOWS
from functions import latest_file_glob, log_per10k, normalize_panel_key, to_numeric_series
from packages import db_data, figs_dir, tables_dir
from script4_treatment import COVARIATE_ORDER

warnings.filterwarnings("ignore", category=FutureWarning)

SCRIPT = "script1c-figures.py"
OUT_FIGS = Path(figs_dir) / "script1c"
OUT_TABLES = Path(tables_dir) / "script1c"
SEED = 20261024                     # box-plot jitter only
FSIS_YEARS = (2017, 2023)           # FSIS county linkage exists only for these years
CENSUS_YEARS = [2002, 2007, 2012, 2017, 2022]
POP, CAFO_TOTAL = "population", "cafo_total_ops_all_animals"
FSIS_TOTAL, FSIS_SLAUGHTER = "n_unique_establishments_fsis", "n_slaughterhouse_present_establishments_fsis"
ANIMALS = ["Cattle", "Hogs", "Chickens"]
SIZES = ["Small", "Medium", "Large"]
X_CAFO = "log(CAFO operations per 10,000 residents + 1)"
STATE_PLOT_LIMIT = 12               # states with the most large operations in `state-trends` (Arkansas is always added)
MAX_TREND_VARS = None               # cap on the variables of `trend-sweep` (--max-trend-vars); None = all
COVERAGE_YEARS = (2015, 2020)       # `control-coverage`: years over which the fill rate of each candidate control is measured
N_SELECTED = 8                      # the selected control set is COVARIATE_ORDER[:N_SELECTED] (D-16)

# One colour per variable, the same in every figure (Okabe-Ito via figstyle).
P = figstyle.PALETTE
ANIMAL_COLOUR = {"Cattle": P["orange"], "Hogs": P["purple"], "Chickens": P["green"]}
COMMODITY_COLOUR = {"cattle": P["orange"], "hogs": P["purple"], "chickens": P["green"], "beef": P["vermillion"], "dairy": P["sky"]}
ANIMAL_MARKER = {"Cattle": "o", "Hogs": "s", "Chickens": "^"}
ANIMAL_HATCH = {"Cattle": "", "Hogs": "//", "Chickens": ".."}
TYPE_HATCH = {"Health": "", "Demographic": "///", "Economic": "....", "Social and education": "xxx"}   # greyscale-legible control types
EXPOSURE_COLOUR = {"total": P["blue"], "large": P["vermillion"], "small": P["sky"], "fsis": P["green"], "slaughter": P["purple"]}
_viridis = plt.get_cmap(figstyle.SEQUENTIAL_CMAP)
SIZE_COLOUR = dict(zip(SIZES, (mcolors.to_hex(_viridis(v)) for v in (0.95, 0.55, 0.10))))   # light -> dark = small -> large
QUARTILE_COLOUR = [mcolors.to_hex(np.array(mcolors.to_rgb("white")) * (1 - a) + np.array(mcolors.to_rgb(P["blue"])) * a)
                   for a in (0.25, 0.5, 0.75, 1.0)]
FIT_COLOUR, NOTE_COLOUR = "black", "#333333"

# (column, name, unit-and-source) of the six outcomes shown in the descriptive figures
OUTCOMES = [
    ("poor_mental_health_days", "Poor mental health days", "days per month, CHR"),
    ("excessive_drinking_per100k", "Excessive drinking", "per 100,000 population, CHR"),
    ("violent_crime", "Violent crime rate", "per 100,000, CHR"),
    ("homicides", "Homicide rate", "per 100,000, CHR"),
    ("aggravated_assault_per100k", "Aggravated assault", "per 100,000, UCR"),
    ("crude_rate_despair", "Deaths of despair, crude rate", "per 100,000, CDC"),
]
OUTCOME_LABEL = {c: f"{n}\n({u})" for c, n, u in OUTCOMES}
OUTCOME_TIGHT = {c: textwrap.fill(f"{n} ({u})", 22) for c, n, u in OUTCOMES}      # three short lines: fits a 25 mm tall panel
# OUTCOMES column -> label of designs.OUTCOME_WINDOWS (same quantity as the stage-4 outcome). Excessive drinking, CHR violent
# crime and homicides are not stage-4 outcomes, have no window, and are drawn over every panel year they are observed.
WINDOW_LABEL_OF = {"poor_mental_health_days": "Poor MH Days", "aggravated_assault_per100k": "Aggravated Assault",
                   "crude_rate_despair": "Deaths of Despair"}
WINDOWS_NOTE = ("Each outcome is drawn over its own window (poor mental health days " + "{}-{}".format(*OUTCOME_WINDOWS["Poor MH Days"])
                + ", aggravated assault " + "{}-{}".format(*OUTCOME_WINDOWS["Aggravated Assault"]) + ", deaths of despair "
                + "{}-{}".format(*OUTCOME_WINDOWS["Deaths of Despair"]) + "; the other outcomes every year observed)")
STATE_NAME = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California", "08": "Colorado", "09": "Connecticut",
    "10": "Delaware", "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine", "24": "Maryland",
    "25": "Massachusetts", "26": "Michigan", "27": "Minnesota", "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York", "37": "North Carolina",
    "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia",
    "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming"}
STATE_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY"}


# =============================================================================
# Shared set-up: data loaders (read once), table and figure writers
# =============================================================================
def _read_panel(pattern):
    path = latest_file_glob(os.path.join(db_data, "merged"), pattern)
    df = normalize_panel_key(pd.read_csv(path, low_memory=False))
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.drop_duplicates(subset=["fips", "year"], keep="first").reset_index(drop=True)
    df["state_fips"] = df["fips"].astype("string").str[:2]
    analysis = [POP, CAFO_TOTAL, FSIS_TOTAL, FSIS_SLAUGHTER] + [c for c, _, _ in OUTCOMES] + \
               [f"cafo_{a.lower()}_{s.lower()}" for a in ANIMALS for s in SIZES]
    for col in analysis:
        if col in df.columns:
            df[col] = to_numeric_series(df[col])
    print(f"loaded {os.path.basename(path)}: {len(df):,} rows, {df['fips'].nunique():,} counties, "
          f"{int(df['year'].min())}-{int(df['year'].max())}")
    return df


@functools.cache
def panel():
    """The analysis panel (rural counties, 2000-2023)."""
    return _read_panel("*_panel.csv")


@functools.cache
def panel_fsis():
    """The 2017-2023 slice that carries the FSIS columns (they are not in the main panel; python-code-conventions section 4)."""
    return _read_panel("*_panel_fsis.csv")


@functools.cache
def cafo_compact():
    """Operations by commodity and size class from the compact clean file, restricted to panel county-years."""
    path = latest_file_glob(os.path.join(db_data, "clean"), "*_cafo_ops_by_size_compact.csv")
    cafo = normalize_panel_key(pd.read_csv(path, low_memory=False))
    cafo = cafo.merge(panel()[["fips", "year"]].drop_duplicates(), on=["fips", "year"], how="inner")
    for col in ["small", "medium", "large"]:
        cafo[col] = pd.to_numeric(cafo[col], errors="coerce").fillna(0) if col in cafo.columns else 0
    for col in ["commodity_desc", "class_desc"]:
        if col not in cafo.columns:
            cafo[col] = "unknown"
        cafo[col] = cafo[col].astype("string").str.lower().str.strip()
    # Keep one canonical class per animal. The compact file has no class_desc column, so every cattle, hog and chicken row
    # fails this test and only beef and dairy survive; this is the behaviour of script2a, kept so the numbers are unchanged.
    canonical = {"cattle": "incl calves", "hogs": "all classes", "chickens": "layers"}
    keep = (cafo["commodity_desc"].map(canonical) == cafo["class_desc"]) | ~cafo["commodity_desc"].isin(canonical)
    dropped = cafo.loc[~keep, "commodity_desc"].value_counts()
    if len(dropped):
        print("WARNING cafo_compact: rows dropped by the canonical-class filter (no class_desc column): "
              + ", ".join(f"{k} {v:,}" for k, v in dropped.items()))
    cafo = cafo[keep].copy()
    long = cafo.melt(id_vars=["fips", "year", "commodity_desc", "class_desc"], value_vars=["small", "medium", "large"],
                     var_name="size_bucket", value_name="ops_count").dropna(subset=["ops_count"])
    return cafo, long


@functools.cache
def numeric_columns():
    """Panel columns that read as numeric (numeric dtype, or at least 80% of the non-missing values parse), as floats."""
    pan = panel()
    out = {}
    for c in [c for c in pan.columns if c not in ("fips", "year", "state_fips")]:   # state_fips is added by _read_panel
        s = to_numeric_series(pan[c])
        n_raw, n_num = int(pan[c].notna().sum()), int(s.notna().sum())
        if pd.api.types.is_numeric_dtype(pan[c]) or (n_raw and n_num / n_raw >= 0.8 and n_num > 0):
            out[c] = s
    return out


@functools.cache
def cafo_counts():
    """Operations by county, year and commodity and size class (small, medium, large), from the compact file."""
    cafo, _ = cafo_compact()
    return cafo.groupby(["fips", "year", "commodity_desc"], as_index=False)[["small", "medium", "large"]].sum()


def safe_stem(text):
    """File-name-safe stem (column names such as %_black or fondling_(incident_liberties/child_molest)_crime)."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(text)).strip("_").lower()[:90]


def write_table(df, stem, index=False):
    """Numbers behind a figure, dated, in tables/script1c."""
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    path = OUT_TABLES / f"{figstyle.dated(stem)}.csv"
    df.to_csv(path, index=index)
    print("  table", path.name, f"({len(df):,} rows)")
    return path


def save(fig, stem, note):
    """Source note under the figure, then PDF + PNG through figstyle. The note names data, sample, method and this script."""
    w_mm, h_mm = np.array(fig.get_size_inches()) * figstyle.MM_PER_INCH
    text = textwrap.fill(f"{note} Script: {SCRIPT}.", width=int(w_mm / 0.95))
    frac = ((text.count("\n") + 1) * 2.5 + 2.0) / h_mm
    fig.get_layout_engine().set(rect=(0, frac, 1, 1 - frac))
    fig.text(0.005, 0.004, text, fontsize=5, va="bottom", ha="left", color=NOTE_COLOUR)
    pdf, png = figstyle.save_figure(fig, OUT_FIGS, stem)
    plt.close(fig)
    print("  figure", png.name)


def letters(axes):
    """Bold panel letters a, b, c ... on the visible axes, in reading order."""
    for ax, letter in zip([a for a in np.ravel(axes) if a.get_visible()], "abcdefghijklmnopqrstuvwxyz"):
        figstyle.panel_label(ax, letter)


def hide_unused(axes, n_used):
    for ax in np.ravel(axes)[n_used:]:
        ax.set_visible(False)


def text_colour_on(hex_colour):
    r, g, b = mcolors.to_rgb(hex_colour)
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def window(df, years):
    return df[df["year"].between(*years)].copy()


def window_text(col):
    """'2010-2023' for an outcome with a designs.OUTCOME_WINDOWS entry, 'all years' otherwise."""
    key = WINDOW_LABEL_OF.get(col)
    return "{}-{}".format(*OUTCOME_WINDOWS[key]) if key else "all years"


def outcome_windowed(df):
    """Copy of df (all years) with every OUTCOMES column set to missing outside its own window; exposures are untouched, so a
    pair of columns is complete only where each is observed inside its window (pairwise complete, as the figures drop NaN)."""
    out = df.copy()
    for col, _, _ in OUTCOMES:
        key = WINDOW_LABEL_OF.get(col)
        if key and col in out.columns:
            out[col] = to_numeric_series(out[col]).where(out["year"].between(*OUTCOME_WINDOWS[key]))
    return out


# =============================================================================
# binned-scatter (script2b A1 A2 A3 I J; script2d P R)
# =============================================================================
def bin_summary(x_vals, y_vals, n_bins, variant):
    """Equal-count bins of x: bin mean of y, 95% CI and a fitted line.
    variant "micro" (script2b): CI = 1.96 x SE, line = OLS on the county-years, reported with r and p.
    variant "bins"  (script2d): CI = t quantile x SE, line = OLS on the bin means (no r, p)."""
    x, y = pd.Series(x_vals, dtype=float), pd.Series(y_vals, dtype=float)
    ok = x.notna() & y.notna()
    x, y, n = x[ok].to_numpy(), y[ok].to_numpy(), int(ok.sum())
    if n < n_bins * 3:
        return None, None, n
    if variant == "micro":
        try:
            cut = pd.qcut(x, q=n_bins, duplicates="drop")
        except ValueError:
            cut = pd.cut(x, bins=n_bins)
    else:
        cut = pd.qcut(x, q=n_bins, duplicates="drop")
    frame = pd.DataFrame({"x": x, "y": y, "bin": cut}).groupby("bin", observed=True)
    bins = pd.DataFrame({"x_mid": frame["x"].mean().to_numpy(), "y_mean": frame["y"].mean().to_numpy(),
                         "se": frame["y"].sem().to_numpy(), "n_bin": frame["y"].count().to_numpy()})
    if variant == "micro":
        bins["ci95"] = 1.96 * bins["se"]
        bins = bins.dropna(subset=["x_mid", "y_mean"]).reset_index(drop=True)
        slope, intercept, r, p, _ = stats.linregress(x, y)
        fit = {"kind": "OLS on county-years", "slope": slope, "intercept": intercept, "r": r, "p": p}
        fit["x_lo"], fit["x_hi"] = x.min(), x.max()
    else:
        bins["ci95"] = bins["se"] * stats.t.ppf(0.975, df=np.maximum(bins["n_bin"] - 1, 1))
        good = np.isfinite(bins["x_mid"]) & np.isfinite(bins["y_mean"])
        fit = None
        if good.sum() >= 3:
            slope, intercept = np.polyfit(bins.loc[good, "x_mid"], bins.loc[good, "y_mean"], 1)
            fit = {"kind": "OLS on bin means", "slope": slope, "intercept": intercept, "r": np.nan, "p": np.nan,
                   "x_lo": bins.loc[good, "x_mid"].min(), "x_hi": bins.loc[good, "x_mid"].max()}
    return bins, fit, n


def draw_binned(ax, cell, variant, bin_rows, fit_rows, fig_name):
    bins, fit, n = bin_summary(cell["x"], cell["y"], cell.get("n_bins", 20), variant)
    ax.set_xlabel(cell["xlabel"])
    ax.set_ylabel(cell["ylabel"])
    if cell.get("title"):
        ax.set_title(cell["title"], fontsize=6.5, pad=2)
    if bins is None:
        ax.text(0.5, 0.5, f"Insufficient data\n(n = {n:,})", ha="center", va="center", transform=ax.transAxes, color="#555555")
        return
    ax.errorbar(bins["x_mid"], bins["y_mean"], yerr=bins["ci95"], fmt="o", color=cell["colour"], ms=3, lw=0.8,
                capsize=1.5, elinewidth=0.6, zorder=3, label=f"Bin mean ± 95% CI (n = {n:,})")
    if fit is not None:
        xs = np.linspace(fit["x_lo"], fit["x_hi"], 100)
        if variant == "micro":
            p_txt = "p < 0.001" if fit["p"] < 0.001 else f"p = {fit['p']:.3f}"
            lab = f"OLS: β = {fit['slope']:.3f}, r = {fit['r']:.2f}, {p_txt}"
        else:
            lab = f"OLS on bin means: β = {fit['slope']:.3f}"
        ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color=FIT_COLOUR, lw=0.8, ls="--", zorder=4, label=lab)
        fit_rows.append({"figure": fig_name, "panel": cell["name"], "n": n, **{k: fit[k] for k in ("kind", "slope", "intercept", "r", "p")}})
    ax.legend(loc="best", fontsize=5, handlelength=1.6)
    for i, row in bins.iterrows():
        bin_rows.append({"figure": fig_name, "panel": cell["name"], "bin": i + 1, "x_mid": row["x_mid"], "y_mean": row["y_mean"],
                         "ci95": row["ci95"], "n_bin": int(row["n_bin"])})


def binned_figure(stem, cells, shape, height_mm, note, bin_rows, fit_rows, variant="micro", sharey_row=False, ylabel_size=None):
    nrows, ncols = shape
    fig, axes = figstyle.new_grid(nrows, ncols, width="double", height_mm=height_mm, sharey="row" if sharey_row else False)
    flat = axes.ravel()
    for ax, cell in zip(flat, cells):
        draw_binned(ax, cell, variant, bin_rows, fit_rows, stem)
        if ylabel_size:
            ax.set_ylabel(textwrap.fill(cell["ylabel"].replace("\n", " "), 24), fontsize=ylabel_size)
    hide_unused(flat, len(cells))
    letters(flat)
    save(fig, stem, note)


def section_binned_scatter():
    """Binned means of each outcome against log(exposure per 10,000 residents + 1); one figure per exposure family."""
    pan, fs = panel(), panel_fsis()
    dp = outcome_windowed(pan)
    dp["x_total"] = log_per10k(dp[CAFO_TOTAL], dp[POP])
    base = f"Rural US county-years. {WINDOWS_NOTE}. Dots: equal-count bin means of the outcome " \
           "with 95% CI (1.96 x SE); dashed line: OLS fit on the county-years with r and p; exposure = log(operations per 10,000 residents + 1)."
    bins_, fits_ = [], []

    cells = [dict(name=c, x=dp["x_total"], y=dp[c], colour=EXPOSURE_COLOUR["total"], xlabel=X_CAFO, ylabel=OUTCOME_LABEL[c])
             for c, _, _ in OUTCOMES if c in dp.columns]
    binned_figure("binned_cafo_total_vs_outcomes", cells, (2, 3), 112, base, bins_, fits_)

    cells = []
    for c, label in ((OUTCOMES[0][0], "Poor mental health days\n(days per month)"), (OUTCOMES[1][0], "Excessive drinking\n(per 100,000)"),
                     (OUTCOMES[2][0], "Violent crime rate\n(per 100,000)")):
        for animal in ANIMALS:
            cols = [f"cafo_{animal.lower()}_{s.lower()}" for s in SIZES if f"cafo_{animal.lower()}_{s.lower()}" in dp.columns]
            x = log_per10k(dp[cols].sum(axis=1, min_count=1), dp[POP])
            cells.append(dict(name=f"{c}|{animal}", x=x, y=dp[c], colour=ANIMAL_COLOUR[animal], title=animal if c == OUTCOMES[0][0] else None,
                              xlabel=f"log({animal.lower()} operations per 10,000 + 1)", ylabel=label if animal == "Cattle" else ""))
    binned_figure("binned_cafo_by_animal_vs_outcomes", cells, (3, 3), 160, base + " Columns: animal type (small + medium + large operations); "
                  "rows: outcome.", bins_, fits_)

    df = outcome_windowed(window(fs, FSIS_YEARS))
    if FSIS_TOTAL in df.columns and df[FSIS_TOTAL].notna().sum() > 50:
        x = log_per10k(df[FSIS_TOTAL], df[POP])
        cells = [dict(name=c, x=x, y=df[c], colour=EXPOSURE_COLOUR["fsis"], xlabel="log(FSIS establishments per 10,000 residents + 1)",
                      ylabel=OUTCOME_LABEL[c]) for c, _, _ in OUTCOMES if c in df.columns]
        binned_figure("binned_fsis_vs_outcomes", cells, (2, 3), 112,
                      f"Rural US counties, {FSIS_YEARS[0]}-{FSIS_YEARS[1]} county-years with at least one FSIS establishment (the FSIS count is missing, "
                      "not zero, for the others). Dots: bin means with 95% CI; dashed line: OLS on the county-years.", bins_, fits_)

    subtypes = [("Meat slaughter", "n_meat_slaughter_establishments_fsis"), ("Poultry slaughter", "n_poultry_slaughter_establishments_fsis"),
                ("Slaughter only", "n_type_slaughter_only_fsis"), ("Processing only", "n_type_processing_only_fsis"),
                ("Slaughter and processing", "n_type_both_slaughter_and_processing_fsis")]
    focus = [("crude_rate_despair", "Deaths of despair, crude rate\n(per 100,000, CDC)"),
             ("poor_mental_health_days", "Poor mental health days\n(days per month, CHR)")]
    cells = []
    for sub_label, col in subtypes:
        if col not in df.columns:
            continue
        x = log_per10k(df[col], df[POP])
        for oc, olab in focus:
            cells.append(dict(name=f"{col}|{oc}", x=x, y=df[oc], colour=EXPOSURE_COLOUR["fsis"], n_bins=15,
                              xlabel=f"log({sub_label.lower()} establishments per 10,000 + 1)", ylabel=olab))
    binned_figure("binned_fsis_subtypes_vs_despair_mhdays", cells, (len(cells) // 2, 2), 200,
                  f"Rural US counties, {FSIS_YEARS[0]}-{FSIS_YEARS[1]} county-years with the FSIS subtype count observed. 15 equal-count bins; "
                  "dots: bin means with 95% CI; dashed line: OLS on the county-years.", bins_, fits_, ylabel_size=5.5)

    cells = []
    for animal in ANIMALS:
        for size in SIZES:
            col = f"cafo_{animal.lower()}_{size.lower()}"
            if col in dp.columns:
                cells.append(dict(name=col, x=log_per10k(dp[col], dp[POP]), y=dp["crude_rate_despair"], colour=ANIMAL_COLOUR[animal], n_bins=15,
                                  title=f"{size} operations" if animal == "Cattle" else None,
                                  xlabel=f"log({animal.lower()}, {size.lower()}, per 10,000 + 1)",
                                  ylabel=f"{animal}\nDeaths of despair (per 100,000, CDC)" if size == "Small" else ""))
    binned_figure("binned_cafo_animal_size_vs_despair", cells, (3, 3), 160,
                  f"Rural US county-years; deaths of despair over {window_text('crude_rate_despair')}. Rows: animal type; columns: operation size class. 15 equal-count bins; "
                  "dots: bin means with 95% CI; dashed line: OLS on the county-years.", bins_, fits_)

    # script2d: the same plot for large operations only, and small vs large side by side (fit on bin means, t-based CI)
    dp["x_large"] = log_per10k(dp["large_cafo"], dp[POP])
    dp["x_small"] = log_per10k(dp["small_cafo"], dp[POP])
    cells = [dict(name=c, x=dp["x_large"], y=dp[c], colour=EXPOSURE_COLOUR["large"], xlabel="log(large CAFO operations per 10,000 + 1)",
                  ylabel=OUTCOME_LABEL[c]) for c, _, _ in OUTCOMES if c in dp.columns]
    binned_figure("binned_large_cafo_vs_outcomes", cells, (2, 3), 112,
                  f"Rural US county-years. {WINDOWS_NOTE}; large size class only (compare with the total-CAFO figure). 20 equal-count "
                  "bins; dots: bin means with t-based 95% CI; dashed line: OLS on the bin means.", bins_, fits_, variant="bins")
    cells = []
    for c, label in ((o[0], OUTCOME_LABEL[o[0]]) for o in (OUTCOMES[5], OUTCOMES[0], OUTCOMES[4])):
        cells.append(dict(name=f"{c}|small", x=dp["x_small"], y=dp[c], colour=EXPOSURE_COLOUR["small"], ylabel=label,
                          title="Small CAFO operations" if c == "crude_rate_despair" else None, xlabel="log(small CAFO operations per 10,000 + 1)"))
        cells.append(dict(name=f"{c}|large", x=dp["x_large"], y=dp[c], colour=EXPOSURE_COLOUR["large"], ylabel="",
                          title="Large CAFO operations" if c == "crude_rate_despair" else None, xlabel="log(large CAFO operations per 10,000 + 1)"))
    binned_figure("binned_small_vs_large_cafo", cells, (3, 2), 150,
                  f"Rural US county-years. {WINDOWS_NOTE}. Left: small, right: large operations; y-axes are matched within each row. "
                  "Dots: bin means with t-based 95% CI; dashed line: OLS on the bin means.", bins_, fits_, variant="bins", sharey_row=True)

    write_table(pd.DataFrame(bins_), "binned_scatter_bins")
    write_table(pd.DataFrame(fits_), "binned_scatter_fits")


# =============================================================================
# outcome-correlation (script2b C)
# =============================================================================
def section_outcome_correlation():
    d = outcome_windowed(panel())
    short = {"poor_mental_health_days": "Poor mental health days (CHR)", "excessive_drinking_per100k": "Excessive drinking (CHR)",
             "violent_crime": "Violent crime (CHR)", "homicides": "Homicides (CHR)",
             "aggravated_assault_per100k": "Aggravated assault (UCR)", "crude_rate_despair": "Deaths of despair rate (CDC)"}
    data = pd.DataFrame({short[c]: to_numeric_series(d[c]) for c, _, _ in OUTCOMES if c in d.columns})
    corr = data.corr(method="spearman", min_periods=200)
    write_table(corr, "outcome_crosscorrelation", index=True)
    fig, ax = figstyle.new_figure("one-and-half", height_mm=100)
    shown = corr.rename(index=lambda t: t.replace(" (", "\n("), columns=lambda t: t.replace(" (", "\n("))      # two-line labels fit the figure
    sns.heatmap(shown, ax=ax, mask=np.triu(np.ones(shown.shape, dtype=bool), k=1), cmap=figstyle.DIVERGING_CMAP, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 5.5}, linewidths=0.4, square=True,
                cbar_kws={"label": "Spearman rank correlation (ρ)", "shrink": 0.7})
    ax.tick_params(axis="x", rotation=30, labelsize=5.5)
    ax.tick_params(axis="y", rotation=0, labelsize=5.5)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    save(fig, "outcome_crosscorrelation",
         f"Spearman correlations between the six outcomes, rural US county-years, pairwise complete. {WINDOWS_NOTE}; pairs need at least 200 joint observations. "
         "Lower triangle shown. Source: analysis panel.")


# =============================================================================
# quartile-distributions (script2b D E)
# =============================================================================
def section_quartile_distributions():
    rng = np.random.default_rng(SEED)
    d = outcome_windowed(panel())
    d["cafo_log_per10k"] = log_per10k(d[CAFO_TOTAL], d[POP])
    labels = ["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"]
    d["cafo_quartile"] = pd.qcut(d["cafo_log_per10k"], q=4, labels=labels, duplicates="drop")
    xlabel = "CAFO-intensity quartile\n(log operations per 10,000 + 1)"
    rows = []
    for kind in ("violin", "box"):
        fig, axes = figstyle.new_grid(2, 3, width="double", height_mm=118)
        flat = axes.ravel()
        for ax, (col, _, _) in zip(flat, OUTCOMES):
            sub = d[["cafo_quartile", col]].dropna().copy()
            if sub.empty or sub["cafo_quartile"].nunique() < 2:
                ax.set_visible(False)
                continue
            sub["y"] = sub[col]
            if kind == "violin":
                sns.violinplot(data=sub, x="cafo_quartile", y="y", hue="cafo_quartile", palette=QUARTILE_COLOUR, legend=False,
                               inner="quartile", linewidth=0.5, cut=0, ax=ax)
                cats = list(sub["cafo_quartile"].cat.categories)
                meds = [sub.loc[sub["cafo_quartile"] == q, "y"].median() for q in cats]
                ax.set_xticks(range(len(cats)), [f"{q}\nmedian {m:,.0f}" if abs(m) >= 100 else f"{q}\nmedian {m:.2f}" for q, m in zip(cats, meds)])
            else:
                groups = [sub.loc[sub["cafo_quartile"] == q, "y"].to_numpy() for q in sub["cafo_quartile"].cat.categories]
                for j, g in enumerate(groups):   # raw county-years behind the boxes
                    ax.scatter(j + rng.uniform(-0.2, 0.2, len(g)), g, s=0.6, color="#555555", alpha=0.12, lw=0, rasterized=True, zorder=1)
                bp = ax.boxplot(groups, positions=range(len(groups)), widths=0.5, patch_artist=True, showfliers=False,
                                medianprops={"color": P["vermillion"], "lw": 1.2}, boxprops={"lw": 0.5}, whiskerprops={"lw": 0.5},
                                capprops={"lw": 0.5})
                for patch, colour in zip(bp["boxes"], QUARTILE_COLOUR):
                    patch.set_facecolor(colour)
                    patch.set_zorder(2)
                ax.set_xticks(range(len(groups)), list(sub["cafo_quartile"].cat.categories))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(OUTCOME_LABEL[col])
            if kind == "violin":
                for q, g in sub.groupby("cafo_quartile", observed=True)["y"]:
                    rows.append({"outcome": col, "quartile": str(q).replace("\n", " "), "n": len(g), "mean": g.mean(),
                                 "p25": g.quantile(0.25), "median": g.median(), "p75": g.quantile(0.75)})
        letters(flat)
        stem = f"{kind}_cafo_quartile_vs_outcomes"
        if kind == "violin":
            save(fig, stem, f"Rural US county-years, grouped by quartile of log(total CAFO operations per 10,000 residents + 1) over all panel years. {WINDOWS_NOTE}. "
                 "Violin width = density (cut at the observed range); inner lines = Q1, median, Q3; the median is printed under each quartile label.")
        else:
            save(fig, stem, f"Rural US county-years, grouped by quartile of log(total CAFO operations per 10,000 residents + 1) over all panel years. {WINDOWS_NOTE}. "
                 "Box = interquartile range, red line = median, whiskers = 1.5 x IQR, grey dots = county-years (jittered, seed fixed).")
    write_table(pd.DataFrame(rows), "quartile_distribution_summary")


# =============================================================================
# cafo-top-counties (script2c K L)
# =============================================================================
def section_cafo_top_counties():
    """Top-20 counties by operations per 10,000 residents in each census year: all sizes combined, then large only."""
    df = panel().copy()
    state_col = "state" if "state" in df.columns else "state_code"
    df["county_label"] = df["fips"].astype("string") + " (" + df[state_col].astype("string") + ")"
    n_top = 20
    rows = []
    for tag, scope in (("total", "all sizes"), ("large", "large operations")):
        for animal in ANIMALS:
            fig, axes = figstyle.new_grid(2, 3, width="double", height_mm=190)
            flat = axes.ravel()
            for ax, yr in zip(flat, CENSUS_YEARS):
                dy = df[df["year"] == yr].copy()
                if tag == "total":
                    cols = [f"cafo_{animal.lower()}_{s.lower()}" for s in SIZES if f"cafo_{animal.lower()}_{s.lower()}" in dy.columns]
                    dy["ops"] = dy[cols].sum(axis=1, min_count=1)
                else:
                    dy["ops"] = dy[f"cafo_{animal.lower()}_large"]
                sub = dy[dy["ops"].notna() & (dy["ops"] > 0)].copy()
                sub["log_per10k"] = log_per10k(sub["ops"], sub[POP])
                top = sub.nlargest(n_top, "log_per10k")
                for rank, (_, r) in enumerate(top.iterrows(), 1):
                    rows.append({"animal": animal, "scope": scope, "year": yr, "rank": rank, "fips": r["fips"], "county_label": r["county_label"],
                                 "operations": r["ops"], "population": r[POP], "log_ops_per10k": r["log_per10k"]})
                top = top.sort_values("log_per10k")
                ax.barh(top["county_label"], top["log_per10k"], color=ANIMAL_COLOUR[animal], edgecolor="white", linewidth=0.3)
                ax.set_xlabel(f"log({'large ' if tag == 'large' else ''}operations per 10,000 + 1)")
                ax.set_title(str(yr), fontsize=6.5, pad=2)
                ax.tick_params(axis="y", labelsize=5)
                ax.set_xlim(left=0)
            hide_unused(flat, len(CENSUS_YEARS))
            letters(flat)
            save(fig, f"top{n_top}_{animal.lower()}_{tag}_by_census_year",
                 f"Top {n_top} rural US counties by {animal.lower()} operations per 10,000 residents ({scope}), by census year; county label = FIPS (state). "
                 "Counties with no operations excluded. Source: analysis panel (USDA Census of Agriculture).")
    write_table(pd.DataFrame(rows), "top_counties_by_census_year")


# =============================================================================
# cafo-census-composition (script2c M N P)
# =============================================================================
def section_cafo_census_composition():
    """National totals of operations by animal x size class x census year, and the within-animal size shares."""
    df = panel()
    recs = []
    for animal in ANIMALS:
        for size in SIZES:
            col = f"cafo_{animal.lower()}_{size.lower()}"
            for yr in CENSUS_YEARS:
                dy = df[df["year"] == yr]
                recs.append({"animal": animal, "size": size, "year": yr, "total_ops": dy[col].sum(skipna=True) if col in dy.columns else 0.0})
    tot = pd.DataFrame(recs)
    tot["share_pct"] = tot["total_ops"] / tot.groupby(["animal", "year"])["total_ops"].transform("sum").replace(0, np.nan) * 100
    tot["share_pct"] = tot["share_pct"].fillna(0.0)
    write_table(tot, "cafo_national_totals_and_shares_by_census_year")
    shares = tot.pivot_table(index=["animal", "year"], columns="size", values="share_pct")
    n_yr = len(CENSUS_YEARS)
    x = np.arange(n_yr)

    # M: 100% stacked bars of the size shares, one panel per animal
    fig, axes = figstyle.new_grid(1, 3, width="double", height_mm=72)
    for ax, animal in zip(axes.ravel(), ANIMALS):
        bottom = np.zeros(n_yr)
        for size in SIZES:
            vals = np.array([shares.loc[(animal, yr), size] for yr in CENSUS_YEARS])
            ax.bar(x, vals, bottom=bottom, color=SIZE_COLOUR[size], edgecolor="white", linewidth=0.4, width=0.6, label=size)
            for xi, (b, v) in enumerate(zip(bottom, vals)):
                if v >= 5:
                    ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=5, color=text_colour_on(SIZE_COLOUR[size]), fontweight="bold")
            bottom += vals
        ax.set_xticks(x, [str(y) for y in CENSUS_YEARS])
        ax.set_xlabel("Census of Agriculture year")
        ax.set_ylabel("Share of operations (%)")
        ax.set_ylim(0, 100)
        ax.set_title(animal, fontsize=6.5, pad=2)
    axes.ravel()[-1].legend(title="Size class", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=5.5, title_fontsize=5.5)
    letters(axes)
    save(fig, "cafo_size_composition_by_animal",
         "Share of operations by size class within each animal type, summed over all rural US counties in the analysis panel, by census year; labels shown for shares of "
         "5% or more. Source: analysis panel (USDA Census of Agriculture).")

    # N: national counts, stacked by size within animal, animal side by side; opacity = size class
    fig, ax = figstyle.new_figure("double", height_mm=88)
    width, gap = 0.22, 0.04
    for ai, animal in enumerate(ANIMALS):
        rgb = mcolors.to_rgb(ANIMAL_COLOUR[animal])
        for yi, yr in enumerate(CENSUS_YEARS):
            bottom = 0.0
            for size, alpha in zip(SIZES, (0.35, 0.65, 1.0)):
                val = tot[(tot.animal == animal) & (tot["size"] == size) & (tot.year == yr)]["total_ops"].iloc[0]
                ax.bar(yi + (ai - 1) * (width + gap), val, bottom=bottom, width=width, color=(*rgb, alpha), edgecolor="#333333", linewidth=0.3,
                       hatch=ANIMAL_HATCH[animal])
                bottom += val
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=ANIMAL_COLOUR[a], edgecolor="#333333", hatch=ANIMAL_HATCH[a], label=a) for a in ANIMALS] + \
              [plt.Rectangle((0, 0), 1, 1, facecolor="grey", alpha=al, label=f"{s} size") for s, al in zip(SIZES, (0.35, 0.65, 1.0))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.5, title="Animal / size class", title_fontsize=5.5)
    ax.set_xticks(np.arange(n_yr), [str(y) for y in CENSUS_YEARS])
    ax.set_xlabel("Census of Agriculture year")
    ax.set_ylabel("Operations (count, all rural counties)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K"))
    save(fig, "cafo_operations_by_animal_and_size",
         "National count of operations over all rural US counties in the analysis panel, by census year. Colour and hatch = animal type (cattle, hogs: //, chickens: ..); "
         "opacity = size class (light = small, full = large); bars stack small, medium, large.")

    # P: size shares of the three animals side by side within each census year
    fig, ax = figstyle.new_figure("double", height_mm=88)
    bar_w, gap, span = 0.22, 0.04, 3 * 0.22 + 2 * 0.04
    centres = np.arange(n_yr) * (span + 0.35)
    for yi, yr in enumerate(CENSUS_YEARS):
        for ai, animal in enumerate(ANIMALS):
            xb, bottom = centres[yi] + (ai - 1) * (bar_w + gap), 0.0
            for size in SIZES:
                val = shares.loc[(animal, yr), size]
                ax.bar(xb, val, bottom=bottom, width=bar_w, color=SIZE_COLOUR[size], edgecolor="#333333", linewidth=0.3, hatch=ANIMAL_HATCH[animal])
                if val >= 6:
                    ax.text(xb, bottom + val / 2, f"{val:.0f}%", ha="center", va="center", fontsize=5, fontweight="bold", color="black",
                            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.85})    # readable over hatching
                bottom += val
            ax.text(xb, 101, ("C", "H", "Ch")[ai], ha="center", va="bottom", fontsize=5, color="#555555")
    ax.set_xticks(centres, [str(y) for y in CENSUS_YEARS])
    ax.set_ylim(0, 108)
    ax.set_xlabel("Census of Agriculture year")
    ax.set_ylabel("Share of operations within animal type (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SIZE_COLOUR[s], edgecolor="grey", label=s) for s in SIZES] + \
              [plt.Rectangle((0, 0), 1, 1, facecolor="#cccccc", edgecolor="#333333", hatch=ANIMAL_HATCH[a], label=a) for a in ANIMALS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.5, title="Size class / animal", title_fontsize=5.5)
    save(fig, "cafo_size_composition_all_animals_combined",
         "Size-class shares within animal type (C = cattle, H = hogs: //, Ch = chickens: ..), side by side for each census year; colour = size class (light = small, dark = large). "
         "All rural US counties in the analysis panel.")


# =============================================================================
# threshold-presence (script2d O Q)
# =============================================================================
def mean_ci(series, alpha=0.05):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return np.nan, np.nan, np.nan
    h = stats.sem(s) * stats.t.ppf(1 - alpha / 2, df=len(s) - 1)
    return s.mean(), s.mean() - h, s.mean() + h


def _positive(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.gt(0).astype("Int64").where(s.notna())


def section_threshold_presence():
    """Mean outcome with 95% CI by presence of large CAFOs / FSIS establishments (0/1) and by count tier."""
    dp, df = outcome_windowed(panel()), outcome_windowed(window(panel_fsis(), FSIS_YEARS))
    df["fsis_any"] = _positive(df[FSIS_TOTAL])
    df["fsis_slaughter_any"] = _positive(df[FSIS_SLAUGHTER])
    win_p, win_f = "outcome windows", f"{FSIS_YEARS[0]}-{FSIS_YEARS[1]}"
    rows = []

    specs = [("Any large CAFO", "any_large_cafo", dp, ["No large CAFO", "Large CAFO"], EXPOSURE_COLOUR["large"], win_p),
             ("Any FSIS establishment", "fsis_any", df, ["No FSIS", "FSIS present"], EXPOSURE_COLOUR["fsis"], win_f),
             ("Any FSIS slaughterhouse", "fsis_slaughter_any", df, ["No slaughterhouse", "Slaughterhouse"], EXPOSURE_COLOUR["slaughter"], win_f)]
    fig, axes = figstyle.new_grid(len(OUTCOMES), 3, width="double", height_mm=200, sharey="row")
    for ci, (title, col, data, glabels, colour, win) in enumerate(specs):
        for ri, (oc, _, _) in enumerate(OUTCOMES):
            ax = axes[ri, ci]
            ticks = []
            for gi, (gval, glab) in enumerate(((0, glabels[0]), (1, glabels[1]))):
                s = data.loc[data[col] == gval, oc]
                m, lo, hi = mean_ci(s)
                n = int(s.notna().sum())
                rows.append({"figure": "presence", "exposure": title, "window": window_text(oc) if win == win_p else win, "outcome": oc, "group": glab, "n": n, "mean": m, "ci95_lo": lo, "ci95_hi": hi})
                ax.bar(gi, m, color=colour if gval else "#bbbbbb", width=0.55, edgecolor="white")
                if np.isfinite(lo):
                    ax.errorbar(gi, m, yerr=[[m - lo], [hi - m]], fmt="none", color="black", capsize=2, lw=0.6)
                ticks.append(f"{glab}\nn = {n:,}")
            ax.set_xticks([0, 1], ticks, fontsize=5)
            if ci == 0:
                ax.set_ylabel(OUTCOME_TIGHT[oc], fontsize=5.5)
            if ri == 0:
                ax.set_title(f"{title}\n({win})", fontsize=6.5, pad=2)
    letters(axes)
    save(fig, "presence_vs_absence_outcome_means",
         f"Mean outcome with 95% CI (t-based) for rural US county-years with and without a large CAFO (all panel years; {WINDOWS_NOTE}) and with and without an FSIS establishment or "
         f"slaughterhouse ({win_f}, FSIS panel). An FSIS count is missing, not zero, for county-years without an establishment, so the 'No FSIS' groups are empty (n = 0). "
         "Y-axes are shared along each row.")

    tiers = [("Large CAFO operations", "large_cafo", dp, [-0.001, 0, 3, 10, 1e9], ["None (0)", "1-3", "4-10", "11+"], EXPOSURE_COLOUR["large"], win_p),
             ("FSIS establishments", FSIS_TOTAL, df, [-0.001, 0, 1, 3, 1e9], ["None (0)", "1", "2-3", "4+"], EXPOSURE_COLOUR["fsis"], win_f),
             ("FSIS slaughterhouses", FSIS_SLAUGHTER, df, [-0.001, 0, 1, 3, 1e9], ["None (0)", "1", "2-3", "4+"], EXPOSURE_COLOUR["slaughter"], win_f)]
    fig, axes = figstyle.new_grid(len(OUTCOMES), 3, width="double", height_mm=200, sharey="row")
    for ci, (title, col, data, cuts, tlabels, colour, win) in enumerate(tiers):
        data = data.copy()
        data["_tier"] = pd.cut(pd.to_numeric(data[col], errors="coerce"), bins=cuts, labels=tlabels, right=True)
        for ri, (oc, _, _) in enumerate(OUTCOMES):
            ax = axes[ri, ci]
            ticks = []
            for ti, tl in enumerate(tlabels):
                s = data.loc[data["_tier"] == tl, oc]
                m, lo, hi = mean_ci(s)
                n = int(s.notna().sum())
                rows.append({"figure": "tiers", "exposure": title, "window": window_text(oc) if win == win_p else win, "outcome": oc, "group": tl, "n": n, "mean": m, "ci95_lo": lo, "ci95_hi": hi})
                ax.bar(ti, m, color=colour, width=0.65, edgecolor="white")
                if np.isfinite(lo):
                    ax.errorbar(ti, m, yerr=[[m - lo], [hi - m]], fmt="none", color="black", capsize=2, lw=0.6)
                ticks.append(f"{tl}\nn = {n:,}")
            ax.set_xticks(range(len(tlabels)), ticks, fontsize=4.8)
            if ci == 0:
                ax.set_ylabel(OUTCOME_TIGHT[oc], fontsize=5.5)
            if ri == 0:
                ax.set_title(f"{title}\n(count tier, {win})", fontsize=6.5, pad=2)
    letters(axes)
    save(fig, "dose_response_count_tiers",
         f"Mean outcome with 95% CI (t-based) by count tier of large CAFO operations per county-year (all panel years; {WINDOWS_NOTE}) and of FSIS establishments and slaughterhouses "
         f"({win_f}, FSIS panel; county-years without an establishment have a missing count and are not in any tier). Y-axes are shared along each row.")
    write_table(pd.DataFrame(rows), "presence_and_tier_means")


# =============================================================================
# fsis-size-bins (script2a section 3)
# =============================================================================
def section_fsis_size_bins():
    mental = "poor_mental_health_days"
    size_cols = [f"n_size_bucket_{i}_fsis" for i in range(1, 6)]
    df = panel_fsis()
    d17 = df[df["year"] == 2017].copy()
    d17[mental] = to_numeric_series(d17[mental])
    for c in size_cols:
        d17[c] = to_numeric_series(d17[c]).fillna(0)
    d17 = d17[d17[mental].notna()].copy()
    long = d17.melt(id_vars=["fips", "year", mental], value_vars=size_cols, var_name="size_bin_col", value_name="fsis_establishments_in_bin")
    long["size_bin"] = long["size_bin_col"].map({c: f"Size Bin {i + 1}" for i, c in enumerate(size_cols)})
    write_table(long, "fsis_size_bins_vs_pmhd_2017_plotdata")
    fig, axes = figstyle.new_grid(2, 3, width="double", height_mm=105)
    flat = axes.ravel()
    for i, c in enumerate(size_cols):
        ax = flat[i]
        ax.scatter(d17[c], d17[mental], s=3, color=EXPOSURE_COLOUR["fsis"], alpha=0.5, lw=0, rasterized=True)
        ax.set_xlabel("FSIS establishments in size bin (count per county)")
        ax.set_ylabel("Poor mental health days\n(days per month, CHR)" if i % 3 == 0 else "")
        ax.set_title(f"Size bin {i + 1}", fontsize=6.5, pad=2)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    flat[5].set_visible(False)
    flat[1].sharey(flat[0])
    letters(flat)
    save(fig, "fsis_size_bins_vs_poor_mental_health_days_2017",
         f"One dot per rural US county in 2017 with the mental-health outcome observed (n = {len(d17):,} counties); the count of FSIS establishments in each size bin is set to zero when "
         "missing. Size bins are the FSIS establishment size codes 1-5 (script0h). Source: FSIS panel, CHR.")


# =============================================================================
# high-coverage-violins (script2a section 6)
# =============================================================================
def section_high_coverage_violins():
    """Violin of each of the 20 best-covered numeric panel variables (at least 90% filled, 20+ distinct values)."""
    pan = panel()
    numeric = numeric_columns()
    rows = [{"variable": c, "fill_pct": float(s.notna().mean() * 100), "std": float(s.std()), "nunique": int(s.nunique(dropna=True))}
            for c, s in numeric.items()]
    summary = pd.DataFrame(rows).sort_values(["fill_pct", "std"], ascending=[False, False])
    high = summary[(summary["fill_pct"] >= 90) & (summary["nunique"] >= 20)]
    if high.empty:
        high = summary[(summary["fill_pct"] >= 60) & (summary["nunique"] >= 8)]
    chosen = high["variable"].head(20).tolist()
    write_table(high.head(20), "high_coverage_violin_variables")
    ncols = 4
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = figstyle.new_grid(nrows, ncols, width="double", height_mm=min(41.0 * nrows + 14, figstyle.MAX_HEIGHT_MM))
    flat = axes.ravel()
    for ax, var in zip(flat, chosen):
        s = numeric[var].dropna()
        sns.violinplot(y=s, inner="quartile", cut=0, color=P["sky"], linewidth=0.5, ax=ax)
        ax.set_title(var, fontsize=5, pad=2)
        ax.set_ylabel("Value (variable's own unit)" if ax in axes[:, 0] else "")
        ax.set_xticks([])
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    hide_unused(flat, len(chosen))
    letters(flat)
    save(fig, "high_coverage_variable_violins",
         f"Distribution over all {len(pan):,} rural US county-years of the 20 numeric panel variables with at least 90% non-missing values and 20 or more distinct values "
         "(ranked by fill rate, then standard deviation). Violin = density cut at the observed range; inner lines = Q1, median, Q3. Panel titles are the panel column names; "
         "units are those of the source variable (see the variable register).")


# =============================================================================
# cafo-class-size (script2a section 7)
# =============================================================================
def section_cafo_class_size():
    """Operations by commodity and size class from the compact CAFO file: distribution per county-year and census-year shares."""
    cafo, long = cafo_compact()
    present = sorted(long["commodity_desc"].dropna().unique())
    census = long[long["year"].isin(CENSUS_YEARS)].copy()

    # violins: operations per county-year by class and size for each commodity, then all commodities side by side
    n_panels = len(present) + 1
    fig, axes = figstyle.new_grid(1, n_panels, width="double", height_mm=70)
    for ax, commodity in zip(axes.ravel(), present):
        sub = long[long["commodity_desc"] == commodity]
        top = sub.groupby("class_desc")["ops_count"].sum().sort_values(ascending=False).head(12).index.tolist()
        sub = sub[sub["class_desc"].isin(top)]
        sns.violinplot(data=sub, x="class_desc", y="ops_count", hue="size_bucket", hue_order=["small", "medium", "large"],
                       palette=[SIZE_COLOUR[s] for s in SIZES], cut=0, inner="quartile", linewidth=0.5, ax=ax)
        ax.set_title(commodity.title(), fontsize=6.5, pad=2)
        ax.set_xlabel("Class")
        ax.set_ylabel("Operations per county-year (count)")
        ax.legend(title="Size class", fontsize=5, title_fontsize=5.5)
    ax = axes.ravel()[-1]
    sns.violinplot(data=long, x="commodity_desc", y="ops_count", hue="size_bucket", hue_order=["small", "medium", "large"],
                   palette=[SIZE_COLOUR[s] for s in SIZES], cut=0, inner="quartile", linewidth=0.5, ax=ax)
    ax.set_title("All commodities", fontsize=6.5, pad=2)
    ax.set_xlabel("Commodity")
    ax.set_ylabel("Operations per county-year (count)")
    ax.legend(title="Size class", fontsize=5, title_fontsize=5.5)
    letters(axes)
    save(fig, "cafo_operations_violin_by_commodity_and_size",
         "Number of operations per rural US county-year by commodity, class and size class (compact CAFO file, panel county-years). Violin = density cut at the observed range; "
         f"inner lines = Q1, median, Q3. Commodities present after the class filter: {', '.join(present)} (see the script2a note in the report).")

    # shares of operations by census year
    size_commodity = census.groupby(["year", "size_bucket", "commodity_desc"], as_index=False)["ops_count"].sum()
    size_commodity["pct_within_size_year"] = size_commodity["ops_count"] / size_commodity.groupby(["year", "size_bucket"])["ops_count"].transform("sum") * 100
    write_table(size_commodity, "size_commodity_share")
    commodity_size = census.groupby(["year", "commodity_desc", "size_bucket"], as_index=False)["ops_count"].sum()
    commodity_size["pct_within_commodity_year"] = commodity_size["ops_count"] / commodity_size.groupby(["year", "commodity_desc"])["ops_count"].transform("sum") * 100
    write_table(commodity_size, "commodity_size_share")
    x = np.arange(len(CENSUS_YEARS))

    def stacked(ax, pivot, order, colours, title):
        bottom = np.zeros(len(CENSUS_YEARS))
        for key in order:
            vals = pivot[key].fillna(0).to_numpy()
            ax.bar(x, vals, 0.6, bottom=bottom, color=colours[key], label=key.title(), edgecolor="white", linewidth=0.3)
            bottom += vals
        ax.set_xticks(x, [str(y) for y in CENSUS_YEARS])
        ax.set_xlabel("Census of Agriculture year")
        ax.set_ylabel("Share of operations (%)")
        ax.set_ylim(0, 120)
        ax.set_yticks(range(0, 101, 20))
        ax.set_title(title, fontsize=6.5, pad=2)
        ax.legend(loc="upper center", fontsize=5, ncol=3)

    fig, axes = figstyle.new_grid(1, 3, width="double", height_mm=65)
    for ax, size in zip(axes.ravel(), ["small", "medium", "large"]):
        pivot = size_commodity[size_commodity["size_bucket"] == size].pivot_table(index="year", columns="commodity_desc",
                                                                                  values="pct_within_size_year", aggfunc="sum").reindex(CENSUS_YEARS)
        stacked(ax, pivot, present, COMMODITY_COLOUR, f"Within {size} operations")
    letters(axes)
    save(fig, "share_commodity_within_size_class",
         "Commodity shares of operations within each size class, rural US counties, census years (compact CAFO file, panel county-years); years without data show no bar.")

    fig, axes = figstyle.new_grid(1, len(present), width="double", height_mm=65)
    for ax, commodity in zip(axes.ravel(), present):
        pivot = commodity_size[commodity_size["commodity_desc"] == commodity].pivot_table(index="year", columns="size_bucket",
                                                                                        values="pct_within_commodity_year", aggfunc="sum").reindex(CENSUS_YEARS)
        stacked(ax, pivot, ["small", "medium", "large"], {s.lower(): SIZE_COLOUR[s] for s in SIZES}, f"Within {commodity} operations")
    letters(axes)
    save(fig, "share_size_class_within_commodity",
         "Size-class shares of operations within each commodity, rural US counties, census years (compact CAFO file, panel county-years); years without data show no bar.")


# =============================================================================
# cafo-concentration (script2a section 8, plots)
# =============================================================================
def gini_from_values(values):
    x = np.array(values, dtype=float)
    x = x[np.isfinite(x) & (x >= 0)]
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    cumx = np.cumsum(x)
    return (x.size + 1 - 2 * np.sum(cumx) / cumx[-1]) / x.size


def section_cafo_concentration():
    """Concentration of operations across counties by commodity and year; entry and exit of large operations."""
    cafo, long = cafo_compact()
    by_county = long.groupby(["year", "fips", "commodity_desc"], as_index=False)["ops_count"].sum()
    rows = []
    for (yr, cmd), g in by_county.groupby(["year", "commodity_desc"]):
        x = g["ops_count"].fillna(0).astype(float).to_numpy()
        total = float(x.sum())
        rows.append({"year": int(yr), "commodity_desc": cmd, "total_ops": total, "n_counties_positive": int((x > 0).sum()),
                     "hhi": float(np.sum((x / total) ** 2)) if total > 0 else np.nan, "gini": gini_from_values(x),
                     "top10_county_share_pct": float(np.sort(x)[-10:].sum() / total * 100) if total > 0 else np.nan})
    conc = pd.DataFrame(rows).sort_values(["commodity_desc", "year"])
    write_table(conc, "cafo_concentration_by_year_commodity")

    overall = cafo.groupby(["fips", "year"], as_index=False)[["small", "medium", "large"]].sum().sort_values(["fips", "year"])
    overall["lag_large"] = overall.groupby("fips")["large"].shift(1).fillna(0)
    overall["entry_large"] = ((overall["lag_large"] <= 0) & (overall["large"].fillna(0) > 0)).astype(int)
    overall["exit_large"] = ((overall["lag_large"] > 0) & (overall["large"].fillna(0) <= 0)).astype(int)
    ee = overall.groupby("year", as_index=False).agg(entries_large=("entry_large", "sum"), exits_large=("exit_large", "sum"), counties=("fips", "nunique"))
    ee["entry_rate_pct"] = ee["entries_large"] / ee["counties"] * 100
    ee["exit_rate_pct"] = ee["exits_large"] / ee["counties"] * 100
    write_table(ee, "cafo_entry_exit_large_by_year")

    fig, axes = figstyle.new_grid(2, 2, width="double", height_mm=112)
    flat = axes.ravel()
    metrics = [("hhi", "Herfindahl index, HHI (0-1)"), ("gini", "Gini coefficient (0-1)"),
               ("top10_county_share_pct", "Top-10 county share (%)")]
    commodities = sorted(conc["commodity_desc"].unique())
    for ax, (metric, ylabel) in zip(flat, metrics):
        for k, cmd in enumerate(commodities):
            d = conc[conc["commodity_desc"] == cmd].dropna(subset=[metric])
            ax.plot(d["year"], d[metric], marker="osd^"[k % 4], ms=3, lw=0.8, color=COMMODITY_COLOUR.get(cmd, P["black"]), label=cmd.title())
        ax.set_xlabel("Census of Agriculture year")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(conc["year"].unique()))
        ax.legend(fontsize=5)
    ax = flat[3]
    ax.plot(ee["year"], ee["entry_rate_pct"], marker="o", ms=3, lw=0.8, color=P["black"], label="Entry rate")
    ax.plot(ee["year"], ee["exit_rate_pct"], marker="s", ms=3, lw=0.8, ls="--", color=P["black"], label="Exit rate")
    ax.set_xlabel("Census of Agriculture year")
    ax.set_ylabel("Counties with entry / exit (%)")
    ax.set_xticks(sorted(ee["year"].unique()))
    ax.legend(fontsize=5)
    letters(flat)
    save(fig, "cafo_concentration_and_entry_exit",
         "a-c: concentration of operations across rural US counties by commodity (compact CAFO file, panel county-years). d: share of counties in which large operations "
         "appear (entry) or disappear (exit) relative to the county's previous observed census year; the first census year counts every county with large operations as an entry.")


# =============================================================================
# ridgelines (script2a section 9)
# =============================================================================
RIDGE_TARGETS = [("aggravated_assault_crime", "Aggravated assault\n(offences, count, UCR)"),
                 ("violent_crime", "Violent crime rate\n(per 100,000, CHR)"),
                 ("poor_mental_health_days", "Poor mental health days\n(days per month, CHR)"),
                 ("frequent_mental_distress_per100k", "Frequent mental distress\n(per 100,000, CHR)")]


def section_ridgelines():
    """One density curve per year (years with at least 40 observations), stacked and overlapping; shared density scale within a panel."""
    pan = panel()
    fig, axes = figstyle.new_grid(1, len(RIDGE_TARGETS), width="double", height_mm=100)
    rows, clipped = [], []
    for ax, (col, label) in zip(axes.ravel(), RIDGE_TARGETS):
        d = pan[["year", col]].dropna()
        counts = d["year"].value_counts()
        years = sorted(int(y) for y in counts.index[counts >= 40])
        d = d[d["year"].isin(years)]
        lo, hi = d[col].quantile([0.005, 0.995])
        grid = np.linspace(lo, hi, 300)
        dens = {y: stats.gaussian_kde(d.loc[d["year"] == y, col].to_numpy())(grid) for y in years}
        top = max(v.max() for v in dens.values())
        colours = _viridis(np.linspace(0.05, 0.95, len(years)))
        for k, y in enumerate(years):
            base = k * 1.0
            h = dens[y] / top * 2.5                                  # ridges overlap by 2.5 rows at the highest density
            ax.fill_between(grid, base, base + h, color=colours[k], alpha=0.9, lw=0, zorder=len(years) - k)
            ax.plot(grid, base + h, color="white", lw=0.3, zorder=len(years) - k)
            g = d.loc[d["year"] == y, col]
            rows.append({"variable": col, "year": y, "n": len(g), "mean": g.mean(), "p25": g.quantile(0.25), "median": g.median(), "p75": g.quantile(0.75)})
        ax.set_yticks([k + 0.3 for k in range(len(years))][::3], [str(y) for y in years][::3])
        ax.set_xlabel(label)
        ax.set_ylabel("Year" if ax is axes.ravel()[0] else "")
        ax.set_xlim(lo, hi)
        clipped.append(f"{col}: {int((pan[col] > hi).sum() + (pan[col] < lo).sum()):,} values outside the plotted range")
    letters(axes)
    save(fig, "ridgelines_outcomes_by_year",
         "Kernel density (Gaussian, Scott bandwidth) of each outcome over rural US counties, one curve per year with at least 40 observations; common density scale within a panel; "
         "x-axis limited to the 0.5th-99.5th percentile (" + "; ".join(clipped) + "). Source: analysis panel.")
    write_table(pd.DataFrame(rows), "ridgeline_year_summary")


# =============================================================================
# trend-sweep and binned-by-size (script2a section 9)
# =============================================================================
TREND_SKIP = {"year", "non_large_metro", "cafo_total_ops_all_animals", "cafo_total_ops_chickens", "any_large_cafo", "any_medium_or_large_cafo",
              "large_cafo", "medium_cafo", "small_cafo"}


def _twin(ax, x, y_out, y_ops, left_label, right_label):
    """Outcome mean (left axis, blue circles, solid) and operations count (right axis, vermillion triangles, dashed) against year."""
    ax.plot(x, y_out, marker="o", ms=2.2, lw=0.8, color=P["blue"])
    ax.tick_params(axis="y", colors=P["blue"], labelsize=5)
    ax.spines["left"].set_color(P["blue"])
    ax2 = ax.twinx()
    ax2.plot(x, y_ops, marker="^", ms=2.2, lw=0.8, ls="--", color=P["vermillion"])
    ax2.tick_params(axis="y", colors=P["vermillion"], labelsize=5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(P["vermillion"])
    ax2.spines["top"].set_visible(False)
    if left_label:
        ax.set_ylabel(left_label, fontsize=5.5)
    if right_label:
        ax2.set_ylabel(right_label, fontsize=5.5)


def section_trend_sweep():
    """For each numeric panel variable: yearly mean over the county-years that have a CAFO row, against the yearly sum of small, medium and large
    operations, by commodity (compact CAFO file; beef and dairy survive its class filter, see cafo_compact)."""
    pan, numeric, cc_all = panel(), numeric_columns(), cafo_counts()
    variables = [c for c in numeric if c not in TREND_SKIP and not c.startswith("n_rows")]
    if MAX_TREND_VARS:
        variables = variables[:MAX_TREND_VARS]
    write_table(pd.DataFrame({"variable": variables}), "trend_variable_list")
    print(f"  {len(variables)} variables to plot")
    commodities = sorted(cc_all["commodity_desc"].unique())
    rows, plotted = [], 0
    for var in variables:
        s = numeric[var]
        if s.notna().sum() < 30:
            continue
        keep = ["fips", "year", var] if var != POP else ["fips", "year", var]
        base = pd.DataFrame({"fips": pan["fips"], "year": pan["year"], var: s})
        panels = []
        for cmd in commodities:
            cc = cc_all[cc_all["commodity_desc"] == cmd][["fips", "year", "small", "medium", "large"]]
            sub = cc.merge(base, on=["fips", "year"], how="left")
            if sub[var].notna().sum() < 30:
                continue
            panels.append((cmd, sub))
        if not panels:
            continue
        fig, axes = figstyle.new_grid(len(panels), 3, width="double", height_mm=48.0 * len(panels) + 16)
        for ri, (cmd, sub) in enumerate(panels):
            for ci, size in enumerate(["small", "medium", "large"]):
                agg = sub.groupby("year", as_index=False).agg(outcome_mean=(var, "mean"), cafo_count=(size, "sum"))
                ax = axes[ri, ci]
                _twin(ax, agg["year"].astype(int), agg["outcome_mean"], agg["cafo_count"],
                      f"{var} (mean)" if ci == 0 else "", "Operations (count)" if ci == 2 else "")
                ax.set_title(f"{cmd.title()}, {size} operations", fontsize=6, pad=2)
                if ri == len(panels) - 1:
                    ax.set_xlabel("Year", fontsize=5.5)
                for _, r in agg.iterrows():
                    rows.append({"variable": var, "commodity": cmd, "size_class": size, "year": int(r["year"]),
                                 "outcome_mean": r["outcome_mean"], "cafo_count": r["cafo_count"]})
        letters(axes)
        fig.legend([plt.Line2D([0], [0], color=P["blue"], marker="o", ms=3, lw=0.8), plt.Line2D([0], [0], color=P["vermillion"], marker="^", ms=3, lw=0.8, ls="--")],
                   ["Yearly mean of the variable (left axis, own unit, solid)", "Yearly sum of operations of that size class (right axis, dashed)"],
                   loc="outside upper center", ncol=2, fontsize=5.5)
        save(fig, f"trend_{safe_stem(var)}_vs_cafo_counts",
             f"{var}: yearly mean over rural US county-years that have a CAFO row of that commodity (compact CAFO file, panel county-years), against the yearly sum of operations by size class. "
             "Source: analysis panel.")
        plotted += 1
        if plotted % 20 == 0:
            print(f"  trend progress: {plotted}/{len(variables)}")
    write_table(pd.DataFrame(rows), "trend_sweep_values")


def section_binned_by_size():
    """Equal-count bins (20) of the county-year operation count of each commodity and size class (counties with at least one) against four outcomes."""
    pan, numeric, cc_all = panel(), numeric_columns(), cafo_counts()
    key = [("aggravated_assault_crime", "Aggravated assault (offences, count, UCR)"), ("violent_crime", "Violent crime rate (per 100,000, CHR)"),
           ("some_college_per100k", "Some college (per 100,000)"), (POP, "Population (residents)")]
    commodities = sorted(cc_all["commodity_desc"].unique())
    rows = []
    for var, label in key:
        if var not in numeric:
            continue
        v = pd.DataFrame({"fips": pan["fips"], "year": pan["year"], var: numeric[var]})
        fig, axes = figstyle.new_grid(len(commodities), 3, width="double", height_mm=52.0 * len(commodities) + 16)
        for ri, cmd in enumerate(commodities):
            cc = cc_all[cc_all["commodity_desc"] == cmd][["fips", "year", "small", "medium", "large"]]
            m = cc.merge(v, on=["fips", "year"], how="left")
            for ci, size in enumerate(["small", "medium", "large"]):
                ax = axes[ri, ci]
                ax.set_title(f"{cmd.title()}, {size} operations", fontsize=6, pad=2)
                ax.set_xlabel("Operations per county-year (bin mean, count)")
                ax.set_ylabel(label if ci == 0 else "")
                d = m[[size, var]].dropna()
                d = d[d[size] > 0]
                if len(d) < 50 or d[size].nunique() < 10:
                    ax.text(0.5, 0.5, "Fewer than 50 county-years\nor 10 distinct counts", ha="center", va="center", transform=ax.transAxes, color="#555555")
                    continue
                d = d.assign(bin=pd.qcut(d[size], q=20, duplicates="drop"))
                g = d.groupby("bin", observed=True).agg(x=(size, "mean"), y=(var, "mean"), n=("bin", "size")).reset_index(drop=True)
                ax.plot(g["x"], g["y"], color="#555555", lw=0.6, zorder=1)
                ax.scatter(g["x"], g["y"], s=3 + 25 * g["n"] / g["n"].max(), color=EXPOSURE_COLOUR["total"], zorder=2, lw=0)
                for i, r in g.iterrows():
                    rows.append({"variable": var, "commodity": cmd, "size_class": size, "bin": i + 1, "x_mean": r["x"], "y_mean": r["y"], "n": int(r["n"])})
        letters(axes)
        save(fig, f"binned_{var}_by_commodity_and_size",
             f"{label}: means of the outcome in 20 equal-count bins of the number of operations per county-year (counties with at least one operation; compact CAFO file, "
             "panel county-years); dot area proportional to the number of county-years in the bin; line joins the bin means. Source: analysis panel.")
    write_table(pd.DataFrame(rows), "binned_by_size_bins")


# =============================================================================
# state-trends (script2a section 10)
# =============================================================================
def section_state_trends():
    """For the states with the most large operations (plus Arkansas): yearly state mean of five variables against yearly operation counts."""
    pan, numeric, cc_all = panel(), numeric_columns(), cafo_counts()
    cc = cc_all.assign(state_fips=cc_all["fips"].astype(str).str[:2])
    state_large = cc.groupby("state_fips", as_index=False)["large"].sum().rename(columns={"large": "large_ops_total"})
    state_large = state_large[state_large["large_ops_total"] > 0].copy()
    state_large["state_name"] = state_large["state_fips"].map(STATE_NAME).fillna(state_large["state_fips"])
    state_large = state_large.sort_values("large_ops_total", ascending=False)
    write_table(state_large, "states_with_large_cafo")
    chosen = state_large["state_fips"].head(STATE_PLOT_LIMIT).tolist()
    if "05" in state_large["state_fips"].values and "05" not in chosen:
        chosen.append("05")
    outcomes = [(c, l) for c, l in (("aggravated_assault_crime", "Aggravated assault (offences, UCR)"), ("violent_crime", "Violent crime rate (per 100,000)"),
                                    ("poor_mental_health_days", "Poor mental health days (per month)"), ("some_college_per100k", "Some college (per 100,000)"),
                                    (POP, "Population (residents)")) if c in numeric]
    frame = pd.DataFrame({"fips": pan["fips"], "year": pan["year"], **{c: numeric[c] for c, _ in outcomes}})
    frame["state_fips"] = frame["fips"].astype(str).str[:2]
    rows = []
    for st in chosen:
        name = STATE_NAME.get(st, st)
        for cmd in sorted(cc["commodity_desc"].unique()):
            ca = cc[(cc["state_fips"] == st) & (cc["commodity_desc"] == cmd)].groupby("year", as_index=False)[["small", "medium", "large"]].sum()
            if ca.empty:
                continue
            usable = []
            for var, label in outcomes:
                y = frame[frame["state_fips"] == st].groupby("year", as_index=False).agg(outcome_mean=(var, "mean"))
                z = ca.merge(y, on="year", how="inner").dropna(subset=["outcome_mean"])
                if not z.empty:
                    usable.append((var, label, z))
            if not usable:
                continue
            fig, axes = figstyle.new_grid(len(usable), 3, width="double", height_mm=min(40.0 * len(usable) + 16, figstyle.MAX_HEIGHT_MM))
            for ri, (var, label, z) in enumerate(usable):
                for ci, size in enumerate(["small", "medium", "large"]):
                    ax = axes[ri, ci]
                    _twin(ax, z["year"].astype(int), z["outcome_mean"], z[size], label if ci == 0 else "", "Operations (count)" if ci == 2 else "")
                    ax.set_title(f"{size.title()} operations", fontsize=6, pad=2)
                    if ri == len(usable) - 1:
                        ax.set_xlabel("Year", fontsize=5.5)
                    for _, r in z.iterrows():
                        rows.append({"state_fips": st, "state": name, "commodity": cmd, "variable": var, "size_class": size, "year": int(r["year"]),
                                     "outcome_mean": r["outcome_mean"], "operations": r[size]})
            letters(axes)
            fig.legend([plt.Line2D([0], [0], color=P["blue"], marker="o", ms=3, lw=0.8), plt.Line2D([0], [0], color=P["vermillion"], marker="^", ms=3, lw=0.8, ls="--")],
                       ["State mean of the variable (left axis, own unit, solid)", "Operations of that size class in the state (right axis, dashed)"],
                       loc="outside upper center", ncol=2, fontsize=5.5)
            save(fig, f"state_{safe_stem(name)}_{cmd}_cafo_vs_outcomes",
                 f"{name}, {cmd}: yearly mean over the state's rural counties against yearly operations by size class (compact CAFO file, panel county-years); "
                 f"one of the {STATE_PLOT_LIMIT} states with the most large operations (Arkansas added). Source: analysis panel.")
    write_table(pd.DataFrame(rows), "state_trends_values")


# =============================================================================
# state-dots and chicken-facets (script2a section 12)
# =============================================================================
def section_state_dots():
    """2017 county dots by state: poor mental health days against total CAFO operations (counties with at least one operation)."""
    pan = panel()
    mh, tot = "poor_mental_health_days", CAFO_TOTAL
    d = pan[(pan["year"] == 2017) & pan[tot].notna() & (pan[tot] > 0) & pan[mh].notna()].copy()
    write_table(d[["fips", "state_fips", "year", tot, mh]], "state_county_dots_2017")
    states = sorted(d["state_fips"].dropna().unique())
    n_pages = int(np.ceil(len(states) / 20))
    for page, chunk in enumerate(np.array_split(states, n_pages), 1):
        nrows = int(np.ceil(len(chunk) / 5))
        fig, axes = figstyle.new_grid(nrows, 5, width="double", height_mm=min(34.0 * nrows + 14, figstyle.MAX_HEIGHT_MM))
        flat = axes.ravel()
        for ax, st in zip(flat, chunk):
            g = d[d["state_fips"] == st]
            ax.scatter(g[tot], g[mh], s=4, color=EXPOSURE_COLOUR["total"], alpha=0.65, lw=0)
            ax.set_title(f"{STATE_ABBR.get(st, st)} (n = {len(g)})", fontsize=5.5, pad=2)
            ax.tick_params(labelsize=5)
        for i, ax in enumerate(flat[:len(chunk)]):
            if i % 5 == 0:
                ax.set_ylabel("Poor mental health\ndays per month", fontsize=5.5)
            if i + 5 >= len(chunk):
                ax.set_xlabel("CAFO operations (count)", fontsize=5.5)
        hide_unused(flat, len(chunk))
        save(fig, f"state_county_dots_2017_part{page}of{n_pages}",
             f"One dot per rural US county in 2017 with at least one CAFO operation (all animals) and the mental-health outcome observed (page {page} of {n_pages}); "
             "n = counties in the state. Source: analysis panel (CHR, USDA Census of Agriculture).")


def section_chicken_facets():
    """Chicken CAFO operations against poor mental health days, one panel per year 2010-2020, colour and marker by the largest size class present."""
    pan = panel()
    mh = "poor_mental_health_days"
    f = pan[pan["year"].between(2010, 2020) & pan["cafo_total_ops_chickens"].notna() & (pan["cafo_total_ops_chickens"] > 0) & pan[mh].notna()].copy()
    f["chicken_size_class"] = pd.NA
    f.loc[f["cafo_chickens_large"] > 0, "chicken_size_class"] = "large_present"
    f.loc[(f["cafo_chickens_large"] <= 0) & (f["cafo_chickens_medium"] > 0), "chicken_size_class"] = "medium_present"
    f.loc[(f["cafo_chickens_large"] <= 0) & (f["cafo_chickens_medium"] <= 0) & (f["cafo_chickens_small"] > 0), "chicken_size_class"] = "small_only"
    f = f[f["chicken_size_class"].notna()].copy()
    if f.empty:
        print("  no chicken county-years; skipped")
        return
    write_table(f[["fips", "state_fips", "year", "cafo_total_ops_chickens", mh, "cafo_chickens_small", "cafo_chickens_medium", "cafo_chickens_large",
                   "chicken_size_class"]], "chickens_scatter_data_2010_2020")
    style = {"small_only": (P["sky"], "o", "Small only"), "medium_present": (P["orange"], "s", "Medium present, no large"), "large_present": (P["vermillion"], "^", "Large present")}
    years = sorted(f["year"].astype(int).unique())
    ncols = 4
    fig, axes = figstyle.new_grid(int(np.ceil(len(years) / ncols)), ncols, width="double", height_mm=120)
    flat = axes.ravel()
    for ax, yr in zip(flat, years):
        g = f[f["year"] == yr]
        for cls, (colour, marker, _) in style.items():
            h = g[g["chicken_size_class"] == cls]
            ax.scatter(h["cafo_total_ops_chickens"], h[mh], s=5, color=colour, marker=marker, alpha=0.7, lw=0, rasterized=True)
        ax.set_title(str(yr), fontsize=6.5, pad=2)
        ax.set_xlabel("Chicken CAFO operations (count)")
        ax.set_ylabel("Poor mental health days\n(days per month)" if ax in axes[:, 0] else "")
    hide_unused(flat, len(years))
    letters(flat)
    flat[-1].set_visible(True)
    flat[-1].axis("off")
    flat[-1].legend([plt.Line2D([0], [0], color=c, marker=m, ls="", ms=3.5) for c, m, _ in style.values()], [lab for _, _, lab in style.values()],
                    loc="center", fontsize=6, title="Size classes present", title_fontsize=6)
    save(fig, "chickens_cafo_vs_poor_mental_health_by_year_2010_2020",
         "One dot per rural US county-year with at least one chicken CAFO operation and the mental-health outcome observed, 2010-2020; colour and marker = largest chicken size class "
         "present in the county. Source: analysis panel (CHR, USDA Census of Agriculture).")


# =============================================================================
# species-overlap (script4s)
# =============================================================================
def section_species_overlap():
    """Counties by number of species (dairy, beef, hogs, chickens) with at least one large operation; a missing count is zero from 2002 (D-34)."""
    species = ["dairy", "beef", "hogs", "chickens"]
    pan = panel()
    assert (pan["rural"] == 1).all(), "panel is expected to be pre-filtered to rural counties"
    p = pan[pan["year"] >= min(CENSUS_YEARS)]
    present = pd.concat([pd.to_numeric(p[f"cafo_{s}_large"], errors="coerce").rename(s) for s in species], axis=1).fillna(0).gt(0).astype(int)
    k = present.sum(axis=1)
    by_year = pd.crosstab(p["year"], k).reindex(columns=range(len(species) + 1), fill_value=0)
    by_year.columns = [f"n_counties_{c}_species" for c in by_year.columns]
    by_year["n_county_years"] = by_year.sum(axis=1)
    by_year["n_multi_species"] = by_year[[f"n_counties_{c}_species" for c in range(2, len(species) + 1)]].sum(axis=1)
    by_year["share_multi_species"] = by_year["n_multi_species"] / by_year["n_county_years"]
    by_year["census_year"] = by_year.index.isin(CENSUS_YEARS)
    by_year = by_year.reset_index().rename(columns={"index": "year"})
    write_table(by_year, "species_count_by_year")
    cy = by_year[by_year["census_year"]].set_index("year")
    colours = [mcolors.to_hex(_viridis(v)) for v in np.linspace(0.95, 0.05, len(species) + 1)]
    fig, ax = figstyle.new_figure("single", height_mm=68)
    bottom = np.zeros(len(cy))
    for i in range(len(species) + 1):
        vals = cy[f"n_counties_{i}_species"].to_numpy()
        ax.bar(cy.index.astype(str), vals, bottom=bottom, color=colours[i], edgecolor="white", linewidth=0.3, label=f"{i} species")
        bottom += vals
    ax.set_xlabel("Census of Agriculture year")
    ax.set_ylabel("Rural counties (number)")
    ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "species_count_by_census_year",
         "Rural US counties by number of species (dairy, beef, hogs, chickens) with at least one large operation. A missing count is zero from 2002 (D-34). Source: analysis panel.")


# =============================================================================
# control-coverage (new; D-55)
# =============================================================================
def coverage_table(pan):
    """Share of non-missing values of each candidate control over COVERAGE_YEARS, one row per control, in figure order
    (type clusters in designs.CONTROL_TYPE_COLOUR order, then fill rate descending). Raises if a candidate is not in the panel."""
    missing = [c for c in CONTROL_TYPE if c not in pan.columns]
    if missing:
        raise KeyError(f"candidate controls not in the panel (reported, not dropped): {missing}")
    selected = set(COVARIATE_ORDER[:N_SELECTED])
    assert selected <= set(CONTROL_TYPE), f"selected controls outside the 24 candidates: {sorted(selected - set(CONTROL_TYPE))}"
    sub = pan[pan["year"].between(*COVERAGE_YEARS)]
    assert not sub.duplicated(["fips", "year"]).any(), "panel must have one row per county-year"
    rows = [{"variable": c, "type": t, "selected": c in selected, "n_obs": len(sub), "n_filled": int(sub[c].notna().sum())}
            for c, t in CONTROL_TYPE.items()]
    out = pd.DataFrame(rows)
    out["fill_pct"] = out["n_filled"] / out["n_obs"] * 100
    out["_cluster"] = out["type"].map({t: i for i, t in enumerate(CONTROL_TYPE_COLOUR)})
    out = out.sort_values(["_cluster", "fill_pct", "variable"], ascending=[True, False, True]).drop(columns="_cluster").reset_index(drop=True)
    # Second route (all columns at once); the two must agree to rounding error.
    again = sub[out["variable"].tolist()].notna().mean() * 100
    assert np.allclose(again.to_numpy(), out["fill_pct"].to_numpy(), rtol=0, atol=1e-9), "fill rates differ between the two routes"
    return out, sub


def control_label(name):
    """Readable axis label for a panel column name."""
    return name.replace("_per100k", "").replace("%_", "% ").replace("_", " ")


def section_control_coverage():
    """Horizontal bars of the percent of filled county-years, COVERAGE_YEARS, for the 24 candidate controls (designs.CONTROL_TYPE),
    clustered by type and coloured by type; the selected set is outlined and the fill cutoff of script0c is dotted."""
    pan = panel()
    assert (pan["rural"] == 1).all(), "panel is expected to be pre-filtered to rural counties"
    cov, sub = coverage_table(pan)
    lo, hi = COVERAGE_YEARS
    write_table(cov, "control_coverage")
    txt = cov.assign(selected=np.where(cov["selected"], "yes", "no")).rename(columns={"selected": "selected (D-16)"})
    sel = cov[cov["selected"]]
    others = cov[~cov["selected"]]
    note_txt = (f"Percent of non-missing values per variable over rural county-years {lo}-{hi} ({len(sub):,} county-years, {sub['fips'].nunique():,} counties). "
                f"Rows follow the figure: type clusters, then fill rate descending. Cutoff line: {FILL_THRESHOLD_PCT}% (the HIGH FILL cutoff of "
                f"script0c-health-controls.py). Selected = the {N_SELECTED} controls of the pre-committed set (D-16).")
    path = figstyle.save_text_table(txt, OUT_TABLES, "control_coverage", f"Fill rate of the {len(cov)} candidate controls, {lo}-{hi}", note_txt)
    print("  table", path.name)
    print(f"  selected {N_SELECTED}: min {sel['fill_pct'].min():.3f}% max {sel['fill_pct'].max():.3f}%; other {len(others)}: "
          f"max {others['fill_pct'].max():.3f}%, lowest {others.loc[others['fill_pct'].idxmin(), 'variable']} {others['fill_pct'].min():.3f}%")

    # y positions from the top: one gap row between type clusters
    ypos, y, prev = [], 0.0, None
    for t in cov["type"]:
        y += 0.8 if prev is not None and t != prev else 0.0
        ypos.append(y)
        y += 1.0
        prev = t
    cov = cov.assign(y=ypos)
    fig, ax = figstyle.new_figure("double", height_mm=148)
    for _, r in cov.iterrows():
        ax.barh(r["y"], r["fill_pct"], height=0.78, color=P[CONTROL_TYPE_COLOUR[r["type"]]], hatch=TYPE_HATCH[r["type"]],
                edgecolor="#333333", linewidth=0.3, zorder=2)          # same dark hatch on every bar
        if r["selected"]:                                               # outline drawn separately so it does not change the hatch
            ax.barh(r["y"], r["fill_pct"], height=0.78, fill=False, edgecolor="black", linewidth=1.3, zorder=4)
        ax.text(r["fill_pct"] + 0.8, r["y"], f"{r['fill_pct']:.1f}%", va="center", ha="left", fontsize=5, color=NOTE_COLOUR)
    ax.axvline(FILL_THRESHOLD_PCT, color="black", ls=":", lw=0.9, zorder=3)
    ax.set_yticks(cov["y"], [control_label(v) for v in cov["variable"]], fontsize=5.5)
    for tick, sel_flag in zip(ax.get_yticklabels(), cov["selected"]):
        tick.set_fontweight("bold" if sel_flag else "normal")
    ax.set_ylim(cov["y"].max() + 0.7, -0.7)
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel(f"Filled observations, rural county-years {lo}-{hi} (% of {len(sub):,})")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=P[CONTROL_TYPE_COLOUR[t]], hatch=TYPE_HATCH[t], edgecolor="#333333", linewidth=0.3, label=t)
               for t in CONTROL_TYPE_COLOUR]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=1.3, label="selected control set, D-16"),
                plt.Line2D([0], [0], color="black", ls=":", lw=0.9, label=f"fill cutoff of the 0-series script, {FILL_THRESHOLD_PCT}%")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=5.5, title="Control type", title_fontsize=5.5)
    save(fig, "control_coverage",
         f"Percent of non-missing values of the {len(cov)} candidate controls (designs.CONTROL_TYPE), rural US county-years {lo}-{hi} "
         f"({len(sub):,} county-years, {sub['fips'].nunique():,} counties). Bars sorted by fill rate within type; colour and hatch = type; dark outline and bold "
         f"name = the {N_SELECTED} selected controls (D-16); dotted line = HIGH FILL cutoff of script0c-health-controls.py ({FILL_THRESHOLD_PCT}%). "
         "Source: analysis panel.")


# =============================================================================
# trends (script4v)
# =============================================================================
UNIT = {"Poor MH Days": "days per month", "Frequent Mental Distress": "per 100,000", "Deaths of Despair": "per 100,000",
        "Aggravated Assault": "per 100,000", "Assault, all severities": "per 100,000", "Violence index (partial)": "per 100,000",
        "NIBRS curated total": "per 100,000"}
SHORT = {"Poor MH Days": "Poor mental health days", "Frequent Mental Distress": "Frequent mental distress", "Deaths of Despair": "Deaths of despair",
         "Aggravated Assault": "Aggravated assault", "Assault, all severities": "Assault, all severities",
         "Violence index (partial)": "Violence index (partial)", "NIBRS curated total": "NIBRS curated total"}
GROUPS = [("never large dairy", P["blue"], "-"), ("enters", P["vermillion"], "--"), ("expands only", P["green"], "-."),
          ("large dairy throughout", P["orange"], ":")]
COHORT_STYLE = {2007: (P["blue"], "o"), 2012: (P["vermillion"], "s"), 2017: (P["green"], "^"), 2022: (P["orange"], "D")}


def event_paths(df, info, ocol, treated_col, comparison_fips, min_n=10, clip=8):
    """Mean change in the outcome since the year before treatment (g-1), treated vs comparison counties, averaged over cohorts by treated size."""
    wide = df.pivot_table(index="fips", columns="year", values=ocol, aggfunc="first")
    recs = []
    for g in sorted(info[treated_col].dropna().unique()):
        base = g - 1
        if base not in wide.columns:
            continue
        treated = info.index[info[treated_col] == g]
        for e in range(-clip, clip + 1):
            t = g + e
            if t not in wide.columns:
                continue
            dt = (wide[t] - wide[base]).reindex(treated).dropna()
            dc = (wide[t] - wide[base]).reindex(comparison_fips).dropna()
            if len(dt) < min_n or len(dc) < min_n:
                continue
            recs.append({"cohort": int(g), "e": e, "n_t": len(dt), "m_t": dt.mean(), "v_t": dt.var(ddof=1) / len(dt),
                         "m_c": dc.mean(), "v_c": dc.var(ddof=1) / len(dc), "n_c": len(dc)})
    r = pd.DataFrame(recs)
    if r.empty:
        return pd.DataFrame()
    out = []
    for e, g in r.groupby("e"):
        w = g.n_t.to_numpy(float)
        out.append({"event_time": e, "treated": np.average(g.m_t, weights=w), "comparison": np.average(g.m_c, weights=w),
                    "se_t": np.sqrt(np.sum(w ** 2 * g.v_t)) / w.sum(), "se_c": np.sqrt(np.sum(w ** 2 * g.v_c)) / w.sum(),
                    "n_cohorts": g.cohort.nunique(), "n_treated": int(g.n_t.sum())})
    return pd.DataFrame(out)


def _key_panel(ax, handles, labels, text):
    """Eighth cell of a 2 x 4 grid: legend plus a reading note."""
    ax.axis("off")
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=5.5)
    ax.text(0.0, 0.42, textwrap.fill(text, 34), fontsize=5, va="top", transform=ax.transAxes, color=NOTE_COLOUR)


def section_trends():
    """Raw outcome trends by dairy county group, event-time paths of raw changes, and the pre-treatment slope difference (no controls)."""
    import pyfixest as pf
    from designs import county_groups
    from script4_treatment import OUTCOMES as T_OUTCOMES, load_panel

    df = load_panel().sort_values(["fips", "year"]).reset_index(drop=True)
    info = county_groups(df)
    df["group"] = df["fips"].map(info["group"])
    counts = info.group.value_counts()
    print("  counties per group:", counts.to_dict())

    # F1: calendar-year means by group
    rows = []
    fig, axes = figstyle.new_grid(2, 4, width="double", height_mm=112)
    flat = axes.ravel()
    for ax, (okey, ocol) in zip(flat, T_OUTCOMES.items()):
        for yr in (2007, 2012, 2017, 2022):
            ax.axvline(yr, color="#dddddd", lw=0.5, zorder=0)
        for gname, colour, ls in GROUPS:
            g = df[(df.group == gname) & df[ocol].notna()].groupby("year")[ocol].agg(["mean", "std", "count"])
            g = g[g["count"] >= 15]
            se = g["std"] / np.sqrt(g["count"])
            ax.fill_between(g.index, g["mean"] - 1.96 * se, g["mean"] + 1.96 * se, color=colour, alpha=0.15, lw=0)
            ax.plot(g.index, g["mean"], color=colour, ls=ls, lw=0.9)
            for yr, r in g.iterrows():
                rows.append({"outcome": okey, "group": gname, "year": int(yr), "mean": r["mean"], "se": r["std"] / np.sqrt(r["count"]), "n_counties": int(r["count"])})
        ax.set_title(SHORT[okey], fontsize=6.5, pad=2)
        ax.set_xlim(2000, 2023)
        ax.set_xlabel("Year")
        ax.set_ylabel(UNIT[okey])
    handles = [plt.Line2D([0], [0], color=c, ls=ls, lw=0.9) for _, c, ls in GROUPS]
    _key_panel(flat[7], handles, [f"{n} (n = {counts.get(n, 0):,})" for n, _, _ in GROUPS],
               "Each panel has its own vertical scale because the outcomes are in different units. Grey lines: census waves.")
    letters(flat[:7])
    save(fig, "trends_means_by_county_group",
         "Raw county means by calendar year and county group (large-dairy status from the Census of Agriculture waves), no controls; shaded band = ±1.96 standard errors across counties; "
         "years with fewer than 15 counties in a group are not drawn. Source: analysis panel, designs.county_groups.")
    write_table(pd.DataFrame(rows), "trends_means_by_group")

    # F2: event-time paths
    designs = {"first expansion (any positive change in large dairies)": ("add", info.index[info["add"].isna() & info["entry"].isna()]),
               "first entry (0 to 1+ large dairies)": ("entry", info.index[info.group == "never large dairy"])}
    ev_rows = []
    for dname, (tcol, comp) in designs.items():
        fig, axes = figstyle.new_grid(2, 4, width="double", height_mm=112)
        flat = axes.ravel()
        for ax, (okey, ocol) in zip(flat, T_OUTCOMES.items()):
            p = event_paths(df, info, ocol, tcol, comp)
            if p.empty:
                ax.set_title(f"{SHORT[okey]} (no usable cohorts)", fontsize=6.5, pad=2)
                continue
            for colour, ls, m, s in ((P["vermillion"], "--", "treated", "se_t"), (P["blue"], "-", "comparison", "se_c")):
                ax.fill_between(p.event_time, p[m] - 1.96 * p[s], p[m] + 1.96 * p[s], color=colour, alpha=0.15, lw=0)
                ax.plot(p.event_time, p[m], color=colour, ls=ls, lw=0.9)
            ax.axvline(-0.5, color="#555555", lw=0.5, ls=":")
            ax.axhline(0, color="#999999", lw=0.5, zorder=0)
            ax.set_xlim(-8.4, 8.4)
            ax.set_title(SHORT[okey], fontsize=6.5, pad=2)
            ax.set_xlabel("Years from first treated year")
            ax.set_ylabel(f"Change since pre-treatment year\n({UNIT[okey]})")
            for _, r in p.iterrows():
                ev_rows.append({"design": dname, "outcome": okey, **r.to_dict()})
        handles = [plt.Line2D([0], [0], color=P["vermillion"], ls="--", lw=0.9), plt.Line2D([0], [0], color=P["blue"], lw=0.9)]
        _key_panel(flat[7], handles, ["treated counties", "comparison counties"],
                   "Dotted vertical line = treatment date. Fewer cohorts contribute at long leads and lags, so the far ends are noisier.")
        letters(flat[:7])
        tag = "expansion" if "expansion" in dname else "entry"
        save(fig, f"trends_event_time_{tag}",
             f"Design: {dname}. Mean change in the outcome since the year before treatment (g-1), treated vs comparison counties, raw changes without controls or fixed effects, averaged over "
             "cohorts weighted by treated counties; band = ±1.96 naive standard errors across counties. Source: analysis panel, designs.county_groups.")
    write_table(pd.DataFrame(ev_rows), "trends_event_time_paths")

    # F3: pre-treatment slope difference (the one regression of this script: outcome ~ year + treated x year | county, SE clustered by state)
    sd = {o: df[c].std() for o, c in T_OUTCOMES.items()}
    sl = []
    for dname, (tcol, comp) in designs.items():
        for g in sorted(info[tcol].dropna().unique()):
            treated = set(info.index[info[tcol] == g])
            for okey, ocol in T_OUTCOMES.items():
                w = df[df.year.between(g - 5, g - 1) & df.fips.isin(treated | set(comp))][["fips", "state_fips", "year", ocol]].dropna()
                ok = w.groupby("fips").size()
                w = w[w.fips.isin(ok.index[ok >= 3])].copy()
                w["treated"] = w.fips.isin(treated).astype(float)
                w["tx"] = w["treated"] * w["year"]
                n_t, n_c = w[w.treated == 1].fips.nunique(), w[w.treated == 0].fips.nunique()
                if n_t < 10 or n_c < 10:
                    continue
                try:
                    t = pf.feols(f"{ocol} ~ year + tx | fips", data=w, vcov={"CRV1": "state_fips"}).tidy().loc["tx"]
                except Exception as err:
                    print(f"  slope test skipped ({dname}, {g}, {okey}): {err}")
                    continue
                sl.append({"design": dname, "cohort": int(g), "outcome": okey, "slope_diff": float(t["Estimate"]), "se_state": float(t["Std. Error"]),
                           "p_state": float(t["Pr(>|t|)"]), "slope_diff_in_sd": float(t["Estimate"]) / sd[okey], "se_in_sd": float(t["Std. Error"]) / sd[okey],
                           "n_treated": n_t, "n_comparison": n_c, "window": f"{int(g) - 5}-{int(g) - 1}"})
    sl = pd.DataFrame(sl)
    write_table(sl, "trends_pre_window_slopes")
    names = list(T_OUTCOMES)
    fig, axes = figstyle.new_grid(1, 2, width="double", height_mm=78, sharex=True)
    for ax, dname in zip(axes.ravel(), designs):
        ax.axvline(0, color="#555555", lw=0.5)
        x = sl[sl.design == dname]
        for k, (g, (colour, marker)) in enumerate(COHORT_STYLE.items()):
            xs = x[x.cohort == g]
            ypos = [names.index(o) + (k - 1.5) * 0.17 for o in xs.outcome]
            ax.errorbar(xs.slope_diff_in_sd, ypos, xerr=1.96 * xs.se_in_sd, fmt=marker, color=colour, ms=3, mec="white", mew=0.4, elinewidth=0.7,
                        capsize=0, ls="none", label=f"{g} cohort")
        ax.set_yticks(range(len(names)), [SHORT[n] for n in names])
        ax.invert_yaxis()
        ax.set_title(dname, fontsize=6.5, pad=2)
        ax.set_xlabel("Slope difference (SD of the outcome per year)")
    axes.ravel()[0].legend(loc="lower right", fontsize=5.5)
    letters(axes)
    save(fig, "trends_pre_window_slope_difference",
         "Treated minus comparison counties: change in the outcome per year over the 5 years before treatment, in standard deviations of the outcome; bars = ±1.96 standard errors "
         "clustered by state. One OLS slope test per cohort (outcome on year and treated x year, county fixed effect), no controls. Source: analysis panel, designs.county_groups.")


SECTIONS = {
    "binned-scatter": section_binned_scatter,
    "outcome-correlation": section_outcome_correlation,
    "quartile-distributions": section_quartile_distributions,
    "cafo-top-counties": section_cafo_top_counties,
    "cafo-census-composition": section_cafo_census_composition,
    "threshold-presence": section_threshold_presence,
    "fsis-size-bins": section_fsis_size_bins,
    "high-coverage-violins": section_high_coverage_violins,
    "cafo-class-size": section_cafo_class_size,
    "cafo-concentration": section_cafo_concentration,
    "ridgelines": section_ridgelines,
    "trend-sweep": section_trend_sweep,
    "binned-by-size": section_binned_by_size,
    "state-trends": section_state_trends,
    "state-dots": section_state_dots,
    "chicken-facets": section_chicken_facets,
    "species-overlap": section_species_overlap,
    "control-coverage": section_control_coverage,
    "trends": section_trends,
}


def main():
    global MAX_TREND_VARS, STATE_PLOT_LIMIT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="+", choices=list(SECTIONS), help="run only these sections (default: all)")
    ap.add_argument("--max-trend-vars", type=int, help="plot only the first N variables of `trend-sweep` (the full sweep is slow)")
    ap.add_argument("--state-limit", type=int, default=STATE_PLOT_LIMIT, help="states in `state-trends` (default %(default)s; Arkansas is always added)")
    args = ap.parse_args()
    MAX_TREND_VARS, STATE_PLOT_LIMIT = args.max_trend_vars, args.state_limit
    figstyle.apply_style()
    for name in args.only or SECTIONS:
        print(f"\n--- {name} ---")
        SECTIONS[name]()
    print(f"\nfigures -> {OUT_FIGS}\ntables  -> {OUT_TABLES}")


if __name__ == "__main__":
    main()
