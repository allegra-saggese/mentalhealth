#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
designs.py -- the treatment designs, county groups and outcome windows shared by stages 4 (4u-4x) and 1.

load_designs()      panel + the two designs (first expansion; first entry with clean controls)
add_baseline_polynomial   2002 characteristics x time polynomial columns (used by script4a cohorts)
county_groups()     enters / expands only / large dairy throughout / never large dairy
species_treatments()  count and T1, T3, T4, E2 for one species and size scope (script4a species)
cafo_designs()      panel + both designs for one CAFO type and size scope (script4g spillover checks, script4b leads)
fsis_designs()      the same five treatment formats for slaughterhouses (FSIS plants), annual 2017-2023 (script4g)
usable_sample()     estimation sample for one outcome and cohort design (script4d estimators, script4c permutation; script1d sample map)
OUTCOME_WINDOWS    years each outcome is drawn over in the descriptive figures and maps (script1c, script1d); not estimation

Definitions are in METHODS.md section 1.
KNOWN DISCREPANCY (QA #28): 17 counties have no row in any census year. load_designs() leaves them out of the
never-large set (2,103 counties); county_groups() counts them as never-large (2,120). Both are kept as they were so
that no result changes; decide on one definition separately.
"""
import os

import numpy as np
import pandas as pd

from functions import latest_file_glob
from script4_treatment import load_panel, CENSUS_YEARS, OUTCOMES, db_data

BASE_YEAR = 2002
# Years over which each outcome is drawn in the descriptive stage (script1c figures, script1d maps), keyed by the labels
# of script4_treatment.OUTCOMES. Descriptive stage only: the stage-4 estimation samples stay data-defined (script4d estimators
# samples file), and A21 and any use of these windows in estimation are separate decisions. Windows confirmed by the
# owner 2026-10-25: crime stops at 2020 although 2021 exists (the script4d estimators samples run to 2021).
CRIME_WINDOW = (2002, 2020)       # the four crime outcomes; change the end year here (script4d estimators samples run to 2021)
DESPAIR_WINDOW = (2002, 2020)
OUTCOME_WINDOWS = {
    "Poor MH Days": (2010, 2023),
    "Frequent Mental Distress": (2016, 2023),
    "Deaths of Despair": DESPAIR_WINDOW,
    "Aggravated Assault": CRIME_WINDOW,
    "Assault, all severities": CRIME_WINDOW,
    "Violence index (partial)": CRIME_WINDOW,
    "NIBRS curated total": CRIME_WINDOW,
}
assert set(OUTCOME_WINDOWS) == set(OUTCOMES), "OUTCOME_WINDOWS keys must be the script4_treatment.OUTCOMES labels"

# Candidate controls and the type each belongs to (the colour clusters of the control-coverage figure; D-55). The 24 are the register's
# "CONTROL (in use)" and "CONTROL (comprehensive only)" variables; COVARIATE_ORDER[:8] (script4_treatment) is the selected set.
# To move a variable to another cluster, change it here; every figure and text table reads this dict.
CONTROL_TYPE = {
    "adult_obesity_per100k": "Health", "adult_smoking_per100k": "Health", "diabetes_prevalence_per100k": "Health",
    "low_birthweight_per100k": "Health", "physical_inactivity_per100k": "Health", "poor_or_fair_health_per100k": "Health",
    "poor_physical_health_days": "Health", "premature_death": "Health", "preventable_hospital_stays": "Health",
    "teen_births_per100k": "Health", "uninsured_adults_per100k": "Health", "access_to_healthy_foods_per100k": "Health",
    "%_65_and_older": "Demographic", "%_below_18_years_of_age": "Demographic", "%_hispanic": "Demographic",
    "%_black": "Demographic", "%_asian_aian_nhpi": "Demographic",
    "median_household_income": "Economic", "unemployment_per100k": "Economic", "children_in_poverty_per100k": "Economic",
    "driving_alone_to_work_per100k": "Economic",
    "children_in_single-parent_households_per100k": "Social and education", "some_college_per100k": "Social and education",
    "%_not_proficient_in_english_per100k": "Social and education",
}
CONTROL_TYPE_COLOUR = {"Health": "green", "Demographic": "purple", "Economic": "orange", "Social and education": "blue"}   # figstyle.PALETTE keys
FILL_THRESHOLD_PCT = 80   # the HIGH FILL cutoff of script0c-health-controls.py (avg_threshold): a variable is kept when its average fill is at least this percent

BL_VARS = ["log_pop", "median_household_income", "%_hispanic", "%_65_and_older", "unemployment_per100k",
           "cafo_dairy_total", "cafo_dairy_large"]


def add_baseline_polynomial(df, base_year=BASE_YEAR):
    """Add baseline-characteristic x time columns bl{k}_t1, bl{k}_t2 to df (in place) and return their names."""
    _bl = df[df["year"] == base_year].set_index("fips")[BL_VARS]
    _bl = (_bl - _bl.mean()) / _bl.std()                         # standardised so the interactions are well scaled
    _t1 = (df["year"] - base_year) / 10.0
    cols = []
    for k, v in enumerate(BL_VARS):
        b = df["fips"].map(_bl[v])
        df[f"bl{k}_t1"], df[f"bl{k}_t2"] = b * _t1, b * _t1 ** 2
        cols += [f"bl{k}_t1", f"bl{k}_t2"]
    return cols


def load_designs(with_polynomial=False):
    """Returns (df, designs, poly_cols, counts). designs maps a name to dict(data, cohort, treat) with
    t_rel = year - cohort; counts has the sizes of the entry sample for logging."""
    df = load_panel().sort_values(["fips", "year"]).reset_index(drop=True)
    poly_cols = add_baseline_polynomial(df) if with_polynomial else []
    waves = df[df.year.isin(CENSUS_YEARS)].pivot(index="fips", columns="year", values="cafo_dairy_large")
    never_large = set(waves.index[~(waves.fillna(0) > 0).any(axis=1)])
    entry_counties = set(df.loc[df["entry_year"].notna(), "fips"].unique())
    df_entry = df[df["fips"].isin(never_large | entry_counties)].copy()
    designs = {
        "first expansion": dict(data=df, cohort="add_year", treat="tr_e2_add_absorb"),
        "first entry (clean controls)": dict(data=df_entry, cohort="entry_year", treat="tr_e3_entry_absorb"),
    }
    for d in designs.values():
        d["data"] = d["data"].copy()
        d["data"]["t_rel"] = d["data"]["year"] - d["data"][d["cohort"]]
    counts = dict(n_entry=len(entry_counties), n_never_large=len(never_large),
                  n_entry_sample=df_entry.fips.nunique(), n_all=df.fips.nunique())
    return df, designs, poly_cols, counts


def county_groups(df):
    """One row per county with entry/add years, ever_large, and the group label used in the trend figures."""
    waves = df[df.year.isin(CENSUS_YEARS)].pivot(index="fips", columns="year", values="cafo_dairy_large")
    ever_large = (waves.fillna(0) > 0).any(axis=1)
    info = df.groupby("fips").agg(entry=("entry_year", "first"), add=("add_year", "first"))
    info["ever_large"] = ever_large.reindex(info.index).fillna(False)

    def grp(r):
        if pd.notna(r.entry): return "enters"
        if pd.notna(r["add"]): return "expands only"
        if r.ever_large: return "large dairy throughout"
        return "never large dairy"

    info["group"] = info.apply(grp, axis=1)
    return info


def species_treatments(df, species, bins):
    """Count and the five treatment variables for one species x scope. Returns a frame aligned to df."""
    cols = [f"cafo_{species}_{b}" for b in bins]
    raw = df[cols]
    filled = raw.where(df["year"] < min(CENSUS_YEARS), raw.fillna(0))       # D-34: zero from 2002, 2000-01 stay missing
    C = filled.sum(axis=1, min_count=1)
    out = pd.DataFrame(index=df.index)
    out["count"] = C
    out["T1"] = (C > 0).astype(float).where(C.notna())
    out["T3"] = C / df["population"].replace(0, np.nan) * 10_000
    out["T4"] = np.log1p(C)
    # first census wave at which the county's count rises relative to its PREVIOUS OBSERVED wave. Mirrors
    # script4_treatment._wave_events: the lag is the previous wave the county has a row for (a county missing a
    # wave is compared with the one before it), not a calendar-based diff.
    w = pd.DataFrame({"fips": df["fips"], "year": df["year"], "C": C})
    w = w[w.year.isin(CENSUS_YEARS)].dropna(subset=["C"]).sort_values(["fips", "year"])
    w["lag"] = w.groupby("fips")["C"].shift(1)
    rises = w[w["lag"].notna() & (w["C"] - w["lag"] > 0)]
    first = rises.groupby("fips")["year"].min()
    out["add_year"] = df["fips"].map(first)
    D = np.where(out["add_year"].notna() & (df["year"] >= out["add_year"]), 1.0, 0.0)
    out["E2"] = pd.Series(D, index=df.index).where(C.notna())
    return out


def usable_sample(data, ocol, treat, cohort, controls):
    """Estimation sample for one outcome and one cohort design (shared by the triangulation and the permutation test).

    Complete cases on outcome, treatment dummy and controls; a treated county needs a pre-period inside the outcome's years
    (first observed year before its cohort) and a cohort no later than the last year, otherwise it is dropped. Columns are
    renamed y, D, cohort, x1..xk (controls in the order given); fips is an int and state_fips a 0-based integer code.
    """
    d = data[["fips", "year", "state_fips", ocol, treat, cohort] + controls].dropna(subset=[ocol, treat] + controls).copy()
    d = d.rename(columns={ocol: "y", treat: "D", cohort: "cohort", **{c: f"x{i + 1}" for i, c in enumerate(controls)}})
    first = d.groupby("fips").year.min()
    ok = d.cohort.isna() | (d.fips.map(first) < d.cohort)
    d = d[ok & (d.cohort.isna() | (d.cohort <= d.year.max()))]
    d.attrs = {}
    d["fips"], d["state_fips"] = d.fips.astype(int), pd.factorize(d.state_fips)[0]
    return d.reset_index(drop=True)


CAFO_TYPES = ["dairy", "beef", "cattle", "hogs", "chickens", "aggregate"]
TREATMENT_COLUMNS = ["tr_e1_lg_bin", "tr_i4_lg_p10k", "tr_c3_log_lg", "add_year", "tr_e2_add_absorb", "entry_year", "tr_e3_entry_absorb"]


def _two_designs(d, never_present, label_counts):
    """The first-expansion design (every county) and the first-entry design (counties never present plus the entering counties), as load_designs."""
    entry_counties = set(d.loc[d["entry_year"].notna(), "fips"].unique())
    df_entry = d[d["fips"].isin(set(never_present) | entry_counties)].copy()
    designs = {"first expansion": dict(data=d.copy(), cohort="add_year", treat="tr_e2_add_absorb"),
               "first entry (clean controls)": dict(data=df_entry, cohort="entry_year", treat="tr_e3_entry_absorb")}
    for z in designs.values():
        z["data"]["t_rel"] = z["data"]["year"] - z["data"][z["cohort"]]
    print(f"  {label_counts}: {d.loc[d.add_year.notna(), 'fips'].nunique():,} counties with an E2 event, "
          f"{len(entry_counties):,} with an E3 event, {len(set(never_present)):,} never present")
    return designs


def cafo_designs(df, ctype, scope, verbose=True):
    """Panel and designs for one CAFO type and size scope. ctype: dairy, beef, cattle, hogs, chickens, or aggregate (the panel's generic
    small/medium/large_cafo columns: cattle, hogs and chickens only, no double counting, D-51); scope: large, or all (small + medium + large).
    The dairy treatment columns that the stage-4 code reads (TREATMENT_COLUMNS) are overwritten by the chosen type's: T1, T3, T4, add_year and E2 from
    species_treatments, and E3 by the same wave rule (first census wave with a count above zero after a zero). The two designs are rebuilt as
    load_designs does. For dairy / large the rebuilt columns are checked against the panel's own (max abs difference < 1e-12)."""
    bins = ["large"] if scope == "large" else ["small", "medium", "large"]
    d = df.copy()
    if ctype == "aggregate":
        for b in ("small", "medium", "large"):
            d[f"cafo_aggregate_{b}"] = d[f"{b}_cafo"]
    tr = species_treatments(d, ctype, bins)
    C = tr["count"]
    w = pd.DataFrame({"fips": d["fips"], "year": d["year"], "C": C})
    w = w[w.year.isin(CENSUS_YEARS)].dropna(subset=["C"]).sort_values(["fips", "year"])
    w["lag"] = w.groupby("fips")["C"].shift(1)
    entry = w[(w["lag"] == 0) & (w["C"] > 0)].groupby("fips")["year"].min()
    new = pd.DataFrame(index=d.index)
    new["tr_e1_lg_bin"], new["tr_i4_lg_p10k"], new["tr_c3_log_lg"] = tr["T1"], tr["T3"], tr["T4"]
    new["add_year"] = tr["add_year"]
    new["tr_e2_add_absorb"] = tr["E2"]
    new["entry_year"] = d["fips"].map(entry)
    new["tr_e3_entry_absorb"] = (new["entry_year"].notna() & (d["year"] >= new["entry_year"])).astype(float).where(C.notna())
    if ctype == "dairy" and scope == "large":
        for c in TREATMENT_COLUMNS:
            a, b = new[c].to_numpy(float), d[c].to_numpy(float)
            assert np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) < 1e-12, f"dairy/large rebuild differs from the panel in {c}"
        if verbose:
            print("  check: dairy / large treatment columns rebuilt by cafo_designs equal the panel's (7 columns, max abs diff < 1e-12)")
    for c in TREATMENT_COLUMNS:
        d[c] = new[c]
    d["_cafo_count"] = C                                  # the type's operation count (large or all sizes), for the baseline-intensity interaction
    waves = d[d.year.isin(CENSUS_YEARS)].assign(_c=C).pivot(index="fips", columns="year", values="_c")
    never_present = waves.index[~(waves.fillna(0) > 0).any(axis=1)]
    return d, _two_designs(d, never_present, f"cafo frame {ctype} / {scope}")


FSIS_COLUMN = "n_slaughterhouse_present_establishments_fsis"
FSIS_WINDOW = (2017, 2023)


def fsis_designs(df, check=True):
    """Slaughterhouses (FSIS plants) in the five CAFO treatment formats, annual 2017-2023 (the FSIS years; NaN outside them).
    Count = plants with slaughter present (the script4a fsis measure), 0 for a county absent from the FSIS panel inside the window.
    T1 presence (count > 0); T3 plants per 10,000 residents; T4 log(1 + count); E2 absorbing from the first year the count rises over the previous
    year (2018 on; the script4a F_cohort); E3 absorbing from the first year with a count above zero after a year at zero. The same columns as
    cafo_designs are overwritten, so every stage-4 section that reads them runs unchanged. Entries before 2017 are invisible."""
    fp = latest_file_glob(os.path.join(db_data, "merged"), "*_panel_fsis.csv")
    f = pd.read_csv(fp, low_memory=False, usecols=["fips", "year", FSIS_COLUMN])
    f["fips"] = f["fips"].astype(str).str.zfill(5)
    d = df.drop(columns=[FSIS_COLUMN], errors="ignore").merge(f, on=["fips", "year"], how="left").sort_values(["fips", "year"]).reset_index(drop=True)
    inwin = d.year.between(*FSIS_WINDOW)
    count = pd.Series(np.where(inwin, d[FSIS_COLUMN].fillna(0), np.nan), index=d.index)
    prev = count.groupby(d["fips"]).shift(1)
    up = d[(count - prev) > 0].groupby("fips")["year"].min()
    ent = d[(prev == 0) & (count > 0)].groupby("fips")["year"].min()
    new = pd.DataFrame(index=d.index)
    new["tr_e1_lg_bin"] = (count > 0).astype(float).where(count.notna())
    new["tr_i4_lg_p10k"] = count / d["population"].replace(0, np.nan) * 10_000
    new["tr_c3_log_lg"] = np.log1p(count)
    new["add_year"] = d["fips"].map(up)
    new["tr_e2_add_absorb"] = (new["add_year"].notna() & (d["year"] >= new["add_year"])).astype(float).where(count.notna())
    new["entry_year"] = d["fips"].map(ent)
    new["tr_e3_entry_absorb"] = (new["entry_year"].notna() & (d["year"] >= new["entry_year"])).astype(float).where(count.notna())
    if check:                                    # the script4a section_fsis definitions of the count and the first rise, recomputed independently
        w = d[inwin].assign(c=d.loc[inwin, FSIS_COLUMN].fillna(0))
        first_up = w.assign(dd=w.groupby("fips")["c"].diff()).query("dd > 0").groupby("fips")["year"].min()
        assert first_up.reindex(up.index).equals(up) and len(first_up) == len(up), "FSIS add_year differs from the script4a F_cohort definition"
        assert np.array_equal(count[inwin].to_numpy(), w["c"].to_numpy()), "FSIS count differs from F1_count"
        print(f"  check: FSIS first-rise cohorts equal script4a's F_cohort ({len(up):,} counties) and the count equals F1_count")
    for c in TREATMENT_COLUMNS:
        d[c] = new[c]
    never_present = d[inwin].groupby("fips")["tr_e1_lg_bin"].max().pipe(lambda m: m.index[m == 0])
    return d, _two_designs(d, never_present, f"fsis frame ({FSIS_COLUMN})")
