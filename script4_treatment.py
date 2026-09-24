#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4_treatment.py  --  SHARED TREATMENT / SAMPLE MODULE for all script4* analyses.

Importable (underscore name) because stage-numbered scripts use dashes and cannot be
imported. Every script4* file loads its panel and treatment variables from here, so
that a treatment variable is defined in EXACTLY ONE place.

Why this module exists
----------------------
In the previous single-file script4-model-test.py, treatment variables were built
inline in four different blocks, some guarded by `if col not in df.columns`. That let
two different missingness rules coexist for the same underlying variable:
  - Blocks 1/2 used  `cafo_dairy_large.fillna(0) > 0`   (NaN -> untreated)
  - Blocks 1b-1e used `(x > 0).where(x.notna())`         (NaN -> NaN)
CAFO is 100% missing in 2000-2001 (first ag census is 2002), so the first rule silently
codes 158 counties that demonstrably HAD a large dairy CAFO in 2002 as untreated in
2000-01 -- 7.1% of the Deaths of Despair estimating sample.

RULE ENFORCED HERE: treatment is NaN wherever the underlying CAFO count is NaN.
There is no fillna(0) anywhere in this module.

Size definition
---------------
`cafo_dairy_large` = dairy operations with 500+ milk cows (USDA NASS top inventory
bin, code 7; see script0b-usda-raw.py:663). medium = 200-499, small = <200.
NOTE: EPA's regulatory "Large CAFO" threshold for dairy is 700+ mature cows. USDA
publishes no 700 cutpoint, so 500+ is the nearest available bin. This is NOT the
regulatory definition and must be stated as such in the paper.

Treatment timing
----------------
CAFO counts come from the ag census (2002, 2007, 2012, 2017, 2022) and are
forward-filled to inter-census years by script1b. ALL within-county treatment
variation therefore occurs at those 5 dates only. Event "timing" carries up to 5
years of measurement error: a county recorded as changing in 2012 actually changed
somewhere in 2008-2012.

Exits
-----
Per team decision 2026-09-18, treatment is ABSORBING: once a county has a positive
change event it stays treated. Exits and contractions never switch treatment back
off. Exit counties are NOT dropped -- dropping them would select on a treatment path
that may correlate with the outcome.
"""

from packages import *
from functions import *
import numpy as np
import pandas as pd

CENSUS_YEARS = [2002, 2007, 2012, 2017, 2022]
MERGED_DIR   = os.path.join(db_data, "merged")

# Waves where NASS's own total-operations count is available to validate the size
# bins. 2022 is "dark" for all three cattle classes (cows milk / cows beef / incl
# calves) -- nass_total_ops missing, so bin completeness cannot be checked there.
# See Data/clean/diagnostic/*_qa_suppressed_bin_imputation.csv.
VERIFIED_WAVES = [2002, 2007, 2012, 2017]


# =============================================================================
# OUTCOMES
# =============================================================================
# Coverage is NOT uniform across these -- it is the binding constraint on which
# entry cohorts are even observable, so it is recorded here rather than discovered
# at estimation time:
#   poor_mental_health_days        2010-2023  (54.7% of rural county-years)
#   frequent_mental_distress       2016-2023  (32.9%) -- SHORTEST series. Only the
#                                  2017 and 2022 cohorts are observable at all,
#                                  which is why A18 returns just 2 cohorts for it.
#   crude_rate_from_census_pop     2000-2020  (19.8%; hard cliff to 0% after 2020)
#   violent_crime  -- DROPPED 2026-09-21 (D-06): CHR v043 is multi-year pooled
#                     (46.7% of values identical to prior year), release-year dated,
#                     and discontinued after the 2022 release.
#   crime_agg_assault              2000-2021  (38.1%)  aggravated assault, alone
#   assault_total_per100k          2000-2021  (38.1%)  agg + simple (79% simple)
#   violence_index_partial         2000-2021  (38.1%)  agg assault + rape + intimidation
#   total_incidents_per100k        2000-2021  (38.1%)  NIBRS curated 12-offence total
#   NOTE: all NIBRS variables are ARREST counts, not offences known to police.
#   NIBRS crime block                NIBRS arrest coverage, sparser
DESPAIR_COL = "crude_rate_from_census_pop"

OUTCOMES = {
    "Poor MH Days":             "poor_mental_health_days",
    "Frequent Mental Distress": "frequent_mental_distress_per100k",
    "Deaths of Despair":        DESPAIR_COL,
    # CHR "Violent Crime" (v043) removed 2026-09-21 -- multi-year pooled, see D-06.
    "Aggravated Assault":       "crime_agg_assault",
    "Assault, all severities":  "assault_total_per100k",
    "Violence index (partial)": "violence_index_partial",
    "NIBRS curated total":      "total_incidents_per100k",
}


# =============================================================================
# CONTROL SETS
# =============================================================================
# CONTROL_FULL25 is the legacy set from script3-ridge.py / script4-model-test.py.
# It is kept ONLY so the old results can be reproduced. Do not use it as default:
# listwise deletion on it drops the Poor MH Days sample from ~34,000 rows and 284
# switching counties to ~10,500 rows and 96 switching counties -- two thirds of the
# identifying variation -- because several members have <80% coverage.
#
# Diagnostics run 2026-09-18 on the estimating sample:
#   - VIF (within-transformed): max 3.24 (log_pop). Nothing above 5. Condition
#     number 12.7. Collinearity is NOT a reason to trim -- it is almost entirely
#     cross-sectional and county FE removes it.
#   - Treatment VIF against all 25 controls: 1.010 (R^2 = 0.0096). Controls do not
#     absorb the treatment signal, so post-treatment/mediator bias is second-order.
# The screen below is therefore built on COVERAGE, which is the real cost.
CONTROL_FULL25 = [
    "adult_obesity_per100k", "adult_smoking_per100k", "unemployment_per100k",
    "children_in_poverty_per100k", "uninsured_adults_per100k",
    "median_household_income", "income_inequality",
    "%_hispanic", "%_non-hispanic_african_american", "%_65_and_older", "%_female",
    "mental_health_providers_per100k", "primary_care_physicians_per100k",
    "food_insecurity_per100k", "physical_inactivity_per100k",
    "poor_physical_health_days", "teen_births_per100k", "low_birthweight_per100k",
    "premature_death", "preventable_hospital_stays", "some_college_per100k",
    "children_in_single-parent_households_per100k", "diabetes_prevalence_per100k",
    "air_pollution_-_particulate_matter", "social_associations_per100k",
]

# CORE_9: all >=92% coverage in the 2010-2023 window, all plausibly pre-determined
# with respect to a CAFO opening. This is the DEFAULT.
CONTROL_CORE9 = [
    "adult_obesity_per100k", "uninsured_adults_per100k", "unemployment_per100k",
    "children_in_poverty_per100k", "%_female", "%_65_and_older", "%_hispanic",
    "median_household_income", "some_college_per100k",
]

# Members of FULL25 that are plausibly POST-treatment (outcomes of local economic
# or health conditions a CAFO could itself move). Excluded from CORE9. Kept as a
# named list so the exclusion is auditable rather than implicit.
CONTROL_POSTTREATMENT = [
    "premature_death", "preventable_hospital_stays", "poor_physical_health_days",
    "low_birthweight_per100k", "diabetes_prevalence_per100k",
    "food_insecurity_per100k", "physical_inactivity_per100k",
]


# =============================================================================
# LOADER + TREATMENT CONSTRUCTION
# =============================================================================
def _wave_events(df, count_col, total_col, waves):
    """
    Classify each county's wave-to-wave change in `count_col`.
    Returns one row per (fips, wave) where a previous wave exists.

    WHAT DELTA MEANS -- read this before using is_build / is_consolidate.
    ---------------------------------------------------------------------
    Delta is the change between CONSECUTIVE AGRICULTURAL CENSUS WAVES for the
    same county. It is a FIVE-YEAR difference, not a year-over-year change,
    because CAFO counts only exist in census years. There are exactly four
    wave-pairs: 2002->2007, 2007->2012, 2012->2017, 2017->2022.

        d_count = cafo_dairy_large[wave t]  - cafo_dairy_large[wave t-1]
        d_total = cafo_dairy_total[wave t]  - cafo_dairy_total[wave t-1]

    where cafo_dairy_large = operations with 500+ milk cows, and
          cafo_dairy_total = ALL dairy operations (small + medium + large).

    So the two mechanism flags are:

        is_build       d_count > 0  AND  d_total > 0
                       Large operations increased AND the county's total dairy
                       operation count also increased -> consistent with genuinely
                       NEW operations being added.            222 of 938 events.

        is_consolidate d_count > 0  AND  d_total < 0
                       Large operations increased WHILE the total number of dairy
                       operations FELL -> small farms disappearing and surviving
                       herds crossing the 500-cow line. This is CONSOLIDATION, not
                       construction.                          676 of 938 events.

        (40 further events have d_count > 0 and d_total == 0 and fall in neither.)

    Worked example, fips 42071 (Lancaster County, PA) -- a consolidation county:
        wave   small   medium   LARGE   TOTAL    d_LARGE   d_TOTAL
        2002    1870       32       9    1911         .         .
        2007    1889       26      13    1928        +4       +17
        2012    1841       17      20    1878        +7       -50
        2017    1572       22      19    1613        -1      -265
        2022     805       42      20     867        +1      -746
    Large operations roughly double while the county loses over a thousand dairy
    farms. A treatment defined only on d_LARGE > 0 would call this "a CAFO was
    added"; what actually happened is that the local dairy sector collapsed and
    concentrated.

    Once a county has a qualifying event at some wave, the treatment variable is
    ABSORBING: it is 1 from that wave onward in every subsequent year, including
    the forward-filled inter-census years.

    The split is motivated by a data fact, not a modelling preference: 70-75% of positive changes in large-dairy counts occur while the
    county's TOTAL dairy operation count is FALLING (median: small -8, total -7).
    A rise in large-op counts therefore predominantly measures SECTOR
    CONSOLIDATION -- small operations exiting, surviving herds crossing 500 head --
    not new construction. Those two have different mechanisms for mental health
    (environmental exposure vs. farm failure / rural economic decline), so they are
    separated here and estimated against each other rather than averaged.

    Verified against NASS's own total-operations count: `gap` = nass_total_ops
    minus (small+medium+large) is exactly 0 for all 10,110 dairy county-waves where
    it is computable, so the decline in total ops is real and not bin suppression.
    """
    d = df[df["year"].isin(waves)][["fips", "year", count_col, total_col]].copy()
    d = d.sort_values(["fips", "year"])
    d["lag_count"] = d.groupby("fips")[count_col].shift(1)
    d["lag_total"] = d.groupby("fips")[total_col].shift(1)
    d = d.dropna(subset=[count_col, total_col, "lag_count", "lag_total"])
    d["d_count"] = d[count_col] - d["lag_count"]
    d["d_total"] = d[total_col] - d["lag_total"]

    d["is_entry"]     = (d["lag_count"] == 0) & (d[count_col] > 0)
    d["is_expansion"] = (d["lag_count"] > 0) & (d["d_count"] > 0)
    d["is_positive"]  = d["d_count"] > 0
    d["is_exit"]      = (d["lag_count"] > 0) & (d[count_col] == 0)
    d["is_build"]       = d["is_positive"] & (d["d_total"] > 0)
    d["is_consolidate"] = d["is_positive"] & (d["d_total"] < 0)
    return d


def _absorb_from(df, first_year_col):
    """1 from the county's first qualifying wave onward, 0 before, NaN where the
    underlying CAFO count is NaN. Absorbing: never returns to 0."""
    t = np.where(df[first_year_col].notna() & (df["year"] >= df[first_year_col]), 1.0, 0.0)
    return pd.Series(t, index=df.index).where(df["cafo_dairy_large"].notna())


def load_panel(animal="dairy", verified_waves_only=False):
    """
    Returns the rural county-year panel with every script4 treatment variable
    attached. `verified_waves_only=True` drops the 2022 wave from event definitions
    (not from the panel), for the robustness cut where NASS totals are unavailable.
    """
    path = latest_file_glob(MERGED_DIR, "*_panel.csv")
    df = pd.read_csv(path, low_memory=False)
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["state_fips"] = df["fips"].str[:2]
    df = df.sort_values(["fips", "year"]).reset_index(drop=True)

    # The panel written by script1b is ALREADY rural-filtered (rural == 1 for all
    # 65,828 rows), so the filter in the old script4 was a no-op. Asserted, not
    # re-applied, so a future change to script1b surfaces here instead of silently
    # changing the sample.
    assert (df["rural"] == 1).all(), "panel is expected to be pre-filtered to rural counties"

    lg  = f"cafo_{animal}_large"
    md  = f"cafo_{animal}_medium"
    sm_ = f"cafo_{animal}_small"
    tot = f"cafo_{animal}_total"

    df["log_pop"] = np.log(df["population"].where(df["population"] > 0))

    # ---- OUTCOME CONSTRUCTION (moved here from script4-model-test.py) -------
    # Despair: crude_rate_from_census_pop (built in script1b as deaths/census pop
    # x 100k) is used rather than CDC's own crude_rate_despair, which CDC
    # suppresses at low death counts. corr = 0.9999 where both exist, and it
    # recovers ~2x the usable sample. BOTH variants hit a hard cliff to 0%
    # coverage after 2020 in the source data itself -- truncated explicitly here
    # rather than left to be implicitly NaN.
    df.loc[df["year"] > 2020, DESPAIR_COL] = np.nan

    # ---- CRIME OUTCOMES ----------------------------------------------------
    # IMPORTANT: every NIBRS variable in this panel is built from the
    # `nibrs_arrestee_segment` files, i.e. they are ARREST counts, not offences
    # known to police. Arrest rates confound crime incidence with policing
    # intensity and clearance. Standard published crime rates are offence-based.
    # This must be stated wherever these outcomes are reported.
    #
    # `total_incidents_per100k` is NOT all crime. script0d sums a CURATED list of
    # 12 offence types (aggravated assault, simple assault, intimidation, rape,
    # statutory rape, incest, fondling, sexual assault with an object,
    # kidnapping/abduction, DUI, and two human-trafficking codes). Property and
    # drug offences are excluded. It is dominated by assault: simple assault alone
    # is 68% of it, and simple + aggravated is 86%.
    #
    # DROPPED 2026-09-21 (decision D-06): CHR `violent_crime` (v043). CHR pools it
    # over multiple years -- 46.7% of county-year values are identical to the prior
    # year -- so it cannot support annual event-study timing.

    # Aggravated assault, kept ALONE. The single largest genuine UCR violent
    # offence in these counties, annually dated, no pooling.
    df["crime_agg_assault"] = df["aggravated_assault_per100k"]

    # Renamed from `crime_assault`. Aggravated + simple assault. Explicitly NOT a
    # violent-crime measure: 79% of it is SIMPLE assault, which UCR excludes from
    # its violent-crime definition. Named for what it is.
    df["assault_total_per100k"] = (df["aggravated_assault_per100k"].fillna(0)
                                    + df["simple_assault_per100k"].fillna(0))
    df.loc[df["total_incidents_per100k"].isna(), "assault_total_per100k"] = np.nan

    # Partial violence index -- NOT the UCR definition. Aggravated assault + rape
    # + intimidation: the three person-directed violent offences available in the
    # panel with usable coverage. ROBBERY is missing, and robbery is a UCR violent
    # component -- it exists in the raw nibrs_arrestee_segment files but script0d's
    # `crime_cols` does not extract it, so adding it requires re-running stage 0.
    # Homicide is also absent from NIBRS here (`homicides` is CHR-sourced).
    # Label this "violent offences against persons (partial, non-UCR)".
    _vi = (df["aggravated_assault_per100k"].fillna(0)
           + df["rape_per100k"].fillna(0)
           + df["intimidation_per100k"].fillna(0))
    df["violence_index_partial"] = _vi.where(df["total_incidents_per100k"].notna())

    waves = VERIFIED_WAVES if verified_waves_only else CENSUS_YEARS
    ev = _wave_events(df, lg, tot, waves)

    # First wave of each event type, merged back on as a county-level constant.
    for flag, col in [("is_positive", "add_year"), ("is_entry", "entry_year"),
                      ("is_build", "build_year"), ("is_consolidate", "consol_year")]:
        first = ev[ev[flag]].groupby("fips")["year"].min().rename(col)
        df = df.merge(first, on="fips", how="left")

    # ---- GROUP E: extensive margin -----------------------------------------
    df["tr_e1_lg_bin"]      = (df[lg] > 0).astype(float).where(df[lg].notna())
    df["tr_e2_add_absorb"]  = _absorb_from(df, "add_year")
    df["tr_e3_entry_absorb"] = _absorb_from(df, "entry_year")
    # Relative event time for the event study / CS DiD. NaN for never-treated.
    df["t_rel"]  = df["year"] - df["add_year"]
    df["cohort"] = df["add_year"]

    # ---- GROUP I: intensive margin (dose) ----------------------------------
    # Cumulative SUM OF POSITIVE CHANGES ONLY (contractions do not subtract),
    # so the dose is consistent with the absorbing extensive-margin definition.
    pos = ev[ev["is_positive"]][["fips", "year", "d_count"]].rename(columns={"d_count": "_add"})
    cum = pos.sort_values(["fips", "year"]).copy()
    cum["_cum"] = cum.groupby("fips")["_add"].cumsum()
    cum["_n"]   = cum.groupby("fips").cumcount() + 1
    df = df.merge(cum[["fips", "year", "_cum", "_n"]], on=["fips", "year"], how="left")
    df["tr_i1_add_cum"]     = df.groupby("fips")["_cum"].ffill().fillna(0.0).where(df[lg].notna())
    df["tr_i2_add_nevents"] = df.groupby("fips")["_n"].ffill().fillna(0.0).where(df[lg].notna())
    df = df.drop(columns=["_cum", "_n"])
    df["tr_i3_lg_count"] = df[lg]
    df["tr_i4_lg_p10k"]  = (df[lg] / df["population"].replace(0, np.nan)) * 10_000

    # ---- GROUP M: mechanism split ------------------------------------------
    df["tr_m1_build"]       = _absorb_from(df, "build_year")
    df["tr_m2_consolidate"] = _absorb_from(df, "consol_year")
    df["tr_m3_add_any"]     = df["tr_e2_add_absorb"]

    # ---- GROUP C: composition ----------------------------------------------
    tot_safe = df[tot].replace(0, np.nan)
    df["tr_c1_lg_share"]     = df[lg] / tot_safe
    # Screened version: the median county has only 7 total dairy operations and 40%
    # have <=5, so the raw share is mechanically unstable in small counties.
    df["tr_c2_lg_share_scr"] = df["tr_c1_lg_share"].where(df[tot] >= 10)
    df["tr_c3_log_lg"]       = np.log1p(df[lg])
    bins = df[[sm_, md, lg]]
    df["tr_c5_hhi"] = ((bins.div(bins.sum(axis=1).replace(0, np.nan), axis=0)) ** 2).sum(axis=1, min_count=1)

    # ---- CONDITIONING variables (the "conditional on small CAFOs" models) ---
    # cond_log_sm is CONTEMPORANEOUS and is post-treatment if large operations
    # displace small ones -- that is precisely the consolidation mechanism. It is
    # still built, because the contrast against the baseline version is informative.
    # cond_log_sm_base is the county's small-operation count at its FIRST OBSERVED
    # wave: pre-determined, so conditioning on it does not block the causal path.
    df["cond_log_sm"]  = np.log1p(df[sm_])
    df["cond_log_tot"] = np.log1p(df[tot])
    base = (df[df["year"].isin(CENSUS_YEARS) & df[sm_].notna()]
            .sort_values(["fips", "year"]).groupby("fips")[sm_].first().rename("_sm_base"))
    df = df.merge(base, on="fips", how="left")
    df["cond_log_sm_base"] = np.log1p(df["_sm_base"])
    df = df.drop(columns=["_sm_base"])

    # cond_log_sm_base is COUNTY-CONSTANT, so in a county-FE model it is absorbed
    # entirely (verified: 0 of 2,635 counties have any within-variation in it).
    # Entering it as a level does nothing -- the regression silently reduces to the
    # unconditional one. A pre-determined baseline can only enter a FE model
    # INTERACTED WITH TIME, which lets counties with different baseline small-farm
    # structure follow different trends. That is what C4 uses.
    df["cond_smbase_x_t"] = df["cond_log_sm_base"] * (df["year"] - df["year"].min())

    df.attrs["panel_path"] = path
    df.attrs["events"] = ev
    return df


# =============================================================================
# TREATMENT REGISTRY -- what gets crossed with every model spec
# =============================================================================
# `conditioning`: extra regressors entered ALONGSIDE the treatment. For Group C
# this is what makes the model "conditional on small-CAFO presence": the treatment
# coefficient is then the association with large operations HOLDING the small-farm
# count fixed, and the small-farm coefficient is reported separately so the two
# rival mechanisms can be read against each other.
TREATMENTS = {
    # id            column                 conditioning                label
    "E1": ("tr_e1_lg_bin",      [],                    "Large-dairy presence (binary)"),
    "E2": ("tr_e2_add_absorb",  [],                    "Any positive change, absorbing"),
    "E3": ("tr_e3_entry_absorb",[],                    "First entry (0->+), absorbing"),
    "I1": ("tr_i1_add_cum",     ["log_pop"],           "Cumulative ops added (dose)"),
    "I2": ("tr_i2_add_nevents", ["log_pop"],           "Number of add events"),
    "I3": ("tr_i3_lg_count",    ["log_pop"],           "Large-op raw count"),
    "I4": ("tr_i4_lg_p10k",     [],                    "Large ops per 10k residents"),
    "C1": ("tr_c1_lg_share",    [],                    "Large share of dairy ops"),
    "C2": ("tr_c2_lg_share_scr",[],                    "Large share, total>=10"),
    # C3 REPLACES the old M1/M2 build-vs-consolidation split. M1/M2 partitioned
    # counties into two arbitrary binary bins and returned near-identical
    # coefficients (+0.112 vs +0.104), which cannot say WHICH margin matters.
    # The conditional model estimates both margins directly: log(large) holding
    # small-farm count fixed, and log(small) holding large fixed.
    "C3": ("tr_c3_log_lg",      ["cond_log_sm"],       "log(large) | log(small), contemp."),
    "C4": ("tr_c3_log_lg",      ["cond_smbase_x_t"],   "log(large) | baseline small x trend"),
    "C5": ("tr_c5_hhi",         [],                    "HHI across size bins"),
}


# -----------------------------------------------------------------------------
# COMPREHENSIVE control sets, DERIVED from the candidate pool (2026-09-18)
# -----------------------------------------------------------------------------
# Replaces the inherited CONTROL_FULL25 with sets built under a stated rule:
#   candidate pool = panel columns that are not identifiers, CHR companion columns
#   (numerator/denominator/CI bounds), treatment columns, outcomes, or population
#   denominators  -> 107 columns, 105 numeric
#   selection rule = numeric AND >=92% complete on rows with a non-missing outcome
#                 -> 26 columns
#
# NON-NUMERIC CONTROLS: checked and there are none usable. The only two
# non-numeric candidates are `pct_of_total_deaths_raw_despair` (a string
# percentage at 29.6% coverage, and derived from the despair OUTCOME) and
# `state_abbrev` (the state identifier, already absorbed by state clustering and
# the state FE in pooled specs). Nothing binary or categorical is being lost.
CONTROL_COMPREHENSIVE = [
    "%_65_and_older", "%_asian", "%_below_18_years_of_age", "%_female",
    "%_hispanic", "%_native_hawaiian/other_pacific_islander",
    "%_not_proficient_in_english_per100k", "%_rural",
    "access_to_healthy_foods_per100k", "adult_obesity_per100k",
    "adult_smoking_per100k", "children_in_poverty_per100k",
    "children_in_single-parent_households_per100k", "diabetes_prevalence_per100k",
    "driving_alone_to_work_per100k", "low_birthweight_per100k",
    "median_household_income", "physical_inactivity_per100k",
    "poor_or_fair_health_per100k", "poor_physical_health_days", "premature_death",
    "preventable_hospital_stays", "some_college_per100k", "teen_births_per100k",
    "unemployment_per100k", "uninsured_adults_per100k",
]

# Health/economic measures a CAFO could itself move. Excluded from the DEFAULT set
# because conditioning on a mediator blocks part of the causal path. Kept in
# CONTROL_COMPREHENSIVE so the two can be compared as a robustness arm.
CONTROL_MEDIATORS = [
    "poor_physical_health_days", "premature_death", "preventable_hospital_stays",
    "low_birthweight_per100k", "diabetes_prevalence_per100k",
    "physical_inactivity_per100k", "poor_or_fair_health_per100k",
]

# DEFAULT for the core models: comprehensive, minus plausible mediators.
CONTROL_PRETREAT = [c for c in CONTROL_COMPREHENSIVE if c not in CONTROL_MEDIATORS]

# -----------------------------------------------------------------------------
# THE THREE CORE TREATMENTS (team decision, 2026-09-18)
# -----------------------------------------------------------------------------
# Everything else in TREATMENTS stays available as robustness, but these three
# carry the headline results and are the ones run through Callaway-Sant'Anna.
# ---------------------------------------------------------------------------
# D_i : EXPOSURE TO CAFOs  -- the treatment set (labels fixed 2026-09-23)
# ---------------------------------------------------------------------------
# These are TREATMENT DEFINITIONS ONLY: the exposure measure itself, nothing
# else. Controls -- including log_pop -- belong to the REGRESSION SPECIFICATION
# and are defined separately, further down. Keeping them apart matters: log_pop
# is a control in the T2 and T4 regressions, but it is not part of what T2 or T4
# measure.
#
# All defined on LARGE DAIRY operations only (500+ milk cows, USDA top inventory
# bin -- NOT the EPA 700+ regulatory threshold; see the size note above).
#
#   T1  Presence      binary 0/1: does the county have a large dairy CAFO?
#   T2  Count         raw count of large dairy CAFOs. beta reads as the marginal
#                     impact of ONE additional CAFO.
#   T3  Per capita    raw count / population, per 10,000 residents
#   T4  Log(count)    log(1 + count). The +1 is required, not cosmetic: 81.4% of
#                     county-years have zero large dairy CAFOs, and plain log
#                     would discard them.
#   T6  Conditional   given the BASELINE number of SMALL dairy CAFOs, the effect
#                     of introducing a large one -- market concentration, the
#                     loss of small farms to large. DEFINED BUT NOT RUN with
#                     T1-T4: it answers a different question. Its controls are
#                     still to be specified.
#
# T5 (log per capita) was DROPPED 2026-09-23. The numbering gap is deliberate --
# T6 keeps its label so it matches the slides.
#
# T1-T4 are four functional forms of the SAME underlying exposure and should
# agree in sign. Divergence between them is itself a finding.
CORE_TREATMENTS = {
    "T1": "tr_e1_lg_bin",     # Presence of a large dairy CAFO (0/1)
    "T2": "tr_i3_lg_count",   # Count of large dairy CAFOs
    "T3": "tr_i4_lg_p10k",    # Large dairy CAFOs per 10,000 residents
    "T4": "tr_c3_log_lg",     # Log(1 + count of large dairy CAFOs)
}
CORE_TREATMENT_LABELS = {
    "T1": "Presence of a large dairy CAFO (0/1)",
    "T2": "Count of large dairy CAFOs",
    "T3": "Large dairy CAFOs per 10,000 residents",
    "T4": "Log(1 + count of large dairy CAFOs)",
    "T6": "Log(1+count) given baseline small dairy CAFOs",
}
# Defined for the record; excluded from the T1-T4 runs. Controls TBD.
T6_TREATMENT = "tr_c3_log_lg"

# ---------------------------------------------------------------------------
# REGRESSION SPECIFICATION -- separate from the treatment definition above.
# ---------------------------------------------------------------------------
# Extra regressors a given treatment requires in its estimating equation. These
# are NOT part of the exposure measure; they are there so the coefficient reads
# correctly.
#   T2, T4  enter log_pop, because a raw or logged COUNT otherwise partly picks
#           up county size -- larger counties have more of everything.
#   T1      binary presence: no size adjustment needed.
#   T3      already population-normalised by construction.
SPEC_EXTRA_REGRESSORS = {
    "T1": [],
    "T2": ["log_pop"],
    "T3": [],
    "T4": ["log_pop"],
    "T6": [],          # to be specified
}

# Single source of truth for the headline spec. Estimator named explicitly:
# pooled/state-FE results are a benchmark, never the headline.
HEADLINE_ESTIMATOR = "within (county + year FE)"
HEADLINE_FE        = "fips + year"
HEADLINE_TREATMENT = "T1"
HEADLINE_OUTCOME   = "poor_mental_health_days"

# ---------------------------------------------------------------------------
# COVARIATE ORDER for the sequential A1/A2 runs (fixed 2026-09-23)
# ---------------------------------------------------------------------------
# Forward stepwise results depend entirely on the order variables are added, so
# the order is FIXED HERE and stated, not left implicit. 13 covariates enter one
# at a time; VIF, the change in beta, partial R2 and surviving switcher count are
# recorded at each step.
#
# NOTE ON THE COVERAGE CLIFF: positions 1-5 and 8-10 are full-coverage
# (99.7-99.9%, re-sourced by script0g); the rest are still CHR-sourced at
# 52.8-65.9%. Adding a partial-coverage variable collapses the estimating sample
# through listwise deletion, so every step after the first one confounds "what
# this covariate does" with "what losing half the sample does".
# The order is therefore grouped: all EIGHT full-coverage covariates first, then
# the six partial-coverage ones ranked by coverage descending. This gives eight
# steps where a change in beta is attributable to the covariate alone, and pushes
# the sample loss as late as possible.
#
# The ACS 5-year block (some_college, children_in_single-parent,
# %_not_proficient_in_english) all share the SAME 53.4% window, so whichever
# enters first pays the entire sample cost and the other two then look free. The
# real question is whether the ACS block earns its place AS A GROUP, not whether
# any individual member does.
# %_not_proficient_in_english enters last, against a specification that already
# contains %_hispanic -- the honest test of whether the two are redundant.
COVARIATE_ORDER = [
    # --- full coverage (99.6-99.9%) : eight clean steps, sample intact --------
    "median_household_income",                      #  1  SAIPE      99.9%
    "unemployment_per100k",                         #  2  BLS LAUS   99.7%
    "%_65_and_older",                               #  3  Census PEP 99.9%
    "%_hispanic",                                   #  4  Census PEP 99.9%
    "%_below_18_years_of_age",                      #  5  Census PEP 99.9%
    "children_in_poverty_per100k",                  #  6  SAIPE      99.9%
    "%_female",                                     #  7  Census PEP 99.9%
    "%_asian",                                      #  8  Census PEP 99.9%
    # --- partial coverage : the sample falls from here ------------------------
    "uninsured_adults_per100k",                     #  9  SAHIE      65.9%  (2008+)
    "adult_obesity_per100k",                        # 10  BRFSS      57.6%
    "access_to_healthy_foods_per100k",              # 11  USDA       57.3%
    "some_college_per100k",                         # 12  ACS 5-yr   53.4%
    "children_in_single-parent_households_per100k", # 13  ACS 5-yr   53.4%
    "%_not_proficient_in_english_per100k",          # 14  ACS 5-yr   53.4%
]

# Excluded from the sequential runs, with the reason recorded for the paper.
COVARIATE_EXCLUDED = {
    "adult_smoking_per100k":
        "COLLIDER -- responds to distress and to local economic shocks, so it is "
        "caused by both the outcome and the treatment. Conditioning induces bias.",
    "teen_births_per100k":
        "COLLIDER -- same reasoning as adult_smoking.",
    "%_rural":
        "DUPLICATED -- the sample is already filtered on the NCHS urban-rural "
        "classification, and %_rural is decennial (93.7% of values identical to "
        "the prior year; within-SD share 0.080).",
    "driving_alone_to_work_per100k":
        "Could not obtain consistent data across the panel window.",
    "%_native_hawaiian/other_pacific_islander":
        "Negligible in non-metro counties; contributes no usable variation.",
}

# %_hispanic was initially dropped for collinearity with
# %_not_proficient_in_english, then REINSTATED at position 14 once that reasoning
# was checked: the two correlate +0.773 in LEVELS but only -0.243 WITHIN county,
# and the within correlation is what a county-FE model uses. Placing it last
# means its marginal contribution is read against a specification that already
# contains the English-proficiency measure, which is the honest test of whether
# the two are redundant.
