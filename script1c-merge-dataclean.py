#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 01:10:00 2025

Builds one county-year merged panel from the latest version of each clean file.

Panel frame: ALL counties where rural-key non_large_metro == 1.
  The rural-key (NCHS urban-rural classification) is the backbone of the panel.
  Every other dataset — including CAFO — is LEFT JOINED onto this frame, so
  rural counties with no data for a given source are always present as NaN rows
  (or 0 for CAFO counts, see zero-fill section below).

Key design decisions:
- CAFO (USDA Agricultural Census): zero-fill for census-covered years.
  Absence in the census = confirmed zero operations, not a missing observation.
  Years not yet covered (2022+) remain NaN until the raw .dta is added.
- CDC deaths-of-despair: three-state missingness flag (cdc_in_query,
  deaths_is_zero, crude_rate_from_census_pop). See inline comments.
- CHR (County Health Rankings): CHR rate-only variables get imputed count
  columns (*_count_imputed). Num/denom variables get a *_ratio_flag QA column.
- FSIS (slaughterhouses): available 2017–present only; panel years 2017–2023
  overlap with the analysis window.
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

# --- CAFO zero-fill for rural counties absent from USDA Census file ---
# The USDA Agricultural Census is comprehensive: a rural county that does not
# appear in the CAFO file for a census-covered year genuinely had zero animal
# feeding operations — the absence is structural, not a missing observation.
# We fill all numeric CAFO count columns with 0 for those county-years.
#
# Only years where the CAFO file contributed ≥1 observation are treated this
# way. Years 2022+ currently have no coverage because 2022.dta has not been
# added to raw/usda/ yet — those rows remain NaN until the file is added.
#
# Text/metadata columns (commodity descriptor, county name, class) are left
# as NaN: there is no meaningful string value for a zero-operations county.
_CAFO_TEXT_COLS = {
    f"county_fips_name_{BASE_DESCRIPTOR}",
    f"commodity_desc_{BASE_DESCRIPTOR}",
    f"class_desc_{BASE_DESCRIPTOR}",
}
_cafo_covered_years = set(
    merged_all.loc[merged_all["cafo_total_ops_all_animals"].notna(), "year"].unique()
)
_cafo_fill_cols = [
    c for c in merged_all.columns
    if (BASE_DESCRIPTOR in c or c.startswith("cafo_"))
    and c not in _CAFO_TEXT_COLS
    and pd.api.types.is_numeric_dtype(merged_all[c])
]
_in_cafo_year = merged_all["year"].isin(_cafo_covered_years)
_all_cafo_null = merged_all[_cafo_fill_cols].isna().all(axis=1)
_zero_fill_mask = _in_cafo_year & _all_cafo_null
merged_all.loc[_zero_fill_mask, _cafo_fill_cols] = 0

print(
    f"CAFO zero-fill: {int(_zero_fill_mask.sum()):,} county-years → 0 "
    f"(rural counties absent from USDA Census = confirmed no operations). "
    f"Years covered by CAFO data: {sorted(_cafo_covered_years)}."
)


# Merge remaining selected datasets on top of base keys
for descriptor in sorted(MERGE_DESCRIPTORS):
    path = latest[descriptor]
    part = _read_filter_reduce(path, descriptor, allowed_keys)
    if part is None:
        continue

    merged_all = merged_all.merge(part, on=["fips", "year"], how="left")


merged_all = merged_all.sort_values(["fips", "year"]).reset_index(drop=True)
print("Final merged panel shape:", merged_all.shape)


# --- Derived: crude rate from census population ---
# Deaths column sourced from CDC WONDER county-year downloads (script0c).
# Population from US Census via population_full (100% county-year coverage).
#
# cdc_in_query: indicator = 1 if county-year appeared in any CDC WONDER file.
#   A value of 0 means the county was absent from the CDC download — could be
#   a true zero or simply outside the query scope (ambiguous; see script1d).
#   This flag is important for the ML framework to distinguish missingness types.
#
# crude_rate_from_census_pop: recomputed deaths-of-despair crude rate using
#   the census population denominator instead of the CDC internal population.
#   Allows full-panel rate comparison, bypassing CDC suppression of crude_rate
#   when deaths ≤ 20. Deaths column retains the raw count regardless of suppression.
_deaths_col = "deaths_cdc_county_year_deathsofdespair"
_pop_col = "population_population_full"

if _deaths_col in merged_all.columns and _pop_col in merged_all.columns:
    _deaths = pd.to_numeric(merged_all[_deaths_col], errors="coerce")
    _pop = pd.to_numeric(merged_all[_pop_col], errors="coerce")

    # Three missingness states for the crude rate calculation:
    #   cdc_in_query == 0 : county-year absent from CDC WONDER download — unknown
    #                        whether true zero deaths or outside query scope.
    #   cdc_in_query == 1, deaths_is_zero == 1 : CDC explicitly returned Deaths=0.
    #                        Plausibly a true zero but cannot rule out suppression
    #                        artefact; crude rate is set to NaN (not 0) to avoid
    #                        treating ambiguous zeros as confirmed no-death counties.
    #   cdc_in_query == 1, deaths_is_zero == 0 : Deaths > 0; crude rate computed.
    merged_all["cdc_in_query"] = _deaths.notna().astype("Int64")
    merged_all["deaths_is_zero"] = ((_deaths == 0) & _deaths.notna()).astype("Int64")

    # Compute rate only where deaths are positive and census pop is positive.
    _rate_mask = (_deaths > 0) & (_pop > 0)
    merged_all["crude_rate_from_census_pop"] = (
        (_deaths / _pop * 100_000).where(_rate_mask)
    )
    merged_all["deaths_per_10k_census_pop"] = (
        (_deaths / _pop * 10_000).where(_rate_mask)
    )

    n_in_query = int(merged_all["cdc_in_query"].sum())
    n_zero = int(merged_all["deaths_is_zero"].sum())
    n_rate = int(_rate_mask.sum())
    n_total = len(merged_all)
    print(
        f"Derived crude rate columns added.\n"
        f"  cdc_in_query == 1 : {n_in_query:,} county-years ({100*n_in_query/n_total:.1f}%)\n"
        f"  deaths_is_zero == 1 : {n_zero:,} (rate set to NaN — ambiguous zeros)\n"
        f"  crude rate computed : {n_rate:,} county-years with Deaths > 0"
    )
else:
    _missing = [c for c in [_deaths_col, _pop_col] if c not in merged_all.columns]
    print(f"Warning: cannot compute crude rate from census pop — missing columns: {_missing}")


# --- Derived: CHR count imputation and numerator/denominator verification ---
# Pass 1: County Health Rankings variables that report only a raw_value (a rate
#   or proportion) with no accompanying numerator column.  We back-calculate an
#   implied count using the census population denominator, clearly labelled
#   *_count_imputed so downstream users know these are derived, not measured.
#
# Pass 2: CHR variables that have both numerator and denominator columns.
#   We verify numerator / denominator ≈ raw_value (within tolerance) and write
#   a *_ratio_flag column (1 = discrepancy) for QA auditing.  No values are
#   modified.
#
# These steps are logically downstream of the merge (they need population and
# the full CHR column set), so they are done here rather than in a separate script.

_CHR_SUFFIX = "_mentalhealthrank_full"
_CHR_RATE_ONLY_VARS = [
    "frequent_mental_distress",
    "frequent_physical_distress",
    "insufficient_sleep",
    "access_to_exercise_opportunities",
    "%_american_indian_and_alaskan_native",
    "%_non-hispanic_african_american",
]
_CHR_ABS_TOL = 0.05   # 5 pp in proportion space
_CHR_REL_TOL = 0.10   # 10% relative in proportion space


def _chr_detect_scale(series):
    """
    Classify CHR raw_value units for count imputation (Pass 1).
      Returns 100 when raw_value is a proportion (median ≤ 1.0), meaning
      we multiply by 100 then divide by 100 × pop — equivalent to raw × pop.
      Returns 1  when raw_value is already in percent (0–100) space, meaning
      we divide by 100 × pop.
    NOT used for ratio verification (Pass 2), which has its own unit logic.
    """
    med = series.dropna().median()
    return 100.0 if med <= 1.0 else 1.0


def _chr_safe_stem(var):
    return re.sub(r"[^a-z0-9]+", "_", var.lower()).strip("_")


# Pass 1
_pop_series = pd.to_numeric(merged_all.get(POP_COL := "population_population_full"), errors="coerce")
_pass1_cols = []
for _var in _CHR_RATE_ONLY_VARS:
    _raw_col = f"{_var}_raw_value{_CHR_SUFFIX}"
    _num_col = f"{_var}_numerator{_CHR_SUFFIX}"
    if _raw_col not in merged_all.columns or _num_col in merged_all.columns:
        continue
    _raw = pd.to_numeric(merged_all[_raw_col], errors="coerce")
    _scale = _chr_detect_scale(_raw)
    _out = f"{_chr_safe_stem(_var)}_count_imputed"
    merged_all[_out] = ((_raw * _scale / 100.0) * _pop_series).round(0).astype("Int64")
    _pass1_cols.append(_out)
print(f"CHR Pass 1 complete: {len(_pass1_cols)} count_imputed columns added.")

# Pass 2
_all_cols = set(merged_all.columns)
_flag_counts = []
for _col in sorted(_all_cols):
    if not _col.endswith(f"_numerator{_CHR_SUFFIX}"):
        continue
    _stem = _col[: -(len("_numerator") + len(_CHR_SUFFIX))]
    _den_col = f"{_stem}_denominator{_CHR_SUFFIX}"
    _raw_col = f"{_stem}_raw_value{_CHR_SUFFIX}"
    if _den_col not in _all_cols or _raw_col not in _all_cols:
        continue
    _num = pd.to_numeric(merged_all[_col], errors="coerce")
    _den = pd.to_numeric(merged_all[_den_col], errors="coerce")
    _raw = pd.to_numeric(merged_all[_raw_col], errors="coerce")
    _implied = (_num / _den).where(_den > 0)   # always in proportion (0–1) space

    # Convert raw_value to proportion space for comparison.
    # CHR raw_values come in three unit types:
    #   proportion (median ≤ 1)  → raw_prop = raw_value          (e.g. adult_obesity=0.35)
    #   percentage (1 < med ≤ 100) → raw_prop = raw_value / 100  (e.g. some_college=65.0)
    #   rate per 100k (med > 100) → skip: can't normalize without knowing the rate base
    _med = _raw.dropna().median()
    if pd.isna(_med) or _med > 100:
        # Rate-type variable (e.g. violent_crime per 100k) — skip comparison
        continue
    _raw_prop = _raw if _med <= 1.0 else (_raw / 100.0)

    _abs_diff = (_implied - _raw_prop).abs()
    _rel_diff = (_abs_diff / _raw_prop.abs()).where(_raw_prop.abs() > 1e-9)
    _checkable = _implied.notna() & _raw_prop.notna()
    _flag_mask = ((_abs_diff > _CHR_ABS_TOL) | (_rel_diff > _CHR_REL_TOL)) & _checkable
    _flag_col = f"{_chr_safe_stem(_stem)}_ratio_flag"
    merged_all[_flag_col] = pd.NA
    merged_all.loc[_checkable, _flag_col] = _flag_mask[_checkable].astype("Int64")
    n_flagged = int(_flag_mask.sum())
    n_checkable = int(_checkable.sum())
    _flag_counts.append((_stem, n_checkable, n_flagged))

_high_flag = [(s, nc, nf) for s, nc, nf in _flag_counts if nc > 0 and nf / nc > 0.05]
print(f"CHR Pass 2 complete: {len(_flag_counts)} ratio_flag columns added.")
if _high_flag:
    print("  WARNING — >5% flag rate in:")
    for _s, _nc, _nf in _high_flag:
        print(f"    {_s}: {_nf}/{_nc} ({100*_nf/_nc:.1f}%)")


# ── Rate Standardization ──────────────────────────────────────────────────────
# Derive *_per100k columns for crime counts and CHR variables not already
# expressed per 100,000. Denominator is always population_population_full.
# Purpose: put all rate variables on a common scale before regression/ML.
#
# Variables skipped (not population rates):
#   - premature_death (YPLL, not an incidence rate)
#   - preventable_hospital_stays (per Medicare enrollee, non-standard base)
#   - poor_mental_health_days, poor_physical_health_days (day-count, 0–30 scale)
#   - income_inequality (Gini coefficient)
#   - food_environment_index (composite index, 0–10)
#   - air_pollution_-_particulate_matter (µg/m³)
#   - median_household_income (dollars)
#   - drinking_water_violations (binary flag)
#   - alcohol-impaired_driving_deaths (fraction of driving deaths, not per pop)
#   - demographic composition columns (% by age/race/sex, %_rural, population)

_pop_col = "population_population_full"
_pop = merged_all[_pop_col]

# --- Part 1: UCR crime counts → per 100k ------------------------------------
# Raw incident counts from FBI UCR / state systems aggregated to county-year.
# Metadata columns (state label, county label, n_rows) are excluded.
_CRIME_COUNT_COLS = [
    c for c in merged_all.columns
    if c.endswith("_crime_fips_level_final")
    and not any(c.startswith(p) for p in ("state_", "county_", "n_rows_"))
]
for _col in _CRIME_COUNT_COLS:
    _new = _col.replace("_crime_fips_level_final", "") + "_per100k"
    merged_all[_new] = (merged_all[_col] / _pop * 100_000).where(_pop > 0)

print(f"Crime rate derivation: {len(_CRIME_COUNT_COLS)} count columns → *_per100k.")

# --- Part 2: CHR variables → standardize to per 100,000 ---------------------
# CHR distributes raw_value columns in mixed units. We create *_per100k
# variants using the documented unit for each variable group.

_CHR_SUFFIX = "_raw_value_mentalhealthrank_full"

# 2a. Provider ratios: raw value = providers / population
#     (e.g. 0.0005 ≈ 50 providers per 100,000). Multiply by 100,000.
_CHR_PROVIDER_RATIO = [
    "primary_care_physicians",
    "mental_health_providers",
    "dentists",
    "other_primary_care_providers",
]

# 2b. Per-1,000 base (per 1,000 live births or per 1,000 females 15–19).
#     Multiply by 100 to convert to per 100,000.
_CHR_PER1000 = ["teen_births", "infant_mortality"]

# 2c. Per-10,000 base (civic/social organizations per 10,000 population).
#     Multiply by 10 to convert to per 100,000.
_CHR_PER10000 = ["social_associations"]

# 2d. Proportions (0–1): rate/prevalence variables expressed as a fraction of
#     the relevant population. Multiply by 100,000 to convert to per 100,000.
#     Demographic composition variables (% by race, sex, age, % rural) excluded
#     — they describe population composition, not event rates.
_CHR_PROPORTION = [
    "adult_obesity",
    "children_in_poverty",
    "unemployment",
    "uninsured_adults",
    "access_to_healthy_foods",
    "low_birthweight",
    "poor_or_fair_health",
    "adult_smoking",
    "smoking_during_pregnancy",
    "physical_inactivity",
    "some_college",
    "driving_alone_to_work",
    "%_not_proficient_in_english",
    "diabetes_prevalence",
    "children_in_single-parent_households",
    "mammography_screening",
    "diabetes_monitoring",
    "excessive_drinking",
    "uninsured",
    "uninsured_children",
    "food_insecurity",
    "severe_housing_problems",
    "long_commute_-_driving_alone",
    "access_to_exercise_opportunities",
    "insufficient_sleep",
    "frequent_mental_distress",
    "frequent_physical_distress",
]

# 2e. Already per 100,000 — no new column needed.
_CHR_ALREADY_PER100K = [
    "violent_crime", "homicides", "motor_vehicle_crash_deaths",
    "motor_vehicle_crash_occupancy_rate",
    "on-road_motor_vehicle_crash-related_er_visits",
    "sexually_transmitted_infections", "hiv_prevalence",
    "premature_age-adjusted_mortality", "child_mortality", "injury_deaths",
]


def _make_chr_per100k(stem: str, factor: float) -> None:
    raw_col = f"{stem}{_CHR_SUFFIX}"
    out_col = f"{stem}_per100k"
    if raw_col not in merged_all.columns:
        print(f"  WARNING: {raw_col} not found — skipped.")
        return
    merged_all[out_col] = merged_all[raw_col] * factor


for _stem in _CHR_PROVIDER_RATIO:
    _make_chr_per100k(_stem, 100_000)
for _stem in _CHR_PER1000:
    _make_chr_per100k(_stem, 100)
for _stem in _CHR_PER10000:
    _make_chr_per100k(_stem, 10)
for _stem in _CHR_PROPORTION:
    _make_chr_per100k(_stem, 100_000)

_n_chr = len(_CHR_PROVIDER_RATIO) + len(_CHR_PER1000) + len(_CHR_PER10000) + len(_CHR_PROPORTION)
print(f"CHR rate standardization: {_n_chr} variables → *_per100k.")
print(f"  Provider ratios (×100,000): {_CHR_PROVIDER_RATIO}")
print(f"  Per-1,000   (×100): {_CHR_PER1000}")
print(f"  Per-10,000  (×10):  {_CHR_PER10000}")
print(f"  Proportions (×100,000): {len(_CHR_PROPORTION)} variables")
print(f"  Already per 100k (no copy): {_CHR_ALREADY_PER100K}")


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
