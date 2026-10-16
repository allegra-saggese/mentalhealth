#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4s-species-overlap.py  --  HOW MANY COUNTIES HAVE LARGE CAFOs OF MORE THAN ONE SPECIES?
(added 2026-10-15, team feedback)

QUESTION
--------
Dairy is the treatment in Part A. If the same counties also host large hog,
cattle or chicken operations, a dairy-only model attributes their effects to
dairy. This script counts how often that happens, by year, so the team can decide
whether to model species jointly. DESCRIPTIVE ONLY: no regressions.

DEFINITIONS
-----------
Species      dairy, beef, hogs, chickens    (the four modelled)
             cattle is NOT a species here: USDA's cattle inventory contains the
             dairy and beef herds, so cattle overlaps dairy and beef by
             construction. Counting it would make nearly every dairy county look
             multi-species. It appears only in the appendix nesting check.
Present      cafo_{species}_large > 0 in the county-year. "Large" is the USDA NASS
             top size bin for that species (500+ milk cows for dairy).
Missing      a missing large-CAFO count is treated as ZERO (no large operation),
             not as a dropped observation. Every county-year stays in every table.
             The number of values recoded this way, by species, is written to
             <date>_species_nan_as_zero.csv so the recode is visible.

CENSUS YEARS VS ALL YEARS
-------------------------
CAFO counts come from the ag census (2002, 2007, 2012, 2017, 2022) and are
forward-filled to the years between (script1b). Years between census waves are
therefore copies of the last wave, not new observations. The pair and
combination tables use census years only so every row is independent evidence.
The by-year count table covers every panel year, flagged `census_year`, because
the regression sample is county-years.

OUTPUTS  ->  Data/output/tables/script4s/   and   Data/output/figs/script4s/
    <date>_species_count_by_year.csv       counties with 0,1,2,3,4 species, every year
    <date>_species_combination_census.csv  exact species combinations, census years
    <date>_species_pair_census.csv         pairwise co-occurrence and P(B | A), census years
    <date>_species_county_summary.csv      county-level: ever / persistently multi-species
    <date>_species_nan_as_zero.csv         how many missing counts were set to zero, by species and year
    <date>_species_cattle_check.csv        appendix: cattle contains dairy and beef
    <date>_F1_species_count_census.png     stacked bars, census years
"""
import warnings; warnings.filterwarnings("ignore")
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from packages import db_data, figs_dir, tables_dir, os
from functions import latest_file_glob
from datetime import date

OUT_TABLES = os.path.join(tables_dir, "script4s"); os.makedirs(OUT_TABLES, exist_ok=True)
OUT_FIGS   = os.path.join(figs_dir,   "script4s"); os.makedirs(OUT_FIGS,   exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")

SPECIES        = ["dairy", "beef", "hogs", "chickens"]
ALL_COLUMNS    = SPECIES + ["cattle"]      # cattle is read only for the appendix check
CENSUS_YEARS   = [2002, 2007, 2012, 2017, 2022]


# =============================================================================
# Load: same panel and same rural sample as every script4 file
# =============================================================================
path = latest_file_glob(os.path.join(db_data, "merged"), "*_panel.csv")
panel = pd.read_csv(path, low_memory=False)
panel["fips"] = panel["fips"].astype(str).str.zfill(5)
assert (panel["rural"] == 1).all(), "panel is expected to be pre-filtered to rural counties"
print(f"panel: {os.path.basename(path)}  {len(panel):,} rows, {panel.fips.nunique():,} counties, "
      f"{int(panel.year.min())}-{int(panel.year.max())}")

large = {s: panel[f"cafo_{s}_large"] for s in ALL_COLUMNS}


def presence(species):
    """0/1 frame: does the county-year have >=1 large operation of each species?
    A missing count is a zero (no large operation); no county-year is dropped."""
    cols = pd.concat([large[s].rename(s) for s in species], axis=1)
    return panel[["fips", "year"]].copy(), (cols.fillna(0) > 0).astype(int)


# How many missing counts the zero rule recodes, so the choice is auditable.
nan_rows = (panel[["year"]].join(pd.concat([large[s].isna().rename(s) for s in ALL_COLUMNS], axis=1))
            .groupby("year").sum().reset_index())
nan_rows.insert(1, "n_county_years", panel.groupby("year").size().values)
nan_rows.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_species_nan_as_zero.csv"), index=False)


# =============================================================================
# 1. Counties by number of species present, every year
# =============================================================================
keys, pres = presence(SPECIES)
k = pres.sum(axis=1)
by_year = (pd.crosstab(keys["year"], k)
             .reindex(columns=range(len(SPECIES) + 1), fill_value=0))
by_year.columns = [f"n_counties_{c}_species" for c in by_year.columns]
by_year["n_county_years"] = by_year.sum(axis=1)
by_year["n_multi_species"] = by_year[[f"n_counties_{c}_species" for c in range(2, len(SPECIES) + 1)]].sum(axis=1)
by_year["share_multi_species"] = by_year["n_multi_species"] / by_year["n_county_years"]
by_year["census_year"] = by_year.index.isin(CENSUS_YEARS)
by_year = by_year.reset_index().rename(columns={"index": "year"})
by_year.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_species_count_by_year.csv"), index=False)
n_recoded = int(sum(large[s].isna().sum() for s in SPECIES))
print(f"\nspecies tracked: {SPECIES}   missing counts set to zero: {n_recoded:,} of "
      f"{len(panel) * len(SPECIES):,} species-county-years")
print(by_year[by_year.census_year].to_string(index=False))


# =============================================================================
# 2. Exact combinations, census years
# =============================================================================
keys["combo"] = pres.apply(lambda r: "+".join(s for s in SPECIES if r[s]) or "none", axis=1)
census_mask = keys["year"].isin(CENSUS_YEARS)
combo = (pd.crosstab(keys.loc[census_mask, "combo"], keys.loc[census_mask, "year"])
           .assign(total=lambda d: d.sum(axis=1))
           .sort_values("total", ascending=False).reset_index())
combo.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_species_combination_census.csv"), index=False)
print("\nExact combinations, census years:")
print(combo.to_string(index=False))


# =============================================================================
# 3. Pairwise co-occurrence, census years
# =============================================================================
# n_both: counties with large operations of BOTH species.
# p_b_given_a: of counties with species A, the share that also has B. The row
# that matters for the dairy question is a = dairy.
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
pairs.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_species_pair_census.csv"), index=False)
dp = pairs[pairs.species_a == "dairy"][["year", "species_b", "n_a", "n_both", "p_b_given_a"]]
print("\nOf counties with a large dairy, share that also have species B:")
print(dp.pivot(index="year", columns="species_b", values="p_b_given_a").round(3).to_string())


# =============================================================================
# 4. County-level summary
# =============================================================================
# 'ever'      >=2 species in at least one census wave
# 'persistent' >=2 species in at least two census waves
cens = keys[census_mask].assign(k=k[census_mask.values])
multi_waves = (cens.assign(multi=cens.k >= 2).groupby("fips")["multi"].sum())
dairy_waves = (pres.loc[census_mask.values, "dairy"].groupby(keys.loc[census_mask, "fips"]).sum())
summary = pd.DataFrame({
    "metric": ["counties in the panel (census-year rows)",
               "counties with >=2 species in >=1 census wave",
               "counties with >=2 species in >=2 census waves",
               "counties with large dairy in >=1 census wave",
               "  of those, with another species in the same wave at least once"],
    "n": [int(multi_waves.size), int((multi_waves >= 1).sum()), int((multi_waves >= 2).sum()),
          int((dairy_waves >= 1).sum()),
          int(((dairy_waves >= 1) & (multi_waves >= 1)).sum())]})
summary.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_species_county_summary.csv"), index=False)
print("\nCounty-level summary:"); print(summary.to_string(index=False))


# =============================================================================
# 5. Appendix: why cattle is not a species
# =============================================================================
# Cattle should contain both dairy and beef: dairy > 0 or beef > 0 with
# cattle == 0 should be (near) zero. Reported so the nesting is documented rather
# than asserted. "cattle only" = large cattle operations with neither a large
# dairy nor a large beef herd (e.g. feedlot or stocker operations).
_, pc = presence(ALL_COLUMNS)
cattle_rows = []
for yr in CENSUS_YEARS:
    p = pc[(keys["year"] == yr).values]
    cattle_rows.append({"year": yr, "n_county_years": len(p),
                        "n_cattle": int(p["cattle"].sum()),
                        "n_dairy": int(p["dairy"].sum()), "n_beef": int(p["beef"].sum()),
                        "n_dairy_without_cattle": int(((p["dairy"] == 1) & (p["cattle"] == 0)).sum()),
                        "n_beef_without_cattle": int(((p["beef"] == 1) & (p["cattle"] == 0)).sum()),
                        "n_cattle_without_dairy_or_beef": int(((p["cattle"] == 1) & (p["dairy"] == 0) & (p["beef"] == 0)).sum())})
cattle = pd.DataFrame(cattle_rows)
cattle.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_species_cattle_check.csv"), index=False)
print("\nAppendix, cattle nesting:"); print(cattle.to_string(index=False))


# =============================================================================
# Figure
# =============================================================================
cy = by_year[by_year.census_year].set_index("year")
cols = [f"n_counties_{c}_species" for c in range(len(SPECIES) + 1)]
fig, ax = plt.subplots(figsize=(7, 4))
bottom = np.zeros(len(cy))
for c, lab in zip(cols, [f"{i} species" for i in range(len(SPECIES) + 1)]):
    ax.bar(cy.index.astype(str), cy[c], bottom=bottom, label=lab)
    bottom += cy[c].values
ax.set_ylabel("rural counties")
ax.set_title("Large-CAFO species present per county, census years")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fp = os.path.join(OUT_FIGS, f"{TODAY}_F1_species_count_census.png")
fig.savefig(fp, dpi=150); plt.close(fig)

print(f"\nscript4s COMPLETE\n  tables -> {OUT_TABLES}\n  figs   -> {fp}")
