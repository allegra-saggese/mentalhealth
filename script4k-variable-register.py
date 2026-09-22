#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4k-variable-register.py  --  the master variable register.

Generates one row per column in the analysis panel, carrying every decision and
diagnostic accumulated in code-audit/plans/CONTROL-DECISION-REGISTER.md so that
each inclusion and exclusion can be justified to a referee from a single file.

Replaces `250710-codebook-fillout.xlsx`, which was built against an earlier version
of the data: only 3 of its 213 rows match the current panel, and none of the 19
controls or 7 outcomes appear in it.

Output -> Dropbox/Mental/Data/output/codebooks/  (xlsx + csv)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from script4_treatment import (load_panel, OUTCOMES, TREATMENTS, CORE_TREATMENTS,
    CONTROL_PRETREAT, CONTROL_COMPREHENSIVE, CONTROL_CORE9, CONTROL_MEDIATORS,
    tables_dir, db_data, os, date)

TODAY = date.today().strftime("%Y-%m-%d")
# Codebooks live in their own folder, not with the result tables -- the folder is
# the running record of every variable listing made for this project.
OUT = os.path.join(db_data, "output", "codebooks")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------
M = {
"%_65_and_older":               dict(v="v053",src="Census PEP",yrs="2023",lag=2,tier="A",role="confounder",ptest="Y",pres="flat (0/5 sig)",why="Age structure predicts agricultural land use and mental-health reporting; not moved by a CAFO."),
"%_asian":                      dict(v="v081",src="Census PEP",yrs="2023",lag=2,tier="A",role="confounder",ptest="Y",pres="flat (0/5 sig)",why="Predetermined composition. Near-zero in non-metro; adds little."),
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
"%_rural":"D-03 drop (pending removal)","access_to_healthy_foods_per100k":"D-05 drop (decided)",
"adult_smoking_per100k":"D-04 collider, removal candidate (open)","teen_births_per100k":"D-04 collider (open) / D-12 keep CHR pooled (decided)",
}
for c in CONTROL_MEDIATORS: DECISION.setdefault(c,"D-01 excluded as mediator")
INTERACTS = {
"%_hispanic":"corr -0.775 with log_pop (within). Used in dairy x %hispanic interaction in the ML block -- that interaction is hard to read cleanly because of this.",
"median_household_income":"VIF 5.35 raw / 1.19 within; the largest raw-level VIF in the set.",
"children_in_poverty_per100k":"VIF 5.35 raw / 1.17 within.",
}


# ---------------------------------------------------------------------------
# SALVAGED from 250710-codebook-fillout.xlsx (the superseded hand-filled codebook).
# Only 3 of its 213 names matched the current panel, but its NIBRS offence-code
# definitions are authoritative and worth carrying forward. The crime-against-a-
# PERSON vs crime-against-SOCIETY split is the principled basis for which offences
# belong in a violence measure -- note DUI is against SOCIETY, not a person.
# Also salvaged: two "not in NIBRS" notes. NOT salvaged: its warning that
# "population does not change for different years" -- verified fixed, population
# now has 24 distinct values per county across 24 years (within-SD share 0.089).
# ---------------------------------------------------------------------------
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
NOT_IN_NIBRS = ["child_molestation", "human_trafficking (as a single category)"]

def nibrs_note(col):
    """Attach the offence-code definition to every derived column of that offence."""
    for stem, desc in NIBRS_CODES.items():
        if col == stem or col.startswith(stem + "_"):
            return desc
    return ""


# ---------------------------------------------------------------------------
# SOURCE FILE MAP -- traces each panel column back to the clean-layer file it
# came from, and from there to the original raw file. Built by matching column
# names against the headers of the files in Data/clean/.
# ---------------------------------------------------------------------------
import json as _json, glob as _glob
_CLEAN = os.path.join(db_data, "clean")
_SRC = {
 "cafo_ops_by_size_compact":        ("script0b", "USDA NASS Quick Stats API (census years only)"),
 "cdc_county_year_deathsofdespair": ("script0c", "CDC WONDER: cty-level-deathsofdespair-YYYY.csv"),
 "crime_fips_level_final":          ("script0d", "FBI NIBRS: nibrs_arrestee_segment_YYYY.csv (ARRESTS)"),
 "fips_full":                       ("script0a", "Census FIPS crosswalk"),
 "mentalhealthrank_full":           ("script0c", "County Health Rankings: analytic_dataYYYY.csv"),
 "population_full":                 ("script0a", "Census Population Estimates Program"),
 "rural-key":                       ("script0e", "NCHS: NCHSurb-rural-codes.csv"),
 "fsis_county_year_fips":           ("script0f", "USDA FSIS establishment file (FOIA) + HUD ZIP crosswalk"),
}
def _build_source_map():
    """Exact header match first; then stem-matching, because script1b renames columns
    during the merge (CHR arrives as `<measure>_raw_value`, panel keeps `<measure>` or
    `<measure>_per100k`). Falls back to a rule on the name itself."""
    out, stems = {}, {}
    for stem, (scr, raw) in _SRC.items():
        hits = sorted(_glob.glob(os.path.join(_CLEAN, f"*{stem}*.csv")))
        if not hits: continue
        f = hits[-1]
        try: cols = list(pd.read_csv(f, nrows=1, low_memory=False).columns)
        except Exception: continue
        for c in cols:
            out.setdefault(c, (os.path.basename(f), scr, raw))
        # index the clean file's measure stems so renamed panel columns still match
        for c in cols:
            base = (c.replace("_raw_value","").replace("_numerator","")
                     .replace("_denominator","").replace("_ci_high","").replace("_ci_low",""))
            stems.setdefault(base, (os.path.basename(f), scr, raw))
    return out, stems

SOURCE_EXACT, SOURCE_STEMS = None, None
def source_of(col):
    """Return (clean_file, script, raw_source) for a panel column."""
    if col in SOURCE_EXACT: return SOURCE_EXACT[col]
    base = col.replace("_per100k","")
    if base in SOURCE_STEMS: return SOURCE_STEMS[base]
    if col in SOURCE_STEMS:  return SOURCE_STEMS[col]
    lc = col.lower()
    if lc.endswith("_fsis") or "fsis" in lc:
        return ("*_fsis_county_year_...csv","script0f","USDA FSIS establishment file (FOIA) + HUD ZIP crosswalk")
    if "cafo" in lc or lc in ("large_cafo","medium_cafo","small_cafo","bin_sum_cafo","total_heads_cafo"):
        return ("*_cafo_ops_by_size_compact.csv","script0b","USDA NASS Quick Stats API (census years only)")
    if lc.startswith("tr_") or lc.startswith("cond_") or lc in ("cohort","t_rel","add_year","entry_year","build_year","consol_year","log_pop"):
        return ("(derived in script4_treatment.py)","script4_treatment","derived from the USDA NASS CAFO block")
    if any(k in lc for k in ["assault","rape","incest","intimidation","kidnap","fondling",
                              "human_traffic","statutory","incident","driving_under","sexual_assault","crime"]):
        return ("*_crime_fips_level_final.csv","script0d","FBI NIBRS: nibrs_arrestee_segment_YYYY.csv (ARRESTS)")
    if "despair" in lc or lc.startswith("crude_rate") or lc in ("deaths","deaths_is_zero","cdc_in_query","homicides"):
        return ("*_cdc_county_year_deathsofdespair.csv","script0c","CDC WONDER: cty-level-deathsofdespair-YYYY.csv")
    if lc in ("rural","non_large_metro","nchs_code","nchs_label"):
        return ("*-rural-key.csv","script0e","NCHS: NCHSurb-rural-codes.csv")
    if "pop" in lc:
        return ("*_population_full.csv","script0a","Census Population Estimates Program")
    if lc in ("fips","state","county","state_fips","county_fips","state_abbrev","state_abbreviation","name","state_code"):
        return ("*_fips_full.csv","script0a","Census FIPS crosswalk")
    return ("","","")
SOURCE_EXACT, SOURCE_STEMS = _build_source_map()

# ---------------------------------------------------------------------------
# INTERPOLATION / AUGMENTATION -- what was done to a variable beyond reading it.
# This is the part a referee will ask about, so it is stated per variable.
# ---------------------------------------------------------------------------
CAFO_AUG = ("FORWARD-FILLED from ag-census years (2002/2007/2012/2017/2022) to all "
            "intervening years by script1b, AND zero-filled where a rural county is "
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
 "crude_rate_from_census_pop":"CONSTRUCTED in script1b: deaths / census population x 100,000. Used instead of CDC's own crude rate because CDC suppresses it below 10 deaths. Truncated after 2020 (source coverage falls to zero).",
 "total_incidents_per100k":"CONSTRUCTED in script0d: sum of a CURATED 12-offence list, per 100k. Not all crime -- no property or drug offences.",
}
for _c in ["cafo_dairy_small","cafo_dairy_medium","cafo_dairy_large","cafo_dairy_total",
           "cafo_hogs_large","cafo_beef_large","cafo_cattle_large","cafo_chickens_large"]:
    AUG.setdefault(_c, CAFO_AUG)

# ---------------------------------------------------------------------------
# WHY a variable is NOT used as a control.
# ---------------------------------------------------------------------------
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

df = load_panel()
ref = df[df["poor_mental_health_days"].notna()]
est = df[["fips","year"]+[c for c in CONTROL_PRETREAT if c in df.columns]].dropna()

def role_of(c):
    if c in OUTCOMES.values(): return "OUTCOME"
    if c in {t[0] for t in TREATMENTS.values()} or c.startswith("tr_"): return "TREATMENT"
    if c in CONTROL_PRETREAT: return "CONTROL (in use)"
    if c in CONTROL_COMPREHENSIVE: return "CONTROL (comprehensive only)"
    if c in {"fips","year","state_fips","rural","population","log_pop"}: return "IDENTIFIER"
    return "not used"

rows=[]
for c in df.columns:
    s=df[c]; num=pd.api.types.is_numeric_dtype(s)
    yrs=df.loc[s.notna(),"year"]
    meta=M.get(c,{}); om=OUTCOME_META.get(c,{})
    r=role_of(c)
    is_cov = "Y" if c in CONTROL_PRETREAT else ("comprehensive only" if c in CONTROL_COMPREHENSIVE else "N")
    wshare=np.nan
    if c in est.columns and num and est[c].std()>0:
        w=est[c]-est.groupby("fips")[c].transform("mean"); wshare=w.std()/est[c].std()
    inv = [k for k,v in OUTCOMES.items() if v==c]
    rows.append({
      "colname":c,
      "role":r,
      "description":meta.get("why","") if r.startswith("CONTROL") else (om.get("note","") if r=="OUTCOME" else ""),
      "outcome_label":inv[0] if inv else "",
      "type":str(s.dtype),
      "source":meta.get("src", om.get("src","")),
      "chr_measure_code":meta.get("v", om.get("v","")),
      "data_years_TRUE":meta.get("yrs", om.get("yrs","")),
      "year_lag_release_minus_data":meta.get("lag", om.get("lag","")),
      "source_tier":meta.get("tier",""),
      "covariate_YN":is_cov,
      "covar_possibility":meta.get("role",""),
      "covar_rationale":meta.get("why",""),
      "pretrend_testable":meta.get("ptest",""),
      "pretrend_result":meta.get("pres",""),
      "in_A1_pooled":"Y" if c in CONTROL_PRETREAT else "",
      "in_A2_within":"Y" if c in CONTROL_PRETREAT else "",
      "in_A7_event_study":"Y" if c in CONTROL_PRETREAT else "",
      "in_A18_CS_with_cov":"Y" if c in CONTROL_PRETREAT else "",
      "in_CORE9_legacy":"Y" if c in CONTROL_CORE9 else "",
      "in_PRETREAT_19":"Y" if c in CONTROL_PRETREAT else "",
      "in_COMPREHENSIVE_26":"Y" if c in CONTROL_COMPREHENSIVE else "",
      "nibrs_offence_code":nibrs_note(c),
      "interactions_collinearity":INTERACTS.get(c,""),
      "decision_id":DECISION.get(c,""),
      "pct_complete_full_panel":float(s.notna().mean()),
      "pct_complete_outcome_rows":float(ref[c].notna().mean()) if c in ref.columns else np.nan,
      "first_year":int(yrs.min()) if len(yrs) else None,
      "last_year":int(yrs.max()) if len(yrs) else None,
      "within_sd_share":wshare,
      "mean":float(s.mean()) if num else np.nan,
      "sd":float(s.std()) if num else np.nan,
      "min":float(s.min()) if num else np.nan,
      "max":float(s.max()) if num else np.nan,
      "coverage_pct":round(100*float(s.notna().mean()),1),
      "coverage_pct_on_outcome_rows":round(100*float(ref[c].notna().mean()),1) if c in ref.columns else np.nan,
      "why_not_used":EXCLUSION.get(c,""),
      "source_file_clean":source_of(c)[0],
      "source_script":source_of(c)[1],
      "source_raw_original":source_of(c)[2],
      "interpolation_augmentation":AUG.get(c,""),
      "notes":"",
    })
R=pd.DataFrame(rows)
ORDER={"OUTCOME":0,"TREATMENT":1,"CONTROL (in use)":2,"CONTROL (comprehensive only)":3,"IDENTIFIER":4,"not used":5}
R["_o"]=R.role.map(ORDER).fillna(9); R=R.sort_values(["_o","colname"]).drop(columns="_o")

# ---------------------------------------------------------------------------
# FSIS sheet. The FSIS columns live in a SEPARATE panel (`*_panel_fsis.csv`,
# 2017+ only) rather than the main panel, so that merging them would not truncate
# the full 2000-2023 time series. They are documented here on their own tab
# rather than being merged in, because their coverage window is different.
# ---------------------------------------------------------------------------
from functions import latest_file_glob
FSIS_NOTE = ("FSIS coverage is 2017-2023 ONLY, at roughly 46-50% of rural counties per "
             "year. Lives in *_panel_fsis.csv, NOT the main panel, to avoid truncating "
             "the 2000-2023 series. Any analysis using an _fsis column must load that "
             "file. The dairy x FSIS interaction (A6) is flagged DO NOT PRESENT: "
             "VIF 34.3 on %_rural in its thin sample.")
try:
    _fp = latest_file_glob(os.path.join(db_data, "merged"), "*_panel_fsis.csv")
    _f = pd.read_csv(_fp, low_memory=False)
    _main = set(df.columns)
    frows=[]
    for c in _f.columns:
        fs=_f[c]; num=pd.api.types.is_numeric_dtype(fs)
        yrs=_f.loc[fs.notna(),"year"] if "year" in _f.columns else pd.Series(dtype=float)
        frows.append({"colname":c,
            "in_main_panel":"Y" if c in _main else "N (FSIS panel only)",
            "is_fsis_column":"Y" if c.endswith("_fsis") else "",
            "type":str(fs.dtype),
            "coverage_pct":round(100*float(fs.notna().mean()),1),
            "first_year":int(yrs.min()) if len(yrs) else None,
            "last_year":int(yrs.max()) if len(yrs) else None,
            "source_file_clean":source_of(c)[0],
            "source_script":source_of(c)[1],
            "source_raw_original":source_of(c)[2],
            "mean":float(fs.mean()) if num else np.nan,
            "sd":float(fs.std()) if num else np.nan,
            "notes":FSIS_NOTE if c.endswith("_fsis") else "",
        })
    FS=pd.DataFrame(frows)
    FS["_o"]=(FS.is_fsis_column!="Y").astype(int); FS=FS.sort_values(["_o","colname"]).drop(columns="_o")
    fsis_src=os.path.basename(_fp)
except Exception as _e:
    FS=pd.DataFrame([{"colname":"<FSIS panel not found>","notes":str(_e)}]); fsis_src="n/a"

README=pd.DataFrame({"item":[
  "Generated by","Source panel","FSIS panel","Rows","Purpose",
  "why_not_used","coverage_pct","source_file_clean / source_raw_original",
  "interpolation_augmentation","pretrend_testable","source_tier","decision_id"],
 "detail":[
  "script4k-variable-register.py (rerun to regenerate)",
  os.path.basename(df.attrs["panel_path"]),
  fsis_src,
  f"{len(R)} main panel columns; {len(FS)} FSIS panel columns",
  "One row per variable, carrying every inclusion/exclusion decision so each can be justified to a referee.",
  "WHY a variable is not used as a control, with its decision ID.",
  "Share of rows with a non-missing value. coverage_pct_on_outcome_rows restricts to rows with a non-missing Poor MH Days.",
  "Traces the variable to the Data/clean file it came from and to the original raw source.",
  "What was done to the variable beyond reading it -- forward-fill, zero-fill, derivation formula.",
  "N where the series is pooled or a multi-year rolling estimate: a pre-trend CANNOT be meaningfully tested on it.",
  "A = re-source at true annual data year; B = multi-year, keep with restrictions; C = drop.",
  "Links to code-audit/plans/CONTROL-DECISION-REGISTER.md",
 ]})

xl=os.path.join(OUT,f"{TODAY}_VARIABLE_REGISTER.xlsx")
with pd.ExcelWriter(xl,engine="openpyxl") as w:
    README.to_excel(w,sheet_name="README",index=False)
    R.to_excel(w,sheet_name="variables",index=False)
    FS.to_excel(w,sheet_name="FSIS_panel",index=False)
    for name,frame in [("README",README),("variables",R),("FSIS_panel",FS)]:
        ws=w.sheets[name]; ws.freeze_panes="B2"
        for i,c in enumerate(frame.columns,1):
            width=max(len(str(c)), int(frame[c].astype(str).str.len().max() or 0))
            ws.column_dimensions[ws.cell(1,i).column_letter].width=min(max(width+2,12),70)
R.to_csv(os.path.join(OUT,f"{TODAY}_VARIABLE_REGISTER.csv"),index=False)
print(f"rows: {len(R)}")
print(R.role.value_counts().to_string())
print(f"\nSaved: {xl}")
