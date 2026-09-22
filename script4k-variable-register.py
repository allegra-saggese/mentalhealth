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

Output -> Dropbox/Mental/Data/output/tables/  (xlsx + csv)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from script4_treatment import (load_panel, OUTCOMES, TREATMENTS, CORE_TREATMENTS,
    CONTROL_PRETREAT, CONTROL_COMPREHENSIVE, CONTROL_CORE9, CONTROL_MEDIATORS,
    tables_dir, db_data, os, date)

TODAY = date.today().strftime("%Y-%m-%d")
OUT   = os.path.join(tables_dir)
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
      "notes":"",
    })
R=pd.DataFrame(rows)
ORDER={"OUTCOME":0,"TREATMENT":1,"CONTROL (in use)":2,"CONTROL (comprehensive only)":3,"IDENTIFIER":4,"not used":5}
R["_o"]=R.role.map(ORDER).fillna(9); R=R.sort_values(["_o","colname"]).drop(columns="_o")

xl=os.path.join(OUT,f"{TODAY}_VARIABLE_REGISTER.xlsx")
with pd.ExcelWriter(xl,engine="openpyxl") as w:
    R.to_excel(w,sheet_name="variables",index=False)
    ws=w.sheets["variables"]; ws.freeze_panes="B2"
    for i,c in enumerate(R.columns,1):
        width=max(len(str(c)), int(R[c].astype(str).str.len().max() or 0))
        ws.column_dimensions[ws.cell(1,i).column_letter].width=min(max(width+2,12),70)
R.to_csv(os.path.join(OUT,f"{TODAY}_VARIABLE_REGISTER.csv"),index=False)
print(f"rows: {len(R)}")
print(R.role.value_counts().to_string())
print(f"\nSaved: {xl}")
