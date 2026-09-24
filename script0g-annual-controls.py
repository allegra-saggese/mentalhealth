#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script0g-annual-controls.py  --  download annual, correctly-dated county controls.

WHY THIS EXISTS
---------------
County Health Rankings dates every measure by its RELEASE year, not the year the
data describes. Confirmed against CHR's own release code and documentation:
`median_household_income`, `children_in_poverty` and `unemployment` are lagged 2
years; `uninsured_adults` is lagged 3. A panel row labelled 2020 therefore holds
2018 income and 2017 uninsured rates, while CAFO treatment is dated by the actual
ag-census year. This script replaces those four with series pulled straight from
the primary source, dated by the year the data actually describes.

SOURCES (all annual, all counties, no population floor)
-------------------------------------------------------
  SAIPE  Small Area Income and Poverty Estimates, Census
         -> median_household_income, children_in_poverty
         est{YY}all.dat  (2000-2002)  /  est{YY}all.xls  (2003+)
  LAUS   Local Area Unemployment Statistics, BLS
         -> unemployment rate
         download.bls.gov (www.bls.gov returns 403 to non-browser clients)
  SAHIE  Small Area Health Insurance Estimates, Census
         -> uninsured adults 18-64
         sahie-{YEAR}-csv.zip

Raw downloads -> Data/raw/annual_controls/<source>/
Cleaned panel -> Data/clean/{today}_annual_controls_county_year.csv
"""
import warnings; warnings.filterwarnings("ignore")
import os, io, re, zipfile, time
import numpy as np, pandas as pd, urllib.request

from packages import *
from functions import *

RAW  = os.path.join(db_data, "raw", "annual_controls")
for sub in ("saipe","laus","sahie","pep"): os.makedirs(os.path.join(RAW,sub), exist_ok=True)
CLEAN = os.path.join(db_data, "clean")
TODAY = date.today().strftime("%Y-%m-%d")
YEARS = range(2000, 2024)
UA = "Mozilla/5.0 (academic research; absagges@ucsc.edu)"

def fetch(url, dest, tries=3):
    """Download once and cache. Returns dest, or None if unavailable."""
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=90) as r, open(dest,"wb") as f:
                f.write(r.read())
            return dest
        except Exception as e:
            if k == tries-1:
                print(f"    MISS {os.path.basename(dest)}: {type(e).__name__}")
                return None
            time.sleep(2)

# =============================================================================
# SAIPE -- median household income + children in poverty
# =============================================================================
# Layout differs by vintage: fixed-width .dat before 2003, Excel after. Both are
# state+county files where county_fips == 0 marks the state total row, which is
# dropped. Column POSITIONS in the .dat files follow the published layout.
def saipe():
    print("SAIPE")
    frames=[]
    for y in YEARS:
        yy=f"{y%100:02d}"
        base=f"https://www2.census.gov/programs-surveys/saipe/datasets/{y}/{y}-state-and-county"
        got=None
        for ext in ("xls","dat","txt"):
            url=f"{base}/est{yy}all.{ext}"
            dest=os.path.join(RAW,"saipe",f"est{yy}all.{ext}")
            if fetch(url,dest): got=(dest,ext); break
        if not got:
            print(f"  {y}: unavailable"); continue
        dest,ext=got
        try:
            # Column positions verified against est10all.xls header row 2:
            #   0 State FIPS | 1 County FIPS | 7 Poverty Percent All Ages
            #   10 Poverty ESTIMATE Under 18 (a COUNT) | 13 Poverty PERCENT Under 18
            #   22 Median Household Income
            # Col 13, not 10 -- col 10 is the count and runs into the hundreds of
            # thousands, which is what made the first run's means nonsensical.
            if ext=="xls":
                d=pd.read_excel(dest,header=None,skiprows=3)
                st=d.iloc[:,0]; ct=d.iloc[:,1]
                mhi=pd.to_numeric(d.iloc[:,22],errors="coerce")
                cip=pd.to_numeric(d.iloc[:,13],errors="coerce")
            else:
                # Fixed-width .dat, 2000-2002. Whitespace-delimited in practice:
                # state, county, then the same field order as the .xls layout.
                raw=[l for l in open(dest,"r",errors="ignore").read().splitlines() if l.strip()]
                parts=[l.split() for l in raw]
                st =pd.Series([q[0] if len(q)>0 else None for q in parts])
                ct =pd.Series([q[1] if len(q)>1 else None for q in parts])
                cip=pd.to_numeric(pd.Series([q[11] if len(q)>11 else None for q in parts]),errors="coerce")
                mhi=pd.to_numeric(pd.Series([q[22] if len(q)>22 else None for q in parts]),errors="coerce")
            # FIPS: both files carry these as numbers or space-padded strings, so
            # coerce to int before zero-padding. zfill on "  1" leaves the spaces.
            st=pd.to_numeric(st,errors="coerce"); ct=pd.to_numeric(ct,errors="coerce")
            f=pd.DataFrame({"state":st,"county":ct,
                            "saipe_median_hh_income":mhi,"saipe_pct_children_poverty":cip})
            f=f.dropna(subset=["state","county"])
            f=f[f.county>0]                      # county 0 is the state total row
            f["fips"]=(f.state.astype(int).astype(str).str.zfill(2)
                       + f.county.astype(int).astype(str).str.zfill(3))
            f["year"]=y
            frames.append(f[["fips","year","saipe_median_hh_income","saipe_pct_children_poverty"]])
            print(f"  {y}: {len(f):,} counties")
        except Exception as e:
            print(f"  {y}: parse failed ({type(e).__name__}: {e})")
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

# =============================================================================
# LAUS -- county unemployment rate
# =============================================================================
# www.bls.gov returns 403 to scripted clients; download.bls.gov serves the same
# data. la.data.64.County is the full county time series; la.area maps the LAUS
# area code to a state+county FIPS. Annual averages are period M13.
def laus():
    print("LAUS")
    d1=fetch("https://download.bls.gov/pub/time.series/la/la.data.64.County",
             os.path.join(RAW,"laus","la.data.64.County"))
    d2=fetch("https://download.bls.gov/pub/time.series/la/la.series",
             os.path.join(RAW,"laus","la.series"))
    if not d1 or not d2:
        print("  unavailable"); return pd.DataFrame()
    dat=pd.read_csv(d1,sep="\t",dtype=str).rename(columns=lambda c:c.strip())
    ser=pd.read_csv(d2,sep="\t",dtype=str).rename(columns=lambda c:c.strip())
    dat["series_id"]=dat["series_id"].str.strip()
    ser["series_id"]=ser["series_id"].str.strip()
    # measure_code 03 = unemployment rate
    ser=ser[ser["measure_code"].str.strip()=="03"]
    m=dat.merge(ser[["series_id","area_code"]],on="series_id",how="inner")
    m=m[m["period"].str.strip()=="M13"]                      # annual average
    m["year"]=pd.to_numeric(m["year"],errors="coerce")
    m=m[m.year.between(min(YEARS),max(YEARS))]
    # LAUS county area codes look like CN{SSCCC}0000000000
    m["fips"]=m["area_code"].str.extract(r"^CN(\d{5})")
    m=m.dropna(subset=["fips"])
    m["laus_unemployment_rate"]=pd.to_numeric(m["value"].str.strip(),errors="coerce")
    out=m[["fips","year","laus_unemployment_rate"]].drop_duplicates(["fips","year"])
    print(f"  {len(out):,} county-years, {out.year.min()}-{out.year.max()}")
    return out

# =============================================================================
# SAHIE -- uninsured, adults 18-64
# =============================================================================
def sahie():
    print("SAHIE")
    frames=[]
    for y in YEARS:
        url=f"https://www2.census.gov/programs-surveys/sahie/datasets/time-series/estimates-acs/sahie-{y}-csv.zip"
        dest=os.path.join(RAW,"sahie",f"sahie-{y}.zip")
        if not fetch(url,dest): continue
        try:
            # The preamble length varies by vintage (79 lines in some years, 84-85 in
            # others), so locate the header row rather than hardcoding skiprows.
            with zipfile.ZipFile(dest) as z:
                name=[n for n in z.namelist() if n.lower().endswith(".csv")][0]
                lines=z.open(name).read().decode("latin-1").splitlines()
            # Must be the comma-delimited header, not the "PRIMARY KEY: year version
            # statefips countyfips ..." documentation line in the preamble, which also
            # contains both words and sits ~17 lines earlier.
            hdr=next(i for i,l in enumerate(lines)
                     if l.lower().lstrip().startswith("year,") and "countyfips" in l.lower())
            d=pd.read_csv(io.StringIO("\n".join(lines[hdr:])),dtype=str,low_memory=False)
            d.columns=[c.strip().lower() for c in d.columns]
            # agecat 1 = 18-64; all races, both sexes, all incomes
            for _c in ["agecat","racecat","sexcat","iprcat","statefips","countyfips"]:
                d[_c]=d[_c].astype(str).str.strip()
            # agecat 1 = ages 18-64; all races, both sexes, all income levels
            d=d[(d.agecat=="1")&(d.racecat=="0")&(d.sexcat=="0")&(d.iprcat=="0")]
            d=d[pd.to_numeric(d.countyfips,errors="coerce").fillna(0)>0]
            f=pd.DataFrame({"fips":d.statefips.str.zfill(2)+d.countyfips.str.zfill(3),
                            "year":y,
                            "sahie_pct_uninsured_18_64":pd.to_numeric(d.pctui,errors="coerce")})
            frames.append(f); print(f"  {y}: {len(f):,} counties")
        except Exception as e:
            print(f"  {y}: parse failed ({type(e).__name__}: {e})")
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()



# ---------------------------------------------------------------------------
# FINAL NAMING. The new series take over the EXACT column names the CHR versions
# used, so that no downstream code has to change. script1b drops the CHR columns
# before merging these in (see REPLACES below).
#
# UNITS: the panel's `*_per100k` columns store PERCENT x 1000 -- 8.9% is stored
# as 8900. Verified against the existing CHR columns (unemployment mean 6,363.6
# = 6.36%). SAIPE/LAUS/SAHIE publish plain percentages, so they are scaled by
# 1000 here. `median_household_income` is plain dollars in both and is not scaled.
# ---------------------------------------------------------------------------
REPLACES = {
    "saipe_median_hh_income":        ("median_household_income",      1),
    "saipe_pct_children_poverty":    ("children_in_poverty_per100k",  1000),
    "laus_unemployment_rate":        ("unemployment_per100k",         1000),
    "sahie_pct_uninsured_18_64":     ("uninsured_adults_per100k",     1000),
    # PEP shares are fractions (0-1); the panel stores these as plain percent
    # (mean %_hispanic ~0.08 in the panel), so scale by 1.
    "pep_pct_female":                ("%_female",                              1),
    "pep_pct_hispanic":              ("%_hispanic",                            1),
    "pep_pct_asian":                 ("%_asian",                               1),
    "pep_pct_nhpi":                  ("%_native_hawaiian/other_pacific_islander", 1),
    "pep_pct_65_plus":               ("%_65_and_older",                        1),
    "pep_pct_under_18":              ("%_below_18_years_of_age",               1),
}

def apply_final_names(out):
    """Rename to the panel's own column names and match its units. Keeps the
    source-named originals alongside so the provenance is visible in the file."""
    for src,(dst,scale) in REPLACES.items():
        if src in out.columns:
            out[dst] = out[src] * scale
    return out

# =============================================================================
# PEP -- county demographics, dated by TRUE data year
# =============================================================================
# Replaces the six CHR demographic shares, which are dated by CHR's RELEASE year
# and therefore carry a 2-year lag. Census PEP publishes county population by
# age x sex x race x Hispanic origin ANNUALLY for every county, with no
# population floor, so the lag is removable outright.
#
# THREE VINTAGES, each indexing YEAR as a code rather than a calendar year, and
# each indexing it differently. Verified empirically against known Alabama state
# totals (2010, 2012, 2015, 2018 all matched within 0.4%) rather than taken from
# the documentation:
#
#   co-est00int-alldata-{SS}   2000-2010   code 2..12 -> 2000..2010
#                              (code 1 = 4/1/2000 census, 13 = 4/1/2010 census)
#   CC-EST2020-ALLDATA-{SS}    2010-2020   code 3..13 -> 2010..2020
#                              (code 1 = 4/1/2010 census, 2 = estimates base)
#   cc-est2024-alldata         2020-2024   code 2..6  -> 2020..2024
#                              (code 1 = 4/1/2020 estimates base)
#
# VINTAGE OVERLAP: the 2010-2020 series drifted ~2% BELOW the 2020 census count
# by its final year and was never rebased, so the two vintages disagree at the
# seam. Per team decision, the NEWER vintage wins wherever they overlap. Because
# every variable here is a SHARE, a level error largely cancels between numerator
# and denominator -- but the 2010 and 2020 seams are still checked for jumps
# below, since a spurious break there would read as an event-study result.
#
# AGEGRP: 0 = all ages, 1..18 = 0-4, 5-9, ... 85+.
#   under 18  = AGEGRP 1,2,3 plus 3/5 of AGEGRP 4 (15-19) -- see note below
#   65+       = AGEGRP 14..18
# The 15-19 band straddles 18, so "% below 18" cannot be formed exactly from
# 5-year bands. CHR's own measure uses single-year ages. We take AGEGRP 1-3
# (0-14) plus 3/5 of the 15-19 band as a linear approximation, and record it.
PEP_STATES = [f"{i:02d}" for i in list(range(1,57))]
def pep():
    print("PEP demographics")
    VINT = [
        ("2000-2010", "https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-alldata-{ss}.csv", 1998, range(2,13)),
        ("2010-2020", "https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/asrh/CC-EST2020-ALLDATA-{ss}.csv", 2007, range(3,14)),
        ("2020-2024", "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata.csv", 2018, range(2,7)),
    ]
    frames=[]
    for vname, tmpl, offset, codes in VINT:
        parts=[]
        targets = ["NATIONAL"] if "{ss}" not in tmpl else PEP_STATES
        for ss in targets:
            url = tmpl if ss=="NATIONAL" else tmpl.format(ss=ss)
            dest = os.path.join(RAW,"pep", f"{vname}_{ss}.csv")
            if not fetch(url,dest): continue
            try:
                d=pd.read_csv(dest,encoding="latin-1",low_memory=False)
            except Exception as e:
                print(f"    {vname} {ss}: {type(e).__name__}"); continue
            d.columns=[c.upper().strip() for c in d.columns]
            if "AGEGRP" not in d.columns: continue
            d=d[d["YEAR"].isin(list(codes))]
            if "SUMLEV" in d.columns: d=d[d["SUMLEV"]==50]
            parts.append(d)
        if not parts:
            print(f"  {vname}: no files"); continue
        D=pd.concat(parts,ignore_index=True)
        D["year"]=D["YEAR"]+offset
        D["fips"]=(D["STATE"].astype(int).astype(str).str.zfill(2)
                   + D["COUNTY"].astype(int).astype(str).str.zfill(3))
        # AGEGRP CODING DIFFERS BY VINTAGE -- caught by the seam check below.
        #   co-est00int (2000-2010): AGEGRP 0 = AGE 0 (infants), 99 = TOTAL,
        #                            1 = 1-4, 2 = 5-9, 3 = 10-14, 4 = 15-19 ...
        #   CC-EST2020 / cc-est2024: AGEGRP 0 = TOTAL,
        #                            1 = 0-4, 2 = 5-9, 3 = 10-14, 4 = 15-19 ...
        # Using 0 as the denominator for the 2000-2010 files divided by the infant
        # count instead of the population -- ~72x too small, and inf where zero.
        # 65+ is AGEGRP 14-18 in BOTH codings. Under-18 differs: the older vintage
        # needs AGEGRP 0 included, the newer does not.
        TOTAL_CODE = 99 if D["AGEGRP"].max() == 99 else 0
        U18_LOW    = 0  if TOTAL_CODE == 99 else 1
        tot = D[D.AGEGRP==TOTAL_CODE].set_index(["fips","year"])
        num = lambda f,c: pd.to_numeric(f[c],errors="coerce")
        base = num(tot,"TOT_POP")
        out = pd.DataFrame(index=tot.index)
        out["pep_pct_female"]   = num(tot,"TOT_FEMALE")/base
        out["pep_pct_hispanic"] = (num(tot,"H_MALE")+num(tot,"H_FEMALE"))/base
        out["pep_pct_asian"]    = (num(tot,"AA_MALE")+num(tot,"AA_FEMALE"))/base
        out["pep_pct_nhpi"]     = (num(tot,"NA_MALE")+num(tot,"NA_FEMALE"))/base
        g = D[D.AGEGRP.between(0,18)].copy()
        g["_pop"]=pd.to_numeric(g["TOT_POP"],errors="coerce")
        o65 = g[g.AGEGRP.between(14,18)].groupby(["fips","year"])["_pop"].sum()
        u15 = g[g.AGEGRP.between(U18_LOW,3)].groupby(["fips","year"])["_pop"].sum()
        b1519 = g[g.AGEGRP==4].groupby(["fips","year"])["_pop"].sum()
        out["pep_pct_65_plus"] = (o65/base).reindex(out.index)
        out["pep_pct_under_18"] = ((u15 + 0.6*b1519)/base).reindex(out.index)
        out=out.reset_index(); out["_vintage"]=vname
        frames.append(out)
        print(f"  {vname}: {out.fips.nunique():,} counties, {int(out.year.min())}-{int(out.year.max())}")
    if not frames: return pd.DataFrame()
    P=pd.concat(frames,ignore_index=True)
    # NEWER VINTAGE WINS on overlap: sort so the latest vintage lands last, keep last.
    order={v:i for i,(v,_,_,_) in enumerate(VINT)}
    P["_ord"]=P["_vintage"].map(order)
    P=P.sort_values(["fips","year","_ord"]).drop_duplicates(["fips","year"],keep="last")
    P=P.drop(columns=["_ord","_vintage"])
    # seam check -- a jump at 2010 or 2020 would read as a spurious event
    for seam in (2010,2020):
        a=P[P.year==seam-1].set_index("fips")["pep_pct_65_plus"]
        b=P[P.year==seam].set_index("fips")["pep_pct_65_plus"]
        j=(b-a.reindex(b.index)).abs().median()
        print(f"  seam {seam-1}->{seam}: median |change| in pct_65_plus = {j:.5f}")
    return P


if __name__ == "__main__":
    print("="*72); print("script0g -- annual county controls, dated by TRUE data year"); print("="*72)
    S,L,H,P = saipe(), laus(), sahie(), pep()
    out=None
    for f in (S,L,H,P):
        if f is None or f.empty: continue
        out = f if out is None else out.merge(f,on=["fips","year"],how="outer")
    if out is None or out.empty:
        raise SystemExit("nothing downloaded")
    out["fips"]=out["fips"].astype(str).str.zfill(5)
    out=apply_final_names(out)
    out=out.sort_values(["fips","year"])
    p=os.path.join(CLEAN,f"{TODAY}_annual_controls_county_year.csv")
    out.to_csv(p,index=False)
    print("\n"+"="*72)
    print(f"rows {len(out):,} | counties {out.fips.nunique():,} | years {out.year.min()}-{out.year.max()}")
    for c in out.columns:
        if c in ("fips","year"): continue
        print(f"  {c:32s} coverage {out[c].notna().mean():6.1%}")
    print(f"\nSaved: {p}")
