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
for sub in ("saipe","laus","sahie"): os.makedirs(os.path.join(RAW,sub), exist_ok=True)
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
}

def apply_final_names(out):
    """Rename to the panel's own column names and match its units. Keeps the
    source-named originals alongside so the provenance is visible in the file."""
    for src,(dst,scale) in REPLACES.items():
        if src in out.columns:
            out[dst] = out[src] * scale
    return out

if __name__ == "__main__":
    print("="*72); print("script0g -- annual county controls, dated by TRUE data year"); print("="*72)
    S,L,H = saipe(), laus(), sahie()
    out=None
    for f in (S,L,H):
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
