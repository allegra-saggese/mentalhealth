#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 01:10:00 2025

@author: allegrasaggese
"""

from packages import *
from functions import * 
import math

# directories 
outf = os.path.join(db_data, "clean") #output
mergf = os.path.join(db_data, "merged") #merged output
interim = os.path.join(db_me, "interim-visuals")

#  READ IN ALL THE DATA FOR THE MERGE 
file_paths = {
    "cafo": "/Users/allegrasaggese/Library/CloudStorage/Dropbox/Mental/Data/clean/2025-10-21_cafo_annual_df.csv",
    "crime": "/Users/allegrasaggese/Library/CloudStorage/Dropbox/Mental/Data/clean/2025-10-19_crime_fips_level_final.csv",
    "health": "/Users/allegrasaggese/Library/CloudStorage/Dropbox/Mental/Data/clean/2025-10-17_mh_mortality_fips_yr.csv",
    "pop": "/Users/allegrasaggese/Library/CloudStorage/Dropbox/Mental/Data/clean/2025-08-11_population_full.csv",
    "fips": "/Users/allegrasaggese/Library/CloudStorage/Dropbox/Mental/Data/clean/2025-08-11_fips_full.csv"
}


prepared = {}
for name, path in file_paths.items():
    prepared[name] = read_and_prepare(path)
    print(name, "->", prepared[name].shape)

# quick peek (first 3 rows) for each
for name, df_part in prepared.items():
    print("\n---", name, "---")
    print(df_part.head(3))
    
# print all colnames to find the FIPS NAMES 
for name, df_part in prepared.items():
    print(f"--- {name} ({df_part.shape[0]}x{df_part.shape[1]}) ---")
    print(list(df_part.columns))
    print()

# fix fips  - can take out when we QA the original data and fix the FIPS
for c in ("fips","year"):
    if c not in fips.columns:
        raise KeyError(f"column '{c}' not found in df")

df = prepared['fips']            # the DataFrame object named "fips"

# identify duplicates (any group with >1 row)
dup_mask_any = df.duplicated(subset=['fips','year'], keep=False)
total_dup_rows = int(dup_mask_any.sum())
dup_groups_count = int(df[dup_mask_any].groupby(['fips','year']).ngroups)

print(f"duplicate rows: {total_dup_rows}")
print(f"duplicate (fips,year) groups: {dup_groups_count}")

# show up to 10 example duplicate groups
if total_dup_rows:
    examples = df[dup_mask_any].sort_values(['fips','year']).groupby(['fips','year'])
    for i, ((f,y), grp) in enumerate(examples):
        if i >= 10: break
        print(f"\nexample {i+1}: fips={f}, year={y}, rows={len(grp)}")
        print(grp.head(5))

# drop duplicates keeping the first occurrence per (fips,year)
df_no_dup = df.drop_duplicates(subset=['fips','year'], keep='first').reset_index(drop=True)

# summary after dedupe
print("\nafter dedupe:", df_no_dup.shape[0], "rows")
# drop index
df_no_dup = df_no_dup.iloc[:, :6].reset_index(drop=True)

# NEW FIPS DF CREATED 
prepared['fips'] = df_no_dup


#### FINAL MERGE 
def _norm(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    if 'fips' in df.columns:
        df['fips'] = df['fips'].astype('string').str.replace(r'\.0$','',regex=True).str.zfill(5)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    return df

frames = []
for name, part in prepared.items():
    d = _norm(part)
    if not {'fips','year'}.issubset(d.columns):
        continue
    cols = [c for c in d.columns if c not in ('fips','year')]
    if cols:
        d = d.rename(columns={c: f"{c}_{name}" for c in cols})
    frames.append(d)

merged_all = reduce(lambda L,R: L.merge(R, on=['fips','year'], how='outer'), frames) if frames else pd.DataFrame(columns=['fips','year'])
merged_all = merged_all.sort_values(['fips','year']).reset_index(drop=True)
print("merged_all ->", merged_all.shape)

# check the merged data 
merged_all.head(50)

# export 
out_path = os.path.join(os.path.expanduser(outf), "full_df_merged.csv")
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
merged_all.to_csv(out_path, index=False)


########## VISUALIZATIONS  #############
df = merged_all
df = pd.read_csv()


# test on a single variable and single year across all counties -- will later update to do for year and at state level
var = "hog_med_op_sales_count_cafo"   # replace with your variable
year_val = 2010           # replace with the year you want

# prepare
tmp = df.loc[df['year'] == year_val, ['fips', var]].copy()
tmp[var] = pd.to_numeric(tmp[var], errors='coerce')
tmp = tmp.dropna(subset=[var])

if tmp.empty:
    print("no data for", var, "in", year_val)
else:
    plt.figure(figsize=(6,6))
    sns.violinplot(y=tmp[var], inner="quartile", cut=0)
    plt.ylabel(var)
    plt.title(f"{var} — {year_val}")
    plt.tight_layout()
    plt.show()


# TEST OUT GALINA'S CODE FOR VIOLIN PLOT 
plt.figure(figsize=(12, 8),dpi=200)
sns.violinplot(x='cat', y='B_Lgrcarbon1', data=df, inner='quartile', palette='pink_r',bw=0.2)
 

# Manually set category labels
plt.xticks(ticks=[0, 1, 2, 3], labels=[])  

# Limit y-axis range
plt.ylim(-0.2, 0.2)  # Adjust min and max y-axis values


# Customization
#plt.title("Violin Plot by Category")
plt.xlabel("")
plt.ylabel("Beta")


# Show plot
#plt.show()
plt.savefig("mg.png", format="png", bbox_inches="tight")
plt.show()
plt.close()
