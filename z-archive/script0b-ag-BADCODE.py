#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 01:23:03 2025

@author: allegrasaggese
"""

# script containing old / bad code related CAFOs 

################ TAKE ONLY CAFO OBSERVATIONS FROM THE DATA ####################
df = df[['county_code',
 'FIPS_generated',
 'layer_ops_size',
 'layer_head_size',
 'broiler_head_size',
 'broiler_ops_size',
 'cattle_ops_size_inv',
 'cattle_head_size_inv',
 'cattle_ops_size_sales',
 'hog_ops_size_inv',
 'hog_head_size_inv',
 'hog_ops_size_sales',
 'dairy_ops_size_inv',
 'dairy_head_size_inv',
 'breed_hog_ops_size_inv',
 'breed_hog_head_size_inv',
 'cattle_senzcow_ops_size_inv',
 'cattle_senzcow_head_size_inv',
 'cattle_feed_ops_size_inv',
 'cattle_feed_map_head_size_inv',
 'beef_ops_size_inv',
 'beef_map_head_size_inv',
 'cattle_calves_ops_size_sales',
 'cattle_feed_ops_size_sales',
 'cattle_500lbs_ops_size_sales',
 'calves_ops_size_sales',
 'state_fips_code', 
 'year',
 'value',
 'domaincat_desc']].copy()

frames = []

thresholds = [
    broiler_cutoff_lrg, broiler_cutoff_med, layer_cutoff_lrg, layer_cutoff_med,
    cattle_cutoff_lrg, cattle_cutoff_med, hog_cutoff_lrg, hog_cutoff_med, 
    h_dairy_cutoff_lrg, h_dairy_cutoff_med, h_all_cattle_cutoff_lrg, 
    h_all_cattle_cutoff_med, h_calf_cutoff_lrg, h_calf_cutoff_med, h_hogs_cutoff_lrg, 
    h_hogs_cutoff_med, h_broilers_cutoff_lrg, h_broilers_cutoff_med, h_layers_cutoff_lrg,
    h_layers_cutoff_med]



def assign_size(df, col, med_thresh, lrg_thresh):
    df = df.copy()
    # keep Int64Dtype and use nullable comparisons
    cond_large  = df[col].ge(lrg_thresh)
    cond_medium = df[col].ge(med_thresh) & df[col].lt(lrg_thresh)
    cond_small  = df[col].lt(med_thresh)

    df['size'] = pd.Series(pd.NA, index=df.index, dtype="string")
    df.loc[cond_large.fillna(False), 'size']  = 'large'
    df.loc[cond_medium.fillna(False), 'size'] = 'medium'
    df.loc[cond_small.fillna(False), 'size']  = 'small'

    df['size_col'] = col
    return df


# start appending manually 
# chickens 
frames.append(assign_size(df, 'layer_ops_size', layer_cutoff_med, layer_cutoff_lrg))
frames.append(assign_size(df, 'layer_head_size', layer_cutoff_med, layer_cutoff_lrg))

frames.append(assign_size(df, 'broiler_ops_size', broiler_cutoff_med, broiler_cutoff_lrg))
frames.append(assign_size(df, 'broiler_head_size', broiler_cutoff_lrg, broiler_cutoff_lrg))

# cattle - we use the same size cut offs, but we should create a new numeric code for different types of cattle in the future --- this is just for ease 
frames.append(assign_size(df, 'cattle_ops_size_inv', 
                          cattle_cutoff_med, cattle_cutoff_lrg))
frames.append(assign_size(df, 'cattle_head_size_inv', 
                          cattle_cutoff_med, cattle_cutoff_lrg))
frames.append(assign_size(df, 'cattle_ops_size_sales', 
                          cattle_cutoff_med, cattle_cutoff_lrg)) 

frames.append(assign_size(df, 'dairy_ops_size_inv', 
                          cattle_cutoff_med, cattle_cutoff_lrg))
frames.append(assign_size(df, 'dairy_head_size_inv', 
                          cattle_cutoff_med, cattle_cutoff_lrg))

frames.append(assign_size(df, 'cattle_senzcow_ops_size_inv', 
                          cattle_cutoff_med, cattle_cutoff_lrg))
frames.append(assign_size(df, 'cattle_senzcow_head_size_inv', 
                          cattle_cutoff_med, cattle_cutoff_lrg))

frames.append(assign_size(df, 'cattle_feed_ops_size_inv', 
                          cattle_cutoff_med, cattle_cutoff_lrg))
frames.append(assign_size(df, 'cattle_feed_map_head_size_inv', 
                          cattle_cutoff_med, cattle_cutoff_lrg))

frames.append(assign_size(df, 'beef_ops_size_inv', 
                          cattle_cutoff_med, cattle_cutoff_lrg))
frames.append(assign_size(df, 'beef_map_head_size_inv', 
                          cattle_cutoff_med, cattle_cutoff_lrg))

frames.append(assign_size(df, 'cattle_calves_ops_size_sales', 
                          cattle_cutoff_med, cattle_cutoff_lrg))
frames.append(assign_size(df, 'cattle_feed_ops_size_sales', 
                          cattle_cutoff_med, cattle_cutoff_lrg))
frames.append(assign_size(df, 'cattle_500lbs_ops_size_sales', 
                          cattle_cutoff_med, cattle_cutoff_lrg))
frames.append(assign_size(df, 'calves_ops_size_sales', 
                          cattle_cutoff_med, cattle_cutoff_lrg))
# hogs 
frames.append(assign_size(df, 'hog_ops_size_inv', 
                          hog_cutoff_med, hog_cutoff_lrg))
frames.append(assign_size(df, 'hog_head_size_inv', 
                          hog_cutoff_med, hog_cutoff_lrg))
frames.append(assign_size(df, 'hog_ops_size_sales', 
                          hog_cutoff_med, hog_cutoff_lrg))
frames.append(assign_size(df, 'breed_hog_ops_size_inv', 
                          hog_cutoff_med, hog_cutoff_lrg))
frames.append(assign_size(df, 'breed_hog_head_size_inv', 
                          hog_cutoff_med, hog_cutoff_lrg))


#### QA FRAMES BEFORE MERGING 
len(frames)
frames[0].head()      # first frame
frames[5].info()      # sixth frame
frames[-1].sample(10) # random sample from last frame

sum(len(f) for f in frames)  # total rows if you stacked them all
frames[0]["size"].head()
frames[0]["size"].value_counts(dropna=False)
frames[0]["size"].isna().mean()  # percent missing
frames[0]["layer_ops_size"].isna().mean()  # percent missing - should be equivalent to size
frames[0]["hog_ops_size_inv"].isna().mean() # weirdly this one is not empty... 

# check to see if each frames contain the same data 
col = "layer_ops_size"

for i, f in enumerate(frames):
    if col in f.columns:
        pct_missing = f[col].isna().mean() * 100
        print(f"Frame {i}: {pct_missing:.2f}% missing")
    else:
        print(f"Frame {i}: column not found")
# great - each col is the same so the only one I need to keep from each of the assign_size is the size col 

col = "size"

for i, f in enumerate(frames):
    if col in f.columns:
        pct_missing = f[col].isna().mean() * 100
        print(f"Frame {i}: {pct_missing:.2f}% missing")
    else:
        print(f"Frame {i}: column not found")

# can drop frame 1, frame 3, frame 23 as there were no matches here  (AFTERWARDS)




# rename the size cols based on the assign_size function
new_names = ['layer_ops_size',
'layer_head_size',
'broiler_head_size',
'broiler_ops_size',
'cattle_ops_size_inv',
'cattle_head_size_inv',
'cattle_ops_size_sales',
'dairy_ops_size_inv',
'dairy_head_size_inv',
'cattle_senzcow_ops_size_inv',
'cattle_senzcow_head_size_inv',
'cattle_feed_ops_size_inv',
'cattle_feed_map_head_size_inv',
'beef_ops_size_inv',
'beef_map_head_size_inv',
'cattle_calves_ops_size_sales',
'cattle_feed_ops_size_sales',
'cattle_500lbs_ops_size_sales',
'calves_ops_size_sales',
'hog_ops_size_inv',
'hog_head_size_inv',
'hog_ops_size_sales',
'breed_hog_ops_size_inv',
'breed_hog_head_size_inv']


for f, name in zip(frames, new_names):
    if "size" in f.columns:
        f.rename(columns={"size": f"{name}_size_class"}, inplace=True)

# check the rename worked  - successful 
frames[0]["layer_ops_size_size_class"].head()
for i, f in enumerate(frames[:3]):
    print(f.columns)

# sense checking once more they are all the same 
for i, f in enumerate(frames):
    print(f"Frame {i}: {len(f)} rows")

keep_fixed = ["FIPS_generated", "year"] #set of cols to keep 

# create reduced copies for all frames except the first
reduced_frames = []
for f in frames[1:]:
    # find the second-to-last column
    second_last = f.columns[-2]
    subset = f[keep_fixed + [second_last]].copy()
    reduced_frames.append(subset)

# check the slice worked
print(reduced_frames[0].head())
lengths = [len(f) for f in reduced_frames]
print(lengths, "All equal?" , len(set(lengths)) == 1) # TRUE 

final_df = frames[0].copy()

# column bind them all together to get size classification
for rf in reduced_frames:
    size_cols = [c for c in rf.columns if c.endswith("_size_class")]
    if not size_cols:
        continue  # skip if missing
    size_col = size_cols[0]
    final_df[size_col] = rf[size_col].values

# check equality of length - all equal 
print([len(f) for f in reduced_frames])
print(len(final_df))


#### LETS TAKE ONLY THE COUNT OF CAFO OPS TO START WITH SIZE CLASSIFICATION
size_cols = [c for c in final_df.columns if c.endswith("_size_class")]

# select identifiers and size cols
cols = ["FIPS_generated", "year"] + size_cols
subset = final_df[cols].copy()

subset.to_csv("interim_ag_raw.csv") # save it incase the kernel crashes again 

# delete the extra vars in storage
del df, final_df, rf

# melt (collapse) and then count for each size category
results = []

for c in size_cols:
    tmp = (
        subset[["FIPS_generated", "year", c]]
        .dropna(subset=[c])
        .groupby(["FIPS_generated", "year", c])
        .size()
        .reset_index(name="count_cafo")
    )
    tmp["animal_type"] = c
    results.append(tmp)
    
count_by_group = pd.concat(results, ignore_index=True)

### QA FINAL TABLE MERGE 
count_by_group.info()        # column types and non-missing counts
count_by_group.head()        # preview first few rows
count_by_group.sample(10)    # random sample

# summary of the final df 
count_by_group.describe(include='all')  # numeric + categorical summary

# pct filled in and counts 
(count_by_group.isna().mean() * 100).round(2)
count_by_group.nunique()

# Distribution of CAFO counts
count_by_group['count_cafo'].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])


# Quick shape + memory use
print(len(count_by_group), "rows ×", len(count_by_group.columns), "columns")
count_by_group.memory_usage(deep=True).sum() / 1e6  # MB

# drop empty rows (i.e. observations with ZERO CAFOs)
keep_cols = ["FIPS_generated", "year", "count_cafo", "animal_type"]

# drop rows where all other (besides key) columns are NaN --- also need to drop totally empty cols 
count_by_group = count_by_group.dropna(
    how="all",
    subset=[c for c in count_by_group.columns if c not in keep_cols]
)

print(len(count_by_group), "rows remain after dropping fully missing rows")
count_by_group.isna().mean().round(3) 

count_by_group = count_by_group.dropna(axis=1, how="all")
print("Remaining columns:", len(count_by_group.columns))
count_by_group.info()
