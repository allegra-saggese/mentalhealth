#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 16:31:27 2025

@author: allegrasaggese
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 18:55:21 2025

@author: allegrasaggese
"""

### NOTE THIS SCRIPT IS OUT OF DATE AS OF OCT 2025 DO NOT USE! 


# purpose (1) iterate the CAFO data to have annual figures instead of every 5 years
# (2) create scatterplots for important pairwise matches


# load packages, functions, databases 
from packages import *
from functions import * 


# set additional directories
db_scatter = os.path.join(db_me, "2025-scatterplots") # for saving scatterplots 


# import data
agg_file_path = os.path.join(db_me, "25-07-01-dta-copy.dta")
df = pd.read_stata(agg_file_path, convert_categoricals=False)

# sort data / review bf iteration 
df_sorted = df.sort_values(by=["STATE", "FIPS", "SURVEY_YEAR"])
df_sorted.head(20)

# save test head for manual review 
test_df_for_iteration = df_sorted.head(500)
save_path = os.path.join(interim, "test_df_for_iteration.csv")
test_df_for_iteration.to_csv(save_path, index=False)

print("File exists:", os.path.exists(save_path)) # check it worked 



## ITERATION

# start by dropping sub-group rows
df_totals = df[df["aggregation_type"] != "DEMOGRAPHICS"].copy()


# create new years
new_years = [2003, 2004, 2005, 2006, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
fips_list = df_totals["FIPS"].dropna().unique()

new_rows = pd.DataFrame(
    [(fips, year) for fips in fips_list for year in new_years],
    columns=["FIPS", "year"]
)

# grab CAIFO cols 
cols = df_totals.columns.tolist()
start_idx = cols.index("all_animal_OP")
end_idx = cols.index("cattle_share_med_CAFO")

# CAIFO cols 
cols_to_copy = cols[start_idx:end_idx + 1]


# check data shape 
print(df_totals["year"].unique())
print(df_totals['FIPS'].nunique()) # 3149

print(df_totals[df_totals["year"] == 2002].shape) # 3111
print(df_totals[df_totals["year"] == 2007].shape) # 3143 missing 6 
print(df_totals[df_totals["year"] == 2012].shape) # 3143 missing 6 

# check types for the merge --- types remain the same (no issue in merge)
print(df_totals.dtypes["FIPS"])
print(new_rows.dtypes["FIPS"])


# Subset for base years
base_2002 = df_totals[df_totals["year"] == 2002][["FIPS"] + cols_to_copy].copy()
base_2007 = df_totals[df_totals["year"] == 2007][["FIPS"] + cols_to_copy].copy()
base_2012 = df_totals[df_totals["year"] == 2012][["FIPS"] + cols_to_copy].copy()


# Filter new rows
new_rows_03_06 = new_rows[new_rows["year"].isin([2003, 2004, 2005, 2006])].copy()
new_rows_08_11 = new_rows[new_rows["year"].isin([2008, 2009, 2010, 2011])].copy()
new_rows_13_16 = new_rows[new_rows["year"].isin([2013, 2014, 2015, 2016])].copy()


# checking FIPS match rate across each match across dataframes
filled_03_06 = new_rows_03_06.merge(
    base_2002, on="FIPS", how="left", indicator=True
)

filled_08_11 = new_rows_08_11.merge(
    base_2007, on="FIPS", how="left", indicator=True
)

filled_13_16 = new_rows_13_16.merge(
    base_2012, on="FIPS", how="left", indicator=True
)

# checking that the merge went through 
print(filled_03_06["_merge"].value_counts())
print(filled_08_11["_merge"].value_counts())
print(filled_13_16["_merge"].value_counts())

print(filled_13_16.head(50))

# concat 
df_full = pd.concat(
    [df, filled_03_06, filled_08_11, filled_13_16],
    ignore_index=True,
    sort=False
)


# save new full iterated df 
today = datetime.today().strftime('%Y%m%d')
filename = f"{today}_annual_CAIFO_df.csv"
df_full.to_csv(os.path.join(db_data, filename), index=False)


#### SCATTERPLOTS 

# drop empty cols 
missing_path = os.path.join(interim, 'cols_all_missing.csv')
cols_to_drop = pd.read_csv(missing_path)['col'].tolist()
df_full.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# create two separate dataframes at diff cross sections
df_cty_yr = df_full[df_full["aggregation_type"] != "DEMOGRAPHICS"].copy()
df_cty_yr_r_s = df_full[df_full["aggregation_type"] == "DEMOGRAPHICS"].copy()


# drop any other empty columns from the subset so the scatterplots aren't blank
full_na_cols = df_cty_yr.columns[df_cty_yr.isna().all()].tolist()
print("dropping:", full_na_cols)
df_cty_yr.drop(columns=full_na_cols, inplace=True)

full_na_2 = df_cty_yr_r_s.columns[df_cty_yr_r_s.isna().all()].tolist()
print("dropping:", full_na_2)
df_cty_yr_r_s.drop(columns=full_na_2, inplace=True)


# define groups of vars 
print(df_cty_yr.columns.tolist())

cols1 = set(df_cty_yr.columns)
cols2 = set(df_cty_yr_r_s.columns)

# columns in df1 but not in df2
only_in_df1 = cols1 - cols2
print("In df1 not in df2:", only_in_df1)

# columns in df2 but not in df1
only_in_df2 = cols2 - cols1
print("In df2 not in df1:", only_in_df2)

CAIFO_vars = ['all_animal_OP', 'aqua_OP', 'cattle_HEAD_INV', 'cattle_OP_INV', 
         'cattle_OP_SALES', 'cattle_HEAD_SALES', 'broiler_chickens_HEAD_INV', 
         'broiler_chickens_OP_INV', 'broiler_chickens_OP_SALES', 
         'broiler_chickens_HEAD_SALES', 'layer_chickens_HEAD_INV', 
         'layer_chickens_OP_INV', 'layer_chickens_OP_SALES', 
         'layer_chickens_HEAD_SALES', 'pullets_chickens_HEAD_INV', 
         'pullets_chickens_OP_SALES', 'pullets_chickens_OP_INV', 
         'pullets_chickens_HEAD_SALES', 'hogs_HEAD_INV', 'hogs_OP_INV', 
         'hogs_OP_SALES', 'hogs_HEAD_SALES', 'dairy_OP', 'poultry_other_OP', 
         'livestock_other_OP', 'specialty_OP', 'broiler_cafos_lrg_op', 
         'broiler_cafos_med_op', 'layer_cafos_lrg_op', 'layer_cafos_med_op', 
         'cattle_cafos_INV_lrg_op', 'cattle_cafos_INV_med_op', 
         'cattle_cafos_INV_lrg_head', 'cattle_cafos_INV_med_head', 
         'cattle_cafos_SALES_lrg_op', 'cattle_cafos_SALES_med_op', 
         'cattle_cafos_SALES_lrg_head', 'cattle_cafos_SALES_med_head', 
         'hog_cafos_INV_lrg_op', 'hog_cafos_INV_med_op', 
         'hog_cafos_INV_lrg_head', 'hog_cafos_INV_med_head', 
         'hog_cafos_SALES_lrg_op', 'hog_cafos_SALES_med_op', 
         'hog_cafos_SALES_lrg_head', 'hog_cafos_SALES_med_head', 
         'all_animal_dollar', 'aqua_dollar', 'cattle_dollar', 'hogs_dollar', 
         'dairy_dollar', 'poultry_other_dollar', 'livestock_other_dollar', 
         'specialty_dollar', 'log_all_animal_dollar', 'log_cattle_dollar', 
         'log_dairy_dollar', 'log_livestock_other_dollar', 'log_aqua_dollar',
         'log_hogs_dollar', 'log_poultry_other_dollar', 'log_specialty_dollar',
         'cattle_nonCAFO', 'fips_num', 'log_med_house_inc', 
         'cattle_share_lrg_CAFO', 'cattle_share_med_CAFO']

crime_vars = ['aggravated_assault', 'driving_under_the_influence', 
         'child_molestation', 'incest', 'intimidation', 'kidnapping_abduction',
         'rape', 'sexual_assault_with_an_object', 'simple_assault', 
         'human_trafficking']


# use without denominator to cut run time 
health_agg_vars = ['adult_obesity_numer', 'adult_obesity_raw', 
                   'adult_smoking_numer', 'adult_smoking_raw', 
                   'air_pollution_raw',
                   'excessive_drinking_numer', 'excessive_drinking_raw', 
                   'frequent_mental_distress_raw', 'frequent_phys_distress_raw', 
                   'injury_deaths_aian_', 
                   'injury_deaths_asipacisl', 'injury_deaths_black_','injury_deaths_hispanic_', 
                   'injury_deaths_numer', 'injury_deaths_raw', 
                   'injury_deaths_white_', 'med_house_inc_aian_',
                   'poor_mental_health_days_raw',
                   'poor_or_fair_health_numer', 'poor_or_fair_health_raw', 
                   'poor_phys_health_days_raw', 'prem_ageadj_aian_', 
                   'prem_ageadj_asipacisl', 'prem_ageadj_black_', 
                   'prem_ageadj_denom', 'prem_ageadj_hispanic_', 
                   'prem_ageadj_numer', 'prem_ageadj_raw',
                   'prem_ageadj_white_', 'prem_death_aian_',
                   'prem_death_asipacisl', 'prem_death_black_','prem_death_hispanic_', 
                   'prem_death_numer', 'prem_death_raw', 'prem_death_white_', 
                    'suicides_aian_',
                   'suicides_asipacisl', 'suicides_black_','suicides_hispanic_', 'suicides_numer', 
                   'suicides_raw', 'suicides_white_','violent_crime_numer', 
                   'violent_crime_raw',
                   'alc_driving_deaths_numer', 'alc_driving_deaths_raw']



# not present in race/sex cross-section
droplist = ['injury_deaths_aian_', 'injury_deaths_asipacisl', 'injury_deaths_black_', 
 'injury_deaths_hispanic_', 'injury_deaths_white_', 'med_house_inc_aian_', 
 'med_house_inc_asian_', 'prem_ageadj_aian_', 'prem_ageadj_asipacisl', 
 'prem_death_aian_', 'prem_death_asipacisl', 'suicides_aian_', 
 'suicides_asipacisl', 'suicides_black_', 'suicides_denom', 
 'suicides_hispanic_', 'suicides_numer', 'suicides_raw', 'suicides_white_', 
 '_merge', 'prem_ageadj_asipacisl', 'suicides_white_', 'prem_death_asipacisl',
 'injury_deaths_hispanic_', 'injury_deaths_black_', 'med_house_inc_asian_', 
 'suicides_denom', 'prem_ageadj_aian_', 'suicides_numer', 
 'injury_deaths_asipacisl', 'med_house_inc_aian_', '_merge', 
 'suicides_asipacisl', 'suicides_hispanic_', 'suicides_raw', 
 'injury_deaths_aian_', 'injury_deaths_white_', 'suicides_aian_', 
 'prem_death_aian_', 'suicides_black_']

health_sub_vars = [x for x in health_agg_vars if x not in droplist]


socioecon_vars = ['high_school_graduation_numer', 
                  'high_school_graduation_raw', 'inc_inequality_numer', 
                  'inc_inequality_raw','unemployment_numer', 'unemployment_raw', 
                  'severe_housing_cost_numer', 
                  'severe_housing_cost_raw', 
                  'severe_housing_problems_numer', 
                  'severe_housing_problems_raw','some_college_numer', 
                  'some_college_raw', 'air_pollution_days_raw', 
                  'air_pollution_ozone_days_raw', 'child_in_pov_black_', 
                  'child_in_pov_hispanic_', 'child_in_pov_numer', 
                  'child_in_pov_raw', 'child_in_pov_white_', 
                  'drinking_water_violations_raw',
                  'med_house_inc_asian_', 'med_house_inc_black_', 
                  'med_house_inc_hispanic_', 'med_house_inc_raw', 
                  'med_house_inc_white_',
                  'mental_health_providers_numer', 
                  'mental_health_providers_raw',
                  'perc_high_housing_costs']

soc_sub_vars = [x for x in socioecon_vars if x not in droplist]




#### WITH DENOM --- IGNORE --- NOT USEFUL FOR PLOTS
health_agg_vars_w_denom = ['adult_obesity_denom', 'adult_obesity_numer', 
                   'adult_obesity_raw', 'adult_smoking_denom', 
                   'adult_smoking_numer', 'adult_smoking_raw', 
                   'air_pollution_raw',
                   'excessive_drinking_numer', 'excessive_drinking_raw', 
                   'frequent_mental_distress_raw', 'frequent_phys_distress_raw', 
                   'injury_deaths_aian_', 
                   'injury_deaths_asipacisl', 'injury_deaths_black_', 
                   'injury_deaths_denom', 'injury_deaths_hispanic_', 
                   'injury_deaths_numer', 'injury_deaths_raw', 
                   'injury_deaths_white_', 'med_house_inc_aian_', 
                   'poor_mental_health_days_denom', 
                   'poor_mental_health_days_raw', 
                   'poor_or_fair_health_denom',
                   'poor_or_fair_health_numer', 'poor_or_fair_health_raw', 
                   'poor_phys_health_days_denom', 
                   'poor_phys_health_days_raw', 'prem_ageadj_aian_', 
                   'prem_ageadj_asipacisl', 'prem_ageadj_black_', 
                   'prem_ageadj_denom', 'prem_ageadj_hispanic_', 
                   'prem_ageadj_numer', 'prem_ageadj_raw',
                   'prem_ageadj_white_', 'prem_death_aian_',
                   'prem_death_asipacisl', 'prem_death_black_', 
                   'prem_death_denom', 'prem_death_hispanic_', 
                   'prem_death_numer', 'prem_death_raw', 'prem_death_white_', 
                    'suicides_aian_',
                   'suicides_asipacisl', 'suicides_black_', 
                   'suicides_denom', 'suicides_hispanic_', 'suicides_numer', 
                   'suicides_raw', 'suicides_white_',
                   'violent_crime_denom', 'violent_crime_numer', 
                   'violent_crime_raw', 'alc_driving_deaths_denom', 
                   'alc_driving_deaths_numer', 'alc_driving_deaths_raw']


socioecon_vars_w_denom = ['high_school_graduation_denom', 'high_school_graduation_numer', 
                  'high_school_graduation_raw', 
                  'inc_inequality_denom', 'inc_inequality_numer', 
                  'inc_inequality_raw', 'unemployment_denom', 
                  'unemployment_numer', 'unemployment_raw', 
                  'severe_housing_cost_denom', 'severe_housing_cost_numer', 
                  'severe_housing_cost_raw', 'severe_housing_problems_denom', 
                  'severe_housing_problems_numer', 
                  'severe_housing_problems_raw', 
                  'some_college_denom', 'some_college_numer', 
                  'some_college_raw', 'air_pollution_days_raw', 
                  'air_pollution_ozone_days_raw', 'child_in_pov_black_', 
                  'child_in_pov_hispanic_', 'child_in_pov_numer', 
                  'child_in_pov_raw', 'child_in_pov_white_', 
                  'drinking_water_violations_raw', 'excessive_drinking_denom',
                  'med_house_inc_asian_', 'med_house_inc_black_', 
                  'med_house_inc_hispanic_', 'med_house_inc_raw', 
                  'med_house_inc_white_', 'mental_health_providers_denom', 
                  'mental_health_providers_numer', 
                  'mental_health_providers_raw',
                  'perc_high_housing_costs']



# create the plots for agg data 

combos = [
    (socioecon_vars,   health_agg_vars, 'Socioeconomic v Health'),
    (socioecon_vars,   crime_vars,      'Socioeconomic v Crime' ),
    (socioecon_vars,   CAIFO_vars,      'Socioeconomic v CAIFO'),
    (health_agg_vars,  crime_vars,      'Health v Crime'),
    (health_agg_vars,  CAIFO_vars,      'Health v CAIFO'),
    (crime_vars,       CAIFO_vars,      'Crime v CAIFO')
]

# set up output PDFs to not crash 
today = datetime.today().strftime('%Y%m%d')

# change color before to avoid loop weight
years = np.sort(df_cty_yr['year'].unique())
cmap = cm.get_cmap('tab10', len(years))
year_to_color = dict(zip(years, cmap.colors))
point_colors  = df_cty_yr['year'].map(year_to_color)


# clear out manually unused stuff
#del new_rows_03_06, new_rows_08_11, new_rows_13_16, new_rows, filled_03_06, filled_08_11, 
#filled_13_16,


# define loop and store output 

for x_vars, y_vars, combo_name in combos:
    pdf_filename = f"{today}_{combo_name}_2x2.pdf"
    pdf_path     = os.path.join(db_scatter, pdf_filename)

    # build all (x,y) pairs and number of pages
    pairs  = [(x, y) for x in x_vars for y in y_vars]
    n_pages = (len(pairs) + 3) // 4

    with PdfPages(pdf_path) as pdf:
        # progress bar per combo
        for page_idx in tqdm(range(n_pages), desc=combo_name):
            # pick next 4 pairs
            chunk     = pairs[page_idx*4 : page_idx*4 + 4]
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            axes_flat = axes.flatten()

            for ax, (x, y) in zip(axes_flat, chunk):
                ax.scatter(
                    df_cty_yr[x], df_cty_yr[y],
                    c=point_colors,    # fast, precomputed
                    s=10,              # small points
                    rasterized=True    # speeds up PDF output
                )
                ax.set_xlabel(x)
                ax.set_ylabel(y)

            # remove any empty subplots
            for ax in axes_flat[len(chunk):]:
                fig.delaxes(ax)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Saved {len(pairs)} plots ({n_pages} pages) → {pdf_path}")
    
    
    
# define loop for the dissagregated cross section (race / sex)
combo_name_small = [
    (soc_sub_vars,     health_sub_vars, 'Socioeconomic v Health'),
    (soc_sub_vars,     crime_vars,      'Socioeconomic v Crime' ),
    (soc_sub_vars,     CAIFO_vars,      'Socioeconomic v CAIFO'),
    (health_sub_vars,  crime_vars,      'Health v Crime'),
    (health_sub_vars,  CAIFO_vars,      'Health v CAIFO'),
    (crime_vars,       CAIFO_vars,      'Crime v CAIFO')
]


# change color before to avoid loop weight
years = np.sort(df_cty_yr_r_s['year'].unique())
cmap = cm.get_cmap('tab10', len(years))
year_to_color = dict(zip(years, cmap.colors))
point_colors  = df_cty_yr_r_s['year'].map(year_to_color)

year_to_color # check the color for the interpretation

# run loop over the demo vars 
for x_vars, y_vars, combo_name in combo_name_small:
    pdf_filename2 = f"{today}_{combo_name}_2x2_r_s.pdf"
    pdf_path2     = os.path.join(db_scatter, pdf_filename2)

    # build all (x,y) pairs and number of pages
    pairs  = [(x, y) for x in x_vars for y in y_vars]
    n_pages = (len(pairs) + 3) // 4

    with PdfPages(pdf_path2) as pdf:
        # progress bar per combo
        for page_idx in tqdm(range(n_pages), desc=combo_name):
            # pick next 4 pairs
            chunk     = pairs[page_idx*4 : page_idx*4 + 4]
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            axes_flat = axes.flatten()

            for ax, (x, y) in zip(axes_flat, chunk):
                ax.scatter(
                    df_cty_yr_r_s[x],  df_cty_yr_r_s[y],
                    c=point_colors,    # fast, precomputed
                    s=10,              # small points
                    rasterized=True    # speeds up PDF output
                )
                ax.set_xlabel(x)
                ax.set_ylabel(y)

            # remove any empty subplots
            for ax in axes_flat[len(chunk):]:
                fig.delaxes(ax)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Saved {len(pairs)} plots ({n_pages} pages) → {pdf_path}")
    
