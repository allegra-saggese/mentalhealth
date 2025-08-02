#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 19:30:16 2025

@author: allegrasaggese
"""

# purpose: perform ridge regression on 2023 aggregated data (note the data is 
# out of date and will need to be updated), results from this should not be taken for interpretation! 

# load packages, functions, databases 
from packages import *
from functions import * 


# load data
db_base = os.path.expanduser("~/Dropbox/Mental") # base for later
rdfile = "~/Dropbox/Mental/Data/20250725_annual_CAIFO_df.csv"
df = pd.read_csv(rdfile)


# purpose: Ridge regression trial 1
# 1: set cols for learning, split the data for one Y value 

# drop to look at only the race/sex cross section 
df_sub = df[df['aggregation_type'] == 'DEMOGRAPHICS'].copy()


# drop non-numeric and non-encoded vars 
uncoded_cols = ['racen', 'sexn', 'fipsracesex', 'dup', 'SURVEY_YEAR', 'FIPS',
                'STATE', 'COUNTY', 'aggregation_type', 'sex_of_arrestee', 
                'race_of_arrestee']
df_sub = df_sub.drop(columns=uncoded_cols)

# set test ridge on y = simple assault
X = df_sub.drop(columns=['simple_assault'])
y = df_sub['simple_assault']

# create race/sex interaction, and dummies for county and time fixed effects 
X_interact = pd.get_dummies(df_sub['racesex'], drop_first=True) 
fips_dummies = pd.get_dummies(df_sub['fips'], drop_first=True)
year_dummies = pd.get_dummies(df_sub['year'], drop_first=True)


### QA STOP ### note there is a lot of missing data - lets check 
n_total = len(X)

# Count missing per column
missing_counts = X.isna().sum().sort_values(ascending=False)

# Top 5 columns with most missing
top5_missing = missing_counts.head(5)

# Size after dropping missing rows
X_dropped = X.dropna()
n_dropped = len(X_dropped)

print(f"Original rows: {n_total}")
print(f"Rows after dropna: {n_dropped}")
print("Top 5 columns losing the most rows:")
print(top5_missing) # note these are missing all columns 

# create threshold for imputation of missing data 
row_thresh = int(0.1 * X.shape[1])  # must have at least 10% non-missing
X_filtered = X[X.notna().sum(axis=1) >= row_thresh]

col_mask = X_filtered.isna().mean() <= 0.9
X_impute_subset = X_filtered.loc[:, col_mask]

imp = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imp.fit_transform(X_impute_subset),
                         columns=X_impute_subset.columns,
                         index=X_impute_subset.index)

# drop missing
valid_idx = y.loc[X_imputed.index].notna()
X_final = X_imputed.loc[valid_idx]
y_final = y.loc[X_imputed.index].loc[valid_idx]

# split the data for training
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

# verify size 
print(X_train.shape, X_test.shape)

# 2: initialize ridge reg 
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# review outputs without alpha changes
pd.Series(ridge_model.coef_, index=X_train.columns).sort_values(ascending=False)
ridge_model.intercept_

y_pred = ridge_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.3f}, R²: {r2:.3f}")


# 3: set shrinkage param and check against pre-programmed one
ridge_cv = RidgeCV(alphas=[0.1, 1, 10, 100])
ridge_cv.fit(X_train, y_train)

# compare / check best alpha 
print(f"Optimal lambda: {ridge_cv_model.alpha_}")

# 4: evaluate with set alphas
y_pred2 = ridge_cv_model.predict(X_test) # test set 

# Calculate MSE, RMSE, MAE, and R-squared
mse1 = mean_squared_error(y_test, y_pred2)
rmse = mse1 ** 0.5
mae = mean_absolute_error(y_test, y_pred2)
r2 = r2_score(y_test, y_pred2)

# Print eval metrics
print(f"MSE: {mse1}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}") 

ridge_cv.predict(X_test)
# takeaway - strong shrinkage = worse, keep alpha low, very different 

# visualization of output 
alphas = [0.01, 0.1, 1, 10, 100]
models = {}
residuals = {}

for a in alphas:
    model = Ridge(alpha=a)
    model.fit(X_train, y_train)
    y_pred_x = model.predict(X_test)
    res = y_test - y_pred_x
    models[a] = model
    residuals[a] = res
    
# plot residuals 
residual_df = pd.DataFrame(residuals)
residual_df.index = y_test.index

residual_df_melted = residual_df.stack().reset_index()
residual_df_melted.columns = ['index', 'alpha', 'residual']

sns.boxplot(data=residual_df_melted, x='alpha', y='residual')
plt.title('Residual Distribution by Alpha')
plt.xlabel('Alpha')
plt.ylabel('Residual')
plt.axhline(0, color='black', linestyle='--')
plt.show()

# rank the coefficients
coef_series = pd.Series(ridge_cv.coef_, index=X_train.columns)
top50_vars = coef_series.abs().sort_values(ascending=False).head(10).index

plot_df = X_train[top50_vars].copy()
plot_df['simple_assault'] = y_train

test_plot = sns.pairplot(plot_df, corner=True)
test_plot.fig.set_size_inches(12, 12)  

for ax in test_plot.axes.flatten():
    if ax is not None:
        ax.tick_params(axis='x', labelrotation=45, labelsize=6)
        ax.tick_params(axis='y', labelrotation=0, labelsize=6)


test_plot.savefig(os.path.join(db_base,"pairplot_top10.png"), dpi=300, bbox_inches='tight')
plt.close()



# NEXT:  create loop over sets of y to just automate the ridge run 






