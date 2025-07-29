#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 19:30:16 2025

@author: allegrasaggese
"""

# load relevant packages for ridge reg 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import sklearn
print(sklearn.__version__)

from sklearn.preprocessing import scale 
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV


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
fips_dummies = pd.get_dummies(df['fips'], drop_first=True)
year_dummies = pd.get_dummies(df['year'], drop_first=True)


# split the data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# verify size 
print(X_train.shape, X_test.shape)

# 2: initialize ridge reg 
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

# 3: set shrinkage param and check against pre-programmed one

alphas = [0.1, 1.0, 10.0, 100.0]
ridge_cv_model = RidgeCV(alphas=alphas, store_cv_values=True)

ridge_cv_model.fit(X_train, y_train)

# compare / check best alpha 
print(f"Optimal lambda: {ridge_cv_model.alpha_}")

# 4: evaluate 

y_pred = ridge_cv_model.predict(X_test) # test set 

# Calculate MSE, RMSE, MAE, and R-squared
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print eval metrics
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")





# NEXT:  create loop over sets of y to just automate the ridge run 






