#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 12:10:51 2025

@author: allegrasaggese
"""

# quick suicide descriptive stats
## all we want is CAFO data + suicide data and we can run some descriptors 

# create CAFO × suicides (fips-year) and save
import os
import pandas as pd


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