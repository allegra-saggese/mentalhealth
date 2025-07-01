#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 18:55:21 2025

@author: allegrasaggese
"""

# purpose - data review (primary, will not be used in modules)

# load packages
import os
import pandas as pd

# set directories
db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
db_me = os.path.join(db_base, "allegra-dropbox-copy")


# import data
agg_file_path = os.path.join(db_me, "25-07-01-dta-copy.dta")
df = pd.read_stata(agg_file_path, convert_categoricals=False)

# review data 
print(df.columns.tolist())