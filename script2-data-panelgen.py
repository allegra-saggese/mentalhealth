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

# purpose - data review (primary, will not be used in modules)

# load packages
import os
import pandas as pd

# set directories
db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
db_me = os.path.join(db_base, "allegra-dropbox-copy")
interim = os.path.join(db_me, "interim-data") # file for interim datasets or lists used across scripts 


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



