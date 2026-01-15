#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 11:54:38 2026

@author: allegrasaggese
"""

# read in new augmentations to the data - slaughterhouses 
inf = os.path.join(db_data, "raw") # input 
outf = os.path.join(db_data, "clean") #output


# FUTURE CHANGE: JAN 2026 - should change to use API to download relevant data only 
# import ag data 
fsis_data = os.path.join(inf, "fsis")
fsis_df = pd.read_csv(os.path.join(fsis_data, "2026-updated-Dataset_Establishment_Demographic_Data.csv"))