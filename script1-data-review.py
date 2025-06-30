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
db_data = os.path.join(db_base, "/Data")


# import data
file_path = os.path.join(db_base, "raw_data.csv")
