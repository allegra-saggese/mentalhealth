#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 13:14:43 2025

@author: allegrasaggese
"""

# set packages to import 
import os
import pandas as pd
import numpy as np
import datetime as datetime
import time
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


import sklearn
print(sklearn.__version__)

from sklearn.preprocessing import scale 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from datetime import datetime
from tqdm import tqdm


# set directories to keep using 
db_base = os.path.expanduser("~/Dropbox/Mental")
db_data = os.path.join(db_base, "Data")
db_me = os.path.join(db_base, "allegra-dropbox-copy")

interim = os.path.join(db_me, "interim-data") # file for interim datasets or lists used across scripts 