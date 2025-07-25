#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:19:14 2025

@author: allegrasaggese
"""

# purpose: useful functions across scripts 

# load packages
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# data cleaning - checking missing percentages 
def percent_missing_vs_filled(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with, for each column in `df`:
      - missing_pct:    percentage of NA values
      - filled_pct:     percentage of non-NA (numeric) values
    """
    # total rows
    total = len(df)

    # count missing per column
    missing_count = df.isna().sum()

    # compute percentages
    missing_pct = missing_count / total * 100
    filled_pct = 100 - missing_pct

    # assemble summary
    summary = pd.DataFrame({
        'missing_pct': missing_pct,
        'filled_pct':  filled_pct
    })

    return summary
