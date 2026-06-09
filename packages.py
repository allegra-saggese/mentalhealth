#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared imports and directory definitions for all project scripts.
ML/sklearn imports live in script3-ridge.py only.
"""

import sys
import os
import re
import glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from functools import reduce

# All other paths derive from db_base — update this one variable per machine.
db_base        = os.path.expanduser("~/Dropbox/Mental")
db_data        = os.path.join(db_base, "Data")
db_me          = os.path.join(db_base, "allegra-dropbox-copy")
interim        = os.path.join(db_me, "interim-data")
diagnostic_dir = os.path.join(db_data, "clean", "diagnostic")  # QA + intermediates
