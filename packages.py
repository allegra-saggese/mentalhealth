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
output_dir     = os.path.join(db_data, "output")

# ---------------------------------------------------------------------------
# Output layout (2026-10-26): one output folder, split only by kind of file.
#   tables/    every CSV / TXT / MD result        figures/  every PDF / PNG figure
#   maps/      HTML + PNG maps                    latex/    .tex fragments, appendix, frames, slides/assets
#   codebooks/ dated VARIABLE_REGISTER versions
# The script and section that wrote a file are in its name, not in a folder:
#   YYYY-MM-DD_<tag>_<stem>.<ext>      e.g. 2026-10-25_4a-cohorts_twfe_results.csv
# <tag> is the script number plus the --only section for stage 4 (4a-designs, 4d-estimators, ...),
# the script number alone for stages 0-1 (0f, 1a, 1b, 1c, 1d), and gets a "-quick" suffix in --quick runs.
# Build every output path with out_file() and find inputs with out_latest()/out_glob(), so the naming rule
# lives here only. (latex/ is the exception: its .tex names are fixed because LaTeX \input's them.)
# ---------------------------------------------------------------------------
tables_dir     = os.path.join(output_dir, "tables")
figs_dir       = os.path.join(output_dir, "figures")
maps_dir       = os.path.join(output_dir, "maps")
latex_dir      = os.path.join(output_dir, "latex")
codebooks_dir  = os.path.join(output_dir, "codebooks")

_DATED = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+)$")


def tagged(tag, name):
    """File name with the producing script/section tag: '<date>_<stem>' -> '<date>_<tag>_<stem>'; undated
    names get the tag as a prefix ('manifest.csv' -> '<tag>_manifest.csv')."""
    m = _DATED.match(name)
    return f"{m.group(1)}_{tag}_{m.group(2)}" if m else f"{tag}_{name}"


def out_file(folder, tag, name):
    """Full path for an output file in one of the output folders (creates the folder)."""
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, tagged(tag, name))


def out_glob(folder, tag, pattern="*"):
    """Sorted paths in `folder` written under `tag` that match `pattern`, a pattern written for the untagged
    name (e.g. '*_A6_cs_aggregated.csv' or 'manifest.csv'). Dated files sort by date, so the last is newest."""
    core = pattern[2:] if pattern.startswith("*_") else pattern
    hits = glob.glob(os.path.join(folder, f"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_{tag}_{core}"))
    if not pattern.startswith("*_"):
        hits += glob.glob(os.path.join(folder, f"{tag}_{core}"))
    return sorted(set(hits))


def out_latest(folder, tag, pattern="*"):
    """Newest file written under `tag` matching `pattern`; raises FileNotFoundError when there is none."""
    hits = out_glob(folder, tag, pattern)
    if not hits:
        raise FileNotFoundError(f"no output '{tagged(tag, pattern)}' in {folder}")
    return hits[-1]
