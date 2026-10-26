#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/compare_outputs.py -- prove a refactor changed no result.

Compares the NEWEST file of each name in two output trees (a saved baseline and the live outputs):
    *.csv   same columns and rows; numeric cells equal within rtol=1e-6, atol=1e-8
    *.tex   identical text (dates normalised)
    *.png   identical pixels
Files are matched by name without the leading YYYY-MM-DD_ date stamp. Both trees must use the same layout
(since 2026-10-26: tables/, figures/, latex/, with the script/section tag in the file name; packages.py).

Usage:  python3 tests/compare_outputs.py <baseline_root> <live_root>
Exit code 0 if everything matches, 1 otherwise.
"""
import glob, os, re, sys
import numpy as np
import pandas as pd

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}_")
RTOL, ATOL = 1e-6, 1e-8


def newest_by_stem(root, subdirs):
    out = {}
    for sd in subdirs:
        for f in sorted(glob.glob(os.path.join(root, sd, "**", "*.*"), recursive=True)):
            if f.endswith((".csv", ".tex", ".png")) and "__pycache__" not in f:
                rel = os.path.relpath(f, root)
                key = os.path.join(os.path.dirname(rel), DATE.sub("", os.path.basename(rel)))
                out[key] = f            # sorted ascending, so the newest dated file wins
    return out


def cmp_csv(a, b):
    try:
        x, y = pd.read_csv(a), pd.read_csv(b)
    except pd.errors.EmptyDataError:                      # a file with no columns: equal only if both are empty
        return os.path.getsize(a) == os.path.getsize(b), "empty file"
    if list(x.columns) != list(y.columns):
        return False, f"columns differ: {set(x.columns) ^ set(y.columns)}"
    if len(x) != len(y):
        return False, f"rows {len(x)} vs {len(y)}"
    worst = 0.0
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
            u, v = x[c].to_numpy(float), y[c].to_numpy(float)
            if not np.array_equal(np.isnan(u), np.isnan(v)):
                return False, f"{c}: missing values differ"
            m = ~np.isnan(u)
            if m.any():
                d = np.abs(u[m] - v[m]); worst = max(worst, float(d.max()))
                if not np.allclose(u[m], v[m], rtol=RTOL, atol=ATOL):
                    return False, f"{c}: max abs diff {d.max():.3g}"
        elif not x[c].astype(str).equals(y[c].astype(str)):
            return False, f"{c}: text differs"
    return True, f"max abs diff {worst:.2g}"


def cmp_tex(a, b):
    norm = lambda p: re.sub(r"\d{4}-\d{2}-\d{2}", "DATE", open(p, errors="replace").read())
    return norm(a) == norm(b), ""


def cmp_png(a, b):
    import matplotlib.image as mi
    x, y = mi.imread(a), mi.imread(b)
    return x.shape == y.shape and bool(np.array_equal(x, y)), ""


if __name__ == "__main__":
    base, live = sys.argv[1], sys.argv[2]
    subs = ["tables", "figures", "latex"]
    A, B = newest_by_stem(base, subs), newest_by_stem(live, subs)
    ok = bad = missing = 0
    for key in sorted(A):
        if key not in B:
            print(f"MISSING in live : {key}"); missing += 1; continue
        f = {".csv": cmp_csv, ".tex": cmp_tex, ".png": cmp_png}[os.path.splitext(key)[1]]
        good, msg = f(A[key], B[key])
        if good:
            ok += 1
        else:
            bad += 1; print(f"DIFFERENT       : {key}  {msg}")
    print(f"\n{ok} identical, {bad} different, {missing} missing (of {len(A)} baseline files)")
    sys.exit(0 if (bad == 0 and missing == 0) else 1)
