"""qa_cs_crosscheck.py -- does estimators.py (Callaway-Sant'Anna section) agree with independent Callaway-Sant'Anna implementations?

Builds one balanced extract per mental-health outcome (identical data for every implementation), then compares
ATT(g,t) cell by cell and the overall ATT: estimators.att_gt_table vs R `did` 2.5.1, `csdid` 0.4.2, `moderndid` 0.3.1.
Settings on every side: outcome regression, not-yet-treated comparison, base year g-1 (universal), covariates =
the pre-committed 8 controls at the base year, cohort = first expansion (E2). Point estimates only; the
standard errors differ by design (estimators.run_cs bootstraps over states).
Needs:   R with `did`; the isolated ./.venv from requirements-crosscheck.txt (qa_cs_crosscheck_py.py runs there).
Writes:  Data/output/tables/qa_cs_crosscheck/YYYY-MM-DD_{cells,summary}.csv
Run:     PYTHONHASHSEED=0 python3 qa-scripts/qa_cs_crosscheck.py        (about a minute)
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from estimators import att_gt_table, base_period_covariates  # noqa: E402
from script4_treatment import load_panel, OUTCOMES, COVARIATE_ORDER, tables_dir, date  # noqa: E402

CHECKED = ["Poor MH Days", "Frequent Mental Distress"]
CONTROLS = COVARIATE_ORDER[:8]
XCOLS = [f"x{i + 1}" for i in range(len(CONTROLS))]
COVERAGE = 0.90          # a year is in the window if at least 90% of the best-covered year's counties report the outcome
TOLERANCE = 1e-6         # replication-protocol: estimates at a fixed seed
TAG = "qa-cs-crosscheck"     # output-name tag (packages.py): <date>_qa-cs-crosscheck_<stem>; was folder tables/qa_cs_crosscheck
OUT = Path(tables_dir)
OUT.mkdir(parents=True, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")


def balanced_extract(df, ocol):
    """Counties observed with the outcome and all 8 controls in every window year; usable cohorts only."""
    d = df[["fips", "year", "cohort", ocol] + CONTROLS].copy()
    reporting = d.dropna(subset=[ocol]).groupby("year").fips.nunique()
    years = sorted(reporting[reporting >= COVERAGE * reporting.max()].index)
    complete = d[d.year.isin(years)].dropna(subset=[ocol] + CONTROLS)
    full = complete.groupby("fips").year.nunique()
    d = complete[complete.fips.isin(full[full == len(years)].index)]
    # a cohort needs its base year g-1 and at least one year g..end inside the window; others are never comparisons
    d = d[d.cohort.isna() | ((d.cohort - 1 >= min(years)) & (d.cohort <= max(years)))]
    d = d.rename(columns={ocol: "y", **dict(zip(CONTROLS, XCOLS))})
    d["g"] = d.cohort.fillna(0).astype(int)
    d["fips"] = d.fips.astype(int)
    return d, years


def own_estimates(d):
    """estimators.att_gt_table on the extract: ATT(g,t) cells and the overall post-treatment ATT."""
    wide = d.pivot_table(index="fips", columns="year", values="y", aggfunc="first")
    cohort_of = d.groupby("fips").cohort.first()
    cohorts = sorted(d.cohort.dropna().unique())
    years = sorted(d.year.unique())
    xbase = base_period_covariates(d, cohorts, XCOLS)
    cells = pd.DataFrame(att_gt_table(wide, cohort_of, cohorts, years, Xbase=xbase, use_cov=True))
    post = cells[cells.event_time >= 0]
    overall = float(np.average(post.att, weights=post.n_treated.astype(float)))
    return cells[["cohort", "year", "att", "n_treated", "n_control", "fallback"]], overall


def run_external(extract, tag):
    """R `did` and the venv packages (csdid, moderndid) on the same file; returns (cells, overall) frames."""
    paths = {k: OUT / f"{TODAY}_{TAG}_{tag}_{k}.csv" for k in ("r_cells", "r_overall", "py_cells", "py_overall")}
    subprocess.run(["Rscript", str(ROOT / "R" / "cs_crosscheck.R"), str(extract),
                    str(paths["r_cells"]), str(paths["r_overall"])], check=True, capture_output=True)
    subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "qa-scripts" / "qa_cs_crosscheck_py.py"),
                    str(extract), str(paths["py_cells"]), str(paths["py_overall"])], check=True, capture_output=True)
    r_cells = pd.read_csv(paths["r_cells"]).assign(package="did (R)")[["package", "cohort", "year", "att"]]
    r_over = pd.read_csv(paths["r_overall"]).assign(package="did (R)")[["package", "overall_att"]]
    cells = pd.concat([r_cells, pd.read_csv(paths["py_cells"])])
    overall = pd.concat([r_over, pd.read_csv(paths["py_overall"])])
    return cells, overall


all_cells, summary = [], []
df = load_panel()
for okey in CHECKED:
    d, years = balanced_extract(df, OUTCOMES[okey])
    extract = OUT / f"{TODAY}_{TAG}_{okey.replace(' ', '_')}_extract.csv"
    d[["fips", "year", "g", "y"] + XCOLS].to_csv(extract, index=False)
    ours, ours_overall = own_estimates(d)
    ext_cells, ext_overall = run_external(extract, okey.replace(" ", "_"))
    print(f"\n{okey}: window {years[0]}-{years[-1]}, {d.fips.nunique():,} counties "
          f"(treated {d[d.g > 0].fips.nunique()}, never {d[d.g == 0].fips.nunique()}), "
          f"cohorts {sorted(d[d.g > 0].g.unique())}, own cells {len(ours)} ({int(ours.fallback.sum())} fallback)")
    for package, cells in ext_cells.groupby("package"):
        m = ours.merge(cells, on=["cohort", "year"], how="outer", suffixes=("_own", "_pkg"), indicator=True)
        both = m[m._merge == "both"]
        # packages also list the base-year cell g-1 (ATT = 0 by construction); estimators.att_gt_table skips it
        pkg_only = m[(m._merge == "right_only") & (m.year != m.cohort - 1)]
        diff = (both.att_own - both.att_pkg).abs()
        pkg_overall = float(ext_overall.loc[ext_overall.package == package, "overall_att"].iloc[0])
        summary.append({"outcome": okey, "package": package, "cells_compared": len(both),
                        "cells_only_in_did_cs": int((m._merge == "left_only").sum()), "cells_only_in_package": len(pkg_only),
                        "max_abs_diff_cells": float(diff.max()), "overall_att_did_cs": ours_overall,
                        "overall_att_package": pkg_overall, "abs_diff_overall": abs(ours_overall - pkg_overall),
                        "agrees_at_1e-6": bool(diff.max() < TOLERANCE and abs(ours_overall - pkg_overall) < TOLERANCE)})
        all_cells.append(both.assign(outcome=okey, package=package, abs_diff=diff)[
            ["outcome", "package", "cohort", "year", "att_own", "att_pkg", "abs_diff"]])

cells_out, summary_out = pd.concat(all_cells), pd.DataFrame(summary)
cells_out.to_csv(OUT / f"{TODAY}_{TAG}_cells.csv", index=False)
summary_out.to_csv(OUT / f"{TODAY}_{TAG}_summary.csv", index=False)
pd.set_option("display.width", 220)
print("\n" + summary_out.round(8).to_string(index=False))
print(f"\nwrote {OUT}/{TODAY}_{TAG}_cells.csv and _summary.csv")
