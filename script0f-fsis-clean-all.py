#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSIS cleaning pipeline end-to-end w/in a single 

1) download and parse FOIA requested inspection dta on slaughterhouses 
2) create a panel with establishment ID w/ plant size classification
3) HUD fill for partially filled geo data (fips recovery): 
   - bulk mode (default): using HUD zip code to fips walk, fill in for establishments only with city  / state / zip 
   - targeted mode: isolate only missing establishments and use fips walk 
4) Optional manual ZIP template step: using manually found matches, backfilling in fips 
5) merge all outputs together to create panel of fsis with highest coverage possible 

    """
# ----------------------- SET UP PART 1: DEFINE -------------------- -#

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR) if os.path.basename(THIS_DIR) == "fsis-scripts" else THIS_DIR
FSIS_SCRIPT_DIR = os.path.join(REPO_ROOT, "fsis-scripts")
DB_BASE = os.path.expanduser("~/Dropbox/Mental")
DB_DATA = os.path.join(DB_BASE, "Data")
QA_DIR = os.path.join(DB_DATA, "FOIA-USDA-request", "qa-fsis")

# ----------------------- DATA PART 1: EXECUTE FOLDER OF SCRIPTS -------------------- -#

SCRIPT_ORDER = {
    "extract_raw": "script0e-fsis-slaughterhouses.py",
    "build_size_panel": "script0h-fsis-establishment-size-panel.py",
    "hud_zip_fill": "script0i-fsis-hud-zip-fips-fill.py",
    "hud_zipyear_refill": "script0j-fsis-hud-zipyear-refill.py",
    "hud_bulk_fill": "script0k-fsis-hud-bulk-zipyear-fill.py",
    "manual_zip_fill": "script0l-fsis-apply-manual-zip-fips.py",
    "deterministic_fill": "script0m-fsis-deterministic-completion.py",
    "manual_county_fill": "script0n-fsis-apply-manual-county-fips.py",
}


def _latest_file_by_regex(dirpath: str, pattern: str):
    pat = re.compile(pattern)
    candidates = []
    if not os.path.isdir(dirpath):
        return None
    for fn in os.listdir(dirpath):
        m = pat.match(fn)
        if m:
            candidates.append((m.group(1), os.path.join(dirpath, fn)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _run_script(script_name: str, dry_run=False):
    candidate_paths = [
        os.path.join(FSIS_SCRIPT_DIR, script_name),
        os.path.join(REPO_ROOT, script_name),
    ]
    path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing script: {script_name}. Checked: {candidate_paths}"
        )

    cmd = [sys.executable, path]
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] RUN {path}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def _build_steps(hud_strategy: str, run_manual_zip: bool, run_manual_county: bool):
    steps = [
        SCRIPT_ORDER["extract_raw"],
        SCRIPT_ORDER["build_size_panel"],
    ]

    if hud_strategy == "bulk":
        steps.append(SCRIPT_ORDER["hud_bulk_fill"])
    elif hud_strategy == "targeted":
        steps.extend([SCRIPT_ORDER["hud_zip_fill"], SCRIPT_ORDER["hud_zipyear_refill"]])
    elif hud_strategy == "skip":
        pass
    else:
        raise ValueError(f"Unknown hud strategy: {hud_strategy}")

    if run_manual_zip:
        steps.append(SCRIPT_ORDER["manual_zip_fill"])

    steps.append(SCRIPT_ORDER["deterministic_fill"])

    if run_manual_county:
        steps.append(SCRIPT_ORDER["manual_county_fill"])

    return steps


def main():
    parser = argparse.ArgumentParser(description="Single-entry FSIS cleaning pipeline.")
    parser.add_argument(
        "--hud-strategy",
        choices=["bulk", "targeted", "skip"],
        default="bulk",
        help="HUD fill strategy. Default: bulk",
    )
    parser.add_argument(
        "--skip-manual-zip",
        action="store_true",
        help="Skip manual ZIP template application.",
    )
    parser.add_argument(
        "--skip-manual-county",
        action="store_true",
        help="Skip manual county-label template application.",
    )
    parser.add_argument(
        "--require-manual-zip",
        action="store_true",
        help="Fail if manual ZIP template is not found.",
    )
    parser.add_argument(
        "--require-manual-county",
        action="store_true",
        help="Fail if manual county template is not found.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print step plan without executing scripts.",
    )
    args = parser.parse_args()

    if args.hud_strategy != "skip" and not os.environ.get("HUD_API_TOKEN"):
        raise RuntimeError("HUD_API_TOKEN is required unless --hud-strategy skip is used.")

    zip_template = _latest_file_by_regex(
        QA_DIR,
        r"^(\d{4}-\d{2}-\d{2})_fsis_unmatched_unique_zip_for_manual_fips\.xlsx$",
    )
    county_template = _latest_file_by_regex(
        QA_DIR,
        r"^(\d{4}-\d{2}-\d{2})_fsis_missing_county_labels_manual_template\.csv$",
    )

    run_manual_zip = (not args.skip_manual_zip) and (zip_template is not None)
    run_manual_county = (not args.skip_manual_county) and (county_template is not None)

    if args.require_manual_zip and zip_template is None:
        raise FileNotFoundError("Manual ZIP template not found in qa-fsis folder.")
    if args.require_manual_county and county_template is None:
        raise FileNotFoundError("Manual county template not found in qa-fsis folder.")

    if zip_template is None and not args.skip_manual_zip:
        print("Manual ZIP template not found: skipping script0l-fsis-apply-manual-zip-fips.py")
    if county_template is None and not args.skip_manual_county:
        print("Manual county template not found: skipping script0n-fsis-apply-manual-county-fips.py")

    steps = _build_steps(
        hud_strategy=args.hud_strategy,
        run_manual_zip=run_manual_zip,
        run_manual_county=run_manual_county,
    )

    print("FSIS pipeline steps:")
    for idx, step in enumerate(steps, start=1):
        print(f"{idx}. {step}")

    for step in steps:
        _run_script(step, dry_run=args.dry_run)

    print("FSIS pipeline complete.")


if __name__ == "__main__":
    main()
