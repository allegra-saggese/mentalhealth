#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSIS cleaning pipeline end-to-end w/in a single 

1) download and parse FOIA requested inspection dta on slaughterhouses 
2) create a panel with establishment ID w/ plant size classification
3) HUD fill for partially filled geo data (fips recovery): 
   - bulk mode (default): using HUD zip code to fips walk, fill in for establishments only with city  / state / zip 
   - targeted mode: isolate only missing establishments and use fips walk 
4) Last FIPS-recovery step (one script): manual ZIP template, deterministic fills (ZIP, Connecticut, city+state, establishment
   history), manual county-label template; a missing template skips its step 
5) merge all outputs together to create panel of fsis with highest coverage possible 

    """
# ----------------------- SET UP PART 1: DEFINE -------------------- -#

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

from packages import db_data


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR) if os.path.basename(THIS_DIR) == "fsis-scripts" else THIS_DIR
FSIS_SCRIPT_DIR = os.path.join(REPO_ROOT, "fsis-scripts")
QA_DIR = os.path.join(db_data, "FOIA-USDA-request", "qa-fsis")

# ----------------------- DATA PART 1: EXECUTE FOLDER OF SCRIPTS -------------------- -#

SCRIPT_ORDER = {
    "build_establishment_panel": "script0e-fsis-slaughterhouses.py",   # sections extract, size-panel
    "hud_fill": "script0i-fsis-hud-fill.py",                            # --strategy bulk (default) or targeted
    "manual_and_deterministic_fill": "script0l-fsis-manual-and-deterministic-fill.py",   # manual ZIP, deterministic, manual county label
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


def _run_script(script_name: str, dry_run=False, extra_args=()):
    candidate_paths = [
        os.path.join(FSIS_SCRIPT_DIR, script_name),
        os.path.join(REPO_ROOT, script_name),
    ]
    path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing script: {script_name}. Checked: {candidate_paths}"
        )

    cmd = [sys.executable, path, *extra_args]
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] RUN {path} {' '.join(extra_args)}".rstrip())
    if dry_run:
        return
    env = os.environ.copy()
    existing_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = REPO_ROOT if not existing_py else f"{REPO_ROOT}:{existing_py}"
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def _build_steps(hud_strategy: str):
    """The steps as (script, extra arguments) pairs, in run order."""
    steps = [(SCRIPT_ORDER["build_establishment_panel"], ())]

    if hud_strategy in ("bulk", "targeted"):
        steps.append((SCRIPT_ORDER["hud_fill"], ("--strategy", hud_strategy)))
    elif hud_strategy != "skip":
        raise ValueError(f"Unknown hud strategy: {hud_strategy}")

    steps.append((SCRIPT_ORDER["manual_and_deterministic_fill"], ()))
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

    if args.require_manual_zip and zip_template is None:
        raise FileNotFoundError("Manual ZIP template not found in qa-fsis folder.")
    if args.require_manual_county and county_template is None:
        raise FileNotFoundError("Manual county template not found in qa-fsis folder.")

    if zip_template is None and not args.skip_manual_zip:
        print("Manual ZIP template not found: the manual ZIP step will be skipped")
    if county_template is None and not args.skip_manual_county:
        print("Manual county template not found: the manual county-label step will be skipped")

    steps = _build_steps(hud_strategy=args.hud_strategy)
    # the merged last step skips a manual step whose template is missing, and honours the same four flags
    last_step_args = [f for f, on in (("--skip-manual-zip", args.skip_manual_zip), ("--skip-manual-county", args.skip_manual_county),
                                      ("--require-manual-zip", args.require_manual_zip),
                                      ("--require-manual-county", args.require_manual_county)) if on]

    print("FSIS pipeline steps:")
    for idx, (script, script_args) in enumerate(steps, start=1):
        print(f"{idx}. {script} {' '.join(script_args)}".rstrip())

    for step in steps:
        script, script_args = step
        _run_script(script, dry_run=args.dry_run, extra_args=(*script_args, *(last_step_args if step == steps[-1] else ())))

    print("FSIS pipeline complete.")


if __name__ == "__main__":
    main()
