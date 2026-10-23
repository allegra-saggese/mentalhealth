# Project Memory

Corrections and learned facts that persist across sessions. This file is committed to a **public** repo: no keys, machine paths, data values or private names.
When a mistake is corrected, append a `[LEARN:category]` entry below (most recent at the bottom): `wrong -> right`, plus the reason in one clause.

---

## Communication

[LEARN:communication] Private shorthand ("E2", "E3", "leads", "dark", "step-8 sequence") -> plain terms ("first expansion", "first entry", "pre-treatment years", "the control set"), defined before first use; show the estimator equation when asking for a method choice. Reason: the user defends these specifications to a team and to referees.

[LEARN:communication] Interpretation mixed into results -> show results first, with no interpretation; give flags separately and only when asked.

[LEARN:communication] The wrap-up box fired before the written answer and the answer was lost -> write the full answer as text first; the box ends the turn.

## Scope and safety

[LEARN:scope] Deleting a specification the team narrowed -> keep estimating it and filter at the reporting level (`REPORT_*` in `script4_treatment.py`); archive old outputs by moving, never deleting.

[LEARN:scope] Changing shared or major files, or re-running estimation, without showing a diff -> ask first, show the diff, then run. Never push; licence choice belongs to the owner.

## Verification

[LEARN:verification] Syntax that may silently do nothing (a pyfixest `fips[year]` term was ignored) -> test that each term changes the fit before relying on it; build checks that reproduce an earlier result exactly.

[LEARN:verification] Unseeded pyfixest runs differ by about 2e-5 in SEs across processes -> always run with `PYTHONHASHSEED=0`; compare outputs with `tests/compare_outputs.py`.

[LEARN:citations] A plausible-looking citation that was not checked -> search and verify every citation; state what could not be verified.

[LEARN:verification] A test fixture or regex that contains the forbidden pattern trips the hook that guards it (the pre-commit gate blocked its own self-test and its own scorer) -> build such literals by concatenation (`api_%s` filled with `key`; the path regex split as `~/` `Dropbox`). Reason: both false positives appeared only when the hook ran on its own files.

[LEARN:verification] A planted-effect test fell just under its threshold (0.78 vs 0.8 on 40 simulated panels) -> measure the reference test's power on the same simulated data first (here the wild bootstrap 0.85 vs the analytic test 0.87); if they match it was sampling noise, so raise the planted effect or the number of simulations instead of lowering the threshold. Reason: a lowered threshold hides a real power loss.

## Data

[LEARN:data] Treating blank `cafo_*` counts as missing for 2002 onward -> the agricultural census is exhaustive, so blank means zero from 2002 (rule D-34); 2000-01 stay missing because no census was taken.

[LEARN:data] Loading FSIS columns from the main panel -> FSIS columns exist only in `*_panel_fsis.csv` (2017+).
