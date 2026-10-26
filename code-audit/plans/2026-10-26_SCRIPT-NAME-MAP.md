# Script name map after the stage-4 consolidation (2026-10-26, D-54)
Each new script holds the old scripts' code unchanged, one section per old script, `--only <section> ...` to run a subset. Output folders and file stems are unchanged (readers find files by folder and stem).
| New script | Section | Old script (removed with `git rm`; history keeps it) |
|---|---|---|
| `script4a-estimates.py` | `designs` | `script4a-design-estimates.py` |
| | `cohorts` | `script4u-did-by-treatment.py` |
| | `species` | `script4x-species-forest.py` |
| | `fsis` | `script4r-fsis-treatment.py` |
| `script4b-diagnostics.py` | `controls` | `script4j-control-diagnostics.py` |
| | `leads` | `script4t-pretrend-leads.py` |
| | `balance` | `script4d-balance-panel.py` |
| | `control-set` | `script4y-control-set-selection.py` |
| | `honestdid` | `script4w-honestdid-export.py` (the R half is `R/honestdid.R`) |
| `script4c-inference.py` | `wild` | `script4z-inference.py` |
| | `permutation` | `script4ze-permutation-inference.py` |
| `script4d-triangulation.py` | `estimators` | `script4za-estimator-triangulation.py` |
| | `treatment-by-model` | `script4zc-treatment-by-model.py` |
| `script4e-tables.py` | `tables`, `appendix`, `frames` | `script4n-regression-tables.py`, `script4p-full-appendix.py`, `script4q-beamer-frames.py` |
| `script4f-figures.py` | `triangulation`, `treatment-by-model`, `permutation` | `script4zb-triangulation-figure.py`, `script4zd-treatment-by-model-figures.py`, `script4zf-permutation-figure.py` |
Earlier in the consolidation: `did_tools.py` and `did_cs.py` are sections of `estimators.py`; `script4b-covariate-sequence.py` and `script4c-covariate-stability.py` are in `z-archive/`; R helpers are in `R/`.
**Folder names written by each section:** designs `script4a`; cohorts `script4u`; species `script4x`; fsis `script4r`; controls `script4a` (control_roster, control_pretrends, vif_headline_within); leads `script4t`; balance LaTeX output folder; control-set `script4y`; honestdid `script4w`; wild `script4z`; permutation `script4ze`; estimators `script4za`; treatment-by-model `script4zc`; figures `script4zb`, `script4zd`, `script4zf`.
**Equality evidence (all at rtol 1e-6, atol 1e-8, `PYTHONHASHSEED=0`, old tree built from HEAD a7dd4ca, scratch HOME, the 2026-10-25 panel):** `designs` and `fsis` full runs identical to the old scripts; `cohorts` and `species` `--quick` identical (21 CSV files across the four estimation sections); `script4b-diagnostics.py` 41 CSV and `.tex` files identical; `script4c-inference.py` and `script4d-triangulation.py` `--quick` files identical; `script4e-tables.py` and `script4f-figures.py` read result files and reproduced the old `.tex` and figure CSVs. Not run: full-length `species`, `cohorts`, `permutation`, `treatment-by-model`, the HonestDiD R run.

## Stage E (FSIS sub-folder), 2026-10-26
| New script (in `fsis-scripts/`) | Section | Old script |
|---|---|---|
| `script0e-fsis-slaughterhouses.py` | `extract` | `script0e-fsis-slaughterhouses.py` (same name, new content) |
| | `size-panel` | `script0h-fsis-establishment-size-panel.py` |
| `script0i-fsis-hud-fill.py` | `bulk` (default `--strategy bulk`) | `script0k-fsis-hud-bulk-zipyear-fill.py` |
| | `zip`, `zipyear` (`--strategy targeted` runs both) | `script0i-fsis-hud-zip-fips-fill.py`, `script0j-fsis-hud-zipyear-refill.py` |
`script0e-fsis-clean-all.py` now runs three steps (build, HUD fill, `script0l` manual and deterministic fill). Each section's code is the old script's code unchanged except one statement: `db_base = os.path.expanduser(...)` is now `from packages import db_base` (the same folder), checked by AST comparison (one differing statement per section).
**Equality (offline):** the `extract` and `size-panel` sections reproduce the old scripts on the raw FSIS files: 10 output files, 5 byte-identical and 5 identical once the scratch HOME inside their path columns is normalised. The three HUD sections need `HUD_API_TOKEN` and the network and were **not run**: they are verified only by AST identity, compilation and the orchestrator dry run for `bulk`, `targeted` and `skip`. A real run with the token is the owner's end-to-end check.
