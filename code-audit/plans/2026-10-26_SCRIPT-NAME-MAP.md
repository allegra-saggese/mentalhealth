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
| `script4e-tables.py` | `tables`, `appendix`, `frames`, `slides` (new, no old script) | `script4n-regression-tables.py`, `script4p-full-appendix.py`, `script4q-beamer-frames.py` |
| `script4g-spillovers.py` | `moran`, `donut`, `conley`, `division-fe`, `durbin` (new 2026-10-29, no old script; D-58) | |
| `script4f-figures.py` | `triangulation`, `treatment-by-model`, `permutation`, `slides`, `slides-other` (the last two new, no old script) | `script4zb-triangulation-figure.py`, `script4zd-treatment-by-model-figures.py`, `script4zf-permutation-figure.py` |
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

## Additions after the consolidation, 2026-10-26 (D-55)
New sections, no old script behind them: `script4b-diagnostics.py --only pretrends` (raw pre-period paths and fits; folder `tables/script4b`); `script4f-figures.py --only pretrends-leads pretrends-controls pretrends-raw pretrends-controls-raw` (folder `figs/script4f`); `script1c-figures.py --only control-coverage`. The `controls` section of `script4b` also writes `control_pretrend_lags.csv`, `control_correlations_selected8.csv` and `.txt` tables to `tables/script4a`; its existing three CSVs are identical to before (checked).

## Output layout, 2026-10-26 (after D-56)
Per-script output folders are replaced by one flat folder per kind of file (`Data/output/{tables,figures,maps,latex,codebooks}`), with the
writing script and section in the file name: `YYYY-MM-DD_<tag>_<stem>`. Helpers `tagged`, `out_file`, `out_glob`, `out_latest` in `packages.py`
are the only place the rule is coded. Existing files were renamed, not rewritten; the file-by-file record is
`Data/output/_archive/2026-10-26_layout-migration_MANIFEST.csv` (763 rows). Notes written before this date (MODEL-RUN-STATE, session notes)
cite the old folders; read them through this table.
| Old folder | New tag | Section |
|---|---|---|
| `tables/script4a`, `figs/script4a` | `4a-designs` | `script4a --only designs` |
| `tables/script4a` (control_roster, control_pretrends, vif_headline_within, control_pretrend_lags, control_correlations_selected8, VIF_audit.txt) | `4b-controls` | `script4b --only controls` |
| `tables/script4u` | `4a-cohorts` | `script4a --only cohorts` |
| `tables/script4x`, `figs/script4x` | `4a-species` | `script4a --only species` |
| `tables/script4r` | `4a-fsis` | `script4a --only fsis` |
| `tables/script4t` | `4b-leads` | `script4b --only leads` |
| `tables/script4b` (pretrend_*) | `4b-pretrends` | `script4b --only pretrends` |
| `tables/script4b` (covariate_sequence, 09-23) | `4b-covariate-sequence` | `z-archive/script4b-covariate-sequence.py` |
| `latex-outputs-run2/*_balance_panel.csv` | `4b-balance` | `script4b --only balance` (the .tex stay in `latex/`) |
| `tables/script4y` | `4b-control-set` | `script4b --only control-set` |
| `tables/script4w` | `4b-honestdid` | `R/honestdid.R` |
| `tables/script4w/honestdid_inputs` | `4b-honestdid-inputs` (undated) | `script4b --only honestdid` |
| `tables/script4z` | `4c-wild` | `script4c --only wild` |
| `tables/script4ze` | `4c-permutation` | `script4c --only permutation` |
| `tables/script4za` | `4d-estimators` | `script4d --only estimators` |
| `tables/script4zc` | `4d-treatment-by-model` | `script4d --only treatment-by-model` |
| `figs/script4zb` / `script4zd` / `script4zf` | `4f-triangulation` / `4f-treatment-by-model` / `4f-permutation` | `script4f` |
| `figs/script4f` | `4f-pretrends` | `script4f --only pretrends-*` |
| `figs/slides` | `4f-slides` | `script4f --only slides slides-other` |
| `tables/script1b`, `tables/script1c`, `figs/script1c`, `maps/script1d`, `tables/script1a`, `figs/script1a` | `1b`, `1c`, `1d`, `1a` | stage-1 scripts |
| `tables/<date>_ratio_flags_qa.csv` | `0f` | `script0f-generate-panel.py` |
| `tables/qa_cs_crosscheck` | `qa-cs-crosscheck` | `qa-scripts/qa_cs_crosscheck.py` |
| `latex-outputs-run2/` | `latex/` (folder renamed; names unchanged) | `script4b --only balance`, `script4e` |
| `*_quick` folders | `<tag>-quick` | any `--quick` run |
