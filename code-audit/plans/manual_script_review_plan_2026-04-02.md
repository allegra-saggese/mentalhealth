# Manual Script Review Plan (2026-04-02)

## Goal
Manually review the full pipeline so we can trust that updated clean files and merged outputs are reproducible, correctly keyed, and using intended latest inputs.

## Review Order (dependency-first)
1. Raw -> clean builders
2. FSIS FIPS completion chain
3. Merge + replacement scripts
4. QA/analysis/visual scripts
5. Legacy/retired scripts

## Stage 1: Raw -> Clean Builders
Scripts:
- script0a-pop-fips-raw-merge.py
- script0b-ag-raw.py
- script0b-ag-raw-v2.py
- script0c-health-raw.py
- script0d-crime-raw.py
- script0e-fsis-slaughterhouses.py
- script0f-nchs-urban.py
- script0o-cdc-county-deathsofdespair-panel.py

Checks per script:
- Verify input glob/patterns only pull intended raw sources.
- Confirm key construction logic for `fips` and `year` (padding, coercion, drop rules).
- Confirm output filenames follow date-prefix conventions and do not overwrite wrong descriptors.
- Spot-check row counts and unique `(fips, year)` before/after major transformations.
- Validate any hard-coded paths and fallback branches still match current folder layout.

Deliverable:
- 1-page checklist table: script, inputs, outputs, key assumptions, pass/fail.

## Stage 2: FSIS Completion Chain
Scripts:
- script0h-fsis-establishment-size-panel.py
- script0i-fsis-hud-zip-fips-fill.py
- script0j-fsis-hud-zipyear-refill.py
- script0k-fsis-hud-bulk-zipyear-fill.py
- script0l-fsis-apply-manual-zip-fips.py
- script0m-fsis-deterministic-completion.py
- script0n-fsis-apply-manual-county-fips.py

Checks per script:
- Confirm each script reads the expected *latest* predecessor file descriptor.
- Verify fill methods are blank-only or intended overwrite behavior.
- Confirm state-FIPS consistency checks are enforced for newly filled `fips_code`.
- Ensure QA artifacts are written for unresolved rows and method counts.
- Verify final outputs used by merge are the intended descriptor variant (`*_hudbulk_manualzip`).

Deliverable:
- Chain diagram showing input/output file per FSIS step plus unresolved-count trend by step.

## Stage 3: Merge + Replacement
Scripts:
- script1b-merge-dataclean.py
- script1c-replace-cdc-in-merged.py
- script1a-QA-dataclean.py

Checks:
- Review `MERGE_DESCRIPTORS` and confirm this list matches current intended clean datasets.
- Confirm rural-key filter (`non_large_metro == 1`) is intentional for all downstream analysis.
- Verify merge collision behavior (suffixing, descriptor tags) and no accidental column clobbering.
- Confirm spliced outputs (`2005_2010`, `2010_2020`, census years) match expected year windows.
- After CDC replacement, verify only intended columns/rows changed.

Deliverable:
- Merge manifest: descriptor -> exact dated file used -> row/column contribution.

## Stage 4: QA + Analysis Scripts
Scripts:
- script2a-panel-sumstats-by-farms.py
- script2b-qa-memo-correlations.py
- script2c-fsis-sizebins-vs-mental-2017.py
- script2d-mental-coverage-audit.py
- script2-visualizations.py

Checks:
- Confirm each script points to latest merged panel and latest required clean inputs.
- Validate variable name assumptions against current 556-variable inventory.
- Verify plots/tables fail loudly (not silently) if expected variables are missing.
- Confirm outputs are versioned/date-stamped and written to correct subfolders.

Deliverable:
- QA matrix: script, required columns, required files, generated outputs, failure mode.

## Stage 5: Legacy / Cleanup Decision
Scripts:
- script3-ridge.py
- z-archive/*

Checks:
- Decide keep/retire status.
- If retained, update paths and input file assumptions.
- If retired, add clear deprecation note in file header.

Deliverable:
- Short "active vs retired" script registry.

## Priority Hotspots to Review First
- `script0b-ag-raw-v2.py`: contains API and local `.dta` fallback logic plus hard-coded raw path block; likely highest drift risk.
- `script1b-merge-dataclean.py`: hard-coded merge descriptor set may exclude newer clean products unless intentionally omitted.
- FSIS chain (`0h`-`0n`): many similarly named outputs; easy to read/write the wrong variant.
- `script1c-replace-cdc-in-merged.py`: post-merge mutation step; needs strict before/after checks.

## Suggested Manual Workflow (single pass)
1. Build a review sheet with one row per script and columns for inputs, outputs, key cols, assumptions, status.
2. Run each stage in order and log row counts + unique `(fips, year)` at stage boundaries.
3. Diff output manifests against previous run to isolate expected vs unexpected changes.
4. Sign off stage-by-stage before proceeding to the next stage.
