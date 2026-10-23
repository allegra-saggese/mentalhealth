---
paths:
  - "script*.py"
  - "*.py"
  - "tests/**/*.py"
  - "qa-scripts/**/*.py"
  - "explorations/**/*.py"
---

# Python Code Standards

Code clarity and auditability beat cleverness. A reviewer who has never met us must be able to follow, run and check every script.

## 1. Reproducibility

- Run with `PYTHONHASHSEED=0` (set in `environment.yml`); pyfixest results differ by about 2e-5 in SEs otherwise.
- Stochastic code: one `rng = numpy.random.default_rng(<seed>)` at the top, passed to whatever draws. No global `np.random.seed` scattered through the file; no unseeded `random`, `sample()` or bootstrap.
- Imports at the top, grouped stdlib / third party / local. Project code does `from packages import *` for directory definitions only; new code imports names explicitly.
- Versions are pinned in `requirements.txt` and `environment.yml`; any new dependency is added there in the same change.

## 2. Paths and secrets

- Every path derives from `db_base` in `packages.py` (`db_data`, `output_dir`, `figs_dir`, `tables_dir`, `diagnostic_dir`). No `/Users/...` or `~/Dropbox` anywhere else.
- Credentials are read with `os.environ[...]` (`USDA_NASS_API_KEY`, `HUD_API_TOKEN`). Never in code, defaults, logs or commit messages.
- Create output directories with `Path(...).mkdir(parents=True, exist_ok=True)`.

## 3. Naming and structure

- Scripts: `script{stage}{letter}-{description}.py`, letters consecutive within a stage. Shared importable modules have no stage prefix (or use `script4_treatment.py`-style for stage-wide definitions).
- Header: about 10 lines (purpose, inputs, outputs, run order, runtime). Longer history goes in `code-audit/`.
- Outputs are dated `YYYY-MM-DD_{descriptor}.ext`; readers pick the newest with `latest_file_glob()`.
- Functions: `snake_case`, verb-noun, one job, typed arguments where it helps, no magic numbers (name them at the top or in the shared module).
- Comments say WHY a non-obvious step exists, not what the line does. Identifiers carry the what.
- Superseded code moves to `z-archive/` (or `first-stage-testing/`) with its output. Never deleted.

## 4. Data handling

- FIPS are 5-digit zero-padded strings via `normalize_panel_key()`.
- Never fill, drop or recode silently: print counts before and after (rows, counties, county-years), or assert them.
- After every merge, assert the key is unique on the side that should be unique and report unmatched rows.
- Missing is not zero unless a documented rule says so (D-34: `cafo_*` counts from 2002).
- Use `pandas` nullable handling deliberately; check `dtype` after reading CSVs (`low_memory=False`, explicit `dtype` for FIPS).
- FSIS columns come only from `*_panel_fsis.csv`.

## 5. Estimation

- Treatments, sample, outcomes, controls come from `script4_treatment.py`. Estimators come from `did_tools.py` / `did_cs.py`. Do not re-implement them in an exhibit script.
- State the formula, fixed effects, clustering and weights in the result file's columns or metadata, not only in the code.
- Test syntax that might silently do nothing: confirm each new term changes the fit.
- A refactor must keep outputs identical: run `tests/compare_outputs.py` against a baseline.

## 6. Verification before "done"

1. `python -m py_compile` passes.
2. The script runs end to end from a clean shell with `PYTHONHASHSEED=0`.
3. Outputs exist, are dated, and sizes and row counts are as expected.
4. Printed sanity checks (counts, ranges, reproduce-an-earlier-number) are reviewed, not skimmed.

## 7. Common pitfalls

| Pitfall | Impact | Prevention |
|---------|--------|------------|
| Reading the newest file when the run state is stale | Cites a pre-rebuild output | Check `MODEL-RUN-STATE` note first; archive old outputs before re-running |
| `from packages import *` shadows a name | Silent wrong variable | Explicit imports in new code |
| Chained assignment / `inplace` on a slice | Silent no-op | Assign to a copy; assert the result |
| Float equality on estimates | Spurious diffs | `numpy.isclose` with the project tolerances |
| Hardcoded absolute path | Breaks for the reviewer | Only via `packages.py` |
| Unseeded resampling | Non-reproducible CIs | `default_rng(seed)` once |
