# First-stage testing scripts (kept for provenance, not part of the pipeline)

These were the first phase of development: pilots and test versions of the analysis and its
reporting. They are **not run** for the results we present, and several read output files that
no longer exist. They are kept, with history, so a reader can see how the final models were
reached. Files carry a `firststage_` prefix so they cannot be mistaken for pipeline scripts.

| File | What it was | Replaced by |
|---|---|---|
| `firststage_script3-ridge.py` | LASSO/Ridge variable selection, panel OLS and DiD pilot; "not for interpretation" | Part A design-based estimates (`script4a`); ML selection remains specified-only (B-series, `2026-09-27_MODEL-NAMING.md`) |
| `firststage_script4-model-test.py` | Independent replication of an external dairy-CAFO memo; first single-file model test (1,538 lines) | `script4_treatment` (definitions) + `script4a`/`4b`/`4r` (estimation) |
| `firststage_script4d-power-balance.py` | Sample / power / coverage workbook for the pre-rebuild Part A | `script4d-balance-panel.py` |
| `firststage_script4e-latex-tables.py` | LaTeX power tables from the 4d workbook (pre-rebuild) | `script4n-regression-tables.py`, `script4p-full-appendix.py` |
| `firststage_script4g-slides.py` | First beamer deck from the Part A results (read `script4f` output, retired D-30) | `script4q-beamer-frames.py` |
| `firststage_script4h-result-tables.py` | First beamer result tables (read the retired `A1_A2` filenames) | `script4n`, `script4q` |

Moved 2026-10-17 with `git mv`; nothing was edited. Some comments elsewhere in the repo still use the
old file names.
