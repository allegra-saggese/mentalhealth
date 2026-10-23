---
paths:
  - "script*.py"
  - "tests/**"
  - "Data/output/**"
---

# Task Completion Verification Protocol

**Before reporting a task done, run it and look at the output.** A script that exits 0 can still have written an empty file, the wrong panel, or a figure with a missing panel.

## For a pipeline script

1. `python -m py_compile <file>`.
2. Run with `PYTHONHASHSEED=0` from a clean shell; read the printed counts, not just the exit code.
3. Confirm the expected output files exist, are dated today, and are non-empty (row counts, column counts).
4. Spot-check magnitudes against a known number (an earlier output, a published figure, a NASS total).
5. For estimation scripts: confirm the self-checks print zero difference (`script4u`, `script4x` reproduce `script4a` exactly for the first-expansion/dairy-large case).
6. State which panel file and run date the output used.

## For a figure or table

1. Open the saved PDF/PNG (Read the image) and check: panel labels, axis labels and units, legend, CI note, no clipped text.
2. Check size and dpi against `figure-and-table-standards.md`; run `/figure-check`.
3. For a table, compile the `.tex` fragment (if `pdflatex` is available) or state that it was not compiled.

## For a refactor

Run `tests/compare_outputs.py <baseline> <live>`; a refactor is done only when it reports no difference at the project tolerances.

## Claims about code carry a revision

Any statement about what the code does now carries the revision it was read at; see [`agent-authored-code.md`](agent-authored-code.md).

## Checklist

```
[ ] Script compiles and runs end to end (PYTHONHASHSEED=0)
[ ] Output files exist, dated, non-empty
[ ] Counts and magnitudes sanity-checked against something known
[ ] Figures/tables opened and inspected, or named for the user to inspect
[ ] Which panel and run date produced the numbers is stated
[ ] Anything not run or not verified is named as such
```
