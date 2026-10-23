---
paths:
  - "script2*.py"
  - "script4*.py"
  - "figstyle.py"
---

# Figure and Table Standards (Nature / Science family first)

Every exhibit that leaves the repo is publication-ready: correct size at final print, vector where possible, readable at final size, fully labelled, reproducible from a script. Target venue is the Nature / Science family; widths below are those journals' column widths. If the venue changes, change the values in `figstyle.py` once (it holds the widths, palette, fonts and `save_figure`).

## Format and size

- Single column 89 mm (3.50 in), 1.5 column 120 mm, double column 183 mm (7.20 in). Height at most 247 mm. Set `figsize` in inches at final print size; never rescale afterwards.
- Save **vector PDF** (fonts embedded, `pdf.fonttype = 42`) and a **600 dpi PNG** for line art, 300 dpi minimum for images and maps. Maps and choropleths: PNG at 300+ dpi is acceptable; keep a vector layer where the library allows.
- File name `YYYY-MM-DD_{descriptor}.pdf|png`, written to the script's `figs_dir` subfolder.

## Typography and style

- Sans-serif (Arial or Helvetica; fall back to DejaVu Sans only if unavailable, and say so).
- Text 5-7 pt at final size; panel labels (a, b, c) bold, 8 pt, lower case, top left. Line width 0.5-1 pt; marker size consistent across panels.
- No in-figure title (the caption carries it); no chart junk (no top/right spines, no 3-D, no gradients, no gridlines unless they carry meaning).
- Axis labels name the variable and unit ("Deaths of despair per 100,000"); the zero line is drawn where zero is meaningful.

## Colour

- A single colour-blind-safe palette defined once (Okabe-Ito for categories; perceptually uniform sequential map such as `viridis`/`cividis` for continuous; a diverging map centred on zero for signed quantities).
- Information is never carried by colour alone: add marker shape, line style or direct labels. Must remain legible in greyscale.
- Same variable, same colour, in every exhibit.

## Uncertainty and inference (what a referee checks first)

- Show the confidence interval and state it: level, estimator, fixed effects, clustering (for example "95% CI, SE clustered by state").
- Event-study plots: reference period marked and labelled, leads and lags on a common y-axis, zero line, treated and comparison sample sizes in the note, pre-trend test p-value reported, not hidden.
- Forest/coefficient plots: fixed ordering across exhibits, units stated, estimator labelled (TWFE vs Callaway-Sant'Anna).
- Report the sample: N counties, N county-years, window, in a source note below the figure.
- Anything that weakens the headline (a failed pre-trend, a null in the full window) appears with the same prominence as the supportive result.

## Source note and caption

- Every exhibit has a note: data sources, sample, specification, estimator, clustering, and the script that produced it (`script4x-...py`).
- Captions are written by the author; draft captions are internal until rewritten (see `writing-with-ai.md`).

## Tables

- LaTeX `booktabs`; no vertical rules; one shared writer function for all regression tables so formatting cannot drift.
- Coefficients with SE in parentheses (clustered, stated); **no significance stars** by default for the Nature/Science family (report CIs or exact p-values); N, fixed effects, controls ticked in rows below the coefficients.
- Consistent decimals (3 significant figures for coefficients, same for SEs); units in column headers; a note line gives the specification.
- Tables are generated from result CSVs, never typed. The table script does not estimate.

## Before an exhibit is cited

Run `/figure-check` on it: size, dpi, fonts embedded, label/unit, CI note, palette, greyscale, source note, script provenance. A figure that has not passed is marked DRAFT in its file name.

## Label overlap (matplotlib)

Check tick labels, annotations and legends against each other with `fig.canvas.get_renderer()` and `Text.get_window_extent()`; shorten, rotate or direct-label rather than shrink fonts below 5 pt.
