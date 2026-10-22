# Original header of `script4q-beamer-frames.py`

Verbatim module docstring before the 2026-10-22 header trim (see METHODS.md and the decision register for the current versions of this material).

````text

script4q-beamer-frames.py  --  drop-in beamer frames, one topic per slide.

Emits frames in the deck's own style: adjustbox-wrapped tabular, coefficient with
	iny(SE) inline, lert{} on 5% significance, a 	iny notes line below.

SCOPE, DELIBERATELY NARROW
--------------------------
DESCRIPTIVE AND TECHNICAL ONLY. No summary slides, no verdicts, no "what this
means" frames. Every note states what was estimated, on what sample, with what
assumption -- never what it implies. Interpretation belongs to the speaker, not
to the generated slides.

ORDER
    1     treatments T1-T4
    2-3   E2, with the worked county
    4-6   step-8 controls: the set, how it was arrived at, the sequence
    7-8   excluded covariates, with reasons
    9     outcome variables: units and interpretation
    10-19 one model per specification + its regression table
    20-21 VIF
    22-23 balance / detectable effect

REPORTED vs ROBUSTNESS (2026-10-15, team feedback). Frames in BEAMER_FRAMES.tex
show only the reported set (script4_treatment.REPORT_*): T1, T3, T4, E2; county +
year FE; state-clustered SE; CS covariate-adjusted. Frames for what is no longer
reported (A1 pooled state-FE) go to BEAMER_FRAMES_robustness.tex. Estimation is
unchanged. Hard-coded sample figures in frame text describe the 2026-10-12 run.

Output -> LATEX_OUT/BEAMER_FRAMES.tex, LATEX_OUT/BEAMER_FRAMES_robustness.tex

````
