#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4p-full-appendix.py  --  ONE document: every regression and balance table.

Combines the fragments written by
    script4d-balance-panel.py     -> _BODY_balance.tex
    script4n-regression-tables.py -> _BODY_regressions.tex
plus the two covariate-exclusion tables, into a single compilable appendix.

NO estimation and NO table construction happens here. The tables are built once,
in the scripts that own them, and this file only orders and frames them. Editing
a table means editing 4d or 4n and re-running; nothing here needs to change.

RUN ORDER
---------
    python3 script4a-design-estimates.py      # A1, A2, A4, A5, A6
    python3 script4l-a3-horserace.py          # A3
    python3 script4d-balance-panel.py         # balance fragment
    python3 script4n-regression-tables.py     # regression fragment
    python3 script4p-full-appendix.py         # this file

Output -> Data/output/latex-outputs-2809/FULL_RESULTS_APPENDIX.tex / .pdf
"""
import warnings; warnings.filterwarnings("ignore")
import subprocess
from script4_treatment import db_data, os, date

OUT = os.path.join(db_data, "output", "latex-outputs-2809")
TODAY = date.today().strftime("%Y-%m-%d")

FRAGMENTS = {
    "balance":     os.path.join(OUT, "_BODY_balance.tex"),
    "regressions": os.path.join(OUT, "_BODY_regressions.tex"),
    "excl_cause":  os.path.join(OUT, "TABLE_covariates_EXCLUDED_FOR_CAUSE.tex"),
    "excl_pretr":  os.path.join(OUT, "TABLE_covariates_EXCLUDED_NO_PRETREND_TEST.tex"),
}
missing = [k for k, v in FRAGMENTS.items() if not os.path.exists(v)]
if missing:
    raise SystemExit(f"missing fragments: {missing}\nRun the scripts listed in the docstring first.")

def read(k):
    return open(FRAGMENTS[k]).read()

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in,landscape]{geometry}
\usepackage{booktabs,adjustbox,amsmath,longtable}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue]{hyperref}
\setlength{\parindent}{0pt}
\begin{document}

\begin{titlepage}
\centering
\vspace*{2cm}
{\Huge\bfseries Results appendix\par}
\vspace{0.8cm}
{\Large Large dairy CAFOs, rural mental health and crime\par}
\vspace{2cm}
{\large
\begin{tabular}{rl}
Panel        & 2026-09-23 (CHR year misalignment corrected) \\
Sample       & rural US counties, NCHS codes 3--6, 2000--2023 \\
Treatments   & T1--T4, plus E2 for the staggered-DiD estimators \\
Outcomes     & 7 (3 mental health / mortality, 4 crime) \\
Controls     & step-8 pre-committed set \\
Inference    & state-clustered standard errors throughout \\
\end{tabular}\par}
\vspace{2cm}
{\large\today\par}
\vfill
{\small\itshape Stars: $^{*}\,p<0.10$, $^{**}\,p<0.05$.\par}
\end{titlepage}

\tableofcontents
\clearpage

\section{How to read this appendix}

\textbf{The headline specification is A2} --- two-way fixed effects, county $+$ year,
with the eight pre-committed step-8 covariates and state-clustered standard errors.
A1 (pooled) is a benchmark, never a headline: it is identified off between-county
variation and is far more sensitive to the control set.

\textbf{The step-8 control set was fixed before the results were read.} Choosing a
specification after seeing which one gives the nicest answer is post-selection
inference and invalidates the reported $p$-values. Section~\ref{sec:controls}
states why each excluded covariate is excluded.

\textbf{Three diagnostics qualify almost everything here}, and they do not agree
with each other:
\begin{itemize}
  \item \textbf{Pre-trends (A5).} Three of the four crime outcomes FAIL. Counties
        that later gain a large dairy were already on a different crime trajectory.
  \item \textbf{Estimator (A6).} Six of seven outcomes are significant under TWFE
        and \emph{none} survives Callaway--Sant'Anna once both sides condition on
        the same covariates. Under staggered adoption TWFE uses already-treated
        counties as controls for later-treated ones; those comparisons can carry
        negative weights.
  \item \textbf{Identification (A4).} Five of seven outcomes have zero treatment
        switchers inside the 2017--2023 FSIS window and are suppressed rather than
        reported as nulls.
\end{itemize}
The mental-health outcomes have clean pre-trends but fail the estimator test; the
crime outcomes fail the pre-trend test. \textbf{No outcome passes both.}

\textbf{Switcher counts are printed on every row.} Under county fixed effects only
counties whose treatment changes identify $\beta$. A row with 128 switchers is not
the same evidence as one with 707, whatever $N$ says.

\clearpage
\section{Regression results, models A1--A6}
"""

MID_BALANCE = r"""
\clearpage
\section{Balance and detectable effect}

Each table reports, for one treatment across all seven outcomes: the estimating
sample, the treated/control split, the number of counties that actually switch,
and the minimum detectable effect at 80\% power expressed in within-county standard
deviations of the outcome.

\textbf{MDE is the honest answer to ``is this a null or is it underpowered?''}
An MDE of 0.15$\sigma_w$ means effects smaller than that could not have been
detected at conventional power, so a non-significant coefficient below that
magnitude is uninformative rather than evidence of no effect.

For the continuous treatments (T2, T3, T4) the treated/control split is
\emph{descriptive only} --- the estimator uses the full continuous variation, not a
dichotomy.
"""

MID_CONTROLS = r"""
\clearpage
\section{Control set: what is included and why}\label{sec:controls}

The headline specification uses eight covariates: median household income,
unemployment, \% 65 and older, \% hispanic, \% below 18, children in poverty,
\% female, and \% asian. All eight are annually measured at 99.7--99.9\% coverage
on the estimating sample, so the sample stays intact and each can be tested for
pre-trends.

\textbf{Coverage is not the reason the others are excluded.} On the headline sample
the broader 18-variable set costs only 15\% of rows. Each excluded covariate is
excluded for a stated substantive reason, given in the two tables below.
"""

doc = "\n".join([
    PREAMBLE,
    read("regressions"),
    MID_BALANCE,
    read("balance"),
    MID_CONTROLS,
    read("excl_cause"),
    read("excl_pretr"),
    r"\end{document}",
])

p = os.path.join(OUT, "FULL_RESULTS_APPENDIX.tex")
open(p, "w").write(doc)
print(f"wrote {p}")

# two passes so \tableofcontents resolves
for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-output-directory", OUT, p],
                   capture_output=True, text=True)
pdf = os.path.join(OUT, "FULL_RESULTS_APPENDIX.pdf")
if os.path.exists(pdf):
    print(f"compiled -> {pdf}")
else:
    print("PDF COMPILE FAILED")
for ext in (".aux", ".log", ".out", ".toc"):
    f = os.path.join(OUT, "FULL_RESULTS_APPENDIX" + ext)
    if os.path.exists(f):
        os.remove(f)
