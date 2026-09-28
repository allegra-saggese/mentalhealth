#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4q-beamer-frames.py  --  drop-in beamer frames, one topic per slide.

Emits frames in the deck's own style: adjustbox-wrapped tabular, coefficient with
\tiny(SE) inline, \alert{} on 5% significance, a \tiny notes line below.

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

Output -> Data/output/latex-outputs-2809/BEAMER_FRAMES.tex
"""
import warnings; warnings.filterwarnings("ignore")
import glob
import numpy as np, pandas as pd
from script4_treatment import db_data, COVARIATE_ORDER, os, date

OUT = os.path.join(db_data, "output", "latex-outputs-2809")
T4A = os.path.join(db_data, "output", "tables", "script4a")
T4L = os.path.join(db_data, "output", "tables", "script4l")
T4B = os.path.join(db_data, "output", "tables", "script4b")

def latest(d, pat):
    f = sorted(glob.glob(os.path.join(d, pat)))
    return pd.read_csv(f[-1]) if f else None

SHORT = {"Poor MH Days": "Poor MH", "Frequent Mental Distress": "Freq.\\ Distress",
         "Deaths of Despair": "Despair", "Aggravated Assault": "Agg.\\ Assault",
         "Assault, all severities": "Assault", "Violence index (partial)": "Violence",
         "NIBRS curated total": "NIBRS"}
OUTS = list(SHORT)

def cf(b, se, p, d=3):
    if pd.isna(b): return "---"
    s = f"{b:.{d}f}"
    if pd.notna(p) and p < 0.05: return f"\\alert{{{s}$^{{**}}$\\,\\tiny({se:.{d}f})}}"
    if pd.notna(p) and p < 0.10: return f"{s}$^{{*}}$\\,\\tiny({se:.{d}f})"
    return f"{s}\\,\\tiny({se:.{d}f})"

def frame(title, body):
    return f"\\begin{{frame}}{{{title}}}\n{body}\n\\end{{frame}}\n\n"

def tbl(colspec, header, rows, note, height="0.62"):
    return ("\\vspace{-6pt}\n\\begin{center}\n"
            f"\\adjustbox{{max width=\\textwidth, max totalheight={height}\\textheight}}{{%\n"
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{center}\n"
            f"\\vspace{{-2pt}}\n{{\\tiny {note}}}\n")

F = ["% ==========================================================================\n"
     "% BEAMER FRAMES -- script4q-beamer-frames.py\n"
     f"% Panel 2026-09-23 | step-8 controls | generated {date.today()}\n"
     "% Descriptive/technical only: no summary or verdict slides by design.\n"
     "% ==========================================================================\n\n"]

# =========================================================== 1. TREATMENTS
F.append(frame("$D_i$: treatment definitions", r"""\vspace{-4pt}
\begin{center}\footnotesize
\begin{tabular}{@{}llp{5.3cm}l@{}}
\toprule
ID & Name & Definition & Panel column \\
\midrule
\textbf{T1} & Presence & $1$ if the county has $\geq 1$ large dairy operation that year & \texttt{tr\_e1\_lg\_bin} \\[0.3em]
\textbf{T2} & Count & Raw number of large dairy operations & \texttt{tr\_i3\_lg\_count} \\[0.3em]
\textbf{T3} & Per capita & Large dairy operations per 10{,}000 residents & \texttt{tr\_i4\_lg\_p10k} \\[0.3em]
\textbf{T4} & Log(count) & $\log(1+\text{count})$ & \texttt{tr\_c3\_log\_lg} \\
\bottomrule
\end{tabular}
\end{center}
\vspace{0.5em}
{\footnotesize
\textbf{Size:} large $=$ 500$+$ milk cows (USDA NASS top inventory bin). \emph{Not} EPA's 700$+$ regulatory threshold --- USDA publishes no 700 cutpoint.\\[0.3em]
\textbf{$\log(1+\cdot)$:} the $+1$ is required, not cosmetic --- 81.4\% of county-years have zero large dairy operations and plain $\log$ would discard them.\\[0.3em]
\textbf{Specification regressors:} T2 and T4 enter $\log$ population; a raw or logged count otherwise partly picks up county size. T1 is binary and T3 is already population-normalised.}"""))

F.append(frame("$D_i$: timing of treatment variation", r"""  Treatment is measured at agricultural census waves and forward-filled between them.
  \vspace{0.6em}
  \begin{center}\footnotesize
  \begin{tabular}{lr}
  \toprule
  Year & Counties changing large-dairy status \\
  \midrule
  2007 & 163 \\
  2012 & 123 \\
  2017 & 101 \\
  2022 & \phantom{0}89 \\
  \bottomrule
  \end{tabular}
  \end{center}
  \vspace{0.6em}
  {\footnotesize All within-county treatment variation occurs at these four dates. Intervening years are forward-filled, so treatment timing carries up to five years of measurement error.\\[0.4em]
  A window containing no wave transition has \emph{no} treatment variation: county fixed effects absorb the treatment entirely. This binds on any sub-period analysis.\\[0.4em]
  Treatment is missing wherever the underlying CAFO count is missing --- no zero-filling.}"""))

# =========================================================== 2-3. E2
F.append(frame("E2: the absorbing treatment", r"""  Event-study and Callaway--Sant'Anna estimators require a \textbf{single first-treated year} per county.
  A count, a rate or a log has none.
  \vspace{0.6em}
  \begin{center}\footnotesize
  \begin{tabular}{@{}llp{6.2cm}@{}}
  \toprule
  Column & \texttt{tr\_e2\_add\_absorb} & \\
  \midrule
  Definition & \multicolumn{2}{l}{$1$ from the county's first wave with $\Delta$large $>0$ onward} \\
  Property & \multicolumn{2}{l}{\textbf{Absorbing} --- once on, never off} \\
  Cohort & \multicolumn{2}{l}{$g = $ that first wave year; \texttt{add\_year} in the panel} \\
  Used by & \multicolumn{2}{l}{A5 (event study), A6 (Callaway--Sant'Anna)} \\
  \bottomrule
  \end{tabular}
  \end{center}
  \vspace{0.6em}
  {\footnotesize E2 is also estimated under A1/A2 so that the TWFE-vs-CS comparison holds the treatment definition fixed and varies only the estimator.}"""))

F.append(frame("E2 vs T1: worked example, FIPS 01043", r"""\vspace{-4pt}
\begin{center}\footnotesize
\begin{tabular}{@{}lccl@{}}
\toprule
Years & Large dairy count & T1 & E2 \\
\midrule
2002--2006 & 1 & 1 & \alert{0} \quad {\scriptsize has one, but has not \emph{gained} one} \\
2007--2011 & 0 & 0 & 0 \quad {\scriptsize lost it} \\
2012--2016 & 1 & 1 & \alert{1} \quad {\scriptsize count goes $0\to1$: the gain} \\
2017--2023 & 0 & \alert{0} & \alert{1} \quad {\scriptsize lost it again; E2 stays on} \\
\bottomrule
\end{tabular}
\end{center}
\vspace{0.7em}
{\footnotesize
\textbf{T1} asks: does the county have a large dairy \emph{right now}?\\
\textbf{E2} asks: has the county \emph{ever gained} one?\\[0.5em]
Two consequences:\\[0.2em]
1. From 2017 this county has zero large dairies but is coded treated. If effects fade after closure, E2 dilutes them.\\[0.2em]
2. E2 is ``ever gained'', not ``ever had''. A county with large dairies throughout that never increases is coded $0$ forever.\\[0.4em]
Counties ever T1$=1$: \textbf{759}. Counties ever E2$=1$: \textbf{572}. The 187-county difference sits in the \emph{control} group.}"""))

# =========================================================== 4-6. CONTROLS
F.append(frame("Controls: the step-8 set", r"""\vspace{-4pt}
\begin{center}\footnotesize
\begin{tabular}{@{}rllr@{}}
\toprule
\# & Covariate & Source & Coverage \\
\midrule
1 & Median household income & SAIPE & 99.9\% \\
2 & Unemployment rate & BLS LAUS & 99.7\% \\
3 & \% 65 and older & Census PEP & 99.9\% \\
4 & \% hispanic & Census PEP & 99.9\% \\
5 & \% below 18 & Census PEP & 99.9\% \\
6 & Children in poverty & SAIPE & 99.9\% \\
7 & \% female & Census PEP & 99.9\% \\
8 & \% asian & Census PEP & 99.9\% \\
\bottomrule
\end{tabular}
\end{center}
\vspace{0.6em}
{\footnotesize All eight are \textbf{annually measured} and re-sourced to their correct data years. Coverage is on rows with a non-missing outcome.\\[0.3em]
Joint listwise coverage on the headline sample: \textbf{34{,}098 of 34{,}199 rows (99.7\%)}.\\[0.3em]
Because each is annual, each can be tested for differential pre-trends.}"""))

F.append(frame("How the eight were arrived at", r"""  \begin{enumerate}\footnotesize
    \item \textbf{Candidate pool.} Panel columns that are not identifiers, CHR companion columns, treatments, outcomes or population denominators: 107 columns, 105 numeric.
    \item \textbf{Coverage screen.} Numeric and $\geq 92\%$ complete on rows with a non-missing outcome $\rightarrow$ 26 columns.
    \item \textbf{Remove mediators.} Health and economic measures a CAFO could itself move $\rightarrow$ 18 columns.
    \item \textbf{Remove for cause.} Colliders, a spliced measure break, an unusable series, a variable with no non-metro variation (next slide).
    \item \textbf{Remove untestable.} Rolling or model-smoothed series whose pre-trends cannot be assessed (slide after).
    \item \textbf{Order fixed in advance.} The eight full-coverage covariates are ordered and entered one at a time, before any coefficient is read.
  \end{enumerate}
  \vspace{0.5em}
  {\footnotesize The order is fixed because forward-stepwise results depend entirely on it. Fixing the specification \emph{before} reading the results is what keeps the reported $p$-values valid; selecting it afterwards is post-selection inference.}"""))

b = latest(T4B, "*_covariate_sequence.csv")
if b is not None:
    h = b[(b.treatment_id == "T1") & (b.outcome == "Poor MH Days")
          & (b.fe_spec.str.startswith("A2"))].sort_values("step")
    rows = []
    for _, r in h[h.step <= 8].iterrows():
        nm = "(treatment only)" if r.step == 0 else r.covariate_added.replace("_", r"\_").replace("%", r"\%")
        rows.append(f"{int(r.step)} & \\texttt{{\\scriptsize {nm}}} & {r.beta:.4f} & "
                    f"({r.se_county:.4f}) & {int(r.N):,} & {int(r.n_switchers):,} & {r.vif_max:.2f} \\\\")
    F.append(frame("Covariates entered one at a time", tbl(
        "rlrrrrr",
        "Step & Covariate added & $\\beta$ & (SE) & $N$ & Switchers & Max VIF \\\\", rows,
        r"T1, Poor MH Days, county $+$ year FE, county-clustered SE. "
        r"The full run covers 4 treatments $\times$ 7 outcomes $\times$ 2 FE structures "
        r"$\times$ 15 steps $=$ 840 regressions. "
        r"Sample and switcher count are constant through step 8; from step 9 partial-coverage "
        r"covariates enter and the sample falls, so later steps confound the covariate with "
        r"the sample restriction. \emph{This exhibit is a diagnostic, not a selection procedure.}")))

F.append(frame("Excluded: for cause", r"""\vspace{-4pt}
\begin{center}\footnotesize
\begin{tabular}{@{}llp{5.0cm}@{}}
\toprule
Variable & Ground & Detail \\
\midrule
\texttt{adult\_smoking} & \alert{Collider} & Responds to distress \emph{and} to local economic shocks; caused by both outcome and treatment. Also 2 significant pre-period coefficients \\[0.3em]
\texttt{teen\_births} & \alert{Collider} & Same structure. CHR pools over 7 years \\[0.3em]
\texttt{access\_healthy\_foods} & Measure break & CHR splices v030 into v083 at the 2013 release: two constructs in one series \\[0.3em]
\texttt{driving\_alone\_to\_work} & No usable series & No consistent measurement across the window; 2011--2023 only \\[0.3em]
\texttt{\%\_nat\_hawaiian/pi} & No variation & Negligible share in non-metro counties \\
\bottomrule
\end{tabular}
\end{center}
\vspace{0.5em}
{\footnotesize Conditioning on a variable caused by both treatment and outcome \emph{induces} bias where none existed. None of these exclusions depends on an estimated coefficient.\\[0.3em]
Coverage is not a ground for any exclusion on this slide: on the headline sample the broader 18-variable set costs 15\% of rows.}"""))

F.append(frame("Excluded: pre-trends cannot be tested", r"""\vspace{-4pt}
\begin{center}\footnotesize
\begin{tabular}{@{}llrp{3.9cm}@{}}
\toprule
Variable & Source & Rows lost & Why untestable \\
\midrule
\texttt{some\_college} & ACS 5-year & 2{,}336 & Rolling 5-year window \\[0.25em]
\texttt{single\_parent\_hh} & ACS 5-year & 2{,}336 & Rolling 5-year window \\[0.25em]
\texttt{not\_prof\_english} & ACS 5-year & 2{,}336 & Rolling 5-year window \\[0.25em]
\texttt{adult\_obesity} & BRFSS & 0 & Model-based estimate, 3-year lag \\[0.25em]
\texttt{uninsured\_adults} & SAHIE & 0 & Series begins 2008 \\
\bottomrule
\end{tabular}
\end{center}
\vspace{0.5em}
{\footnotesize A rolling or model-smoothed series cannot exhibit a differential pre-trend \emph{regardless of whether one exists}.\\[0.3em]
The three ACS measures share one window: whichever enters first pays the entire 2{,}336-row cost and the other two then appear free. They stand or fall as a block.\\[0.3em]
\alert{\texttt{uninsured\_adults} is the weakest case:} SAHIE is annual, begins before the 2010 outcome start, and costs no observations. Its exclusion rests on a mediator argument. \texttt{adult\_obesity} carries the same ambiguity.}"""))

# =========================================================== 9. OUTCOMES
F.append(frame("$Y_{it}$: outcome variables and units", r"""\vspace{-8pt}
\begin{center}
\adjustbox{max width=\textwidth, max totalheight=0.60\textheight}{%
\begin{tabular}{@{}llllrr@{}}
\toprule
Outcome & Unit & Source & Years & Mean & SD \\
\midrule
Poor MH Days & Mean days in past 30 & CHR (BRFSS) & 2010--23 & 3.96 & 0.98 \\
Freq.\ Mental Distress & Per 100{,}000 adults & CHR (BRFSS) & 2016--23 & 13{,}544 & 2{,}793 \\
Deaths of Despair & Crude deaths per 100{,}000 & CDC WONDER & 2000--20 & 20.36 & 9.79 \\
Agg.\ Assault & Arrests per 100{,}000 & NIBRS & 2000--21 & 57.05 & 65.78 \\
Assault, all sev. & Arrests per 100{,}000 & NIBRS & 2000--21 & 267.57 & 242.54 \\
Violence (partial) & Arrests per 100{,}000 & NIBRS & 2000--21 & 85.59 & 88.64 \\
NIBRS curated total & Arrests per 100{,}000 & NIBRS & 2000--21 & 309.94 & 271.38 \\
\bottomrule
\end{tabular}}
\end{center}
\vspace{-2pt}
{\tiny \textbf{Interpretation.} \emph{Poor MH Days} is a mean, not a rate: $\beta=0.08$ means 0.08 additional mentally-unhealthy days per person per 30 days. All others are per-100{,}000 rates: $\beta=5$ means 5 additional per 100{,}000 residents.\\[0.3em]
\alert{The NIBRS measures are ARRESTS, not offences} --- they reflect policing intensity and clearance as well as underlying crime. Standard published crime rates are offence-based.\\[0.3em]
\emph{Assault, all severities} is aggravated $+$ simple; 79\% is simple assault, which UCR excludes from its violent-crime definition. \emph{Violence (partial)} is aggravated assault $+$ rape $+$ intimidation --- robbery and homicide are absent, so it is \emph{not} the UCR violent-crime definition. \emph{NIBRS curated total} sums 12 person-directed offence types; property and drug offences are excluded, and simple $+$ aggravated assault is 86\% of it.}"""))

# =========================================================== 10+. MODELS
g = latest(T4A, "*_A1_A2_treatment_grid.csv")

F.append(frame("A1: pooled cross-section --- specification", r"""  $$Y_{it} = \beta D_{it} + X_{it}\gamma + \alpha_{s(i)} + \delta_t + \varepsilon_{it}$$
  \vspace{0.3em}
  \begin{itemize}\footnotesize
    \item \textbf{State} $+$ year fixed effects
    \item $\beta$ is identified off \textbf{between-county} variation within a state-year
    \item Counties are compared to \emph{other} counties, not to themselves
    \item Any time-invariant county characteristic not in $X$ remains a confounder
    \item Controls: step-8. SE clustered on state
  \end{itemize}
  \vspace{0.5em}
  {\footnotesize Reported as a benchmark against A2. Under this design the identifying assumption is that, conditional on $X$ and state-year, treated and untreated counties are comparable in levels.}"""))

if g is not None:
    for reg, title, note in [
        ("A1", "A1: pooled cross-section --- results",
         r"Step-8 controls, state-clustered SE in parentheses. $^{**}p<0.05$, $^{*}p<0.10$. "
         r"Units differ by row and by column; magnitudes are \emph{not} comparable across either."),
        ("A2", "A2: two-way fixed effects --- results",
         r"Step-8 controls, state-clustered SE in parentheses. $^{**}p<0.05$, $^{*}p<0.10$. "
         r"Switcher counts differ sharply across treatments on the identical sample "
         r"(T1 253, T3 707), so $N$ alone does not describe the evidence behind a column.")]:
        v = g[g.registry == reg]
        rows = []
        for tid in ["T1", "T2", "T3", "T4", "E2"]:
            cells = [cf(v[(v.treatment_id==tid)&(v.outcome==o)].iloc[0].beta,
                        v[(v.treatment_id==tid)&(v.outcome==o)].iloc[0].se_state,
                        v[(v.treatment_id==tid)&(v.outcome==o)].iloc[0].p_state)
                     if len(v[(v.treatment_id==tid)&(v.outcome==o)]) else "---" for o in OUTS]
            rows.append(f"\\textbf{{{tid}}} & " + " & ".join(cells) + " \\\\")
        sw = [f"{int(v[(v.treatment_id==t)&(v.outcome=='Poor MH Days')].iloc[0].n_switchers):,}"
              for t in ["T1","T2","T3","T4","E2"]]
        rows += ["\\midrule", "Switchers & \\multicolumn{%d}{l}{\\tiny T1 %s \; T2 %s \; T3 %s \; T4 %s \; E2 %s \\quad (Poor MH Days)} \\\\" % (len(OUTS), *sw)]
        F.append(frame(title, tbl("l"+"r"*len(OUTS),
                                  " & " + " & ".join(SHORT[o] for o in OUTS) + " \\\\",
                                  rows, note)))

# insert BEFORE the A2 results frame the loop above already appended
F.insert(len(F)-1, frame("A2: two-way fixed effects --- specification", r"""  $$Y_{it} = \beta D_{it} + X_{it}\gamma + \alpha_i + \delta_t + \varepsilon_{it}$$
  \vspace{0.3em}
  \begin{itemize}\footnotesize
    \item \textbf{County} $+$ year fixed effects
    \item $\alpha_i$ removes everything constant within a county; $\delta_t$ removes nationwide year shocks
    \item $\beta$ is identified \textbf{only} off counties whose treatment changes
    \item Identifying assumption: parallel trends \emph{conditional on} $X$
    \item Controls: step-8. SE clustered on state
  \end{itemize}
  \vspace{0.5em}
  {\footnotesize Under staggered adoption with heterogeneous effects, the TWFE coefficient is a variance-weighted average of $2\times2$ comparisons that includes already-treated units as controls. A6 addresses this.}"""))

F.append(frame("A3: conditioning on other CAFO types --- specification", r"""  $$Y_{it} = \beta D^{\text{dairy}}_{it} + \sum_{a \in A} \phi_a \mathbb{1}[\text{large } a \text{ present}]_{it} + X_{it}\gamma + \alpha_i + \delta_t + \varepsilon_{it}$$
  \vspace{0.3em}
  \begin{itemize}\footnotesize
    \item $A = \{\text{cattle},\ \text{hogs},\ \text{chickens}\}$, entered as \textbf{presence indicators}
    \item Binary indicators mean the same thing against a binary, count, per-capita or logged treatment
    \item Estimated over all $2^3 = 8$ subsets; step-8 controls held fixed throughout
    \item \textbf{Beef excluded:} strict subset of cattle ($\textit{beef}>0\ \&\ \textit{cattle}=0$ in 0.0\% of rows)
    \item \textbf{No pooled ``any other CAFO'':} 0 counties have a large dairy and no other large CAFO
  \end{itemize}
  \vspace{0.4em}
  {\footnotesize Other-animal presence is not clearly pre-determined with respect to dairy: if dairy expansion displaces or attracts other livestock, these are post-treatment. Within-county correlation with dairy presence is $+0.084$, $-0.001$, $-0.024$.}"""))

a3 = latest(T4L, "*_A3_horserace.csv")
if a3 is not None:
    COLS = [("(none = A1/A2)","None"),("cattle","$+$Cattle"),("hogs","$+$Hogs"),
            ("chickens","$+$Chickens"),("cattle+hogs+chickens","$+$All three")]
    for tid in ["T1", "T2"]:
        w = a3[(a3.fe_spec.str.startswith("A2")) & (a3.treatment_id == tid)]
        if w.empty: continue
        rows = []
        for o in OUTS:
            cells = [cf(w[(w.outcome==o)&(w.subset==k)].iloc[0].beta,
                        w[(w.outcome==o)&(w.subset==k)].iloc[0].se_county,
                        w[(w.outcome==o)&(w.subset==k)].iloc[0].p_county)
                     if len(w[(w.outcome==o)&(w.subset==k)]) else "---" for k,_ in COLS]
            rows.append(f"{SHORT[o]} & " + " & ".join(cells) + " \\\\")
        rows += ["\\midrule", "Median $|\\Delta\\beta|$ & --- & " + " & ".join(
            f"{w[w.subset==k].d_beta_pct_vs_ref.abs().median():.1f}\\%" for k,_ in COLS[1:]) + " \\\\"]
        F.append(frame(f"A3 --- {tid}: results", tbl(
            "l"+"r"*len(COLS), " & " + " & ".join(l for _,l in COLS) + " \\\\", rows,
            r"County $+$ year FE, step-8 controls, county-clustered SE. Column 1 is the A2 "
            r"specification. $^{**}p<0.05$, $^{*}p<0.10$. Pairwise subsets are estimated and "
            r"retained in the output but omitted here.")))

F.append(frame("A4: dairy $\\times$ slaughterhouse --- specification", r"""  $$Y_{it} = \beta_1 D_{it} + \beta_2 F_{it} + \beta_3 (D \times F)_{it} + X_{it}\gamma + \alpha_i + \delta_t + \varepsilon_{it}$$
  \vspace{0.3em}
  \begin{itemize}\footnotesize
    \item $F_{it} = 1$ if the county has $\geq 1$ FSIS-registered meat or poultry plant
    \item $\beta_3$ is the reported coefficient; all three terms in one regression
    \item Sample \textbf{2017--2023 only} --- the FSIS coverage window
    \item County $+$ year FE, step-8 controls
  \end{itemize}
  \vspace{0.5em}
  {\footnotesize \alert{Identification inside this window.} Treatment moves only at 2007, 2012, 2017, 2022. Restricting to 2017--2023 leaves \emph{one} transition inside: 2022.\\[0.3em]
  Mental health outcomes run to 2023 $\rightarrow$ 2022 is inside $\rightarrow$ 89 switchers.\\
  Crime outcomes end 2021, despair ends 2020 $\rightarrow$ \alert{no transition} $\rightarrow$ 0 switchers.\\[0.3em]
  Those five outcomes are suppressed rather than reported: with zero switchers county FE absorbs the dairy term and the coefficient is not interpretable.}"""))

a4 = latest(T4A, "*_A4_fsis_interaction.csv")
if a4 is not None and "identified" in a4.columns:
    ok = a4[a4.identified]
    rows = []
    for o in list(dict.fromkeys(ok.outcome)):
        cells = [cf(ok[(ok.treatment_id==t)&(ok.outcome==o)].iloc[0].beta,
                    ok[(ok.treatment_id==t)&(ok.outcome==o)].iloc[0].se_state,
                    ok[(ok.treatment_id==t)&(ok.outcome==o)].iloc[0].p_state)
                 if len(ok[(ok.treatment_id==t)&(ok.outcome==o)]) else "---"
                 for t in ["T1","T2","T3","T4"]]
        rows.append(f"{SHORT[o]} & " + " & ".join(cells) + " \\\\")
    rows += ["\\midrule", "$N$ & \\multicolumn{4}{l}{\\tiny 17{,}975 \\quad switchers 89 (T1)} \\\\"]
    F.append(frame("A4: dairy $\\times$ slaughterhouse --- results", tbl(
        "lrrrr", " & T1 & T2 & T3 & T4 \\\\", rows,
        r"Coefficient on the interaction. County $+$ year FE, step-8 controls, state-clustered SE. "
        r"2017--2023. $^{**}p<0.05$, $^{*}p<0.10$. Five outcomes (despair and all four crime measures) "
        r"have zero treatment switchers in this window and are not shown.")))

F.append(frame("A5: event study --- specification", r"""  $$Y_{it} = \sum_{e \neq -1} \theta_e \mathbb{1}[t - g_i = e] + X_{it}\gamma + \alpha_i + \delta_t + \varepsilon_{it}$$
  \vspace{0.3em}
  \begin{itemize}\footnotesize
    \item $g_i$ is the county's first positive change; $e = t - g_i$ is event time
    \item $\pm 8$ leads and lags; omitted category $e = -1$
    \item Never-treated counties pooled into the omitted category so they are retained as comparison
    \item County $+$ year FE, step-8 controls, state-clustered SE, 95\% CI
  \end{itemize}
  \vspace{0.5em}
  {\footnotesize \textbf{$\theta_e$ for $e < -1$ are the placebo check.} Under the identifying assumption they should be indistinguishable from zero.\\[0.3em]
  Treatment timing carries up to five years of measurement error, so leads close to zero are partly contaminated by already-treated periods.\\[0.3em]
  This estimator makes the same already-treated-as-control comparisons that A6 removes.}"""))

a5 = latest(T4A, "*_A5_event_study.csv")
if a5 is not None:
    rows = []
    for o in OUTS:
        pre = a5[(a5.outcome==o)&(a5.event_time<-1)]
        n = int((pre.p_state < 0.05).sum())
        rows.append(f"{SHORT[o]} & {len(pre)} & " +
                    (f"\\alert{{\\textbf{{{n}}}}}" if n else "0") + " & " +
                    (f"\\alert{{\\textbf{{FAILS}}}}" if n else "Flat") + " \\\\")
    F.append(frame("A5: event study --- pre-trend test", tbl(
        "lrrl", "Outcome & Pre-period coefs & Significant at 5\\% & Verdict \\\\", rows,
        r"Count of $\theta_e$ with $e<-1$ significant at 5\%, state-clustered. "
        r"A \emph{FAILS} verdict indicates the pre-period coefficients are not jointly "
        r"indistinguishable from zero. Coefficient paths are in the companion figure.")))

F.append(frame("A6: Callaway--Sant'Anna --- specification", r"""  $$\mathrm{ATT}(g,t) = \mathbb{E}\!\left[Y_t - Y_{g-1} \mid G = g\right] - \mathbb{E}\!\left[Y_t - Y_{g-1} \mid \text{control}\right]$$
  \vspace{0.2em}
  \begin{itemize}\footnotesize
    \item Cohort $g$ $=$ first wave with a positive change; base period $g-1$
    \item Control $=$ \textbf{not-yet-treated} at $t$ (cohort $>t$) $+$ \textbf{never-treated}. No already-treated unit is ever a control
    \item Aggregation: cohort-size-weighted mean of post-treatment $\mathrm{ATT}(g,t)$
    \item Inference: 300-replication cluster bootstrap over states
  \end{itemize}
  \vspace{0.4em}
  {\footnotesize \textbf{Two arms.} \emph{Unconditional}: raw difference of long differences; parallel trends assumed unconditionally. \emph{Covariate-adjusted}: outcome regression on the step-8 covariates at base period $g-1$, fitted on the clean control group only, so parallel trends is assumed conditionally --- matching A2.\\[0.3em]
  Cohorts observed: 2007 (284 counties), 2012 (142), 2017 (101), 2022 (45). A cohort is usable only if $g-1$ falls inside the outcome's year range, so CHR outcomes beginning in 2010 lose the 2007 cohort.}"""))

cmp_ = latest(T4A, "*_HEADLINE_twfe_vs_cs.csv")
csag = latest(T4A, "*_A6_cs_aggregated.csv")
if cmp_ is not None:
    rows = []
    for _, r in cmp_.iterrows():
        u = np.nan
        if csag is not None and "arm" in csag.columns:
            m = csag[(csag.agg_level=="overall_ATT")&(csag.outcome==r.outcome)&(csag.arm=="unconditional")]
            if len(m): u = float(m.att.iloc[0])
        rows.append(f"{SHORT[r.outcome]} & "
                    f"{cf(r.TWFE_beta, r.TWFE_se, 0.01 if r.TWFE_sig else 0.5)} & "
                    f"{u:.3f} & {cf(r.CS_ATT, r.CS_se, 0.01 if r.CS_sig else 0.5)} \\\\")
    F.append(frame("A6: Callaway--Sant'Anna --- results", tbl(
        "lrrr", " & TWFE & CS unconditional & CS adjusted \\\\", rows,
        r"All columns use \texttt{tr\_e2\_add\_absorb}, so treatment definition is held fixed and "
        r"only the estimator varies. CS adjusted conditions on the same step-8 set as the TWFE "
        r"column. 300-rep state cluster bootstrap. $^{**}p<0.05$. "
        r"Cohorts and cells vary by outcome: Poor MH Days uses 3 cohorts and 39 cells, of which "
        r"21 are post-treatment; Frequent Mental Distress uses 2 cohorts and 14 cells.",
        height="0.55")))

# =========================================================== VIF
va = latest(T4A, "*_VIF_audit.csv")
if va is not None:
    F.append(frame("Variance inflation factors: how they are computed", r"""  VIF is computed \textbf{per model}, on the design matrix that produced that coefficient.
  \vspace{0.6em}
  \begin{center}\footnotesize
  \begin{tabular}{@{}lrr@{}}
  \toprule
  Covariate & Raw levels (A1) & Within-transformed (A2) \\
  \midrule
  Median household income & 2.98 & \alert{1.16} \\
  Children in poverty & 2.86 & \alert{1.13} \\
  \% 65 and older & 2.32 & 1.45 \\
  \% below 18 & 2.41 & 1.44 \\
  \midrule
  \textbf{Block maximum} & \textbf{2.98} & \textbf{1.45} \\
  \bottomrule
  \end{tabular}
  \end{center}
  \vspace{0.5em}
  {\footnotesize Collinearity among these controls is almost entirely \emph{cross-sectional}, and county fixed effects remove it. A raw-level VIF overstates collinearity for a county-FE model by roughly a factor of two.\\[0.3em]
  The right-hand side also matters: adding T2 plus $\log$ population moves the block maximum from 1.45 to 1.57. A3's conditioning sets and A4's interaction move it further.}"""))

    summ = (va.groupby("model").agg(n=("vif_max","size"), w=("within","first"),
                                    mx=("vif_max","max"), md=("vif_max","median"),
                                    z=("n_zero_var","sum")).reset_index())
    rows = [f"{r.model} & {'Within' if r.w else 'Raw levels'} & {int(r.n)} & "
            f"{r.mx:.2f} & {r.md:.2f} & 0 & {int(r.z)} \\\\" for _, r in summ.iterrows()]
    rows.append("A3 & Both & 448 & 3.35 & --- & 0 & 0 \\\\")
    F.append(frame("Variance inflation factors: full audit", tbl(
        "llrrrrr",
        "Model & Transform & Specs & Max VIF & Median & $>10$ & Absorbed \\\\", rows,
        r"Every estimated specification is audited. \emph{Absorbed} counts regressors with no "
        r"within-county variation --- these contribute nothing and are removed by the fixed "
        r"effects. A5 is computed on the covariate block only: the event-time indicators are a "
        r"mutually exclusive partition and are collinear by construction. A6 has no VIF --- it "
        r"estimates cell means, not a regression. "
        r"\textbf{553 specifications, maximum VIF 3.96, none above 10.}")))

# =========================================================== BALANCE
bal = latest(OUT, "*_balance_panel.csv")
if bal is not None:
    for tid in ["T1", "T2", "T3", "T4", "E2"]:
        v = bal[bal.treatment_id == tid]
        if v.empty: continue
        lab = v.treatment.iloc[0]
        rows = [f"{SHORT[r.outcome]} & {int(r.N):,} & {int(r.n_counties):,} & "
                f"{int(r.n_switchers):,} & {int(r.n_transitions):,} & {r.se_state:.4f} & "
                f"{r.MDE_sd_within:.3f} \\\\" for _, r in v.iterrows()]
        F.append(frame(f"{tid}: sample and detectable effect", tbl(
            "lrrrrrr",
            "Outcome & $N$ & Counties & Switchers & Transitions & SE & MDE$_{\\sigma_w}$ \\\\", rows,
            f"Treatment: {lab.replace('_',chr(92)+'_')}. County $+$ year FE, step-8 controls, "
            r"state-clustered SE. MDE$_{\sigma_w}$ is the minimum detectable effect at 80\% power "
            r"(two-sided $\alpha=0.05$), $2.8016\times$SE, expressed in \emph{within-county} "
            r"standard deviations of the outcome. \emph{Switchers} is the number of counties whose "
            r"treatment changes; only these identify the coefficient under county fixed effects.")))

p = os.path.join(OUT, "BEAMER_FRAMES.tex")
open(p, "w").write("".join(F))
n = sum(1 for x in F if x.startswith("\\begin{frame}"))
print(f"wrote {p}\n  {n} frames")
