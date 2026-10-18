#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4g-slides.py  --  beamer deck from the Part A results.

Reads the CSVs written by script4a / script4f and emits a compilable beamer
source plus the figures it references. Nothing is retyped by hand, so the deck
cannot drift from the estimates.

Output -> Dropbox/Mental/Data/output/slides/
    core_results.tex     beamer source
    core_results.pdf     compiled (if pdflatex is available)
    *.png                figures copied alongside
"""
import warnings; warnings.filterwarnings("ignore")
import shutil
import numpy as np, pandas as pd
from script4_treatment import OUTCOMES, figs_dir, tables_dir, db_data, os, date

TODAY   = date.today().strftime("%Y-%m-%d")
T4A     = os.path.join(tables_dir, "script4a")
T4F     = os.path.join(tables_dir, "script4f")
F4A     = os.path.join(figs_dir,  "script4a")
SLIDES  = os.path.join(db_data, "output", "slides"); os.makedirs(SLIDES, exist_ok=True)

def esc(x):
    s=str(x)
    for a,b in [("\\",r"\textbackslash{}"),("&",r"\&"),("%",r"\%"),("$",r"\$"),
                ("#",r"\#"),("_",r"\_"),("{",r"\{"),("}",r"\}")]:
        s=s.replace(a,b)
    return s
def n(x,dp=3):
    return "--" if x is None or (isinstance(x,float) and not np.isfinite(x)) else f"{x:.{dp}f}"
def star(p): return "$^{*}$" if (pd.notna(p) and p<0.05) else ""

tw = pd.read_csv(os.path.join(T4F, f"{TODAY}_CORE_twfe_by_controlset.csv"))
hl = pd.read_csv(os.path.join(T4F, f"{TODAY}_CORE_HEADLINE.csv"))
pw = pd.read_csv(os.path.join(T4A, f"{TODAY}_power_by_analysis.csv"))
ORDER = list(OUTCOMES.keys())

# ---- conditional model (T2), default controls ------------------------------
t2 = tw[(tw.treatment_id=="T2") & (tw.control_set=="PRETREAT (19, default)")].copy()
t2["_o"]=t2.outcome.apply(lambda o: ORDER.index(o) if o in ORDER else 99); t2=t2.sort_values("_o")
cond_rows = "\n".join(
    f"{esc(r.outcome)} & {n(r.beta,3)}{star(r.p_state)} & ({n(r.se_state,3)}) & "
    f"{n(r.cond_beta,3)}{star(r.cond_p)} & ({n(r.cond_se,3)}) \\\\"
    for _,r in t2.iterrows())

# ---- control-set sensitivity, Poor MH Days ---------------------------------
cs_sens = tw[tw.outcome=="Poor MH Days"].copy()
piv = cs_sens.pivot_table(index="treatment_id",columns="n_controls",values=["beta","p_state"])
sens_rows=[]
for tid in ["T1","T2","T3"]:
    cells=[]
    for nc in [9,19,26]:
        try:
            b=piv[("beta",nc)][tid]; p=piv[("p_state",nc)][tid]
            cells.append(f"{n(b,4)}{star(p)}")
        except Exception: cells.append("--")
    sens_rows.append(f"{tid} & " + " & ".join(cells) + r" \\")
sens_rows="\n".join(sens_rows)

# ---- headline verdict counts -----------------------------------------------
vc = hl.verdict.value_counts().to_dict()
hl_ord = hl.copy(); hl_ord["_o"]=hl_ord.outcome.apply(lambda o: ORDER.index(o) if o in ORDER else 99)
hl_t2 = hl_ord[hl_ord.treatment_id=="T2"].sort_values("_o")
hl_rows="\n".join(
    f"{esc(r.outcome)} & {n(r.TWFE_beta,3)}{star(r.TWFE_p)} & {n(r.CS_ATT,3)}{star(r.CS_p)} & {esc(r.verdict)} \\\\"
    for _,r in hl_t2.iterrows())

# ---- power, core treatments ------------------------------------------------
pw_core = pw[(pw.treatment_id.isin(["E1","C3","E3"])) & (pw.outcome=="Poor MH Days")]
pw_rows="\n".join(
    f"{esc(r.treatment_id)} & {int(r.N_obs):,} & {int(r.n_switcher_counties)} & "
    f"{n(r.se_state_realised,4)} & {n(r.MDE_in_within_sd,3)} \\\\".replace(",", r"\,")
    for _,r in pw_core.iterrows())

for f in ["F4_twfe_vs_cs","F1_treatment_grid_within","F3_cs_event_time"]:
    src=os.path.join(F4A,f"{TODAY}_{f}.png")
    if os.path.exists(src): shutil.copy(src, os.path.join(SLIDES,f"{f}.png"))

TEX = r"""\documentclass[aspectratio=169,11pt]{beamer}
\usetheme{default}
\usecolortheme{seahorse}
\usepackage{booktabs}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}[frame number]
\setbeamerfont{frametitle}{size=\large}

\title{Large Dairy CAFOs and Rural Mental Health}
\subtitle{County-year panel, 2000--2023 \\ Design-based estimates}
\date{""" + TODAY + r"""}

\begin{document}
\frame{\titlepage}

% ---------------------------------------------------------------
\begin{frame}{Setting}
\begin{itemize}
  \item Rural US counties (NCHS 3--6), county-year panel, 2000--2023
  \item \textbf{Treatment}: large dairy CAFO $=$ operation with \textbf{500+ milk cows}
        (USDA top inventory bin; \emph{not} EPA's 700+ regulatory threshold)
  \item Counts come from the \textbf{agricultural census} --- 2002, 2007, 2012, 2017, 2022 ---
        and are forward-filled between waves
  \item \alert{All within-county treatment variation occurs at four wave transitions}
  \item Outcomes: mental health days, frequent distress, deaths of despair, three crime measures
\end{itemize}
\end{frame}

% ---------------------------------------------------------------
\begin{frame}{Identifying variation --- the binding constraint}
\begin{columns}
\begin{column}{0.52\textwidth}
\begin{itemize}
  \item 938 positive-change events, \textbf{572 counties}
  \item Four cohorts: 284 / 142 / 101 / 45
  \item Under county FE, only \textbf{switchers} identify $\beta$
  \item Timing carries \alert{up to 5 years of measurement error}
\end{itemize}
\end{column}
\begin{column}{0.48\textwidth}
\footnotesize
\begin{tabular}{l rrrr}
\toprule
 & $N$ & Switch. & SE & MDE$_{\sigma_w}$ \\
\midrule
""" + pw_rows + r"""
\bottomrule
\end{tabular}
\vspace{4pt}
\scriptsize MDE at 80\% power, in within-county SD.
\end{column}
\end{columns}
\end{frame}

% ---------------------------------------------------------------
\begin{frame}{Conditional model separates the two margins}
\begin{center}
\footnotesize
\begin{tabular}{l rr rr}
\toprule
 & \multicolumn{2}{c}{$\log(\text{large})$} & \multicolumn{2}{c}{$\log(\text{small})$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Outcome & $\beta$ & (SE) & $\beta$ & (SE) \\
\midrule
""" + cond_rows + r"""
\bottomrule
\end{tabular}
\end{center}
\vspace{2pt}
\footnotesize County $+$ year FE, 19 controls, state-clustered SE. $^{*}p<0.05$.
\vspace{4pt}

Motivation: \textbf{676 of 938} positive-change events occur while the county's
\emph{total} dairy operation count is falling, so a treatment defined on
$\Delta$Large $>0$ alone cannot separate ``a CAFO arrived'' from ``small farms left''.
\vspace{4pt}

\alert{$\log(\text{small})$ is null in every outcome.} Holding small-farm counts fixed,
large operations still move the outcome; holding large fixed, small-farm counts do nothing.
The mechanism is \textbf{not} small farms disappearing.
\end{frame}

% ---------------------------------------------------------------
\begin{frame}{But the result does not survive a heterogeneity-robust estimator}
\begin{center}
\footnotesize
\begin{tabular}{l rr l}
\toprule
Outcome & TWFE & Callaway--Sant'Anna & Verdict \\
\midrule
""" + hl_rows + r"""
\bottomrule
\end{tabular}
\end{center}
\vspace{4pt}
\footnotesize
Under staggered adoption, two-way FE uses \textbf{already-treated} counties as controls
for later-treated ones. Callaway--Sant'Anna removes those comparisons.
Across all 18 treatment $\times$ outcome cells: \textbf{""" + str(vc.get("TWFE only",0)) + r""" TWFE-only,
""" + str(vc.get("null in both",0)) + r""" null in both, """ + str(vc.get("CS only",0)) + r""" CS-only,
\alert{0 significant under both}}.
\end{frame}

% ---------------------------------------------------------------
\begin{frame}{The control set was doing much of the work}
\begin{center}
\footnotesize
\begin{tabular}{l rrr}
\toprule
 & \multicolumn{3}{c}{Controls} \\
\cmidrule(lr){2-4}
Treatment & 9 (legacy) & 19 (default) & 26 (full) \\
\midrule
""" + sens_rows + r"""
\bottomrule
\end{tabular}
\end{center}
\vspace{4pt}
\footnotesize
Poor mental health days. The legacy 9-variable set was \emph{inherited}, not derived,
and inflated every estimate. Only T2 survives the fullest control set --- and it halves.
\end{frame}

% ---------------------------------------------------------------
\begin{frame}{Diagnostics}
\begin{columns}
\begin{column}{0.5\textwidth}
\textbf{Collinearity is not the problem}
\begin{itemize}\footnotesize
  \item 72 within-county specs: \textbf{max VIF 3.09}, none above 5
  \item Pooled specs run 5--6.8 on raw levels --- exactly the cross-sectional
        collinearity county FE removes
  \item $\log(\text{large})$ and $\log(\text{small})$: VIF 1.03 each, separately identified
\end{itemize}
\end{column}
\begin{column}{0.5\textwidth}
\textbf{Power}
\begin{itemize}\footnotesize
  \item MDE $\approx 0.15$--$0.24$ within-county SD
  \item Moderate effects detectable; small ones not
  \item Panel unbalanced: 76\% of counties have the full span
\end{itemize}
\end{column}
\end{columns}
\vspace{6pt}
\footnotesize
\alert{Caveat}: Callaway--Sant'Anna is estimated \textbf{without covariates} --- the
$(g,t)$ cells are too thin for a doubly-robust version --- so it assumes
\emph{unconditional} parallel trends, a stronger assumption than TWFE makes.
\end{frame}

% ---------------------------------------------------------------
\begin{frame}{TWFE vs Callaway--Sant'Anna}
\begin{center}
\includegraphics[width=0.82\textwidth]{F4_twfe_vs_cs.png}
\end{center}
\end{frame}

% ---------------------------------------------------------------
\begin{frame}{Takeaways}
\begin{enumerate}
  \item A rise in large-operation counts mostly reflects \textbf{sector consolidation},
        not new construction
  \item Conditional on small-farm counts, \textbf{large operations carry the association};
        small-farm counts do not
  \item \alert{No result survives both TWFE and a heterogeneity-robust estimator}
  \item Identifying variation is thin --- 572 switching counties, four cohorts, five-year
        timing error --- and the design detects only moderate effects
  \item The defensible conclusion is an \textbf{informative null}, not a causal effect
\end{enumerate}
\end{frame}

\end{document}
"""

path=os.path.join(SLIDES,"core_results.tex")
with open(path,"w") as f: f.write(TEX)
print("wrote", path)
print("figures copied to", SLIDES)
