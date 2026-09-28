#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4n-regression-tables.py  --  regression tables, journal layout.

Reads the CSVs written by script4a (A1, A2, A4, A5, A6) and script4l (A3).
NO estimation happens here.

LAYOUT
------
Standard published-paper form throughout:
    COLUMNS are specifications, numbered (1) (2) (3) ...
    ROWS are coefficients, with the standard error in parentheses DIRECTLY BELOW
    a bottom panel carries N, counties, switchers, fixed effects and controls

This replaces the earlier long-format dump, in which every treatment x outcome x
specification was one row. That format is readable by a machine and by nobody
else -- the A3 table alone ran to 56 rows and no coefficient could be compared
against its neighbour without counting lines.

STARS: * p<0.10, ** p<0.05.  SEs state-clustered throughout.

A3 IN PARTICULAR
----------------
One table per treatment, columns (1)-(5):
    (1) no conditioning -- identical to the A2 specification
    (2) + large cattle present
    (3) + large hogs present
    (4) + large chickens present
    (5) + all three
Reading across a row shows what happens to the dairy coefficient as other animal
types are conditioned on, which is the entire question A3 asks. The three
pairwise subsets are estimated and kept in the CSV but omitted from the table:
they carry no information the single and full columns do not.

Output -> Data/output/latex-outputs-2809/REGRESSION_TABLES.tex / .pdf
          plus _BODY_regressions.tex for the combined appendix (script4p)
"""
import warnings; warnings.filterwarnings("ignore")
import glob, subprocess
import numpy as np, pandas as pd
from script4_treatment import db_data, os, date

OUT = os.path.join(db_data, "output", "latex-outputs-2809"); os.makedirs(OUT, exist_ok=True)
T4A = os.path.join(db_data, "output", "tables", "script4a")
T4L = os.path.join(db_data, "output", "tables", "script4l")

def latest(d, pat):
    f = sorted(glob.glob(os.path.join(d, pat)))
    return pd.read_csv(f[-1]) if f else None

def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

def cell(beta, se, p, fmt="{:.4f}"):
    """A coefficient and its SE, as the two stacked lines a paper table uses."""
    if pd.isna(beta):
        return "", ""
    st = "$^{**}$" if (pd.notna(p) and p < 0.05) else ("$^{*}$" if (pd.notna(p) and p < 0.10) else "")
    return fmt.format(beta) + st, f"({fmt.format(se)})" if pd.notna(se) else ""

def table(caption, label, ncol, colhead, rows, foot, note):
    spec = "l" + "r" * ncol
    nums = " & ".join(f"({i+1})" for i in range(ncol))
    return ("\\begin{table}[htbp]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\begin{adjustbox}{max width=\\textwidth, max totalheight=0.78\\textheight}\n"
            f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
            f" & {nums} \\\\\n"
            f" & {colhead} \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\midrule\n" + "\n".join(foot) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\n"
            "\\vspace{0.3em}\n\\begin{minipage}{\\textwidth}\\footnotesize\n"
            f"\\textit{{Notes.}} {note}\n"
            "\\end{minipage}\n\\end{table}\n")

tex = [r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs,adjustbox,amsmath,graphicx}
\usepackage[colorlinks=true,linkcolor=blue]{hyperref}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}
{\Large\bfseries Regression results, models A1--A6}\\[0.3em]
{\large Large dairy CAFOs and rural mental health}\\[0.6em]
Panel 2026-09-23 \quad$\cdot$\quad step-8 pre-committed controls \quad$\cdot$\quad
state-clustered SEs\\[0.2em]
\today
\end{center}
\vspace{0.5em}
\textit{Stars:} $^{*}\,p<0.10$, $^{**}\,p<0.05$. Standard errors in parentheses.
\vspace{1em}
"""]

TRT = ["T1", "T2", "T3", "T4", "E2"]

# ============================================================ A1 / A2
g = latest(T4A, "*_A1_A2_treatment_grid.csv")
if g is not None:
    for reg, nm, fe in [("A2", "A2 --- TWFE within county", "County $+$ year"),
                        ("A1", "A1 --- Pooled OLS", "State $+$ year")]:
        v = g[g.registry == reg]
        rows = []
        for okey in v.outcome.unique():
            b_line, s_line = [esc(okey)], [""]
            for tid in TRT:
                r = v[(v.treatment_id == tid) & (v.outcome == okey)]
                if r.empty:
                    b_line.append(""); s_line.append(""); continue
                r = r.iloc[0]
                b, s = cell(r.beta, r.se_state, r.p_state)
                b_line.append(b); s_line.append(s)
            rows.append(" & ".join(b_line) + " \\\\")
            rows.append(" & ".join(s_line) + " \\\\")
            rows.append("\\addlinespace")
        foot = []
        for lbl, key in [("Observations", "N"), ("Counties", "n_counties"),
                         ("Switchers", "n_switchers")]:
            vals = []
            for tid in TRT:
                r = v[(v.treatment_id == tid) & (v.outcome == "Poor MH Days")]
                vals.append(f"{int(r.iloc[0][key]):,}" if len(r) else "")
            foot.append(f"{lbl} (Poor MH Days) & " + " & ".join(vals) + " \\\\")
        foot.append(f"Fixed effects & " + " & ".join([fe.split()[0]] * len(TRT)) + " \\\\")
        foot.append("Step-8 controls & " + " & ".join(["Yes"] * len(TRT)) + " \\\\")
        tex.append(table(
            f"{nm} ({fe.lower()} fixed effects)", f"tab:{reg}", len(TRT),
            " & ".join(f"\\textbf{{{t}}}" for t in TRT), rows, foot,
            "Each column is a separate regression; each cell is the treatment coefficient "
            "for that treatment--outcome pair. "
            "T1 presence, T2 count, T3 per 10{,}000 residents, T4 $\\log(1+$count$)$, "
            "E2 first positive change (absorbing). T2 and T4 additionally control for "
            "$\\log$ population. Sample sizes vary by outcome; the bottom panel reports "
            "them for the headline outcome. "
            + ("\\textit{Switchers} is the number of counties whose treatment changes --- "
               "under county fixed effects only these identify $\\beta$."
               if reg == "A2" else
               "A1 is a \\textit{benchmark}, not a headline: identified off between-county "
               "variation and far more sensitive to the control set than A2.")))

# ============================================================ A3
a3 = latest(T4L, "*_A3_horserace.csv")
if a3 is not None:
    COLS = [("(none = A1/A2)", "None"), ("cattle", "$+$ Cattle"), ("hogs", "$+$ Hogs"),
            ("chickens", "$+$ Chickens"), ("cattle+hogs+chickens", "$+$ All three")]
    w = a3[a3.fe_spec.str.startswith("A2")]
    for tid in ["T1", "T2", "T3", "T4"]:
        v = w[w.treatment_id == tid]
        if v.empty:
            continue
        tlab = v.treatment.iloc[0]
        rows = []
        for okey in v.outcome.unique():
            b_line, s_line = [esc(okey)], [""]
            for key, _ in COLS:
                r = v[(v.outcome == okey) & (v.subset == key)]
                if r.empty:
                    b_line.append(""); s_line.append(""); continue
                r = r.iloc[0]
                b, s = cell(r.beta, r.se_county, r.p_county)
                b_line.append(b); s_line.append(s)
            rows.append(" & ".join(b_line) + " \\\\")
            rows.append(" & ".join(s_line) + " \\\\")
            rows.append("\\addlinespace")
        base = v[v.subset == "(none = A1/A2)"]
        full = v[v.subset == "cattle+hogs+chickens"]
        foot = ["Median $|\\Delta\\beta|$ vs (1) & --- & "
                + " & ".join(
                    f"{v[v.subset==k].d_beta_pct_vs_ref.abs().median():.1f}\\%"
                    for k, _ in COLS[1:]) + " \\\\",
                f"Observations & " + " & ".join(
                    f"{int(v[v.subset==k].N.median()):,}" for k, _ in COLS) + " \\\\",
                f"Switchers & " + " & ".join(
                    f"{int(v[v.subset==k].n_switchers.median()):,}" for k, _ in COLS) + " \\\\",
                "Fixed effects & " + " & ".join(["County"] * len(COLS)) + " \\\\",
                "Step-8 controls & " + " & ".join(["Yes"] * len(COLS)) + " \\\\"]
        tex.append(table(
            f"A3 --- {tid}: dairy conditional on other large CAFO types",
            f"tab:A3_{tid}", len(COLS),
            " & ".join(lab for _, lab in COLS), rows, foot,
            f"Treatment: {esc(tlab)}. Column (1) is the A2 specification; columns (2)--(5) add "
            "indicators for the presence of other large operations. County $+$ year fixed "
            "effects, step-8 controls, county-clustered SEs. "
            "\\textbf{The near-null is the expected result and was registered before the run:} "
            "within-county correlation between dairy presence and the three indicators is "
            "$+0.084$, $-0.001$ and $-0.024$, so there is little for them to absorb. It is "
            "evidence the within estimate is not capturing ``any large CAFO''. "
            "Beef is excluded --- it is a strict subset of cattle "
            "($\\textit{beef}>0\\ \\&\\ \\textit{cattle}=0$ in 0.0\\% of rows). "
            "Other-animal presence is not clearly pre-determined with respect to dairy, so A3 "
            "is a probe on what the coefficient contains, not a better-identified "
            "specification. Pairwise subsets are estimated and retained in the output CSV."))

# ============================================================ A4
a4 = latest(T4A, "*_A4_fsis_interaction.csv")
if a4 is not None and "identified" in a4.columns:
    ok = a4[a4.identified]
    outs = list(dict.fromkeys(ok.outcome))
    rows = []
    for okey in outs:
        b_line, s_line = [esc(okey)], [""]
        for tid in ["T1", "T2", "T3", "T4"]:
            r = ok[(ok.treatment_id == tid) & (ok.outcome == okey)]
            if r.empty:
                b_line.append(""); s_line.append(""); continue
            r = r.iloc[0]
            b, s = cell(r.beta, r.se_state, r.p_state)
            b_line.append(b); s_line.append(s)
        rows.append(" & ".join(b_line) + " \\\\")
        rows.append(" & ".join(s_line) + " \\\\")
        rows.append("\\addlinespace")
    foot = ["Observations & " + " & ".join(
                f"{int(ok[ok.treatment_id==t].N.median()):,}" for t in ["T1","T2","T3","T4"]) + " \\\\",
            "Switchers & " + " & ".join(
                f"{int(ok[ok.treatment_id==t].n_switchers.median()):,}" for t in ["T1","T2","T3","T4"]) + " \\\\",
            "Fixed effects & " + " & ".join(["County"] * 4) + " \\\\",
            "Sample & " + " & ".join(["2017--23"] * 4) + " \\\\"]
    sup = sorted(a4.loc[~a4.identified, "outcome"].unique())
    tex.append(table(
        "A4 --- Dairy $\\times$ FSIS slaughterhouse interaction", "tab:A4", 4,
        " & ".join(f"\\textbf{{{t}}}" for t in ["T1","T2","T3","T4"]), rows, foot,
        "Reported coefficient is on the interaction; dairy and FSIS main effects are in the "
        "same regression. \\textbf{Only identified specifications appear.} Suppressed: "
        f"{esc(', '.join(sup))}. Treatment is measured at ag-census waves and forward-filled, "
        "so it moves only in 2007, 2012, 2017 and 2022. Those outcomes end in 2020 or 2021, "
        "leaving \\textit{no wave transition} inside the window, so county fixed effects "
        "absorb the dairy term entirely. Their coefficients are not interpretable and are "
        "\\textit{not} reported as nulls --- a null would claim power this design does not "
        "have. Exploratory: 89 switchers over seven years."))

# ============================================================ A5
a5 = latest(T4A, "*_A5_event_study.csv")
if a5 is not None:
    outs = list(dict.fromkeys(a5.outcome))
    rows, foot = [], []
    npre = [len(a5[(a5.outcome == o) & (a5.event_time < -1)]) for o in outs]
    nsig = [int((a5[(a5.outcome == o) & (a5.event_time < -1)].p_state < 0.05).sum()) for o in outs]
    rows.append("Pre-period coefficients & " + " & ".join(str(n) for n in npre) + " \\\\")
    rows.append("Significant at 5\\% & " + " & ".join(
        (f"\\textbf{{{n}}}" if n else "0") for n in nsig) + " \\\\")
    rows.append("\\addlinespace")
    rows.append("Verdict & " + " & ".join(
        ("\\textbf{FAILS}" if n else "Flat") for n in nsig) + " \\\\")
    foot.append("Leads/lags & " + " & ".join(["$\\pm8$"] * len(outs)) + " \\\\")
    foot.append("Omitted period & " + " & ".join(["$t=-1$"] * len(outs)) + " \\\\")
    tex.append(table(
        "A5 --- Event study: pre-trend test", "tab:A5", len(outs),
        " & ".join(f"\\rotatebox{{60}}{{{esc(o)}}}" for o in outs), rows, foot,
        "TWFE event study around the first positive change, never-treated pooled into the "
        "omitted category. A \\textbf{FAILS} verdict means counties that later gain a large "
        "dairy were already on a different trajectory, so that outcome cannot be read causally "
        "from this design. Treatment timing carries up to five years of measurement error "
        "(ag census forward-filled), so leads near zero are partly contaminated by "
        "already-treated periods. Coefficient paths are in the companion figure. This "
        "estimator inherits the forbidden-comparison problem A6 exists to correct."))

# ============================================================ A6
cmp_ = latest(T4A, "*_HEADLINE_twfe_vs_cs.csv")
csag = latest(T4A, "*_A6_cs_aggregated.csv")
if cmp_ is not None:
    rows = []
    for _, r in cmp_.iterrows():
        u = np.nan
        if csag is not None and "arm" in csag.columns:
            m = csag[(csag.agg_level == "overall_ATT") & (csag.outcome == r.outcome)
                     & (csag.arm == "unconditional")]
            if len(m):
                u = float(m.att.iloc[0])
        tb, ts = cell(r.TWFE_beta, r.TWFE_se, 0.01 if r.TWFE_sig else 0.5)
        cb, cs_ = cell(r.CS_ATT, r.CS_se, 0.01 if r.CS_sig else 0.5)
        rows.append(f"{esc(r.outcome)} & {tb} & {u:.4f} & {cb} & {esc(r.verdict)} \\\\")
        rows.append(f" & {ts} & & {cs_} & \\\\")
        rows.append("\\addlinespace")
    foot = ["Treatment & \\multicolumn{3}{c}{\\texttt{tr\\_e2\\_add\\_absorb}} & \\\\",
            "Covariates & Step-8 & None & Step-8 & \\\\",
            "Inference & Cluster & \\multicolumn{2}{c}{300-rep bootstrap} & \\\\"]
    tex.append(table(
        "A6 --- Callaway--Sant'Anna versus TWFE, treatment held fixed", "tab:A6", 4,
        "\\textbf{TWFE} & \\textbf{CS uncond.} & \\textbf{CS adjusted} & \\textbf{Verdict}",
        rows, foot,
        "All columns use \\texttt{tr\\_e2\\_add\\_absorb}, the binary absorbing treatment the "
        "Callaway--Sant'Anna cohort definition implies, so the \\textit{only} difference "
        "between (1) and (3) is the estimator. Column (3) applies outcome-regression "
        "adjustment on the step-8 covariates measured at base period $g-1$, fitted on the "
        "clean control group only --- both sides therefore assume parallel trends "
        "\\textit{conditionally}. \\textbf{Under staggered adoption TWFE uses already-treated "
        "counties as controls for later-treated ones, and those comparisons can carry negative "
        "weights}; Callaway--Sant'Anna removes them by construction. A \\textit{TWFE-only} "
        "verdict is the signature of that problem. Six of seven outcomes are significant under "
        "TWFE and none survives."))

BODY = os.path.join(OUT, "_BODY_regressions.tex")
open(BODY, "w").write("\n".join(tex[1:]))
print(f"wrote {BODY}")

tex.append("\\end{document}\n")
p = os.path.join(OUT, "REGRESSION_TABLES.tex")
open(p, "w").write("\n".join(tex))
print(f"wrote {p}")
subprocess.run(["pdflatex","-interaction=nonstopmode","-output-directory",OUT,p],
               capture_output=True, text=True)
pdf = os.path.join(OUT, "REGRESSION_TABLES.pdf")
print(f"compiled -> {pdf}" if os.path.exists(pdf) else "PDF COMPILE FAILED")
for ext in (".aux",".log",".out"):
    f = os.path.join(OUT, "REGRESSION_TABLES"+ext)
    if os.path.exists(f): os.remove(f)
