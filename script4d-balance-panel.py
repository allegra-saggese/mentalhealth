#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4d-balance-panel.py  --  rebuilds BALANCEpanel.pdf on the corrected panel.

REPLACES the 2026-09-18 version of script4d-power-balance.py, which produced
BALANCEpanel.pdf against the pre-CHR-fix panel, the legacy 12-treatment grid,
6 outcomes and the retired CORE_9 control set.

WHAT CHANGED
------------
    panel       2026-09-23 (10 controls re-sourced to correct data years)
    treatments  T1-T4 + E2   (was 12 legacy definitions; T5 dropped, T6 retired)
    outcomes    7            (was 6; four crime outcomes settled under D-11)
    controls    step-8 pre-committed set, D-16/D-29  (was CORE_9)

WHAT EACH COLUMN MEANS
----------------------
    N                 estimating-sample rows after listwise deletion
    Treated/Control   county-year observations with treatment >0 / ==0. For the
                      continuous treatments (T2, T3, T4) this split is
                      DESCRIPTIVE ONLY -- the estimator uses the full continuous
                      variation, not a dichotomy.
    Counties          total, and those ever treated
    Switch./Trans.    counties whose treatment CHANGES, and the number of changes.
                      Under county FE only switchers identify beta. This is the
                      honest denominator on every result in the deck.
    SE                cluster-robust at the state level, from the actual A2
                      regression (county + year FE, step-8 controls)
    MDE               minimum detectable effect at 80% power, two-sided a=0.05:
                      2.8016 x SE, expressed in WITHIN-COUNTY standard deviations
                      of the outcome. Within-county, not raw, because that is the
                      variation a county-FE model can actually use.

Output -> Data/output/latex-outputs-2809/
              BALANCEpanel.tex / .pdf        (one table per treatment + outcomes)
              <date>_balance_panel.csv       (same numbers, machine readable)
"""
import warnings; warnings.filterwarnings("ignore")
import subprocess
import numpy as np, pandas as pd, pyfixest as pf

from script4_treatment import (
    load_panel, OUTCOMES, CORE_TREATMENTS, CORE_TREATMENT_LABELS,
    SPEC_EXTRA_REGRESSORS, COVARIATE_ORDER, db_data, os, date,
)

OUT = os.path.join(db_data, "output", "latex-outputs-2809")
os.makedirs(OUT, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
CONTROLS = COVARIATE_ORDER[:8]
Z80 = 2.8016                      # 80% power, two-sided alpha = 0.05

# E2 is not a core treatment but carries A5/A6, so it belongs in the balance panel.
GRID = {tid: (col, SPEC_EXTRA_REGRESSORS.get(tid, []), CORE_TREATMENT_LABELS[tid])
        for tid, col in CORE_TREATMENTS.items()}
GRID["E2"] = ("tr_e2_add_absorb", [],
              "First positive change in large dairy ops (absorbing)")
BINARY = {"T1", "E2"}

def _clean(n):
    return (str(n).replace("%","pct").replace("-","_").replace("/","_").replace(" ","_")
                  .replace("(","").replace(")","").replace("+","p").replace(",","_"))

def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

df = load_panel()
print("="*78); print("script4d -- balance / power panel"); print("="*78)
print(f"panel {len(df):,} rows | {len(GRID)} treatments x {len(OUTCOMES)} outcomes")
print(f"controls: step-8 ({len(CONTROLS)})\n")

rows = []
for tid, (tcol, extra, tlab) in GRID.items():
    for okey, ocol in OUTCOMES.items():
        need = ["fips","year","state_fips",ocol,tcol] + list(extra) + CONTROLS
        sub = df[list(dict.fromkeys(need))].dropna()
        if len(sub) < 200:
            continue
        nun = sub.groupby("fips")[tcol].nunique()
        sw  = nun[nun > 1].index
        if len(sw):
            e = sub[sub.fips.isin(sw)].sort_values(["fips","year"]).copy()
            e["_d"] = e.groupby("fips")[tcol].diff()
            ntrans = int((e["_d"].abs() > 0).sum())
        else:
            ntrans = 0
        # within-county SD of the outcome -- the scale MDE is expressed in
        sd_w = float((sub[ocol] - sub.groupby("fips")[ocol].transform("mean")).std())

        ren = {c: _clean(c) for c in sub.columns}
        s2  = sub.rename(columns=ren)
        fml = (f"{ren[ocol]} ~ " + " + ".join([ren[tcol]] + [ren[x] for x in extra]
               + [ren[c] for c in CONTROLS]) + " | fips + year")
        try:
            m = pf.feols(fml, data=s2, vcov={"CRV1":"state_fips"})
            t = m.tidy().loc[ren[tcol]]
            se, beta, pv = float(t["Std. Error"]), float(t["Estimate"]), float(t["Pr(>|t|)"])
        except Exception as ex:
            print(f"  {tid}|{okey} failed: {type(ex).__name__}"); continue

        rows.append({
            "treatment_id": tid, "treatment": tlab, "treatment_col": tcol,
            "outcome": okey, "N": len(sub),
            "obs_treated": int((sub[tcol] > 0).sum()),
            "obs_control": int((sub[tcol] == 0).sum()),
            "n_counties": sub.fips.nunique(),
            "n_counties_treated": int(sub.loc[sub[tcol] > 0, "fips"].nunique()),
            "n_switchers": int(len(sw)), "n_transitions": ntrans,
            "beta": beta, "se_state": se, "p_state": pv,
            "sd_within": sd_w,
            "MDE_sd_within": Z80 * se / sd_w if sd_w else np.nan,
            "binary_split_exact": tid in BINARY,
        })
    print(f"  {tid} done")

R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, f"{TODAY}_balance_panel.csv"), index=False)

# --------------------------------------------------------------------- LaTeX
def table(tid, g):
    lab, col = g.treatment.iloc[0], g.treatment_col.iloc[0]
    exact = bool(g.binary_split_exact.iloc[0])
    body = []
    for _, r in g.iterrows():
        body.append(
            f"{esc(r.outcome)} & {r.N:,} & {r.obs_treated:,} & {r.obs_control:,} & "
            f"{r.n_counties:,} & {r.n_counties_treated:,} & {r.n_switchers:,} & "
            f"{r.n_transitions:,} & {r.se_state:.4f} & {r.MDE_sd_within:.3f} \\\\")
    note = (f"Treatment column \\texttt{{{esc(col)}}}. Rural counties, county $+$ year fixed "
            f"effects, step-8 pre-committed controls. SE is cluster-robust at the state level. "
            f"MDE is the minimum detectable effect at 80\\% power (two-sided $\\alpha=0.05$), "
            f"$2.8016\\times$SE, expressed in {{\\itshape within-county}} standard deviations of "
            f"the outcome. {{\\itshape Switch.}} is the number of counties whose treatment "
            f"changes --- only these identify the coefficient under county fixed effects.")
    note += (" Treated/control counts are exact." if exact else
             " Treatment is continuous; the treated/control split is at $>0$ and is "
             "{\\itshape descriptive only} --- the estimator uses the full continuous variation.")
    if g.treatment_id.iloc[0] in ("T2","T4"):
        note += " Conditioning variable: \\texttt{log\\_pop}."
    return ("\\begin{table}[htbp]\n\\centering\n"
            f"\\caption{{Sample and detectable effect: {tid} --- {esc(lab)}}}\n"
            f"\\label{{tab:bal_{tid}}}\n"
            "\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
            " & \\multicolumn{3}{c}{Observations} & \\multicolumn{2}{c}{Counties}"
            " & \\multicolumn{2}{c}{Variation} & \\multicolumn{2}{c}{Detectable effect} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
            "Outcome & $N$ & Treated & Control & Total & Treated & Switch. & Trans. & SE"
            " & MDE$_{\\sigma_w}$ \\\\\n\\midrule\n"
            + "\n".join(body) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\n"
            "\\vspace{0.3em}\n\\begin{minipage}{\\textwidth}\\footnotesize\n"
            f"\\textit{{Notes.}} {note}\n" "\\end{minipage}\n\\end{table}\n")

tex = [r"""\documentclass[11pt]{article}
\usepackage[margin=1in,landscape]{geometry}
\usepackage{booktabs,adjustbox,amsmath}
\usepackage[colorlinks=true,linkcolor=blue]{hyperref}
\setlength{\parindent}{0pt}
\begin{document}
\begin{center}
{\Large\bfseries Balance and detectable effect panel}\\[0.3em]
{\large Large dairy CAFOs and rural mental health}\\[0.6em]
""" + f"Panel 2026-09-23 \\quad$\\cdot$\\quad {len(GRID)} treatments $\\times$ {len(OUTCOMES)} outcomes"
   + r"""\quad$\cdot$\quad step-8 pre-committed controls\\[0.2em]
\today
\end{center}
\vspace{1em}
"""]
for tid in GRID:
    g = R[R.treatment_id == tid]
    if len(g):
        tex.append(table(tid, g))

# outcome reference table
orow = []
for okey, ocol in OUTCOMES.items():
    d = df[["fips","year",ocol]].dropna()
    orow.append(f"{esc(okey)} & \\texttt{{{esc(ocol)}}} & {int(d.year.min())}--{int(d.year.max())}"
                f" & {len(d):,} & {100*len(d)/len(df):.1f}\\% & {d.fips.nunique():,}"
                f" & {d[ocol].mean():.2f} & {d[ocol].std():.2f} \\\\")
tex.append("\\begin{table}[htbp]\n\\centering\n"
           "\\caption{Outcome variables: source column, coverage, and moments}\n"
           "\\begin{adjustbox}{max width=\\textwidth}\n"
           "\\begin{tabular}{llrrrrrr}\n\\toprule\n"
           "Outcome & Panel column & Years & $N$ & Complete & Counties & Mean & SD \\\\\n"
           "\\midrule\n" + "\n".join(orow) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\end{adjustbox}\n\n\\vspace{0.3em}\n\\begin{minipage}{\\textwidth}\\footnotesize\n"
           "\\textit{Notes.} Rural counties only (NCHS codes 3--6), 2000--2023 panel. "
           "\\textit{Complete} is the share of all rural county-year cells with a non-missing "
           "value. Coverage is the binding constraint on which treatment cohorts are "
           "observable: County Health Rankings outcomes begin in 2010, so the 2007 entry "
           "cohort has no observable pre-period and is dropped by Callaway--Sant'Anna. "
           "Deaths of despair are truncated after 2020.\n"
           "\\end{minipage}\n\\end{table}\n")
tex.append("\\end{document}\n")

# Body-only fragment (no preamble, no \begin{document}) so the master appendix
# in script4p can \input it. Single source of truth: the table code lives here,
# not duplicated in the master.
BODY = os.path.join(OUT, "_BODY_balance.tex")
open(BODY, "w").write("\n".join(tex[1:-1]))
print(f"wrote {BODY}")

p = os.path.join(OUT, "BALANCEpanel.tex")
open(p, "w").write("\n".join(tex))
print(f"\nwrote {p}")
r = subprocess.run(["pdflatex","-interaction=nonstopmode","-output-directory",OUT,p],
                   capture_output=True, text=True)
pdf = os.path.join(OUT, "BALANCEpanel.pdf")
print(f"compiled -> {pdf}" if os.path.exists(pdf) else "PDF COMPILE FAILED")
for ext in (".aux",".log",".out"):
    f = os.path.join(OUT, "BALANCEpanel"+ext)
    if os.path.exists(f): os.remove(f)
print(f"\n{len(R)} treatment x outcome cells")
