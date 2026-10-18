#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4e-latex-tables.py  --  LaTeX table generation for Part A.

Reads the workbook written by script4d-power-balance.py and emits booktabs tables
as standalone .tex fragments, ready to \\input into a paper or beamer deck.

Outputs -> Dropbox/Mental/Data/output/tables/script4a/latex/
    power_<TREATMENT_ID>.tex   one table per treatment definition (14 files),
                               rows = the 6 outcomes
    all_power_tables.tex       master file that \\input's all 14 in order
    outcome_coverage.tex       outcome year-span and completeness table

Each fragment is a complete float (\\begin{table} ... \\end{table}) with its own
caption and \\label, so they can be included individually or all at once.

Preamble required in the calling document:
    \\usepackage{booktabs}
    \\usepackage{siunitx}     % optional, only if you switch to S columns
Nothing here depends on siunitx as written -- numbers are pre-formatted strings,
which keeps the fragments portable into beamer where siunitx is often absent.

ESCAPING: control and outcome names in this project contain `_`, `%` and `-`
(e.g. `%_65_and_older`, `children_in_single-parent_households_per100k`). Every
string that reaches the table body goes through `esc()`.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from script4_treatment import TREATMENTS, OUTCOMES, tables_dir, os, date

IN_DIR  = os.path.join(tables_dir, "script4a")
OUT_DIR = os.path.join(IN_DIR, "latex")
os.makedirs(OUT_DIR, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
XLSX  = os.path.join(IN_DIR, f"{TODAY}_power_and_coverage.xlsx")


def esc(x):
    """Escape LaTeX specials. Order matters: backslash first."""
    s = str(x)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def num(x, dp=3, comma=False):
    """Format a number, or an en-dash for missing."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if comma:
        return f"{int(round(x)):,}".replace(",", r"\,")
    return f"{x:.{dp}f}"


# =============================================================================
if not os.path.exists(XLSX):
    raise SystemExit(f"workbook not found: {XLSX}\nRun script4d-power-balance.py first.")

power = pd.read_excel(XLSX, sheet_name="power_by_analysis")
outc  = pd.read_excel(XLSX, sheet_name="outcome_coverage")
print(f"read {len(power)} power rows, {len(outc)} outcome rows")

OUTCOME_ORDER = list(OUTCOMES.keys())


# =============================================================================
# One power table per treatment definition
# =============================================================================
POWER_HEADER = (
    r"\begin{tabular}{l rrr rr rr rr}" "\n"
    r"\toprule" "\n"
    r" & & \multicolumn{2}{c}{Observations} & \multicolumn{2}{c}{Counties}"
    r" & \multicolumn{2}{c}{Variation} & \multicolumn{2}{c}{Detectable effect} \\" "\n"
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}" "\n"
    r"Outcome & $N$ & Treated & Control & Total & Treated"
    r" & Switch. & Trans. & SE & MDE$_{\sigma_w}$ \\" "\n"
    r"\midrule" "\n"
)

written = []
for tid, (tcol, cond, tlabel) in TREATMENTS.items():
    sub = power[power.treatment_id == tid].copy()
    if sub.empty:
        continue
    sub["_o"] = sub.outcome.apply(lambda o: OUTCOME_ORDER.index(o) if o in OUTCOME_ORDER else 99)
    sub = sub.sort_values("_o")
    ttype = sub.iloc[0]["treatment_type"]
    condtxt = sub.iloc[0]["conditioning"]
    # empty conditioning reads back from Excel as NaN, not ""
    condtxt = "" if (pd.isna(condtxt) or str(condtxt).strip() in ("", "nan")) else str(condtxt)

    body = []
    for _, r in sub.iterrows():
        body.append(
            f"{esc(r.outcome)} & {num(r.N_obs, comma=True)} & "
            f"{num(r.n_treated_obs, comma=True)} & {num(r.n_control_obs, comma=True)} & "
            f"{num(r.n_counties, comma=True)} & {num(r.n_treated_counties, comma=True)} & "
            f"{num(r.n_switcher_counties, comma=True)} & {num(r.n_transitions, comma=True)} & "
            f"{num(r.se_state_realised, 4)} & {num(r.MDE_in_within_sd, 3)} \\\\"
        )

    # The treated/control split is only exact for binary treatments -- say so on
    # the face of the table rather than in a separate methods note.
    split_note = (
        "Treated/control counts are exact."
        if ttype == "binary" else
        rf"Treatment is \emph{{{esc(ttype)}}}; the treated/control split is at "
        r"$>0$ and is descriptive only --- the estimator uses the full continuous variation."
    )
    cond_note = (rf" Conditioning variables: \texttt{{{esc(condtxt)}}}." if condtxt else "")

    tex = (
        r"% ---------------------------------------------------------------" "\n"
        rf"% Power / sample table for treatment {tid} -- generated by script4e-latex-tables.py" "\n"
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\footnotesize" "\n"
        rf"\caption{{Sample and detectable effect: {esc(tid)} --- {esc(tlabel)}}}" "\n"
        rf"\label{{tab:power-{tid.lower()}}}" "\n"
        + POWER_HEADER
        + "\n".join(body) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\begin{minipage}{\linewidth}\vspace{2pt}\scriptsize" "\n"
        rf"\textit{{Notes.}} Treatment column \texttt{{{esc(tcol)}}}. Rural counties, "
        r"county $+$ year fixed effects, CORE\_9 controls. SE is cluster-robust at the "
        r"state level. MDE$_{\sigma_w}$ is the minimum detectable effect at 80\% power "
        r"(two-sided $\alpha=0.05$), $2.8016\times\mathrm{SE}$, expressed in within-county "
        r"standard deviations of the outcome. \emph{Switch.} is the number of counties "
        r"whose treatment changes --- only these identify the coefficient under county "
        r"fixed effects. " + split_note + cond_note + "\n"
        r"\end{minipage}" "\n"
        r"\end{table}" "\n"
    )
    path = os.path.join(OUT_DIR, f"power_{tid}.tex")
    with open(path, "w") as f:
        f.write(tex)
    written.append((tid, tlabel, os.path.basename(path)))
    print(f"  wrote {os.path.basename(path)}")

master = (
    "% Master file -- all per-treatment power tables, in registry group order.\n"
    "% Requires: \\usepackage{booktabs}\n"
    "% Usage: \\input{.../all_power_tables.tex}\n\n"
    + "\n".join(rf"\input{{power_{tid}}}  % {lab}" for tid, lab, _ in written) + "\n"
)
with open(os.path.join(OUT_DIR, "all_power_tables.tex"), "w") as f:
    f.write(master)


# =============================================================================
# Outcome coverage table
# =============================================================================
outc["_o"] = outc.outcome.apply(lambda o: OUTCOME_ORDER.index(o) if o in OUTCOME_ORDER else 99)
outc = outc.sort_values("_o")

rows = []
for _, r in outc.iterrows():
    rows.append(
        f"{esc(r.outcome)} & \\texttt{{{esc(r.column)}}} & "
        f"{int(r.first_year)}--{int(r.last_year)} & "
        f"{num(r.n_nonnull, comma=True)} & {num(100*r.pct_complete_full_panel, 1)}\\% & "
        f"{num(r.n_counties, comma=True)} & {num(r['mean'], 2)} & {num(r['sd'], 2)} \\\\"
    )

tex = (
    r"% Outcome coverage -- generated by script4e-latex-tables.py" "\n"
    r"\begin{table}[htbp]" "\n"
    r"\centering" "\n"
    r"\footnotesize" "\n"
    r"\caption{Outcome variables: source column, coverage, and moments}" "\n"
    r"\label{tab:outcome-coverage}" "\n"
    r"\begin{tabular}{l l c r r r rr}" "\n"
    r"\toprule" "\n"
    r"Outcome & Panel column & Years & $N$ & Complete & Counties & Mean & SD \\" "\n"
    r"\midrule" "\n"
    + "\n".join(rows) + "\n"
    r"\bottomrule" "\n"
    r"\end{tabular}" "\n"
    r"\begin{minipage}{\linewidth}\vspace{2pt}\scriptsize" "\n"
    r"\textit{Notes.} Rural counties only (NCHS codes 3--6), 2000--2023 panel. "
    r"\emph{Complete} is the share of all rural county-year cells with a non-missing "
    r"value. Coverage differs sharply across outcomes and is the binding constraint on "
    r"which treatment cohorts are observable: County Health Rankings outcomes begin in "
    r"2010, so the 2007 entry cohort has no observable pre-period. Deaths of despair "
    r"are truncated after 2020, where source coverage falls to zero." "\n"
    r"\end{minipage}" "\n"
    r"\end{table}" "\n"
)
with open(os.path.join(OUT_DIR, "outcome_coverage.tex"), "w") as f:
    f.write(tex)

print(f"\n  wrote outcome_coverage.tex")
print(f"  wrote all_power_tables.tex ({len(written)} inputs)")
print(f"\nAll LaTeX -> {OUT_DIR}")


# =============================================================================
# VIF table -- headline specification only
# =============================================================================
# Presentation rule: VIF belongs to a SPECIFIC design matrix, so one table for the
# headline spec, and a one-sentence summary of the other 185 in the notes.
vif = pd.read_excel(XLSX, sheet_name="vif_headline_spec")

rows = []
for _, r in vif.iterrows():
    rows.append(
        f"\\texttt{{{esc(r.variable)}}} & {esc(r.role)} & "
        f"{num(r.vif_raw_levels, 2)} & {num(r.vif_within_county_year, 2)} \\\\"
    )

tex = (
    r"% VIF, headline spec -- generated by script4e-latex-tables.py" "\n"
    r"\begin{table}[htbp]" "\n"
    r"\centering" "\n"
    r"\footnotesize" "\n"
    r"\caption{Variance inflation factors, headline specification}" "\n"
    r"\label{tab:vif}" "\n"
    r"\begin{tabular}{l l rr}" "\n"
    r"\toprule" "\n"
    r" & & \multicolumn{2}{c}{VIF} \\" "\n"
    r"\cmidrule(lr){3-4}" "\n"
    r"Variable & Role & Raw levels & Within \\" "\n"
    r"\midrule" "\n"
    + "\n".join(rows) + "\n"
    r"\bottomrule" "\n"
    r"\end{tabular}" "\n"
    r"\begin{minipage}{\linewidth}\vspace{2pt}\scriptsize" "\n"
    r"\textit{Notes.} Specification: treatment E2 (any positive change in large-dairy "
    r"operations, absorbing) on poor mental health days, rural counties, CORE\_9 "
    r"controls. \emph{Raw levels} computes VIF on the untransformed design matrix; "
    r"\emph{Within} computes it after two-way (county and year) demeaning, which is "
    r"the variation the fixed-effects estimator actually uses and therefore the "
    r"relevant diagnostic. The contrast is the substantive point: collinearity among "
    r"county health and demographic indicators is almost entirely cross-sectional and "
    r"is absorbed by county fixed effects (children in poverty falls from 3.66 to "
    r"1.12). VIF is a property of a particular design matrix, so this table reports "
    r"the headline specification only; across all 186 estimated specifications the "
    r"maximum VIF on any right-hand-side variable was 4.56, and no specification "
    r"exceeded the conventional threshold of 5." "\n"
    r"\end{minipage}" "\n"
    r"\end{table}" "\n"
)
with open(os.path.join(OUT_DIR, "vif_headline.tex"), "w") as f:
    f.write(tex)
print("  wrote vif_headline.tex")


# =============================================================================
# Control selection funnel
# =============================================================================
fun = pd.read_excel(XLSX, sheet_name="control_funnel")
never = pd.read_excel(XLSX, sheet_name="controls_never_considered")

frows = []
for _, r in fun.iterrows():
    stage = str(r.stage)
    indent = stage.startswith("  ")
    label = esc(stage.strip())
    if indent:
        label = r"\quad " + label
    if stage.startswith(("=", "USED")):
        label = r"\textbf{" + label + "}"
    frows.append(f"{label} & {num(r['n'], comma=True)} \\\\")

tex = (
    r"% Control selection funnel -- generated by script4e-latex-tables.py" "\n"
    r"\begin{table}[htbp]" "\n"
    r"\centering" "\n"
    r"\footnotesize" "\n"
    r"\caption{From panel columns to the estimating control set}" "\n"
    r"\label{tab:control-funnel}" "\n"
    r"\begin{tabular}{l r}" "\n"
    r"\toprule" "\n"
    r"Stage & Columns \\" "\n"
    r"\midrule" "\n"
    + "\n".join(frows) + "\n"
    r"\bottomrule" "\n"
    r"\end{tabular}" "\n"
    r"\begin{minipage}{\linewidth}\vspace{2pt}\scriptsize" "\n"
    r"\textit{Notes.} The panel carries 162 columns, but most are not candidate "
    r"controls: identifiers, County Health Rankings companion columns (numerator, "
    r"denominator, confidence bounds accompanying each published measure), the "
    r"treatment variables themselves, the outcomes, and population denominators. "
    r"Of the 107 genuine candidates, 105 are numeric and 26 are at least 92\% "
    r"complete on rows with a non-missing outcome. "
    rf"Of those 26, {len(never)} were never considered for the legacy 25-variable "
    r"control set, which was inherited from \texttt{script3-ridge.py} rather than "
    r"derived from this pool: "
    + ", ".join(rf"\texttt{{{esc(c)}}}" for c in never["column"].tolist()) + ". "
    r"This is a known open item, not a stated exclusion criterion." "\n"
    r"\end{minipage}" "\n"
    r"\end{table}" "\n"
)
with open(os.path.join(OUT_DIR, "control_funnel.tex"), "w") as f:
    f.write(tex)
print("  wrote control_funnel.tex")
