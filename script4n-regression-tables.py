#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4n-regression-tables.py  --  regression tables in standard paper format.

REWRITTEN 2026-10-12 (run 2). Reads the latest CSVs from script4a (A1, A2, A4,
A5, A6, A7, VIF, Conley) and script4r (slaughterhouse R0-R4). NO estimation.

FORMAT (applies to every table)
-------------------------------
    columns  = outcomes (1)-(7), grouped Mental health | Mortality | Arrests
    panels   = treatments (Panel A, B, ...), one coefficient row each,
               standard error in parentheses directly below
    bottom   = Observations, Counties, State clusters, Controls, FE rows
    stars    = *** p<0.01, ** p<0.05, * p<0.10   (state-clustered p-values)
Notes go in the document, not inside the tabular, so each fragment can be
\\input into a paper, a slide or the appendix unchanged.

A3 was dropped 2026-10-12 and is no longer read.

REPORTED vs ROBUSTNESS (2026-10-15, team feedback). script4_treatment.REPORT_*
sets what appears in the MAIN exhibits: treatments T1, T3, T4, E2; county + year
FE (A2); state-clustered SEs; Callaway-Sant'Anna covariate-adjusted. Nothing is
dropped: the other specifications (T2, A1 state FE, CS-unconditional, LP-DiD,
Conley, the all-treatment grid) are still written, as separate tables collected
in _BODY_robustness.tex and the second half of REGRESSION_TABLES.tex.

Output -> LATEX_OUT/tables/*.tex, LATEX_OUT/REGRESSION_TABLES.tex (+ .pdf if
pdflatex is available), LATEX_OUT/_BODY_regressions.tex for script4p.
"""
import warnings; warnings.filterwarnings("ignore")
import glob, math, subprocess
import numpy as np, pandas as pd
from script4_treatment import db_data, os, LATEX_OUT, REPORT_TREATMENTS, REPORT_FE, REPORT_CS_ARM

TAB = os.path.join(LATEX_OUT, "tables"); os.makedirs(TAB, exist_ok=True)
T4A = os.path.join(db_data, "output", "tables", "script4a")
T4R = os.path.join(db_data, "output", "tables", "script4r")


def latest(d, pat):
    f = sorted(glob.glob(os.path.join(d, pat)))
    return pd.read_csv(f[-1]) if f else None


OUTS = ["Poor MH Days", "Frequent Mental Distress", "Deaths of Despair", "Aggravated Assault",
        "Assault, all severities", "Violence index (partial)", "NIBRS curated total"]
DEC = {"Poor MH Days": 3, "Frequent Mental Distress": 1, "Deaths of Despair": 3,
       "Aggravated Assault": 2, "Assault, all severities": 2,
       "Violence index (partial)": 2, "NIBRS curated total": 2}
HEAD7 = r""" & \multicolumn{2}{c}{Mental health} & Mortality & \multicolumn{4}{c}{Arrests per 100{,}000} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-8}
 & Poor MH & Freq.\ mental & Deaths of & Agg. & Assault, & Violence & NIBRS \\
 & days & distress & despair & assault & all sev. & (partial) & total \\
 & (1) & (2) & (3) & (4) & (5) & (6) & (7) \\
\midrule"""
# Panel letters are assigned when a table is built, so a table restricted to the
# reported treatments is lettered A, B, C, D rather than skipping T2's letter.
PANELS = [("T1", "Presence of a large dairy CAFO (T1)", "Large dairy CAFO (0/1)"),
          ("T2", "Count of large dairy CAFOs (T2)", "Number of large dairy CAFOs"),
          ("T3", "Large dairy CAFOs per 10,000 residents (T3)", "Large dairy CAFOs per 10,000"),
          ("T4", r"$\log(1+\text{count})$ (T4)", r"$\log(1+$large dairy CAFOs$)$"),
          ("E2", "First gain of a large dairy CAFO, absorbing (E2)", "Post first gain (0/1)")]
STARNOTE = r"$^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$"


def st(p):
    return "" if pd.isna(p) else "^{***}" if p < .01 else "^{**}" if p < .05 else "^{*}" if p < .10 else ""


def coef(b, p, d):
    if pd.isna(b):
        return ""
    sign = "$-$" if b < 0 else ""
    return f"{sign}{abs(b):,.{d}f}" + (f"$\\mkern-2mu{st(p)}$" if st(p) else "")


def se(x, d):
    return "" if pd.isna(x) else f"({x:,.{d}f})"


def pnorm(b, s):
    return math.erfc(abs(b / s) / math.sqrt(2)) if (pd.notna(s) and s > 0) else np.nan


def save(name, body):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(body)
    return name


written = []

# ------------------------------------------------------------- A1 / A2 -------
grid = latest(T4A, "*_A1_A2_treatment_grid.csv")


def twfe(reg, fe_county, panels=None):
    d = grid[grid.registry == reg]
    PN = PANELS if panels is None else [x for x in PANELS if x[0] in panels]
    PN = [(t, f"Panel {chr(65 + i)}. {title}", lab) for i, (t, title, lab) in enumerate(PN)]
    L = [r"\begin{tabular}{l*{7}{c}}", r"\toprule", HEAD7]
    for i, (t, title, lab) in enumerate(PN):
        r = d[d.treatment_id == t].set_index("outcome")
        L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}} \\\\")
        L.append(lab + " & " + " & ".join(coef(r.loc[o, "beta"], r.loc[o, "p_state"], DEC[o]) if o in r.index else "" for o in OUTS) + r" \\")
        L.append(" & " + " & ".join(se(r.loc[o, "se_state"], DEC[o]) if o in r.index else "" for o in OUTS) + r" \\")
        L.append("Switching counties & " + " & ".join(f"{int(r.loc[o,'n_switchers']):,}" if o in r.index else "" for o in OUTS) + r" \\")
        if i < len(PN) - 1:
            L.append(r"\addlinespace")
    r = d[d.treatment_id == "T1"].set_index("outcome")
    yes, no = " & ".join(["Yes"] * 7), " & ".join(["No"] * 7)
    L += [r"\midrule",
          "Observations & " + " & ".join(f"{int(r.loc[o,'N']):,}" for o in OUTS) + r" \\",
          "Counties & " + " & ".join(f"{int(r.loc[o,'n_counties']):,}" for o in OUTS) + r" \\",
          "State clusters & " + " & ".join(f"{int(r.loc[o,'n_states'])}" for o in OUTS) + r" \\",
          "Controls & " + yes + r" \\",
          "County FE & " + (yes if fe_county else no) + r" \\",
          "State FE & " + (no if fe_county else yes) + r" \\",
          "Year FE & " + yes + r" \\", r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


if grid is not None:
    # REPORTED: A2, reported treatments only.
    rep_t = [t for t in ["T1", "T2", "T3", "T4", "E2"] if t in REPORT_TREATMENTS]
    half = (len(rep_t) + 1) // 2
    written.append(save("tab_A2.tex", twfe("A2", True, rep_t)))
    for k, pn in [("p1", rep_t[:half]), ("p2", rep_t[half:])]:
        written.append(save(f"tab_A2_{k}.tex", twfe("A2", True, pn)))
    # ROBUSTNESS: A2 with every treatment (incl. T2), and A1 pooled state-FE.
    written.append(save("tab_A2_all_treatments.tex", twfe("A2", True)))
    written.append(save("tab_A1.tex", twfe("A1", False)))
    for k, pn in [("p1", ["T1", "T2", "T3"]), ("p2", ["T4", "E2"])]:
        written.append(save(f"tab_A1_{k}.tex", twfe("A1", False, pn)))

# ------------------------------------------------------------- A4 ------------
a4 = latest(T4A, "*_A4_fsis_interaction.csv")
TCOL = {"T1": "tr_e1_lg_bin", "T2": "tr_i3_lg_count", "T3": "tr_i4_lg_p10k", "T4": "tr_c3_log_lg"}
def a4_table(tids):
    cols = [(o, t) for o in ["Poor MH Days", "Frequent Mental Distress"] for t in tids]
    n = len(tids)
    L = [r"\begin{tabular}{l*{%d}{c}}" % (2 * n), r"\toprule",
         r" & \multicolumn{%d}{c}{Poor MH days} & \multicolumn{%d}{c}{Freq.\ mental distress} \\" % (n, n),
         r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (n + 1, n + 2, 2 * n + 1),
         r"Dairy measure $D$: & " + " & ".join(tids * 2) + r" \\",
         r" & " + " & ".join(f"({i})" for i in range(1, 2 * n + 1)) + r" \\", r"\midrule"]
    R = {k: [] for k in ["b1", "s1", "b2", "s2", "b3", "s3"]}
    for o, t in cols:
        r = a4[(a4.outcome == o) & (a4.treatment_id == t)].iloc[0]
        d = 3 if o == "Poor MH Days" else 1
        for k, pre in [("1", f"cond_{TCOL[t]}"), ("2", "cond_any_fsis")]:
            R["b" + k].append(coef(r.get(pre + "_beta"), r.get(pre + "_p"), d))
            R["s" + k].append(se(r.get(pre + "_se", np.nan), d))
        R["b3"].append(coef(r.beta, r.p_state, d)); R["s3"].append(se(r.se_state, d))
    r0 = a4[a4.outcome == "Poor MH Days"].iloc[0]; r1 = a4[a4.outcome == "Frequent Mental Distress"].iloc[0]
    yes = " & ".join(["Yes"] * (2 * n))
    L += [r"$D$ ($\beta_1$) & " + " & ".join(R["b1"]) + r" \\", " & " + " & ".join(R["s1"]) + r" \\",
          r"FSIS plant, $F$ ($\beta_2$) & " + " & ".join(R["b2"]) + r" \\", " & " + " & ".join(R["s2"]) + r" \\",
          r"$D \times F$ ($\beta_3$) & " + " & ".join(R["b3"]) + r" \\", " & " + " & ".join(R["s3"]) + r" \\",
          r"\midrule",
          "Observations & " + " & ".join([f"{int(r0.N):,}"] * n + [f"{int(r1.N):,}"] * n) + r" \\",
          "Counties & " + " & ".join([f"{int(r0.n_counties):,}"] * n + [f"{int(r1.n_counties):,}"] * n) + r" \\",
          "Switching counties & " + " & ".join(f"{int(a4[(a4.outcome==o)&(a4.treatment_id==t)].n_switchers.iloc[0]):,}" for o, t in cols) + r" \\",
          "Controls & " + yes + r" \\", "County FE & " + yes + r" \\", "Year FE & " + yes + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


if a4 is not None:
    written.append(save("tab_A4.tex", a4_table([t for t in TCOL if t in REPORT_TREATMENTS])))   # reported
    written.append(save("tab_A4_all_treatments.tex", a4_table(list(TCOL))))                     # robustness

# ------------------------------------------------------------- A5 pre-trends --
a5 = latest(T4A, "*_A5_event_study.csv")
if a5 is not None:
    reg_lab = {"A5": "Baseline", "A5_popbase": r"+ log pop.\ (2000) $\times$ year"}
    L = [r"\begin{tabular}{llccc}", r"\toprule",
         r"Outcome & Specification & Pre-period coefs & Significant at 5\% & Min $p$ \\", r"\midrule"]
    for o in OUTS:
        for reg in ["A5", "A5_popbase"]:
            pre = a5[(a5.registry == reg) & (a5.outcome == o) & (a5.event_time < -1)]
            if pre.empty:
                continue
            L.append(f"{o if reg == 'A5' else ''} & {reg_lab[reg]} & {len(pre)} & "
                     f"{int((pre.p_state < .05).sum())} & {pre.p_state.min():.3f} \\\\")
    pre = a5[(a5.registry == "A5_logpop") & (a5.event_time < -1)]
    if len(pre):
        L += [r"\midrule", f"Log population (placebo) & No controls & {len(pre)} & "
              f"{int((pre.p_state < .05).sum())} & {pre.p_state.min():.3f} \\\\"]
    L += [r"\bottomrule", r"\end{tabular}"]
    written.append(save("tab_A5_pretrend.tex", "\n".join(L)))

# ------------------------------------------------------------- A6 ------------
a6 = latest(T4A, "*_A6_cs_aggregated.csv")
def a6_table(arms):
    ov = a6[a6.agg_level == "overall_ATT"].copy()
    ov["p"] = [pnorm(b, s) for b, s in zip(ov.att, ov.se_state)]
    d2 = grid[(grid.registry == "A2") & (grid.treatment_id == "E2")].set_index("outcome")
    L = [r"\begin{tabular}{l*{7}{c}}", r"\toprule", HEAD7,
         "TWFE (county $+$ year FE) & " + " & ".join(coef(d2.loc[o, "beta"], d2.loc[o, "p_state"], DEC[o]) for o in OUTS) + r" \\",
         " & " + " & ".join(se(d2.loc[o, "se_state"], DEC[o]) for o in OUTS) + r" \\ \addlinespace"]
    for arm, lab in [("unconditional", "Callaway--Sant'Anna, unconditional"),
                     ("covariate-adjusted", "Callaway--Sant'Anna, covariate-adjusted")]:
        if arm not in arms:
            continue
        x = ov[ov.arm == arm].set_index("outcome")
        L.append(lab + " & " + " & ".join(coef(x.loc[o, "att"], x.loc[o, "p"], DEC[o]) if o in x.index else "" for o in OUTS) + r" \\")
        L.append(" & " + " & ".join(se(x.loc[o, "se_state"], DEC[o]) if o in x.index else "" for o in OUTS) + r" \\ \addlinespace")
    x = ov[ov.arm == REPORT_CS_ARM].set_index("outcome")
    L += [r"\midrule",
          "Observations (TWFE) & " + " & ".join(f"{int(d2.loc[o,'N']):,}" for o in OUTS) + r" \\",
          "Switching counties (TWFE) & " + " & ".join(f"{int(d2.loc[o,'n_switchers']):,}" for o in OUTS) + r" \\",
          "Cohorts used (CS) & " + " & ".join(f"{int(x.loc[o,'n_cohorts'])}" if o in x.index else "" for o in OUTS) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


if a6 is not None and grid is not None:
    written.append(save("tab_A6.tex", a6_table([REPORT_CS_ARM])))                 # reported
    written.append(save("tab_A6_both_arms.tex", a6_table(["unconditional", "covariate-adjusted"])))   # robustness

# ------------------------------------------------------------- A7 LP-DiD -----
a7 = latest(T4A, "*_A7_lpdid.csv")


def lp_table(df_, horizons, arm):
    L = [r"\begin{tabular}{l*{7}{c}}", r"\toprule", HEAD7]
    for h in horizons:
        r = df_[(df_.arm == arm) & (df_.h == h)].set_index("outcome")
        lab = f"$h={h}$" if h < 0 else f"$h=+{h}$" if h > 0 else "$h=0$"
        L.append(lab + " & " + " & ".join(coef(r.loc[o, "beta"], r.loc[o, "p"], DEC[o]) if o in r.index else "" for o in OUTS) + r" \\")
        L.append(" & " + " & ".join(se(r.loc[o, "se"], DEC[o]) if o in r.index else "" for o in OUTS) + r" \\")
    r = df_[(df_.arm == arm) & (df_.h == 0)].set_index("outcome")
    L += [r"\midrule",
          "Treated counties ($h=0$) & " + " & ".join(f"{int(r.loc[o,'n_treated']):,}" if o in r.index else "" for o in OUTS) + r" \\",
          "Clean controls ($h=0$) & " + " & ".join(f"{int(r.loc[o,'n_control']):,}" if o in r.index else "" for o in OUTS) + r" \\",
          "Event-year FE & " + " & ".join(["Yes"] * 7) + r" \\",
          "Controls at $t-1$ & " + " & ".join(["Yes" if arm != "no controls" else "No"] * 7) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


if a7 is not None and len(a7):
    written.append(save("tab_A7_lpdid.tex", lp_table(a7, [-5, -3, -2, 0, 2, 5], "step-8 at t-1")))

# ------------------------------------------------------------- VIF -----------
vif = latest(T4A, "*_VIF_audit.csv")
TLAB = {"T1": "T1 presence", "T2": "T2 count", "T3": "T3 per 10,000", "T4": r"T4 $\log(1+$count$)$", "E2": "E2 absorbing"}


def vif_table(model, outs, treats, head, show_treat=True):
    d = vif[vif.model == model]
    L = [r"\begin{tabular}{l*{%d}{c}}" % len(outs), r"\toprule", head]
    for t in treats:
        r = d[d.treatment_id == t].set_index("outcome")
        L.append(TLAB[t] + " & " + " & ".join(f"{r.loc[o,'vif_max']:.2f}" if o in r.index else "" for o in outs) + r" \\")
        if show_treat:
            L.append(" & " + " & ".join(f"({r.loc[o,'vif_treat']:.2f})" if o in r.index else "" for o in outs) + r" \\")
    L += [r"\midrule", "Max VIF in table & \\multicolumn{%d}{c}{%.2f}" % (len(outs), d.vif_max.max()) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


if vif is not None:
    for m in ["A1", "A2"]:
        written.append(save(f"tab_vif_{m}.tex", vif_table(m, OUTS, ["T1", "T2", "T3", "T4", "E2"], HEAD7)))
    written.append(save("tab_vif_A4.tex", vif_table("A4", ["Poor MH Days", "Frequent Mental Distress"],
                        ["T1", "T2", "T3", "T4"], " & Poor MH days & Freq.\\ mental distress \\\\\n & (1) & (2) \\\\\n\\midrule")))
    written.append(save("tab_vif_A5.tex", vif_table("A5", OUTS, ["E2"], HEAD7, show_treat=False)))

# ------------------------------------------------------------- Conley --------
con = latest(T4A, "*_A2_conley_se.csv")
if con is not None and grid is not None:
    L = [r"\begin{tabular}{l*{7}{c}}", r"\toprule", HEAD7]
    for t in ["T1", "E2"]:
        g = grid[(grid.registry == "A2") & (grid.treatment_id == t)].set_index("outcome")
        c = con[con.treatment_id == t].set_index("outcome")
        L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{TLAB[t]}}}}} \\\\")
        L.append(r"$\hat\beta$ & " + " & ".join(f"{g.loc[o,'beta']:,.{DEC[o]}f}" for o in OUTS) + r" \\")
        for lab, colname in [("State cluster", "se_state"), ("Conley 100 km", "se_conley_100km"),
                             ("Conley 250 km", "se_conley_250km"), ("Conley 500 km", "se_conley_500km")]:
            src = g if colname == "se_state" else c
            L.append(lab + " & " + " & ".join(se(src.loc[o, colname], DEC[o]) if o in src.index else "" for o in OUTS) + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    written.append(save("tab_A2_conley.tex", "\n".join(L)))

# ------------------------------------------------------------- script4r ------
r0 = latest(T4R, "*_R0_entry_events.csv")
if r0 is not None:
    L = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Year & First entry & Any increase & Any decrease \\", r"\midrule"]
    for _, r in r0.iterrows():
        if str(r.year) == "total":
            L.append(r"\midrule")
        L.append(f"{r.year} & {int(r.counties_first_entry)} & {int(r.counties_any_increase)} & {int(r.counties_any_decrease)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    written.append(save("tab_R0_entry_events.tex", "\n".join(L)))
tw = latest(T4R, "*_R1_R2_twfe.csv")
if tw is not None:
    L = [r"\begin{tabular}{l*{7}{c}}", r"\toprule", HEAD7]
    for i, (mod, title, lab) in enumerate([("R1", "Panel A. Number of slaughter plants (F1), 2017--2023", "Slaughter plants"),
                                           ("R2", "Panel B. Entry of an additional plant, absorbing (F2), 2012--2023", "Post entry (0/1)")]):
        r = tw[tw.model == mod].set_index("outcome")
        L.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}} \\\\")
        L.append(lab + " & " + " & ".join(coef(r.loc[o, "beta"], r.loc[o, "p_state"], DEC[o]) if o in r.index else "" for o in OUTS) + r" \\")
        L.append(" & " + " & ".join(se(r.loc[o, "se_state"], DEC[o]) if o in r.index else "" for o in OUTS) + r" \\")
        for rl, cname, fmt in [("Observations", "N", "{:,.0f}"), ("Switching counties", "n_switchers", "{:,.0f}"),
                               (r"MDE (within-county SD)", "MDE_sd_within", "{:.3f}")]:
            L.append(rl + " & " + " & ".join(fmt.format(r.loc[o, cname]) if o in r.index else "" for o in OUTS) + r" \\")
        if i == 0:
            L.append(r"\addlinespace")
    yes = " & ".join(["Yes"] * 7)
    L += [r"\midrule", "Controls & " + yes + r" \\", "County FE & " + yes + r" \\", "Year FE & " + yes + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    written.append(save("tab_R1_R2.tex", "\n".join(L)))
r4 = latest(T4R, "*_R4_lpdid.csv")
if r4 is not None and len(r4):
    written.append(save("tab_R4_lpdid.tex", lp_table(r4, [-4, -3, -2, 0, 2, 4], "step-8 at t-1")))

print("written:", written)

# ------------------------------------------------------------- document ------
NOTES = {
 "tab_A2.tex": ("Large dairy CAFOs and county outcomes: two-way fixed effects (A2)",
   "Each coefficient is from a separate regression of the column outcome on the panel's treatment, county and year fixed effects and the step-8 controls (median household income, unemployment, \\% 65 and older, \\% Hispanic, \\% below 18, children in poverty, \\% Asian/AIAN/NHPI, \\% Black). T4 also includes log population. Only switching counties identify the coefficient. Standard errors clustered by state in parentheses."),
 "tab_A2_all_treatments.tex": ("ROBUSTNESS. Two-way fixed effects, all treatments including the count T2 (A2)",
   "As the A2 table, with the raw count of large dairy CAFOs (T2) added. T2 and T4 include log population. Not in the reported set."),
 "tab_A1.tex": ("ROBUSTNESS. Large dairy CAFOs and county outcomes: pooled with state fixed effects (A1)",
   "As the A2 table, with state rather than county fixed effects. Standard errors clustered by state in parentheses."),
 "tab_A4.tex": ("Dairy CAFOs, slaughterhouses and mental health, 2017--2023 (A4)",
   "Each column is one regression of the outcome on dairy measure $D$, FSIS-plant indicator $F$ and $D\\times F$, with county and year fixed effects and step-8 controls. The only agricultural-census transition inside 2017--2023 is 2022; crime and mortality outcomes end earlier and are omitted. Standard errors clustered by state in parentheses."),
 "tab_A5_pretrend.tex": ("Event-study pre-trend tests (A5)",
   "Pre-period event-study coefficients ($e<-1$) significant at 5\\%, state-clustered. The second row per outcome adds 2000 log population interacted with year. The placebo uses log population as the outcome."),
 "tab_A6_both_arms.tex": ("ROBUSTNESS. Callaway--Sant'Anna, unconditional and covariate-adjusted arms (A6)",
   "As the A6 table, adding the unconditional arm: parallel trends assumed with no covariates. Not in the reported set."),
 "tab_A4_all_treatments.tex": ("ROBUSTNESS. Dairy CAFOs, slaughterhouses and mental health, all treatments (A4)",
   "As the A4 table, with the count T2 added. Not in the reported set."),
 "tab_A6.tex": ("Two-way fixed effects versus Callaway--Sant'Anna, absorbing treatment E2 (A6)",
   "All rows use E2. CS: cohort-size-weighted average of post-treatment ATT$(g,t)$ with not-yet- and never-treated controls; standard errors from a 300-replication state cluster bootstrap."),
 "tab_A7_lpdid.tex": ("LP-DiD, absorbing treatment E2 (A7)",
   "Each cell is one stacked regression of $Y_{t+h}-Y_{t-1}$ on a newly-treated indicator with event-year fixed effects and step-8 controls at $t-1$, on newly treated and clean (not-yet- and never-treated) counties. $h<-1$ are pre-trend tests. Standard errors clustered by state."),
 "tab_A2_conley.tex": ("Spatial standard errors for A2",
   "Conley standard errors with a Bartlett kernel within year at the stated cutoff, plus serial correlation within county."),
 "tab_R0_entry_events.tex": ("Slaughterhouse entry events by year",
   "Counties whose number of FSIS slaughter establishments rises, by year. First entry defines the F2 cohort."),
 "tab_R1_R2.tex": ("Slaughterhouses as the treatment (R1, R2)",
   "County and year fixed effects, step-8 controls, standard errors clustered by state. MDE at 80\\% power, $2.80\\times$SE, in within-county SDs of the outcome."),
 "tab_R4_lpdid.tex": ("LP-DiD, slaughterhouse entry (R4)",
   "As the A7 table, with cohorts defined by first slaughterhouse entry (2018--2023)."),
}
MAIN_TABLES   = ["tab_A2.tex", "tab_A4.tex", "tab_A5_pretrend.tex", "tab_A6.tex",
                 "tab_R0_entry_events.tex", "tab_R1_R2.tex"]
ROBUST_TABLES = ["tab_A2_all_treatments.tex", "tab_A1.tex", "tab_A4_all_treatments.tex",
                 "tab_A6_both_arms.tex", "tab_A7_lpdid.tex", "tab_A2_conley.tex", "tab_R4_lpdid.tex"]


def float_block(names):
    out = []
    for name in names:
        if name not in written:
            continue
        cap, note = NOTES[name]
        out.append("\\begin{table}[p]\\centering\n\\caption{%s}\n\\adjustbox{max width=\\linewidth}{\\input{tables/%s}}\n"
                   "\\par\\vspace{4pt}\\begin{minipage}{\\linewidth}\\footnotesize\\textit{Notes:} %s %s.\\end{minipage}\\end{table}\n"
                   % (cap, name, note, STARNOTE))
    return out


body, body_robust = float_block(MAIN_TABLES), float_block(ROBUST_TABLES)
# _BODY_regressions.tex = REPORTED exhibits only (this is what script4p inputs).
open(os.path.join(LATEX_OUT, "_BODY_regressions.tex"), "w").write("\n".join(body))
open(os.path.join(LATEX_OUT, "_BODY_robustness.tex"), "w").write("\n".join(body_robust))
doc = ("\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{booktabs,amsmath,caption,adjustbox}\n"
       "\\captionsetup{labelfont=bf,font=small,skip=6pt}\n\\begin{document}\n" + "\n".join(body)
       + "\n\\clearpage\n\\section*{Robustness: specifications outside the reported set}\n" + "\n".join(body_robust)
       + "\\end{document}\n")
dp = os.path.join(LATEX_OUT, "REGRESSION_TABLES.tex")
open(dp, "w").write(doc)
try:
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "REGRESSION_TABLES.tex"], cwd=LATEX_OUT,
                   capture_output=True, timeout=120)
    print("compiled:", os.path.join(LATEX_OUT, "REGRESSION_TABLES.pdf"))
except Exception as e:
    print("pdflatex not run:", e)
