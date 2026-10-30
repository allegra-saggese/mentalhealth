#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4e-tables.py -- every LaTeX exhibit built from result files: regression tables, the full appendix, the beamer frames (no estimation).

Sections (run in this order; select with --only, e.g. --only tables appendix):
  tables    regression tables in paper format      (was script4n-regression-tables.py)
  appendix  one document with every table          (was script4p-full-appendix.py)
  frames    drop-in beamer frames, no verdicts     (was script4q-beamer-frames.py)
  slides    one .tex body per table frame of slides/deck.tex, newest result files (new 2026-10-26, D-56)
Reads:    (result CSVs from Data/output/tables/ by section tag, packages.py: 4a-designs, 4a-fsis, 4b-controls, 4b-balance, ...)
          tables: newest CSVs of script4a (4a-designs, 4a-fsis)
          appendix: LATEX_OUT/_BODY_balance.tex (script4d), _BODY_regressions.tex (section tables),
                    TABLE_covariates_EXCLUDED_{FOR_CAUSE, NO_PRETREND_TEST}.tex (no generating script, QA #21)
          frames: newest CSVs of script4a and script4b, tables/*_4b-balance_balance_panel.csv (script4b balance)
Writes:   LATEX_OUT/slides/<section>__<frame>.tex (section slides);  LATEX_OUT/tables/*.tex, _BODY_regressions.tex (reported), _BODY_robustness.tex (cut specifications),
          REGRESSION_TABLES.tex/.pdf; FULL_RESULTS_APPENDIX.tex/.pdf; BEAMER_FRAMES.tex, BEAMER_FRAMES_robustness.tex
Run:      PYTHONHASHSEED=0 python3 script4e-tables.py [--only tables|appendix|frames ...]
Method:   the reported set is REPORT_* in script4_treatment (D-32); the other specifications go to the robustness tables.
Notes:    some sample figures in frame text are hard-coded and describe the 2026-10-12 run (QA #11).
"""
import warnings; warnings.filterwarnings("ignore")
import argparse
import glob
import math
import subprocess

import numpy as np
import pandas as pd

from script4_treatment import db_data, os, date, LATEX_OUT, REPORT_TREATMENTS, REPORT_CS_ARM
from packages import tables_dir, figs_dir, maps_dir, out_glob
from estimators import INCLUDE_LPDID                      # False: no LP-DiD tables or columns (D-57)

OUT = LATEX_OUT   # output/latex (packages.latex_dir); .tex names are fixed because LaTeX \input's them
TAB = os.path.join(LATEX_OUT, "tables")
# Inputs are found by the output-name tag of the section that wrote them (packages.py), in the flat tables/ folder.
T4A = "4a-designs"                  # was folder tables/script4a (designs part)
T4C = "4b-controls"                 # was folder tables/script4a (control diagnostics part)
T4B = "4b-covariate-sequence"       # z-archive/script4b-covariate-sequence.py (09-23 run); was folder tables/script4b
T4R = "4a-fsis"                     # was folder tables/script4r

# =============================================================== shared helper
def latest(tag, pat):
    """Newest result file written under `tag` (packages.py naming) matching `pat`, or None."""
    f = out_glob(tables_dir, tag, pat)
    return pd.read_csv(f[-1]) if f else None


def panel_date():
    """Date in the file name of the newest analysis panel (the panel every result file here was run on)."""
    return os.path.basename(sorted(glob.glob(os.path.join(db_data, "merged", "*_panel.csv")))[-1])[:10]


# =============================================================== section tables (was script4n)
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


TCOL = {"T1": "tr_e1_lg_bin", "T2": "tr_i3_lg_count", "T3": "tr_i4_lg_p10k", "T4": "tr_c3_log_lg"}

TLAB = {"T1": "T1 presence", "T2": "T2 count", "T3": "T3 per 10,000", "T4": r"T4 $\log(1+$count$)$", "E2": "E2 absorbing"}

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
if not INCLUDE_LPDID:
    ROBUST_TABLES = [t for t in ROBUST_TABLES if "lpdid" not in t]


def regression_tables():
    """Build the table fragments, the two _BODY_ files and REGRESSION_TABLES.tex/.pdf from the newest script4a and script4r CSVs."""
    os.makedirs(TAB, exist_ok=True)
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


    if INCLUDE_LPDID and a7 is not None and len(a7):
        written.append(save("tab_A7_lpdid.tex", lp_table(a7, [-5, -3, -2, 0, 2, 5], "step-8 at t-1")))

    # ------------------------------------------------------------- VIF -----------
    vif = latest(T4A, "*_VIF_audit.csv")


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
    if INCLUDE_LPDID and r4 is not None and len(r4):
        written.append(save("tab_R4_lpdid.tex", lp_table(r4, [-4, -3, -2, 0, 2, 4], "step-8 at t-1")))

    print("written:", written)

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
    # _BODY_regressions.tex = REPORTED exhibits only (this is what the appendix section inputs).
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


# =============================================================== section appendix (was script4p)
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
Panel        & 2026-10-12 (run 2: revised demographic controls) \\
Sample       & rural US counties, NCHS codes 3--6, 2000--2023 \\
Treatments   & T1--T4, plus E2 for the staggered-DiD estimators \\
Outcomes     & 7 (3 mental health / mortality, 4 crime) \\
Controls     & step-8 set, revised 2026-10-12 \\
Inference    & state-clustered standard errors throughout \\
\end{tabular}\par}
\vspace{2cm}
{\large\today\par}
\vfill
{\small\itshape Stars: $^{*}\,p<0.10$, $^{**}\,p<0.05$, $^{***}\,p<0.01$.\par}
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


def full_appendix():
    """Stitch the balance, regression and excluded-covariate fragments into FULL_RESULTS_APPENDIX.tex and compile it."""
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


# =============================================================== section frames (was script4q)
SHORT = {"Poor MH Days": "Poor MH", "Frequent Mental Distress": "Freq.\\ Distress",
         "Deaths of Despair": "Despair", "Aggravated Assault": "Agg.\\ Assault",
         "Assault, all severities": "Assault", "Violence index (partial)": "Violence",
         "NIBRS curated total": "NIBRS"}
assert OUTS == list(SHORT)   # the frames use the same seven outcomes, in the same order, as the tables


def cf(b, se, p, d=3):
    if pd.isna(b): return "---"
    s = f"{b:.{d}f}"
    if pd.notna(p) and p < 0.01: return f"\\alert{{{s}$^{{***}}$\\,\\tiny({se:.{d}f})}}"
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


def beamer_frames():
    """Write BEAMER_FRAMES.tex (reported set) and BEAMER_FRAMES_robustness.tex (cut specifications)."""
    FR = []   # frames for specifications outside the reported set
    F = ["% ==========================================================================\n"
         "% BEAMER FRAMES -- script4e-tables.py\n"
         f"% Panel {panel_date()} | step-8 controls | generated {date.today()}\n"
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
\textbf{T2}$^\dagger$ & Count & Raw number of large dairy operations & \texttt{tr\_i3\_lg\_count} \\[0.3em]
\textbf{T3} & Per capita & Large dairy operations per 10{,}000 residents & \texttt{tr\_i4\_lg\_p10k} \\[0.3em]
\textbf{T4} & Log(count) & $\log(1+\text{count})$ & \texttt{tr\_c3\_log\_lg} \\
\bottomrule
\end{tabular}
\end{center}
\vspace{0.5em}
{\footnotesize
\textbf{Size:} large $=$ 500$+$ milk cows (USDA NASS top inventory bin). \emph{Not} EPA's 700$+$ regulatory threshold --- USDA publishes no 700 cutpoint.\\[0.3em]
\textbf{$\log(1+\cdot)$:} the $+1$ is required, not cosmetic --- 81.4\% of county-years have zero large dairy operations and plain $\log$ would discard them.\\[0.3em]
$^\dagger$ T2 is estimated but not reported (robustness only).\\[0.3em]
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
                        f"({r.se_state:.4f}) & {int(r.N):,} & {int(r.n_switchers):,} & {r.vif_max:.2f} \\\\")
        F.append(frame("Covariates entered one at a time", tbl(
            "rlrrrrr",
            "Step & Covariate added & $\\beta$ & (SE) & $N$ & Switchers & Max VIF \\\\", rows,
            r"T1, Poor MH Days, county $+$ year FE, state-clustered SE. "
            r"The run estimates 4 treatments $\times$ 7 outcomes $\times$ 2 FE structures "
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

    FR.append(frame("A1: pooled cross-section --- specification", r"""  $$Y_{it} = \beta D_{it} + X_{it}\gamma + \alpha_{s(i)} + \delta_t + \varepsilon_{it}$$
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
             r"Step-8 controls, state-clustered SE in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. "
             r"Units differ by row and by column; magnitudes are \emph{not} comparable across either."),
            ("A2", "A2: two-way fixed effects --- results",
             r"Step-8 controls, state-clustered SE in parentheses. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. "
             r"Switcher counts differ sharply across treatments on the identical sample "
             r"(T1 253, T3 707), so $N$ alone does not describe the evidence behind a column.")]:
            v = g[g.registry == reg]
            rows = []
            # A2 (reported) shows the reported treatments; A1 (robustness) shows all five.
            shown = ([t for t in ["T1", "T2", "T3", "T4", "E2"] if t in REPORT_TREATMENTS]
                     if reg == "A2" else ["T1", "T2", "T3", "T4", "E2"])
            for tid in shown:
                cells = [cf(v[(v.treatment_id==tid)&(v.outcome==o)].iloc[0].beta,
                            v[(v.treatment_id==tid)&(v.outcome==o)].iloc[0].se_state,
                            v[(v.treatment_id==tid)&(v.outcome==o)].iloc[0].p_state)
                         if len(v[(v.treatment_id==tid)&(v.outcome==o)]) else "---" for o in OUTS]
                rows.append(f"\\textbf{{{tid}}} & " + " & ".join(cells) + " \\\\")
            sw = [f"{int(v[(v.treatment_id==t)&(v.outcome=='Poor MH Days')].iloc[0].n_switchers):,}"
                  for t in shown]
            rows += ["\\midrule", "Switchers & \\multicolumn{%d}{l}{\\tiny %s \\quad (Poor MH Days)} \\\\"
                     % (len(OUTS), " \\; ".join(f"{t} {n}" for t, n in zip(shown, sw)))]
            (F if reg == "A2" else FR).append(frame(title, tbl("l"+"r"*len(OUTS),
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

    # A3 removed 2026-10-12 (Sept 30 feedback).

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
    A4_SHOWN = [t for t in ["T1", "T2", "T3", "T4"] if t in REPORT_TREATMENTS]
    if a4 is not None and "identified" in a4.columns:
        ok = a4[a4.identified]
        rows = []
        for o in list(dict.fromkeys(ok.outcome)):
            cells = [cf(ok[(ok.treatment_id==t)&(ok.outcome==o)].iloc[0].beta,
                        ok[(ok.treatment_id==t)&(ok.outcome==o)].iloc[0].se_state,
                        ok[(ok.treatment_id==t)&(ok.outcome==o)].iloc[0].p_state)
                     if len(ok[(ok.treatment_id==t)&(ok.outcome==o)]) else "---"
                     for t in A4_SHOWN]
            rows.append(f"{SHORT[o]} & " + " & ".join(cells) + " \\\\")
        rows += ["\\midrule", "$N$ & \\multicolumn{%d}{l}{\\tiny 17{,}975 \\quad switchers 89 (T1)} \\\\" % len(A4_SHOWN)]
        F.append(frame("A4: dairy $\\times$ slaughterhouse --- results", tbl(
            "l" + "r" * len(A4_SHOWN), " & " + " & ".join(A4_SHOWN) + " \\\\", rows,
            r"Coefficient on the interaction. County $+$ year FE, step-8 controls, state-clustered SE. "
            r"2017--2023. $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. Five outcomes (despair and all four crime measures) "
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
  {\footnotesize \textbf{Two arms estimated; the covariate-adjusted arm is reported.} \emph{Unconditional}: raw difference of long differences; parallel trends assumed unconditionally. \emph{Covariate-adjusted}: outcome regression on the step-8 covariates at base period $g-1$, fitted on the clean control group only, so parallel trends is assumed conditionally --- matching A2.\\[0.3em]
  Cohorts observed: 2007 (284 counties), 2012 (142), 2017 (101), 2022 (45). A cohort is usable only if $g-1$ falls inside the outcome's year range, so CHR outcomes beginning in 2010 lose the 2007 cohort.}"""))

    cmp_ = latest(T4A, "*_HEADLINE_twfe_vs_cs.csv")
    csag = latest(T4A, "*_A6_cs_aggregated.csv")
    if cmp_ is not None:
        rows = []
        for _, r in cmp_.iterrows():
            rows.append(f"{SHORT[r.outcome]} & "
                        f"{cf(r.TWFE_beta, r.TWFE_se, 0.01 if r.TWFE_sig else 0.5)} & "
                        f"{cf(r.CS_ATT, r.CS_se, 0.01 if r.CS_sig else 0.5)} \\\\")
        F.append(frame("A6: Callaway--Sant'Anna --- results", tbl(
            "lrr", " & TWFE & CS adjusted \\\\", rows,
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
  The right-hand side also matters: adding T2 plus $\log$ population moves the block maximum from 1.45 to 1.57. A4's interaction moves it further.}"""))

        summ = (va.groupby("model").agg(n=("vif_max","size"), w=("within","first"),
                                        mx=("vif_max","max"), md=("vif_max","median"),
                                        z=("n_zero_var","sum")).reset_index())
        rows = [f"{r.model} & {'Within' if r.w else 'Raw levels'} & {int(r.n)} & "
                f"{r.mx:.2f} & {r.md:.2f} & 0 & {int(r.z)} \\\\" for _, r in summ.iterrows()]
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
    bal = latest("4b-balance", "*_balance_panel.csv")
    if bal is not None:
        for tid in [t for t in ["T1", "T2", "T3", "T4", "E2"] if t in REPORT_TREATMENTS]:
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
    open(os.path.join(OUT, "BEAMER_FRAMES_robustness.tex"), "w").write("".join(FR))
    n = sum(1 for x in F if x.startswith("\\begin{frame}"))
    print(f"wrote {p}\n  {n} frames")


# =============================================================== section slides (new 2026-10-26, D-56)
# One fragment per table frame of slides/deck.tex: the frame BODY only (the deck keeps the owner's frame titles). Every number comes from the
# newest result file; notes are neutral (source, sample, estimator, clustering, panel date). Wide tables shrink to the text width.
SLIDE_OUT = os.path.join(OUT, "slides")
OUT_NAME = {"Poor MH Days": "Poor MH Days", "Frequent Mental Distress": "Freq. mental distress", "Deaths of Despair": "Deaths of despair",
            "Aggravated Assault": "Aggravated assault", "Assault, all severities": "Assault, all severities",
            "Violence index (partial)": "Violence index (partial)", "NIBRS curated total": "NIBRS curated total"}
OUT_UNIT = {"Poor MH Days": ("Mean days in the past 30", "CHR (BRFSS)"), "Frequent Mental Distress": ("Per 100,000 adults", "CHR (BRFSS)"),
            "Deaths of Despair": ("Crude deaths per 100,000", "CDC WONDER"), "Aggravated Assault": ("Arrests per 100,000", "NIBRS"),
            "Assault, all severities": ("Arrests per 100,000", "NIBRS"), "Violence index (partial)": ("Arrests per 100,000", "NIBRS"),
            "NIBRS curated total": ("Arrests per 100,000", "NIBRS")}
TREATMENT_ROWS = [("T1", "Presence", r"$1$ if the county has $\geq 1$ large dairy operation that year", "tr_e1_lg_bin"),
                  ("T3", "Per capita", "Large dairy operations per 10,000 residents", "tr_i4_lg_p10k"),
                  ("T4", "Log count", r"$\log(1+\text{count})$ of large dairy operations", "tr_c3_log_lg"),
                  ("E2", "First expansion", r"First census wave at which the large-dairy count rises ($n \rightarrow n+k$, $k \geq 1$); absorbing", "tr_e2_add_absorb"),
                  ("E3", "First entry", r"First census wave at which the count goes from $0$ to $\geq 1$; comparison: counties that never had one", "tr_e3_entry_absorb")]


def _esc(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _f(x, d=3):
    return "" if pd.isna(x) else f"{x:,.{d}f}"


def _cs(b, s, d=3):
    return "" if pd.isna(b) else f"{b:,.{d}f} ({s:,.{d}f})"


def _yn(v):
    return r"\textbullet" if bool(v) else ""


def _stars(p):
    """Significance stars for the slide regression tables: * p<0.10, ** p<0.05, *** p<0.01."""
    return "" if pd.isna(p) else ("***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "")


def _est_se_rows(label, cells):
    """Two table rows in the layout of a journal regression table: coefficient with stars, then the standard error in parentheses below.
    cells: list of (beta, se, p); a missing beta leaves the cell blank."""
    est = " & ".join("" if pd.isna(b) else f"{b:,.3f}" + (f"$^{{{_stars(p)}}}$" if _stars(p) else "") for b, _, p in cells)
    se = " & ".join("" if pd.isna(b) else f"({s:,.3f})" for b, s, _ in cells)
    return [f"{label} & {est}", f" & {se}"]


def _reg_tab(names, body, footer):
    """Journal-style regression table: numbered columns with the outcome stacked over two lines, coefficient/SE rows, a rule, then footer rows (N, fixed effects, ...)."""
    head = " & ".join(f"\\shortstack{{{n}}}" for n in names)
    nums = " & ".join(f"({i + 1})" for i in range(len(names)))
    return ("\\begin{tabular}{@{}l" + "c" * len(names) + "@{}}\n\\toprule\n & " + nums + "\\\\\n & " + head + "\\\\\n\\midrule\n"
            + "\\\\\n".join(body) + "\\\\\n\\midrule\n" + "\\\\\n".join(footer) + "\\\\\n\\bottomrule\n\\end{tabular}")


def _fragment(name, tabulars, note, size=r"\scriptsize"):
    """Write one fragment: the tabular(s) side by side or stacked as given, shrunk to the text width when wider, then a neutral note."""
    body = []
    for tab in (tabulars if isinstance(tabulars, list) else [tabulars]):
        body.append("\\setbox0=\\hbox{" + size + "\n" + tab + "}%\n\\ifdim\\wd0>\\textwidth\\resizebox{\\textwidth}{!}{\\box0}\\else\\box0\\fi\\par")
    import re
    note = re.sub(r"(?<!\\)([_%])", r"\\\1", note)                 # file stems contain underscores; a bare % would comment out the closing brace
    text = "{\\centering\n" + "\n\\vspace{3pt}\n".join(body) + "\\par}\n\\vspace{-2pt}\n{\\tiny\\baselineskip=6.5pt " + note + "\\par}\n"
    os.makedirs(SLIDE_OUT, exist_ok=True)
    path = os.path.join(SLIDE_OUT, name + ".tex")
    open(path, "w").write(text)
    print("wrote", path)


def _tab(colspec, header, rows):
    return f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header}\\\\\n\\midrule\n" + "\\\\\n".join(rows) + "\\\\\n\\bottomrule\n\\end{tabular}"


def slide_fragments():
    from designs import load_designs, usable_sample, OUTCOME_WINDOWS
    from script4_treatment import OUTCOMES, COVARIATE_ORDER, load_panel
    pdate = panel_date()
    stem = lambda tag, pat: os.path.basename(out_glob(tables_dir, tag, pat)[-1])[:10]   # date of the newest input
    TZ, TZA = "4c-wild", "4d-estimators"     # were folders tables/script4z, tables/script4za
    base_note = f"Panel {pdate}; county and year fixed effects, 8 pre-committed controls, standard errors clustered by state."

    # ---- treatment definitions
    rows = [f"\\textbf{{{k}}} & {n} & {d} & \\texttt{{{_esc(c)}}}" for k, n, d, c in TREATMENT_ROWS]
    _fragment("treatment__definitions", _tab(r"@{}llp{6.2cm}l@{}", "ID & Name & Definition & Panel column", rows),
              "Large = 500+ milk cows (USDA NASS top inventory bin). Treatment is measured at census waves (2002, 2007, 2012, 2017, 2022) and forward-filled "
              "between them; a blank census-year count is 0 (D-34, D-53). T2, the raw count, is estimated but not reported (D-32). " + base_note, size=r"\footnotesize")

    # ---- sampling frame table (estimation samples of the triangulation run)
    s = pd.read_csv(out_glob(tables_dir, TZA, "*_samples.csv")[-1])
    e2, e3 = s[s.design == "first expansion"].set_index("outcome"), s[s.design != "first expansion"].set_index("outcome")
    rows = [f"{OUT_NAME[o]} & {e2.loc[o, 'years']} & {int(e2.loc[o, 'counties']):,} ({int(e2.loc[o, 'treated_counties'])}) & "
            f"{int(e3.loc[o, 'counties']):,} ({int(e3.loc[o, 'treated_counties'])}) & {_esc(e2.loc[o, 'cohorts'])}" for o in OUTCOMES]
    _fragment("treatment__sampling_frame_table", _tab(r"@{}lcccl@{}", r"Outcome & Years & E2: counties (treated) & E3: counties (treated) & E2 cohorts",
              rows), f"Complete-case estimation samples; a treated county needs at least one observed year before its cohort wave. Source: script4za samples file "
              f"({stem(TZA, '*_samples.csv')}). " + base_note)

    # ---- treated counties by census wave, Poor MH Days sample
    _, D, _, _ = load_designs()
    ctrl = COVARIATE_ORDER[:8]
    cnt = {}
    for dn, tag in (("first expansion", "E2"), ("first entry (clean controls)", "E3")):
        u = usable_sample(D[dn]["data"], OUTCOMES["Poor MH Days"], D[dn]["treat"], D[dn]["cohort"], ctrl).drop_duplicates("fips")
        cnt[tag] = u.cohort.value_counts().sort_index()
        cnt[tag + "_comp"] = int(u.cohort.isna().sum())
    waves = sorted(set(cnt["E2"].index) | set(cnt["E3"].index))
    rows = [f"{int(w)} & {int(cnt['E2'].get(w, 0))} & {int(cnt['E3'].get(w, 0))}" for w in waves]
    rows.append(f"Comparison counties & {cnt['E2_comp']:,} & {cnt['E3_comp']:,}")
    _fragment("treatment__temporal_counts", _tab(r"@{}lcc@{}", "Cohort wave & E2 first expansion & E3 first entry", rows),
              "Counties in the Poor MH Days estimation sample by the census wave of their treatment. Source: designs.usable_sample on the panel. " + base_note, size=r"\footnotesize")

    # ---- controls: the 24 candidates
    cov = pd.read_csv(out_glob(tables_dir, "1c", "*_control_coverage.csv")[-1])
    cov = cov.sort_values(["type", "fill_pct"], ascending=[True, False]).reset_index(drop=True)
    half = (len(cov) + 1) // 2
    blocks = []
    for ch in (cov.iloc[:half], cov.iloc[half:]):
        blocks.append(_tab(r"@{}llrc@{}", "Control & Type & \\% filled & Sel.",
                           [f"{_esc(v.replace('_per100k', '').replace('_', ' ')[:30])} & {t.split(' ')[0]} & {p:.1f} & {_yn(sel)}"
                            for v, t, p, sel in zip(ch.variable, ch.type, ch.fill_pct, ch.selected)]))
    _fragment("controls__possible", [r"\begin{tabular}{@{}ll@{}}" + blocks[0] + "&" + blocks[1] + r"\end{tabular}"],
              f"The 24 candidate controls; percent of observations filled over the rural panel county-years 2015-2020; Sel. = in the pre-committed set of eight. "
              f"Source: script1c control coverage ({stem('1c', '*_control_coverage.csv')}).", size=r"\tiny")

    # ---- controls: the eight
    pre = pd.read_csv(out_glob(tables_dir, T4C, "*_control_pretrends.csv")[-1]).set_index("variable")
    corr = pd.read_csv(out_glob(tables_dir, T4C, "*_control_correlations_selected8.csv")[-1]).set_index("variable")
    ctl = pd.concat([cov.set_index("variable").loc[ctrl, ["type", "fill_pct"]], pre.loc[ctrl, ["description", "n_pre_significant", "n_pre_coefs"]]], axis=1)
    left = _tab(r"@{}rlcr@{}", "\\# & Control & Type & \\% filled",
                [f"{i + 1} & {_esc(ctl.loc[c, 'description'])} & {ctl.loc[c, 'type'].split(' ')[0]} & {ctl.loc[c, 'fill_pct']:.1f}" for i, c in enumerate(ctrl)])
    right = _tab("@{}r" + "r" * 8 + "@{}", "& " + " & ".join(str(i + 1) for i in range(8)),
                 [f"{i + 1} & " + " & ".join(f"{corr.loc[c, d]:.2f}" for d in ctrl) for i, c in enumerate(ctrl)])
    _fragment("controls__selected8", [r"\begin{tabular}{@{}ll@{}}" + left + "&\\hspace{6pt}" + right + r"\end{tabular}"],
              "The eight pre-committed controls (D-16, D-31) and their within-county correlation matrix (numbers refer to the list). Source: script4b controls "
              f"({stem(T4C, '*_control_correlations_selected8.csv')}), script1c coverage.", size=r"\tiny")

    # ---- outcomes and units
    df = load_panel()
    rows = []
    for o, col in OUTCOMES.items():
        lo, hi = OUTCOME_WINDOWS[o]
        x = df[df.year.between(lo, hi)][col].dropna()
        rows.append(f"{OUT_NAME[o]} & {OUT_UNIT[o][0]} & {OUT_UNIT[o][1]} & {lo}--{hi} & {x.mean():,.2f} & {x.std():,.2f}")
    _fragment("outcomes__units", _tab(r"@{}lllrrr@{}", "Outcome & Unit & Source & Years & Mean & SD", rows),
              "Mean and standard deviation over county-years of each outcome's descriptive window (designs.OUTCOME_WINDOWS; the estimation samples are data-defined). "
              "The NIBRS measures are arrests, not offences. Panel " + pdate + ".", size=r"\footnotesize")

    # ---- A2 TWFE results
    g = pd.read_csv(out_glob(tables_dir, T4A, "*_A1_A2_treatment_grid.csv")[-1])
    g = g[(g.registry == "A2") & g.treatment_id.isin(["T1", "T3", "T4", "E2"])]
    cols = [g[(g.outcome == o)] for o in OUTCOMES]
    rows = []
    for t, lab in (("T1", "T1 presence"), ("T3", "T3 per 10k"), ("T4", "T4 log count"), ("E2", "E2 first expansion")):
        cells = [tuple(c[c.treatment_id == t].iloc[0][["beta", "se_state", "p_state"]]) for c in cols]
        rows += _est_se_rows(lab, cells)
    yes = " & ".join("Yes" for _ in cols)
    footer = ["Observations (county-years) & " + " & ".join(f"{int(c.N.iloc[0]):,}" for c in cols),
              "Counties & " + " & ".join(f"{int(c.n_counties.iloc[0]):,}" for c in cols),
              "States (clusters) & " + " & ".join(f"{int(c.n_states.iloc[0])}" for c in cols),
              "County and year fixed effects & " + yes, "Controls (the control set of 8) & " + yes]
    tab = _reg_tab([r"Poor MH\\Days", r"Frequent\\distress", r"Deaths of\\despair", r"Aggravated\\assault", r"Assault,\\all", r"Violence\\index", r"NIBRS\\total"], rows, footer)
    _fragment("results__a2_twfe", tab,
              "Each coefficient is a separate two-way fixed effects regression of the column outcome on the row treatment; standard errors clustered by state in parentheses. "
              "* p < 0.10, ** p < 0.05, *** p < 0.01 (state-clustered p-values). Coefficients are in the outcome's own units (see the outcomes frame). Source: script4a A2 grid "
              f"({stem(T4A, '*_A1_A2_treatment_grid.csv')}). Panel {pdate}.")

    # ---- Callaway-Sant'Anna vs TWFE
    h = pd.read_csv(out_glob(tables_dir, T4A, "*_HEADLINE_twfe_vs_cs.csv")[-1])
    from scipy.stats import norm
    hr = list(h.itertuples())
    twfe_p = {o: g[(g.outcome == o) & (g.treatment_id == "E2")].iloc[0].p_state for o in OUTCOMES}
    cs_p = [2 * norm.sf(abs(r.CS_ATT / r.CS_se)) for r in hr]                   # normal approximation to the bootstrap standard error
    rows = _est_se_rows("TWFE, E2 first expansion", [(r.TWFE_beta, r.TWFE_se, twfe_p[r.outcome]) for r in hr])
    rows += _est_se_rows("Callaway-Sant'Anna", [(r.CS_ATT, r.CS_se, q) for r, q in zip(hr, cs_p)])
    footer = ["CS / TWFE & " + " & ".join(_f(r.ratio_CS_over_TWFE, 2) for r in hr),
              "Treated counties (TWFE) & " + " & ".join(f"{int(r.n_switchers_TWFE):,}" for r in hr),
              "Treated county-periods (CS) & " + " & ".join(f"{int(r.n_treated_CS):,}" for r in hr)]
    short = dict(zip(OUTCOMES, [r"Poor MH\\Days", r"Frequent\\distress", r"Deaths of\\despair", r"Aggravated\\assault", r"Assault,\\all", r"Violence\\index", r"NIBRS\\total"]))
    tab = _reg_tab([short[r.outcome] for r in hr], rows, footer)
    _fragment("results__cs", tab,
              "First expansion (E2). TWFE: two-way fixed effects, standard errors clustered by state. Callaway-Sant'Anna: covariate-adjusted overall ATT, not-yet-treated comparison, state bootstrap standard errors in parentheses. "
              "* p < 0.10, ** p < 0.05, *** p < 0.01 (TWFE: state-clustered p; Callaway-Sant'Anna: normal approximation to the bootstrap standard error). Source: script4a headline file "
              f"({stem(T4A, '*_HEADLINE_twfe_vs_cs.csv')}). Panel {pdate}.", size=r"\footnotesize")

    # ---- inference
    c = pd.read_csv(out_glob(tables_dir, TZ, "*_confirmatory_family.csv")[-1])
    rows = [f"{OUT_NAME[r.outcome]} & {r.treatment_id} & {_f(r.beta, 3)} & {_f(r.p_state, 3)} & {_f(r.p_wild, 3)} & {_f(r.p_romano_wolf_within_outcome, 3)} & {_f(r.p_holm_over_8, 3)}"
            for r in c.itertuples()]
    _fragment("results__inference", _tab(r"@{}llrrrrr@{}", "Outcome & Treatment & Coefficient & p (state-clustered) & p (wild bootstrap) & p (Romano-Wolf, within outcome) & p (Holm, all 8)", rows),
              f"Confirmatory family (D-42): Poor MH Days and Frequent Mental Distress by T1, T3, T4, E2. Wild cluster bootstrap-t, Webb weights, {int(c.reps.iloc[0]):,} draws, "
              f"seed {int(c.seed.iloc[0])}. Source: script4c ({stem(TZ, '*_confirmatory_family.csv')}). " + base_note, size=r"\footnotesize")

    # ---- estimator agreement (D-45)
    a = pd.read_csv(out_glob(tables_dir, TZA, "*_agreement.csv")[-1])
    if not INCLUDE_LPDID:
        a = a[a.estimator != "LP-DiD"]                                   # 3 of 4 rule (D-57); each estimator's flag is against CS alone, so the rest are unchanged
    rows = []
    for o in OUTCOMES:
        cells = []
        for dn in ("first expansion", "first entry (clean controls)"):
            x = a[(a.design == dn) & (a.outcome == o)]
            no = ", ".join(x[~x.agrees].estimator.str.replace(r" \(did2s\)", "", regex=True))
            cells.append(f"{int(x.agrees.sum())} of {len(x)}" + (f" (not: {_esc(no)})" if no else ""))
        rows.append(f"{OUT_NAME[o]} & " + " & ".join(cells))
    _fragment("compare__agreement", _tab(r"@{}lll@{}", "Outcome & E2 first expansion & E3 first entry", rows),
              "Estimators that agree with covariate-adjusted Callaway-Sant'Anna: same sign and overall ATT inside its 95% interval (rule D-45, 3 of 4 since D-57); four estimators compared "
              f"(Sun-Abraham, imputation, dCDH, stacked DiD). Source: script4d agreement file ({stem(TZA, '*_agreement.csv')}).", size=r"\footnotesize")

    # ---- HonestDiD
    hb = pd.read_csv(out_glob(tables_dir, "4b-honestdid", "*_honestdid_breakdown.csv")[-1])
    rows = []
    for dn in ("first expansion", "first entry"):
        for o in OUTCOMES:
            cells = []
            for rs in ("relative magnitudes (Mbar)", "smoothness (M)"):
                for tg in ("first_post", "avg_first5"):
                    x = hb[(hb.design == dn) & (hb.outcome == o) & (hb.restriction == rs) & (hb.target == tg)]
                    cells.append("" if x.empty else (f"{x.breakdown_M.iloc[0]:.2f}" if pd.notna(x.breakdown_M.iloc[0]) else f">{x.grid_max.iloc[0]:.1f}"))
            rows.append(f"{'E2' if dn == 'first expansion' else 'E3'} & {OUT_NAME[o]} & " + " & ".join(cells))
    _fragment("robust__honestdid", _tab(r"@{}llrrrr@{}", r"Design & Outcome & \multicolumn{2}{c}{Relative magnitudes, $\bar M$} & \multicolumn{2}{c}{Smoothness, $M$}\\ & & first post & avg. of five & first post & avg. of five", rows),
              f"Smallest value of the sensitivity parameter at which the 95% confidence set includes zero (0 = zero is already in the set under exact parallel trends). Rambachan-Roth, "
              f"R package HonestDiD; output of {stem('4b-honestdid', '*_honestdid_breakdown.csv')}; the R run was not repeated for the D-53 panel because the inputs are unchanged.",
              size=r"\tiny")

    # ---- control-set sensitivity
    cs_ = pd.read_csv(out_glob(tables_dir, "4b-control-set", "*_selection_table.csv")[-1])
    rows = [f"{_esc(v.covariate.replace('_per100k', '').replace('_', ' '))} & {_yn(v.in_final_8)} & {v.coverage_pct:.1f} & {v.pretrend_verdict} & "
            f"{_f(v.added_alone_median_abs_pct_change, 1)} & {_f(v.dropped_from_8_median_abs_pct_change, 1)}" for v in cs_.itertuples()]
    _fragment("robust__control_sets", _tab(r"@{}lcrlrr@{}", r"Covariate & In 8 & \% filled & Pre-trend & Added alone & Dropped from 8", rows),
              "Median absolute percent change of the treatment coefficient across outcomes and treatments when the covariate is added alone, or dropped from the final eight. Source: script4b control sets "
              f"({stem('4b-control-set', '*_selection_table.csv')}). Panel {pdate}.", size=r"\scriptsize")

    # ---- dairy x FSIS
    f4 = pd.read_csv(out_glob(tables_dir, T4A, "*_A4_fsis_interaction.csv")[-1])
    f4 = f4[f4.treatment_id.isin(["T1", "T3", "T4", "E2"])]
    rows = []
    for o in OUTCOMES:
        cells = []
        for t in ("T1", "T3", "T4", "E2"):
            x = f4[(f4.outcome == o) & (f4.treatment_id == t)]
            cells.append("" if x.empty or not bool(x.identified.iloc[0]) else _cs(x.beta.iloc[0], x.se_state.iloc[0]))
        rows.append(f"{OUT_NAME[o]} & " + " & ".join(cells))
    _fragment("robust__a4_fsis", _tab(r"@{}lllll@{}", "Outcome & T1 presence & T3 per 10k & T4 log count & E2 first expansion", rows),
              "Interaction of the large-dairy treatment with the presence of an FSIS slaughterhouse (2017-2023 only); coefficient (state-clustered standard error); blank = not identified. Source: script4a A4 "
              f"({stem(T4A, '*_A4_fsis_interaction.csv')}). Panel {pdate}.", size=r"\scriptsize")

    # ---- event counts by species and size class (script4b section species-counts)
    sc = pd.read_csv(out_glob(tables_dir, "4b-species-counts", "*_species_counts.csv")[-1])
    waves = [2007, 2012, 2017, 2022]
    rows = []
    for (sp, scope), g in sc.groupby(["species", "scope"], sort=False):
        e2, e3 = g[g.design == "E2"].iloc[0], g[g.design == "E3"].iloc[0]
        rows.append(f"{sp.capitalize()} & {scope} & " + " & ".join(
            [f"{int(e2[f'wave_{y}']):,}" for y in waves] + [f"{int(e2.treated_counties):,}", f"{int(e2.comparison_counties):,}"]
            + [f"{int(e3[f'wave_{y}']):,}" for y in waves] + [f"{int(e3.treated_counties):,}", f"{int(e3.comparison_counties):,}"]))
    head = (r"\multicolumn{2}{@{}l}{} & \multicolumn{6}{c}{E2 first expansion} & \multicolumn{6}{c}{E3 first entry}\\" + "\n"
            + r"\cmidrule(lr){3-8}\cmidrule(lr){9-14}" + "\n"
            + "Species & Size class & 2007 & 2012 & 2017 & 2022 & Treated & Comparison & 2007 & 2012 & 2017 & 2022 & Treated & Comparison")
    _fragment("robust__species_counts", _tab(r"@{}ll" + "r" * 12 + "@{}", head, rows),
              "Counties whose first event falls at each census wave. E2: first wave with a rise in the count of operations; E3: first wave with a count above zero after a zero. "
              "Comparison: E2 = no rise at any wave; E3 = zero count at every wave. Size classes: large, small, and all sizes (small + medium + large). "
              f"Rural counties before any outcome sample restriction; not estimated. Source: script4b species-counts ({stem('4b-species-counts', '*_species_counts.csv')}). Panel {pdate}.",
              size=r"\scriptsize")

    # ---- spatial spillover checks (script4g, D-58); each fragment is skipped, and said so, when its result file has not been written
    spillover_fragments(pdate, base_note)


def spillover_fragments(pdate, base_note):
    """Slide fragments for the script4g spillover checks: Moran's I, Conley standard errors, division x year effects, the Durbin model."""
    from script4_treatment import OUTCOMES
    order = list(OUTCOMES)
    heads = [r"Poor MH\\Days", r"Frequent\\distress", r"Deaths of\\despair", r"Aggravated\\assault", r"Assault,\\all", r"Violence\\index", r"NIBRS\\total"]
    names = dict(zip(order, heads))
    treat = [("T1", "T1 presence"), ("T3", "T3 per 10k"), ("T4", "T4 log count"), ("E2", "E2 first expansion")]

    def newest(section, stem):
        hits = [h for h in out_glob(tables_dir, f"4g-{section}", f"*_{stem}.csv")]
        return pd.read_csv(hits[-1]) if hits else None, (os.path.basename(hits[-1])[:10] if hits else None)

    # Moran's I
    m, d = newest("moran", "moran_summary")
    if m is not None:
        rows = []
        for o in order:
            r, e = (m[(m.outcome == o) & (m.series == k)] for k in ("raw outcome", "A2 residual (E2)"))
            if len(r) and len(e):
                r, e = r.iloc[0], e.iloc[0]
                rows.append(f"{OUT_NAME[o]} & {r.mean_I:.3f} & {e.mean_I:.3f} & {e.min_I:.3f} to {e.max_I:.3f} & {int(round(e.share_p_below_05 * e.years))} of {int(e.years)}")
        _fragment("robust__spill_moran", _tab(r"@{}lrrcr@{}", "Outcome & I, raw outcome & I, A2 residual & Range of I (residual) & Years with p < 0.05 (residual)", rows),
                  "Moran's I by year over the rural counties with data that year (mean over years); weights: counties within 100 km, row-standardised; "
                  "permutation p-value (499 draws). Residual from the A2 regression of the outcome on E2, the control set, county and year fixed effects. "
                  f"Zero would mean no spatial pattern. Source: script4g moran ({d}). Panel {pdate}.", size=r"\footnotesize")
    else:
        print("spillover fragment robust__spill_moran skipped: no 4g-moran result file")

    # Conley standard errors
    c, d = newest("conley", "conley_results")
    if c is not None:
        rows = []
        for t, lab in treat:
            g = c[c.treatment_id == t].set_index("outcome")
            cells = [(g.loc[o, "beta"], g.loc[o, "se_state"], g.loc[o, "p_state"]) if o in g.index else (np.nan, np.nan, np.nan) for o in order]
            rows += _est_se_rows(lab, cells)
            rows.append(" & " + " & ".join(f"[{g.loc[o, 'se_conley_250km']:.3f}]" if o in g.index else "" for o in order))
        rel = (c[["se_conley_100km", "se_conley_500km"]].div(c["se_conley_250km"], axis=0) - 1).abs().max().max()
        footer = ["Observations & " + " & ".join(f"{int(c[c.outcome == o].N.iloc[0]):,}" if (c.outcome == o).any() else "" for o in order)]
        _fragment("robust__spill_conley", _reg_tab([names[o] for o in order], rows, footer),
                  "State-clustered standard error in parentheses (stars from its p-value), Conley standard error in brackets (250 km; 100 and 500 km differ by at most "
                  f"{rel * 100:.0f}%). The 15 counties without an outline are left out (coefficients within 1% of the reported A2 cells). "
                  f"* p < 0.10, ** p < 0.05, *** p < 0.01. Source: script4g conley ({d}). Panel {pdate}.", size=r"\tiny")
    else:
        print("spillover fragment robust__spill_conley skipped: no 4g-conley result file")

    # division x year
    v, d = newest("division-fe", "division_year_results")
    if v is not None:
        rows = []
        for t, lab in treat:
            g = v[v.treatment_id == t].set_index("outcome")
            cells = [(g.loc[o, "beta_division_year"], g.loc[o, "se_division_year"], g.loc[o, "p_division_year"]) if o in g.index else (np.nan, np.nan, np.nan) for o in order]
            rows += _est_se_rows(lab, cells)
            rows.append(" & " + " & ".join(f"[{g.loc[o, 'beta_baseline']:.3f}]" if o in g.index else "" for o in order))
        footer = ["Observations & " + " & ".join(f"{int(v[v.outcome == o].N.iloc[0]):,}" if (v.outcome == o).any() else "" for o in order),
                  "County fixed effects & " + " & ".join(["Yes"] * len(order)), "Division x year fixed effects & " + " & ".join(["Yes"] * len(order))]
        _fragment("robust__spill_division", _reg_tab([names[o] for o in order], rows, footer),
                  "County and Census division x year fixed effects (nine divisions); state-clustered standard error in parentheses; baseline coefficient with year effects only, "
                  "same sample, in brackets (15 counties without an outline left out). * p < 0.10, ** p < 0.05, *** p < 0.01. "
                  f"Source: script4g division-fe ({d}). Panel {pdate}.", size=r"\tiny")
    else:
        print("spillover fragment robust__spill_division skipped: no 4g-division-fe result file")

    # Durbin: sample and spatial dependence, then the bootstrap check on the confirmatory family
    u, d = newest("durbin", "durbin_results")
    if u is not None:
        rows = []
        for o in order:
            g = u[(u.outcome == o) & (u.treatment_id == "E2")]
            if len(g):
                g = g.iloc[0]
                rows.append(f"{OUT_NAME[o]} & {_esc(g.w)} & {int(g.counties):,} & {g.years} & {g.rho:.3f} ({g.se_rho:.3f})")
        _fragment("robust__spill_durbin_sample", _tab(r"@{}llrcr@{}", "Outcome & Weights W & Counties & Years & rho (ML s.e.)", rows),
                  "Spatial Durbin panel model, county and year effects, balanced panel of the counties observed in every year of the outcome's window that have a neighbour within 100 km. "
                  "rho: dependence of a county's outcome on its neighbours' outcomes (from the E2 fit; the other treatments' fits give nearly the same value). "
                  f"Source: script4g durbin ({d}). Panel {pdate}.", size=r"\footnotesize")
        b = u[u.boot_draws > 0]
        if len(b):
            rows = []
            for _, g in b.iterrows():
                rows.append(f"{OUT_NAME[g.outcome]} & {g.treatment_id} & " + " & ".join(
                    f"{g[k]:.3f} ({g['se_' + k]:.3f}) [{g['boot_se_' + k]:.3f}]" for k in ("direct", "indirect", "total")))
            _fragment("robust__spill_durbin_boot", _tab(r"@{}llccc@{}", "Outcome & Treatment & Direct & Indirect (spillover) & Total", rows),
                      "Effects of the treatment on the outcome in outcome units: estimate (maximum-likelihood standard error) [state-block wild-bootstrap standard error, "
                      f"{int(b.boot_draws.iloc[0])} draws, Webb weights]. Confirmatory family only (D-42). Source: script4g durbin ({d}). Panel {pdate}.", size=r"\footnotesize")
    else:
        print("spillover fragments robust__spill_durbin_* skipped: no 4g-durbin result file")


def link_slide_assets():
    """Undated symlinks to the newest slide figures and the three cohort maps, so the deck refers to stable file names (LATEX_OUT/slides/assets)."""
    import re
    assets = os.path.join(SLIDE_OUT, "assets")
    os.makedirs(assets, exist_ok=True)
    # (folder, output-name tag, extensions): the slide figures of script4f and three maps of script1d
    sources = [(figs_dir, "4f-slides", (".pdf",)), (maps_dir, "1d", (".png",))]
    wanted_maps = {"estimation_sample_by_design", "expansion_cohort_year", "entry_cohort_year"}
    newest = {}
    for folder, tag, exts in sources:
        for f in out_glob(folder, tag, "*"):
            m = re.match(r"^\d{4}-\d{2}-\d{2}_" + re.escape(tag) + r"_(.+)(\.[a-z]+)$", os.path.basename(f))
            if m and m.group(2) in exts and (tag == "4f-slides" or m.group(1) in wanted_maps):
                newest[m.group(1) + m.group(2)] = f                 # sorted ascending: the newest date wins
    if not INCLUDE_LPDID:                                          # LP-DiD is off (D-57): no LP-DiD figure is linked, and an earlier link is removed
        for name in [n for n in newest if "lp_did" in n]:
            del newest[name]
        for name in os.listdir(assets):
            if "lp_did" in name and os.path.islink(os.path.join(assets, name)):
                os.remove(os.path.join(assets, name))              # a symlink only; the figure file itself stays in the figures folder
    for name, target in newest.items():
        link = os.path.join(assets, name)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
    print(f"linked {len(newest)} slide assets in {assets}")


def slides_section():
    slide_fragments()
    link_slide_assets()


SECTIONS = {"tables": regression_tables, "appendix": full_appendix, "frames": beamer_frames, "slides": slides_section}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(SECTIONS), help="sections to run (default: all, in file order)")
    args = ap.parse_args()
    for name in args.only or SECTIONS:
        print(f"=== {name}")
        SECTIONS[name]()


if __name__ == "__main__":
    main()
