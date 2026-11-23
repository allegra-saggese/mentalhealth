#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4o-headline-deck.py -- the exhibits of slides/deck_headline.tex: table fragments and one new figure (no estimation).

Reads the newest result files of script4i (headline evidence), 4j (2016 split), 4k/4l/4m/4n (FSIS designs), 4c (family), 4b (leads) and 4g (Conley);
writes LaTeX fragments `headline__*.tex` to the slide folder (packages.latex_dir/slides, linked as slides/generated) and the FSIS event-study figure
`headline_fsis_event.pdf` (figure folder, tag 4o-headline; undated link in slides/generated/assets). Fragments are built with the writer helpers of
script4e-tables.py (loaded as a module). Every number on a table is a cell of a result file; frame wording is in deck_headline.tex.
Run:    PYTHONHASHSEED=0 python3 script4o-headline-deck.py
"""
import warnings; warnings.filterwarnings("ignore")
import importlib.util
import os

import numpy as np
import pandas as pd

import figstyle
from designs import OUTCOME_WINDOWS
from packages import figs_dir, out_glob, tables_dir
from script4_treatment import OUTCOMES

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s4e", os.path.join(HERE, "script4e-tables.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
OUT = "Poor MH Days"
SRC = {}


def newest(tag, name):
    f = out_glob(tables_dir, tag, f"*_{name}")
    if not f:
        raise FileNotFoundError(f"no {tag} {name}")
    SRC[tag + " " + name] = os.path.basename(f[-1])
    return pd.read_csv(f[-1])


def note(tags):
    return "Sources: " + "; ".join(sorted({v for k, v in SRC.items() if any(k.startswith(t) for t in tags)})) + "."


def pstr(p):
    return "" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def cell(b, se=None, d=3):
    return "" if pd.isna(b) else (f"{b:.{d}f} ({se:.{d}f})" if se is not None and not pd.isna(se) else f"{b:.{d}f}")


# ------------------------------------------------------------------------------------------------- fragments
def frag_data(ev):
    """Outcomes, windows and sample sizes of the reported A2 regressions (E2 treatment); event counts of the dairy and plant designs."""
    g = newest("4a-designs", "A1_A2_treatment_grid.csv")
    g = g[(g.registry == "A2") & (g.treatment_id == "E2")].set_index("outcome")
    rows = []
    for okey in OUTCOMES:
        lo, hi = OUTCOME_WINDOWS[okey]
        rows.append(f"{S._esc(okey)} & {lo}--{hi} & {int(g.loc[okey, 'N']):,} & {int(g.loc[okey, 'n_counties']):,}")
    t1 = S._tab("@{}lccc@{}", "Outcome & Years & County-years & Counties", rows)
    S._fragment("headline__data_outcomes", t1, "Reported A2 regressions (county and year effects, control set of 8), first expansion of large dairies. " + note(["4a-designs"]), size=r"\small")
    sp = newest("4j-split", "split_results.csv")
    f = sp[(sp.window == "full 2010-2023") & (sp.estimator == "Callaway-Sant'Anna")].set_index("design")
    pk = newest("4n-fsis-ctrl", "overall.csv")
    pk = pk[(pk.outcome == OUT) & (pk.spec == "without ClosePost") & (pk.estimator == "TWFE")].set_index("version")
    rows2 = [f"First expansion (E2), large dairies & {int(f.loc['E2', 'treated_counties'])} & {int(f.loc['E2', 'counties']):,} & {f.loc['E2', 'cohorts'].replace(',', ', ')}",
             f"First entry (E3), large dairies, clean controls & {int(f.loc['E3', 'treated_counties'])} & {int(f.loc['E3', 'counties']):,} & {f.loc['E3', 'cohorts'].replace(',', ', ')}",
             f"Large FSIS plant, county entry & {int(pk.loc['county entry', 'treated_counties'])} & {int(pk.loc['county entry', 'counties']):,} & 2018--2023",
             f"Large FSIS plant, zone entry (all counties) & {int(pk.loc['zone A all counties in zone', 'treated_counties'])} & {int(pk.loc['zone A all counties in zone', 'counties']):,} & 2018--2023"]
    t2 = S._tab("@{}lrrl@{}", "Design (Poor MH Days sample) & Treated counties & Counties & First treated (cohorts)", rows2)
    S._fragment("headline__data_events", t2, "Dairy cohorts are census waves; plant cohorts are years. Counts are those of the Callaway--Sant'Anna usable sample. " + note(["4j-split", "4n-fsis-ctrl"]), size=r"\small")


def frag_scorecard(ev):
    def v(check, design=None, col="value"):
        q = ev[ev.check == check] if design is None else ev[(ev.check == check) & (ev.design == design)]
        return q.iloc[0][col]
    tw, cs = ev[(ev.check == "TWFE, A2 reported spec (full sample)")].iloc[0], v("Callaway-Sant'Anna", "E2")
    csrow = ev[(ev.check == "Callaway-Sant'Anna") & (ev.design == "E2")].iloc[0]
    sa, dc = v("Sun-Abraham", "E2"), v("de Chaisemartin-D'Haultfoeuille", "E2")
    pre = ev[ev.check.str.startswith("Callaway-Sant'Anna pre-periods significant at 5%, covariate-adjusted")].iloc[0].quantity
    cl = ev[ev.check == "control set: 8 pre-committed -> all 14"].iloc[0]
    dv = ev[ev.check == "Census division x year fixed effects"].iloc[0]
    cn = ev[ev.check == "Conley spatial standard error, 500 km"].iloc[0]
    dt = ev[(ev.check == "donut 100 km, TWFE") & (ev.design == "E2")].iloc[0]
    dc_ = ev[(ev.check == "donut 100 km, Callaway-Sant'Anna") & (ev.design == "E2")].iloc[0]
    holm = ev[ev.check.str.startswith("Holm")].iloc[0]
    rw = ev[ev.check.str.startswith("Romano")].iloc[0]
    sp = {k: ev[ev.check == f"{k}, large operations, TWFE"].iloc[0] for k in ("beef", "hogs", "chickens")}
    da = ev[ev.check == "dairy, all sizes operations, TWFE"].iloc[0]
    j2 = ev[(ev.check.str.startswith("TWFE event study")) & (ev.design == "E2")].iloc[0].p
    j3 = ev[(ev.check.str.startswith("TWFE event study")) & (ev.design == "E3")].iloc[0].p
    sps = newest("4j-split", "split_results.csv")
    g = lambda w, e: sps[(sps.window == w) & (sps.design == "E2") & (sps.estimator == e)].iloc[0]
    rows = [
        f"1. Estimator & TWFE {cell(tw.value, tw.se)}; Callaway--Sant'Anna {cell(csrow.value, csrow.se)}; Sun--Abraham {sa:.3f}; dCDH {dc:.3f} & Significant in TWFE only",
        f"2. Pre-trends & TWFE leads joint $p$ = {j2:.2f} (E2), {j3:.2f} (E3); Callaway--Sant'Anna pre-period estimates significant: {pre} & Mixed",
        f"3. Adjustments & All 14 controls: {cl.value:+.1f}\\% ({cl.reading.split(': ')[1].split(';')[0].replace('%', chr(92) + '%')} with the 8); division$\\times$year {cell(dv.value, dv.se)}, $p$ = {dv.p:.3f}; Conley 500 km $p$ = {cn.p:.3f}; donut 100 km TWFE {dt.value:.3f}, CS {dc_.value:.3f} & TWFE stable; CS never significant",
        f"4. Many tests & Holm over the 8 confirmatory tests $p$ = {holm.p:.2f}; Romano--Wolf within outcome {rw.p:.2f} & Does not survive",
        f"5. Specificity & Dairy all sizes {da.value:.3f}; beef {sp['beef'].value:.3f}, hogs {sp['hogs'].value:.3f}, chickens {sp['chickens'].value:.3f} (TWFE) & Holds in TWFE",
        f"6. 2016 data break & CS 2010--15 {cell(g('pre-break 2010-2015', 'Callaway-Sant' + chr(39) + 'Anna').estimate, g('pre-break 2010-2015', 'Callaway-Sant' + chr(39) + 'Anna').se)}; "
        f"2016--23 {cell(g('post-break 2016-2023', 'Callaway-Sant' + chr(39) + 'Anna').estimate, g('post-break 2016-2023', 'Callaway-Sant' + chr(39) + 'Anna').se)} & Same sign; neither significant",
    ]
    t = ("\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{0.17\\textwidth}>{\\raggedright\\arraybackslash}p{0.60\\textwidth}>{\\raggedright\\arraybackslash}p{0.17\\textwidth}@{}}\n\\toprule\nTest & Poor MH Days, large dairies, E2: days per month (SE) & Reading\\\\\n\\midrule\n"
         + "\\\\[1pt]\n".join(rows) + "\\\\\n\\bottomrule\n\\end{tabular}")
    S._fragment("headline__scorecard", t, "Reading is the author's; Holm and Romano--Wolf over the D-42 family. " + note(["4i-headline"]), size=r"\scriptsize")


def frag_multiple():
    w = newest("4c-wild", "confirmatory_family.csv")
    w = w[w.outcome == OUT].set_index("treatment_id")
    lead = newest("4b-leads", "leads_joint_tests.csv")
    lead = lead[lead.outcome == OUT]
    lp = {}
    for _, r in lead.iterrows():
        lp[r.treatment] = r.joint_p
    names = {"T1": "T1 presence", "T3": "T3 per 10,000 residents", "T4": "T4 log(1 + count)", "E2": "E2 first expansion"}
    rows = []
    for k in ("T1", "T3", "T4", "E2"):
        r = w.loc[k]
        rows.append(f"{names[k]} & {cell(r.beta, r.se_state)} & {pstr(r.p_state)} & {pstr(r.p_wild)} & {pstr(r.p_romano_wolf_within_outcome)} & {pstr(r.p_holm_over_8)} & {pstr(lp.get(k))}")
    t = S._tab("@{}lcccccc@{}", "Treatment (large dairies) & Effect (SE) & $p$ state & $p$ wild & $p$ Romano--Wolf & $p$ Holm (8 tests) & Leads $p$", rows)
    S._fragment("headline__multiple", t, "Poor MH Days. Wild: cluster bootstrap-t, Webb weights, 9,999 draws. Holm over the 8 confirmatory tests (Poor MH Days and Frequent Mental Distress; D-42). "
                "Leads: joint Wald test of the lead coefficients (4 leads for T1, T3, T4; 7 for E2). " + note(["4c-wild", "4b-leads"]), size=r"\small")


def frag_robust(ev):
    def r(check, design=None):
        q = ev[(ev.check == check)] if design is None else ev[(ev.check == check) & (ev.design == design)]
        return q.iloc[0]
    rows = []
    for lab, check in (("Division $\\times$ year effects", "Census division x year fixed effects"), ("Conley errors, 100 km", "Conley spatial standard error, 100 km"),
                       ("Conley errors, 500 km", "Conley spatial standard error, 500 km")):
        x = r(check)
        rows.append(f"{lab} & TWFE & {cell(x.value, x.se)} & {pstr(x.p)}")
    for km in (50, 100):
        for est, lab in (("TWFE", "TWFE"), ("Callaway-Sant'Anna", "CS")):
            x = r(f"donut {km} km, {est}", "E2")
            rows.append(f"Drop never-treated within {km} km & {lab} & {cell(x.value, x.se)} & {pstr(x.p)}")
    t = S._tab("@{}llcc@{}", "Check (E2) & Estimator & Effect (SE), days & $p$", rows)
    S._fragment("headline__robust_adj", t, "Poor MH Days, first expansion. Donut: never-treated counties within the radius of a treated county are dropped; not-yet-treated counties stay. CS = Callaway--Sant'Anna. " + note(["4i-headline"]), size=r"\small")
    rows = []
    for sp_ in ("dairy, large", "dairy, all sizes", "beef, large", "hogs, large", "chickens, large"):
        nm = sp_.split(", ")
        for est, lab in (("TWFE", "TWFE"), ("Callaway-Sant'Anna", "CS")):
            x = r(f"{nm[0]}, {nm[1]} operations, {est}") if nm[1] == "large" else r(f"{nm[0]}, all sizes operations, {est}")
            rows.append(f"{nm[0].capitalize()}, {nm[1]} operations & {lab} & {cell(x.value, x.se)} & {pstr(x.p)}")
    t = S._tab("@{}llcc@{}", "Animal and size (E2) & Estimator & Effect (SE), days & $p$", rows)
    S._fragment("headline__robust_spec", t, "Poor MH Days, first expansion of the animal's operations of that size. CS = Callaway--Sant'Anna. " + note(["4i-headline"]), size=r"\small")


def frag_split():
    s = newest("4j-split", "split_results.csv")
    rows = []
    for w in ("full 2010-2023", "pre-break 2010-2015", "post-break 2016-2023"):
        for d_ in ("E2", "E3"):
            q = s[(s.window == w) & (s.design == d_)]
            tw = q[q.estimator == "TWFE (design sample)"].iloc[0]
            cs = q[q.estimator == "Callaway-Sant'Anna"].iloc[0]
            rows.append(f"{w} & {d_} & {int(cs.treated_counties)} & {cs.cohorts.replace(',', ', ')} & {cell(tw.estimate, tw.se)} & {cell(cs.estimate, cs.se)} & {pstr(cs.p)}")
    t = S._tab("@{}llrlccc@{}", "Window & Design & Treated & Cohorts & TWFE (SE) & CS (SE) & CS $p$", rows)
    S._fragment("headline__split", t, "Poor MH Days, large dairies; control set of 8; state-clustered. CS = Callaway--Sant'Anna covariate-adjusted, 300 draws. Cohorts after a window's last year stay as comparison counties (D-62). " + note(["4j-split"]), size=r"\small")


def frag_fsis():
    k = newest("4k-fsis-large", "overall.csv")
    l = newest("4l-fsis-cz", "overall.csv")
    m = newest("4m-fsis-clean", "overall.csv")
    n = newest("4n-fsis-ctrl", "overall.csv")
    mv = newest("4m-fsis-clean", "manyvsnone.csv")
    rows = []

    def add(label, q, est, pre=""):
        x = q[q.estimator == est].iloc[0]
        rows.append(f"{label} & {est.replace('Callaway-Sant' + chr(39) + 'Anna', 'CS')} & {int(x.treated_counties)} & {cell(x.estimate, x.se)} & {pstr(x.p)} & {pre}")
    b = lambda df, **kw: df[(df.outcome == OUT) & np.logical_and.reduce([df[c] == v for c, v in kw.items()])]
    for est in ("Callaway-Sant'Anna", "TWFE"):
        q = b(k, design="entry", sensitivity="baseline")
        x = q[q.estimator == est].iloc[0]
        pre = f"{int(x.pre_sig_5pct)} of {int(x.pre_periods)} pre-periods sig." if est.startswith("Calla") else ""
        add("County entry", q, est, pre)
    q = b(k, design="exit", sensitivity="baseline")
    rows.append(f"County exit & CS & {int(q[q.estimator == 'TWFE'].iloc[0].treated_counties)} & not estimable & & ")
    add("", q, "TWFE")
    for ver, lab in (("A all counties in zone", "Zone entry, all"), ("B non-host counties only", "Zone entry, non-host")):
        for est in ("Callaway-Sant'Anna", "TWFE"):
            q = b(l, version=ver, sensitivity="baseline")
            x = q[q.estimator == est].iloc[0]
            add(lab, q, est, f"{int(x.pre_sig_5pct)} of {int(x.pre_periods)} pre-periods sig." if est.startswith("Calla") else "")
    q = b(m, version="county entry", sensitivity="baseline")
    x = q[q.estimator.str.startswith("Calla")].iloc[0]
    add("Closures dropped", q, "Callaway-Sant'Anna", f"{int(x.pre_sig_5pct)} of {int(x.pre_periods)} pre-periods sig.")
    for spec in ("without ClosePost", "with ClosePost"):
        q = b(n, version="county entry", spec=spec)
        add("Closure control: " + ("yes" if spec.startswith("with ") else "no"), q, "Imputation (did2s)")
    x = mv[mv.outcome == OUT].iloc[0]
    rows.append(f"Many vs none & TWFE & {int(x.many_counties)}/{int(x.none_counties):,} & {cell(x.estimate, x.se)} & {pstr(x.p)} & leads $p$ {'< 0.001' if x.pre_joint_p < 0.001 else '= ' + pstr(x.pre_joint_p)}; not read")
    t = S._tab("@{}llrccl@{}", "Design & Estimator & Treated & Effect (SE), days & $p$ & Pre-trend", rows)
    S._fragment("headline__fsis", t, "Poor MH Days, 2017--2023, large FSIS plants (size code 4 or 5); zone entry = commuting-zone exposure (D-64); closure rows D-65, D-66. Negative = fewer poor mental health days. "
                + note(["4k-fsis-large", "4l-fsis-cz", "4m-fsis-clean", "4n-fsis-ctrl"]), size=r"\small")


def frag_gp():
    """Residual spatial dependence from the NN-RCM fits (script4h-gp.py): decay length and spatial share of the A2 residual variance, national, smoothness 0.5."""
    g = newest("4h-gp", "range_summary.csv")
    g = g[(g.series.str.startswith("A2")) & (g.region == "US") & (g.kappa == 0.5)].set_index("outcome")
    rows = []
    for okey in OUTCOMES:
        if okey in g.index:
            r = g.loc[okey]
            rows.append(f"{S._esc(okey)} & {int(r.years)} & {r.range_km_median:.0f} & {100 / (1 + r.nugget_median):.0f}\\%")
    t = S._tab("@{}lccc@{}", "A2 residual of & Years fitted & Decay length, km (median) & Spatial share of variance", rows)
    S._fragment("headline__gp", t, "Decay length: distance at which the fitted correlation has fallen to 0.37 (exponential, smoothness 0.5); 50 km is the floor of the simulated calibration (50 km or less). "
                "Spatial share = 1/(1 + nugget ratio). Fits that did not converge are excluded. " + note(["4h-gp"]), size=r"\small")


# ------------------------------------------------------------------------------------------------- figure
def fig_fsis_event():
    k = newest("4k-fsis-large", "event_time.csv")
    l = newest("4l-fsis-cz", "event_time.csv")
    n = newest("4n-fsis-ctrl", "event_time.csv")
    panels = [("County entry (CS)", k[(k.design == "entry") & (k.sensitivity == "baseline") & (k.outcome == OUT)], "att", "se_state"),
              ("County entry, closure\ncontrolled (imputation)", n[(n.version == "county entry") & (n.spec == "with ClosePost") & (n.outcome == OUT)], "est", "se"),
              ("Zone entry, all counties (CS)", l[(l.version == "A all counties in zone") & (l.sensitivity == "baseline") & (l.outcome == OUT)], "att", "se_state")]
    fig, axes = figstyle.new_slide(1, 3, height_mm=figstyle.SLIDE_HEIGHT_MM - 18)
    for ax, (ttl, d, b, s) in zip(axes.ravel(), panels):
        d = d[d.event_time.between(-6, 5)].sort_values("event_time")
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(-0.5, color="black", lw=0.5, ls="--")
        ax.errorbar(d.event_time, d[b], yerr=1.96 * d[s], fmt="o", ms=2.6, lw=0.7, capsize=1.2, color=figstyle.PALETTE["blue"])
        ax.set_title(ttl, fontsize=6.5, pad=2)
        ax.set_xlabel("Years since entry", fontsize=6)
    axes[0, 0].set_ylabel("Poor mental health days\n(effect, 95% CI)")
    stem = "headline_fsis_event"
    pdf, _png = figstyle.save_figure(fig, figs_dir, "4o-headline_" + stem, png_dpi=300)
    assets = os.path.join(S.SLIDE_OUT, "assets")
    os.makedirs(assets, exist_ok=True)
    link = os.path.join(assets, stem + ".pdf")
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(pdf, link)
    print("wrote", pdf, "and linked", link)


def main():
    ev = newest("4i-headline", "evidence.csv")
    frag_data(ev)
    frag_scorecard(ev)
    frag_multiple()
    frag_robust(ev)
    frag_split()
    frag_fsis()
    frag_gp()
    fig_fsis_event()


if __name__ == "__main__":
    main()
