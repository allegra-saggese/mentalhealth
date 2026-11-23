#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4i-headline-evidence.py -- one table of the evidence for and against the headline design (plan 2026-11-23, step 1).

Headline design: large dairies -> Poor MH Days, first expansion (E2) and first entry (E3). Nothing is estimated here: every row is read from the
newest result file of the script that produced it, and carries that file's name (so its date and run state can be checked against
code-audit/plans/2026-10-17_MODEL-RUN-STATE.md). Rows that weaken the headline sit in the same table with the same prominence.

Reads:  Data/output/tables/ results of script4a (designs, cohorts, species), 4b (leads, honestdid, control-set), 4c (wild), 4d (estimators), 4g (conley, donut, division-fe)
Writes: Data/output/tables/<date>_4i-headline_evidence.csv   (columns: block, check, design, quantity, value, se, p, reading, source)
Run:    python3 script4i-headline-evidence.py
LP-DiD rows are left out (switched off, D-57). Counts of significant pre-period estimates use p < 0.05 on the state-clustered p-value.
"""
import os
from datetime import date

import numpy as np
import pandas as pd

from packages import out_file, out_glob, tables_dir

TODAY = date.today().strftime("%Y-%m-%d")
OUT = "Poor MH Days"
DESIGNS = {"E2": "first expansion", "E3": "first entry (clean controls)"}
ROWS = []


def newest(tag, pattern):
    """Newest file of a script's tag, as (DataFrame, file name). Stops if absent: a missing input must not give a quiet gap in the table."""
    f = out_glob(tables_dir, tag, pattern)
    if not f:
        raise FileNotFoundError(f"no {tag} {pattern}")
    return pd.read_csv(f[-1]), os.path.basename(f[-1])


def add(block, check, design, quantity, value=np.nan, se=np.nan, p=np.nan, reading="", source=""):
    ROWS.append(dict(block=block, check=check, design=design, quantity=quantity, value=value, se=se, p=p, reading=reading, source=source))


def estimates():
    h, hs = newest("4a-designs", "*_HEADLINE_twfe_vs_cs.csv")
    r = h[h.outcome == OUT].iloc[0]
    d, ds = newest("4d-estimators", "*_overall_estimates.csv")
    d = d[(d.outcome == OUT) & ~d.estimator.str.contains("LP-DiD")]
    for k, name in DESIGNS.items():
        x = d[d.design == name]
        for _, e in x.iterrows():
            est = e.estimator.replace(" (benchmark)", "").replace(" (static, reference)", "")
            rd = {"Callaway-Sant'Anna": "covariate-adjusted; the reported robust estimator", "TWFE": "static two-way fixed effects; forbidden comparisons possible"}.get(est, "")
            add("1 estimate", est, k, "overall effect (days)", e.est, e.se, e.p, rd, ds)
    add("1 estimate", "TWFE, A2 reported spec (full sample)", "E2", "overall effect (days)", r.TWFE_beta, r.TWFE_se, np.nan,
        f"{int(r.n_switchers_TWFE)} switchers; CS/TWFE = {r.ratio_CS_over_TWFE:.2f}; the 4d TWFE row above uses the CS-usable sample (0.096)", hs)


def pretrends():
    j, js = newest("4b-leads", "*_leads_joint_tests.csv")
    for k in DESIGNS:
        x = j[(j.outcome == OUT) & (j.treatment == k) & (j.design == "event study")].iloc[0]
        add("2 pre-trends", "TWFE event study, joint test of the 7 leads", k, "joint p", p=x.joint_p, reading="flat" if x.joint_p > 0.1 else "FAILS", source=js)
    c, cs_ = newest("4a-cohorts", "*_event_study_leads_joint.csv")
    for _, x in c[c.outcome == OUT].iterrows():
        k = "E2" if x.design == "first expansion" else "E3"
        add("2 pre-trends", f"TWFE leads, adjustment: {x.spec}", k, "joint p", p=x.joint_p, source=cs_)
    a6, a6s = newest("4a-designs", "*_A6_cs_aggregated.csv")
    for arm, g in a6[(a6.outcome == OUT) & (a6.agg_level == "event_time") & (a6.event_time < 0)].groupby("arm"):
        n, s = len(g), int((g.p_state < 0.05).sum())
        add("2 pre-trends", f"Callaway-Sant'Anna pre-periods significant at 5%, {arm} arm", "E2", f"{s} of {n}", float(s), p=np.nan,
            reading="the covariate-adjusted arm is the reported one" if arm == "covariate-adjusted" else "", source=a6s)
    e, es = newest("4a-cohorts", "*_cs_aggregated.csv")
    for (dn, sp), g in e[(e.outcome == OUT) & (e.agg_level == "event_time") & (e.event_time < 0)].groupby(["design", "spec"]):
        add("2 pre-trends", f"Callaway-Sant'Anna pre-periods significant at 5%, spec {sp}", "E2" if dn == "first expansion" else "E3",
            f"{int((g.p_state < 0.05).sum())} of {len(g)}", float((g.p_state < 0.05).sum()), source=es)


def honestdid():
    b, bs = newest("4b-honestdid", "*_honestdid_breakdown.csv")
    b = b[b.outcome == OUT]
    for _, x in b.iterrows():
        k = "E2" if x.design == "first expansion" else "E3"
        add("3 sensitivity", f"HonestDiD breakdown, {x.target}, {x.restriction}", k, "breakdown M", x.breakdown_M,
            reading="zero already inside the set at M = 0" if x.zero_in_set_at_M0 else "result excludes zero at M = 0", source=bs)


def inference():
    w, ws = newest("4c-wild", "*_confirmatory_family.csv")
    x = w[(w.outcome == OUT) & (w.treatment_id == "E2")].iloc[0]
    add("4 inference", "state-clustered p", "E2", "p", x.beta, x.se_state, x.p_state, source=ws)
    add("4 inference", "wild cluster bootstrap-t p (Webb, 9,999 draws)", "E2", "p", p=x.p_wild, source=ws)
    add("4 inference", "Romano-Wolf p within outcome (4 treatments)", "E2", "p", p=x.p_romano_wolf_within_outcome, source=ws)
    add("4 inference", "Holm p over the 8 confirmatory tests", "E2", "p", p=x.p_holm_over_8,
        reading="does not survive" if x.p_holm_over_8 > 0.05 else "survives", source=ws)


def robustness():
    c, cs_ = newest("4b-control-set", "*_path_summary.csv")
    x = c[(c.outcome == OUT) & (c.treatment == "E2")].iloc[0]
    add("5 robustness", "control set: 8 pre-committed -> all 14", "E2", "change in estimate, %", x.change_0_to_14_pct,
        reading=f"0 -> 8: {x.change_0_to_8_pct:.1f}%; sign flips {int(x.sign_flips_along_path)}", source=cs_)
    v, vs = newest("4g-division-fe", "*_division_year_results.csv")
    x = v[(v.outcome == OUT) & (v.treatment_id == "E2")].iloc[0]
    add("5 robustness", "Census division x year fixed effects", "E2", "effect (days)", x.beta_division_year, x.se_division_year, x.p_division_year, source=vs)
    q, qs = newest("4g-conley", "*_conley_results.csv")
    x = q[(q.outcome == OUT) & (q.treatment_id == "E2")].iloc[0]
    for km in (100, 250, 500):
        add("5 robustness", f"Conley spatial standard error, {km} km", "E2", "effect (days)", x.beta, x[f"se_conley_{km}km"], x[f"p_conley_{km}km"], source=qs)
    dn, dns = newest("4g-donut", "*_donut_results.csv")
    for _, x in dn[(dn.outcome == OUT) & dn.design.isin(DESIGNS) & dn.radius_km.isin([50, 100])].iterrows():
        add("5 robustness", f"donut {int(x.radius_km)} km, {x.estimator}", x.design, "effect (days)", x.estimate, x.se, x.p,
            reading=f"{int(x.never_treated_dropped)} never-treated counties dropped", source=dns)


def specificity():
    s, ss = newest("4a-species", "*_species_results.csv")
    s = s[(s.outcome == OUT) & s.estimator.isin(["E2", "CS"])]
    for _, x in s.iterrows():
        add("6 specificity", f"{x.species}, {x.scope} operations, {"TWFE" if x.estimator == "E2" else "Callaway-Sant'Anna"}", "E2", "effect (days)",
            x.estimate, x.se, x.p, "the headline" if (x.species == "dairy" and x.scope == "large") else "", ss)


def main():
    for f in (estimates, pretrends, honestdid, inference, robustness, specificity):
        f()
    t = pd.DataFrame(ROWS)
    assert t.groupby("block").size().min() > 0
    t.to_csv(out_file(tables_dir, "4i-headline", f"{TODAY}_evidence.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 200)
    print(t.drop(columns="source").round(3).to_string(index=False))
    print(f"\n{len(t)} rows; sources: {sorted(set(t.source))}")


if __name__ == "__main__":
    main()
