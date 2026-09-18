#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4d-power-balance.py  --  sample, power, and coverage diagnostics for Part A.

Produces ONE Excel workbook supporting power calculations and the data-appendix
tables, reconstructing the exact estimating sample for every (treatment x outcome)
cell that script4a-twfe-eventstudy.py estimates.

Sheets
------
1. README              what each sheet is, and how to read the power columns
2. power_by_analysis   N, treated/control, switchers, SE, and MDE per analysis
3. panel_balance       balance of the estimating panel per analysis
4. control_coverage    CONTROL variables: used (CORE_9) vs dropped, with coverage
5. treatment_summary   coverage and variation for all 14 treatment definitions
6. outcome_coverage    per-outcome year span and completeness

Power columns (sheet 2)
-----------------------
MDE is the minimum detectable effect at 80% power, two-sided alpha = 0.05:
    MDE = (z_{0.975} + z_{0.80}) * SE = (1.96 + 0.8416) * SE = 2.8016 * SE
It is computed from the REALISED state-clustered SE of the estimate actually run,
so it is an ex-post MDE -- it answers "what is the smallest true effect this design
could reliably have detected", which is the relevant question for interpreting the
nulls in script4a. It is NOT an ex-ante power calculation for a new design.

mde_in_sd divides that by the WITHIN-COUNTY sd of the outcome, because a
county+year FE model identifies off within-county variation; scaling by the raw
cross-sectional sd would overstate the design's sensitivity.

CAVEAT on "treated" / "control": for the binary treatments these are exact. For
count and share treatments the split is at > 0, which is a reasonable reading but
is not what the estimator uses -- those specs use the full continuous variation.
The treatment_type column says which is which. Do not build a power calculation
on the treated/control split for a continuous treatment.
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from script4_treatment import (
    load_panel, OUTCOMES, TREATMENTS, CONTROL_CORE9, CONTROL_FULL25,
    CONTROL_PRETREAT, CONTROL_COMPREHENSIVE,
    HEADLINE_TREATMENT, HEADLINE_OUTCOME, HEADLINE_ESTIMATOR,
    CONTROL_POSTTREATMENT, tables_dir, os, date,
)

OUT_TABLES = os.path.join(tables_dir, "script4a")
os.makedirs(OUT_TABLES, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
XLSX = os.path.join(OUT_TABLES, f"{TODAY}_power_and_coverage.xlsx")

Z_ALPHA, Z_POWER = 1.959964, 0.841621          # two-sided 5%, 80% power
MDE_MULT = Z_ALPHA + Z_POWER

# Binary treatments: treated/control counts are exact. Others are continuous and
# the >0 split is descriptive only.
TREATMENT_TYPE = {
    "E1": "binary", "E2": "binary", "E3": "binary",
    "I1": "count", "I2": "count", "I3": "count", "I4": "rate",
    "C1": "share", "C2": "share", "C3": "log-count", "C4": "log-count", "C5": "index",
}

df = load_panel()
print(f"panel: {df.shape[0]:,} rows, {df['fips'].nunique():,} counties")

# Realised SEs from the estimation run, joined on so MDE uses the actual SE.
grid_path = os.path.join(OUT_TABLES, f"{TODAY}_A1_A2_treatment_grid.csv")
grid = pd.read_csv(grid_path) if os.path.exists(grid_path) else pd.DataFrame()
if grid.empty:
    print("WARNING: treatment grid not found -- run script4a first. SE/MDE will be blank.")


def balance(sub):
    """Balance of the estimating panel actually used by one analysis."""
    n_per = sub.groupby("fips").size()
    yrs = sorted(sub["year"].unique())
    span = len(yrs)
    # a county is 'contiguous' if its observed years have no internal gaps
    gaps = sub.groupby("fips")["year"].apply(
        lambda s: int(s.max() - s.min() + 1) != len(s)).sum()
    return {
        "year_min": int(min(yrs)), "year_max": int(max(yrs)), "n_years_spanned": span,
        "obs_per_county_min": int(n_per.min()),
        "obs_per_county_med": float(n_per.median()),
        "obs_per_county_max": int(n_per.max()),
        "pct_counties_full_span": float((n_per == span).mean()),
        "balance_ratio": float(len(sub) / (sub["fips"].nunique() * span)),
        "n_counties_with_gaps": int(gaps),
        "is_balanced": bool(n_per.nunique() == 1),
    }


# =============================================================================
# Sheet 2 + 3: per-analysis power and balance
# =============================================================================
power_rows, bal_rows = [], []
for tid, (tcol, cond, tlabel) in TREATMENTS.items():
    for okey, ocol in OUTCOMES.items():
        need = ["fips", "year", "state_fips", ocol, tcol] + list(cond) + CONTROL_PRETREAT
        sub = df[list(dict.fromkeys(need))].dropna()
        if len(sub) < 200:
            continue

        t = sub[tcol]
        treated_obs = int((t > 0).sum())
        treated_cty = int(sub.loc[t > 0, "fips"].nunique())
        nun = sub.groupby("fips")[tcol].nunique()
        switchers = int((nun > 1).sum())
        e = sub[sub.fips.isin(nun[nun > 1].index)].sort_values(["fips", "year"]).copy()
        e["_d"] = e.groupby("fips")[tcol].diff()

        # within-county sd of the outcome: the scale the FE estimator works on
        wsd = float((sub[ocol] - sub.groupby("fips")[ocol].transform("mean")).std())

        se = np.nan
        if len(grid):
            g = grid[(grid.registry == "A2") & (grid.treatment_id == tid) & (grid.outcome == okey)]
            if len(g):
                se = float(g.iloc[0]["se_state"])

        power_rows.append({
            "treatment_id": tid, "treatment": tlabel, "treatment_type": TREATMENT_TYPE.get(tid, ""),
            "conditioning": "+".join(cond), "outcome": okey,
            "N_obs": len(sub), "n_counties": sub.fips.nunique(), "n_states": sub.state_fips.nunique(),
            "n_treated_obs": treated_obs, "n_control_obs": len(sub) - treated_obs,
            "pct_treated_obs": treated_obs / len(sub),
            "n_treated_counties": treated_cty,
            "n_control_counties": sub.fips.nunique() - treated_cty,
            "n_switcher_counties": switchers,
            "pct_switchers": switchers / sub.fips.nunique(),
            "n_transitions": int((e["_d"].abs() > 0).sum()) if switchers else 0,
            "outcome_mean": float(sub[ocol].mean()),
            "outcome_sd": float(sub[ocol].std()),
            "outcome_within_sd": wsd,
            "se_state_realised": se,
            "MDE_80pct": MDE_MULT * se if pd.notna(se) else np.nan,
            "MDE_in_within_sd": (MDE_MULT * se / wsd) if (pd.notna(se) and wsd > 0) else np.nan,
        })
        bal_rows.append({"treatment_id": tid, "outcome": okey, "N_obs": len(sub),
                         "n_counties": sub.fips.nunique(), **balance(sub)})
    print(f"  {tid} done")

power = pd.DataFrame(power_rows)
bal   = pd.DataFrame(bal_rows)


# =============================================================================
# Sheet 4: control coverage -- used vs dropped
# =============================================================================
# Completeness is measured on the REFERENCE ESTIMATING SAMPLE (the rows the
# analyses actually ran on), not on the full panel, because that is the population
# a reviewer cares about. The full-panel figure is given alongside for contrast.
REF_OUT = "poor_mental_health_days"
ref_need = ["fips", "year", REF_OUT, "tr_e1_lg_bin"] + CONTROL_CORE9
ref = df[ref_need].dropna()
ref_keys = set(zip(ref.fips, ref.year))
panel_rows = df[df[REF_OUT].notna()]

ctrl_rows = []
for c in sorted(set(CONTROL_FULL25) | set(CONTROL_CORE9)):
    if c not in df.columns:
        ctrl_rows.append({"variable": c, "status": "MISSING FROM PANEL"})
        continue
    s = df[c]
    yrs = df.loc[s.notna(), "year"]
    on_ref = df.set_index(["fips", "year"]).loc[
        df.set_index(["fips", "year"]).index.isin(ref_keys), c]
    used = c in CONTROL_CORE9
    ctrl_rows.append({
        "variable": c,
        "status": "USED (CORE_9)" if used else "DROPPED",
        # Reason labels are derived from the TWO rules actually applied, and
        # anything matching neither is flagged rather than given a post-hoc
        # justification. CORE_9 was inherited from script4-model-test.py's spec
        # curve, not re-derived, so some exclusions have no stated basis --
        # surfacing that is the point of the UNJUSTIFIED label.
        "reason_dropped": "" if used else (
            "post-treatment (possible mediator)" if c in CONTROL_POSTTREATMENT
            else "coverage < 92% on outcome rows"
            if float(panel_rows[c].notna().mean()) < 0.92
            else "UNJUSTIFIED -- inherited exclusion, coverage OK and not post-treatment"),
        "first_year": int(yrs.min()) if len(yrs) else None,
        "last_year": int(yrs.max()) if len(yrs) else None,
        "n_nonnull_full_panel": int(s.notna().sum()),
        "pct_complete_full_panel": float(s.notna().mean()),
        "pct_complete_outcome_rows": float(panel_rows[c].notna().mean()),
        "pct_complete_ref_sample": float(on_ref.notna().mean()) if len(on_ref) else np.nan,
    })
ctrl = pd.DataFrame(ctrl_rows).sort_values(
    ["status", "pct_complete_outcome_rows"], ascending=[True, False])


# =============================================================================
# Sheet 5 + 6: treatment and outcome coverage
# =============================================================================
tr_rows = []
for tid, (tcol, cond, tlabel) in TREATMENTS.items():
    s = df[tcol]
    nun = df.groupby("fips")[tcol].nunique()
    tr_rows.append({
        "treatment_id": tid, "column": tcol, "label": tlabel,
        "type": TREATMENT_TYPE.get(tid, ""), "conditioning": "+".join(cond),
        "n_nonnull": int(s.notna().sum()), "pct_complete": float(s.notna().mean()),
        "mean": float(s.mean()), "sd": float(s.std()),
        "min": float(s.min()), "max": float(s.max()),
        "n_counties_any_variation": int((nun > 1).sum()),
    })
tr = pd.DataFrame(tr_rows)

out_rows = []
for okey, ocol in OUTCOMES.items():
    s = df[ocol]
    yrs = df.loc[s.notna(), "year"]
    out_rows.append({
        "outcome": okey, "column": ocol,
        "first_year": int(yrs.min()) if len(yrs) else None,
        "last_year": int(yrs.max()) if len(yrs) else None,
        "n_nonnull": int(s.notna().sum()), "pct_complete_full_panel": float(s.notna().mean()),
        "n_counties": int(df.loc[s.notna(), "fips"].nunique()),
        "mean": float(s.mean()), "sd": float(s.std()),
    })
outc = pd.DataFrame(out_rows)

readme = pd.DataFrame({
    "sheet": ["power_by_analysis", "panel_balance", "control_coverage",
              "treatment_summary", "outcome_coverage"],
    "contents": [
        "One row per (treatment x outcome). N, treated/control split, switchers, realised state-clustered SE, and MDE at 80% power.",
        "Balance of the estimating panel for each analysis: year span, obs per county, share with full span, balance ratio, counties with gaps.",
        "Every control in CONTROL_FULL25 / CORE_9: whether it was used or dropped, why, coverage years, and completeness on the reference estimating sample.",
        "All 14 treatment definitions: coverage, moments, and how many counties have any within-county variation.",
        "Each outcome's year span and completeness.",
    ],
    "note": [
        f"MDE = {MDE_MULT:.4f} x SE (80% power, two-sided 5%). Ex-post, from the realised SE. mde_in_within_sd scales by the WITHIN-county sd, which is the variation a FE model uses.",
        "is_balanced = every county has the same number of observations. balance_ratio = N / (counties x years spanned); 1.0 is fully balanced.",
        "pct_complete_ref_sample is measured on the rows the analyses actually ran on. pct_complete_outcome_rows is over all rows with a non-missing Poor MH Days value.",
        "treated/control counts are exact for type=binary only. For count/rate/share/index types the >0 split is descriptive; the estimator uses the full continuous variation.",
        "Coverage differences across outcomes are the binding constraint on which entry cohorts are observable at all.",
    ],
})

with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
    readme.to_excel(w, sheet_name="README", index=False)
    power.to_excel(w, sheet_name="power_by_analysis", index=False)
    bal.to_excel(w, sheet_name="panel_balance", index=False)
    ctrl.to_excel(w, sheet_name="control_coverage", index=False)
    tr.to_excel(w, sheet_name="treatment_summary", index=False)
    outc.to_excel(w, sheet_name="outcome_coverage", index=False)
    for name, frame in [("README", readme), ("power_by_analysis", power),
                        ("panel_balance", bal), ("control_coverage", ctrl),
                        ("treatment_summary", tr), ("outcome_coverage", outc)]:
        ws = w.sheets[name]
        ws.freeze_panes = "A2"
        for i, col in enumerate(frame.columns, 1):
            width = max(len(str(col)), int(frame[col].astype(str).str.len().max() if len(frame) else 0))
            ws.column_dimensions[ws.cell(1, i).column_letter].width = min(max(width + 2, 10), 52)

ctrl.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_control_coverage.csv"), index=False)
power.to_csv(os.path.join(OUT_TABLES, f"{TODAY}_power_by_analysis.csv"), index=False)

print(f"\nSaved workbook: {XLSX}")
print(f"  power_by_analysis  {len(power):3d} rows")
print(f"  panel_balance      {len(bal):3d} rows")
print(f"  control_coverage   {len(ctrl):3d} rows")


# =============================================================================
# Sheet 7: VIF for the HEADLINE specification only
# =============================================================================
# WHY ONLY ONE SPEC. VIF is a property of a PARTICULAR design matrix, not of the
# data or of "the controls" in the abstract. It changes when you add or drop a
# regressor, when listwise deletion changes the sample, and -- most of all here --
# when you within-transform. There is therefore no such thing as "the VIF table"
# for this project; there are 186 of them, one per estimated spec.
#
# So the presentation split is:
#   - every spec's max VIF is carried in the VIF_audit.csv written by script4a,
#     and is summarised in one sentence in the table notes;
#   - ONE table, this one, shows the full VIF vector for the headline spec, with
#     raw and within side by side, because that contrast is the substantive point:
#     collinearity among county health/demographic indicators is almost entirely
#     CROSS-SECTIONAL and county fixed effects remove it.
# imported from script4_treatment -- do not redefine here
HEADLINE_TID = HEADLINE_TREATMENT
HEADLINE_OUT = HEADLINE_OUTCOME

_tcol, _cond, _tlab = TREATMENTS[HEADLINE_TID]
_need = ["fips", "year", HEADLINE_OUT, _tcol] + list(_cond) + CONTROL_PRETREAT
_s = df[list(dict.fromkeys(_need))].dropna()
_rhs = [_tcol] + list(_cond) + CONTROL_PRETREAT


def _vif_vector(frame, cols, within):
    W = frame[cols].astype(float).copy()
    if within:
        for c in cols:
            W[c] = (W[c] - frame.groupby("fips")[c].transform("mean")
                        - frame.groupby("year")[c].transform("mean") + frame[c].mean())
    keep = [c for c in cols if W[c].std() > 1e-10]
    R = np.corrcoef(W[keep].values, rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(R)), index=keep)


_raw = _vif_vector(_s, _rhs, within=False)
_wth = _vif_vector(_s, _rhs, within=True)
vif_headline = pd.DataFrame({
    "variable": _rhs,
    "role": ["treatment"] + ["conditioning"] * len(_cond) + ["control"] * len(CONTROL_PRETREAT),
    "vif_raw_levels": [_raw.get(c, np.nan) for c in _rhs],
    "vif_within_county_year": [_wth.get(c, np.nan) for c in _rhs],
}).sort_values("vif_within_county_year", ascending=False)


# =============================================================================
# Sheet 8: control SELECTION FUNNEL -- why 9 controls and not 110
# =============================================================================
# The panel has 162 columns, which is where "why only 20 controls?" comes from.
# Most are not candidate controls at all. This sheet walks the funnel explicitly
# so the selection is auditable rather than asserted.
import re as _re

_NON_CONTROL_EXACT = {
    "fips", "year", "state", "county", "state_fips", "county_fips", "rural",
    "non_large_metro", "state_abbr", "county_name", "state_name", "state_code",
    "cdc_in_query", "deaths_is_zero",
}
_OUTCOME_COLS = set(OUTCOMES.values()) | {
    "aggravated_assault_per100k", "simple_assault_per100k", "crude_rate_despair",
    "deaths_despair", "crime_assault",
}


def _classify(c):
    lc = c.lower()
    if lc in _NON_CONTROL_EXACT:
        return "identifier / flag"
    if _re.search(r"(_numerator|_denominator|_ci_low|_ci_high|_flag|_unreliable|_z_score|_rank|_quartile)$", lc):
        return "CHR companion column"
    if "cafo" in lc:
        return "treatment (CAFO)"
    if "fsis" in lc:
        return "treatment (FSIS)"
    if c in _OUTCOME_COLS:
        return "outcome"
    if "pop" in lc:
        return "population / denominator"
    return "candidate control"


_orig = [c for c in df.columns if not c.startswith(("tr_", "cond_", "any_large_", "thr_"))
         and c not in ("log_pop", "t_rel", "cohort", "add_year", "entry_year",
                       "build_year", "consol_year")]
_ref_rows = df[df[REF_OUT].notna()]
funnel_rows = []
for c in _orig:
    role = _classify(c)
    is_num = pd.api.types.is_numeric_dtype(df[c])
    cov = float(_ref_rows[c].notna().mean()) if c in _ref_rows.columns else np.nan
    funnel_rows.append({
        "column": c, "role": role, "numeric": is_num,
        "pct_complete_outcome_rows": cov,
        "passes_coverage_92": bool(is_num and cov >= 0.92),
        "in_FULL25": c in CONTROL_FULL25,
        "in_CORE9": c in CONTROL_CORE9,
    })
funnel = pd.DataFrame(funnel_rows)

_cand = funnel[funnel.role == "candidate control"]
funnel_summary = pd.DataFrame({
    "stage": [
        "All columns in panel (as written by script1b)",
        "  less identifiers, flags, CHR companion columns",
        "  less treatment (CAFO / FSIS) columns",
        "  less outcome and population/denominator columns",
        "= Candidate control pool",
        "  of which numeric",
        "  of which >=92% complete on outcome rows",
        "Considered in CONTROL_FULL25",
        "USED in CORE_9",
    ],
    "n": [
        len(funnel),
        len(funnel) - (funnel.role.isin(["identifier / flag", "CHR companion column"])).sum(),
        len(funnel) - (funnel.role.isin(["identifier / flag", "CHR companion column",
                                          "treatment (CAFO)", "treatment (FSIS)"])).sum(),
        len(_cand),
        len(_cand),
        int(_cand.numeric.sum()),
        int(_cand.passes_coverage_92.sum()),
        int(funnel.in_FULL25.sum()),
        int(funnel.in_CORE9.sum()),
    ],
})

# High-coverage candidates that were never even considered for FULL25 -- the
# honest answer to "why 20 controls when there are hundreds of columns": the
# control set was INHERITED from script3-ridge.py, not derived from this pool.
never_considered = _cand[(_cand.passes_coverage_92) & (~_cand.in_FULL25)][
    ["column", "pct_complete_outcome_rows"]].sort_values(
    "pct_complete_outcome_rows", ascending=False)

with pd.ExcelWriter(XLSX, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as w:
    vif_headline.to_excel(w, sheet_name="vif_headline_spec", index=False)
    funnel_summary.to_excel(w, sheet_name="control_funnel", index=False)
    funnel.to_excel(w, sheet_name="control_funnel_detail", index=False)
    never_considered.to_excel(w, sheet_name="controls_never_considered", index=False)

print(f"\n=== control selection funnel ===")
print(funnel_summary.to_string(index=False))
print(f"\n=== high-coverage candidates NEVER considered for FULL25: {len(never_considered)} ===")
print(never_considered.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\n=== VIF, headline spec ({HEADLINE_TID}, {REF_OUT}) ===")
print(vif_headline.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
