#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script3-ridge.py

LASSO/Ridge variable selection + Panel OLS with county × year fixed effects
+ DiD event study around USDA Census years.

Theory: LARGE CAFO presence and slaughterhouse (FSIS) operations drive mental
health deterioration — not CAFO count in general. Most CAFOs are small by count,
so large/large+medium treatments isolate the relevant margin.

Sample design:
  Var A — 2010–2023, CAFO treatments only (FSIS NaN pre-2017, noted in output)
  Var B — 2017–2023, CAFO + FSIS treatments jointly

CAFO data: backfilled between USDA Census years (2002 → held through 2006,
2007 → held through 2011, etc.). Within-county year-over-year variation is
valid but reflects census-period averages.

Deaths of despair rate: derived from raw death counts / county population
(not CDC-suppressed). Fill reflects county-year reporting completeness.

Outputs → Dropbox/Mental/Data/merged/figs/script3/
  X1_lasso_heatmap.png          LASSO control selection matrix
  X2_coef_plot.png              Panel OLS: treatment × outcome grid (Var A + Var B)
  X3_event_study.png            DiD event study: CAFO entry + consolidation
  X4_robustness_plot.png        Placebo, Hispanic quartile, regional splits
  Block1_lasso_selection.csv    Raw LASSO selection table
  Block2_panel_ols_results.csv  Full regression results (102 regressions)
  Block3_event_study_results.csv Event study coefficients
  Block4_robustness_results.csv  Robustness results
"""

from packages import *
from functions import *
import statsmodels.api as sm

# ── Directories ───────────────────────────────────────────────────────────────
merged_dir = os.path.join(db_data, "merged")
out_dir    = os.path.join(db_data, "merged", "figs", "script3")
os.makedirs(out_dir, exist_ok=True)
today_str  = date.today().strftime("%Y-%m-%d")

# ── Load panel ────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(latest_file_glob(merged_dir, "*_full_merged.csv"), low_memory=False)
df_raw = df_raw[df_raw["rural"] == 1].copy()
print(f"Rural panel: {len(df_raw):,} rows | {df_raw['fips'].nunique():,} counties | "
      f"years {int(df_raw['year'].min())}–{int(df_raw['year'].max())}")

POP_COL = "population"

# ── Helper: log per 10k ───────────────────────────────────────────────────────
def log_per10k(series, pop):
    x   = pd.to_numeric(series, errors="coerce")
    pop = pd.to_numeric(pop,    errors="coerce").replace(0, np.nan)
    return np.log1p((x / pop) * 10_000)

# ── Derive treatment variables ────────────────────────────────────────────────
def _derive_treatments(df):
    d = df.copy()
    pop = d[POP_COL]

    # Aggregate size sums (cross-animal)
    d["_large_all"]       = d[["cafo_cattle_large","cafo_hogs_large",
                                "cafo_chickens_large"]].sum(axis=1, min_count=1)
    d["_largemedium_all"] = (
        d[["cafo_cattle_large","cafo_hogs_large","cafo_chickens_large"]].sum(axis=1, min_count=1)
      + d[["cafo_cattle_medium","cafo_hogs_medium","cafo_chickens_medium"]].sum(axis=1, min_count=1)
    )
    d["_small_all"]       = d[["cafo_cattle_small","cafo_hogs_small",
                                "cafo_chickens_small"]].sum(axis=1, min_count=1)

    # Per-animal large+medium sums
    for animal in ["cattle","hogs","chickens"]:
        d[f"_largemedium_{animal}"] = (d[f"cafo_{animal}_large"].fillna(0)
                                     + d[f"cafo_{animal}_medium"].fillna(0))

    # Log per-10k transforms
    d["cafo_total_log"]           = log_per10k(d["cafo_total_ops_all_animals"], pop)
    d["cafo_large_log"]           = log_per10k(d["_large_all"], pop)
    d["cafo_largemedium_log"]     = log_per10k(d["_largemedium_all"], pop)
    d["cafo_small_log"]           = log_per10k(d["_small_all"], pop)   # placebo

    for animal in ["cattle","hogs","chickens"]:
        d[f"cafo_{animal}_log"] = log_per10k(
            d[[f"cafo_{animal}_small",f"cafo_{animal}_medium",
               f"cafo_{animal}_large"]].sum(axis=1, min_count=1), pop)
        d[f"cafo_{animal}_large_log"]       = log_per10k(d[f"cafo_{animal}_large"], pop)
        d[f"cafo_{animal}_largemedium_log"] = log_per10k(d[f"_largemedium_{animal}"], pop)

    # FSIS treatments (populated 2017+)
    for key, col in [
        ("fsis_total_log",     "n_unique_establishments_fsis"),
        ("fsis_slaughter_log", "n_slaughterhouse_present_establishments_fsis"),
        ("fsis_meat_log",      "n_meat_slaughter_establishments_fsis"),
        ("fsis_poultry_log",   "n_poultry_slaughter_establishments_fsis"),
    ]:
        d[key] = log_per10k(d[col], pop) if col in d.columns else np.nan

    return d

df_raw = _derive_treatments(df_raw)

# ── Sample windows ────────────────────────────────────────────────────────────
df_a = df_raw[df_raw["year"].between(2010, 2023)].copy()   # Var A: CAFO only
df_b = df_raw[df_raw["year"].between(2017, 2023)].copy()   # Var B: CAFO + FSIS
print(f"Var A (2010–2023): {len(df_a):,} rows | Var B (2017–2023): {len(df_b):,} rows")

# ── Treatment groups ──────────────────────────────────────────────────────────
TREATMENTS_A = {
    # Aggregate
    "CAFO Total":                  "cafo_total_log",
    "CAFO Large (all animals)":    "cafo_large_log",
    "CAFO Large+Med (all animals)":"cafo_largemedium_log",
    # By animal — total
    "Cattle (total)":              "cafo_cattle_log",
    "Hogs (total)":                "cafo_hogs_log",
    "Chickens (total)":            "cafo_chickens_log",
    # By animal — large only
    "Cattle (large)":              "cafo_cattle_large_log",
    "Hogs (large)":                "cafo_hogs_large_log",
    "Chickens (large)":            "cafo_chickens_large_log",
    # By animal — large+medium
    "Cattle (large+med)":          "cafo_cattle_largemedium_log",
    "Hogs (large+med)":            "cafo_hogs_largemedium_log",
    "Chickens (large+med)":        "cafo_chickens_largemedium_log",
}

TREATMENTS_B = {
    **TREATMENTS_A,
    "FSIS Total":                  "fsis_total_log",
    "FSIS Slaughter":              "fsis_slaughter_log",
    "FSIS Meat Slaughter":         "fsis_meat_log",
    "FSIS Poultry Slaughter":      "fsis_poultry_log",
}

# ── Outcomes ──────────────────────────────────────────────────────────────────
OUTCOMES = {
    "Poor MH Days":         "poor_mental_health_days",
    "Deaths of Despair":    "crude_rate_despair",
    "Excessive Drinking":   "excessive_drinking_per100k",
    "Agg Assault":          "aggravated_assault_per100k",
    "Violent Crime":        "violent_crime",
    "Freq Mental Distress": "frequent_mental_distress_per100k",
}

# ── Control candidate pool (all 25 confirmed present in panel) ─────────────────
CONTROL_COLS = [
    "adult_obesity_per100k", "adult_smoking_per100k", "unemployment_per100k",
    "children_in_poverty_per100k", "uninsured_adults_per100k",
    "median_household_income", "income_inequality",
    "%_hispanic", "%_non-hispanic_african_american", "%_65_and_older", "%_female",
    "mental_health_providers_per100k", "primary_care_physicians_per100k",
    "food_insecurity_per100k", "physical_inactivity_per100k",
    "poor_physical_health_days", "teen_births_per100k", "low_birthweight_per100k",
    "premature_death", "preventable_hospital_stays", "some_college_per100k",
    "children_in_single-parent_households_per100k", "diabetes_prevalence_per100k",
    "air_pollution_-_particulate_matter", "social_associations_per100k",
]

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})

_TGROUP_COLORS = {
    "aggregate":  "#4d4d4d",
    "animal_tot": "#1b7837",
    "large_only": "#762a83",
    "large_med":  "#b35806",
    "fsis":       "#d6604d",
}
def _treatment_group(label):
    if "CAFO Total" in label or "Large (all" in label or "Large+Med (all" in label:
        return "aggregate"
    if "FSIS" in label:
        return "fsis"
    if "large+med" in label.lower():
        return "large_med"
    if "(large)" in label.lower():
        return "large_only"
    return "animal_tot"


# =============================================================================
# BLOCK 1: LASSO control selection
# =============================================================================
print("\n" + "="*70)
print("BLOCK 1: LASSO control selection")
print("="*70)

from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SELECTED_CONTROLS = {}    # outcome_key → list of selected control col names
selection_matrix  = {}    # outcome_key → binary Series indexed by control col

for outcome_key, outcome_col in OUTCOMES.items():
    y    = df_a[outcome_col].copy()
    mask = y.notna()
    X_ctrl = df_a.loc[mask, CONTROL_COLS].copy()
    y_fit  = y[mask]

    imp = SimpleImputer(strategy="mean")
    scl = StandardScaler()
    X_imp = scl.fit_transform(imp.fit_transform(X_ctrl))

    lasso = LassoCV(cv=5, max_iter=10_000, random_state=42, n_alphas=100)
    lasso.fit(X_imp, y_fit)

    selected = [c for c, coef in zip(CONTROL_COLS, lasso.coef_) if abs(coef) > 0]
    SELECTED_CONTROLS[outcome_key] = selected
    selection_matrix[outcome_key]  = pd.Series(
        {c: int(abs(coef) > 0) for c, coef in zip(CONTROL_COLS, lasso.coef_)}
    )
    fill = mask.mean()
    print(f"  {outcome_key:25s}  fill={fill:.2f}  α={lasso.alpha_:.4f}  "
          f"N={mask.sum():,}  selected={len(selected)}/{len(CONTROL_COLS)} controls")

sel_df = pd.DataFrame(selection_matrix).T    # rows = outcomes, cols = controls
sel_df.to_csv(os.path.join(out_dir, f"{today_str}_Block1_lasso_selection.csv"))

# Figure X1: LASSO heatmap
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(sel_df, ax=ax, cmap="Blues", vmin=0, vmax=1,
            linewidths=0.4, linecolor="white",
            cbar_kws={"label": "Selected by LASSO (1=yes)", "shrink": 0.6},
            annot=True, fmt="d", annot_kws={"size": 7})
ax.set_title(
    "LASSO Control Selection — Rural US Counties, 2010–2023\n"
    "1 = control retained (non-zero coefficient); 0 = zeroed out",
    fontsize=11,
)
ax.tick_params(axis="x", labelsize=7, rotation=40)
ax.tick_params(axis="y", labelsize=8, rotation=0)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_X1_lasso_heatmap.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)


# =============================================================================
# BLOCK 2: Panel OLS grid — county + year FE, clustered SEs
# =============================================================================
print("\n" + "="*70)
print("BLOCK 2: Panel OLS — treatment × outcome grid")
print("="*70)

def within_transform(df, y_col, x_cols, entity="fips", time="year"):
    """Two-way within estimator: demean by entity then by time period."""
    keep = [y_col] + x_cols + [entity, time]
    out  = df[keep].dropna().copy()
    for col in [y_col] + x_cols:
        grand_mean  = out[col].mean()
        county_mean = out.groupby(entity)[col].transform("mean")
        year_mean   = out.groupby(time)[col].transform("mean")
        out[col]    = out[col] - county_mean - year_mean + grand_mean
    return out

def run_panel_ols(df, treatment_col, outcome_col, control_cols, label=""):
    """Panel OLS with two-way FE (within transformation) and county-clustered SEs."""
    x_cols = [treatment_col] + [c for c in control_cols if c != treatment_col]
    dm = within_transform(df, outcome_col, x_cols)
    if len(dm) < 200:
        return None
    Y      = dm[outcome_col]
    X      = sm.add_constant(dm[x_cols], has_constant="add")
    groups = dm["fips"]
    try:
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    except Exception as e:
        print(f"    OLS failed [{label}]: {e}")
        return None
    beta = res.params.get(treatment_col, np.nan)
    se   = res.bse.get(treatment_col, np.nan)
    pval = res.pvalues.get(treatment_col, np.nan)
    return {
        "beta": beta, "se": se,
        "ci_lo": beta - 1.96*se, "ci_hi": beta + 1.96*se,
        "pval": pval, "N": int(res.nobs), "r2": res.rsquared,
    }

results_rows = []

for var_label, df_var, treatments in [
    ("Var A (2010-2023)", df_a, TREATMENTS_A),
    ("Var B (2017-2023)", df_b, TREATMENTS_B),
]:
    for outcome_key, outcome_col in OUTCOMES.items():
        ctrl = SELECTED_CONTROLS.get(outcome_key, [])
        for treat_label, treat_col in treatments.items():
            res = run_panel_ols(df_var, treat_col, outcome_col, ctrl,
                                label=f"{treat_label}|{outcome_key}|{var_label}")
            if res is None:
                continue
            row = {
                "sample": var_label,
                "outcome": outcome_key, "treatment": treat_label,
                "treatment_col": treat_col, "outcome_col": outcome_col,
                **res,
            }
            results_rows.append(row)
            sig = "*" if res["pval"] < 0.05 else " "
            print(f"  {var_label[:5]} | {outcome_key:20s} | {treat_label:30s} "
                  f"β={res['beta']:+.4f}  p={res['pval']:.3f}{sig}  N={res['N']:,}")

results_df = pd.DataFrame(results_rows)
csv_path = os.path.join(out_dir, f"{today_str}_Block2_panel_ols_results.csv")
results_df.to_csv(csv_path, index=False)
print("Saved:", csv_path)

# Figure X2: Coefficient plot — faceted by outcome, colored by treatment group
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

n_out = len(OUTCOMES)
fig, axes = plt.subplots(1, n_out, figsize=(4.5 * n_out, 10), sharey=False)

for ax, outcome_key in zip(axes, OUTCOMES.keys()):
    sub = results_df[results_df["outcome"] == outcome_key].copy()
    sub = sub.sort_values("beta").reset_index(drop=True)

    for i, row in sub.iterrows():
        grp   = _treatment_group(row["treatment"])
        color = _TGROUP_COLORS[grp]
        mk    = "D" if "Var B" in row["sample"] else "o"
        ax.errorbar(row["beta"], i,
                    xerr=[[row["beta"] - row["ci_lo"]], [row["ci_hi"] - row["beta"]]],
                    fmt=mk, color=color, markersize=4, capsize=2,
                    elinewidth=0.8, markeredgecolor="white")
        if row["pval"] < 0.05:
            ax.text(row["ci_hi"] + 0.001, i, "*", fontsize=8, va="center", color=color)

    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["treatment"].tolist(), fontsize=6)
    ax.set_xlabel("β (panel OLS)", fontsize=8)
    ax.set_title(outcome_key, fontsize=9, fontweight="bold")

_leg = [mpatches.Patch(color=v, label=k.replace("_"," ")) for k, v in _TGROUP_COLORS.items()]
_leg += [
    Line2D([0],[0], ls="-",  color="grey", marker="o", ms=4, label="Var A (2010–23)"),
    Line2D([0],[0], ls="--", color="grey", marker="D", ms=4, label="Var B (2017–23)"),
]
fig.legend(handles=_leg, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.suptitle(
    "Panel OLS: Treatment β ± 95% CI by Outcome — Rural US Counties\n"
    "County + year FE, clustered SEs at county level   |   * = p < 0.05\n"
    "Circle = Var A (2010–23, CAFO only) / Diamond = Var B (2017–23, CAFO+FSIS)",
    fontsize=10, y=1.01,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_X2_coef_plot.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)


# =============================================================================
# BLOCK 3: DiD event study — CAFO entry and consolidation
# =============================================================================
print("\n" + "="*70)
print("BLOCK 3: DiD event study — CAFO entry + consolidation")
print("="*70)

df_raw["large_total"] = df_raw[["cafo_cattle_large","cafo_hogs_large",
                                  "cafo_chickens_large"]].sum(axis=1, min_count=1)
df_raw["total_all"]   = df_raw[["cafo_cattle_small","cafo_cattle_medium","cafo_cattle_large",
                                  "cafo_hogs_small","cafo_hogs_medium","cafo_hogs_large",
                                  "cafo_chickens_small","cafo_chickens_medium",
                                  "cafo_chickens_large"]].sum(axis=1, min_count=1)
df_raw["large_share"] = df_raw["large_total"] / df_raw["total_all"].replace(0, np.nan)

# Detect events at USDA Census years
CENSUS_YEARS = [2002, 2007, 2012, 2017]
df_cens = df_raw[df_raw["year"].isin(CENSUS_YEARS)].sort_values(["fips","year"]).copy()
df_cens["large_total_lag"] = df_cens.groupby("fips")["large_total"].shift(1)
df_cens["large_share_lag"] = df_cens.groupby("fips")["large_share"].shift(1)

# Entry: 0 large CAFOs → >0 large CAFOs between consecutive census years
df_cens["entry_event"] = (
    (df_cens["large_total"] > 0) &
    (df_cens["large_total_lag"] == 0) &
    df_cens["large_total_lag"].notna()
)
# Consolidation: large share rises ≥10 pp between census years
df_cens["consol_event"] = (
    ((df_cens["large_share"] - df_cens["large_share_lag"]) >= 0.10) &
    df_cens["large_share_lag"].notna()
)

entry_events  = df_cens[df_cens["entry_event"]][["fips","year"]].rename(columns={"year":"event_year"})
consol_events = df_cens[df_cens["consol_event"]][["fips","year"]].rename(columns={"year":"event_year"})
print(f"  Entry events:         {len(entry_events):,} county-census-year obs")
print(f"  Consolidation events: {len(consol_events):,} county-census-year obs")

FOCUS_OUTCOME = "poor_mental_health_days"

def run_event_study(df_full, events_df, outcome_col, event_label,
                    n_leads=4, n_lags=4, ctrl_cols=None):
    """
    TWFE event-study regression.
    Omitted category: t_rel = -1 (one period before event).
    Control group: counties that never experience the event.
    """
    if ctrl_cols is None:
        ctrl_cols = []

    treated_fips = set(events_df["fips"].unique())
    first_event  = (events_df.groupby("fips")["event_year"].min()
                              .reset_index().rename(columns={"event_year":"event_year"}))

    df_es = df_full[[outcome_col,"fips","year"] + ctrl_cols].copy()
    df_es = df_es.merge(first_event, on="fips", how="left")
    never_fips = set(df_full["fips"].unique()) - treated_fips

    df_es["t_rel"] = np.where(
        df_es["event_year"].notna(),
        df_es["year"] - df_es["event_year"],
        np.nan,
    )

    df_treated = df_es[
        df_es["fips"].isin(treated_fips) &
        df_es["t_rel"].between(-n_leads, n_lags)
    ].copy()
    df_never = df_es[df_es["fips"].isin(never_fips)].copy()
    df_never["t_rel"] = np.nan
    df_pool = pd.concat([df_treated, df_never], ignore_index=True)

    if df_pool[outcome_col].notna().sum() < 200:
        print(f"    {event_label}: too few obs, skip")
        return pd.DataFrame()

    rel_times     = list(range(-n_leads, n_lags + 1))
    rel_times_use = [t for t in rel_times if t != -1]

    for t in rel_times_use:
        df_pool[f"d_t{t:+d}"] = (df_pool["t_rel"] == t).astype(float)

    dummy_cols = [f"d_t{t:+d}" for t in rel_times_use]
    x_cols     = dummy_cols + ctrl_cols
    dm         = within_transform(df_pool, outcome_col, x_cols)
    if len(dm) < 200:
        return pd.DataFrame()

    Y = dm[outcome_col]
    X = sm.add_constant(dm[x_cols], has_constant="add")
    try:
        res = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": dm["fips"]})
    except Exception as e:
        print(f"    {event_label} OLS failed: {e}")
        return pd.DataFrame()

    rows = []
    for t in rel_times_use:
        col = f"d_t{t:+d}"
        if col in res.params.index:
            beta = res.params[col]; se = res.bse[col]
            rows.append({"t_rel": t, "beta": beta, "se": se,
                         "ci_lo": beta - 1.96*se, "ci_hi": beta + 1.96*se,
                         "pval": res.pvalues[col], "event": event_label})
    # Omitted period: β = 0 by normalization
    rows.append({"t_rel": -1, "beta": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                 "pval": np.nan, "event": event_label})
    return pd.DataFrame(rows).sort_values("t_rel")

ctrl_focus = SELECTED_CONTROLS.get("Poor MH Days", [])
es_entry   = run_event_study(df_a, entry_events,  FOCUS_OUTCOME, "CAFO Entry",
                              ctrl_cols=ctrl_focus)
es_consol  = run_event_study(df_a, consol_events, FOCUS_OUTCOME, "CAFO Consolidation",
                              ctrl_cols=ctrl_focus)
es_df = pd.concat([es_entry, es_consol], ignore_index=True)
es_csv = os.path.join(out_dir, f"{today_str}_Block3_event_study_results.csv")
es_df.to_csv(es_csv, index=False)
print("Saved:", es_csv)

# Figure X3
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
event_colors = {"CAFO Entry": "#762a83", "CAFO Consolidation": "#b35806"}

for ax, event_label in zip(axes, ["CAFO Entry", "CAFO Consolidation"]):
    sub = es_df[es_df["event"] == event_label].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
        ax.set_title(event_label); continue
    sub   = sub.sort_values("t_rel")
    color = event_colors[event_label]

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(-0.5, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax.fill_between([0 - 0.5, sub["t_rel"].max() + 0.5],
                    ax.get_ylim()[0], ax.get_ylim()[1],
                    color=color, alpha=0.04)

    ax.errorbar(sub["t_rel"], sub["beta"],
                yerr=[sub["beta"] - sub["ci_lo"], sub["ci_hi"] - sub["beta"]],
                fmt="o", color=color, markersize=5, capsize=3,
                elinewidth=1, markeredgecolor="white")

    pre_mask = sub["t_rel"] < 0
    if pre_mask.sum() >= 2:
        pre_betas = sub.loc[pre_mask, "beta"]
        avg_pre_p = sub.loc[pre_mask, "pval"].fillna(1).mean()
        ax.text(0.05, 0.95,
                f"Pre-trend: max|β|={pre_betas.abs().max():.3f}\nAvg pre-p={avg_pre_p:.2f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel("Years relative to event (0 = census year of event)", fontsize=9)
    ax.set_ylabel(f"β on {FOCUS_OUTCOME}", fontsize=9)
    ax.set_title(f"{event_label}", fontsize=10, fontweight="bold")
    ax.set_xticks(sorted(sub["t_rel"].unique()))

fig.suptitle(
    "DiD Event Study: CAFO Entry & Consolidation → Poor Mental Health Days\n"
    "Rural US Counties, 2010–2023 | County + year FE, county-clustered SEs\n"
    "Omitted: t = −1 (one period before event) | Shaded = post-event region",
    fontsize=10, y=1.02,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_X3_event_study.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)


# =============================================================================
# BLOCK 4: Robustness
# =============================================================================
print("\n" + "="*70)
print("BLOCK 4: Robustness")
print("="*70)

PRIMARY_OUTCOME = "poor_mental_health_days"
PRIMARY_TREAT   = "cafo_large_log"
PRIMARY_CTRL    = SELECTED_CONTROLS.get("Poor MH Days", [])

robust_rows = []

# 4a: Placebo — small CAFO as treatment (should be near zero if theory holds)
print("  4a: Placebo (small CAFO)")
for lbl, tcol in [("Small CAFO (placebo)", "cafo_small_log"),
                   ("Large CAFO (main)",    "cafo_large_log")]:
    res = run_panel_ols(df_a, tcol, PRIMARY_OUTCOME, PRIMARY_CTRL, lbl)
    if res:
        robust_rows.append({"group": "Placebo", "label": lbl, **res})

# 4b: Hispanic quartile split — Q1 vs Q4
print("  4b: Hispanic quartile split")
HISP_COL  = "%_hispanic"
hisp_data = df_a[df_a[HISP_COL].notna()].copy()
hisp_data["hisp_q"] = pd.qcut(hisp_data[HISP_COL], q=4,
                               labels=["Q1 (Low)","Q2","Q3","Q4 (High)"])
for ql in ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]:
    sub = hisp_data[hisp_data["hisp_q"] == ql]
    res = run_panel_ols(sub, PRIMARY_TREAT, PRIMARY_OUTCOME, PRIMARY_CTRL, f"hispanic_{ql}")
    if res:
        robust_rows.append({"group": "Hispanic Quartile", "label": ql, **res})

# 4c: Regional split
print("  4c: Regional split")
if "region" in df_a.columns:
    for reg in sorted(df_a["region"].dropna().unique()):
        sub = df_a[df_a["region"] == reg]
        res = run_panel_ols(sub, PRIMARY_TREAT, PRIMARY_OUTCOME, PRIMARY_CTRL, f"region_{reg}")
        if res:
            robust_rows.append({"group": "Region", "label": str(reg), **res})

robust_df = pd.DataFrame(robust_rows)
csv_path  = os.path.join(out_dir, f"{today_str}_Block4_robustness_results.csv")
robust_df.to_csv(csv_path, index=False)
print("Saved:", csv_path)

# Figure X4
groups_uniq = robust_df["group"].unique().tolist()
n_grp = len(groups_uniq)
fig, axes = plt.subplots(1, n_grp, figsize=(5 * n_grp, 6), sharey=False)
if n_grp == 1:
    axes = [axes]

grp_colors = {"Placebo": "#4d4d4d", "Hispanic Quartile": "#4393c3", "Region": "#1b7837"}

for ax, grp in zip(axes, groups_uniq):
    sub   = robust_df[robust_df["group"] == grp].copy().reset_index(drop=True)
    color = grp_colors.get(grp, "grey")

    for i, row in sub.iterrows():
        ax.errorbar(row["beta"], i,
                    xerr=[[row["beta"] - row["ci_lo"]], [row["ci_hi"] - row["beta"]]],
                    fmt="o", color=color, markersize=5, capsize=3,
                    elinewidth=0.9, markeredgecolor="white")
        if row["pval"] < 0.05:
            ax.text(row["ci_hi"] + 0.001, i, "*", fontsize=9, va="center", color=color)

    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["label"].tolist(), fontsize=8)
    ax.set_xlabel("β (panel OLS)", fontsize=9)
    ax.set_title(grp, fontsize=10, fontweight="bold")

fig.suptitle(
    f"Robustness — Outcome: Poor Mental Health Days | Treatment: Large CAFO (log/10k)\n"
    "Rural US Counties, 2010–2023 | County + year FE, county-clustered SEs | * = p < 0.05",
    fontsize=10, y=1.01,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_X4_robustness_plot.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)

print(f"\nAll script3 outputs saved to: {out_dir}")
