#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script2i-hispanic-chr-explorer.py

Explores the broader CHR health landscape across rural US counties, with explicit
attention to Hispanic population concentration as both a potential confounder and
a dimension of heterogeneity in CAFO-health relationships.

Research motivation:
  - Hispanic workers are disproportionately employed in meatpacking / CAFO operations.
  - Counties with higher CAFO/FSIS presence may differ on %_hispanic, which itself
    correlates with health outcomes (access, insurance, SES, etc.).
  - Controlling for %_hispanic in scatter plots tests whether observed CAFO-health
    correlations are spurious proxies for Hispanic population share.

Outputs saved to: Dropbox/Mental/Data/merged/figs/hispanic-chr/

Figures:
  S_chr_correlation_heatmap.png  — Spearman correlation heatmap, 15 CHR health vars
                                    + 4 treatment vars, 2010–2015 rural panel
  T_hispanic_quartile_profiles.png — Standardized (z-score) mean health outcome
                                      profiles by %_hispanic quartile (Q1–Q4)
  U_cafo_by_hispanic_quartile.png — CAFO + FSIS exposure distributions (box) by
                                     %_hispanic quartile
  V_partial_scatter_hispanic.png  — CAFO vs. despair/poor MH days residualized on
                                     %_hispanic; 4 panels per outcome (one per quartile)

Summary stats:
  S_chr_summary_by_hispanic_quartile.csv — mean±SD for all CHR vars × quartile
"""

from packages import *
from functions import *
from scipy import stats
from scipy.stats import spearmanr

# ── Directories ───────────────────────────────────────────────────────────────
merged_dir = os.path.join(db_data, "merged")
out_dir    = os.path.join(db_data, "merged", "figs", "hispanic-chr")
os.makedirs(out_dir, exist_ok=True)
today_str  = date.today().strftime("%Y-%m-%d")

# ── Load panel ────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(latest_file_glob(merged_dir, "*_full_merged.csv"), low_memory=False)
print(f"Panel loaded: {len(df_raw):,} rows | {df_raw['fips'].nunique():,} counties | "
      f"years {int(df_raw['year'].min())}–{int(df_raw['year'].max())}")

# Primary analysis window: 2010–2015 (best MH outcome coverage)
df = df_raw[df_raw["year"].between(2010, 2015)].copy()
print(f"Restricted to 2010–2015: {len(df):,} rows | {df['fips'].nunique():,} counties")

POP_COL  = "population"
HISP_COL = "%_hispanic"

# ── Helper: log per 10k ───────────────────────────────────────────────────────
def log_per10k(series, pop):
    x   = pd.to_numeric(series, errors="coerce")
    pop = pd.to_numeric(pop,    errors="coerce").replace(0, np.nan)
    return np.log1p((x / pop) * 10_000)

# ── Treatment variables (log per 10k) ─────────────────────────────────────────
df["cafo_total_log"]      = log_per10k(df["cafo_total_ops_all_animals"], df[POP_COL])
df["large_cafo_log"]      = log_per10k(
    df[["cafo_cattle_large","cafo_hogs_large","cafo_chickens_large"]].sum(axis=1, min_count=1),
    df[POP_COL]
)
# FSIS available 2017+; carry forward for structure but will be sparse in 2010-2015
df["fsis_total_log"]      = log_per10k(df.get("n_unique_establishments_fsis", pd.Series(np.nan, index=df.index)), df[POP_COL])
df["fsis_slaughter_log"]  = log_per10k(df.get("n_slaughterhouse_present_establishments_fsis", pd.Series(np.nan, index=df.index)), df[POP_COL])

TREATMENT_VARS = {
    "CAFO Total (log/10k)":     "cafo_total_log",
    "CAFO Large (log/10k)":     "large_cafo_log",
    "FSIS Total (log/10k)":     "fsis_total_log",
    "FSIS Slaughter (log/10k)": "fsis_slaughter_log",
}

# ── CHR health outcomes (fill ≥ 40% in 2010–2015) ────────────────────────────
CHR_HEALTH = {
    "Poor Mental Health Days":       "poor_mental_health_days",
    "Poor Physical Health Days":     "poor_physical_health_days",
    "Poor/Fair Health (%)":          "poor_or_fair_health_per100k",
    "Frequent Mental Distress (%)":  "frequent_mental_distress_per100k",
    "Frequent Physical Distress (%)":"frequent_physical_distress_per100k",
    "Adult Obesity (%)":             "adult_obesity_per100k",
    "Adult Smoking (%)":             "adult_smoking_per100k",
    "Excessive Drinking (%)":        "excessive_drinking_per100k",
    "Physical Inactivity (%)":       "physical_inactivity_per100k",
    "Diabetes Prevalence (%)":       "diabetes_prevalence_per100k",
    "Teen Births (per 100k)":        "teen_births_per100k",
    "Low Birthweight (%)":           "low_birthweight_per100k",
    "Premature Death (YPLL)":        "premature_death",
    "Preventable Hospital Stays":    "preventable_hospital_stays",
    "Uninsured Adults (%)":          "uninsured_adults_per100k",
    "Food Insecurity (%)":           "food_insecurity_per100k",
    "Children in Poverty (%)":       "children_in_poverty_per100k",
    "Unemployment (%)":              "unemployment_per100k",
    "Deaths of Despair (crude rate)":"crude_rate_cdc_county_year_deathsofdespair",
}
# Filter to cols that actually exist in the frame
CHR_HEALTH = {k: v for k, v in CHR_HEALTH.items() if v in df.columns}

# Core MH focus outcomes (for partial scatter)
FOCUS_OUTCOMES = {
    "Deaths of Despair (crude rate)": "crude_rate_cdc_county_year_deathsofdespair",
    "Poor Mental Health Days":        "poor_mental_health_days",
    "Aggravated Assault (per 100k)":  "aggravated_assault_per100k",
}
FOCUS_OUTCOMES = {k: v for k, v in FOCUS_OUTCOMES.items() if v in df.columns}

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linewidth":  0.5,
})

QUARTILE_COLORS = ["#d1e5f0", "#4393c3", "#2166ac", "#053061"]
QUARTILE_LABELS = ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]


# =============================================================================
# FIGURE S: Expanded CHR + treatment Spearman correlation heatmap
# =============================================================================
_corr_labels = list(CHR_HEALTH.keys()) + list(TREATMENT_VARS.keys())
_corr_cols   = [CHR_HEALTH[k] for k in CHR_HEALTH] + [TREATMENT_VARS[k] for k in TREATMENT_VARS]

# Keep rows where at least half the variables are non-null
_hm_df = df[_corr_cols + [HISP_COL]].copy()
_hm_df.columns = _corr_labels + ["%_hispanic"]
_all_labels = _corr_labels + ["%_hispanic"]

n = len(_all_labels)
corr_mat  = np.full((n, n), np.nan)
pval_mat  = np.full((n, n), np.nan)

for i, col_i in enumerate(_all_labels):
    for j, col_j in enumerate(_all_labels):
        _sub = _hm_df[[col_i, col_j]].dropna()
        if len(_sub) > 30:
            _res = spearmanr(_sub[col_i], _sub[col_j])
            _stat = np.asarray(_res.statistic)
            _pval = np.asarray(_res.pvalue)
            corr_mat[i, j] = float(_stat.flat[0])
            pval_mat[i, j] = float(_pval.flat[0])

corr_df = pd.DataFrame(corr_mat, index=_all_labels, columns=_all_labels)

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.eye(n, dtype=bool)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(
    corr_df, ax=ax, cmap=cmap, center=0, vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 6},
    linewidths=0.4, linecolor="white",
    mask=mask,
    cbar_kws={"shrink": 0.7, "label": "Spearman ρ"},
)
# Bold annotations where |ρ| ≥ 0.3 and p < 0.05
for txt in ax.texts:
    try:
        val = float(txt.get_text())
        _r_i = round(val, 2)
        # Find matching cell in corr_mat to get p-value
        _match = np.argwhere(np.round(corr_mat, 2) == _r_i)
        if len(_match) > 0:
            _ii, _jj = _match[0]
            if abs(corr_mat[_ii, _jj]) >= 0.30 and pval_mat[_ii, _jj] < 0.05:
                txt.set_fontweight("bold")
    except (ValueError, IndexError):
        pass

# Draw a separator between health vars and treatment vars
sep = len(CHR_HEALTH)
ax.axhline(sep, color="black", lw=1.5, ls="--")
ax.axvline(sep, color="black", lw=1.5, ls="--")

ax.set_title(
    "Spearman Correlations: CHR Health Outcomes × CAFO/FSIS Treatments × %_hispanic\n"
    "Rural US Counties, 2010–2015   |   Bold = |ρ| ≥ 0.30 and p < 0.05   |   Dashed line separates outcomes from treatments",
    fontsize=10, pad=12,
)
ax.tick_params(axis="x", labelsize=7, rotation=40)
ax.tick_params(axis="y", labelsize=7, rotation=0)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_S_chr_correlation_heatmap.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)


# =============================================================================
# FIGURE T: Health outcome profiles by %_hispanic quartile (standardized z-scores)
# Rows with non-null %_hispanic only.
# =============================================================================
_t_df = df[[HISP_COL] + list(CHR_HEALTH.values())].copy()
_t_df = _t_df[_t_df[HISP_COL].notna()].copy()
_t_df["hisp_q"] = pd.qcut(_t_df[HISP_COL], q=4, labels=QUARTILE_LABELS)

# Standardize each health variable (z-score using pooled mean/SD)
_z_df = _t_df.copy()
for col in CHR_HEALTH.values():
    _mu = _t_df[col].mean()
    _sd = _t_df[col].std()
    if _sd > 0:
        _z_df[col] = (_t_df[col] - _mu) / _sd

# Compute mean z-score per quartile × health var
_q_means = _z_df.groupby("hisp_q", observed=True)[list(CHR_HEALTH.values())].mean()

# Sort health vars by Q4 mean (descending) so the most-elevated-in-high-Hispanic appear first
_sort_order = _q_means.loc["Q4 (High)"].sort_values(ascending=False).index.tolist()
_q_means_sorted = _q_means[_sort_order]
_labels_sorted  = [k for v in _sort_order for k, vv in CHR_HEALTH.items() if vv == v]

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(_sort_order))
bar_w = 0.2
for qi, (qlabel, color) in enumerate(zip(QUARTILE_LABELS, QUARTILE_COLORS)):
    vals = _q_means_sorted.loc[qlabel].values
    ax.bar(x + qi * bar_w - 1.5 * bar_w, vals, width=bar_w,
           color=color, label=qlabel, edgecolor="white", linewidth=0.5)

ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xticks(x)
ax.set_xticklabels(_labels_sorted, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean Standardized z-score (relative to pooled mean)", fontsize=10)
ax.set_title(
    "CHR Health Outcome Profiles by % Hispanic Quartile — Rural US Counties, 2010–2015\n"
    "z-scores: 0 = pooled mean; sorted by Q4 (High Hispanic) mean",
    fontsize=11,
)
ax.legend(title="% Hispanic", fontsize=9, loc="upper right")
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_T_hispanic_quartile_profiles.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)


# =============================================================================
# FIGURE U: CAFO/FSIS exposure by %_hispanic quartile — boxplots
# =============================================================================
_u_df = df[[HISP_COL] + list(TREATMENT_VARS.values())].copy()
_u_df = _u_df[_u_df[HISP_COL].notna()].copy()
_u_df["hisp_q"] = pd.qcut(_u_df[HISP_COL], q=4, labels=QUARTILE_LABELS)

n_treat = len(TREATMENT_VARS)
fig, axes = plt.subplots(1, n_treat, figsize=(5 * n_treat, 6), sharey=False)
if n_treat == 1:
    axes = [axes]

for ax, (tlabel, tcol) in zip(axes, TREATMENT_VARS.items()):
    data_by_q = [_u_df.loc[_u_df["hisp_q"] == ql, tcol].dropna().values
                 for ql in QUARTILE_LABELS]
    bp = ax.boxplot(
        data_by_q,
        patch_artist=True,
        medianprops=dict(color="black", lw=1.5),
        whiskerprops=dict(lw=0.8),
        capprops=dict(lw=0.8),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], QUARTILE_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Overlay group means
    for qi, ql in enumerate(QUARTILE_LABELS, start=1):
        _m = _u_df.loc[_u_df["hisp_q"] == ql, tcol].mean()
        if not np.isnan(_m):
            ax.plot(qi, _m, "D", color="white", markersize=5,
                    markeredgecolor="black", markeredgewidth=0.8, zorder=5)

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(QUARTILE_LABELS, fontsize=8, rotation=20)
    ax.set_title(tlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel("log(ops per 10,000 residents + 1)", fontsize=9)

fig.suptitle(
    "CAFO & FSIS Exposure by % Hispanic Quartile — Rural US Counties, 2010–2015\n"
    "Box = IQR / Whiskers = 1.5×IQR / Diamond = group mean",
    fontsize=11, y=1.01,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_U_cafo_by_hispanic_quartile.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)


# =============================================================================
# FIGURE V: Partial scatter — CAFO (total) vs. focus outcomes, residualized on
# %_hispanic. Each outcome gets a row; columns are Hispanic quartiles (Q1–Q4).
# This answers: does the CAFO-health relationship persist within Hispanic quartile?
# =============================================================================
from sklearn.linear_model import LinearRegression

def _residualize(y_col, x_col, cov_col, df_sub):
    """Return residuals of y ~ cov and x ~ cov (partial regression)."""
    _rows = df_sub[[y_col, x_col, cov_col]].dropna()
    if len(_rows) < 30:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    def _resid(dep, indep):
        X = indep.values.reshape(-1, 1)
        Y = dep.values
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), Y, rcond=None)[0]
        return Y - (beta[0] + beta[1] * X.ravel())

    y_res = _resid(_rows[y_col], _rows[cov_col])
    x_res = _resid(_rows[x_col], _rows[cov_col])
    return pd.Series(x_res, index=_rows.index), pd.Series(y_res, index=_rows.index)

_v_df = df[[HISP_COL, "cafo_total_log"] + list(FOCUS_OUTCOMES.values())].copy()
_v_df = _v_df[_v_df[HISP_COL].notna() & _v_df["cafo_total_log"].notna()].copy()
_v_df["hisp_q"] = pd.qcut(_v_df[HISP_COL], q=4, labels=QUARTILE_LABELS)

N_BINS = 15

n_outcomes = len(FOCUS_OUTCOMES)
n_quartiles = 4
fig, axes = plt.subplots(n_outcomes, n_quartiles,
                          figsize=(5 * n_quartiles, 4 * n_outcomes),
                          sharex="row", sharey="row")

for ri, (olabel, ocol) in enumerate(FOCUS_OUTCOMES.items()):
    for ci, qlabel in enumerate(QUARTILE_LABELS):
        ax = axes[ri][ci]
        _sub = _v_df[_v_df["hisp_q"] == qlabel].copy()
        x_res, y_res = _residualize(ocol, "cafo_total_log", HISP_COL, _sub)

        if len(x_res) < 30:
            ax.text(0.5, 0.5, "n<30", transform=ax.transAxes, ha="center", va="center")
            ax.set_visible(True)
            continue

        # Bin into N_BINS quantiles
        _bins_df = pd.DataFrame({"xr": x_res.values, "yr": y_res.values}).dropna()
        _bins_df["bin"] = pd.qcut(_bins_df["xr"], q=N_BINS, duplicates="drop")
        _agg = _bins_df.groupby("bin", observed=True)["yr"].agg(["mean", "sem", "count"]).reset_index()
        _agg["x_mid"] = _agg["bin"].apply(lambda b: b.mid).astype(float)
        _agg = _agg[_agg["count"] >= 3]

        ax.errorbar(
            _agg["x_mid"], _agg["mean"],
            yerr=1.96 * _agg["sem"],
            fmt="o", markersize=4, capsize=3, color=QUARTILE_COLORS[ci],
            elinewidth=0.8, markeredgecolor="white",
        )
        # OLS fit
        if len(_agg) >= 4:
            _cx = _agg["x_mid"].values
            _cy = _agg["mean"].values
            _ok = np.isfinite(_cx) & np.isfinite(_cy)
            if _ok.sum() >= 4:
                _fit = np.polyfit(_cx[_ok], _cy[_ok], 1)
                _xl  = np.linspace(_cx[_ok].min(), _cx[_ok].max(), 100)
                ax.plot(_xl, np.polyval(_fit, _xl), "--", color="black", lw=0.9, alpha=0.7)
                # Annotate slope
                _r, _p = spearmanr(_cx[_ok], _cy[_ok])
                _pstr  = f"p={_p:.3f}" if _p >= 0.001 else "p<0.001"
                ax.text(0.05, 0.93, f"ρ={_r:.2f}, {_pstr}",
                        transform=ax.transAxes, fontsize=7, color="black",
                        va="top", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        # n label
        ax.text(0.98, 0.05, f"n={len(x_res):,}", transform=ax.transAxes,
                fontsize=6, ha="right", color="grey")

        if ri == 0:
            ax.set_title(qlabel, fontsize=10, fontweight="bold", color=QUARTILE_COLORS[ci])
        if ci == 0:
            ax.set_ylabel(f"{olabel}\n(residual)", fontsize=8)
        if ri == n_outcomes - 1:
            ax.set_xlabel("CAFO Total (log/10k, residual)", fontsize=8)

fig.suptitle(
    "Partial Scatter: CAFO Total vs. Health Outcomes — Residualized on % Hispanic\n"
    "Each column = % Hispanic quartile   |   Both axes are residuals after regressing on %_hispanic\n"
    "Tests whether CAFO-health relationship persists within Hispanic population share strata",
    fontsize=10, y=1.01,
)
plt.tight_layout()
path = os.path.join(out_dir, f"{today_str}_V_partial_scatter_hispanic.png")
fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", path)


# =============================================================================
# TABLE W: Summary stats (mean ± SD, n) for all CHR vars × Hispanic quartile
# =============================================================================
_w_df = df[[HISP_COL] + list(CHR_HEALTH.values())].copy()
_w_df = _w_df[_w_df[HISP_COL].notna()].copy()
_w_df["hisp_q"] = pd.qcut(_w_df[HISP_COL], q=4, labels=QUARTILE_LABELS)

records = []
for col, label in zip(CHR_HEALTH.values(), CHR_HEALTH.keys()):
    row = {"variable": label, "column": col}
    for ql in QUARTILE_LABELS:
        _sub = _w_df.loc[_w_df["hisp_q"] == ql, col].dropna()
        row[f"{ql}_mean"]   = round(_sub.mean(), 3)
        row[f"{ql}_sd"]     = round(_sub.std(), 3)
        row[f"{ql}_n"]      = len(_sub)
    # Add pooled stats
    _all = _w_df[col].dropna()
    row["pooled_mean"] = round(_all.mean(), 3)
    row["pooled_sd"]   = round(_all.std(), 3)
    row["pooled_n"]    = len(_all)
    records.append(row)

stats_df = pd.DataFrame(records)
csv_path = os.path.join(out_dir, f"{today_str}_W_chr_summary_by_hispanic_quartile.csv")
stats_df.to_csv(csv_path, index=False)
print("Saved:", csv_path)


print(f"\nAll hispanic-CHR figures saved to: {out_dir}")
