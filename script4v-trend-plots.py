#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script4v-trend-plots.py  --  WHAT DO THE OUTCOME TRENDS LOOK LIKE?  (added 2026-10-17)

Descriptive pictures and simple time regressions, no controls and no fixed effects, so the raw
paths are visible before any model is applied. Nothing here is an estimate of a treatment effect.

GROUPS (county level, from the zero-filled CAFO counts in load_panel)
    never large dairy        no large dairy operation in any census wave                      (comparison)
    enters                   first goes from 0 to 1+ large dairies at some wave (first entry)
    expands only             a positive change in an existing large-dairy count, never enters
    large dairy throughout   has a large dairy at some wave, never enters or expands

FIGURES  (-> Data/output/figs/script4v/)
    F1  outcome means by calendar year, one line per group, 7 panels (own y-axis each)
    F2  event-time path: mean change in the outcome since the year before treatment (g-1), treated
        counties vs comparison counties, for each of the two treatment definitions
    F3  pre-treatment slope difference by cohort: treated minus comparison, in outcome SDs per year

TABLES  (-> Data/output/tables/script4v/): the numbers behind every figure.

SIMPLE TIME REGRESSION (F3 / pre_window_slopes.csv)
    For cohort g, window = the 5 years before treatment (g-5 ... g-1), counties with >= 3 years of data:
        Y_it = a_i + b * year_t + c * (treated_i x year_t) + e_it       county FE, SE clustered by state
    c = how much faster (or slower) treated counties' outcomes were already moving per year.
    Outcomes that start late (poor mental-health days from 2010, frequent mental distress from 2016) have
    no usable window for early cohorts; those cells are simply absent.

COLOURS: categorical slots 1-4 of the reference palette (validated; the aqua/yellow contrast warning is
met with a legend, different line styles and the CSV tables).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyfixest as pf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from script4_treatment import load_panel, OUTCOMES, CENSUS_YEARS, figs_dir, tables_dir, os, date

OUT_F = os.path.join(figs_dir, "script4v"); os.makedirs(OUT_F, exist_ok=True)
OUT_T = os.path.join(tables_dir, "script4v"); os.makedirs(OUT_T, exist_ok=True)
TODAY = date.today().strftime("%Y-%m-%d")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SHORT = {"Poor MH Days": "Poor mental-health days", "Frequent Mental Distress": "Frequent mental distress",
         "Deaths of Despair": "Deaths of despair", "Aggravated Assault": "Aggravated assault",
         "Assault, all severities": "Assault, all severities", "Violence index (partial)": "Violence index (partial)",
         "NIBRS curated total": "NIBRS curated total"}

df = load_panel().sort_values(["fips", "year"]).reset_index(drop=True)

# ------------------------------------------------------------------ groups
waves = df[df.year.isin(CENSUS_YEARS)].pivot(index="fips", columns="year", values="cafo_dairy_large")
ever_large = (waves.fillna(0) > 0).any(axis=1)
info = df.groupby("fips").agg(entry=("entry_year", "first"), add=("add_year", "first"))
info["ever_large"] = ever_large.reindex(info.index).fillna(False)


def grp(r):
    if pd.notna(r.entry): return "enters"
    if pd.notna(r["add"]): return "expands only"
    if r.ever_large: return "large dairy throughout"
    return "never large dairy"


info["group"] = info.apply(grp, axis=1)
df["group"] = df["fips"].map(info["group"])
GROUPS = [("never large dairy", "#2a78d6", "-"), ("enters", "#eb6834", "--"),
          ("expands only", "#1baf7a", "-."), ("large dairy throughout", "#eda100", ":")]
counts = info.group.value_counts()
print("counties per group:", counts.to_dict())


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.7); ax.set_axisbelow(True)


def panel_grid(title, subtitle):
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4), facecolor=SURFACE)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.99)
    fig.text(0.01, 0.945, subtitle, ha="left", fontsize=9, color=MUTED)
    return fig, axes.ravel()


def legend_panel(ax, handles, labels, note):
    ax.axis("off")
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    ax.text(0.0, 0.40, note, fontsize=8, color=MUTED, va="top", wrap=True, transform=ax.transAxes)


# ================================================================== F1: calendar-year means by group
rows = []
fig, axes = panel_grid("Outcome means by calendar year, by county group",
                       "Raw county averages, no controls. Shaded band = +/-1.96 standard errors across counties. "
                       "Grey lines mark the census waves (2007, 2012, 2017, 2022).")
for ax, (okey, ocol) in zip(axes, OUTCOMES.items()):
    style(ax)
    for yr in (2007, 2012, 2017, 2022):
        ax.axvline(yr, color=GRID, linewidth=1.0, zorder=0)
    for gname, col, ls in GROUPS:
        g = df[(df.group == gname) & df[ocol].notna()].groupby("year")[ocol].agg(["mean", "std", "count"])
        g = g[g["count"] >= 15]
        se = g["std"] / np.sqrt(g["count"])
        ax.fill_between(g.index, g["mean"] - 1.96 * se, g["mean"] + 1.96 * se, color=col, alpha=0.13, linewidth=0)
        ax.plot(g.index, g["mean"], color=col, linestyle=ls, linewidth=2, solid_capstyle="round")
        for yr, r in g.iterrows():
            rows.append({"outcome": okey, "group": gname, "year": int(yr), "mean": r["mean"], "se": r["std"] / np.sqrt(r["count"]),
                         "n_counties": int(r["count"])})
    ax.set_title(SHORT[okey], fontsize=10, color=INK, loc="left")
    ax.set_xlim(2000, 2023)
handles = [plt.Line2D([0], [0], color=c, linestyle=l, linewidth=2) for _, c, l in GROUPS]
legend_panel(axes[7], handles, [f"{n}  (n = {counts.get(n, 0):,})" for n, _, _ in GROUPS],
             "Each panel has its own vertical scale: outcomes are in different units.")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT_F, f"{TODAY}_F1_means_by_group.png"), dpi=150, facecolor=SURFACE); plt.close(fig)
pd.DataFrame(rows).to_csv(os.path.join(OUT_T, f"{TODAY}_F1_means_by_group.csv"), index=False)


# ================================================================== F2: event-time paths (raw changes since g-1)
def event_paths(ocol, treated_col, comparison_fips, min_n=10, clip=8):
    wide = df.pivot_table(index="fips", columns="year", values=ocol, aggfunc="first")
    cohorts = sorted(info[treated_col].dropna().unique())
    recs = []
    for g in cohorts:
        base = g - 1
        if base not in wide.columns:
            continue
        tr = info.index[info[treated_col] == g]
        for e in range(-clip, clip + 1):
            t = g + e
            if t not in wide.columns:
                continue
            dt = (wide[t] - wide[base]).reindex(tr).dropna()
            dc = (wide[t] - wide[base]).reindex(comparison_fips).dropna()
            if len(dt) < min_n or len(dc) < min_n:
                continue
            recs.append({"cohort": int(g), "e": e, "n_t": len(dt), "m_t": dt.mean(), "v_t": dt.var(ddof=1) / len(dt),
                         "m_c": dc.mean(), "v_c": dc.var(ddof=1) / len(dc), "n_c": len(dc)})
    r = pd.DataFrame(recs)
    if r.empty:
        return pd.DataFrame()
    out = []
    for e, g in r.groupby("e"):
        w = g.n_t.to_numpy(float)
        out.append({"event_time": e, "treated": np.average(g.m_t, weights=w), "comparison": np.average(g.m_c, weights=w),
                    "se_t": np.sqrt(np.sum(w ** 2 * g.v_t)) / w.sum(), "se_c": np.sqrt(np.sum(w ** 2 * g.v_c)) / w.sum(),
                    "n_cohorts": g.cohort.nunique(), "n_treated": int(g.n_t.sum())})
    return pd.DataFrame(out)


DESIGNS = {
    "first expansion (any positive change in large dairies)": ("add", info.index[info["add"].isna() & info["entry"].isna()]),
    "first entry (0 to 1+ large dairies)": ("entry", info.index[info.group == "never large dairy"]),
}
ev_rows = []
for dname, (tcol, comp) in DESIGNS.items():
    fig, axes = panel_grid(f"Event-time path: {dname}",
                           "Mean change in the outcome since the year before treatment (g-1): treated counties vs comparison counties. "
                           "Raw changes, no controls. Band = +/-1.96 naive standard errors across counties.")
    for ax, (okey, ocol) in zip(axes, OUTCOMES.items()):
        style(ax)
        p = event_paths(ocol, tcol, comp)
        if p.empty:
            ax.set_title(SHORT[okey] + " (no usable cohorts)", fontsize=10, color=MUTED, loc="left"); continue
        for lab, col, ls, m, s in [("treated", "#eb6834", "--", "treated", "se_t"), ("comparison", "#2a78d6", "-", "comparison", "se_c")]:
            ax.fill_between(p.event_time, p[m] - 1.96 * p[s], p[m] + 1.96 * p[s], color=col, alpha=0.13, linewidth=0)
            ax.plot(p.event_time, p[m], color=col, linestyle=ls, linewidth=2)
        ax.axvline(-0.5, color=MUTED, linewidth=0.9, linestyle=(0, (1, 2)))
        ax.axhline(0, color=GRID, linewidth=1.0, zorder=0)
        ax.set_xlim(-8.4, 8.4)
        ax.set_title(SHORT[okey], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("years from first treated year", fontsize=8, color=MUTED)
        for _, r in p.iterrows():
            ev_rows.append({"design": dname, "outcome": okey, **r.to_dict()})
    handles = [plt.Line2D([0], [0], color="#eb6834", linestyle="--", linewidth=2), plt.Line2D([0], [0], color="#2a78d6", linewidth=2)]
    legend_panel(axes[7], handles, ["treated counties", "comparison counties"],
                 "Dotted vertical line = treatment date. The number of cohorts contributing falls at long leads and lags, "
                 "so the far ends are noisier and can change composition.")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    tag = "expansion" if "expansion" in dname else "entry"
    fig.savefig(os.path.join(OUT_F, f"{TODAY}_F2_event_time_{tag}.png"), dpi=150, facecolor=SURFACE); plt.close(fig)
pd.DataFrame(ev_rows).to_csv(os.path.join(OUT_T, f"{TODAY}_F2_event_time_paths.csv"), index=False)

# ================================================================== F3 + table: pre-window slope difference
sd = {o: df[c].std() for o, c in OUTCOMES.items()}
sl = []
for dname, (tcol, comp) in DESIGNS.items():
    for g in sorted(info[tcol].dropna().unique()):
        tr = set(info.index[info[tcol] == g])
        for okey, ocol in OUTCOMES.items():
            w = df[df.year.between(g - 5, g - 1) & df.fips.isin(tr | set(comp))][["fips", "state_fips", "year", ocol]].dropna()
            ok = w.groupby("fips").size()
            w = w[w.fips.isin(ok.index[ok >= 3])].copy()
            w["treated"] = w.fips.isin(tr).astype(float)
            w["tx"] = w["treated"] * w["year"]
            n_t, n_c = w[w.treated == 1].fips.nunique(), w[w.treated == 0].fips.nunique()
            if n_t < 10 or n_c < 10:
                continue
            try:
                m = pf.feols(f"{ocol} ~ year + tx | fips", data=w, vcov={"CRV1": "state_fips"}); t = m.tidy().loc["tx"]
            except Exception as e:
                continue
            sl.append({"design": dname, "cohort": int(g), "outcome": okey, "slope_diff": float(t["Estimate"]),
                       "se_state": float(t["Std. Error"]), "p_state": float(t["Pr(>|t|)"]),
                       "slope_diff_in_sd": float(t["Estimate"]) / sd[okey], "se_in_sd": float(t["Std. Error"]) / sd[okey],
                       "n_treated": n_t, "n_comparison": n_c, "window": f"{int(g) - 5}-{int(g) - 1}"})
sl = pd.DataFrame(sl)
sl.to_csv(os.path.join(OUT_T, f"{TODAY}_F3_pre_window_slopes.csv"), index=False)

COH = {2007: ("#2a78d6", "o"), 2012: ("#eb6834", "s"), 2017: ("#1baf7a", "^"), 2022: ("#eda100", "D")}
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), facecolor=SURFACE, sharex=True)
fig.suptitle("Were treated counties' outcomes already moving differently? Pre-treatment slope difference",
             x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold", y=0.99)
fig.text(0.01, 0.93, "Treated minus comparison, change in outcome per year over the 5 years before treatment, in outcome standard deviations. "
         "Bars = +/-1.96 state-clustered SE.", ha="left", fontsize=9, color=MUTED)
names = list(OUTCOMES)
for ax, (dname, _) in zip(axes, DESIGNS.items()):
    style(ax); ax.grid(axis="x", color=GRID, linewidth=0.7); ax.grid(axis="y", visible=False)
    ax.axvline(0, color=MUTED, linewidth=0.9)
    x = sl[sl.design == dname]
    for k, (g, (col, mk)) in enumerate(COH.items()):
        xs = x[x.cohort == g]
        ypos = [names.index(o) + (k - 1.5) * 0.17 for o in xs.outcome]
        ax.errorbar(xs.slope_diff_in_sd, ypos, xerr=1.96 * xs.se_in_sd, fmt=mk, color=col, ms=6, mec=SURFACE, mew=1.2,
                    elinewidth=1.3, capsize=0, label=f"{g} cohort", linestyle="none")
    ax.set_yticks(range(len(names))); ax.set_yticklabels([SHORT[n] for n in names], fontsize=9, color=INK)
    ax.invert_yaxis(); ax.set_title(dname, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("slope difference, SDs of the outcome per year", fontsize=8, color=MUTED)
axes[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(OUT_F, f"{TODAY}_F3_pre_window_slope_difference.png"), dpi=150, facecolor=SURFACE); plt.close(fig)

print("\nPRE-WINDOW SLOPE DIFFERENCE (treated minus comparison; outcome units per year; p state-clustered)")
for dname in DESIGNS:
    print(f"\n{dname}")
    for okey in OUTCOMES:
        x = sl[(sl.design == dname) & (sl.outcome == okey)].sort_values("cohort")
        print(f"  {okey[:24]:24s} " + "  ".join(f"{r.cohort}: {r.slope_diff:+.3g} (p={r.p_state:.2f}, nT={r.n_treated})" for r in x.itertuples()))
print(f"\nSaved figures -> {OUT_F}\nSaved tables  -> {OUT_T}")
